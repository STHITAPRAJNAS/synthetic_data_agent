"""PII Handler specialist agent.

ISOLATION CONTRACT
------------------
This agent NEVER receives or processes real data rows.  It only receives:
  1. Column metadata: PII category, distribution type, locale/domain hints.
  2. ``entity_hashes``: SHA-256 hashes of original values (computed by the
     generator agent which has access to real data).  These hashes let the
     agent maintain cross-table consistency via the SyntheticValueLedger
     without ever seeing the original values.
  3. ``pipeline_run_id`` and ``pipeline_salt``: scoping context for the ledger.

Cross-table consistency
-----------------------
When ``entity_hashes`` is provided for a column, each hash is looked up in the
SyntheticValueLedger.  If the same entity appeared in another table that was
already processed in this pipeline run, the same synthetic value is returned.
If the hash is new, a fresh synthetic value is generated, stored in the ledger,
and returned.

This guarantees that "John Smith" in ``customers.name`` and "John Smith" in
``orders.billing_name`` always become the same synthetic person name within a
single pipeline run, even though the PII handler never saw the original value.
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from google.adk.agents import Agent
from google.adk.tools import tool  # type: ignore[attr-defined]

from ....agents.callbacks import (
    after_model_callback,
    after_tool_callback,
    before_model_callback,
    before_tool_callback,
)
from ....models.column_profile import DistributionType, PIICategory
from ....pii.generators import (
    generate_synthetic_address,
    generate_synthetic_card_pan,
    generate_synthetic_email,
    generate_synthetic_instruction,
    generate_synthetic_ip,
    generate_synthetic_name,
    generate_synthetic_phone,
    generate_synthetic_ssn,
    recursive_rehydrate,
)
from ....tools.value_ledger import (
    SemanticType,
    SyntheticValueLedger,
    infer_semantic_type,
)

logger = structlog.get_logger()

# Module-level ledger singleton — shared across all tool calls in the same process
_ledger = SyntheticValueLedger()


@tool
async def populate_pii_columns(
    table_fqn: str,
    row_count: int,
    pii_spec: dict[str, dict[str, Any]],
    json_templates: dict[str, list[str]] | None = None,
    entity_hashes: dict[str, list[str]] | None = None,
    pipeline_run_id: str | None = None,
    pipeline_salt: str | None = None,
) -> dict[str, list[Any]]:
    """Generate synthetic PII values for specified columns.

    Cross-table consistency: when ``entity_hashes`` is provided, values for
    the same original entity are guaranteed to be identical across tables in
    the same pipeline run.

    Args:
        table_fqn: Fully-qualified table name (used for logging only).
        row_count: Number of rows to generate for each column.
        pii_spec: Dict mapping column name → spec dict with keys:
            - 'category': PIICategory string
            - 'dist_type': DistributionType string
            - 'locale_distribution': optional dict[locale, weight]
            - 'domain_distribution': optional dict[domain, weight]
        json_templates: Optional scrubbed JSON structure templates per column.
        entity_hashes: Dict mapping column name → list of 32-char hex hashes
            (one per row) computed by the generator from the original values.
            When present, the SyntheticValueLedger is used for consistency.
        pipeline_run_id: Unique ID for the current pipeline run.
        pipeline_salt: Per-run secret (not logged, not stored).

    Returns:
        Dict mapping column name → list of synthetic values (length = row_count).
    """
    logger.info(
        "pii_handler_start",
        table=table_fqn,
        rows=row_count,
        columns=list(pii_spec.keys()),
        cross_table_consistency=bool(entity_hashes and pipeline_run_id),
    )

    # Ensure the ledger DB schema exists
    if entity_hashes and pipeline_run_id:
        try:
            await _ledger.init_db()
        except Exception as exc:
            logger.warning("ledger_init_failed_falling_back", error=str(exc))
            entity_hashes = None  # fall back to independent generation

    data: dict[str, list[Any]] = {}

    for col_name, spec in pii_spec.items():
        category_str: str = spec.get("category", "SAFE")
        dist_type_str: str = spec.get("dist_type", "")
        locale_dist: dict[str, float] | None = spec.get("locale_distribution")
        domain_dist: dict[str, float] | None = spec.get("domain_distribution")

        col_hashes: list[str] | None = (entity_hashes or {}).get(col_name)
        use_ledger = (
            col_hashes is not None
            and pipeline_run_id is not None
            and pipeline_salt is not None
            and len(col_hashes) >= row_count
        )

        # ── JSON columns — re-hydrate from template structures ────────────
        if dist_type_str in (str(DistributionType.JSON), "DistributionType.JSON"):
            templates = (json_templates or {}).get(col_name, ['{"value": "placeholder"}'])
            synthetic_jsons: list[Any] = []
            for i in range(row_count):
                template = templates[i % len(templates)]
                if use_ledger and col_hashes:
                    # For JSON, we use a text-level lookup keyed by the template hash
                    # since JSON structures can't be byte-compared directly
                    val = await _ledger.lookup_or_generate(
                        pipeline_run_id=pipeline_run_id,  # type: ignore[arg-type]
                        pipeline_salt=pipeline_salt,  # type: ignore[arg-type]
                        semantic_type=SemanticType.FREE_TEXT,
                        original_value=col_hashes[i],  # already hashed — double-hashed in ledger
                        generator_fn=lambda t=template: _rehydrate_json(t),
                    )
                    synthetic_jsons.append(val)
                else:
                    synthetic_jsons.append(_rehydrate_json(template))
            data[col_name] = synthetic_jsons
            continue

        # ── Standard PII columns ──────────────────────────────────────────
        semantic_type = infer_semantic_type(col_name, category_str)
        generator_fn = _make_generator(col_name, category_str, locale_dist, domain_dist)

        if use_ledger and col_hashes:
            # Batch ledger lookup — handles all rows in one DB round-trip
            assert pipeline_run_id is not None
            assert pipeline_salt is not None
            values = await _ledger.bulk_lookup_or_generate(
                pipeline_run_id=pipeline_run_id,
                pipeline_salt=pipeline_salt,
                semantic_type=semantic_type,
                original_values=col_hashes[:row_count],  # already hashed
                generator_fn=generator_fn,
            )
            # Trim/pad to row_count (should already be exact)
            values = values[:row_count]
            if len(values) < row_count:
                values += [generator_fn() for _ in range(row_count - len(values))]
        else:
            # No hashes provided — generate independently (no cross-table guarantee)
            values = [generator_fn() for _ in range(row_count)]

        data[col_name] = values

    logger.info(
        "pii_handler_complete",
        table=table_fqn,
        generated_columns=list(data.keys()),
        used_ledger=bool(entity_hashes and pipeline_run_id),
    )
    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_generator(
    col_name: str,
    category_str: str,
    locale_dist: dict[str, float] | None,
    domain_dist: dict[str, float] | None,
) -> Any:
    """Return a zero-argument callable that produces one synthetic value."""
    col_lower = col_name.lower()

    if category_str in (str(PIICategory.DIRECT_PII), "DIRECT_PII", "PIICategory.DIRECT_PII"):
        if "ssn" in col_lower or "social_security" in col_lower:
            return generate_synthetic_ssn
        if "email" in col_lower:
            return lambda: generate_synthetic_email(domain_distribution=domain_dist)
        if "company" in col_lower or "employer" in col_lower or "org" in col_lower:
            from faker import Faker
            _f = Faker()
            return _f.company
        if "name" in col_lower or "contact" in col_lower or "person" in col_lower:
            return lambda: generate_synthetic_name(locale_distribution=locale_dist)
        if "address" in col_lower or "street" in col_lower:
            return lambda: str(generate_synthetic_address())
        # Generic DIRECT_PII fallback
        return lambda: generate_synthetic_name(locale_distribution=locale_dist)

    if category_str in (str(PIICategory.FINANCIAL_PII), "FINANCIAL_PII", "PIICategory.FINANCIAL_PII"):
        if "iban" in col_lower or "account" in col_lower:
            from ....pii.generators import generate_synthetic_iban
            return generate_synthetic_iban
        return generate_synthetic_card_pan

    if category_str in (str(PIICategory.QUASI_PII), "QUASI_PII", "PIICategory.QUASI_PII"):
        if "phone" in col_lower or "mobile" in col_lower:
            return generate_synthetic_phone
        if "ip" in col_lower:
            return generate_synthetic_ip
        if "zip" in col_lower or "postal" in col_lower:
            return lambda: generate_synthetic_address()["zip"]
        if "city" in col_lower:
            return lambda: generate_synthetic_address()["city"]
        return generate_synthetic_ip

    if category_str in (str(PIICategory.SENSITIVE), "SENSITIVE", "PIICategory.SENSITIVE"):
        from faker import Faker
        _f = Faker()
        return lambda: f"[REDACTED] {_f.word()}"

    # SAFE or unknown
    return lambda: None


def _rehydrate_json(template: str) -> str:
    """Re-hydrate a JSON template, replacing any residual PII structure."""
    try:
        parsed = json.loads(template)
        return json.dumps(recursive_rehydrate(parsed))
    except json.JSONDecodeError:
        return json.dumps({"value": generate_synthetic_instruction(template)})


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="pii_handler_agent",
    model="gemini-2.5-flash",
    description=(
        "PII isolation specialist.  Generates format-valid, provably non-real synthetic "
        "values for PII columns.  Uses the SyntheticValueLedger to ensure the same "
        "original entity maps to the same synthetic value across all tables in a "
        "pipeline run — without ever receiving real data."
    ),
    instruction="""You are a PII isolation specialist.

CRITICAL CONSTRAINT: You must never receive, process, or log real PII data.
You only receive:
  1. Column metadata (categories, distribution specs, format hints).
  2. entity_hashes — pre-computed hashes of original values from the generator.
  3. pipeline_run_id and pipeline_salt — ledger scoping context.

When generating PII data:
1. Call populate_pii_columns with pii_spec, entity_hashes, pipeline_run_id,
   and pipeline_salt.
2. When entity_hashes is provided, the SyntheticValueLedger ensures that
   the same original entity always maps to the same synthetic value, even
   if it appears in multiple tables.
3. For JSON columns, provide json_templates with scrubbed structure examples.
4. Return the generated column dict to the orchestrator for merging.

All generated values use provably fictional ranges:
- SSNs: area codes 900-999 (never issued)
- Phones: 555-01XX exchange (permanently reserved)
- Emails: @example.com / @test.invalid (RFC 2606)
- Card PANs: BIN starting with 9 (unassigned) + valid Luhn checksum
- IPs: RFC 5737 TEST-NET ranges only""",
    tools=[populate_pii_columns],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
