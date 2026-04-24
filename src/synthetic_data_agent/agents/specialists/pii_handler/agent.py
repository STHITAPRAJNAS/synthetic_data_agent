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

logger = structlog.get_logger()


@tool
async def populate_pii_columns(
    table_fqn: str,
    row_count: int,
    pii_spec: dict[str, dict[str, Any]],
    json_templates: dict[str, list[str]] | None = None,
) -> dict[str, list[Any]]:
    """Generate synthetic PII data for specified columns.

    This agent is intentionally isolated: it receives only metadata (column
    categories, distribution specs, and optional JSON structure templates) —
    never actual real data rows.

    Args:
        table_fqn: Fully-qualified table name (used for logging only).
        row_count: Number of rows to generate for each column.
        pii_spec: Dict mapping column name → spec dict with keys:
            - 'category': PIICategory string (e.g. 'DIRECT_PII')
            - 'dist_type': DistributionType string (e.g. 'JSON')
            - 'locale_distribution': optional dict[locale, weight]
            - 'domain_distribution': optional dict[domain, weight]
        json_templates: Optional dict mapping column name → list of example
            JSON *structure* strings (values already scrubbed — used to infer
            schema shape only, no real PII should be present).

    Returns:
        Dict mapping column name → list of synthetic values (length = row_count).
    """
    logger.info("Populating PII columns", table=table_fqn, rows=row_count, columns=list(pii_spec.keys()))
    data: dict[str, list[Any]] = {}

    for col_name, spec in pii_spec.items():
        category_str: str = spec.get("category", "SAFE")
        dist_type_str: str = spec.get("dist_type", "")
        locale_dist: dict[str, float] | None = spec.get("locale_distribution")
        domain_dist: dict[str, float] | None = spec.get("domain_distribution")

        # --- JSON columns: re-hydrate from provided template structures ---
        if dist_type_str == str(DistributionType.JSON) or dist_type_str == "DistributionType.JSON":
            templates = (json_templates or {}).get(col_name, ['{"value": "placeholder"}'])
            synthetic_jsons: list[Any] = []
            for i in range(row_count):
                template = templates[i % len(templates)]
                try:
                    parsed = json.loads(template)
                    rehydrated = recursive_rehydrate(parsed)
                    synthetic_jsons.append(json.dumps(rehydrated))
                except json.JSONDecodeError:
                    synthetic_jsons.append(json.dumps({"value": generate_synthetic_instruction(template)}))
            data[col_name] = synthetic_jsons
            continue

        # --- Standard flat PII generation ---
        col_lower = col_name.lower()
        values: list[Any]

        if category_str in (str(PIICategory.DIRECT_PII), "PIICategory.DIRECT_PII"):
            if "ssn" in col_lower or "social_security" in col_lower:
                values = [generate_synthetic_ssn() for _ in range(row_count)]
            elif "email" in col_lower:
                values = [generate_synthetic_email(domain_distribution=domain_dist) for _ in range(row_count)]
            elif "name" in col_lower:
                values = [generate_synthetic_name(locale_distribution=locale_dist) for _ in range(row_count)]
            elif "address" in col_lower or "street" in col_lower:
                values = [str(generate_synthetic_address()) for _ in range(row_count)]
            else:
                # Generic DIRECT_PII fallback — use name
                values = [generate_synthetic_name(locale_distribution=locale_dist) for _ in range(row_count)]

        elif category_str in (str(PIICategory.FINANCIAL_PII), "PIICategory.FINANCIAL_PII"):
            values = [generate_synthetic_card_pan() for _ in range(row_count)]

        elif category_str in (str(PIICategory.QUASI_PII), "PIICategory.QUASI_PII"):
            if "phone" in col_lower:
                values = [generate_synthetic_phone() for _ in range(row_count)]
            elif "ip" in col_lower:
                values = [generate_synthetic_ip() for _ in range(row_count)]
            elif "zip" in col_lower or "postal" in col_lower:
                addr_list = [generate_synthetic_address() for _ in range(row_count)]
                values = [a["zip"] for a in addr_list]
            else:
                values = [generate_synthetic_ip() for _ in range(row_count)]

        elif category_str in (str(PIICategory.SENSITIVE), "PIICategory.SENSITIVE"):
            # Sensitive but non-identifying — generate a safe placeholder
            values = [f"[REDACTED_{i}]" for i in range(row_count)]

        else:
            # SAFE or unknown — nothing to generate; orchestrator handles non-PII
            values = [None] * row_count

        data[col_name] = values

    logger.info("PII column generation complete", table=table_fqn, generated_columns=list(data.keys()))
    return data


# ---------------------------------------------------------------------------
# ADK root agent — ISOLATED: receives metadata only, never real data rows
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="pii_handler_agent",
    model="gemini-2.5-flash",
    description=(
        "PII isolation specialist that generates format-valid but provably non-real "
        "values for PII columns. Receives only column metadata and distribution specs — "
        "never actual real data rows."
    ),
    instruction="""You are a PII isolation specialist.

CRITICAL CONSTRAINT: You must never receive, process, or log real PII data.
You only receive column metadata (categories, locale distributions, format specs).

When asked to generate PII data:
1. Call populate_pii_columns with the pii_spec metadata dict.
2. For JSON columns, provide json_templates with scrubbed structure examples (no real values).
3. Return the generated column data to the orchestrator for merging.

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
