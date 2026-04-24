from __future__ import annotations

from typing import Any

import structlog
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse  # type: ignore[attr-defined]
from google.adk.tools import tool  # type: ignore[attr-defined]

from .callbacks import (
    after_model_callback,
    after_tool_callback,
    before_model_callback as _shared_before_model,
    before_tool_callback,
)
from ..config import get_settings
from ..ml.model_registry import ModelRegistry
from ..models.generation_plan import GenerationPlan, TableGenConfig
from ..models.quality_report import QualityReport
from ..tools.knowledge_base import KnowledgeBase
from ..tools.registry_tools import SyntheticIDRegistry
from ..tools.semantic_memory import SemanticMemory

logger = structlog.get_logger()

cfg = get_settings()

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised infra connections)
# ---------------------------------------------------------------------------
_knowledge_base = KnowledgeBase()
_semantic_memory = SemanticMemory()
_registry = SyntheticIDRegistry()
_model_registry = ModelRegistry()

_db_tools_ref: Any = None  # populated on first use


def _get_db_tools() -> Any:
    global _db_tools_ref
    if _db_tools_ref is None:
        from ..tools.databricks_tools import DatabricksTools
        _db_tools_ref = DatabricksTools()
    return _db_tools_ref


# ---------------------------------------------------------------------------
# Orchestrator tools
# ---------------------------------------------------------------------------

@tool
async def remember_business_rule(table_fqn: str, description: str, pandas_query: str) -> str:
    """Persist a business rule and index it in semantic memory for future retrieval.

    Args:
        table_fqn: Fully-qualified table name this rule applies to.
        description: Human-readable description of the constraint.
        pandas_query: pandas-compatible query string (e.g. 'amount > 0').

    Returns:
        Confirmation string.
    """
    await _knowledge_base.add_business_rule(table_fqn, description, pandas_query)
    await _semantic_memory.store_memory(table_fqn, description, {"query": pandas_query, "type": "rule"})
    return f"Remembered rule for {table_fqn}: {description}"


@tool
async def search_semantic_knowledge(query: str) -> str:
    """Search the pgvector semantic memory for similar past experiences or rules.

    Args:
        query: Free-text query describing what you're looking for.

    Returns:
        Formatted string with up to 3 relevant results.
    """
    results = await _semantic_memory.search_similar(query)
    if not results:
        return "No similar past experiences found."
    return f"Found {len(results)} relevant items:\n" + "\n".join(
        f"- [{r['table']}] {r['content']}" for r in results
    )


@tool
async def run_pipeline(table_fqns: list[str]) -> list[dict[str, Any]]:
    """Execute the full synthetic data generation pipeline with adaptive self-correction.

    Phases:
    1. Profile all tables in parallel.
    2. Build entity graph and topological order.
    3. Generate non-PII + PII per table (respecting FK order).
    4. Validate each table; self-correct up to 3 times on failure.
    5. Return aggregated QualityReports.

    Args:
        table_fqns: List of source table FQNs to synthesise.

    Returns:
        List of serialised QualityReport dicts.
    """
    from .specialists.profiler.agent import profile_tables
    from .specialists.entity_graph.agent import create_generation_plan
    from .specialists.generator.agent import generate_table_data
    from .specialists.pii_handler.agent import populate_pii_columns
    from .specialists.validator.agent import validate_table

    logger.info("Starting synthetic data pipeline", tables=table_fqns)

    # Ensure DB schemas exist
    await _knowledge_base.init_db()
    await _semantic_memory.init_db()
    await _registry.init_db()

    # -------------------------------------------------------------------------
    # Phase 1 — Profile
    # -------------------------------------------------------------------------
    profiles_json: list[dict[str, Any]] = await profile_tables(table_fqns)
    logger.info("Profiling complete", table_count=len(profiles_json))

    # -------------------------------------------------------------------------
    # Phase 2 — Entity graph + generation plan
    # -------------------------------------------------------------------------
    plan_json: dict[str, Any] = await create_generation_plan(profiles_json)
    plan = GenerationPlan.model_validate(plan_json)
    logger.info("Generation plan created", ordered_tables=plan.tables_ordered)

    final_reports: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Phase 3+4 — Generate + validate (table by table, topological order)
    # -------------------------------------------------------------------------
    for table_fqn in plan.tables_ordered:
        config: TableGenConfig = plan.table_configs[table_fqn]

        # Adaptive strategy selection — use best historical strategy if available
        best = _model_registry.best_strategy_for(table_fqn)
        if best:
            logger.info("Using historical best strategy", table=table_fqn, strategy=best)
            config = config.model_copy(update={"ml_strategy": best})  # type: ignore[assignment]

        max_attempts = 3
        report_json: dict[str, Any] | None = None

        for attempt in range(1, max_attempts + 1):
            logger.info("Generation attempt", table=table_fqn, attempt=attempt, strategy=config.ml_strategy)

            try:
                # 3a. Generate non-PII columns
                await generate_table_data(config.model_dump(mode="json"))

                # 3b. Build PII spec from column profiles (metadata only — no real data)
                profile = next(
                    (p for p in profiles_json if p["table_fqn"] == table_fqn), None
                )
                if profile and config.pii_columns:
                    pii_spec = {
                        c["name"]: {
                            "category": c.get("pii_category", "SAFE"),
                            "dist_type": c.get("distribution_type", "CATEGORICAL"),
                        }
                        for c in profile.get("columns", [])
                        if c["name"] in config.pii_columns
                    }
                    pii_data = await populate_pii_columns(
                        table_fqn=table_fqn,
                        row_count=config.target_row_count,
                        pii_spec=pii_spec,
                    )

                    # 3c. Merge PII columns into the output table
                    output_fqn = _output_fqn(table_fqn)
                    db = _get_db_tools()
                    synth_df = await db.sample_dataframe(output_fqn, config.target_row_count)
                    for col, values in pii_data.items():
                        if col in config.pii_columns and values:
                            synth_df[col] = values[: len(synth_df)]
                    await db.write_synthetic_table(output_fqn, synth_df)

                # 3d. Validate
                output_fqn = _output_fqn(table_fqn)
                quasi_ids = [
                    c["name"] for c in (profile or {}).get("columns", [])
                    if c.get("pii_category") == "QUASI_PII"
                ]
                report_json = await validate_table(table_fqn, output_fqn, quasi_ids)
                report = QualityReport.model_validate(report_json)

                await _model_registry.record_run(table_fqn, config.ml_strategy, report)

                if report.overall_pass:
                    logger.info("Quality gate passed", table=table_fqn, attempt=attempt)
                    break

                # Self-correction — adjust strategy/params for next attempt
                logger.warning("Quality gate failed", table=table_fqn, attempt=attempt)
                suggestions = _model_registry.get_tuning_suggestions(table_fqn, report)
                if "switch_strategy" in suggestions:
                    new_strategy: str = suggestions["switch_strategy"]
                    logger.info("Switching strategy", table=table_fqn, new=new_strategy)
                    config = config.model_copy(update={"ml_strategy": new_strategy})  # type: ignore[assignment]

            except Exception as exc:
                logger.error("Pipeline error", table=table_fqn, attempt=attempt, error=str(exc))
                if attempt == max_attempts:
                    report_json = QualityReport(
                        table_fqn=table_fqn,
                        overall_pass=False,
                        failure_details={"exception": str(exc)},
                    ).model_dump(mode="json")

        if report_json:
            final_reports.append(report_json)
        else:
            final_reports.append(
                QualityReport(
                    table_fqn=table_fqn,
                    overall_pass=False,
                    failure_details={"reason": "No validation result after all attempts"},
                ).model_dump(mode="json")
            )

    passed = sum(1 for r in final_reports if r.get("overall_pass"))
    logger.info(
        "Pipeline complete",
        tables_total=len(final_reports),
        tables_passed=passed,
        tables_failed=len(final_reports) - passed,
    )
    return final_reports


def _output_fqn(source_fqn: str) -> str:
    table_name = source_fqn.split(".")[-1]
    return f"{cfg.output_catalog}.{cfg.databricks_schema}.{table_name}"


# ---------------------------------------------------------------------------
# Orchestrator-specific before_model_callback
# Wraps the shared callback to also warm up DB connections before the first
# LLM call, then delegates to the shared logging callback.
# ---------------------------------------------------------------------------

async def _before_model_orchestrator(
    ctx: CallbackContext,
    request: Any,
) -> LlmResponse | None:
    """Warm up DB connections then run the shared logging callback.

    Ensures all async DB tables are initialised before the first LLM call so
    the pipeline never hits a "table does not exist" error mid-run.
    """
    await _knowledge_base.init_db()
    await _semantic_memory.init_db()
    await _registry.init_db()
    # Delegate to shared callback for request logging + timing
    return await _shared_before_model(ctx, request)


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="orchestrator",
    model="gemini-2.5-pro",
    description=(
        "Synthetic Data Generation Orchestrator — end-to-end pipeline coordinator "
        "that profiles tables, builds the entity graph, generates synthetic data with "
        "full referential integrity, isolates PII generation, and validates quality."
    ),
    instruction="""You are the Synthetic Data Generation Orchestrator.

When given a list of Databricks table FQNs, call run_pipeline to execute:

PHASE 1 — PROFILE: Sample and profile all tables, detecting PII columns.
PHASE 2 — ENTITY GRAPH: Build FK relationships and topological generation order.
PHASE 3 — GENERATE: For each table (parent before child):
  a. Generate non-PII columns via ML model (CTGAN/TVAE/Copula/TimeGAN).
  b. Generate PII columns via the isolated PII handler (metadata only).
  c. Merge non-PII + PII into the output table.
  d. Run all quality gates (KS test, correlation, FK integrity, privacy).
  e. Self-correct up to 3 times if quality gates fail.
PHASE 4 — REPORT: Summarise all QualityReports.

Additional tools:
- remember_business_rule: Persist domain constraints before running the pipeline.
- search_semantic_knowledge: Retrieve past learnings and tuning suggestions.""",
    tools=[run_pipeline, remember_business_rule, search_semantic_knowledge],
    before_model_callback=_before_model_orchestrator,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
