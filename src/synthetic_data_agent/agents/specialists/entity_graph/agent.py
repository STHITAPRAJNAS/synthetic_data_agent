from __future__ import annotations

import re
from typing import Any

import networkx as nx
import structlog
from google.adk.agents import Agent
from google.adk.tools import tool  # type: ignore[attr-defined]

from ....agents.callbacks import (
    after_model_callback,
    after_tool_callback,
    before_model_callback,
    before_tool_callback,
)
from ....models.column_profile import PIICategory, TableProfile
from ....models.entity_graph import EntityNode, FKRelation
from ....models.generation_plan import GenerationPlan, TableGenConfig

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# FK inference helpers
# ---------------------------------------------------------------------------

def _infer_fk_from_column_name(
    col_name: str,
    table_fqn: str,
    all_profiles: list[TableProfile],
) -> FKRelation | None:
    """Heuristic: if a column name matches ``{other_table}_id`` or ``{other_table}id``,
    treat it as a FK to that table's PK.

    Args:
        col_name: Column being inspected.
        table_fqn: FQN of the table that owns this column.
        all_profiles: All profiled tables (for candidate parent lookup).

    Returns:
        FKRelation if a plausible parent is found, else None.
    """
    # Normalise: strip trailing _id / id
    normalised = re.sub(r"_?id$", "", col_name.lower().replace("-", "_"))
    if not normalised:
        return None

    for profile in all_profiles:
        if profile.table_fqn == table_fqn:
            continue
        candidate_table = profile.table_fqn.split(".")[-1].lower()
        if candidate_table == normalised or candidate_table.startswith(normalised):
            # Find the PK column in the parent (first column marked as low-cardinality int)
            pk_col = "id"
            for c in profile.columns:
                if c.name.lower() in {"id", f"{candidate_table}_id", "pk"}:
                    pk_col = c.name
                    break
            logger.debug(
                "Inferred FK",
                child_table=table_fqn,
                fk_col=col_name,
                parent_table=profile.table_fqn,
                parent_pk=pk_col,
            )
            return FKRelation(
                fk_col=col_name,
                parent_table_fqn=profile.table_fqn,
                parent_pk_col=pk_col,
                cardinality="many_to_many",
                null_rate=0.0,
                fanout_mean=10.0,
                fanout_distribution="poisson",
            )
    return None


# ---------------------------------------------------------------------------
# ADK tool function
# ---------------------------------------------------------------------------

@tool
async def create_generation_plan(
    profiles_json: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyse TableProfiles and produce a topologically ordered GenerationPlan.

    Args:
        profiles_json: List of TableProfile dicts (serialised via model.model_dump()).

    Returns:
        GenerationPlan serialised as a dict.
    """
    profiles = [TableProfile.model_validate(p) for p in profiles_json]

    G: nx.DiGraph = nx.DiGraph()
    table_configs: dict[str, TableGenConfig] = {}

    for p in profiles:
        G.add_node(p.table_fqn)

        # Classify columns
        pii_cols: list[str] = [
            c.name for c in p.columns
            if c.pii_category is not None and c.pii_category != PIICategory.SAFE
        ]
        non_pii_cols: list[str] = [
            c.name for c in p.columns
            if c.pii_category is None or c.pii_category == PIICategory.SAFE
        ]

        # Infer FK relationships from Unity Catalog + naming heuristics
        fk_relations: list[FKRelation] = []
        for col in p.columns:
            # Skip obvious non-FK columns
            if col.name.lower() in {"created_at", "updated_at", "deleted_at", "timestamp"}:
                continue
            if col.name.lower().endswith("_id") or col.name.lower().endswith("id"):
                fk = _infer_fk_from_column_name(col.name, p.table_fqn, profiles)
                if fk is not None:
                    fk_relations.append(fk)
                    # Add edge: parent → child
                    G.add_edge(fk.parent_table_fqn, p.table_fqn)

        # Infer primary key columns (first column named id / table_id)
        pk_cols: list[str] = []
        table_name = p.table_fqn.split(".")[-1].lower()
        for c in p.columns:
            if c.name.lower() in {"id", f"{table_name}_id", "pk"}:
                pk_cols.append(c.name)
                break
        if not pk_cols and p.columns:
            pk_cols = [p.columns[0].name]

        table_configs[p.table_fqn] = TableGenConfig(
            table_fqn=p.table_fqn,
            target_row_count=p.row_count,
            ml_strategy="ctgan",
            pii_columns=pii_cols,
            non_pii_columns=non_pii_cols,
            foreign_keys=fk_relations,
            primary_key_cols=pk_cols,
        )

    # Topological sort — parents before children
    try:
        ordered_tables = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        logger.error("Cyclic FK dependency detected — falling back to profile order")
        ordered_tables = [p.table_fqn for p in profiles]

    plan = GenerationPlan(
        tables_ordered=ordered_tables,
        table_configs=table_configs,
        estimated_total_rows=sum(p.row_count for p in profiles),
    )
    return plan.model_dump(mode="json")


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="entity_graph_agent",
    model="gemini-2.5-flash",
    description=(
        "Data modeling specialist that builds the entity-relationship graph from "
        "TableProfiles and returns a topologically ordered GenerationPlan with "
        "FK relations and fanout ratios."
    ),
    instruction="""You are a data modeling specialist.

Given a list of serialised TableProfile objects, call create_generation_plan to:
1. Build the complete FK graph by inferring relationships from column names.
2. Compute topological generation order (parents before children).
3. Classify PII vs non-PII columns per table.
4. Return the serialised GenerationPlan dict.""",
    tools=[create_generation_plan],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
