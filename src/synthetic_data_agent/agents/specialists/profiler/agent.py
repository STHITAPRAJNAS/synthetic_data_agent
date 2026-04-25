from __future__ import annotations

import asyncio
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
from ....config import get_settings
from ....models.column_profile import ColumnProfile, DistributionType, PIICategory, TableProfile
from ....pii.detector import PIIDetector
from ....tools.databricks_tools import DatabricksTools

logger = structlog.get_logger()

_db_tools = DatabricksTools()
_pii_detector = PIIDetector()

# Max concurrent table profiles — prevents overwhelming Spark / LLM quotas
_PROFILE_CONCURRENCY = 4
_profile_semaphore: asyncio.Semaphore | None = None


def _get_profile_semaphore() -> asyncio.Semaphore:
    global _profile_semaphore
    if _profile_semaphore is None:
        _profile_semaphore = asyncio.Semaphore(_PROFILE_CONCURRENCY)
    return _profile_semaphore


# ---------------------------------------------------------------------------
# Per-table profiling logic (runs in parallel)
# ---------------------------------------------------------------------------

async def _profile_single_table(fqn: str, cfg: Any) -> dict[str, Any]:
    """Profile one table and return its serialised TableProfile."""
    sem = _get_profile_semaphore()
    async with sem:
        logger.info("Profiling table", table=fqn)

        # 1. Schema (actual row count from Unity Catalog when available)
        schema = await _db_tools.read_table_schema(fqn)

        # 2. Sample for column-level profiling
        df = await _db_tools.sample_dataframe(fqn, cfg.max_profiling_sample_rows)

        # 3. Prefer schema-reported row count; fall back to sample length
        actual_row_count: int = int(
            schema.get("properties", {}).get("numRows", len(df))
        )

        # 4. Column profiles — PII detection runs concurrently per column
        async def _profile_col(col: str) -> ColumnProfile:
            pii_cat = await _pii_detector.detect(col, df[col].dropna().tolist()[:200])
            cardinality = int(df[col].nunique())
            null_rate = float(df[col].isnull().mean())

            if pii_cat != PIICategory.SAFE:
                dist_type = DistributionType.HIGH_CARD_STRING
            elif df[col].dtype == "bool":
                dist_type = DistributionType.BOOLEAN
            elif df[col].dtype == "object":
                dist_type = _infer_string_dist(df[col].dropna().astype(str).iloc[:5].tolist())
            elif str(df[col].dtype).startswith("datetime"):
                dist_type = DistributionType.TEMPORAL
            elif cardinality <= 20:
                dist_type = DistributionType.CATEGORICAL
            else:
                import numpy as np  # type: ignore[import]
                vals = df[col].dropna()
                dist_type = (
                    DistributionType.LOG_NORMAL
                    if float(vals.skew()) > 1.5
                    else DistributionType.GAUSSIAN
                )

            return ColumnProfile(
                name=col,
                dtype=str(df[col].dtype),
                pii_category=pii_cat,
                distribution_type=dist_type,
                null_rate=null_rate,
                cardinality=cardinality,
                sample_values=df[col].dropna().unique()[:5].tolist(),
            )

        col_profiles = await asyncio.gather(*[_profile_col(c) for c in df.columns])

        # Detect temporal column (first datetime column, or one named *_at / *_date / *_time)
        temporal_col: str | None = None
        for cp in col_profiles:
            if cp.distribution_type == DistributionType.TEMPORAL:
                temporal_col = cp.name
                break
        if temporal_col is None:
            for cp in col_profiles:
                lc = cp.name.lower()
                if any(lc.endswith(s) for s in ("_at", "_date", "_time", "_ts", "_timestamp")):
                    temporal_col = cp.name
                    break

        profile = TableProfile(
            table_fqn=fqn,
            row_count=actual_row_count,
            columns=list(col_profiles),
            temporal_col=temporal_col,
        )
        logger.info("Profiling complete", table=fqn, columns=len(col_profiles))
        return profile.model_dump(mode="json")


# ---------------------------------------------------------------------------
# ADK tool
# ---------------------------------------------------------------------------

@tool
async def profile_tables(table_fqns: list[str]) -> list[dict[str, Any]]:
    """Profile a list of Databricks tables and return their serialised TableProfiles.

    Tables are profiled in parallel (up to 4 concurrently) to minimise wall-clock
    time for large schemas.  Each table profile includes:
    - Full column inventory with dtype, cardinality, null rate.
    - PII classification via 3-layer detection (regex → Presidio → LLM).
    - Distribution type inference (gaussian, log-normal, categorical, temporal, JSON…).
    - Actual row count from Unity Catalog (or sample-based estimate for local files).

    Args:
        table_fqns: Fully-qualified table names (catalog.schema.table or file://path).

    Returns:
        List of TableProfile dicts (serialised via model.model_dump()).
    """
    cfg = get_settings()

    results = await asyncio.gather(
        *[_profile_single_table(fqn, cfg) for fqn in table_fqns],
        return_exceptions=True,
    )

    profiles: list[dict[str, Any]] = []
    for fqn, result in zip(table_fqns, results):
        if isinstance(result, BaseException):
            logger.error("Table profiling failed", table=fqn, error=str(result))
            # Emit a minimal error profile rather than aborting the whole pipeline
            profiles.append({"table_fqn": fqn, "error": str(result), "columns": [], "row_count": 0})
        else:
            profiles.append(result)

    return profiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_string_dist(samples: list[str]) -> DistributionType:
    """Detect JSON or categorical from string samples."""
    for s in samples:
        if s and s[0] in {"{", "["}:
            try:
                json.loads(s)
                return DistributionType.JSON
            except json.JSONDecodeError:
                pass
    return DistributionType.CATEGORICAL


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="profiler_agent",
    model="gemini-2.5-flash",
    description=(
        "Data profiling specialist that analyses Databricks tables in parallel, detects PII "
        "columns using 3-layer detection (regex, Presidio, LLM), and returns serialised "
        "TableProfile objects with full column statistics and distribution types."
    ),
    instruction="""You are a data profiling specialist.

When given a list of table FQNs, call profile_tables to:
1. Profile all tables concurrently (up to 4 at a time).
2. Detect PII columns using regex → Presidio NLP → LLM metadata heuristic.
3. Compute null rates, cardinality, and distribution types per column.
4. Return the list of serialised TableProfile dicts.

Always pass the full list to profile_tables in a single call — parallel processing
is handled internally and minimises total pipeline latency.""",
    tools=[profile_tables],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
