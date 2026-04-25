"""Generator specialist agent.

This agent is the heart of the pipeline.  For each table it:

1. Resolves the training DataFrame from the data source (generic pandas I/O).
2. Selects the best synthesis strategy (data-driven heuristic + quality history).
3. Checks the ADK artifact store for a cached trained model (skip retraining).
4. If no cache hit, trains the model via a LongRunningFunctionTool that emits
   real-time progress events the orchestrator can observe.
5. Saves the trained model to the ADK artifact store.
6. Generates synthetic non-PII rows with optional conditional constraints.
7. Resolves FK columns from the SyntheticIDRegistry.
8. Registers generated PKs so child tables can reference them.

The agent exposes two tools:
  - ``plan_generation`` — fast, synchronous pre-flight check
  - ``train_and_generate`` — long-running training + generation (wrapped in
    ``LongRunningFunctionTool`` so the agent can report progress and recover
    from partial failures)
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator
from typing import Any

import pandas as pd
import structlog
from google.adk.agents import Agent
from google.adk.tools.long_running_tool import LongRunningFunctionTool  # type: ignore[import]

from ....agents.callbacks import (
    after_model_callback,
    after_tool_callback,
    before_model_callback,
    before_tool_callback,
)
from ....config import get_settings
from ....ml.artifact_store import load_model_artifact, save_model_artifact
from ....ml.base import TrainingConfig
from ....ml.copula_trainer import CopulaTrainer
from ....ml.ctgan_trainer import CTGANTrainer
from ....ml.data_fingerprint import artifact_key, fingerprint_dataframe
from ....ml.model_registry import ModelRegistry
from ....ml.strategy_selector import adapt_training_config_for_profile, select_strategy
from ....ml.timegan_trainer import TimeGANTrainer
from ....ml.tvae_trainer import TVAETrainer
from ....models.column_profile import TableProfile
from ....models.generation_plan import TableGenConfig
from ....tools.knowledge_base import KnowledgeBase
from ....tools.registry_tools import SyntheticIDRegistry

logger = structlog.get_logger()

# Module-level singletons — shared across tool calls within the same process
_registry = SyntheticIDRegistry()
_knowledge_base = KnowledgeBase()
_model_registry = ModelRegistry()

_TRAINER_MAP = {
    "ctgan": CTGANTrainer,
    "tvae": TVAETrainer,
    "copula": CopulaTrainer,
    "timegan": TimeGANTrainer,
}


# ---------------------------------------------------------------------------
# Helper: load training data from any supported source
# ---------------------------------------------------------------------------

async def _load_training_df(config: TableGenConfig, cfg: Any) -> pd.DataFrame:
    """Load training data regardless of source (Databricks, file, or in-memory)."""
    fqn = config.table_fqn

    if fqn.startswith("file://"):
        from ....tools.databricks_tools import DatabricksTools
        db = DatabricksTools()
        return await db.sample_dataframe(fqn, cfg.max_profiling_sample_rows)

    if fqn.startswith("df://"):
        # In-memory: agent passed a pre-loaded DataFrame via shared state
        # Convention: fqn = "df://<uuid>" where uuid maps to a session artifact
        raise NotImplementedError(
            "In-memory DataFrames must be passed via the file:// convention "
            "after writing to a temp CSV. See upload endpoint."
        )

    # Default: Databricks / any SQL source
    from ....tools.databricks_tools import DatabricksTools
    db = DatabricksTools()
    return await db.sample_dataframe(fqn, cfg.max_profiling_sample_rows)


async def _write_output(
    fqn: str,
    df: pd.DataFrame,
    cfg: Any,
) -> dict[str, Any]:
    """Write synthetic DataFrame to the configured output location."""
    if fqn.startswith("file://"):
        from ....tools.databricks_tools import DatabricksTools
        db = DatabricksTools()
        return await db.write_synthetic_table(fqn, df)

    output_fqn = _output_fqn(fqn, cfg)
    from ....tools.databricks_tools import DatabricksTools
    db = DatabricksTools()
    return await db.write_synthetic_table(output_fqn, df)


def _output_fqn(source_fqn: str, cfg: Any) -> str:
    table_name = source_fqn.split(".")[-1]
    return f"{cfg.output_catalog}.{cfg.databricks_schema}.{table_name}_synthetic"


# ---------------------------------------------------------------------------
# Tool 1: plan_generation  (fast, synchronous pre-flight)
# ---------------------------------------------------------------------------

async def plan_generation(
    config_json: dict[str, Any],
    profile_json: dict[str, Any],
) -> dict[str, Any]:
    """Pre-flight check — selects strategy and checks artifact cache.

    Returns a plan dict so the orchestrator can make an informed decision
    before committing to a potentially long training run.

    Args:
        config_json: Serialised TableGenConfig.
        profile_json: Serialised TableProfile from the profiler agent.

    Returns:
        Dict with keys:
          - recommended_strategy: chosen strategy name
          - strategy_reason: explanation
          - cache_hit: True if a trained model artifact exists
          - artifact_key: artifact key to pass to train_and_generate
          - run_count: number of previous attempts for this table
    """
    config = TableGenConfig.model_validate(config_json)
    profile = TableProfile.model_validate(profile_json)

    # Strategy selection — prefer history over heuristic
    best_historical = _model_registry.best_strategy_for(config.table_fqn)
    if best_historical:
        strategy = best_historical
        reason = f"Using historically best strategy from {_model_registry.run_count(config.table_fqn)} recorded runs."
    else:
        decision = select_strategy(profile)
        strategy = decision.strategy
        reason = decision.reason

    # Check artifact cache
    cfg = get_settings()
    best_artifact = _model_registry.best_artifact_for(config.table_fqn)
    cache_hit = best_artifact is not None

    return {
        "table_fqn": config.table_fqn,
        "recommended_strategy": strategy,
        "strategy_reason": reason,
        "cache_hit": cache_hit,
        "artifact_key": best_artifact["artifact_key"] if best_artifact else None,
        "run_count": _model_registry.run_count(config.table_fqn),
        "target_row_count": config.target_row_count,
    }


# ---------------------------------------------------------------------------
# Tool 2: train_and_generate  (long-running)
# ---------------------------------------------------------------------------

async def train_and_generate(
    config_json: dict[str, Any],
    profile_json: dict[str, Any],
    strategy: str,
    cached_artifact_key: str | None = None,
    tool_context: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """Train a synthesis model (or load from cache) then generate synthetic data.

    This is a long-running operation wrapped in ``LongRunningFunctionTool``.
    It yields progress dicts at each milestone so the orchestrator can
    surface live status to the user.

    Progress dict schema:
        {
          "status":   "starting" | "running" | "completed" | "failed",
          "progress": 0-100,
          "message":  human-readable description,
          "result":   {...}  # only present in "completed" yield
        }

    Args:
        config_json: Serialised TableGenConfig.
        profile_json: Serialised TableProfile.
        strategy: Selected strategy ('ctgan', 'tvae', 'copula', 'timegan').
        cached_artifact_key: If provided, try to load this model instead of training.
        tool_context: ADK ToolContext (injected by framework — used for artifact store).

    Yields:
        Progress update dicts.
    """
    cfg = get_settings()
    config = TableGenConfig.model_validate(config_json)
    profile = TableProfile.model_validate(profile_json)
    t_start = time.monotonic()

    yield {
        "status": "starting",
        "progress": 0,
        "message": f"Starting generation for {config.table_fqn} using {strategy}",
    }

    # ── Phase 1: Load training data ────────────────────────────────────────
    yield {"status": "running", "progress": 5, "message": "Loading training data…"}
    try:
        real_df = await _load_training_df(config, cfg)
    except Exception as exc:
        yield {"status": "failed", "progress": 5, "message": f"Failed to load training data: {exc}"}
        return

    non_pii_cols = [c for c in config.non_pii_columns if c in real_df.columns]
    if not non_pii_cols:
        yield {"status": "failed", "progress": 5, "message": "No non-PII columns available for training."}
        return

    train_df = real_df[non_pii_cols].copy()

    yield {
        "status": "running",
        "progress": 10,
        "message": f"Loaded {len(train_df):,} training rows × {len(train_df.columns)} columns.",
    }

    # ── Phase 2: Resolve hyperparameters ───────────────────────────────────
    overrides = adapt_training_config_for_profile(profile, strategy)  # type: ignore[arg-type]
    train_config = TrainingConfig(**{**dataclasses.asdict(TrainingConfig()), **overrides})

    # ── Phase 3: Check artifact cache ─────────────────────────────────────
    model = None
    used_cache = False

    if cached_artifact_key and tool_context:
        yield {"status": "running", "progress": 15, "message": "Checking artifact cache…"}
        try:
            model = await load_model_artifact(tool_context, cached_artifact_key, strategy)
            if model:
                used_cache = True
                logger.info(
                    "generator_cache_hit",
                    table=config.table_fqn,
                    artifact_key=cached_artifact_key,
                )
        except Exception as exc:
            logger.warning("generator_cache_load_failed", error=str(exc))

    # If no cached model, also check by fingerprint
    if not model and tool_context:
        fp = fingerprint_dataframe(train_df, train_config, config.table_fqn, strategy)
        a_key = artifact_key(config.table_fqn, strategy, fp)
        yield {"status": "running", "progress": 18, "message": f"Checking fingerprint cache ({fp[:8]}…)"}
        try:
            model = await load_model_artifact(tool_context, a_key, strategy)
            if model:
                used_cache = True
                cached_artifact_key = a_key
                logger.info("generator_fingerprint_cache_hit", table=config.table_fqn, fp=fp)
        except Exception as exc:
            logger.debug("generator_fingerprint_cache_miss", error=str(exc))

    # ── Phase 4: Train (if cache miss) ────────────────────────────────────
    if not model:
        trainer_cls = _TRAINER_MAP.get(strategy, CTGANTrainer)
        trainer = trainer_cls()

        yield {
            "status": "running",
            "progress": 20,
            "message": f"Training {strategy.upper()} — {len(train_df):,} rows × {len(train_df.columns)} cols, {train_config.epochs} epochs…",
        }

        try:
            training_result = await asyncio.to_thread(trainer.train, train_df, train_config)
        except Exception as exc:
            yield {"status": "failed", "progress": 20, "message": f"Training failed: {exc}"}
            return

        yield {
            "status": "running",
            "progress": 75,
            "message": f"Training complete in {training_result.training_duration_s:.0f}s. Saving to artifact store…",
        }

        # Save to artifact store
        if tool_context:
            try:
                fp = fingerprint_dataframe(train_df, train_config, config.table_fqn, strategy)
                a_key = artifact_key(config.table_fqn, strategy, fp)
                await save_model_artifact(tool_context, trainer, a_key)
                cached_artifact_key = a_key
            except Exception as exc:
                logger.warning("generator_artifact_save_failed", error=str(exc))

        model = trainer
    else:
        yield {
            "status": "running",
            "progress": 75,
            "message": f"Cache hit — skipped training, loaded model from {cached_artifact_key}.",
        }

    # ── Phase 5: Generate synthetic rows ──────────────────────────────────
    target = config.target_row_count
    oversample = int(target * 1.5)  # over-generate, trim after business rules

    yield {"status": "running", "progress": 80, "message": f"Generating {oversample:,} synthetic rows…"}

    try:
        synth_df = await asyncio.to_thread(model.sample, oversample)
    except Exception as exc:
        yield {"status": "failed", "progress": 80, "message": f"Sampling failed: {exc}"}
        return

    # ── Phase 6: Apply business rules ─────────────────────────────────────
    yield {"status": "running", "progress": 85, "message": "Applying business rules…"}
    rules = await _knowledge_base.get_business_rules(config.table_fqn)
    for rule in rules:
        try:
            before = len(synth_df)
            synth_df = synth_df.query(rule["code"])
            logger.info(
                "business_rule_applied",
                table=config.table_fqn,
                rule=rule["description"],
                rows_before=before,
                rows_after=len(synth_df),
            )
        except Exception as exc:
            logger.warning("business_rule_failed", rule=rule.get("description"), error=str(exc))

    if len(synth_df) < target:
        logger.warning(
            "insufficient_rows_after_rules",
            table=config.table_fqn,
            generated=len(synth_df),
            target=target,
        )
    synth_df = synth_df.head(target)

    # ── Phase 7: Resolve FK columns ────────────────────────────────────────
    yield {"status": "running", "progress": 88, "message": "Resolving foreign key references…"}
    for fk in config.foreign_keys:
        if fk.fk_col not in synth_df.columns:
            continue
        try:
            parent_ids = await _registry.sample_fk(
                fk.parent_table_fqn, fk.parent_pk_col, len(synth_df)
            )
            if parent_ids:
                synth_df[fk.fk_col] = parent_ids[: len(synth_df)]
                logger.debug(
                    "fk_resolved",
                    table=config.table_fqn,
                    fk_col=fk.fk_col,
                    parent=fk.parent_table_fqn,
                )
            else:
                logger.warning(
                    "fk_parent_not_registered",
                    table=config.table_fqn,
                    fk_col=fk.fk_col,
                    parent=fk.parent_table_fqn,
                )
        except Exception as exc:
            logger.error("fk_resolution_failed", fk_col=fk.fk_col, error=str(exc))

    # ── Phase 8: Write output ─────────────────────────────────────────────
    yield {"status": "running", "progress": 92, "message": "Writing synthetic data to output…"}
    try:
        write_result = await _write_output(config.table_fqn, synth_df, cfg)
    except Exception as exc:
        yield {"status": "failed", "progress": 92, "message": f"Write failed: {exc}"}
        return

    # ── Phase 9: Register PKs ─────────────────────────────────────────────
    yield {"status": "running", "progress": 96, "message": "Registering primary keys…"}
    for pk_col in config.primary_key_cols:
        if pk_col in synth_df.columns:
            try:
                await _registry.register_ids(
                    config.table_fqn, pk_col, synth_df[pk_col].tolist()
                )
            except Exception as exc:
                logger.error("pk_registration_failed", pk_col=pk_col, error=str(exc))

    total_s = time.monotonic() - t_start
    logger.info(
        "generation_complete",
        table=config.table_fqn,
        rows=len(synth_df),
        strategy=strategy,
        used_cache=used_cache,
        total_s=round(total_s, 1),
    )

    yield {
        "status": "completed",
        "progress": 100,
        "message": f"Generation complete: {len(synth_df):,} rows in {total_s:.1f}s",
        "result": {
            "table_fqn": config.table_fqn,
            "rows_generated": len(synth_df),
            "strategy": strategy,
            "used_cache": used_cache,
            "artifact_key": cached_artifact_key,
            "duration_s": round(total_s, 1),
            **write_result,
        },
    }


# ---------------------------------------------------------------------------
# ADK tool wrappers
# ---------------------------------------------------------------------------

# Synchronous plan tool — registered as a standard @tool via function reference
from google.adk.tools import tool as _adk_tool  # type: ignore[import]
plan_generation_tool = _adk_tool(plan_generation)

# Long-running training + generation tool
train_and_generate_tool = LongRunningFunctionTool(func=train_and_generate)


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="generator_agent",
    model="gemini-2.5-flash",
    description=(
        "Synthetic data generation specialist.  Selects the best ML strategy "
        "(CTGAN / TVAE / Copula / TimeGAN) from data characteristics and quality "
        "history, loads cached models from the artifact store when available, and "
        "trains new models for first-run or schema-changed tables.  Generates "
        "non-PII columns, resolves FK references, and registers PKs."
    ),
    instruction="""You are a synthetic data generation specialist.

For each table, follow this exact sequence:

STEP 1 — PLAN: Call plan_generation_tool with the TableGenConfig and TableProfile.
  - Inspect the returned strategy, cache_hit, and run_count.
  - If cache_hit is True, you can skip training and use the cached_artifact_key.

STEP 2 — GENERATE: Call train_and_generate_tool with:
  - The TableGenConfig and TableProfile
  - The recommended strategy from STEP 1
  - The cached_artifact_key if cache_hit was True (otherwise pass null)

  This is a long-running operation.  Monitor progress updates:
  - status="starting" → report table name and strategy to user
  - status="running"  → relay progress percentage and message
  - status="completed" → extract result dict and return to orchestrator
  - status="failed"   → report error and return failure result

STEP 3 — REPORT: Return the result dict (rows_generated, strategy, used_cache,
  artifact_key, duration_s) to the orchestrator for quality gate evaluation.

Strategy guidance (used when no history exists):
  - ctgan:   mixed-type tables, complex categoricals (default)
  - tvae:    high-fidelity behavioural data, numeric-heavy tables
  - timegan: any table with a temporal sequence key column
  - copula:  small datasets, simple numeric tables, rapid prototyping""",
    tools=[plan_generation_tool, train_and_generate_tool],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
