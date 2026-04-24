from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

import pandas as pd
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
from ....ml.copula_trainer import CopulaTrainer
from ....ml.ctgan_trainer import CTGANTrainer
from ....ml.timegan_trainer import TimeGANTrainer
from ....ml.tvae_trainer import TVAETrainer
from ....models.generation_plan import TableGenConfig
from ....tools.databricks_tools import DatabricksTools
from ....tools.knowledge_base import KnowledgeBase
from ....tools.registry_tools import SyntheticIDRegistry

logger = structlog.get_logger()

_db_tools = DatabricksTools()
_registry = SyntheticIDRegistry()
_knowledge_base = KnowledgeBase()


# ---------------------------------------------------------------------------
# Trainer protocol — enables duck-typed dispatch without bare Any
# ---------------------------------------------------------------------------

@runtime_checkable
class _Trainer(Protocol):
    def train(self, df: pd.DataFrame) -> None: ...
    def sample(self, n_rows: int) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# ADK tool
# ---------------------------------------------------------------------------

@tool
async def generate_table_data(config_json: dict[str, Any]) -> dict[str, Any]:
    """Train an ML model and generate synthetic non-PII data for a single table.

    Resolves all FK columns from the SyntheticIDRegistry to guarantee referential
    integrity.  Registers generated PKs so child tables can reference them.

    Args:
        config_json: Serialised TableGenConfig dict.

    Returns:
        Dict with rows_written, table_fqn, and write_timestamp.
    """
    cfg = get_settings()
    config = TableGenConfig.model_validate(config_json)
    logger.info("Generating data", table=config.table_fqn, strategy=config.ml_strategy)

    # 1. Fetch training sample (non-PII columns only)
    real_df = await _db_tools.sample_dataframe(config.table_fqn, cfg.max_profiling_sample_rows)
    train_cols = [c for c in config.non_pii_columns if c in real_df.columns]
    if not train_cols:
        raise ValueError(f"No non-PII columns found for table {config.table_fqn}")
    train_df = real_df[train_cols]

    # 2. Select and train model — training is CPU-bound, run in thread
    trainer: _Trainer
    if config.ml_strategy == "tvae":
        _t = TVAETrainer(config.table_fqn)
    elif config.ml_strategy == "copula":
        _t = CopulaTrainer(config.table_fqn)
    elif config.ml_strategy == "timegan":
        _t = TimeGANTrainer(config.table_fqn)
    else:
        _t = CTGANTrainer(config.table_fqn)
    trainer = _t

    await asyncio.to_thread(trainer.train, train_df)
    logger.info("Model training complete", table=config.table_fqn, strategy=config.ml_strategy)

    # 3. Generate ~50% extra rows so business rule filtering doesn't leave us short
    target = config.target_row_count
    synth_df = await asyncio.to_thread(trainer.sample, int(target * 1.5))

    # 4. Apply business rules
    rules = await _knowledge_base.get_business_rules(config.table_fqn)
    for rule in rules:
        logger.info("Applying business rule", table=config.table_fqn, rule=rule["description"])
        try:
            synth_df = synth_df.query(rule["code"])
        except Exception as exc:
            logger.error("Failed to apply rule", rule=rule["description"], error=str(exc))

    # Trim to target
    if len(synth_df) > target:
        synth_df = synth_df.head(target)

    # 5. Resolve FK columns from SyntheticIDRegistry
    for fk in config.foreign_keys:
        if fk.fk_col not in synth_df.columns:
            continue
        parent_ids = await _registry.sample_fk(
            fk.parent_table_fqn, fk.parent_pk_col, len(synth_df)
        )
        if parent_ids:
            synth_df[fk.fk_col] = parent_ids[: len(synth_df)]
            logger.debug(
                "Resolved FK",
                table=config.table_fqn,
                fk_col=fk.fk_col,
                parent=fk.parent_table_fqn,
            )
        else:
            logger.warning(
                "No parent IDs available for FK — parent table not yet generated?",
                table=config.table_fqn,
                fk_col=fk.fk_col,
                parent=fk.parent_table_fqn,
            )

    # 6. Write non-PII output
    output_fqn = _output_fqn(config.table_fqn)
    result = await _db_tools.write_synthetic_table(output_fqn, synth_df)

    # 7. Register PKs so child tables can reference them
    for pk_col in config.primary_key_cols:
        if pk_col in synth_df.columns:
            await _registry.register_ids(
                config.table_fqn, pk_col, synth_df[pk_col].tolist()
            )

    return result


def _output_fqn(source_fqn: str) -> str:
    cfg = get_settings()
    table_name = source_fqn.split(".")[-1]
    return f"{cfg.output_catalog}.{cfg.databricks_schema}.{table_name}"


# ---------------------------------------------------------------------------
# ADK root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="generator_agent",
    model="gemini-2.5-flash",
    description=(
        "Synthetic data generation specialist that trains ML models (CTGAN, TVAE, "
        "Copula, TimeGAN) on real table samples, generates non-PII columns, resolves "
        "FK references from the ID registry, and writes output to Databricks."
    ),
    instruction="""You are a synthetic data generation specialist.

For each table in the generation plan:
1. Call generate_table_data with the serialised TableGenConfig.
2. The tool trains the selected ML model, generates data, resolves FKs, and writes to Databricks.
3. Report any tables that failed so the orchestrator can trigger self-correction.

Strategy selection guidance:
- ctgan: mixed-type tables with complex distributions (default)
- tvae: high-fidelity behavioural / account-level tables
- timegan: tables with a temporal sequence key
- copula: simple tables where interpretability matters more than fidelity""",
    tools=[generate_table_data],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
