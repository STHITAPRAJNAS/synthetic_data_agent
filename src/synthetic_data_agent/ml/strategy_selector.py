"""Intelligent strategy selection based on dataset characteristics.

The selector examines the TableProfile and answers: "which synthesis model
will most likely produce the highest-quality output for this particular table?"

Decision rules (applied in priority order)
-------------------------------------------
1. Temporal column present → ``timegan``  (PAR preserves time dependencies)
2. Very small dataset (< 500 rows) → ``copula``  (deep models overfit)
3. High cardinality + many columns + large dataset → ``tvae``  (VAE generalises)
4. Mostly categorical / mixed types → ``ctgan``  (GAN handles mode collapse)
5. Mostly numeric + low column count → ``copula``  (fast, interpretable)
6. Default → ``ctgan``

The ModelRegistry overrides this selection if it has quality history showing
a different strategy performed better for the same table in the past.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import structlog

from ..models.column_profile import DistributionType, PIICategory, TableProfile

logger = structlog.get_logger()

Strategy = Literal["ctgan", "tvae", "copula", "timegan"]

_DEEP_MODEL_MIN_ROWS = 500
_TVAE_MIN_ROWS = 5_000
_LARGE_DATASET_ROWS = 50_000


@dataclass
class StrategyDecision:
    strategy: Strategy
    reason: str
    confidence: float  # 0–1, how confident the selector is


def select_strategy(profile: TableProfile) -> StrategyDecision:
    """Choose the best synthesis strategy for *profile*.

    Args:
        profile: Profiled TableProfile from the profiler agent.

    Returns:
        StrategyDecision with the recommended strategy and reasoning.
    """
    n_rows = profile.row_count
    n_cols = len(profile.columns)
    safe_cols = [c for c in profile.columns if c.pii_category in (None, PIICategory.SAFE)]

    # ── Rule 1: Temporal data ───────────────────────────────────────────────
    if profile.temporal_col:
        return StrategyDecision(
            strategy="timegan",
            reason=f"Temporal column '{profile.temporal_col}' detected — PAR preserves time dependencies.",
            confidence=0.95,
        )

    # ── Rule 2: Very small dataset ───────────────────────────────────────────
    if n_rows < _DEEP_MODEL_MIN_ROWS:
        return StrategyDecision(
            strategy="copula",
            reason=f"Only {n_rows} rows — deep models overfit; Gaussian Copula is more stable.",
            confidence=0.9,
        )

    # ── Compute column type ratios ───────────────────────────────────────────
    num_count = sum(
        1 for c in safe_cols
        if c.distribution_type in {
            DistributionType.GAUSSIAN,
            DistributionType.LOG_NORMAL,
            DistributionType.UNIFORM,
        }
    )
    cat_count = sum(
        1 for c in safe_cols
        if c.distribution_type in {
            DistributionType.CATEGORICAL,
            DistributionType.BOOLEAN,
        }
    )
    high_card_count = sum(
        1 for c in safe_cols
        if c.distribution_type in {
            DistributionType.HIGH_CARD_STRING,
            DistributionType.JSON,
        }
    )

    safe_n = max(len(safe_cols), 1)
    num_ratio = num_count / safe_n
    cat_ratio = cat_count / safe_n
    high_card_ratio = high_card_count / safe_n

    avg_cardinality = (
        sum(c.cardinality for c in safe_cols) / safe_n if safe_cols else 0
    )

    logger.debug(
        "strategy_selector_features",
        table=profile.table_fqn,
        n_rows=n_rows,
        n_cols=n_cols,
        num_ratio=round(num_ratio, 2),
        cat_ratio=round(cat_ratio, 2),
        high_card_ratio=round(high_card_ratio, 2),
        avg_cardinality=round(avg_cardinality, 1),
    )

    # ── Rule 3: High-cardinality strings → TVAE generalises better ──────────
    if high_card_ratio > 0.4 and n_rows >= _TVAE_MIN_ROWS:
        return StrategyDecision(
            strategy="tvae",
            reason=(
                f"{high_card_ratio:.0%} of columns are high-cardinality strings; "
                "TVAE latent space generalises better than GAN discriminator for such data."
            ),
            confidence=0.8,
        )

    # ── Rule 4: Large dataset with complex mixed types → TVAE ───────────────
    if n_rows >= _LARGE_DATASET_ROWS and n_cols >= 15 and num_ratio > 0.3:
        return StrategyDecision(
            strategy="tvae",
            reason=(
                f"Large dataset ({n_rows:,} rows, {n_cols} columns) with "
                f"{num_ratio:.0%} numeric columns — TVAE handles this class well."
            ),
            confidence=0.75,
        )

    # ── Rule 5: Mostly categorical / mixed → CTGAN ──────────────────────────
    if cat_ratio > 0.5 or (cat_ratio + num_ratio > 0.8 and n_rows >= _DEEP_MODEL_MIN_ROWS):
        return StrategyDecision(
            strategy="ctgan",
            reason=(
                f"{cat_ratio:.0%} categorical columns — CTGAN's conditional generator "
                "handles multi-modal categorical distributions well."
            ),
            confidence=0.8,
        )

    # ── Rule 6: Mostly numeric + few columns → Copula ───────────────────────
    if num_ratio > 0.7 and n_cols <= 12:
        return StrategyDecision(
            strategy="copula",
            reason=(
                f"{num_ratio:.0%} numeric columns, {n_cols} total — "
                "Gaussian Copula is fast and captures pairwise correlations accurately."
            ),
            confidence=0.7,
        )

    # ── Default ─────────────────────────────────────────────────────────────
    return StrategyDecision(
        strategy="ctgan",
        reason="Default selection — CTGAN handles general mixed-type tabular data.",
        confidence=0.6,
    )


def adapt_training_config_for_profile(
    profile: TableProfile,
    strategy: Strategy,
) -> dict[str, object]:
    """Return recommended TrainingConfig overrides based on profile characteristics.

    The generator agent merges these into its base TrainingConfig before
    training so hyperparameters are tuned to the actual data.
    """
    n_rows = profile.row_count
    n_cols = len(profile.columns)
    overrides: dict[str, object] = {}

    # Adaptive epoch count — small data needs more passes to converge
    if n_rows < 1_000:
        overrides["epochs"] = 500
    elif n_rows < 10_000:
        overrides["epochs"] = 400
    elif n_rows < 100_000:
        overrides["epochs"] = 300
    else:
        overrides["epochs"] = 200  # large data converges faster

    # Wider networks for wider tables
    if n_cols > 30:
        overrides["generator_dim"] = (512, 512, 256)
        overrides["discriminator_dim"] = (512, 512, 256)
    elif n_cols > 15:
        overrides["generator_dim"] = (256, 256)
        overrides["discriminator_dim"] = (256, 256)

    # Larger batch on larger data for training stability
    if n_rows > 100_000:
        overrides["batch_size"] = 1000
    elif n_rows > 10_000:
        overrides["batch_size"] = 500
    else:
        overrides["batch_size"] = 256

    # Sequence key hint for TimeGAN
    if strategy == "timegan" and profile.temporal_col:
        overrides["sequence_key"] = profile.temporal_col

    return overrides
