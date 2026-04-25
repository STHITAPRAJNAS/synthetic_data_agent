"""Production Gaussian Copula trainer.

The Gaussian Copula is the fastest and most interpretable strategy.  It models
each column's marginal distribution independently, then links them using a
Gaussian copula to preserve pairwise correlations.

Best for
---------
- Simple numeric tables (≤ 12 columns, mostly continuous)
- Small datasets (< 500 rows — deep models overfit)
- Rapid prototyping where training speed matters
- When you need to understand WHY the model generates certain distributions

Not suitable for
-----------------
- Tables with complex multi-modal categoricals (use CTGAN)
- High-cardinality string columns (use TVAE)
- Temporal sequences (use TimeGAN)
"""
from __future__ import annotations

import io
import time
from typing import Any

import pandas as pd
import structlog
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from .base import SynthesisModel, TrainingConfig, TrainingResult

logger = structlog.get_logger()


class CopulaTrainer(SynthesisModel):
    """Gaussian Copula wrapper implementing the ``SynthesisModel`` contract."""

    strategy = "copula"

    def __init__(self) -> None:
        self._model: GaussianCopulaSynthesizer | None = None
        self._training_result: TrainingResult | None = None

    # ------------------------------------------------------------------
    # SynthesisModel implementation
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
        """Fit a Gaussian Copula to *df*.

        The Copula does not use epochs or GPU — it fits marginal distributions
        and a correlation matrix analytically.  Training is fast even on large
        datasets but quality plateaus for complex distributions.

        Args:
            df: Training DataFrame.
            config: Only ``random_state`` and ``max_training_rows`` are used.

        Returns:
            TrainingResult.
        """
        self._validate_training_df(df, min_rows=5)  # Copula can fit tiny datasets

        if len(df) > config.max_training_rows:
            logger.warning(
                "copula_training_rows_capped",
                original=len(df),
                cap=config.max_training_rows,
            )
            df = df.sample(config.max_training_rows, random_state=config.random_state)

        logger.info("copula_training_start", rows=len(df), cols=len(df.columns))
        t0 = time.monotonic()

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        self._model = GaussianCopulaSynthesizer(metadata)
        self._model.fit(df)

        duration = time.monotonic() - t0
        logger.info("copula_training_complete", duration_s=round(duration, 1))

        self._training_result = TrainingResult(
            strategy=self.strategy,
            training_rows=len(df),
            training_cols=len(df.columns),
            epochs_run=1,  # Copula fits analytically in one pass
            training_duration_s=duration,
        )
        return self._training_result

    def sample(
        self,
        n_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate *n_rows* synthetic rows."""
        if self._model is None:
            raise RuntimeError("CopulaTrainer: call train() before sample().")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be positive, got {n_rows}.")

        logger.info("copula_sampling", n_rows=n_rows)

        if conditions:
            from sdv.sampling import Condition  # type: ignore[import]
            cond = Condition(num_rows=n_rows, column_values=conditions)
            return self._model.sample_from_conditions(conditions=[cond])

        return self._model.sample(num_rows=n_rows)

    def to_bytes(self) -> bytes:
        if self._model is None:
            raise RuntimeError("CopulaTrainer: model has not been trained — cannot serialize.")
        buf = io.BytesIO()
        self._model.save(buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CopulaTrainer":
        trainer = cls()
        buf = io.BytesIO(data)
        trainer._model = GaussianCopulaSynthesizer.load(buf)
        logger.info("copula_model_loaded_from_bytes")
        return trainer
