"""Production TimeGAN / PAR trainer for temporal sequential data.

Uses SDV's PARSynthesizer (Probabilistic AutoRegressive model) which is the
production-grade approach for time-series tabular data.  PAR models:
- Inter-row temporal dependencies (e.g. account balance changes over time)
- Context columns that shape a whole sequence (e.g. customer tier)
- Variable-length sequences

Best for
---------
- Transaction logs with a customer/account sequence key
- IoT sensor readings grouped by device
- Medical visits grouped by patient
- Any table where row ORDER within a group matters

How to configure
-----------------
Set ``config.sequence_key`` to the column that identifies each sequence
(e.g. ``"customer_id"``).  Optionally set ``config.context_columns`` to
columns that are constant within a sequence (e.g. ``"customer_tier"``).

The ``sample`` method generates *n_rows* complete sequences.  The actual
output row count may differ if sequences have variable lengths.
"""
from __future__ import annotations

import io
import time
from typing import Any

import pandas as pd
import structlog
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer

from .base import SynthesisModel, TrainingConfig, TrainingResult

logger = structlog.get_logger()


class TimeGANTrainer(SynthesisModel):
    """PARSynthesizer wrapper implementing the ``SynthesisModel`` contract."""

    strategy = "timegan"

    def __init__(self) -> None:
        self._model: PARSynthesizer | None = None
        self._sequence_key: str | None = None
        self._context_columns: list[str] = []
        self._training_result: TrainingResult | None = None

    # ------------------------------------------------------------------
    # SynthesisModel implementation
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
        """Fit PARSynthesizer on sequential training data.

        Args:
            df: Training DataFrame.  Must include ``config.sequence_key`` if set.
            config: Hyperparameters including sequence_key and context_columns.

        Returns:
            TrainingResult.

        Raises:
            ValueError: If sequence_key specified but not present in df.
        """
        self._validate_training_df(df, min_rows=config.min_rows_for_deep_model)

        self._sequence_key = config.sequence_key
        self._context_columns = list(config.context_columns)

        if self._sequence_key and self._sequence_key not in df.columns:
            raise ValueError(
                f"TimeGANTrainer: sequence_key='{self._sequence_key}' not found in "
                f"DataFrame columns {list(df.columns)}."
            )

        if len(df) > config.max_training_rows:
            logger.warning(
                "timegan_training_rows_capped",
                original=len(df),
                cap=config.max_training_rows,
            )
            # For sequential data: keep whole sequences rather than sampling rows
            if self._sequence_key:
                seq_ids = df[self._sequence_key].unique()
                keep = seq_ids[:config.max_training_rows // max(1, len(df) // max(len(seq_ids), 1))]
                df = df[df[self._sequence_key].isin(keep)]
            else:
                df = df.head(config.max_training_rows)

        use_cuda = (
            config.use_cuda
            if config.use_cuda is not None
            else self._detect_cuda()
        )

        logger.info(
            "timegan_training_start",
            rows=len(df),
            cols=len(df.columns),
            epochs=config.epochs,
            sequence_key=self._sequence_key,
            context_columns=self._context_columns,
            cuda=use_cuda,
        )
        t0 = time.monotonic()

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        if self._sequence_key:
            metadata.update_column(column_name=self._sequence_key, sdtype="id")
            metadata.set_sequence_key(column_name=self._sequence_key)

        self._model = PARSynthesizer(
            metadata,
            context_columns=self._context_columns,
            epochs=config.epochs,
            cuda=use_cuda,
        )
        self._model.fit(df)

        duration = time.monotonic() - t0
        logger.info("timegan_training_complete", duration_s=round(duration, 1))

        self._training_result = TrainingResult(
            strategy=self.strategy,
            training_rows=len(df),
            training_cols=len(df.columns),
            epochs_run=config.epochs,
            training_duration_s=duration,
            extra={
                "sequence_key": self._sequence_key,
                "context_columns": self._context_columns,
            },
        )
        return self._training_result

    def sample(
        self,
        n_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate *n_rows* sequences.

        Note: *n_rows* here means sequences, not flat rows.  The returned
        DataFrame may have more rows if sequences have length > 1.

        Args:
            n_rows: Number of sequences to generate.
            conditions: Ignored for PAR (use context_columns at train time).

        Returns:
            DataFrame of generated sequences.

        Raises:
            RuntimeError: If not trained.
            ValueError: If n_rows is not positive.
        """
        if self._model is None:
            raise RuntimeError("TimeGANTrainer: call train() before sample().")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be positive, got {n_rows}.")

        logger.info("timegan_sampling", n_sequences=n_rows)
        return self._model.sample(num_sequences=n_rows)

    def to_bytes(self) -> bytes:
        if self._model is None:
            raise RuntimeError("TimeGANTrainer: model has not been trained — cannot serialize.")
        payload = {
            "sequence_key": self._sequence_key,
            "context_columns": self._context_columns,
        }
        buf = io.BytesIO()
        self._model.save(buf)
        import pickle
        return pickle.dumps({"model_bytes": buf.getvalue(), "meta": payload})

    @classmethod
    def from_bytes(cls, data: bytes) -> "TimeGANTrainer":
        import pickle
        payload = pickle.loads(data)  # noqa: S301 — trusted internal artifact
        trainer = cls()
        trainer._sequence_key = payload["meta"].get("sequence_key")
        trainer._context_columns = payload["meta"].get("context_columns", [])
        buf = io.BytesIO(payload["model_bytes"])
        trainer._model = PARSynthesizer.load(buf)
        logger.info("timegan_model_loaded_from_bytes")
        return trainer
