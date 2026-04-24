from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import structlog
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer

from ..config import get_settings

logger = structlog.get_logger()


class TimeGANTrainer:
    """Trainer for temporal/sequential tabular data using SDV PARSynthesizer.

    Note: PAR generates *sequences* rather than flat rows.  The ``n_rows``
    parameter in :meth:`sample` is treated as the number of sequences to
    generate.  Callers that need a specific flat row count should divide by
    the expected average sequence length before calling.
    """

    def __init__(
        self,
        table_fqn: str,
        metadata: Optional[SingleTableMetadata] = None,
        sequence_key: Optional[str] = None,
        context_columns: Optional[list[str]] = None,
    ) -> None:
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.sequence_key = sequence_key
        self.context_columns: list[str] = context_columns or []
        self.model: Optional[PARSynthesizer] = None

    def train(self, df: pd.DataFrame, epochs: int = get_settings().ctgan_epochs) -> None:
        """Fit PARSynthesizer on sequential training data.

        Args:
            df: Training DataFrame that must include the sequence key column.
            epochs: Number of training epochs.
        """
        logger.info("Training PARSynthesizer (TimeGAN)", table=self.table_fqn, rows=len(df), epochs=epochs)

        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)
            if self.sequence_key:
                self.metadata.update_column(column_name=self.sequence_key, sdtype="id")
                self.metadata.set_primary_key(column_name=self.sequence_key)

        self.model = PARSynthesizer(
            self.metadata,
            context_columns=self.context_columns,
            epochs=epochs,
            cuda=True,
        )
        self.model.fit(df)
        logger.info("PARSynthesizer training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate ``n_rows`` synthetic sequences.

        Args:
            n_rows: Number of sequences to generate (not flat row count).

        Returns:
            DataFrame containing generated sequences.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        logger.info("Sampling from PARSynthesizer", table=self.table_fqn, n_sequences=n_rows)
        return self.model.sample(num_sequences=max(1, n_rows))

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Destination file path.
        """
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info("PARSynthesizer model saved", path=str(path))

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> TimeGANTrainer:
        """Load a previously saved model from disk.

        Args:
            table_fqn: Table the model was trained on.
            path: Path to the saved model file.

        Returns:
            Loaded TimeGANTrainer instance.
        """
        trainer = cls(table_fqn)
        trainer.model = PARSynthesizer.load(path)
        logger.info("PARSynthesizer model loaded", path=str(path))
        return trainer
