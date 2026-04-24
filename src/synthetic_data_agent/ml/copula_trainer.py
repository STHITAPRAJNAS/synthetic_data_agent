from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import structlog
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

logger = structlog.get_logger()


class CopulaTrainer:
    """Gaussian Copula wrapper — fast, interpretable path for simple tables."""

    def __init__(
        self,
        table_fqn: str,
        metadata: Optional[SingleTableMetadata] = None,
    ) -> None:
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.model: Optional[GaussianCopulaSynthesizer] = None

    def train(self, df: pd.DataFrame) -> None:
        """Fit a Gaussian Copula to the training data.

        Args:
            df: Training DataFrame (non-PII columns only).
        """
        logger.info("Training Gaussian Copula", table=self.table_fqn, rows=len(df))
        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)
        self.model = GaussianCopulaSynthesizer(self.metadata)
        self.model.fit(df)
        logger.info("Gaussian Copula training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic rows.

        Args:
            n_rows: Number of rows to generate.

        Returns:
            DataFrame with n_rows synthetic rows.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        return self.model.sample(num_rows=n_rows)

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Destination file path.
        """
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info("Copula model saved", path=str(path))

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> CopulaTrainer:
        """Load a previously saved model from disk.

        Args:
            table_fqn: Table the model was trained on.
            path: Path to the saved model file.

        Returns:
            Loaded CopulaTrainer instance.
        """
        trainer = cls(table_fqn)
        trainer.model = GaussianCopulaSynthesizer.load(path)
        logger.info("Copula model loaded", path=str(path))
        return trainer
