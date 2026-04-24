import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger()

class CopulaTrainer:
    def __init__(self, table_fqn: str, metadata: Optional[SingleTableMetadata] = None):
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.model: Optional[GaussianCopulaSynthesizer] = None

    def train(self, df: pd.DataFrame):
        """Train Gaussian Copula model."""
        logger.info("Training Gaussian Copula", table=self.table_fqn, rows=len(df))
        
        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)
            
        self.model = GaussianCopulaSynthesizer(self.metadata)
        self.model.fit(df)
        logger.info("Gaussian Copula training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        return self.model.sample(num_rows=n_rows)

    def save(self, path: Path):
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> 'CopulaTrainer':
        trainer = cls(table_fqn)
        trainer.model = GaussianCopulaSynthesizer.load(path)
        return trainer
