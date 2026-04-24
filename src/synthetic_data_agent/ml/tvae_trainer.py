import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path
from typing import Optional
from ..config import settings
import structlog

logger = structlog.get_logger()

class TVAETrainer:
    def __init__(self, table_fqn: str, metadata: Optional[SingleTableMetadata] = None):
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.model: Optional[TVAESynthesizer] = None

    def train(self, df: pd.DataFrame, epochs: int = settings.ctgan_epochs):
        """Train TVAE model on the provided dataframe."""
        logger.info("Training TVAE", table=self.table_fqn, rows=len(df), epochs=epochs)
        
        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)
            
        self.model = TVAESynthesizer(
            self.metadata,
            epochs=epochs,
            cuda=True
        )
        
        self.model.fit(df)
        logger.info("TVAE training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic rows."""
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        
        logger.info("Sampling from TVAE", table=self.table_fqn, n_rows=n_rows)
        return self.model.sample(num_rows=n_rows)

    def save(self, path: Path):
        """Save model to disk."""
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info("TVAE model saved", path=str(path))

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> 'TVAETrainer':
        """Load model from disk."""
        trainer = cls(table_fqn)
        trainer.model = TVAESynthesizer.load(path)
        logger.info("TVAE model loaded", path=str(path))
        return trainer
