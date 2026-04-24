import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path
from typing import Optional
from ..config import settings
import structlog

logger = structlog.get_logger()

class CTGANTrainer:
    def __init__(self, table_fqn: str, metadata: Optional[SingleTableMetadata] = None):
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.model: Optional[CTGANSynthesizer] = None

    def train(self, df: pd.DataFrame, epochs: int = settings.ctgan_epochs):
        """Train CTGAN model on the provided dataframe."""
        logger.info("Training CTGAN", table=self.table_fqn, rows=len(df), epochs=epochs)
        
        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)
            
        self.model = CTGANSynthesizer(
            self.metadata,
            epochs=epochs,
            verbose=True,
            cuda=True # Use GPU if available
        )
        
        self.model.fit(df)
        logger.info("CTGAN training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic rows."""
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        
        logger.info("Sampling from CTGAN", table=self.table_fqn, n_rows=n_rows)
        return self.model.sample(num_rows=n_rows)

    def save(self, path: Path):
        """Save model to disk."""
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info("CTGAN model saved", path=str(path))

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> 'CTGANTrainer':
        """Load model from disk."""
        trainer = cls(table_fqn)
        trainer.model = CTGANSynthesizer.load(path)
        logger.info("CTGAN model loaded", path=str(path))
        return trainer
