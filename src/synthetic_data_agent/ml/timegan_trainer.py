import pandas as pd
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path
from typing import Optional
from ..config import settings
import structlog

logger = structlog.get_logger()

class TimeGANTrainer:
    """Trainer for temporal/sequential tabular data using SDV PARSynthesizer."""
    def __init__(self, table_fqn: str, metadata: Optional[SingleTableMetadata] = None, sequence_key: str = None, context_columns: list[str] = None):
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.sequence_key = sequence_key
        self.context_columns = context_columns or []
        self.model: Optional[PARSynthesizer] = None

    def train(self, df: pd.DataFrame, epochs: int = settings.ctgan_epochs):
        """Train sequential model on the provided dataframe."""
        logger.info("Training TimeGAN (PARSynthesizer)", table=self.table_fqn, rows=len(df), epochs=epochs)
        
        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)
            if self.sequence_key:
                self.metadata.update_column(column_name=self.sequence_key, sdtype='id')
                self.metadata.set_primary_key(column_name=self.sequence_key)
            
        self.model = PARSynthesizer(
            self.metadata,
            context_columns=self.context_columns,
            epochs=epochs,
            cuda=True
        )
        
        self.model.fit(df)
        logger.info("TimeGAN training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic sequences."""
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        
        logger.info("Sampling from TimeGAN", table=self.table_fqn, n_rows=n_rows)
        # Assuming n_rows relates to the number of sequences or total rows
        # For PAR, usually we sample by context or number of sequences
        return self.model.sample(num_sequences=max(1, n_rows // 10))

    def save(self, path: Path):
        """Save model to disk."""
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info("TimeGAN model saved", path=str(path))

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> 'TimeGANTrainer':
        """Load model from disk."""
        trainer = cls(table_fqn)
        trainer.model = PARSynthesizer.load(path)
        logger.info("TimeGAN model loaded", path=str(path))
        return trainer
