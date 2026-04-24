from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

from ..config import get_settings

logger = structlog.get_logger()


class TVAETrainer:
    """SDV TVAE wrapper with latent-space encoding and interpolation support."""

    def __init__(
        self,
        table_fqn: str,
        metadata: Optional[SingleTableMetadata] = None,
    ) -> None:
        self.table_fqn = table_fqn
        self.metadata = metadata or SingleTableMetadata()
        self.model: Optional[TVAESynthesizer] = None
        # Store the fitted data transformer for encoding
        self._fitted_df: Optional[pd.DataFrame] = None

    def train(self, df: pd.DataFrame, epochs: int = get_settings().ctgan_epochs) -> None:
        """Train TVAE model on the provided DataFrame.

        Args:
            df: Training data (non-PII columns only).
            epochs: Number of training epochs.
        """
        logger.info("Training TVAE", table=self.table_fqn, rows=len(df), epochs=epochs)

        if not self.metadata.is_initialized():
            self.metadata.detect_from_dataframe(df)

        self.model = TVAESynthesizer(
            self.metadata,
            epochs=epochs,
            cuda=True,
        )
        self.model.fit(df)
        self._fitted_df = df.copy()
        logger.info("TVAE training complete", table=self.table_fqn)

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic rows.

        Args:
            n_rows: Number of rows to generate.

        Returns:
            DataFrame with n_rows synthetic rows.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before sampling")
        logger.info("Sampling from TVAE", table=self.table_fqn, n_rows=n_rows)
        return self.model.sample(num_rows=n_rows)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a DataFrame into TVAE latent vectors.

        Args:
            df: DataFrame with the same schema as training data.

        Returns:
            numpy array of shape (n_rows, latent_dim).
        """
        if not self.model:
            raise RuntimeError("Model must be trained before encoding")

        # SDV's TVAESynthesizer exposes the underlying synthesizer via ._model
        # We use the data transformer to convert to numeric, then the encoder
        try:
            transformed = self.model._data_processor.transform(df)  # type: ignore[attr-defined]
            import torch  # type: ignore[import]
            tensor = torch.tensor(transformed.values, dtype=torch.float32)
            with torch.no_grad():
                mu, _ = self.model._model.encoder(tensor)  # type: ignore[attr-defined]
            return mu.numpy()
        except AttributeError as exc:
            logger.warning("TVAE encoder access failed, falling back to zeros", error=str(exc))
            latent_dim = 128
            return np.zeros((len(df), latent_dim), dtype=np.float32)

    def interpolate(self, z1: np.ndarray, z2: np.ndarray, steps: int = 5) -> list[pd.DataFrame]:
        """Generate synthetic data interpolated between two latent vectors.

        Useful for producing behaviorally similar but distinct synthetic records.

        Args:
            z1: Source latent vector, shape (latent_dim,).
            z2: Target latent vector, shape (latent_dim,).
            steps: Number of interpolation steps.

        Returns:
            List of DataFrames, one per interpolation step.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before interpolation")

        import torch  # type: ignore[import]

        results: list[pd.DataFrame] = []
        for alpha in np.linspace(0.0, 1.0, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            tensor = torch.tensor(z_interp[None, :], dtype=torch.float32)
            try:
                with torch.no_grad():
                    decoded = self.model._model.decoder(tensor)  # type: ignore[attr-defined]
                decoded_np = decoded[0].numpy() if isinstance(decoded, tuple) else decoded.numpy()
                # Inverse-transform back to original schema
                cols = self.model._data_processor.get_transformed_columns()  # type: ignore[attr-defined]
                decoded_df = pd.DataFrame(decoded_np, columns=cols)
                original = self.model._data_processor.reverse_transform(decoded_df)  # type: ignore[attr-defined]
                results.append(original)
            except Exception as exc:
                logger.warning("Interpolation step failed", step=alpha, error=str(exc))
                results.append(self.sample(1))
        return results

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Destination file path (e.g. ml_models/my_table/tvae.pkl).
        """
        if self.model:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info("TVAE model saved", path=str(path))

    @classmethod
    def load(cls, table_fqn: str, path: Path) -> TVAETrainer:
        """Load a previously saved model from disk.

        Args:
            table_fqn: Table the model was trained on.
            path: Path to the saved model file.

        Returns:
            Loaded TVAETrainer instance.
        """
        trainer = cls(table_fqn)
        trainer.model = TVAESynthesizer.load(path)
        logger.info("TVAE model loaded", path=str(path))
        return trainer
