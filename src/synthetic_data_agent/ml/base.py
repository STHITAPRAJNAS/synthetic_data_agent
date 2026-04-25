"""Abstract base for all synthesis models.

Every model implementation (CTGAN, TVAE, Copula, TimeGAN) must satisfy this
interface.  Callers only ever depend on ``SynthesisModel`` — making it trivial
to swap or extend strategies without touching agent code.

Serialization contract
-----------------------
``to_bytes()`` must produce a self-contained binary blob that ``from_bytes()``
can reconstruct on a different process / machine that has the same package
versions installed.  The ADK artifact store uses these blobs verbatim.
"""
from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Typed training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters and hints for a single training run.

    Defaults are chosen to be appropriate for medium-scale datasets (~50 k rows).
    The strategy selector and model registry may override them.
    """
    epochs: int = 300
    batch_size: int = 500
    embedding_dim: int = 128
    generator_dim: tuple[int, ...] = (256, 256)
    discriminator_dim: tuple[int, ...] = (256, 256)
    # Whether to attempt GPU acceleration (auto-detected if None)
    use_cuda: bool | None = None
    # Conditional columns — model will preserve their distributions exactly
    condition_on: list[str] = field(default_factory=list)
    # Sequence key for temporal models (TimeGAN / PAR)
    sequence_key: str | None = None
    context_columns: list[str] = field(default_factory=list)
    # Maximum rows per training batch (limits peak memory)
    max_training_rows: int = 200_000
    # Minimum acceptable row count to fit a deep model
    min_rows_for_deep_model: int = 100
    # Random seed for reproducibility
    random_state: int = 42

    def fingerprint(self) -> str:
        """SHA-256 of the config dict — used as part of artifact cache keys."""
        raw = str(sorted(self.__dict__.items())).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Metadata produced after a successful ``SynthesisModel.train()`` call."""
    strategy: str
    trained_at: datetime = field(default_factory=datetime.utcnow)
    training_rows: int = 0
    training_cols: int = 0
    epochs_run: int = 0
    training_duration_s: float = 0.0
    # Self-reported quality hint (0-1) — used by ModelRegistry for ranking
    quality_hint: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SynthesisModel(ABC):
    """Contract every synthesis model must implement."""

    strategy: str  # class-level constant — set by each subclass

    @abstractmethod
    def train(self, df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
        """Fit the model on *df*.  Must be called before ``sample``.

        Args:
            df: Training DataFrame (non-PII columns only, already cleaned).
            config: Hyperparameters and hints for this run.

        Returns:
            TrainingResult with metadata about the completed run.

        Raises:
            ValueError: If *df* is empty or has too few rows.
        """
        ...

    @abstractmethod
    def sample(
        self,
        n_rows: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate *n_rows* synthetic rows.

        Args:
            n_rows: Number of rows to generate.
            conditions: Optional column→value constraints for conditional
                generation (e.g. ``{"country": "US"}``).

        Returns:
            DataFrame with *n_rows* rows matching the training schema.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        ...

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serialize the trained model to a portable binary blob.

        The blob must be deserializable with ``from_bytes`` on any machine
        running the same package versions.
        """
        ...

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> "SynthesisModel":
        """Reconstruct a trained model from bytes produced by ``to_bytes``."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_training_df(df: pd.DataFrame, min_rows: int = 50) -> None:
        """Raise ``ValueError`` with a clear message if training data is unsuitable."""
        if df is None or len(df) == 0:
            raise ValueError("Training DataFrame is empty — cannot fit a synthesis model.")
        if len(df) < min_rows:
            raise ValueError(
                f"Training DataFrame has only {len(df)} rows; "
                f"minimum required is {min_rows}.  "
                "Consider using GaussianCopulaSynthesizer for small datasets."
            )
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names in training data: {dupes}")

    @staticmethod
    def _detect_cuda() -> bool:
        """Return True iff a CUDA-capable GPU is available."""
        try:
            import torch  # type: ignore[import]
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _safe_pickle_load(data: bytes) -> Any:
        """Load pickle bytes, raising a clear error on version mismatch."""
        try:
            return pickle.loads(data)  # noqa: S301 — trusted internal artifact
        except Exception as exc:
            raise RuntimeError(
                f"Failed to deserialize model artifact — "
                f"ensure the same sdv/torch versions are installed. Error: {exc}"
            ) from exc

    def quality_score(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Compute a 0–1 fidelity score comparing real vs synthetic distributions.

        Uses mean column-wise KS statistic (lower KS distance = higher score).
        Subclasses may override with a more sophisticated metric.
        """
        from scipy.stats import ks_2samp  # type: ignore[import]

        scores: list[float] = []
        num_cols = real_df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if col in synth_df.columns:
                real_v = real_df[col].dropna()
                synth_v = synth_df[col].dropna()
                if len(real_v) > 0 and len(synth_v) > 0:
                    stat, _ = ks_2samp(real_v, synth_v)
                    scores.append(1.0 - stat)

        return float(sum(scores) / len(scores)) if scores else 0.0
