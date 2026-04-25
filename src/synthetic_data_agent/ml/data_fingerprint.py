"""Data fingerprinting for model artifact cache keys.

A fingerprint is a short deterministic string that identifies a specific
(dataset shape, column set, training config) triple.  If the fingerprint
matches a stored artifact, we can skip retraining entirely.

The fingerprint intentionally ignores row *values* — it is based only on:
  - Column names and dtypes (schema)
  - Row count (data volume)
  - Training config hyperparameters

This means schema changes, new columns, or significantly different data
volumes invalidate the cache, while row-level updates (e.g. new transactions
added to the same table) do NOT.  If callers want value-sensitive caching they
can pass a data_hash produced externally.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from .base import TrainingConfig


def fingerprint_dataframe(
    df: pd.DataFrame,
    config: TrainingConfig,
    table_fqn: str,
    strategy: str,
    extra_salt: str = "",
) -> str:
    """Produce a stable 24-char hex fingerprint for the (data, config, strategy) triple.

    Args:
        df: Training DataFrame — only schema (columns + dtypes) is hashed.
        config: Training hyperparameters.
        table_fqn: Fully-qualified table name (catalog.schema.table or file://…).
        strategy: Strategy name ('ctgan', 'tvae', 'copula', 'timegan').
        extra_salt: Optional extra string mixed into the hash (e.g. a data hash
            if callers want value-sensitive caching).

    Returns:
        24-character lowercase hex string.
    """
    schema_sig = "|".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes))
    row_bucket = _row_bucket(len(df))

    parts = [
        _normalise_fqn(table_fqn),
        strategy,
        schema_sig,
        row_bucket,
        config.fingerprint(),
        extra_salt,
    ]
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def artifact_key(table_fqn: str, strategy: str, fingerprint: str) -> str:
    """Build the ADK artifact filename for a trained model.

    Format: ``models/{safe_fqn}/{strategy}/{fingerprint}.pkl``

    This keeps artifacts scoped per table and strategy, making it easy to
    list all models for a given table.
    """
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", table_fqn)
    return f"models/{safe}/{strategy}/{fingerprint}.pkl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_fqn(fqn: str) -> str:
    """Strip file:// prefix and lowercase for consistent hashing."""
    return fqn.removeprefix("file://").lower()


def _row_bucket(n: int) -> str:
    """Bucket row count into coarse ranges so minor additions don't invalidate cache."""
    if n < 1_000:
        return "tiny"
    if n < 10_000:
        return "small"
    if n < 100_000:
        return "medium"
    if n < 1_000_000:
        return "large"
    return "xlarge"
