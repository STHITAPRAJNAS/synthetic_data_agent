"""ADK Artifact Store integration for trained synthesis models.

Trained models are expensive to produce (minutes to hours of GPU time).
This module persists them as binary blobs in the ADK artifact service so
they survive across agent sessions and can be reused for future generation
runs against the same table without retraining.

Cache key format
-----------------
``models/{safe_table_fqn}/{strategy}/{fingerprint}.pkl``

The fingerprint encodes the dataset schema + row-count bucket + training
config so cache hits are only returned when the model is still valid.

Fallback
--------
If the ADK artifact service is not configured (e.g. local development with
``use_local_storage=True`` and no GCS backend), the store falls back to the
local filesystem under ``{model_storage_path}``.  The agent code is
unaware of which backend is active.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ..config import get_settings
from .base import SynthesisModel

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()

# mime type for binary model blobs
_MODEL_MIME = "application/octet-stream"


# ---------------------------------------------------------------------------
# Artifact store helpers
# ---------------------------------------------------------------------------

async def save_model_artifact(
    tool_context: Any,
    model: SynthesisModel,
    artifact_key: str,
) -> str:
    """Persist a trained model to the ADK artifact store.

    Args:
        tool_context: ADK ToolContext (injected by the framework).
        model: Trained model — must implement ``to_bytes()``.
        artifact_key: Logical filename (e.g. ``models/users/ctgan/abc123.pkl``).

    Returns:
        The artifact_key (for logging/chaining).
    """
    model_bytes = model.to_bytes()
    logger.info(
        "saving_model_artifact",
        key=artifact_key,
        size_kb=round(len(model_bytes) / 1024, 1),
    )

    # Try ADK artifact service first
    try:
        from google.genai import types as genai_types  # type: ignore[import]
        artifact = genai_types.Part.from_bytes(
            data=model_bytes,
            mime_type=_MODEL_MIME,
        )
        await tool_context.save_artifact(filename=artifact_key, artifact=artifact)
        logger.info("model_artifact_saved_adk", key=artifact_key)
        return artifact_key
    except Exception as exc:
        logger.warning(
            "adk_artifact_save_failed_falling_back",
            key=artifact_key,
            error=str(exc),
        )

    # Fallback: local filesystem
    await _save_local(artifact_key, model_bytes)
    return artifact_key


async def load_model_artifact(
    tool_context: Any,
    artifact_key: str,
    strategy: str,
) -> SynthesisModel | None:
    """Load a previously persisted model from the ADK artifact store.

    Args:
        tool_context: ADK ToolContext (injected by the framework).
        artifact_key: Logical filename used when saving.
        strategy: Strategy name used to reconstruct the correct class.

    Returns:
        Reconstructed SynthesisModel, or None if no artifact found.
    """
    logger.info("loading_model_artifact", key=artifact_key)

    # Try ADK artifact service first
    try:
        artifact = await tool_context.load_artifact(filename=artifact_key)
        if artifact and artifact.inline_data and artifact.inline_data.data:
            model_bytes = artifact.inline_data.data
            model = _reconstruct(model_bytes, strategy)
            logger.info("model_artifact_loaded_adk", key=artifact_key)
            return model
    except Exception as exc:
        logger.debug("adk_artifact_load_miss", key=artifact_key, error=str(exc))

    # Fallback: local filesystem
    model_bytes = await _load_local(artifact_key)
    if model_bytes:
        model = _reconstruct(model_bytes, strategy)
        logger.info("model_artifact_loaded_local", key=artifact_key)
        return model

    logger.info("model_artifact_not_found", key=artifact_key)
    return None


async def artifact_exists(tool_context: Any, artifact_key: str) -> bool:
    """Return True if an artifact with *artifact_key* exists in any backend."""
    try:
        artifact = await tool_context.load_artifact(filename=artifact_key)
        if artifact and artifact.inline_data and artifact.inline_data.data:
            return True
    except Exception:
        pass
    return await _local_exists(artifact_key)


# ---------------------------------------------------------------------------
# Local filesystem fallback
# ---------------------------------------------------------------------------

def _local_path(artifact_key: str) -> Path:
    base = get_settings().model_storage_path
    return base / artifact_key


async def _save_local(artifact_key: str, data: bytes) -> None:
    path = _local_path(artifact_key)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_bytes, data)
    logger.info("model_artifact_saved_local", path=str(path), size_kb=round(len(data) / 1024, 1))


async def _load_local(artifact_key: str) -> bytes | None:
    path = _local_path(artifact_key)
    if not await asyncio.to_thread(path.exists):
        return None
    return await asyncio.to_thread(path.read_bytes)


async def _local_exists(artifact_key: str) -> bool:
    path = _local_path(artifact_key)
    return await asyncio.to_thread(path.exists)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(data: bytes, strategy: str) -> SynthesisModel:
    """Reconstruct the correct SynthesisModel subclass from bytes."""
    from .ctgan_trainer import CTGANTrainer
    from .tvae_trainer import TVAETrainer
    from .copula_trainer import CopulaTrainer
    from .timegan_trainer import TimeGANTrainer

    cls_map: dict[str, type[SynthesisModel]] = {
        "ctgan": CTGANTrainer,
        "tvae": TVAETrainer,
        "copula": CopulaTrainer,
        "timegan": TimeGANTrainer,
    }
    cls = cls_map.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown strategy '{strategy}' — cannot reconstruct model.")
    return cls.from_bytes(data)
