"""FastAPI application entry point.

Exposes all ADK agents via the A2A protocol and a Web UI through
``get_fast_api_app``.  Additional endpoints:
  - GET  /health   — liveness probe (always 200 if the process is up)
  - GET  /ready    — readiness probe (checks DB + Databricks circuit-breaker)
  - POST /upload   — upload a CSV/Parquet/JSON file as a local source table
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from ..config import get_settings
from ..tools.databricks_tools import DatabricksTools

logger = structlog.get_logger()

UPLOAD_DIR = Path("/tmp/synthetic_data_uploads")

# Module-level start time for uptime reporting
_START_TIME: float = time.monotonic()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup; release on shutdown."""
    logger.info("server_startup", version=get_settings().app_version if hasattr(get_settings(), "app_version") else "1.0.0")

    # Pre-create upload directory
    await asyncio.to_thread(UPLOAD_DIR.mkdir, parents=True, exist_ok=True)

    # Eagerly initialise the orchestrator DB singletons so the first request
    # does not block while waiting for schema migrations.
    try:
        from ..tools.knowledge_base import KnowledgeBase
        from ..tools.registry_tools import SyntheticIDRegistry
        from ..tools.semantic_memory import SemanticMemory

        kb = KnowledgeBase()
        sm = SemanticMemory()
        reg = SyntheticIDRegistry()
        await asyncio.gather(kb.init_db(), sm.init_db(), reg.init_db())
        logger.info("server_db_ready")
    except Exception as exc:
        logger.warning("server_db_init_failed", error=str(exc))

    yield  # application is running

    logger.info("server_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Any:
    """Build the FastAPI application with A2A routing and custom endpoints."""
    from google.adk.cli.fast_api import get_fast_api_app  # type: ignore[import]

    agents_dir = str(Path(__file__).parent.parent / "agents")

    app: FastAPI = get_fast_api_app(
        agents_dir=agents_dir,
        web=True,
        a2a=True,
        use_local_storage=True,
        memory_service_uri=None,
        artifact_service_uri=None,
    )

    # Attach lifespan — ADK's app may not expose it directly, so we swap it in
    app.router.lifespan_context = _lifespan  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Liveness probe — process-level only, never blocks
    # ------------------------------------------------------------------
    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, Any]:
        """Liveness probe: 200 iff the process is running."""
        return {
            "status": "healthy",
            "uptime_s": round(time.monotonic() - _START_TIME, 1),
        }

    # ------------------------------------------------------------------
    # Readiness probe — checks actual dependencies
    # ------------------------------------------------------------------
    @app.get("/ready", tags=["ops"])
    async def ready() -> dict[str, Any]:
        """Readiness probe: 200 iff all dependencies are reachable.

        Checks:
        - Databricks circuit-breaker state (OPEN → not ready)
        - DB schema tables exist (quick existence ping)
        """
        checks: dict[str, Any] = {}

        # Databricks circuit breaker
        cb_health = DatabricksTools.circuit_breaker_health()
        checks["databricks_circuit_breaker"] = cb_health
        db_ready = cb_health["state"] != "OPEN"

        # DB connectivity — lightweight ping
        db_ok = True
        try:
            from ..tools.knowledge_base import KnowledgeBase
            kb = KnowledgeBase()
            await kb.init_db()
            checks["knowledge_base"] = "ok"
        except Exception as exc:
            checks["knowledge_base"] = f"error: {exc}"
            db_ok = False

        overall = db_ready and db_ok
        status_code = 200 if overall else 503

        response = {
            "ready": overall,
            "checks": checks,
        }

        if not overall:
            raise HTTPException(status_code=status_code, detail=response)

        return response

    # ------------------------------------------------------------------
    # File upload — CSV / Parquet / JSON → local source table
    # ------------------------------------------------------------------
    @app.post("/upload", tags=["data"])
    async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
        """Upload a CSV, Parquet, or JSON file as a local source table.

        The returned ``file_path`` value (e.g. ``file:///tmp/.../data.csv``)
        can be passed directly to the pipeline as a table FQN.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".csv", ".parquet", ".json", ".jsonl"}:
            raise HTTPException(
                status_code=400,
                detail="Only .csv, .parquet, .json, .jsonl files are supported",
            )

        # Strip directory components — prevents path traversal
        safe_name = Path(file.filename).name
        file_path = UPLOAD_DIR / safe_name

        try:
            contents = await file.read()
            await asyncio.to_thread(file_path.write_bytes, contents)
            logger.info("file_uploaded", path=str(file_path), bytes=len(contents))
            return {"status": "success", "file_path": f"file://{file_path}"}
        except Exception as exc:
            logger.error("upload_failed", error=str(exc))
            raise HTTPException(status_code=500, detail="Could not save file") from exc

    return app


# ---------------------------------------------------------------------------
# Module-level app — discoverable by ADK / uvicorn
# ---------------------------------------------------------------------------
app = create_app()


def main() -> None:
    cfg = get_settings()
    uvicorn.run(
        "synthetic_data_agent.server.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=cfg.log_level.lower(),
        # Production settings
        workers=1,          # ADK sessions are stateful; keep single process
        timeout_keep_alive=75,
        access_log=True,
    )


if __name__ == "__main__":
    main()
