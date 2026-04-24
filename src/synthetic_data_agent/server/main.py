import uvicorn
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.adk.cli.fast_api import get_fastapi_app
from ..config import settings
import structlog

logger = structlog.get_logger()

def create_app() -> FastAPI:
    agents_dir = str(Path(__file__).parent.parent / "agents")
    
    # We use get_fastapi_app which natively wires ADK services.
    # For a persistent ADK MemoryStore, you can pass a URI like:
    # memory_service_uri="vertex-ai-memory-bank://projects/<project>/locations/<loc>/memoryBanks/<id>"
    # Or rely on standard ADK in-memory / local SQLite defaults if none provided.
    app = get_fastapi_app(
        agents_dir=agents_dir,
        web=True,
        a2a=True,
        use_local_storage=True,
        memory_service_uri=None, # Update this with your actual persistent memory backend (e.g. Vertex AI Memory Bank)
        artifact_service_uri=None, # Uses local file artifacts natively; no native S3 support out of the box.
    )
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a CSV or Parquet file to be used as a source table."""
        if not file.filename.endswith((".csv", ".parquet")):
            raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")
            
        # Natively, ADK's FileArtifactService would manage artifacts, 
        # but since S3 is not native and we need local ingestion quickly:
        upload_dir = Path("/tmp/synthetic_data_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info("File uploaded successfully", path=str(file_path))
            return {"status": "success", "file_path": f"file://{file_path}"}
        except Exception as e:
            logger.error("Failed to save uploaded file", error=str(e))
            raise HTTPException(status_code=500, detail="Could not save file")
        
    return app

def main():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
