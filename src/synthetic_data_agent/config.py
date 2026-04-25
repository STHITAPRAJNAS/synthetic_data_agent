from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Databricks Config
    databricks_host: str = Field(default="", description="Databricks workspace host")
    databricks_token: SecretStr = Field(default=SecretStr(""), description="Databricks personal access token")
    databricks_catalog: str = Field(default="", description="Source Unity Catalog catalog")
    databricks_schema: str = Field(default="", description="Source Unity Catalog schema")
    output_catalog: str = Field(default="", description="Databricks catalog to write synthetic tables to")

    # AI Config
    gemini_api_key: SecretStr = Field(default=SecretStr(""), description="Google Gemini API key")
    gemini_model: str = "gemini-2.5-flash"

    # Infrastructure Config
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/synthetic_data"
    model_storage_path: Path = Field(
        default=Path("./ml_models"),
        description="Path to store trained ML models",
    )

    # Generation Config
    max_profiling_sample_rows: int = 500_000
    ctgan_epochs: int = 300
    generation_batch_size: int = 50_000
    log_level: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Python logging level",
    )
    app_version: str = "1.0.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call this instead of using the module-level singleton."""
    return Settings()


# Module-level convenience — only evaluated once at first import
settings = get_settings()
