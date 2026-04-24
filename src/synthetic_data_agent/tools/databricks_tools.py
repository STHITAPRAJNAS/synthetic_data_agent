from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from ..config import get_settings
from .circuit_breaker import CircuitBreaker
from .retry import retry_async

logger = structlog.get_logger()

# Allowlist: only catalog.schema.table or file:// paths
_FQN_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+$")

# Module-level circuit breaker — shared across all DatabricksTools instances.
# Trips after 5 consecutive Spark/Databricks failures; recovers after 60 s.
_databricks_cb = CircuitBreaker(
    name="databricks",
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=2,
)


def _validate_fqn(table_fqn: str) -> None:
    """Raise ValueError if the table FQN is not a safe catalog.schema.table identifier."""
    if not table_fqn.startswith("file://") and not _FQN_PATTERN.match(table_fqn):
        raise ValueError(
            f"Invalid table FQN '{table_fqn}'. "
            "Expected 'catalog.schema.table' or 'file://path'."
        )


class DatabricksTools:
    """Async wrapper around Databricks SDK and Spark Connect for table operations."""

    def __init__(self) -> None:
        # Clients are created lazily to avoid connecting at import time
        self._workspace_client: Any | None = None
        self._spark: Any | None = None

    # ------------------------------------------------------------------
    # Lazy client accessors — token extracted only when needed
    # ------------------------------------------------------------------

    def _get_workspace_client(self) -> Any:
        if self._workspace_client is None:
            from databricks.sdk import WorkspaceClient  # type: ignore[import]

            cfg = get_settings()
            self._workspace_client = WorkspaceClient(
                host=cfg.databricks_host,
                token=cfg.databricks_token.get_secret_value(),
            )
        return self._workspace_client

    def _get_spark(self) -> Any:
        if self._spark is None:
            from databricks.connect import DatabricksSession  # type: ignore[import]

            cfg = get_settings()
            self._spark = (
                DatabricksSession.builder.remote(
                    host=cfg.databricks_host,
                    token=cfg.databricks_token.get_secret_value(),
                    cluster_id=None,
                ).getOrCreate()
            )
        return self._spark

    # ------------------------------------------------------------------
    # Public async tools
    # ------------------------------------------------------------------

    async def read_table_schema(self, table_fqn: str) -> dict[str, Any]:
        """Read Unity Catalog schema for a table, or infer from a local file.

        Args:
            table_fqn: Fully-qualified table name (catalog.schema.table) or file://path.

        Returns:
            Dict with 'columns' list (name, type, comment) and 'comment'.
        """
        _validate_fqn(table_fqn)
        logger.info("Reading table schema", table=table_fqn)

        if table_fqn.startswith("file://"):
            return await asyncio.to_thread(self._read_local_schema, table_fqn)

        def _fetch() -> dict[str, Any]:
            w = self._get_workspace_client()
            table_info = w.tables.get(table_fqn)
            return {
                "columns": [
                    {"name": c.name, "type": str(c.type_name), "comment": c.comment}
                    for c in (table_info.columns or [])
                ],
                "comment": table_info.comment,
                "properties": table_info.properties or {},
            }

        async def _safe_fetch() -> dict[str, Any]:
            async with _databricks_cb:
                return await asyncio.to_thread(_fetch)

        try:
            return await retry_async(_safe_fetch, max_attempts=3, label=f"read_schema:{table_fqn}")
        except Exception as exc:
            logger.error("Failed to read schema", table=table_fqn, error=str(exc))
            raise

    @staticmethod
    def _read_local_schema(table_fqn: str) -> dict[str, Any]:
        path = table_fqn.removeprefix("file://")
        if path.endswith(".csv"):
            df = pd.read_csv(path, nrows=5)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path, engine="pyarrow")
        elif path.endswith((".json", ".jsonl")):
            df = pd.read_json(path, lines=path.endswith(".jsonl"), nrows=5)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return {
            "columns": [
                {"name": str(c), "type": str(df[c].dtype), "comment": "inferred"}
                for c in df.columns
            ],
            "comment": f"Local file: {path}",
        }

    async def profile_column_statistics(
        self, table_fqn: str, sample_rows: int = 500_000
    ) -> dict[str, Any]:
        """Run statistical profiling using TABLESAMPLE for large Databricks tables.

        Args:
            table_fqn: Fully-qualified table name or file://path.
            sample_rows: Maximum rows to sample.

        Returns:
            Dict mapping column name → statistics dict.
        """
        _validate_fqn(table_fqn)

        if table_fqn.startswith("file://"):
            df = await self.sample_dataframe(table_fqn, sample_rows)
            return await asyncio.to_thread(lambda: df.describe(include="all").to_dict())

        def _spark_profile() -> dict[str, Any]:
            spark = self._get_spark()
            query = f"SELECT * FROM {table_fqn} TABLESAMPLE ({sample_rows} ROWS)"
            sampled_df = spark.sql(query)
            stats: dict[str, Any] = {}
            for col in sampled_df.columns:
                col_stats = (
                    sampled_df.select(col)
                    .summary("count", "mean", "stddev", "min", "max", "25%", "50%", "75%")
                    .toPandas()
                )
                stats[col] = col_stats.to_dict()
            return stats

        return await asyncio.to_thread(_spark_profile)

    async def sample_dataframe(self, table_fqn: str, n_rows: int) -> pd.DataFrame:
        """Pull a sample from Databricks or a local file as a pandas DataFrame.

        Args:
            table_fqn: Fully-qualified table name or file://path.
            n_rows: Number of rows to return.

        Returns:
            pandas DataFrame with up to n_rows rows.
        """
        _validate_fqn(table_fqn)
        logger.info("Sampling dataframe", table=table_fqn, n_rows=n_rows)

        if table_fqn.startswith("file://"):
            return await asyncio.to_thread(self._read_local_file, table_fqn, n_rows)

        def _spark_sample() -> pd.DataFrame:
            spark = self._get_spark()
            query = f"SELECT * FROM {table_fqn} TABLESAMPLE ({n_rows} ROWS)"
            return spark.sql(query).toPandas()

        async def _safe_sample() -> pd.DataFrame:
            async with _databricks_cb:
                return await asyncio.to_thread(_spark_sample)

        return await retry_async(_safe_sample, max_attempts=3, label=f"sample_df:{table_fqn}")

    @staticmethod
    def _read_local_file(table_fqn: str, n_rows: int) -> pd.DataFrame:
        path = table_fqn.removeprefix("file://")
        if path.endswith(".csv"):
            df: pd.DataFrame = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path, engine="pyarrow")
        elif path.endswith((".json", ".jsonl")):
            df = pd.read_json(path, lines=path.endswith(".jsonl"))
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return df.sample(n=min(n_rows, len(df))) if len(df) > 0 else df

    async def extract_foreign_keys(self, table_fqn: str) -> list[dict[str, Any]]:
        """Extract FK relationships from Unity Catalog constraints.

        Args:
            table_fqn: Fully-qualified table name.

        Returns:
            List of FK relation dicts with fk_col, parent_table, parent_col keys.
        """
        _validate_fqn(table_fqn)
        if table_fqn.startswith("file://"):
            return []

        def _fetch() -> list[dict[str, Any]]:
            w = self._get_workspace_client()
            try:
                constraints = w.tables.get(table_fqn).table_constraints or []
                fks: list[dict[str, Any]] = []
                for c in constraints:
                    if hasattr(c, "foreign_key_constraint") and c.foreign_key_constraint:
                        fk = c.foreign_key_constraint
                        for col_pair in fk.child_columns:
                            fks.append({
                                "fk_col": col_pair,
                                "parent_table_fqn": fk.parent_table,
                                "parent_pk_col": fk.parent_columns[0] if fk.parent_columns else "id",
                            })
                return fks
            except Exception:
                return []

        return await asyncio.to_thread(_fetch)

    async def write_synthetic_table(
        self,
        table_fqn: str,
        df: pd.DataFrame,
        mode: str = "overwrite",
    ) -> dict[str, Any]:
        """Write synthetic data to Databricks Delta or a local file.

        Args:
            table_fqn: Target fully-qualified table name or file://path.
            df: Synthetic DataFrame to write.
            mode: 'overwrite' or 'append'.

        Returns:
            Dict with rows_written, table_fqn, write_timestamp.
        """
        _validate_fqn(table_fqn)
        logger.info("Writing synthetic table", table=table_fqn, rows=len(df))

        if table_fqn.startswith("file://"):
            await asyncio.to_thread(self._write_local_file, table_fqn, df, mode)
            return {"rows_written": len(df), "table_fqn": table_fqn}

        def _spark_write() -> None:
            spark = self._get_spark()
            spark_df = spark.createDataFrame(df)
            spark_df.write.format("delta").mode(mode).saveAsTable(table_fqn)

        async def _safe_write() -> dict[str, Any]:
            async with _databricks_cb:
                await asyncio.to_thread(_spark_write)
            return {"rows_written": len(df), "table_fqn": table_fqn}

        return await retry_async(_safe_write, max_attempts=2, label=f"write_table:{table_fqn}")

    @staticmethod
    def circuit_breaker_health() -> dict[str, Any]:
        """Return the current circuit-breaker snapshot for /ready and /metrics."""
        return _databricks_cb.health()

    @staticmethod
    def _write_local_file(table_fqn: str, df: pd.DataFrame, mode: str) -> None:
        path = table_fqn.removeprefix("file://")
        append = mode == "append" and Path(path).exists()

        if path.endswith(".csv"):
            df.to_csv(path, mode="a", header=not append, index=False) if append else df.to_csv(path, index=False)
        elif path.endswith(".parquet"):
            if append:
                existing = pd.read_parquet(path)
                pd.concat([existing, df]).to_parquet(path)
            else:
                df.to_parquet(path)
        elif path.endswith((".json", ".jsonl")):
            lines = path.endswith(".jsonl")
            if append:
                existing = pd.read_json(path, lines=lines)
                pd.concat([existing, df]).to_json(path, orient="records", lines=lines)
            else:
                df.to_json(path, orient="records", lines=lines)
        else:
            raise ValueError(f"Unsupported file format: {path}")
