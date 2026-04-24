from typing import Any, Optional
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.connect import DatabricksSession
from ..config import settings
import structlog

logger = structlog.get_logger()

class DatabricksTools:
    def __init__(self):
        self.w = WorkspaceClient(
            host=settings.databricks_host,
            token=settings.databricks_token.get_secret_value()
        )
        # spark-connect session for large scale data handling
        self.spark = DatabricksSession.builder.remote(
            host=settings.databricks_host,
            token=settings.databricks_token.get_secret_value(),
            cluster_id=None # Usually handled via env or config in production
        ).getOrCreate()

    async def read_table_schema(self, table_fqn: str) -> dict[str, Any]:
        """Read Unity Catalog schema for a table, or infer from local file (CSV, Parquet, JSON)."""
        logger.info("Reading table schema", table=table_fqn)
        try:
            if table_fqn.startswith("file://"):
                path = table_fqn.replace("file://", "")
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
                    "comment": f"Local file: {path}"
                }

            catalog, schema, table = table_fqn.split(".")
            table_info = self.w.tables.get(table_fqn)
            return {
                "columns": [
                    {"name": c.name, "type": c.type_name, "comment": c.comment}
                    for c in table_info.columns
                ],
                "comment": table_info.comment
            }
        except Exception as e:
            logger.error("Failed to read schema", table=table_fqn, error=str(e))
            raise

    async def profile_column_statistics(self, table_fqn: str, sample_rows: int = 500_000) -> dict[str, Any]:
        """Run statistical profiling queries using TABLESAMPLE for scale."""
        logger.info("Profiling column statistics", table=table_fqn, sample_rows=sample_rows)
        
        if table_fqn.startswith("file://"):
            df = await self.sample_dataframe(table_fqn, sample_rows)
            return df.describe(include='all').to_dict()

        # Use Spark for scalable profiling
        df = self.spark.table(table_fqn).sample(withReplacement=False, fraction=None) # We use limit or TABLESAMPLE in SQL
        
        # SQL based profiling is often faster for large tables
        query = f"SELECT * FROM {table_fqn} TABLESAMPLE ({sample_rows} ROWS)"
        sampled_df = self.spark.sql(query)
        
        stats = {}
        for col in sampled_df.columns:
            # Basic stats using Spark native functions
            col_stats = sampled_df.select(col).summary("count", "mean", "stddev", "min", "max", "25%", "50%", "75%").toPandas()
            stats[col] = col_stats.to_dict()
            
        return stats

    async def sample_dataframe(self, table_fqn: str, n_rows: int) -> pd.DataFrame:
        """Pull a stratified sample from Databricks or local file as a pandas DataFrame."""
        logger.info("Sampling dataframe", table=table_fqn, n_rows=n_rows)
        if table_fqn.startswith("file://"):
            path = table_fqn.replace("file://", "")
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path, engine="pyarrow")
            elif path.endswith((".json", ".jsonl")):
                df = pd.read_json(path, lines=path.endswith(".jsonl"))
            else:
                raise ValueError(f"Unsupported file format: {path}")
            return df.sample(n=min(n_rows, len(df))) if len(df) > 0 else df

        query = f"SELECT * FROM {table_fqn} TABLESAMPLE ({n_rows} ROWS)"
        return self.spark.sql(query).toPandas()

    async def extract_foreign_keys(self, table_fqn: str) -> list[dict[str, Any]]:
        """Extract FK relationships from Unity Catalog constraints."""
        if table_fqn.startswith("file://"):
            return []
        # Implementation depends on UC API availability for constraints
        # For now, placeholder for manual/inferred FKs
        return []

    async def write_synthetic_table(self, table_fqn: str, df: pd.DataFrame, mode: str = "overwrite") -> dict[str, Any]:
        """Write synthetic data to Databricks or local file (CSV, Parquet, JSON)."""
        logger.info("Writing synthetic table", table=table_fqn, rows=len(df))
        
        if table_fqn.startswith("file://"):
            path = table_fqn.replace("file://", "")
            is_json = path.endswith((".json", ".jsonl"))
            is_lines = path.endswith(".jsonl")

            if mode == "append" and Path(path).exists():
                if path.endswith(".csv"):
                    df.to_csv(path, mode='a', header=False, index=False)
                elif path.endswith(".parquet"):
                    existing = pd.read_parquet(path)
                    pd.concat([existing, df]).to_parquet(path)
                elif is_json:
                    existing = pd.read_json(path, lines=is_lines)
                    pd.concat([existing, df]).to_json(path, orient='records', lines=is_lines)
            else:
                if path.endswith(".csv"):
                    df.to_csv(path, index=False)
                elif path.endswith(".parquet"):
                    df.to_parquet(path)
                elif is_json:
                    df.to_json(path, orient='records', lines=is_lines)
            return {"rows_written": len(df), "table_fqn": table_fqn}

        spark_df = self.spark.createDataFrame(df)
        spark_df.write.format("delta").mode(mode).saveAsTable(table_fqn)
        return {
            "rows_written": len(df),
            "table_fqn": table_fqn
        }
