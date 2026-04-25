from __future__ import annotations

import secrets
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .entity_graph import FKRelation


class TableGenConfig(BaseModel):
    model_config = ConfigDict(strict=False)

    table_fqn: str
    target_row_count: int = Field(gt=0)
    ml_strategy: Literal["ctgan", "tvae", "copula", "timegan"] = "ctgan"
    pii_columns: list[str] = Field(default_factory=list)
    non_pii_columns: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    # FK relations — populated by EntityGraphAgent
    foreign_keys: list[FKRelation] = Field(default_factory=list)
    # PK columns — needed so GeneratorAgent can register synthetic IDs
    primary_key_cols: list[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    model_config = ConfigDict(strict=False)

    plan_id: UUID = Field(default_factory=uuid4)
    # Stable identifier for the entire multi-table pipeline run.
    # All tables in the same run share this ID so the value ledger
    # can maintain cross-table synthetic value consistency.
    pipeline_run_id: str = Field(default_factory=lambda: str(uuid4()))
    # Random secret mixed into every entity hash.
    # Generated once per plan so different runs produce independent datasets.
    # NEVER logged or exposed to users.
    pipeline_salt: str = Field(default_factory=lambda: secrets.token_hex(32))
    tables_ordered: list[str]
    table_configs: dict[str, TableGenConfig]
    estimated_total_rows: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
