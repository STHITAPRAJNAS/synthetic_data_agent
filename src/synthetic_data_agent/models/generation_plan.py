from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .entity_graph import FKRelation


class TableGenConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    table_fqn: str
    target_row_count: int
    ml_strategy: Literal["ctgan", "tvae", "copula", "timegan"]
    pii_columns: list[str] = Field(default_factory=list)
    non_pii_columns: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    # FK relations for this table — populated by EntityGraphAgent
    foreign_keys: list[FKRelation] = Field(default_factory=list)
    # PK columns — needed so GeneratorAgent can register synthetic IDs
    primary_key_cols: list[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    model_config = ConfigDict(strict=True)

    plan_id: UUID = Field(default_factory=uuid4)
    tables_ordered: list[str]
    table_configs: dict[str, TableGenConfig]
    estimated_total_rows: int
    created_at: datetime = Field(default_factory=datetime.now)
