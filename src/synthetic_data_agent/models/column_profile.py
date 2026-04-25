from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PIICategory(str, Enum):
    DIRECT_PII = "DIRECT_PII"
    QUASI_PII = "QUASI_PII"
    FINANCIAL_PII = "FINANCIAL_PII"
    SENSITIVE = "SENSITIVE"
    SAFE = "SAFE"


class DistributionType(str, Enum):
    GAUSSIAN = "GAUSSIAN"
    LOG_NORMAL = "LOG_NORMAL"
    CATEGORICAL = "CATEGORICAL"
    UNIFORM = "UNIFORM"
    TEMPORAL = "TEMPORAL"
    BOOLEAN = "BOOLEAN"
    HIGH_CARD_STRING = "HIGH_CARD_STRING"
    JSON = "JSON"


class ColumnProfile(BaseModel):
    model_config = ConfigDict(strict=False)  # lenient for JSON round-trips

    name: str
    dtype: str
    pii_category: Optional[PIICategory] = None
    distribution_type: Optional[DistributionType] = None
    distribution_params: dict[str, Any] = Field(default_factory=dict)
    null_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cardinality: int = Field(default=0, ge=0)
    sample_values: list[Any] = Field(default_factory=list)
    conditional_on: Optional[str] = None
    conditional_map: Optional[dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Column name must not be empty.")
        return v

    @field_validator("distribution_params")
    @classmethod
    def params_keys_are_strings(cls, v: dict[str, Any]) -> dict[str, Any]:
        for k in v:
            if not isinstance(k, str):
                raise ValueError(f"distribution_params keys must be strings, got {type(k)}")
        return v


class TableProfile(BaseModel):
    model_config = ConfigDict(strict=False)

    table_fqn: str
    row_count: int = Field(default=0, ge=0)
    columns: list[ColumnProfile]
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    functional_dependencies: list[tuple[str, str]] = Field(default_factory=list)
    # Temporal column — set by profiler when a datetime column is detected
    temporal_col: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("table_fqn")
    @classmethod
    def fqn_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("table_fqn must not be empty.")
        return v

    @field_validator("columns")
    @classmethod
    def at_least_one_column(cls, v: list[ColumnProfile]) -> list[ColumnProfile]:
        # Allow empty for error-profile case (partial profile on failure)
        return v
