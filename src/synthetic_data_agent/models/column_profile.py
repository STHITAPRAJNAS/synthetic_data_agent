from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

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
    model_config = ConfigDict(strict=True)
    
    name: str
    dtype: str
    pii_category: Optional[PIICategory] = None
    distribution_type: Optional[DistributionType] = None
    distribution_params: dict[str, Any] = Field(default_factory=dict)
    null_rate: float = 0.0
    cardinality: int = 0
    sample_values: list[Any] = Field(default_factory=list)
    conditional_on: Optional[str] = None
    conditional_map: Optional[dict[str, Any]] = None

class TableProfile(BaseModel):
    model_config = ConfigDict(strict=True)
    
    table_fqn: str
    row_count: int
    columns: list[ColumnProfile]
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    functional_dependencies: list[tuple[str, str]] = Field(default_factory=list)
    temporal_col: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
