from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QualityReport(BaseModel):
    model_config = ConfigDict(strict=False)

    table_fqn: str
    ks_test_results: dict[str, float] = Field(default_factory=dict)
    correlation_delta_frobenius: float = Field(default=0.0, ge=0.0)
    fk_integrity_pass: bool = False
    business_rule_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    pii_leakage_detected: bool = False
    k_anonymity_min: int = Field(default=0, ge=0)
    overall_pass: bool = False
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    failure_details: dict[str, str] = Field(default_factory=dict)

    @field_validator("ks_test_results")
    @classmethod
    def ks_values_are_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        for col, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"KS p-value for column '{col}' must be in [0, 1], got {p}."
                )
        return v

    @field_validator("table_fqn")
    @classmethod
    def fqn_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("table_fqn must not be empty.")
        return v
