from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

class QualityReport(BaseModel):
    model_config = ConfigDict(strict=True)
    
    table_fqn: str
    ks_test_results: dict[str, float] = Field(default_factory=dict)
    correlation_delta_frobenius: float = 0.0
    fk_integrity_pass: bool = False
    business_rule_pass_rate: float = 0.0
    pii_leakage_detected: bool = False
    k_anonymity_min: int = 0
    overall_pass: bool = False
    generated_at: datetime = Field(default_factory=datetime.now)
    failure_details: dict[str, str] = Field(default_factory=dict)
