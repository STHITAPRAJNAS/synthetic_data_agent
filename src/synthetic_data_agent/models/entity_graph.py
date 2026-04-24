from typing import Literal
from pydantic import BaseModel, ConfigDict

class FKRelation(BaseModel):
    model_config = ConfigDict(strict=True)
    
    fk_col: str
    parent_table_fqn: str
    parent_pk_col: str
    cardinality: Literal["one_to_one", "one_to_many", "many_to_many"]
    null_rate: float = 0.0
    fanout_mean: float = 1.0
    fanout_distribution: str = "poisson"

class EntityNode(BaseModel):
    model_config = ConfigDict(strict=True)
    
    table_fqn: str
    primary_key_cols: list[str]
    foreign_keys: list[FKRelation] = []
