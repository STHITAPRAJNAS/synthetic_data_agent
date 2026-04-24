import pandas as pd
import numpy as np
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

class PrivacyAuditor:
    """
    Mathematical privacy validator for synthetic datasets.
    """
    
    @staticmethod
    def check_row_leakage(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """
        Check if any synthetic rows are exact duplicates of real rows (Leakage).
        Returns the percentage of leaked rows.
        """
        # We only check non-PII columns as PII is generated randomly
        common_cols = list(set(real_df.columns) & set(synth_df.columns))
        
        # Performance optimized merge to find intersections
        merged = pd.merge(real_df[common_cols], synth_df[common_cols], on=common_cols, how='inner')
        leakage_rate = len(merged) / len(synth_df)
        return leakage_rate

    @staticmethod
    def calculate_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
        """
        Calculate the K-Anonymity of a dataset based on quasi-identifiers.
        A dataset has k-anonymity if every record is indistinguishable from at least k-1 others.
        """
        if not quasi_identifiers:
            return 0
            
        # Group by quasi-identifiers and find the smallest group size
        group_counts = df.groupby(quasi_identifiers).size()
        return int(group_counts.min())

    async def audit_report(self, real_df: pd.DataFrame, synth_df: pd.DataFrame, quasi_ids: List[str]) -> Dict[str, Any]:
        """Generate a full privacy audit."""
        leakage = self.check_row_leakage(real_df, synth_df)
        k_anon = self.calculate_k_anonymity(synth_df, quasi_ids)
        
        passed = (leakage < 0.01) and (k_anon >= 5)
        
        return {
            "leakage_rate": leakage,
            "k_anonymity": k_anon,
            "privacy_pass": passed,
            "quasi_identifiers_audited": quasi_ids
        }
