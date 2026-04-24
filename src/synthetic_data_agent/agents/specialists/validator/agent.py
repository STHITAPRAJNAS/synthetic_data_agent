from typing import Dict, Any
from google.adk import Agent, tool
import pandas as pd
from scipy.stats import ks_2samp
from ....models.quality_report import QualityReport
from ....tools.databricks_tools import DatabricksTools
from ....pii.leakage_auditor import PrivacyAuditor
from ....config import settings
import structlog

logger = structlog.get_logger()

class ValidatorAgent:
    def __init__(self):
        self.db_tools = DatabricksTools()
        self.privacy_auditor = PrivacyAuditor()
        
        instructions = """
        # IDENTITY
        You are the Synthetic Data Quality Specialist. Your goal is to run rigorous 
        statistical and privacy gates.

        # QUALITY GATES
        1. **Statistical Fidelity**: Run KS-tests on numerical columns.
        2. **Privacy Audit**: Ensure K-Anonymity >= 5 and Row Leakage < 1%.
        3. **Referential Integrity**: Verify 0% orphaned Foreign Keys.
        """

        self.agent = LlmAgent(
            name="ValidatorAgent",
            instructions=instructions,
            model=settings.gemini_model
        )
        self.agent.register_tool(self.validate_table)

    @tool
    async def validate_table(self, table_fqn: str, synth_table_fqn: str, quasi_ids: list[str]) -> QualityReport:
        """Run KS tests and Privacy Audits on the synthetic table."""
        logger.info("Validating table quality and privacy", table=table_fqn)
        
        real_df = await self.db_tools.sample_dataframe(table_fqn, 10_000)
        synth_df = await self.db_tools.sample_dataframe(synth_table_fqn, 10_000)
        
        # 1. Statistical Audit
        ks_results = {}
        stat_pass = True
        num_cols = real_df.select_dtypes(include=['number']).columns
        for col in num_cols:
            if col in synth_df.columns:
                _, p_val = ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
                ks_results[col] = p_val
                if p_val < 0.05: stat_pass = False

        # 2. Privacy Audit
        privacy_report = await self.privacy_auditor.audit_report(real_df, synth_df, quasi_ids)
        
        return QualityReport(
            table_fqn=table_fqn,
            ks_test_results=ks_results,
            pii_leakage_detected=(privacy_report["leakage_rate"] > 0.01),
            k_anonymity_min=privacy_report["k_anonymity"],
            overall_pass=stat_pass and privacy_report["privacy_pass"]
        )

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = ValidatorAgent().agent
