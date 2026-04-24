from typing import Dict, Any
from google.adk import Agent, tool
import pandas as pd
from scipy.stats import ks_2samp
from ....models.quality_report import QualityReport
from ....tools.databricks_tools import DatabricksTools
from ....config import settings
import structlog

logger = structlog.get_logger()

class ValidatorAgent:
    def __init__(self):
        self.db_tools = DatabricksTools()
        self.agent = Agent(
            name="ValidatorAgent",
            instructions="""You are a synthetic data quality specialist.
            Given a synthetic table and its original TableProfile, run all 
            quality gates and return a QualityReport."""
        )
        self.agent.register_tool(self.validate_table)

    @tool
    async def validate_table(self, table_fqn: str, synth_table_fqn: str) -> QualityReport:
        """Run KS tests and other quality gates on the synthetic table."""
        logger.info("Validating table", table=table_fqn)
        
        # 1. Pull samples
        real_df = await self.db_tools.sample_dataframe(table_fqn, 10_000)
        synth_df = await self.db_tools.sample_dataframe(synth_table_fqn, 10_000)
        
        ks_results = {}
        overall_pass = True
        
        # 2. Run Kolmogorov-Smirnov test for numerical columns
        numerical_cols = real_df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if col in synth_df.columns:
                statistic, p_value = ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
                ks_results[col] = p_value
                if p_value < 0.05:
                    overall_pass = False
                    
        return QualityReport(
            table_fqn=table_fqn,
            ks_test_results=ks_results,
            overall_pass=overall_pass
        )

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = ValidatorAgent().agent
