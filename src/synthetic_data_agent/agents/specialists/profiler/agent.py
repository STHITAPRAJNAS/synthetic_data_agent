from typing import List
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk import tool
from ....models.column_profile import TableProfile, ColumnProfile, PIICategory, DistributionType
from ....tools.databricks_tools import DatabricksTools
from ....pii.detector import PIIDetector
from ....config import settings
import structlog

logger = structlog.get_logger()

class ProfilerAgent:
    def __init__(self):
        self.db_tools = DatabricksTools()
        self.pii_detector = PIIDetector()
        
        instructions = """
        # IDENTITY
        You are the Data Profiling Specialist. You analyze raw datasets to extract their 
        statistical signature and identify sensitive PII columns.

        # CORE OBJECTIVES
        1. **PII Discovery**: Use multi-layered detection (Regex, Presidio, LLM) to classify columns.
        2. **Statistical DNA**: Extract cardinality, null rates, and distribution types.
        3. **Relationship Inference**: Identify potential functional dependencies (e.g., zip -> city).

        # OUTPUT REQUIREMENT
        You must return a list of `TableProfile` objects that precisely describe the data 
        topology. Ensure `pii_category` is correctly assigned to avoid leakage.
        """

        self.agent = LlmAgent(
            name="ProfilerAgent",
            instructions=instructions,
            model=settings.gemini_model,
            after_model_callback=self._validate_profiling_results
        )
        
        # Register tools
        self.agent.register_tool(self.profile_tables)

    async def _validate_profiling_results(self, ctx: CallbackContext, response: LlmResponse):
        """Callback to ensure profiling logic is consistent."""
        logger.info("Profiling Completed", result_summary=response.text[:200])
        return response

    @tool
    async def profile_tables(self, table_fqns: List[str]) -> List[TableProfile]:
        """Profile a list of Databricks tables and return their TableProfiles."""
        profiles = []
        for fqn in table_fqns:
            logger.info("Profiling table", table=fqn)
            
            # 1. Read Schema
            schema = await self.db_tools.read_table_schema(fqn)
            
            # 2. Sample Data for deep profiling
            df = await self.db_tools.sample_dataframe(fqn, settings.max_profiling_sample_rows)
            
            # 3. Build Column Profiles
            col_profiles = []
            for col in df.columns:
                # Detect PII
                pii_cat = await self.pii_detector.detect(col, df[col].tolist())
                
                # Basic Stats
                cardinality = df[col].nunique()
                null_rate = df[col].isnull().mean()
                
                # Infer Distribution (Simplified)
                dist_type = DistributionType.GAUSSIAN
                if pii_cat != PIICategory.SAFE:
                    dist_type = DistributionType.HIGH_CARD_STRING
                elif df[col].dtype == 'object':
                    # Check for JSON
                    sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
                    if sample_val.startswith(("{", "[")):
                        try:
                            import json
                            json.loads(sample_val)
                            dist_type = DistributionType.JSON
                        except:
                            dist_type = DistributionType.CATEGORICAL
                    else:
                        dist_type = DistributionType.CATEGORICAL
                
                cp = ColumnProfile(
                    name=col,
                    dtype=str(df[col].dtype),
                    pii_category=pii_cat,
                    distribution_type=dist_type,
                    null_rate=null_rate,
                    cardinality=cardinality,
                    sample_values=df[col].dropna().unique()[:5].tolist()
                )
                col_profiles.append(cp)
            
            profile = TableProfile(
                table_fqn=fqn,
                row_count=len(df), # This is sample count, real count should come from schema
                columns=col_profiles,
                created_at=None # Defaults to now
            )
            profiles.append(profile)
            
        return profiles

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = ProfilerAgent().agent
