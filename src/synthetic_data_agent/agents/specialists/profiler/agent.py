from typing import List
from google.adk import Agent, tool
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
        
        self.agent = Agent(
            name="ProfilerAgent",
            instructions="""You are a data profiling specialist. Given a list of 
            Databricks table FQNs, produce a complete TableProfile for each table.
            For each column: detect PII using PIIDetector, compute statistics using 
            profile_column_statistics, infer distribution type from the statistics,
            detect functional dependencies between column pairs (e.g. zip→city).
            Store the profile as JSON. Return a summary of what was profiled."""
        )
        
        # Register tools
        self.agent.register_tool(self.profile_tables)

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
