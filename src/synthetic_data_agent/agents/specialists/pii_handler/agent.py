from typing import List, Dict, Any
from google.adk import Agent, tool
import pandas as pd
from ....pii.generators import (
    generate_synthetic_ssn, generate_synthetic_email, 
    generate_synthetic_name, generate_synthetic_phone,
    generate_synthetic_address, recursive_rehydrate
)
from ....models.column_profile import PIICategory, DistributionType
import structlog
import json

logger = structlog.get_logger()

class PIIHandlerAgent:
    def __init__(self):
        self.agent = Agent(
            name="PIIHandlerAgent",
            instructions="""You are a PII isolation specialist. You generate 
            format-valid but provably non-real values for PII columns.
            You must never receive real PII data, only metadata and distributions."""
        )
        self.agent.register_tool(self.populate_pii_columns)

    @tool
    async def populate_pii_columns(self, table_fqn: str, row_count: int, pii_spec: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Generate synthetic PII data for the specified columns.
        Supports deep re-hydration of JSON structures.
        """
        logger.info("Populating PII columns", table=table_fqn, rows=row_count)
        data = {}
        
        # Pull templates for JSON columns from Databricks/File
        from ....tools.databricks_tools import DatabricksTools
        db = DatabricksTools()
        
        for col_name, spec in pii_spec.items():
            category = spec.get('category')
            dist_type = spec.get('dist_type')

            if dist_type == "DistributionType.JSON":
                # 1. Sample 'Shell' structures from real data to ensure syntax validity
                real_samples = await db.sample_dataframe(table_fqn, row_count)
                raw_json_strings = real_samples[col_name].tolist()
                
                # 2. Deep Recursive Re-hydration
                synthetic_jsons = []
                for sj in raw_json_strings:
                    try:
                        parsed = json.loads(sj)
                        rehydrated = recursive_rehydrate(parsed)
                        synthetic_jsons.append(json.dumps(rehydrated))
                    except:
                        # Fallback to string re-hydration if not valid JSON
                        synthetic_jsons.append(generate_synthetic_instruction(str(sj)))
                data[col_name] = synthetic_jsons
                continue

            # Standard flat PII generation
            if category == "PIICategory.DIRECT_PII":
                if "ssn" in col_name.lower():
                    data[col_name] = [generate_synthetic_ssn() for _ in range(row_count)]
                elif "email" in col_name.lower():
                    data[col_name] = [generate_synthetic_email() for _ in range(row_count)]
                elif "name" in col_name.lower():
                    data[col_name] = [generate_synthetic_name() for _ in range(row_count)]
                else:
                    data[col_name] = [f"PII_{i}" for i in range(row_count)]
            elif category == "PIICategory.QUASI_PII":
                if "phone" in col_name.lower():
                    data[col_name] = [generate_synthetic_phone() for _ in range(row_count)]
                else:
                    data[col_name] = [f"QUASI_{i}" for i in range(row_count)]
            else:
                data[col_name] = [None] * row_count
                
        return data

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = PIIHandlerAgent().agent
