import pandas as pd
from typing import List, Dict, Any, Optional
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk import tool
from ....ml.ctgan_trainer import CTGANTrainer
from ....ml.tvae_trainer import TVAETrainer
from ....ml.copula_trainer import CopulaTrainer
from ....ml.timegan_trainer import TimeGANTrainer
from ....tools.databricks_tools import DatabricksTools
from ....tools.registry_tools import SyntheticIDRegistry
from ....tools.knowledge_base import KnowledgeBase
from ....models.generation_plan import TableGenConfig
from ....config import settings
import structlog

logger = structlog.get_logger()

class GeneratorAgent:
    def __init__(self):
        self.db_tools = DatabricksTools()
        self.registry = SyntheticIDRegistry()
        self.knowledge_base = KnowledgeBase()
        
        # mature structured instructions
        instructions = """
        # IDENTITY
        You are the Synthetic Data Generation Specialist. Your goal is to produce high-fidelity synthetic 
        data that preserves the statistical properties of the source while ensuring 0% row-leakage.

        # OPERATIONAL RULES
        1. ALWAYS train the appropriate ML model based on the `ml_strategy` provided.
        2. ALWAYS fetch and apply business rules from the Knowledge Base after generation.
        3. Ensure referential integrity by resolving Foreign Keys using the SyntheticIDRegistry.
        4. Write the final validated DataFrame back to the designated Databricks/File path.

        # STRATEGY SELECTION
        - use 'ctgan' for mixed tabular data.
        - use 'tvae' for high-fidelity behavioral data.
        - use 'timegan' for sequential/temporal data.
        - use 'copula' for simple, fast interpretable data.
        """

        self.agent = LlmAgent(
            name="GeneratorAgent",
            instructions=instructions,
            model=settings.gemini_model,
            after_model_callback=self._log_llm_activity
        )
        self.agent.register_tool(self.generate_table_data)

    async def _log_llm_activity(self, ctx: CallbackContext, response: LlmResponse):
        """Callback to monitor agent reasoning and token usage."""
        logger.info("GeneratorAgent LLM Activity", 
                    text=response.text,
                    usage=response.usage_metadata if hasattr(response, 'usage_metadata') else None)
        return response

    @tool
    async def generate_table_data(self, config: TableGenConfig) -> Dict[str, Any]:
        """Train model and generate synthetic data for a single table."""
        logger.info("Generating data for table", table=config.table_fqn)
        
        # 1. Get training data
        real_df = await self.db_tools.sample_dataframe(config.table_fqn, settings.max_profiling_sample_rows)
        
        # 2. Select and train model
        trainer: Any
        if config.ml_strategy == "tvae":
            trainer = TVAETrainer(config.table_fqn)
        elif config.ml_strategy == "copula":
            trainer = CopulaTrainer(config.table_fqn)
        elif config.ml_strategy == "timegan":
            trainer = TimeGANTrainer(config.table_fqn)
        else:
            trainer = CTGANTrainer(config.table_fqn)
            
        trainer.train(real_df[config.non_pii_columns])
        
        # 3. Sample
        # Generate extra rows in case business rules filter some out
        synth_df = trainer.sample(int(config.target_row_count * 1.5))
        
        # 4. Apply Business Rules
        rules = await self.knowledge_base.get_business_rules(config.table_fqn)
        for rule in rules:
            logger.info("Applying business rule", table=config.table_fqn, rule=rule["description"])
            try:
                synth_df = synth_df.query(rule["code"])
            except Exception as e:
                logger.error("Failed to apply rule", rule=rule, error=str(e))
                
        # Truncate back to target size if needed
        if len(synth_df) > config.target_row_count:
            synth_df = synth_df.head(config.target_row_count)
            
        # 5. Resolve FKs (Placeholder logic)
        # for fk in config.foreign_keys:
        #    synth_df[fk.col] = await self.registry.sample_fk(fk.parent_table, fk.parent_pk, len(synth_df))
        
        # 6. Write to Databricks
        output_fqn = f"{settings.output_catalog}.{settings.databricks_schema}.{config.table_fqn.split('.')[-1]}"
        result = await self.db_tools.write_synthetic_table(output_fqn, synth_df)
        
        # 7. Register PKs if any
        # await self.registry.register_ids(config.table_fqn, pk_col, synth_df[pk_col].tolist())
        
        return result

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = GeneratorAgent().agent
