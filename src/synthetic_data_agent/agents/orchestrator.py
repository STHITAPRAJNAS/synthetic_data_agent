from typing import List, Dict, Any
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk import tool
from .specialists.profiler.agent import ProfilerAgent
from .specialists.entity_graph.agent import EntityGraphAgent
from .specialists.generator.agent import GeneratorAgent
from .specialists.pii_handler.agent import PIIHandlerAgent
from .specialists.validator.agent import ValidatorAgent
from ..models.generation_plan import GenerationPlan
from ..tools.knowledge_base import KnowledgeBase
from ..tools.semantic_memory import SemanticMemory
from ..config import settings
import structlog

logger = structlog.get_logger()

class Orchestrator:
    def __init__(self):
        self.profiler = ProfilerAgent()
        self.entity_graph = EntityGraphAgent()
        self.generator = GeneratorAgent()
        self.pii_handler = PIIHandlerAgent()
        self.validator = ValidatorAgent()
        self.knowledge_base = KnowledgeBase()
        self.semantic_memory = SemanticMemory()
        
        instructions = """
        # IDENTITY
        You are the Synthetic Data Generation Orchestrator. You are responsible for 
        end-to-end coordination of the data synthesis pipeline.

        # PIPELINE PHASES
        1. **Profiling**: Identify PII and statistical DNA via `ProfilerAgent`.
        2. **Graphing**: Determine FK relationships and generation order via `EntityGraphAgent`.
        3. **Synthesis**: Coordinate parallel generation of Non-PII (via `GeneratorAgent`) 
           and PII (via `PIIHandlerAgent`).
        4. **Validation**: Enforce quality gates via `ValidatorAgent`.

        # KNOWLEDGE MANAGEMENT
        - Use `search_semantic_knowledge` to leverage past learnings.
        - Use `remember_business_rule` to persist new constraints.
        - Prioritize existing business rules in the Knowledge Base for every run.
        """

        self.agent = LlmAgent(
            name="Orchestrator",
            instructions=instructions,
            model="gemini-2.0-pro-exp-02-05", # Use Pro for Orchestration
            before_model_callback=self._ensure_db_init
        )
        self.agent.register_tool(self.remember_business_rule)
        self.agent.register_tool(self.search_semantic_knowledge)
        self.agent.register_tool(self.run_pipeline)

    async def _ensure_db_init(self, ctx: CallbackContext, request: LlmRequest):
        """Ensure persistent stores are ready before agent starts thinking."""
        await self.knowledge_base.init_db()
        await self.semantic_memory.init_db()
        await self.generator.registry.init_db()
        return request

    @tool
    async def search_semantic_knowledge(self, query: str) -> str:
        """Search the pgvector semantic memory for similar past experiences or rules."""
        results = await self.semantic_memory.search_similar(query)
        if not results:
            return "No similar past experiences found."
        return f"Found {len(results)} similar items: " + str(results)

    @tool
    async def remember_business_rule(self, table_fqn: str, description: str, pandas_query: str) -> str:
        """
        Store a business rule for a given table and index it semantically.
        """
        await self.knowledge_base.add_business_rule(table_fqn, description, pandas_query)
        # Index in semantic memory for future RAG
        await self.semantic_memory.store_memory(table_fqn, description, {"query": pandas_query, "type": "rule"})
        return f"Successfully remembered and indexed rule for {table_fqn}: {description}"

    async def run_pipeline(self, table_fqns: List[str]):
        """Execute the full synthetic data generation pipeline with self-correction."""
        logger.info("Starting pipeline with Self-Correction Loop", tables=table_fqns)
        
        await self.knowledge_base.init_db()
        await self.semantic_memory.init_db()
        await self.generator.registry.init_db()
        
        profiles = await self.profiler.profile_tables(table_fqns)
        plan = await self.entity_graph.create_generation_plan(profiles)
        
        final_reports = []
        for table_fqn in plan.tables_ordered:
            config = plan.table_configs[table_fqn]
            attempts = 0
            max_attempts = 3
            pass_gate = False
            
            while attempts < max_attempts and not pass_gate:
                attempts += 1
                logger.info("Generation attempt", table=table_fqn, attempt=attempts)
                
                # 1. Generate Non-PII
                gen_result = await self.generator.generate_table_data(config)
                
                # 2. Generate PII (Metadata only)
                profile = next(p for p in profiles if p.table_fqn == table_fqn)
                pii_spec = {
                    c.name: {"category": str(c.pii_category), "dist_type": str(c.distribution_type)} 
                    for c in profile.columns if c.pii_category != "SAFE" or c.distribution_type == "JSON"
                }
                pii_data = await self.pii_handler.populate_pii_columns(table_fqn, config.target_row_count, pii_spec)
                
                # 3. Validation Gate
                output_fqn = f"{settings.output_catalog}.{settings.databricks_schema}.{table_fqn.split('.')[-1]}"
                quasi_ids = [c.name for c in profile.columns if c.pii_category == "QUASI_PII"]
                
                report = await self.validator.validate_table(table_fqn, output_fqn, quasi_ids)
                
                if report.overall_pass:
                    logger.info("Quality Gate Passed", table=table_fqn)
                    pass_gate = True
                else:
                    logger.warn("Quality Gate Failed, attempting self-correction", table=table_fqn)
                    # Get tuning suggestions from the agent's memory
                    suggestions = await self.search_semantic_knowledge(f"Tuning suggestions for failed quality gate on {table_fqn}")
                    # Update config for next attempt (e.g. switch strategy or increase epochs)
                    if "tvae" in suggestions.lower(): config.ml_strategy = "tvae"
                    
            final_reports.append(report)
            
        return final_reports

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = Orchestrator().agent
