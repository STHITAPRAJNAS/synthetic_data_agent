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
        """Execute the full synthetic data generation pipeline."""
        logger.info("Starting pipeline", tables=table_fqns)
        
        # Initialize all database tables
        await self.knowledge_base.init_db()
        await self.semantic_memory.init_db()
        await self.generator.registry.init_db()
        
        # Phase 1: Profile
        profiles = await self.profiler.profile_tables(table_fqns)
        
        # Phase 2: Entity Graph
        plan = await self.entity_graph.create_generation_plan(profiles)
        
        # Phase 3: Generate and Validate
        results = []
        for table_fqn in plan.tables_ordered:
            config = plan.table_configs[table_fqn]
            
            # Generate Non-PII
            gen_result = await self.generator.generate_table_data(config)
            
            # Generate PII (metadata only)
            pii_spec = {c.name: str(c.pii_category) for c in next(p for p in profiles if p.table_fqn == table_fqn).columns if c.pii_category != "SAFE"}
            pii_data = await self.pii_handler.populate_pii_columns(table_fqn, config.target_row_count, pii_spec)
            
            # TODO: Join PII and Non-PII and write final table
            
            # Phase 4: Validate
            # validation_report = await self.validator.validate_table(table_fqn, output_fqn)
            # results.append(validation_report)
            
        return results

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = Orchestrator().agent
