from typing import List
from google.adk import Agent, tool
import networkx as nx
from ....models.column_profile import TableProfile
from ....models.entity_graph import EntityNode, FKRelation
from ....models.generation_plan import GenerationPlan, TableGenConfig
import structlog

logger = structlog.get_logger()

class EntityGraphAgent:
    def __init__(self):
        self.agent = Agent(
            name="EntityGraphAgent",
            instructions="""You are a data modeling specialist. Given a set of 
            TableProfiles, build the complete entity relationship graph.
            Identify all FK relationships (from Unity Catalog constraints first,
            then infer from column naming patterns and cardinality analysis).
            Compute topological generation order — parents must be generated before
            children. Compute fanout ratios for each parent-child relationship.
            Return the ordered generation plan."""
        )
        self.agent.register_tool(self.create_generation_plan)

    @tool
    async def create_generation_plan(self, profiles: List[TableProfile]) -> GenerationPlan:
        """Analyze profiles to create a topological generation plan."""
        G = nx.DiGraph()
        table_configs = {}
        
        for p in profiles:
            G.add_node(p.table_fqn)
            # Simplified FK inference: if col ends in _id and matches another table name
            pii_cols = [c.name for c in p.columns if c.pii_category and c.pii_category != "SAFE"]
            non_pii_cols = [c.name for c in p.columns if not c.pii_category or c.pii_category == "SAFE"]
            
            # Placeholder for actual FK detection logic
            # for col in p.columns:
            #     if col.name.endswith("_id"): ...
            
            table_configs[p.table_fqn] = TableGenConfig(
                table_fqn=p.table_fqn,
                target_row_count=p.row_count,
                ml_strategy="ctgan", # Default
                pii_columns=pii_cols,
                non_pii_columns=non_pii_cols
            )

        # Compute topological order
        try:
            ordered_tables = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            logger.error("Cyclic dependency detected in entity graph")
            ordered_tables = [p.table_fqn for p in profiles] # Fallback

        return GenerationPlan(
            tables_ordered=ordered_tables,
            table_configs=table_configs,
            estimated_total_rows=sum(p.row_count for p in profiles)
        )

    async def run(self, input_text: str):
        return await self.agent.run(input_text)

# Export root_agent for ADK AgentLoader
root_agent = EntityGraphAgent().agent
