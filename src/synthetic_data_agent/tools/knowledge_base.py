import redis.asyncio as redis
import json
from typing import List, Optional
from ..config import settings
import structlog

logger = structlog.get_logger()

class KnowledgeBase:
    """
    Session/Memory store for Business Rules and Metadata.
    Allows the agent to remember constraints for specific tables.
    """
    def __init__(self, redis_url: str = settings.redis_url):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    async def add_business_rule(self, table_fqn: str, rule_description: str, rule_code: str):
        """
        Store a business rule for a table.
        rule_description: Human readable constraint.
        rule_code: pandas query string or python eval code (e.g. 'age >= 18').
        """
        key = f"kb:rules:{table_fqn}"
        rule = {
            "description": rule_description,
            "code": rule_code
        }
        await self.redis.sadd(key, json.dumps(rule))
        logger.info("Added business rule to Knowledge Base", table=table_fqn, rule=rule_description)

    async def get_business_rules(self, table_fqn: str) -> List[dict]:
        """Retrieve all business rules for a given table."""
        key = f"kb:rules:{table_fqn}"
        rules_raw = await self.redis.smembers(key)
        return [json.loads(r) for r in rules_raw]

    async def clear_rules(self, table_fqn: str):
        """Clear all rules for a given table."""
        key = f"kb:rules:{table_fqn}"
        await self.redis.delete(key)
        logger.info("Cleared business rules from Knowledge Base", table=table_fqn)
