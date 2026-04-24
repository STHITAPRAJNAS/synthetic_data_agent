import redis.asyncio as redis
import json
import random
from typing import Any, Optional, List, Dict
from ..config import settings
import structlog

logger = structlog.get_logger()

class SyntheticIDRegistry:
    """
    Central FK coherence ledger backed by Redis.
    Ensures that synthetic child tables always reference valid synthetic parent IDs.
    """

    def __init__(self, redis_url: str = settings.redis_url):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    async def register_ids(self, table_fqn: str, pk_col: str, ids: List[Any]) -> int:
        """Register generated PKs for a specific table."""
        key = f"pk_registry:{table_fqn}:{pk_col}"
        logger.info("Registering IDs", table=table_fqn, count=len(ids))
        if not ids:
            return 0
            
        # Store as a set in Redis for efficient sampling and membership checks
        await self.redis.sadd(key, *[str(i) for i in ids])
        return len(ids)

    async def sample_fk(self, parent_table_fqn: str, parent_pk_col: str, n: int) -> List[Any]:
        """Sample n FK values from registered parent PKs."""
        key = f"pk_registry:{parent_table_fqn}:{parent_pk_col}"
        ids = await self.redis.srandmember(key, n)
        if not ids:
            logger.warning("No parent IDs found in registry", table=parent_table_fqn)
            return []
        return ids

    async def get_fanout_sample(self, parent_table_fqn: str, 
                                 parent_pk_col: str, 
                                 fanout_mean: float, 
                                 n_parents: Optional[int] = None) -> Dict[str, int]:
        """
        Returns {parent_id: child_count} respecting a Poisson distribution.
        Used to determine how many child rows to generate per parent.
        """
        key = f"pk_registry:{parent_table_fqn}:{parent_pk_col}"
        if n_parents:
            parent_ids = await self.redis.srandmember(key, n_parents)
        else:
            parent_ids = await self.redis.smembers(key)
            
        import numpy as np
        # Generate counts using Poisson distribution to simulate real-world fanout
        counts = np.random.poisson(fanout_mean, len(parent_ids))
        
        return dict(zip(parent_ids, [int(c) for c in counts]))

    async def clear_registry(self):
        """Wipe the registry for a fresh run."""
        keys = await self.redis.keys("pk_registry:*")
        if keys:
            await self.redis.delete(*keys)
        logger.info("Registry cleared")
