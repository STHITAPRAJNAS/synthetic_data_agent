import json
import random
from typing import Any, Optional, List, Dict
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, select, delete, func
from ..config import settings
import structlog

logger = structlog.get_logger()

class RegistryBase(DeclarativeBase):
    pass

class SyntheticID(RegistryBase):
    __tablename__ = "synthetic_ids"
    id: Mapped[int] = mapped_column(primary_key=True)
    table_fqn: Mapped[str] = mapped_column(String(255), index=True)
    pk_col: Mapped[str] = mapped_column(String(255), index=True)
    pk_value: Mapped[str] = mapped_column(String(255))

class SyntheticIDRegistry:
    """
    Central FK coherence ledger backed by Postgres.
    """

    def __init__(self, database_url: str = settings.database_url):
        self.engine = create_async_engine(database_url)
        self.async_session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(RegistryBase.metadata.create_all)

    async def register_ids(self, table_fqn: str, pk_col: str, ids: List[Any]) -> int:
        """Register generated PKs for a specific table."""
        logger.info("Registering IDs", table=table_fqn, count=len(ids))
        if not ids:
            return 0
            
        async with self.async_session() as session:
            for i in ids:
                session.add(SyntheticID(
                    table_fqn=table_fqn,
                    pk_col=pk_col,
                    pk_value=str(i)
                ))
            await session.commit()
        return len(ids)

    async def sample_fk(self, parent_table_fqn: str, parent_pk_col: str, n: int) -> List[Any]:
        """Sample n FK values from registered parent PKs."""
        async with self.async_session() as session:
            # Use RANDOM() for sampling in Postgres
            result = await session.execute(
                select(SyntheticID.pk_value)
                .where(SyntheticID.table_fqn == parent_table_fqn)
                .where(SyntheticID.pk_col == parent_pk_col)
                .order_by(func.random())
                .limit(n)
            )
            return [r[0] for r in result.all()]

    async def get_fanout_sample(self, parent_table_fqn: str, 
                                 parent_pk_col: str, 
                                 fanout_mean: float, 
                                 n_parents: Optional[int] = None) -> Dict[str, int]:
        async with self.async_session() as session:
            query = select(SyntheticID.pk_value).where(
                SyntheticID.table_fqn == parent_table_fqn,
                SyntheticID.pk_col == parent_pk_col
            )
            if n_parents:
                query = query.order_by(func.random()).limit(n_parents)
            
            result = await session.execute(query)
            parent_ids = [r[0] for r in result.all()]
            
        import numpy as np
        counts = np.random.poisson(fanout_mean, len(parent_ids))
        return dict(zip(parent_ids, [int(c) for c in counts]))

    async def clear_registry(self):
        """Wipe the registry for a fresh run."""
        async with self.async_session() as session:
            await session.execute(delete(SyntheticID))
            await session.commit()
        logger.info("Registry cleared")
