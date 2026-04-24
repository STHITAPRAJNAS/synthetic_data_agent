from typing import List, Optional
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, select, delete
from ..config import settings
import structlog

logger = structlog.get_logger()

class Base(DeclarativeBase):
    pass

class BusinessRule(Base):
    __tablename__ = "business_rules"
    id: Mapped[int] = mapped_column(primary_key=True)
    table_fqn: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str] = mapped_column(Text)
    code: Mapped[str] = mapped_column(Text)

class KnowledgeBase:
    """
    Postgres-backed Session/Memory store for Business Rules.
    """
    def __init__(self, database_url: str = settings.database_url):
        self.engine = create_async_engine(database_url)
        self.async_session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_business_rule(self, table_fqn: str, rule_description: str, rule_code: str):
        async with self.async_session() as session:
            rule = BusinessRule(
                table_fqn=table_fqn,
                description=rule_description,
                code=rule_code
            )
            session.add(rule)
            await session.commit()
        logger.info("Added business rule to Knowledge Base", table=table_fqn, rule=rule_description)

    async def get_business_rules(self, table_fqn: str) -> List[dict]:
        async with self.async_session() as session:
            result = await session.execute(
                select(BusinessRule).where(BusinessRule.table_fqn == table_fqn)
            )
            rules = result.scalars().all()
            return [{"description": r.description, "code": r.code} for r in rules]

    async def clear_rules(self, table_fqn: str):
        async with self.async_session() as session:
            await session.execute(
                delete(BusinessRule).where(BusinessRule.table_fqn == table_fqn)
            )
            await session.commit()
        logger.info("Cleared business rules from Knowledge Base", table=table_fqn)
