from typing import List, Dict, Any, Optional
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, select, func
from pgvector.sqlalchemy import Vector
from ..config import settings
import structlog

logger = structlog.get_logger()

class VectorBase(DeclarativeBase):
    pass

class SemanticEntry(VectorBase):
    __tablename__ = "semantic_memory"
    id: Mapped[int] = mapped_column(primary_key=True)
    table_fqn: Mapped[str] = mapped_column(String(255), index=True)
    content: Mapped[str] = mapped_column(Text) # The actual text (e.g. rule or profile)
    metadata_json: Mapped[str] = mapped_column(Text) # Extra tags
    embedding: Mapped[List[float]] = mapped_column(Vector(768)) # Gemini embedding size

class SemanticMemory:
    """
    pgvector-backed Semantic Memory for RAG and Cross-table Reasoning.
    """
    def __init__(self, database_url: str = settings.database_url, adk_client: Any = None):
        self.engine = create_async_engine(database_url)
        self.async_session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        self.adk_client = adk_client # Used for embedding generation via Gemini

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.execute(func.text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(VectorBase.metadata.create_all)

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini."""
        if not self.adk_client:
            # Fallback for local testing if Gemini is not hooked
            return [0.0] * 768
            
        # In a real ADK app, we'd use the configured embedding model
        # response = await self.adk_client.embed_content(text)
        # return response.embedding
        return [0.1] * 768 # Mock for now

    async def store_memory(self, table_fqn: str, text: str, metadata: Dict[str, Any]):
        """Store a semantic insight about a table."""
        embedding = await self._get_embedding(text)
        async with self.async_session() as session:
            entry = SemanticEntry(
                table_fqn=table_fqn,
                content=text,
                metadata_json=json.dumps(metadata),
                embedding=embedding
            )
            session.add(entry)
            await session.commit()
        logger.info("Stored semantic memory", table=table_fqn)

    async def search_similar(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for semantically similar insights."""
        query_embedding = await self._get_embedding(query)
        async with self.async_session() as session:
            # Native pgvector cosine distance search (<=>)
            result = await session.execute(
                select(SemanticEntry)
                .order_by(SemanticEntry.embedding.cosine_distance(query_embedding))
                .limit(limit)
            )
            entries = result.scalars().all()
            return [
                {
                    "table": e.table_fqn,
                    "content": e.content,
                    "metadata": json.loads(e.metadata_json)
                } 
                for e in entries
            ]
