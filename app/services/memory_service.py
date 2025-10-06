import hashlib
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from app.models.memory import Memory
from app.db.chromadb import chroma_client
from app.db.redis import redis_client
from app.services.embedding_service import EmbeddingService
from app.core.config import settings

logger = structlog.get_logger()

class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService()

    async def create_memory(
        self,
        user_id: UUID,
        content: str,
        extra_data: Dict[str, Any]
    ) -> Memory:
        """Create a new memory"""
        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for duplicate
        existing = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.content_hash == content_hash
                )
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Duplicate memory content")

        # Create memory record
        memory = Memory(
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            extra_data=extra_data
        )

        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)

        logger.info("Memory created",
                   memory_id=str(memory.id),
                   user_id=str(user_id))

        return memory

    async def generate_and_store_embedding(self, memory_id: UUID, content: str):
        """Generate and store embedding for a memory."""
        try:
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content)

            # Get the memory record
            memory = await self.db.get(Memory, memory_id)
            if not memory:
                logger.warning("Memory not found for embedding generation", memory_id=str(memory_id))
                return

            # Update memory with embedding
            memory.embedding = embedding

            # Add to ChromaDB
            await chroma_client.add_embedding(
                memory_id=str(memory.id),
                embedding=embedding,
                metadata={
                    "user_id": str(memory.user_id),
                    "created_at": memory.created_at.isoformat()
                }
            )

            # Cache the embedding in Redis
            cache_key = f"embedding:{memory_id}"
            await redis_client.cache_set(cache_key, embedding, ttl=settings.EMBEDDING_CACHE_TTL)

            await self.db.commit()

            logger.info("Embedding generated and stored", memory_id=str(memory_id))

        except Exception as e:
            logger.error("Embedding generation failed", memory_id=str(memory_id), error=str(e))
            # Consider adding retry logic or dead-letter queue here
            raise

    async def search_memories(
        self,
        user_id: UUID,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Memory]:
        """Search memories using semantic similarity"""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)

        # Search in ChromaDB
        where_clause = {"user_id": str(user_id)}
        if filters:
            # This is a simple merge. For complex filters, you might need more logic.
            where_clause.update(filters)

        results = await chroma_client.search_similar(
            query_embedding=query_embedding,
            n_results=limit + offset,
            where=where_clause
        )

        # Get memory IDs from results
        memory_ids_str = results['ids'][0]
        if not memory_ids_str:
            return []

        memory_ids = [UUID(mid) for mid in memory_ids_str][offset:offset + limit]

        if not memory_ids:
            return []

        # Fetch memories from database
        stmt = select(Memory).where(Memory.id.in_(memory_ids))
        result = await self.db.execute(stmt)
        memories = result.scalars().all()

        # Sort by relevance score from ChromaDB
        memory_dict = {m.id: m for m in memories}
        sorted_memories = [memory_dict[mid] for mid in memory_ids if mid in memory_dict]

        # Update access timestamps
        for memory in sorted_memories:
            memory.accessed_at = datetime.utcnow()
            extra_data = memory.extra_data or {}
            extra_data["access_count"] = extra_data.get("access_count", 0) + 1
            memory.extra_data = extra_data

        await self.db.commit()

        logger.info("Memories searched",
                   user_id=str(user_id),
                   query=query[:50],
                   results=len(sorted_memories))

        return sorted_memories

    async def get_memory(
        self,
        memory_id: UUID,
        user_id: UUID
    ) -> Optional[Memory]:
        """Get a specific memory"""
        stmt = select(Memory).where(
            and_(
                Memory.id == memory_id,
                Memory.user_id == user_id
            )
        )
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()

        if memory:
            # Update access timestamp
            memory.accessed_at = datetime.utcnow()
            extra_data = memory.extra_data or {}
            extra_data["access_count"] = extra_data.get("access_count", 0) + 1
            memory.extra_data = extra_data
            await self.db.commit()

        return memory

    async def update_memory(
        self,
        memory_id: UUID,
        user_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Memory]:
        """Update an existing memory"""
        memory = await self.get_memory(memory_id, user_id)
        if not memory:
            return None

        # Update fields
        for field, value in updates.items():
            if hasattr(memory, field) and value is not None:
                setattr(memory, field, value)

        # Update content hash if content changed
        if "content" in updates and updates["content"] is not None:
            memory.content_hash = hashlib.sha256(
                updates["content"].encode()
            ).hexdigest()

        memory.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(memory)

        logger.info("Memory updated", memory_id=str(memory_id))

        return memory

    async def delete_memory(
        self,
        memory_id: UUID,
        user_id: UUID
    ) -> bool:
        """Delete a memory"""
        stmt = select(Memory).where(
            and_(
                Memory.id == memory_id,
                Memory.user_id == user_id
            )
        )
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()

        if not memory:
            return False

        # Delete from ChromaDB
        await chroma_client.delete_embedding(str(memory_id))

        # Delete from cache
        cache_key = f"embedding:{memory_id}"
        await redis_client.cache_delete(cache_key)

        # Delete from database
        await self.db.delete(memory)
        await self.db.commit()

        logger.info("Memory deleted",
                   memory_id=str(memory_id),
                   user_id=str(user_id))

        return True