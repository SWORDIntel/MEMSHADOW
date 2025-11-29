import hashlib
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog
import numpy as np

from app.models.memory import Memory
from app.db.chromadb import chroma_client
from app.db.redis import redis_client
from app.services.embedding_service import EmbeddingService
from app.services.consciousness.confidence_signals import (
    SignalExtractor,
    ConfidenceAggregator
)
from app.schemas.memory import ConfidenceMetadata
from app.core.config import settings, MemoryOperationMode

logger = structlog.get_logger()

class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService()

        # Initialize metacognitive confidence system (Phase 8.3 v2)
        if settings.ENABLE_METACOGNITIVE_CONFIDENCE:
            self.signal_extractor = SignalExtractor()
            self.confidence_aggregator = ConfidenceAggregator(
                confidence_threshold=settings.METACOGNITIVE_CONFIDENCE_THRESHOLD,
                low_similarity_threshold=settings.METACOGNITIVE_LOW_SIMILARITY_THRESHOLD
            )
            logger.info("Metacognitive confidence v4 enabled (32 signals: 14 v1 + 7 Phase2 + 3 Phase3 local-first + 8 Phase4 heavy)")
        else:
            self.signal_extractor = None
            self.confidence_aggregator = None

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
            # Check operation mode to determine processing level
            mode = settings.MEMORY_OPERATION_MODE

            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content)

            # Get the memory record
            memory = await self.db.get(Memory, memory_id)
            if not memory:
                logger.warning("Memory not found for embedding generation", memory_id=str(memory_id))
                return

            # Update memory with embedding
            memory.embedding = embedding

            # Mode-specific storage strategies
            if mode == MemoryOperationMode.LIGHTWEIGHT:
                # LIGHTWEIGHT: Skip ChromaDB, only cache and PostgreSQL
                logger.debug("LIGHTWEIGHT mode: Skipping ChromaDB storage", memory_id=str(memory_id))
            else:
                # LOCAL and ONLINE: Store in ChromaDB
                await chroma_client.add_embedding(
                    memory_id=str(memory.id),
                    embedding=embedding,
                    metadata={
                        "user_id": str(memory.user_id),
                        "created_at": memory.created_at.isoformat()
                    }
                )

            # Always cache the embedding in Redis (useful for all modes)
            cache_key = f"embedding:{memory_id}"
            cache_ttl = settings.EMBEDDING_CACHE_TTL

            # ONLINE mode: Extended cache TTL for better performance
            if mode == MemoryOperationMode.ONLINE:
                cache_ttl = cache_ttl * 2

            await redis_client.cache_set(cache_key, embedding, ttl=cache_ttl)

            await self.db.commit()

            logger.info("Embedding generated and stored",
                       memory_id=str(memory_id),
                       mode=mode.value)

        except Exception as e:
            logger.error("Embedding generation failed", memory_id=str(memory_id), error=str(e))
            # Consider adding retry logic or dead-letter queue here
            raise

    def _calculate_confidence_v2(
        self,
        query: str,
        similarity_scores: List[float],
        result_metadata: List[Dict[str, Any]],
        result_index: int,
        retrieval_mode: str
    ) -> ConfidenceMetadata:
        """
        Calculate metacognitive confidence using v2 signal-based system (Phase 8.3).

        Implements 14 mandatory signals across 5 tiers:
        - Tier 1: Retrieval Geometry (magnitude, mean, margin)
        - Tier 2: Result Quality (trust, completeness, integrity)
        - Tier 3: Query Semantics (length, specificity, ambiguity)
        - Tier 4: Temporal (recency, index freshness)
        - Tier 5: System Health (retrieval path)
        - Tier 7: Meta (signal completeness)

        Args:
            query: The search query string
            similarity_scores: List of all similarity scores (sorted descending)
            result_metadata: Metadata for each result
            result_index: Position in result list
            retrieval_mode: LIGHTWEIGHT/LOCAL/ONLINE

        Returns:
            ConfidenceMetadata with confidence, meta-confidence, and uncertainty sources
        """
        if not self.signal_extractor or not self.confidence_aggregator:
            # Metacognitive confidence disabled - return neutral metadata
            similarity_score = similarity_scores[result_index] if result_index < len(similarity_scores) else 0.5
            return ConfidenceMetadata(
                confidence=1.0,
                meta_confidence=1.0,
                similarity_score=similarity_score,
                uncertainty_sources=[],
                should_review=False
            )

        # Extract signal vector
        signals = self.signal_extractor.extract(
            query=query,
            similarity_scores=similarity_scores,
            result_metadata=result_metadata,
            result_index=result_index,
            retrieval_mode=retrieval_mode,
            index_last_updated=None  # TODO: track index update time
        )

        # Aggregate into confidence estimate
        estimate = self.confidence_aggregator.aggregate(signals)

        # Convert to ConfidenceMetadata schema
        return ConfidenceMetadata(
            confidence=estimate['confidence'],
            meta_confidence=estimate['meta_confidence'],
            similarity_score=estimate['similarity_score'],
            uncertainty_sources=estimate['uncertainty_sources'],
            should_review=estimate['should_review']
        )

    async def search_memories(
        self,
        user_id: UUID,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Tuple[List[Memory], Optional[List[ConfidenceMetadata]]]:
        """Search memories using semantic similarity"""
        mode = settings.MEMORY_OPERATION_MODE
        query_length = len(query)

        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)

        # Track similarity scores for confidence calculation
        similarity_scores = {}

        # Mode-specific search strategies
        if mode == MemoryOperationMode.LIGHTWEIGHT:
            # LIGHTWEIGHT: Use PostgreSQL vector search only (faster, less accurate)
            from sqlalchemy import func
            stmt = select(Memory).where(
                Memory.user_id == user_id
            ).order_by(
                func.cosine_distance(Memory.embedding, query_embedding)
            ).limit(limit).offset(offset)

            result = await self.db.execute(stmt)
            sorted_memories = list(result.scalars().all())

            # Calculate similarity scores (approximate for PostgreSQL)
            for memory in sorted_memories:
                if memory.embedding:
                    similarity = 1.0 - np.linalg.norm(
                        np.array(query_embedding) - np.array(memory.embedding)
                    ) / 2.0
                    similarity_scores[memory.id] = float(np.clip(similarity, 0.0, 1.0))
                else:
                    similarity_scores[memory.id] = 0.5  # Unknown similarity

        else:
            # LOCAL and ONLINE: Use ChromaDB for hybrid search
            where_clause = {"user_id": str(user_id)}
            if filters:
                where_clause.update(filters)

            # ONLINE mode: Reduce n_results for faster search
            search_limit = limit + offset
            if mode == MemoryOperationMode.ONLINE:
                search_limit = min(search_limit, limit * 2)  # Cap to reduce latency

            results = await chroma_client.search_similar(
                query_embedding=query_embedding,
                n_results=search_limit,
                where=where_clause
            )

            # Get memory IDs and distances from results
            memory_ids_str = results['ids'][0]
            distances = results.get('distances', [[]])[0]

            if not memory_ids_str:
                return [], None

            # Convert distances to similarity scores (1 - normalized_distance)
            for mid, distance in zip(memory_ids_str, distances):
                # ChromaDB returns L2 distance, convert to similarity
                similarity = 1.0 / (1.0 + distance)  # Inverse distance as similarity
                similarity_scores[UUID(mid)] = float(similarity)

            memory_ids = [UUID(mid) for mid in memory_ids_str][offset:offset + limit]

            if not memory_ids:
                return [], None

            # Fetch memories from database
            stmt = select(Memory).where(Memory.id.in_(memory_ids))
            result = await self.db.execute(stmt)
            memories = result.scalars().all()

            # Sort by relevance score from ChromaDB
            memory_dict = {m.id: m for m in memories}
            sorted_memories = [memory_dict[mid] for mid in memory_ids if mid in memory_dict]

        # Update access timestamps (skip in LIGHTWEIGHT mode for speed)
        if mode != MemoryOperationMode.LIGHTWEIGHT:
            for memory in sorted_memories:
                memory.accessed_at = datetime.utcnow()
                extra_data = memory.extra_data or {}
                extra_data["access_count"] = extra_data.get("access_count", 0) + 1
                memory.extra_data = extra_data

            await self.db.commit()

        # Calculate confidence metadata for each result (v2 signal-based system)
        confidence_metadata = None
        if self.signal_extractor and self.confidence_aggregator and sorted_memories:
            confidence_metadata = []

            # Build similarity scores list (sorted descending)
            similarity_list = [similarity_scores.get(m.id, 0.5) for m in sorted_memories]

            # Build metadata list for signal extraction
            result_metadata_list = []
            for memory in sorted_memories:
                metadata = {
                    'content': memory.content,
                    'created_at': memory.created_at,
                    'user_id': str(memory.user_id),
                    'source_trust': memory.extra_data.get('source_trust', 'unknown') if memory.extra_data else 'unknown',
                    'parse_errors': memory.extra_data.get('parse_errors', False) if memory.extra_data else False,
                    'truncated': memory.extra_data.get('truncated', False) if memory.extra_data else False,
                }
                result_metadata_list.append(metadata)

            # Calculate confidence for each result
            for idx, memory in enumerate(sorted_memories):
                confidence = self._calculate_confidence_v2(
                    query=query,
                    similarity_scores=similarity_list,
                    result_metadata=result_metadata_list,
                    result_index=idx,
                    retrieval_mode=mode.value
                )
                confidence_metadata.append(confidence)

                logger.debug(
                    "Confidence v2 calculated",
                    memory_id=str(memory.id),
                    confidence=confidence.confidence,
                    meta_confidence=confidence.meta_confidence,
                    similarity=confidence.similarity_score,
                    uncertainty_count=len(confidence.uncertainty_sources),
                    should_review=confidence.should_review
                )

        logger.info("Memories searched",
                   user_id=str(user_id),
                   query=query[:50],
                   results=len(sorted_memories),
                   mode=mode.value,
                   confidence_enabled=confidence_metadata is not None)

        return sorted_memories, confidence_metadata

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