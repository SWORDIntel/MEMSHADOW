"""
Neural Storage Integration Service

Bridges the existing MemoryService with the new Neural Storage system.
Provides seamless integration without breaking existing functionality.
"""

import asyncio
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.core.config import settings
from app.models.memory import Memory
from .orchestrator import NeuralStorageOrchestrator, NeuralStorageConfig, get_neural_storage
from .tiered_database import StorageTier

logger = structlog.get_logger()


class NeuralStorageIntegration:
    """
    Integration layer between existing MemoryService and Neural Storage.

    This class provides methods to:
    - Sync memories from PostgreSQL to Neural Storage
    - Use neural storage for enhanced search
    - Leverage brain-like connections for better retrieval
    - Maintain consistency between systems
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._orchestrator: Optional[NeuralStorageOrchestrator] = None

    async def get_orchestrator(self) -> NeuralStorageOrchestrator:
        """Get or initialize the neural storage orchestrator"""
        if self._orchestrator is None:
            self._orchestrator = await get_neural_storage()
        return self._orchestrator

    async def sync_memory_to_neural(
        self,
        memory: Memory,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Sync a memory from the database to neural storage.

        This should be called after creating or updating a memory
        to keep neural storage in sync.
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return False

        orchestrator = await self.get_orchestrator()

        # Use existing embedding or generate placeholder
        if embedding is None and memory.embedding is not None:
            embedding = np.array(memory.embedding, dtype=np.float32)
        elif embedding is None:
            # No embedding available yet
            logger.debug("No embedding for memory, skipping neural sync",
                        memory_id=str(memory.id))
            return False

        try:
            # Determine initial tier based on memory characteristics
            initial_tier = self._determine_initial_tier(memory)

            # Store in neural storage
            await orchestrator.store_memory(
                memory_id=memory.id,
                content_hash=memory.content_hash,
                embedding=embedding,
                metadata={
                    "user_id": str(memory.user_id),
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "extra_data": memory.extra_data or {},
                },
                initial_tier=initial_tier,
                discover_connections=True
            )

            logger.info("Memory synced to neural storage",
                       memory_id=str(memory.id),
                       tier=initial_tier.description)

            return True

        except Exception as e:
            logger.error("Failed to sync memory to neural storage",
                        memory_id=str(memory.id),
                        error=str(e))
            return False

    def _determine_initial_tier(self, memory: Memory) -> StorageTier:
        """Determine the appropriate initial tier for a memory"""
        # New memories start at HIGH tier
        # Important/frequently accessed can be promoted later

        extra_data = memory.extra_data or {}

        # Check for priority hints
        if extra_data.get("priority") == "high":
            return StorageTier.RAMDISK
        elif extra_data.get("priority") == "low":
            return StorageTier.MEDIUM

        # Check access count
        access_count = extra_data.get("access_count", 0)
        if access_count > 10:
            return StorageTier.LOW  # Hot memory

        return StorageTier.HIGH  # Default

    async def enhanced_search(
        self,
        query_embedding: np.ndarray,
        user_id: UUID,
        top_k: int = 10,
        use_spreading_activation: bool = True
    ) -> List[Tuple[UUID, float, str]]:
        """
        Enhanced search using neural storage.

        Returns list of (memory_id, score, source) tuples.
        Source indicates where the result came from (tier name or 'association').
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return []

        orchestrator = await self.get_orchestrator()
        results = []

        # Primary search across tiers
        tier_results = await orchestrator.search(
            query_embedding,
            top_k=top_k * 2,  # Get more for filtering
            threshold=settings.NEURAL_STORAGE_SIMILARITY_THRESHOLD
        )

        # Add tier results
        for memory_id, score, tier in tier_results:
            results.append((memory_id, score, tier.name))

        # Optionally use spreading activation to find associated memories
        if use_spreading_activation and tier_results:
            top_result_ids = [mid for mid, _, _ in tier_results[:3]]
            for seed_id in top_result_ids:
                associated = await orchestrator.get_associated_memories(
                    seed_id, max_depth=2, top_k=5
                )
                for assoc_id, activation in associated:
                    if assoc_id not in [r[0] for r in results]:
                        results.append((assoc_id, activation * 0.8, "association"))

        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get_related_memories(
        self,
        memory_id: UUID,
        max_depth: int = 2,
        top_k: int = 10
    ) -> List[Tuple[UUID, float]]:
        """
        Get memories related to a given memory using neural connections.

        This uses the brain-like spreading activation to find
        semantically and contextually related memories.
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return []

        orchestrator = await self.get_orchestrator()
        return await orchestrator.get_associated_memories(
            memory_id, max_depth=max_depth, top_k=top_k
        )

    async def find_hidden_connections(
        self,
        memory_a: UUID,
        memory_b: UUID
    ) -> List[List[UUID]]:
        """
        Find memories that connect two seemingly unrelated memories.

        This discovers relationships that might have been missed by
        the various AI components.
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return []

        orchestrator = await self.get_orchestrator()
        return await orchestrator.find_bridge(memory_a, memory_b)

    async def complete_memory_pattern(
        self,
        partial_memory_ids: List[UUID],
        top_k: int = 5
    ) -> List[Tuple[UUID, float]]:
        """
        Complete a partial memory pattern.

        Given a set of memories that appear together, find other
        memories that commonly appear with them.
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return []

        orchestrator = await self.get_orchestrator()
        return await orchestrator.complete_pattern(partial_memory_ids, top_k)

    async def suggest_related_memories(
        self,
        memory_id: UUID,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Suggest memories that should be connected to a given memory.

        Returns suggestions with confidence and reasoning.
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return []

        orchestrator = await self.get_orchestrator()
        suggestions = await orchestrator.suggest_connections(memory_id, top_k)

        return [
            {
                "memory_id": str(mid),
                "confidence": conf,
                "reason": reason
            }
            for mid, conf, reason in suggestions
        ]

    async def promote_memory(self, memory_id: UUID) -> bool:
        """Promote a memory to faster storage"""
        if not settings.NEURAL_STORAGE_ENABLED:
            return False

        orchestrator = await self.get_orchestrator()
        return await orchestrator.migration_manager.force_promote(
            memory_id, StorageTier.RAMDISK
        )

    async def demote_memory(self, memory_id: UUID) -> bool:
        """Demote a memory to slower storage"""
        if not settings.NEURAL_STORAGE_ENABLED:
            return False

        orchestrator = await self.get_orchestrator()
        return await orchestrator.migration_manager.force_demote(
            memory_id, StorageTier.ULTRA_HIGH
        )

    async def get_memory_temperature(self, memory_id: UUID) -> Optional[float]:
        """Get the current temperature (hotness) of a memory"""
        if not settings.NEURAL_STORAGE_ENABLED:
            return None

        orchestrator = await self.get_orchestrator()
        record = orchestrator.tiered_db.global_index.get(memory_id)
        if record:
            return record.temperature
        return None

    async def get_memory_connections(self, memory_id: UUID) -> List[Dict[str, Any]]:
        """Get all connections for a memory"""
        if not settings.NEURAL_STORAGE_ENABLED:
            return []

        orchestrator = await self.get_orchestrator()
        connections = await orchestrator.tiered_db.get_connections(memory_id)

        return [
            {
                "connected_to": str(conn_id),
                "weight": weight
            }
            for conn_id, weight in connections
        ]

    async def bulk_sync(
        self,
        memories: List[Memory],
        embeddings: Dict[UUID, np.ndarray]
    ) -> Dict[str, int]:
        """
        Bulk sync multiple memories to neural storage.

        More efficient than syncing one by one.
        """
        if not settings.NEURAL_STORAGE_ENABLED:
            return {"synced": 0, "failed": 0, "skipped": 0}

        orchestrator = await self.get_orchestrator()

        # Request burst capacity for bulk operation
        await orchestrator.cpu_manager.request_burst(30.0)

        synced = 0
        failed = 0
        skipped = 0

        for memory in memories:
            embedding = embeddings.get(memory.id)
            if embedding is None:
                skipped += 1
                continue

            try:
                success = await self.sync_memory_to_neural(memory, embedding)
                if success:
                    synced += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Bulk sync error",
                           memory_id=str(memory.id),
                           error=str(e))
                failed += 1

        logger.info("Bulk sync completed",
                   synced=synced,
                   failed=failed,
                   skipped=skipped)

        return {"synced": synced, "failed": failed, "skipped": skipped}

    async def get_neural_stats(self) -> Dict[str, Any]:
        """Get neural storage statistics"""
        if not settings.NEURAL_STORAGE_ENABLED:
            return {"enabled": False}

        orchestrator = await self.get_orchestrator()
        stats = await orchestrator.get_stats()
        stats["enabled"] = True
        return stats


async def create_neural_storage_config() -> NeuralStorageConfig:
    """Create neural storage config from application settings"""
    return NeuralStorageConfig(
        enable_ultra_high_tier=settings.NEURAL_STORAGE_ENABLE_ULTRA_HIGH_TIER,
        max_ramdisk_mb=settings.NEURAL_STORAGE_RAMDISK_MAX_MB,
        min_workers=settings.NEURAL_STORAGE_MIN_WORKERS,
        max_workers=settings.NEURAL_STORAGE_MAX_WORKERS,
        target_cpu_utilization=settings.NEURAL_STORAGE_TARGET_CPU_UTILIZATION,
        promote_temperature=settings.NEURAL_STORAGE_PROMOTE_TEMPERATURE,
        demote_temperature=settings.NEURAL_STORAGE_DEMOTE_TEMPERATURE,
        semantic_dedup_threshold=settings.NEURAL_STORAGE_SEMANTIC_DEDUP_THRESHOLD,
        auto_dedup_interval_minutes=settings.NEURAL_STORAGE_AUTO_DEDUP_INTERVAL_MINUTES,
        similarity_threshold=settings.NEURAL_STORAGE_SIMILARITY_THRESHOLD,
        max_connections_per_memory=settings.NEURAL_STORAGE_MAX_CONNECTIONS_PER_MEMORY,
        hebbian_learning_rate=settings.NEURAL_STORAGE_HEBBIAN_LEARNING_RATE,
        enable_background_tasks=settings.NEURAL_STORAGE_ENABLE_BACKGROUND_TASKS,
        maintenance_interval_minutes=settings.NEURAL_STORAGE_MAINTENANCE_INTERVAL_MINUTES,
    )
