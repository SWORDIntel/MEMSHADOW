"""
Neural Storage Orchestrator - Brain-like Memory Coordination

The main orchestrator that coordinates all components of the neural storage system:
- Tiered Database Manager (multi-dimensional storage)
- Neural Connection Engine (brain-like pattern discovery)
- Dynamic CPU Manager (adaptive resource allocation)
- RAMDISK Storage (ultra-fast working memory)
- Memory Migration Manager (hot/cold storage)
- Cross-tier Deduplicator (storage optimization)

This is the primary interface for the neural storage system, providing
a unified API that abstracts the complexity of the underlying components.
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
import structlog

from .tiered_database import TieredDatabaseManager, StorageTier, TieredMemoryRecord
from .neural_connection_engine import NeuralConnectionEngine, SynapticConnection
from .dynamic_cpu_manager import DynamicCPUManager, WorkloadIntensity
from .ramdisk_storage import RAMDiskStorage
from .memory_migration import MemoryMigrationManager, MigrationPolicy
from .deduplication import CrossTierDeduplicator

logger = structlog.get_logger()


@dataclass
class NeuralStorageConfig:
    """Configuration for the Neural Storage system"""
    # Tiered Database
    enable_ultra_high_tier: bool = True
    max_ramdisk_mb: int = 512

    # CPU Management
    min_workers: int = 1
    max_workers: Optional[int] = None
    target_cpu_utilization: float = 0.7

    # Migration
    promote_temperature: float = 0.8
    demote_temperature: float = 0.2

    # Deduplication
    semantic_dedup_threshold: float = 0.95
    auto_dedup_interval_minutes: int = 30

    # Connection Engine
    similarity_threshold: float = 0.7
    max_connections_per_memory: int = 100
    hebbian_learning_rate: float = 0.1

    # Background Tasks
    enable_background_tasks: bool = True
    maintenance_interval_minutes: int = 15


class NeuralStorageOrchestrator:
    """
    Main orchestrator for the brain-like neural storage system.

    Provides a unified interface for:
    - Storing memories with automatic tier placement
    - Searching across all tiers with cascade search
    - Discovering and creating connections between memories
    - Managing hot/cold storage migration
    - Deduplicating across tiers
    - Dynamic CPU resource allocation

    Usage:
        orchestrator = NeuralStorageOrchestrator(config)
        await orchestrator.start()

        # Store a memory
        record = await orchestrator.store_memory(
            memory_id=uuid,
            content_hash=hash,
            embedding=embedding,
            metadata={"source": "user_input"}
        )

        # Search for similar memories
        results = await orchestrator.search(query_embedding, top_k=10)

        # Get related memories (brain-like association)
        related = await orchestrator.get_associated_memories(memory_id)
    """

    def __init__(self, config: Optional[NeuralStorageConfig] = None):
        self.config = config or NeuralStorageConfig()

        # Initialize components
        self.tiered_db = TieredDatabaseManager(
            enable_ultra_high=self.config.enable_ultra_high_tier,
            max_ramdisk_mb=self.config.max_ramdisk_mb,
            auto_migrate=True
        )

        self.cpu_manager = DynamicCPUManager(
            min_workers=self.config.min_workers,
            max_workers=self.config.max_workers,
            target_cpu_utilization=self.config.target_cpu_utilization
        )

        self.ramdisk = RAMDiskStorage(
            max_memory_mb=self.config.max_ramdisk_mb,
            embedding_dim=256  # RAMDISK uses compressed embeddings
        )

        self.connection_engine = NeuralConnectionEngine(
            tiered_db=self.tiered_db,
            similarity_threshold=self.config.similarity_threshold,
            max_connections_per_memory=self.config.max_connections_per_memory,
            hebbian_learning_rate=self.config.hebbian_learning_rate
        )

        self.migration_manager = MemoryMigrationManager(
            tiered_db=self.tiered_db,
            ramdisk=self.ramdisk,
            cpu_manager=self.cpu_manager,
            policy=MigrationPolicy(
                promote_threshold=self.config.promote_temperature,
                demote_threshold=self.config.demote_temperature
            )
        )

        self.deduplicator = CrossTierDeduplicator(
            tiered_db=self.tiered_db,
            ramdisk=self.ramdisk,
            semantic_threshold=self.config.semantic_dedup_threshold
        )

        # Background tasks
        self._running = False
        self._maintenance_task: Optional[asyncio.Task] = None
        self._connection_discovery_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_stores": 0,
            "total_searches": 0,
            "total_retrievals": 0,
            "connections_discovered": 0,
            "patterns_learned": 0,
        }

        logger.info("NeuralStorageOrchestrator initialized")

    async def start(self):
        """Start all components and background tasks"""
        logger.info("Starting Neural Storage Orchestrator...")

        # Start components
        await self.cpu_manager.start()
        await self.ramdisk.start()
        await self.migration_manager.start()

        self._running = True

        # Start background tasks
        if self.config.enable_background_tasks:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            self._connection_discovery_task = asyncio.create_task(
                self._connection_discovery_loop()
            )

        logger.info("Neural Storage Orchestrator started successfully")

    async def stop(self):
        """Stop all components and cleanup"""
        logger.info("Stopping Neural Storage Orchestrator...")

        self._running = False

        # Cancel background tasks
        if self._maintenance_task:
            self._maintenance_task.cancel()
        if self._connection_discovery_task:
            self._connection_discovery_task.cancel()

        # Stop components
        await self.migration_manager.stop()
        await self.ramdisk.stop()
        await self.cpu_manager.stop()

        logger.info("Neural Storage Orchestrator stopped")

    # ==================== Core Operations ====================

    async def store_memory(
        self,
        memory_id: UUID,
        content_hash: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        initial_tier: StorageTier = StorageTier.HIGH,
        discover_connections: bool = True
    ) -> TieredMemoryRecord:
        """
        Store a new memory in the neural storage system.

        The memory is stored at the specified tier with embeddings
        projected to all lower-dimensional tiers for fast searching.
        Connections to similar memories are automatically discovered.
        """
        self.stats["total_stores"] += 1

        # Use CPU manager for the operation
        async def store_operation():
            # Store in tiered database
            record = await self.tiered_db.store(
                memory_id=memory_id,
                content_hash=content_hash,
                embedding=embedding,
                initial_tier=initial_tier,
                metadata=metadata
            )

            # Also store in RAMDISK for fast access (hot storage)
            await self.ramdisk.store(
                memory_id=memory_id,
                embedding=embedding,
                content_hash=content_hash,
                metadata=metadata
            )

            return record

        record = await self.cpu_manager.execute_task(
            f"store_{memory_id}",
            store_operation,
            intensity=WorkloadIntensity.LIGHT
        )

        # Discover connections in background
        if discover_connections:
            asyncio.create_task(self._discover_connections_for_memory(memory_id))

        return record

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5,
        search_all_tiers: bool = True,
        user_id: Optional[UUID] = None
    ) -> List[Tuple[UUID, float, StorageTier]]:
        """
        Search for similar memories across the neural storage system.

        Implements cascade search: starts from fastest tier (RAMDISK)
        and progressively searches slower tiers if needed.
        """
        self.stats["total_searches"] += 1

        # First check RAMDISK (fastest)
        ramdisk_results = await self.ramdisk.batch_search(
            query_embedding, top_k=top_k, threshold=threshold
        )

        if len(ramdisk_results) >= top_k and not search_all_tiers:
            # Found enough in RAMDISK
            return [(mid, score, StorageTier.RAMDISK) for mid, score in ramdisk_results]

        # Search across all tiers
        all_results = await self.tiered_db.search_all_tiers(
            query_embedding,
            top_k=top_k,
            threshold=threshold
        )

        # Merge with RAMDISK results
        seen = set()
        merged = []

        for mid, score, tier in all_results:
            if mid not in seen:
                seen.add(mid)
                merged.append((mid, score, tier))

        for mid, score in ramdisk_results:
            if mid not in seen:
                seen.add(mid)
                merged.append((mid, score, StorageTier.RAMDISK))

        # Sort by score
        merged.sort(key=lambda x: x[1], reverse=True)

        # Learn from co-activation (Hebbian learning)
        activated_ids = [mid for mid, _, _ in merged[:min(5, len(merged))]]
        if len(activated_ids) > 1:
            await self.connection_engine.learn_from_coactivation(activated_ids)
            self.stats["patterns_learned"] += 1

        return merged[:top_k]

    async def retrieve(
        self,
        memory_id: UUID,
        promote_on_access: bool = True
    ) -> Optional[Tuple[np.ndarray, StorageTier, Dict[str, Any]]]:
        """
        Retrieve a specific memory by ID.

        Returns (embedding, tier, metadata) tuple.
        Optionally promotes frequently accessed memories to faster tiers.
        """
        self.stats["total_retrievals"] += 1

        # Try RAMDISK first
        ramdisk_entry = await self.ramdisk.retrieve(memory_id)
        if ramdisk_entry:
            if promote_on_access:
                await self.ramdisk.promote(memory_id)
            return (ramdisk_entry.embedding, StorageTier.RAMDISK, ramdisk_entry.metadata)

        # Fall back to tiered database
        result = await self.tiered_db.retrieve(memory_id)
        if result:
            embedding, tier = result
            record = self.tiered_db.global_index.get(memory_id)
            metadata = record.metadata if record else {}

            # Promote to RAMDISK if accessed frequently
            if promote_on_access and record:
                if record.temperature > self.config.promote_temperature:
                    await self.migration_manager.force_promote(memory_id, StorageTier.RAMDISK)

            return (embedding, tier, metadata)

        return None

    # ==================== Association & Connection Operations ====================

    async def get_associated_memories(
        self,
        memory_id: UUID,
        max_depth: int = 2,
        top_k: int = 10
    ) -> List[Tuple[UUID, float]]:
        """
        Get memories associated with a given memory using spreading activation.

        This mimics how the brain retrieves related memories through
        neural pathway activation.
        """
        # Use spreading activation
        activations = await self.connection_engine.spreading_activation(
            [memory_id],
            max_depth=max_depth
        )

        # Filter and sort
        results = [
            (mid, act) for mid, act in activations.items()
            if mid != memory_id
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    async def complete_pattern(
        self,
        partial_memories: List[UUID],
        top_k: int = 5
    ) -> List[Tuple[UUID, float]]:
        """
        Complete a partial memory pattern by finding likely associated memories.

        Given a set of memories, finds other memories that commonly
        appear together with them.
        """
        return await self.connection_engine.pattern_completion(
            partial_memories, top_k
        )

    async def find_bridge(
        self,
        memory_a: UUID,
        memory_b: UUID
    ) -> List[List[UUID]]:
        """
        Find memories that bridge (connect) two seemingly unrelated memories.

        Discovers hidden relationships that might have been missed.
        """
        return await self.connection_engine.find_bridging_memories(
            memory_a, memory_b
        )

    async def suggest_connections(
        self,
        memory_id: UUID,
        top_k: int = 5
    ) -> List[Tuple[UUID, float, str]]:
        """
        Suggest potential connections for a memory.

        Returns (memory_id, confidence, reason) tuples.
        """
        return await self.connection_engine.suggest_connections(memory_id, top_k)

    # ==================== Smart Management Operations ====================

    async def optimize(self):
        """
        Run optimization routines:
        - Deduplicate across tiers
        - Clean cold memories
        - Rebalance tier distribution
        """
        logger.info("Running optimization...")

        # Request burst capacity
        await self.cpu_manager.request_burst(30.0)

        # Deduplicate
        dedup_results = await self.deduplicator.auto_deduplicate(
            batch_size=200,
            merge_semantic=True
        )

        # Clean cold memories
        cleaned = await self.tiered_db.cleanup_cold_memories()

        # Clean orphaned connections
        orphans = await self.deduplicator.clean_orphaned_connections()

        # Decay old connections
        decayed = await self.connection_engine.decay_connections()

        logger.info("Optimization complete",
                   deduped=dedup_results.get("total_merged", 0),
                   cold_cleaned=cleaned,
                   orphans_removed=orphans,
                   connections_decayed=decayed)

        return {
            "deduplication": dedup_results,
            "cold_cleaned": cleaned,
            "orphans_removed": orphans,
            "connections_decayed": decayed
        }

    async def rebalance_tiers(self):
        """
        Rebalance memory distribution across tiers based on access patterns.
        """
        distribution = await self.migration_manager.get_tier_distribution()
        logger.info("Current tier distribution", **distribution)

        # Promote hot memories from slow tiers
        for tier in [StorageTier.ULTRA_HIGH, StorageTier.HIGH, StorageTier.MEDIUM]:
            for memory_id, record in self.tiered_db.tier_indexes[tier].items():
                if record.temperature > self.config.promote_temperature:
                    await self.migration_manager.force_promote(
                        memory_id,
                        StorageTier.RAMDISK
                    )

        # Demote cold memories from fast tiers
        coldest = await self.ramdisk.get_coldest_memories(top_k=50)
        for memory_id, temp in coldest:
            if temp < self.config.demote_temperature:
                await self.migration_manager.force_demote(
                    memory_id,
                    StorageTier.MEDIUM
                )

        new_distribution = await self.migration_manager.get_tier_distribution()
        logger.info("Rebalanced tier distribution", **new_distribution)

        return new_distribution

    # ==================== Discovery Operations ====================

    async def discover_all_connections(
        self,
        batch_size: int = 100
    ) -> int:
        """
        Run connection discovery on all memories.
        This is the brain-like "making connections" operation.
        """
        logger.info("Running global connection discovery...")

        total_discovered = 0
        processed = 0

        # Process each tier
        for tier in StorageTier:
            for memory_id in list(self.tiered_db.tier_indexes[tier].keys()):
                if processed >= batch_size:
                    break

                connections = await self.connection_engine.discover_connections(
                    memory_id, search_tier=tier
                )
                total_discovered += len(connections)
                processed += 1

        # Detect clusters
        await self.connection_engine.detect_clusters()

        self.stats["connections_discovered"] += total_discovered
        logger.info("Connection discovery complete",
                   processed=processed,
                   discovered=total_discovered)

        return total_discovered

    async def _discover_connections_for_memory(self, memory_id: UUID):
        """Background task to discover connections for a new memory"""
        try:
            connections = await self.connection_engine.discover_connections(memory_id)
            self.stats["connections_discovered"] += len(connections)
        except Exception as e:
            logger.error("Connection discovery failed", memory_id=str(memory_id), error=str(e))

    # ==================== Background Tasks ====================

    async def _maintenance_loop(self):
        """Background maintenance loop"""
        interval = self.config.maintenance_interval_minutes * 60

        while self._running:
            try:
                await asyncio.sleep(interval)

                # Run deduplication
                if self.config.auto_dedup_interval_minutes > 0:
                    await self.deduplicator.auto_deduplicate(batch_size=50)

                # Clean orphaned connections
                await self.deduplicator.clean_orphaned_connections()

                # Decay connections
                await self.connection_engine.decay_connections()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Maintenance loop error", error=str(e))

    async def _connection_discovery_loop(self):
        """Background loop for continuous connection discovery"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Discover connections for random sample
                await self.discover_all_connections(batch_size=20)

                # Update clusters
                await self.connection_engine.detect_clusters()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Connection discovery loop error", error=str(e))

    # ==================== Statistics & Monitoring ====================

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the neural storage system"""
        tier_stats = await self.tiered_db.get_tier_stats()
        cpu_stats = await self.cpu_manager.get_resource_stats()
        ramdisk_stats = await self.ramdisk.get_stats()
        connection_stats = await self.connection_engine.get_connection_stats()
        migration_stats = await self.migration_manager.get_migration_stats()
        dedup_stats = await self.deduplicator.get_deduplication_stats()

        return {
            "orchestrator": self.stats,
            "tiered_database": tier_stats,
            "cpu_manager": cpu_stats,
            "ramdisk": ramdisk_stats,
            "connections": connection_stats,
            "migration": migration_stats,
            "deduplication": dedup_stats,
            "config": {
                "enable_ultra_high_tier": self.config.enable_ultra_high_tier,
                "max_ramdisk_mb": self.config.max_ramdisk_mb,
                "similarity_threshold": self.config.similarity_threshold,
            }
        }

    async def get_health(self) -> Dict[str, Any]:
        """Get health status of all components"""
        ramdisk_stats = await self.ramdisk.get_stats()
        cpu_stats = await self.cpu_manager.get_resource_stats()

        return {
            "status": "healthy" if self._running else "stopped",
            "components": {
                "tiered_db": "healthy",
                "ramdisk": "healthy" if ramdisk_stats["entries"] >= 0 else "degraded",
                "cpu_manager": "healthy" if cpu_stats["allocation"]["cpu_cores"] > 0 else "degraded",
                "connection_engine": "healthy",
                "migration_manager": "healthy",
                "deduplicator": "healthy",
            },
            "metrics": {
                "total_memories": self.stats["total_stores"],
                "ramdisk_utilization": ramdisk_stats["current_size_mb"] / self.config.max_ramdisk_mb,
                "cpu_utilization": len(self.cpu_manager.active_tasks) / max(1, cpu_stats["allocation"]["max_concurrent_tasks"]),
            }
        }


# Singleton instance for global access
_orchestrator_instance: Optional[NeuralStorageOrchestrator] = None


async def get_neural_storage() -> NeuralStorageOrchestrator:
    """Get or create the global neural storage orchestrator"""
    global _orchestrator_instance

    if _orchestrator_instance is None:
        _orchestrator_instance = NeuralStorageOrchestrator()
        await _orchestrator_instance.start()

    return _orchestrator_instance


async def shutdown_neural_storage():
    """Shutdown the global neural storage orchestrator"""
    global _orchestrator_instance

    if _orchestrator_instance is not None:
        await _orchestrator_instance.stop()
        _orchestrator_instance = None
