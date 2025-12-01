"""
Memory Fabric - Unified Brain-Inspired Memory System for Multi-AI

Central coordination layer that:
- Schedules: indexing, migration, dedup, Hebbian sweeps
- Manages cluster-wide placement if distributed
- Exposes internal "plans" for audit + tuning
- Provides unified API for all AI systems
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
import structlog

from .core_abstractions import (
    MemoryObject, MemoryGraph, MemoryGraphView, RelationType,
    ShareScope, StorageTierLevel, EmbeddingProjection
)
from .embedding_adapter_hub import EmbeddingAdapterHub
from .spreading_activation import (
    SpreadingActivationKernel, ActivationConstraints, ActivationConfig, ActivationResult
)
from .pattern_completion import PatternCompletionEngine, CompletionResult
from .policy_engine import PolicyEngine, AgentProfile, AccessLevel
from .telemetry_tuner import TelemetryAutoTuner
from .neural_connection_engine import NeuralConnectionEngine
from .tiered_database import TieredDatabaseManager, StorageTier
from .ramdisk_storage import RAMDiskStorage
from .memory_migration import MemoryMigrationManager
from .deduplication import CrossTierDeduplicator
from .dynamic_cpu_manager import DynamicCPUManager, WorkloadIntensity

logger = structlog.get_logger()


@dataclass
class FabricConfig:
    """Configuration for the Memory Fabric"""
    # Embedding
    canonical_dimension: int = 256
    enable_multi_model: bool = True

    # Tiers
    t0_cache_size_mb: int = 64
    t1_ramdisk_size_mb: int = 512
    enable_nvme_tier: bool = True
    enable_cold_tier: bool = True

    # Activation
    max_activation_depth: int = 3
    activation_decay: float = 0.7
    activation_threshold: float = 0.1

    # Learning
    hebbian_learning_rate: float = 0.1
    connection_decay_rate: float = 0.01

    # Multi-AI
    enable_policy_engine: bool = True
    enable_compartments: bool = True

    # Auto-tuning
    enable_auto_tuning: bool = True
    tuning_interval_seconds: int = 300

    # Background tasks
    enable_background_tasks: bool = True
    maintenance_interval_minutes: int = 15


class MemoryFabric:
    """
    Unified brain-inspired memory fabric for multiple AI systems.

    Provides:
    - Fast associative recall via spreading activation
    - Cross-agent knowledge sharing (under policy)
    - Self-optimizing tiered storage
    - Pattern completion for partial cues
    - Hebbian learning for connection strengthening
    """

    def __init__(self, config: Optional[FabricConfig] = None):
        self.config = config or FabricConfig()

        # Core components
        self.graph = MemoryGraph()
        self.embedding_hub = EmbeddingAdapterHub(
            canonical_dimension=self.config.canonical_dimension
        )

        # Storage tiers (using existing components)
        self.tiered_db = TieredDatabaseManager(
            enable_ultra_high=self.config.enable_cold_tier,
            max_ramdisk_mb=self.config.t1_ramdisk_size_mb
        )
        self.ramdisk = RAMDiskStorage(
            max_memory_mb=self.config.t1_ramdisk_size_mb,
            embedding_dim=self.config.canonical_dimension
        )
        self.cpu_manager = DynamicCPUManager()

        # Connection and migration
        self.connection_engine = NeuralConnectionEngine(
            tiered_db=self.tiered_db,
            similarity_threshold=0.7,
            hebbian_learning_rate=self.config.hebbian_learning_rate
        )
        self.migration_manager = MemoryMigrationManager(
            tiered_db=self.tiered_db,
            ramdisk=self.ramdisk,
            cpu_manager=self.cpu_manager
        )
        self.deduplicator = CrossTierDeduplicator(
            tiered_db=self.tiered_db,
            ramdisk=self.ramdisk
        )

        # Activation and completion
        self.activation_kernel = SpreadingActivationKernel(
            graph=self.graph,
            embedding_hub=self.embedding_hub,
            config=ActivationConfig(
                max_depth=self.config.max_activation_depth,
                decay_factor=self.config.activation_decay,
                activation_threshold=self.config.activation_threshold
            )
        )
        self.pattern_engine = PatternCompletionEngine(
            graph=self.graph,
            activation_kernel=self.activation_kernel,
            embedding_hub=self.embedding_hub
        )

        # Policy
        self.policy_engine = PolicyEngine() if self.config.enable_policy_engine else None

        # Telemetry
        self.telemetry = TelemetryAutoTuner()

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self.stats = {
            "memories_stored": 0,
            "queries_processed": 0,
            "activations_performed": 0,
            "completions_performed": 0,
            "connections_created": 0,
        }

        logger.info("MemoryFabric initialized",
                   canonical_dim=self.config.canonical_dimension)

    async def start(self):
        """Start the memory fabric"""
        logger.info("Starting Memory Fabric...")

        # Start components
        await self.cpu_manager.start()
        await self.ramdisk.start()
        await self.migration_manager.start()

        if self.config.enable_auto_tuning:
            await self.telemetry.start()

        self._running = True

        # Start background tasks
        if self.config.enable_background_tasks:
            self._tasks.append(asyncio.create_task(self._maintenance_loop()))
            self._tasks.append(asyncio.create_task(self._hebbian_sweep_loop()))

        logger.info("Memory Fabric started")

    async def stop(self):
        """Stop the memory fabric"""
        logger.info("Stopping Memory Fabric...")

        self._running = False

        for task in self._tasks:
            task.cancel()

        await self.telemetry.stop()
        await self.migration_manager.stop()
        await self.ramdisk.stop()
        await self.cpu_manager.stop()

        logger.info("Memory Fabric stopped")

    # ==================== Data Plane API ====================

    async def put_memory(
        self,
        content: str,
        embedding: np.ndarray,
        model_id: str = "default",
        agent_id: Optional[str] = None,
        share_scope: ShareScope = ShareScope.GLOBAL,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> MemoryObject:
        """
        Store a memory object.

        Data Plane API: PutMemoryObject(mem_object, scope)
        """
        self.stats["memories_stored"] += 1

        # Create memory object
        memory = MemoryObject(
            id=uuid4(),
            content=content,
            metadata=metadata or {}
        )

        # Add embedding with canonical projection
        canonical = self.embedding_hub.project_to_canonical(embedding, model_id)
        memory.add_embedding(model_id, embedding)
        memory.add_embedding("canonical", canonical, is_canonical=True)

        # Set policy
        memory.policy.share_scope = share_scope
        memory.policy.owner_agent_id = agent_id

        # Add to graph
        self.graph.add_node(memory)

        # Add to session timeline
        if session_id:
            memory.session_id = session_id
            self.graph.add_to_session_timeline(session_id, memory.id)

        # Store in tiered DB
        await self.tiered_db.store(
            memory_id=memory.id,
            content_hash=memory.content_hash,
            embedding=canonical,
            initial_tier=StorageTier.HIGH,
            metadata=metadata
        )

        # Record telemetry
        self.telemetry.telemetry.record_hit("write", 0, agent_id)

        # Discover connections in background
        asyncio.create_task(self._discover_connections(memory.id))

        return memory

    async def query_by_embedding(
        self,
        embedding: np.ndarray,
        agent_id: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryObject, float]]:
        """
        Query memories by embedding similarity.

        Data Plane API: QueryByEmbedding(embedding, filters, k)
        """
        self.stats["queries_processed"] += 1
        start_time = datetime.utcnow()

        # Project to canonical
        canonical = self.embedding_hub.project_to_canonical(embedding)

        # Search tiered DB
        results = await self.tiered_db.search_all_tiers(
            canonical, top_k=k * 2, threshold=0.3
        )

        # Get memory objects with policy filtering
        memories = []
        for mem_id, score, tier in results:
            memory = self.graph.nodes.get(mem_id)
            if memory:
                # Policy check
                if self.policy_engine and agent_id:
                    decision = self.policy_engine.check_access(agent_id, memory)
                    if not decision.allowed:
                        continue
                memories.append((memory, score))

        # Record telemetry
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.telemetry.telemetry.record_hit("query", latency_ms, agent_id)

        return memories[:k]

    async def query_graph(
        self,
        seed_ids: List[UUID],
        agent_id: Optional[str] = None,
        depth: int = 2,
        budget: int = 50
    ) -> ActivationResult:
        """
        Query by graph traversal from seeds.

        Data Plane API: QueryGraph(seed_ids, depth, budget)
        """
        self.stats["activations_performed"] += 1

        constraints = ActivationConstraints(agent_id=agent_id) if agent_id else None

        result = await self.activation_kernel.activate(
            seed_ids=seed_ids,
            constraints=constraints,
            config=ActivationConfig(
                max_depth=depth,
                top_k_total=budget
            )
        )

        # Record telemetry
        self.telemetry.telemetry.record_activation(
            depth, len(result.activations), agent_id
        )

        return result

    async def recall_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        agent_id: str,
        context_memories: Optional[List[UUID]] = None,
        constraints: Optional[ActivationConstraints] = None,
        max_results: int = 20
    ) -> Tuple[List[MemoryObject], List[Tuple[UUID, UUID, float]]]:
        """
        Recall context for an AI agent.

        Data Plane API: RecallContext(query, agent_id, constraints) → ranked MemoryObjects + edges

        Combines:
        1. ANN search for semantic matches
        2. Spreading activation from context
        3. Policy filtering
        """
        self.stats["completions_performed"] += 1

        # Set agent constraints
        if constraints is None:
            constraints = ActivationConstraints(agent_id=agent_id)

        # Pattern completion
        result = await self.pattern_engine.complete(
            query_embedding=self.embedding_hub.project_to_canonical(query_embedding),
            partial_cues=context_memories,
            constraints=constraints,
            top_k=max_results
        )

        # Get memory objects
        memories = []
        for cand in result.candidates:
            memory = self.graph.nodes.get(cand.memory_id)
            if memory:
                memories.append(memory)

        # Get edges from activation subgraph
        if context_memories:
            activation = await self.activation_kernel.activate(
                seed_ids=context_memories,
                query_embedding=query_embedding,
                constraints=constraints
            )
            edges = activation.subgraph_edges
        else:
            edges = []

        return memories, edges

    # ==================== Agent Management ====================

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        groups: Optional[Set[str]] = None,
        clearance: AccessLevel = AccessLevel.INTERNAL
    ) -> Optional[AgentProfile]:
        """Register a new AI agent"""
        if not self.policy_engine:
            return None

        return self.policy_engine.register_agent(
            agent_id=agent_id,
            name=name,
            groups=groups,
            clearance=clearance
        )

    def get_agent_view(
        self,
        agent_id: str,
        agent_groups: Optional[Set[str]] = None
    ) -> MemoryGraphView:
        """Get filtered graph view for an agent"""
        policy_filter = None
        if self.policy_engine:
            policy_filter = self.policy_engine.get_agent_view_filter(agent_id)

        return self.graph.get_view(
            agent_id=agent_id,
            agent_groups=agent_groups or set(),
            policy_filter=policy_filter
        )

    # ==================== Connection Operations ====================

    async def add_connection(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: RelationType,
        weight: float = 1.0,
        agent_id: Optional[str] = None
    ) -> bool:
        """Add a connection between memories"""
        # Policy check
        if self.policy_engine and agent_id:
            source = self.graph.nodes.get(source_id)
            target = self.graph.nodes.get(target_id)
            if source and target:
                decision = self.policy_engine.check_link_access(
                    agent_id, source, target, relation_type
                )
                if not decision.allowed:
                    return False

        edge = self.graph.add_edge(
            source_id, target_id, relation_type, weight,
            source_agent=agent_id
        )

        if edge:
            self.stats["connections_created"] += 1
            return True
        return False

    async def _discover_connections(self, memory_id: UUID):
        """Background task to discover connections"""
        try:
            await self.connection_engine.discover_connections(memory_id)
        except Exception as e:
            logger.error("Connection discovery failed", error=str(e))

    # ==================== Learning Operations ====================

    async def hebbian_update(
        self,
        co_activated_ids: List[UUID],
        agent_id: Optional[str] = None
    ):
        """Apply Hebbian learning to co-activated memories"""
        await self.connection_engine.learn_from_coactivation(
            co_activated_ids, agent_id
        )

    async def strengthen_path(
        self,
        path: List[UUID],
        reward: float = 1.0
    ):
        """Strengthen connections along a retrieval path"""
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            # Find and strengthen edge
            for key, edge in self.graph.edges.items():
                if key[0] == source_id and key[1] == target_id:
                    edge.weight = min(1.0, edge.weight + self.config.hebbian_learning_rate * reward)
                    edge.activation_count += 1
                    edge.last_activated = datetime.utcnow()

    # ==================== Background Tasks ====================

    async def _maintenance_loop(self):
        """Background maintenance"""
        interval = self.config.maintenance_interval_minutes * 60

        while self._running:
            try:
                await asyncio.sleep(interval)

                # Deduplication
                await self.deduplicator.auto_deduplicate(batch_size=50)

                # Connection decay
                await self.connection_engine.decay_connections()

                # Clean orphans
                await self.deduplicator.clean_orphaned_connections()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Maintenance error", error=str(e))

    async def _hebbian_sweep_loop(self):
        """Background Hebbian sweep"""
        while self._running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes

                # Decay all connections slightly
                for edge in self.graph.edges.values():
                    hours_since = (datetime.utcnow() - edge.last_activated).total_seconds() / 3600
                    decay = np.exp(-self.config.connection_decay_rate * hours_since)
                    edge.weight *= decay

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Hebbian sweep error", error=str(e))

    # ==================== Statistics ====================

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive fabric statistics"""
        return {
            "fabric": self.stats,
            "graph": {
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
                "agents": len(self.graph.agent_overlays),
            },
            "telemetry": self.telemetry.get_combined_stats(),
            "policy": self.policy_engine.export_policy() if self.policy_engine else None,
        }

    async def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "status": "healthy" if self._running else "stopped",
            "components": {
                "graph": "healthy",
                "activation": "healthy",
                "policy": "healthy" if self.policy_engine else "disabled",
                "telemetry": "healthy",
            }
        }


# Singleton instance
_fabric_instance: Optional[MemoryFabric] = None


async def get_memory_fabric() -> MemoryFabric:
    """Get or create the global memory fabric"""
    global _fabric_instance

    if _fabric_instance is None:
        _fabric_instance = MemoryFabric()
        await _fabric_instance.start()

    return _fabric_instance


async def shutdown_memory_fabric():
    """Shutdown the global memory fabric"""
    global _fabric_instance

    if _fabric_instance is not None:
        await _fabric_instance.stop()
        _fabric_instance = None
