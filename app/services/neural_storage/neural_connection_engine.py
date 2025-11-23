"""
Neural Connection Engine - Brain-like Pattern Discovery

Implements a neural network-inspired system for discovering and creating
connections between memories, similar to how the human brain forms synapses.

Features:
- Automatic connection discovery based on semantic similarity
- Hebbian learning: "neurons that fire together wire together"
- Connection strengthening through repeated co-activation
- Pattern completion and association retrieval
- Spreading activation for related memory discovery
"""

import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import structlog

from .tiered_database import TieredDatabaseManager, StorageTier, TieredMemoryRecord

logger = structlog.get_logger()


@dataclass
class SynapticConnection:
    """Represents a connection between two memories"""
    source_id: UUID
    target_id: UUID
    weight: float = 0.5  # Connection strength (0.0 to 1.0)
    connection_type: str = "semantic"  # semantic, temporal, causal, associative
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activated: datetime = field(default_factory=datetime.utcnow)
    activation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationPattern:
    """Pattern of neural activations across memories"""
    activated_memories: Dict[UUID, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trigger_query: Optional[str] = None


class NeuralConnectionEngine:
    """
    Brain-like connection discovery and management engine.

    Implements:
    - Hebbian learning for connection strengthening
    - Spreading activation for association discovery
    - Pattern completion for partial recall
    - Automatic connection creation from co-activation
    """

    def __init__(
        self,
        tiered_db: TieredDatabaseManager,
        similarity_threshold: float = 0.7,
        connection_decay_rate: float = 0.01,
        max_connections_per_memory: int = 100,
        hebbian_learning_rate: float = 0.1
    ):
        self.tiered_db = tiered_db
        self.similarity_threshold = similarity_threshold
        self.connection_decay_rate = connection_decay_rate
        self.max_connections = max_connections_per_memory
        self.learning_rate = hebbian_learning_rate

        # Connection graph
        self.connections: Dict[Tuple[UUID, UUID], SynapticConnection] = {}
        self.outgoing: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.incoming: Dict[UUID, Set[UUID]] = defaultdict(set)

        # Activation history for pattern learning
        self.activation_history: List[ActivationPattern] = []
        self.max_history = 1000

        # Cluster detection
        self.memory_clusters: Dict[int, Set[UUID]] = {}
        self.memory_to_cluster: Dict[UUID, int] = {}

        # Statistics
        self.stats = {
            "connections_created": 0,
            "connections_strengthened": 0,
            "connections_pruned": 0,
            "patterns_learned": 0,
            "spreading_activations": 0,
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("NeuralConnectionEngine initialized",
                   similarity_threshold=similarity_threshold,
                   max_connections=max_connections_per_memory)

    async def discover_connections(
        self,
        memory_id: UUID,
        search_tier: StorageTier = StorageTier.MEDIUM
    ) -> List[SynapticConnection]:
        """
        Discover potential connections for a memory based on semantic similarity.
        This is the primary mechanism for "making connections like a human brain".
        """
        # Get the memory's embedding
        result = await self.tiered_db.retrieve(memory_id)
        if not result:
            return []

        embedding, _ = result

        # Search for similar memories across tiers
        similar_memories = await self.tiered_db.search_all_tiers(
            embedding,
            top_k=50,
            threshold=self.similarity_threshold * 0.8,  # Slightly lower threshold for discovery
            start_tier=search_tier
        )

        new_connections = []

        async with self._lock:
            for other_id, similarity, tier in similar_memories:
                if other_id == memory_id:
                    continue

                # Check if connection already exists
                conn_key = self._connection_key(memory_id, other_id)
                if conn_key in self.connections:
                    # Strengthen existing connection
                    await self._strengthen_connection(conn_key, similarity)
                    continue

                # Create new connection
                connection = await self._create_connection(
                    memory_id, other_id,
                    weight=similarity,
                    connection_type="semantic"
                )
                if connection:
                    new_connections.append(connection)

        logger.info("Connections discovered",
                   memory_id=str(memory_id),
                   new_connections=len(new_connections))

        return new_connections

    async def _create_connection(
        self,
        source_id: UUID,
        target_id: UUID,
        weight: float,
        connection_type: str = "semantic"
    ) -> Optional[SynapticConnection]:
        """Create a new synaptic connection"""
        # Check connection limits
        if len(self.outgoing[source_id]) >= self.max_connections:
            # Prune weakest connection
            await self._prune_weakest(source_id)

        connection = SynapticConnection(
            source_id=source_id,
            target_id=target_id,
            weight=min(1.0, weight),
            connection_type=connection_type
        )

        conn_key = self._connection_key(source_id, target_id)
        self.connections[conn_key] = connection
        self.outgoing[source_id].add(target_id)
        self.incoming[target_id].add(source_id)

        # Also add to tiered DB for persistence
        await self.tiered_db.add_connection(source_id, target_id, weight, bidirectional=True)

        self.stats["connections_created"] += 1
        return connection

    async def _strengthen_connection(self, conn_key: Tuple[UUID, UUID], activation: float):
        """Strengthen connection using Hebbian learning"""
        if conn_key not in self.connections:
            return

        conn = self.connections[conn_key]

        # Hebbian update: Δw = η * activation
        delta = self.learning_rate * activation
        conn.weight = min(1.0, conn.weight + delta)
        conn.last_activated = datetime.utcnow()
        conn.activation_count += 1

        self.stats["connections_strengthened"] += 1

    async def _prune_weakest(self, memory_id: UUID):
        """Prune the weakest connection from a memory"""
        if not self.outgoing[memory_id]:
            return

        # Find weakest connection
        weakest_target = None
        weakest_weight = float('inf')

        for target_id in self.outgoing[memory_id]:
            conn_key = self._connection_key(memory_id, target_id)
            conn = self.connections.get(conn_key)
            if conn and conn.weight < weakest_weight:
                weakest_weight = conn.weight
                weakest_target = target_id

        if weakest_target:
            await self._remove_connection(memory_id, weakest_target)
            self.stats["connections_pruned"] += 1

    async def _remove_connection(self, source_id: UUID, target_id: UUID):
        """Remove a connection"""
        conn_key = self._connection_key(source_id, target_id)
        if conn_key in self.connections:
            del self.connections[conn_key]
        self.outgoing[source_id].discard(target_id)
        self.incoming[target_id].discard(source_id)

    def _connection_key(self, id1: UUID, id2: UUID) -> Tuple[UUID, UUID]:
        """Generate canonical connection key (smaller UUID first)"""
        return (id1, id2) if str(id1) < str(id2) else (id2, id1)

    async def spreading_activation(
        self,
        seed_memories: List[UUID],
        max_depth: int = 3,
        activation_threshold: float = 0.3,
        decay_factor: float = 0.7
    ) -> Dict[UUID, float]:
        """
        Perform spreading activation from seed memories.
        Returns activation levels for all reached memories.

        This mimics how the brain retrieves associated memories through
        neural pathway activation.
        """
        self.stats["spreading_activations"] += 1

        # Initialize activation levels
        activations: Dict[UUID, float] = {mid: 1.0 for mid in seed_memories}
        frontier = set(seed_memories)
        visited = set()

        for depth in range(max_depth):
            if not frontier:
                break

            next_frontier = set()
            current_decay = decay_factor ** depth

            for memory_id in frontier:
                if memory_id in visited:
                    continue
                visited.add(memory_id)

                source_activation = activations.get(memory_id, 0)
                if source_activation < activation_threshold:
                    continue

                # Spread to connected memories
                for target_id in self.outgoing[memory_id]:
                    conn_key = self._connection_key(memory_id, target_id)
                    conn = self.connections.get(conn_key)
                    if not conn:
                        continue

                    # Calculate propagated activation
                    propagated = source_activation * conn.weight * current_decay

                    if propagated >= activation_threshold:
                        # Accumulate activation (multiple paths can activate same memory)
                        current = activations.get(target_id, 0)
                        activations[target_id] = min(1.0, current + propagated)
                        next_frontier.add(target_id)

            frontier = next_frontier - visited

        # Record activation pattern for learning
        pattern = ActivationPattern(
            activated_memories=activations.copy()
        )
        self.activation_history.append(pattern)
        if len(self.activation_history) > self.max_history:
            self.activation_history.pop(0)

        return activations

    async def pattern_completion(
        self,
        partial_memory_ids: List[UUID],
        top_k: int = 10
    ) -> List[Tuple[UUID, float]]:
        """
        Complete a partial memory pattern by finding most likely associated memories.
        This mimics how the brain recalls complete memories from partial cues.
        """
        # Perform spreading activation
        activations = await self.spreading_activation(
            partial_memory_ids,
            max_depth=2,
            activation_threshold=0.2
        )

        # Filter out seed memories and sort by activation
        completions = [
            (mid, act) for mid, act in activations.items()
            if mid not in partial_memory_ids
        ]
        completions.sort(key=lambda x: x[1], reverse=True)

        return completions[:top_k]

    async def find_bridging_memories(
        self,
        memory_a: UUID,
        memory_b: UUID,
        max_path_length: int = 4
    ) -> List[List[UUID]]:
        """
        Find memories that bridge (connect) two seemingly unrelated memories.
        This discovers hidden relationships the AI components might have missed.
        """
        paths = []

        # BFS for shortest paths
        queue = [(memory_a, [memory_a])]
        visited = {memory_a}

        while queue and len(paths) < 5:
            current, path = queue.pop(0)

            if len(path) > max_path_length:
                continue

            if current == memory_b:
                paths.append(path)
                continue

            for neighbor in self.outgoing[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return paths

    async def learn_from_coactivation(
        self,
        activated_memories: List[UUID],
        query_context: Optional[str] = None
    ):
        """
        Learn connections from co-activated memories (Hebbian learning).
        "Memories that are retrieved together become connected."
        """
        if len(activated_memories) < 2:
            return

        async with self._lock:
            # Create/strengthen connections between all co-activated pairs
            for i, mem_a in enumerate(activated_memories):
                for mem_b in activated_memories[i+1:]:
                    conn_key = self._connection_key(mem_a, mem_b)

                    if conn_key in self.connections:
                        # Strengthen existing connection
                        await self._strengthen_connection(conn_key, 0.5)
                    else:
                        # Create new associative connection
                        await self._create_connection(
                            mem_a, mem_b,
                            weight=0.3,  # Initial weight for co-activation
                            connection_type="associative"
                        )

            self.stats["patterns_learned"] += 1

        logger.debug("Learned from coactivation",
                    memories=len(activated_memories),
                    context=query_context[:50] if query_context else None)

    async def detect_clusters(self, min_cluster_size: int = 3) -> Dict[int, Set[UUID]]:
        """
        Detect clusters of strongly connected memories.
        These represent conceptual groupings in the memory space.
        """
        # Reset clusters
        self.memory_clusters = {}
        self.memory_to_cluster = {}

        visited = set()
        cluster_id = 0

        for memory_id in self.outgoing.keys():
            if memory_id in visited:
                continue

            # BFS to find cluster
            cluster = set()
            queue = [memory_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                # Add strongly connected neighbors
                for neighbor in self.outgoing[current]:
                    if neighbor in visited:
                        continue

                    conn_key = self._connection_key(current, neighbor)
                    conn = self.connections.get(conn_key)
                    if conn and conn.weight > 0.5:  # Strong connection
                        queue.append(neighbor)

            if len(cluster) >= min_cluster_size:
                self.memory_clusters[cluster_id] = cluster
                for mid in cluster:
                    self.memory_to_cluster[mid] = cluster_id
                cluster_id += 1

        logger.info("Clusters detected", num_clusters=len(self.memory_clusters))
        return self.memory_clusters

    async def decay_connections(self, time_decay_hours: float = 24):
        """
        Apply time-based decay to connections that haven't been activated.
        This mimics memory consolidation and forgetting.
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=time_decay_hours)
        decayed = 0

        async with self._lock:
            for conn_key, conn in list(self.connections.items()):
                if conn.last_activated < cutoff:
                    # Apply decay
                    conn.weight *= (1 - self.connection_decay_rate)

                    # Remove very weak connections
                    if conn.weight < 0.1:
                        await self._remove_connection(conn.source_id, conn.target_id)
                        decayed += 1
                    else:
                        decayed += 1

        if decayed > 0:
            logger.info("Connection decay applied", affected=decayed)

        return decayed

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection graph"""
        if not self.connections:
            return {
                "total_connections": 0,
                "avg_weight": 0,
                "stats": self.stats
            }

        weights = [c.weight for c in self.connections.values()]
        types = defaultdict(int)
        for conn in self.connections.values():
            types[conn.connection_type] += 1

        return {
            "total_connections": len(self.connections),
            "total_memories_connected": len(self.outgoing),
            "avg_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights)),
            "connection_types": dict(types),
            "num_clusters": len(self.memory_clusters),
            "stats": self.stats
        }

    async def suggest_connections(
        self,
        memory_id: UUID,
        top_k: int = 5
    ) -> List[Tuple[UUID, float, str]]:
        """
        Suggest potential connections for a memory based on patterns.
        Returns (memory_id, confidence, reason) tuples.
        """
        suggestions = []

        # 1. Find memories with similar connection patterns
        current_connections = self.outgoing.get(memory_id, set())
        for other_id in self.tiered_db.global_index.keys():
            if other_id == memory_id or other_id in current_connections:
                continue

            other_connections = self.outgoing.get(other_id, set())
            if not other_connections:
                continue

            # Jaccard similarity of connection sets
            intersection = len(current_connections & other_connections)
            union = len(current_connections | other_connections)
            if union > 0:
                similarity = intersection / union
                if similarity > 0.3:
                    suggestions.append((other_id, similarity, "similar_connections"))

        # 2. Find memories in same cluster but not connected
        if memory_id in self.memory_to_cluster:
            cluster_id = self.memory_to_cluster[memory_id]
            cluster_members = self.memory_clusters.get(cluster_id, set())
            for member in cluster_members:
                if member != memory_id and member not in current_connections:
                    suggestions.append((member, 0.7, "same_cluster"))

        # Sort by confidence and deduplicate
        suggestions.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique = []
        for mid, conf, reason in suggestions:
            if mid not in seen:
                seen.add(mid)
                unique.append((mid, conf, reason))

        return unique[:top_k]
