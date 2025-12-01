"""
SpreadingActivationKernel - Neural Activation Propagation

Implements spreading activation for associative memory retrieval:
- Limited-depth activation with decay
- Top-K pruning per level
- Policy-aware traversal
- Query embedding influence

activation_next = f(activation_current * edge_weight * decay)
"""

import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from uuid import UUID
import structlog

from .core_abstractions import (
    MemoryGraph, MemoryGraphView, MemoryObject,
    RelationType, GraphEdge
)

logger = structlog.get_logger()


@dataclass
class ActivationConstraints:
    """Constraints for spreading activation"""
    agent_id: Optional[str] = None
    agent_groups: Set[str] = field(default_factory=set)
    max_time_window: Optional[datetime] = None  # Only activate memories before this time
    min_time_window: Optional[datetime] = None  # Only activate memories after this time
    allowed_relation_types: Optional[Set[RelationType]] = None
    blocked_compartments: Set[str] = field(default_factory=set)
    max_clearance_level: int = 10
    custom_filter: Optional[Callable[[MemoryObject], bool]] = None


@dataclass
class ActivationResult:
    """Result of spreading activation"""
    activations: Dict[UUID, float]  # node_id → activation level
    traversal_paths: Dict[UUID, List[UUID]]  # node_id → path from seed
    activation_sources: Dict[UUID, Set[UUID]]  # node_id → which seeds activated it
    subgraph_edges: List[Tuple[UUID, UUID, float]]  # (source, target, weight)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationConfig:
    """Configuration for activation parameters"""
    max_depth: int = 3
    decay_factor: float = 0.7
    activation_threshold: float = 0.1
    top_k_per_level: int = 50
    top_k_total: int = 100
    use_query_embedding: bool = True
    query_weight: float = 0.3
    edge_weight_power: float = 1.0  # Higher = stronger edge influence


class SpreadingActivationKernel:
    """
    Performs spreading activation for associative memory retrieval.

    Input: seed nodes + query embedding + constraints
    Output: ranked subgraph as candidate context set

    The activation spreads through the graph following weighted edges,
    with decay at each hop and pruning of low-activation nodes.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedding_hub: Any = None,  # EmbeddingAdapterHub
        config: Optional[ActivationConfig] = None
    ):
        self.graph = graph
        self.embedding_hub = embedding_hub
        self.config = config or ActivationConfig()

        # Statistics
        self.stats = {
            "total_activations": 0,
            "avg_nodes_activated": 0,
            "avg_depth_reached": 0,
        }

        logger.info("SpreadingActivationKernel initialized",
                   max_depth=self.config.max_depth,
                   decay_factor=self.config.decay_factor)

    async def activate(
        self,
        seed_ids: List[UUID],
        query_embedding: Optional[np.ndarray] = None,
        constraints: Optional[ActivationConstraints] = None,
        config: Optional[ActivationConfig] = None
    ) -> ActivationResult:
        """
        Perform spreading activation from seed nodes.

        Args:
            seed_ids: Starting nodes for activation
            query_embedding: Optional embedding to bias activation toward similar nodes
            constraints: Policy and time constraints
            config: Override default configuration

        Returns:
            ActivationResult with ranked subgraph
        """
        cfg = config or self.config
        constraints = constraints or ActivationConstraints()

        # Get graph view for agent
        if constraints.agent_id:
            view = self.graph.get_view(
                constraints.agent_id,
                constraints.agent_groups
            )
        else:
            view = self.graph.get_view("__global__", set())

        # Initialize activations
        activations: Dict[UUID, float] = {seed_id: 1.0 for seed_id in seed_ids}
        traversal_paths: Dict[UUID, List[UUID]] = {seed_id: [seed_id] for seed_id in seed_ids}
        activation_sources: Dict[UUID, Set[UUID]] = {seed_id: {seed_id} for seed_id in seed_ids}

        visited = set(seed_ids)
        frontier = set(seed_ids)
        all_edges = []

        # Spread activation layer by layer
        for depth in range(cfg.max_depth):
            if not frontier:
                break

            current_decay = cfg.decay_factor ** depth
            next_frontier = set()
            level_activations: Dict[UUID, float] = {}

            for node_id in frontier:
                source_activation = activations.get(node_id, 0)
                if source_activation < cfg.activation_threshold:
                    continue

                # Get neighbors
                neighbors = view.get_neighbors(node_id, constraints.allowed_relation_types)

                for neighbor_id, edge in neighbors:
                    # Apply constraints
                    if not self._passes_constraints(neighbor_id, constraints, view):
                        continue

                    # Calculate propagated activation
                    edge_weight = edge.weight ** cfg.edge_weight_power
                    propagated = source_activation * edge_weight * current_decay

                    # Add query embedding influence
                    if cfg.use_query_embedding and query_embedding is not None:
                        neighbor_node = view.get_node(neighbor_id)
                        if neighbor_node and neighbor_node.canonical_embedding is not None:
                            similarity = self._compute_similarity(
                                query_embedding, neighbor_node.canonical_embedding
                            )
                            propagated = (
                                (1 - cfg.query_weight) * propagated +
                                cfg.query_weight * similarity * source_activation * current_decay
                            )

                    if propagated >= cfg.activation_threshold:
                        # Accumulate activation (multiple paths can activate same node)
                        current = level_activations.get(neighbor_id, 0)
                        level_activations[neighbor_id] = min(1.0, current + propagated)

                        # Track path and sources
                        if neighbor_id not in traversal_paths:
                            traversal_paths[neighbor_id] = traversal_paths[node_id] + [neighbor_id]
                        if neighbor_id not in activation_sources:
                            activation_sources[neighbor_id] = set()
                        activation_sources[neighbor_id].update(activation_sources[node_id])

                        # Track edge
                        all_edges.append((node_id, neighbor_id, edge_weight))

                        if neighbor_id not in visited:
                            next_frontier.add(neighbor_id)

            # Prune to top-K for this level
            if level_activations:
                sorted_activations = sorted(
                    level_activations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:cfg.top_k_per_level]

                for node_id, activation in sorted_activations:
                    activations[node_id] = activation
                    visited.add(node_id)

                next_frontier = {nid for nid, _ in sorted_activations if nid in next_frontier}

            frontier = next_frontier

        # Final top-K pruning
        final_activations = dict(
            sorted(
                activations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:cfg.top_k_total]
        )

        # Filter edges to only include activated nodes
        activated_set = set(final_activations.keys())
        filtered_edges = [
            (s, t, w) for s, t, w in all_edges
            if s in activated_set and t in activated_set
        ]

        self.stats["total_activations"] += 1

        return ActivationResult(
            activations=final_activations,
            traversal_paths={k: v for k, v in traversal_paths.items() if k in activated_set},
            activation_sources={k: v for k, v in activation_sources.items() if k in activated_set},
            subgraph_edges=filtered_edges,
            stats={
                "nodes_activated": len(final_activations),
                "edges_traversed": len(all_edges),
                "max_depth_reached": min(cfg.max_depth, depth + 1) if frontier else depth,
            }
        )

    def _passes_constraints(
        self,
        node_id: UUID,
        constraints: ActivationConstraints,
        view: MemoryGraphView
    ) -> bool:
        """Check if a node passes activation constraints"""
        node = view.get_node(node_id)
        if node is None:
            return False

        # Time window constraints
        if constraints.max_time_window:
            if node.timestamps.created_at > constraints.max_time_window:
                return False

        if constraints.min_time_window:
            if node.timestamps.created_at < constraints.min_time_window:
                return False

        # Compartment constraints
        if constraints.blocked_compartments:
            if node.policy.compartment_tags & constraints.blocked_compartments:
                return False

        # Clearance level
        if node.policy.clearance_level > constraints.max_clearance_level:
            return False

        # Custom filter
        if constraints.custom_filter:
            if not constraints.custom_filter(node):
                return False

        return True

    def _compute_similarity(
        self,
        query: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings"""
        if self.embedding_hub:
            return self.embedding_hub.compute_similarity(query, target)

        # Fallback to simple cosine
        query = np.asarray(query, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        # Handle dimension mismatch
        if len(query) != len(target):
            min_dim = min(len(query), len(target))
            query = query[:min_dim]
            target = target[:min_dim]

        q_norm = np.linalg.norm(query)
        t_norm = np.linalg.norm(target)

        if q_norm > 0 and t_norm > 0:
            return float(np.dot(query, target) / (q_norm * t_norm))
        return 0.0

    async def activate_with_context(
        self,
        context_memories: List[UUID],
        query_embedding: np.ndarray,
        constraints: Optional[ActivationConstraints] = None,
        boost_recent: bool = True
    ) -> ActivationResult:
        """
        Activate from context (recent dialog, task context) with query focus.

        Useful for retrieving context-relevant memories for an AI.
        """
        constraints = constraints or ActivationConstraints()

        # Boost recent context memories
        config = ActivationConfig(
            max_depth=self.config.max_depth,
            decay_factor=self.config.decay_factor,
            activation_threshold=self.config.activation_threshold,
            top_k_per_level=self.config.top_k_per_level * 2,  # More exploration
            top_k_total=self.config.top_k_total,
            use_query_embedding=True,
            query_weight=0.4,  # Higher query influence
        )

        return await self.activate(
            seed_ids=context_memories,
            query_embedding=query_embedding,
            constraints=constraints,
            config=config
        )

    async def bidirectional_activation(
        self,
        source_ids: List[UUID],
        target_ids: List[UUID],
        constraints: Optional[ActivationConstraints] = None
    ) -> Tuple[ActivationResult, ActivationResult, Set[UUID]]:
        """
        Perform bidirectional activation to find meeting points.

        Useful for finding bridging memories between two sets.
        """
        # Activate from sources
        source_result = await self.activate(
            seed_ids=source_ids,
            constraints=constraints,
            config=ActivationConfig(
                max_depth=self.config.max_depth,
                decay_factor=self.config.decay_factor,
                top_k_total=self.config.top_k_total * 2,
            )
        )

        # Activate from targets
        target_result = await self.activate(
            seed_ids=target_ids,
            constraints=constraints,
            config=ActivationConfig(
                max_depth=self.config.max_depth,
                decay_factor=self.config.decay_factor,
                top_k_total=self.config.top_k_total * 2,
            )
        )

        # Find intersection (meeting points)
        source_activated = set(source_result.activations.keys())
        target_activated = set(target_result.activations.keys())
        meeting_points = source_activated & target_activated

        return source_result, target_result, meeting_points

    def get_ranked_context(
        self,
        result: ActivationResult,
        include_paths: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Convert activation result to ranked context list.

        Returns list of dicts with node info sorted by activation.
        """
        ranked = []

        for node_id, activation in sorted(
            result.activations.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            entry = {
                "memory_id": str(node_id),
                "activation": activation,
                "sources": [str(s) for s in result.activation_sources.get(node_id, set())],
            }

            if include_paths:
                path = result.traversal_paths.get(node_id, [])
                entry["path"] = [str(p) for p in path]
                entry["path_length"] = len(path) - 1

            ranked.append(entry)

        return ranked

    def get_stats(self) -> Dict[str, Any]:
        """Get kernel statistics"""
        return {
            "total_activations": self.stats["total_activations"],
            "config": {
                "max_depth": self.config.max_depth,
                "decay_factor": self.config.decay_factor,
                "activation_threshold": self.config.activation_threshold,
                "top_k_per_level": self.config.top_k_per_level,
                "top_k_total": self.config.top_k_total,
            }
        }
