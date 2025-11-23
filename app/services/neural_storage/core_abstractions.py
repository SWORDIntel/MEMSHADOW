"""
Core Abstractions for Brain-Inspired Memory Fabric

MemoryObject: Unified memory representation with multi-model embeddings
MemoryGraph: Global graph + per-AI overlay graphs with typed relations
"""

import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4
import structlog

logger = structlog.get_logger()


class RelationType(str, Enum):
    """Types of relations between memory objects"""
    CAUSED_BY = "caused_by"
    CAUSES = "causes"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SAME_EVENT = "same_event"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    CONTAINS = "contains"
    TEMPORAL_NEXT = "temporal_next"
    TEMPORAL_PREV = "temporal_prev"
    ASSOCIATED = "associated"
    DERIVED_FROM = "derived_from"
    ALIAS_OF = "alias_of"
    SESSION_LINK = "session_link"
    EPISODIC_CHAIN = "episodic_chain"


class ShareScope(str, Enum):
    """Sharing scope for memory objects"""
    GLOBAL = "global"           # Visible to all AIs
    GROUP = "group"             # Visible to specific AI groups
    AGENT_LOCAL = "agent_local" # Private to specific AI


class StorageTierLevel(int, Enum):
    """Storage tier levels with associated characteristics"""
    T0_CACHE = 0      # Process-local KV cache (shared memory/mmap, 256d only)
    T1_RAMDISK = 1    # Hot working set, full vectors (up to 4096d)
    T2_NVME = 2       # NVMe vector store + graph store
    T3_COLD = 3       # Cold object store (compressed embeddings)


@dataclass
class EmbeddingProjection:
    """Embedding for a specific model"""
    model_id: str
    vector: np.ndarray
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_canonical: bool = False  # True for fabric space (256d/512d)


@dataclass
class MemoryLink:
    """A typed link between memory objects"""
    target_id: UUID
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activated: datetime = field(default_factory=datetime.utcnow)
    activation_count: int = 0
    source_agent_id: Optional[str] = None  # Which AI created this link


@dataclass
class MemoryTimestamps:
    """Timestamp tracking for memory objects"""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    last_activated: datetime = field(default_factory=datetime.utcnow)
    last_migrated: Optional[datetime] = None


@dataclass
class MemoryCounters:
    """Access and usage counters"""
    access_count: int = 0
    activation_count: int = 0
    retrieval_count: int = 0
    link_traversal_count: int = 0
    per_agent_access: Dict[str, int] = field(default_factory=dict)


@dataclass
class TierMetadata:
    """Tier placement and replication metadata"""
    current_tier: StorageTierLevel = StorageTierLevel.T1_RAMDISK
    replica_tiers: Set[StorageTierLevel] = field(default_factory=set)
    temperature: float = 0.5  # 0.0 = cold, 1.0 = hot
    hotness_score: float = 0.0
    last_tier_change: Optional[datetime] = None
    compression_level: int = 0  # 0 = none, higher = more compressed


@dataclass
class PolicyMetadata:
    """Access control and sharing policy"""
    share_scope: ShareScope = ShareScope.GLOBAL
    owner_agent_id: Optional[str] = None
    allowed_agents: Set[str] = field(default_factory=set)
    allowed_groups: Set[str] = field(default_factory=set)
    clearance_level: int = 0  # Higher = more restricted
    compartment_tags: Set[str] = field(default_factory=set)
    is_shadow_link: bool = False  # Global knowledge, local view


@dataclass
class MemoryObject:
    """
    Unified memory representation with multi-model embeddings.

    One logical object, many projections: per-model embeddings + graph links.
    This is the core data structure for the brain-inspired memory fabric.
    """
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    content_hash: str = ""

    # Multi-model embeddings: model_id → EmbeddingProjection
    embeddings: Dict[str, EmbeddingProjection] = field(default_factory=dict)

    # Canonical fabric space embedding (256d or 512d)
    canonical_embedding: Optional[np.ndarray] = None
    canonical_dimension: int = 256

    # Typed links to other memory objects
    links: List[MemoryLink] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamps: MemoryTimestamps = field(default_factory=MemoryTimestamps)
    counters: MemoryCounters = field(default_factory=MemoryCounters)
    tier_meta: TierMetadata = field(default_factory=TierMetadata)
    policy: PolicyMetadata = field(default_factory=PolicyMetadata)

    # Lineage tracking (for dedup merges)
    lineage: List[UUID] = field(default_factory=list)
    aliases: Set[UUID] = field(default_factory=set)

    # Session/episodic tracking
    session_id: Optional[str] = None
    episode_sequence: int = 0

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()

    def add_embedding(
        self,
        model_id: str,
        vector: np.ndarray,
        is_canonical: bool = False
    ):
        """Add or update embedding for a model"""
        self.embeddings[model_id] = EmbeddingProjection(
            model_id=model_id,
            vector=vector.astype(np.float32),
            dimension=len(vector),
            is_canonical=is_canonical
        )
        if is_canonical:
            self.canonical_embedding = vector.astype(np.float32)
            self.canonical_dimension = len(vector)

    def add_link(
        self,
        target_id: UUID,
        relation_type: RelationType,
        weight: float = 1.0,
        source_agent: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Add a link to another memory object"""
        link = MemoryLink(
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            source_agent_id=source_agent,
            metadata=metadata or {}
        )
        self.links.append(link)

    def get_links_by_type(self, relation_type: RelationType) -> List[MemoryLink]:
        """Get all links of a specific type"""
        return [l for l in self.links if l.relation_type == relation_type]

    def get_outgoing_ids(self) -> Set[UUID]:
        """Get all linked memory IDs"""
        return {link.target_id for link in self.links}

    def update_access(self, agent_id: Optional[str] = None):
        """Update access tracking"""
        self.timestamps.last_accessed = datetime.utcnow()
        self.counters.access_count += 1
        if agent_id:
            self.counters.per_agent_access[agent_id] = \
                self.counters.per_agent_access.get(agent_id, 0) + 1

    def calculate_hotness(
        self,
        recency_weight: float = 0.4,
        frequency_weight: float = 0.3,
        centrality_weight: float = 0.2,
        agent_importance_weight: float = 0.1,
        agent_importance: float = 1.0
    ) -> float:
        """
        Calculate hotness score for tier placement.

        hotness = f(access_frequency, recency, centrality, agent_importance)
        """
        now = datetime.utcnow()

        # Recency score (exponential decay)
        hours_since_access = (now - self.timestamps.last_accessed).total_seconds() / 3600
        recency_score = np.exp(-hours_since_access / 24)  # Half-life ~24h

        # Frequency score (log scale)
        frequency_score = min(1.0, np.log1p(self.counters.access_count) / 5)

        # Centrality score (based on link count)
        centrality_score = min(1.0, len(self.links) / 50)

        # Combine scores
        hotness = (
            recency_weight * recency_score +
            frequency_weight * frequency_score +
            centrality_weight * centrality_score +
            agent_importance_weight * agent_importance
        )

        self.tier_meta.hotness_score = hotness
        return hotness

    def is_accessible_by(self, agent_id: str, agent_groups: Set[str] = None) -> bool:
        """Check if an agent can access this memory"""
        policy = self.policy

        # Global scope is accessible to all
        if policy.share_scope == ShareScope.GLOBAL:
            return True

        # Owner always has access
        if policy.owner_agent_id == agent_id:
            return True

        # Check agent-level access
        if agent_id in policy.allowed_agents:
            return True

        # Check group-level access
        if agent_groups and policy.allowed_groups & agent_groups:
            return True

        return False


@dataclass
class GraphEdge:
    """Edge in the memory graph"""
    source_id: UUID
    target_id: UUID
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activated: datetime = field(default_factory=datetime.utcnow)
    activation_count: int = 0
    source_agent_id: Optional[str] = None
    is_bidirectional: bool = False


class MemoryGraph:
    """
    Global graph + per-AI overlay graphs.

    Nodes = MemoryObjects, edges = typed relations.
    Supports:
    - Global graph visible to all AIs
    - Per-AI overlay graphs with private nodes/edges
    - Policy-filtered views
    """

    def __init__(self):
        # Global graph
        self.nodes: Dict[UUID, MemoryObject] = {}
        self.edges: Dict[Tuple[UUID, UUID, RelationType], GraphEdge] = {}

        # Adjacency lists for fast traversal
        self.outgoing: Dict[UUID, Dict[UUID, List[GraphEdge]]] = {}
        self.incoming: Dict[UUID, Dict[UUID, List[GraphEdge]]] = {}

        # Per-agent overlays
        self.agent_overlays: Dict[str, 'AgentOverlay'] = {}

        # Temporal sequences
        self.session_timelines: Dict[str, List[UUID]] = {}
        self.episodic_chains: Dict[str, List[UUID]] = {}

        # Index structures
        self.hash_index: Dict[str, UUID] = {}  # content_hash → node_id
        self.tag_index: Dict[str, Set[UUID]] = {}  # tag → node_ids

        logger.info("MemoryGraph initialized")

    def add_node(self, memory_object: MemoryObject) -> bool:
        """Add a memory object to the graph"""
        if memory_object.id in self.nodes:
            return False

        self.nodes[memory_object.id] = memory_object
        self.outgoing[memory_object.id] = {}
        self.incoming[memory_object.id] = {}

        # Update indexes
        if memory_object.content_hash:
            self.hash_index[memory_object.content_hash] = memory_object.id

        # Add edges from the memory object's links
        for link in memory_object.links:
            self._add_edge_internal(
                memory_object.id, link.target_id,
                link.relation_type, link.weight,
                link.metadata, link.source_agent_id
            )

        return True

    def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
        source_agent: Optional[str] = None,
        bidirectional: bool = False
    ) -> Optional[GraphEdge]:
        """Add an edge between two memory objects"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge = self._add_edge_internal(
            source_id, target_id, relation_type, weight,
            metadata, source_agent, bidirectional
        )

        # Also update the source node's links
        source_node = self.nodes[source_id]
        source_node.add_link(target_id, relation_type, weight, source_agent, metadata)

        return edge

    def _add_edge_internal(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
        source_agent: Optional[str] = None,
        bidirectional: bool = False
    ) -> GraphEdge:
        """Internal edge creation"""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata or {},
            source_agent_id=source_agent,
            is_bidirectional=bidirectional
        )

        key = (source_id, target_id, relation_type)
        self.edges[key] = edge

        # Update adjacency
        if source_id not in self.outgoing:
            self.outgoing[source_id] = {}
        if target_id not in self.outgoing[source_id]:
            self.outgoing[source_id][target_id] = []
        self.outgoing[source_id][target_id].append(edge)

        if target_id not in self.incoming:
            self.incoming[target_id] = {}
        if source_id not in self.incoming[target_id]:
            self.incoming[target_id][source_id] = []
        self.incoming[target_id][source_id].append(edge)

        # Handle bidirectional
        if bidirectional:
            reverse_key = (target_id, source_id, relation_type)
            reverse_edge = GraphEdge(
                source_id=target_id,
                target_id=source_id,
                relation_type=relation_type,
                weight=weight,
                metadata=metadata or {},
                source_agent_id=source_agent,
                is_bidirectional=True
            )
            self.edges[reverse_key] = reverse_edge

        return edge

    def get_neighbors(
        self,
        node_id: UUID,
        relation_types: Optional[Set[RelationType]] = None,
        direction: str = "outgoing"  # "outgoing", "incoming", "both"
    ) -> List[Tuple[UUID, GraphEdge]]:
        """Get neighboring nodes with their edges"""
        neighbors = []

        if direction in ("outgoing", "both"):
            for target_id, edges in self.outgoing.get(node_id, {}).items():
                for edge in edges:
                    if relation_types is None or edge.relation_type in relation_types:
                        neighbors.append((target_id, edge))

        if direction in ("incoming", "both"):
            for source_id, edges in self.incoming.get(node_id, {}).items():
                for edge in edges:
                    if relation_types is None or edge.relation_type in relation_types:
                        neighbors.append((source_id, edge))

        return neighbors

    def get_view(
        self,
        agent_id: str,
        agent_groups: Set[str] = None,
        policy_filter: Optional[callable] = None
    ) -> 'MemoryGraphView':
        """
        Get a filtered view of the graph for a specific agent.

        View = global_graph ⊕ agent_overlay ⊕ policy_filter
        """
        return MemoryGraphView(
            graph=self,
            agent_id=agent_id,
            agent_groups=agent_groups or set(),
            policy_filter=policy_filter,
            overlay=self.agent_overlays.get(agent_id)
        )

    def get_or_create_overlay(self, agent_id: str) -> 'AgentOverlay':
        """Get or create an overlay for an agent"""
        if agent_id not in self.agent_overlays:
            self.agent_overlays[agent_id] = AgentOverlay(agent_id)
        return self.agent_overlays[agent_id]

    def add_to_session_timeline(
        self,
        session_id: str,
        memory_id: UUID
    ):
        """Add a memory to a session timeline"""
        if session_id not in self.session_timelines:
            self.session_timelines[session_id] = []
        self.session_timelines[session_id].append(memory_id)

        # Link to previous in session
        timeline = self.session_timelines[session_id]
        if len(timeline) > 1:
            prev_id = timeline[-2]
            self.add_edge(
                prev_id, memory_id,
                RelationType.TEMPORAL_NEXT, 1.0,
                {"session_id": session_id}
            )
            self.add_edge(
                memory_id, prev_id,
                RelationType.TEMPORAL_PREV, 1.0,
                {"session_id": session_id}
            )

    def merge_nodes(
        self,
        primary_id: UUID,
        secondary_id: UUID,
        mode: str = "soft"  # "soft" = alias, "hard" = unify
    ) -> bool:
        """
        Merge two nodes.

        Soft merge: link objects as aliases
        Hard merge: unify into single MemoryObject with lineage
        """
        if primary_id not in self.nodes or secondary_id not in self.nodes:
            return False

        primary = self.nodes[primary_id]
        secondary = self.nodes[secondary_id]

        if mode == "soft":
            # Create alias relationship
            self.add_edge(primary_id, secondary_id, RelationType.ALIAS_OF, 1.0)
            self.add_edge(secondary_id, primary_id, RelationType.ALIAS_OF, 1.0)
            primary.aliases.add(secondary_id)
            secondary.aliases.add(primary_id)

        elif mode == "hard":
            # Merge secondary into primary
            # Transfer embeddings
            for model_id, proj in secondary.embeddings.items():
                if model_id not in primary.embeddings:
                    primary.embeddings[model_id] = proj

            # Transfer links
            for link in secondary.links:
                if link.target_id != primary_id:
                    primary.add_link(
                        link.target_id, link.relation_type,
                        link.weight, link.source_agent_id, link.metadata
                    )

            # Update incoming edges to point to primary
            for source_id, edges in self.incoming.get(secondary_id, {}).items():
                for edge in edges:
                    self.add_edge(
                        source_id, primary_id, edge.relation_type,
                        edge.weight, edge.metadata, edge.source_agent_id
                    )

            # Track lineage
            primary.lineage.append(secondary_id)
            primary.aliases.add(secondary_id)

            # Remove secondary from graph (but keep in hash_index for lookups)
            del self.nodes[secondary_id]
            self.outgoing.pop(secondary_id, None)
            self.incoming.pop(secondary_id, None)

        return True


@dataclass
class AgentOverlay:
    """Per-agent overlay graph for private nodes and edges"""
    agent_id: str
    private_nodes: Dict[UUID, MemoryObject] = field(default_factory=dict)
    private_edges: Dict[Tuple[UUID, UUID, RelationType], GraphEdge] = field(default_factory=dict)
    notes: Dict[UUID, Dict[str, Any]] = field(default_factory=dict)  # Agent's notes on global nodes
    custom_embeddings: Dict[UUID, EmbeddingProjection] = field(default_factory=dict)


class MemoryGraphView:
    """
    A filtered view of the memory graph for a specific agent.

    Combines global graph + agent overlay + policy filtering.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        agent_id: str,
        agent_groups: Set[str],
        policy_filter: Optional[callable],
        overlay: Optional[AgentOverlay]
    ):
        self.graph = graph
        self.agent_id = agent_id
        self.agent_groups = agent_groups
        self.policy_filter = policy_filter
        self.overlay = overlay

    def get_node(self, node_id: UUID) -> Optional[MemoryObject]:
        """Get a node if accessible"""
        # Check overlay first
        if self.overlay and node_id in self.overlay.private_nodes:
            return self.overlay.private_nodes[node_id]

        # Check global graph
        if node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            if self._is_accessible(node):
                return node

        return None

    def _is_accessible(self, node: MemoryObject) -> bool:
        """Check if node is accessible to this agent"""
        if self.policy_filter and not self.policy_filter(node, self.agent_id):
            return False
        return node.is_accessible_by(self.agent_id, self.agent_groups)

    def get_neighbors(
        self,
        node_id: UUID,
        relation_types: Optional[Set[RelationType]] = None
    ) -> List[Tuple[UUID, GraphEdge]]:
        """Get accessible neighbors"""
        neighbors = self.graph.get_neighbors(node_id, relation_types)

        # Filter by accessibility
        accessible = []
        for neighbor_id, edge in neighbors:
            neighbor = self.get_node(neighbor_id)
            if neighbor is not None:
                accessible.append((neighbor_id, edge))

        # Add overlay neighbors
        if self.overlay:
            for key, edge in self.overlay.private_edges.items():
                if key[0] == node_id:
                    if relation_types is None or edge.relation_type in relation_types:
                        accessible.append((key[1], edge))

        return accessible

    def iter_accessible_nodes(self):
        """Iterate over all accessible nodes"""
        for node_id, node in self.graph.nodes.items():
            if self._is_accessible(node):
                yield node_id, node

        # Include overlay nodes
        if self.overlay:
            for node_id, node in self.overlay.private_nodes.items():
                yield node_id, node
