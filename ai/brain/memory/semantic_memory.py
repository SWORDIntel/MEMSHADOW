#!/usr/bin/env python3
"""
Semantic Memory (L3) for DSMIL Brain

Long-term knowledge graph storage:
- Concept nodes with relationship edges
- Confidence scoring on facts
- Auto-extraction from episodic memory
- Forgetting curves with importance weighting
- Temporal validity tracking
"""

import time
import hashlib
import threading
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple, Iterator
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ConceptType(Enum):
    """Types of concepts in the knowledge graph"""
    ENTITY = auto()       # Named entity (person, org, location)
    EVENT = auto()        # Event or occurrence
    ATTRIBUTE = auto()    # Property or attribute
    ACTION = auto()       # Action or behavior
    STATE = auto()        # State or condition
    ABSTRACT = auto()     # Abstract concept
    RELATIONSHIP = auto() # Relationship type
    THREAT = auto()       # Threat indicator
    PATTERN = auto()      # Behavioral pattern
    RULE = auto()         # Business/security rule


class RelationType(Enum):
    """Types of relationships between concepts"""
    # Hierarchical
    IS_A = auto()           # Type hierarchy
    PART_OF = auto()        # Composition
    HAS_PART = auto()       # Has component
    INSTANCE_OF = auto()    # Instance relationship

    # Causal
    CAUSES = auto()         # Causal relationship
    ENABLES = auto()        # Enabling condition
    PREVENTS = auto()       # Prevention
    INFLUENCES = auto()     # Influence

    # Associative
    RELATED_TO = auto()     # General relation
    SIMILAR_TO = auto()     # Similarity
    OPPOSITE_OF = auto()    # Opposition
    CORRELATES_WITH = auto() # Correlation

    # Temporal
    PRECEDES = auto()       # Temporal order
    FOLLOWS = auto()        # Temporal order
    DURING = auto()         # Temporal containment

    # Attribution
    HAS_ATTRIBUTE = auto()  # Attribute assignment
    PERFORMED_BY = auto()   # Actor
    TARGETS = auto()        # Target of action
    USES = auto()           # Tool/method usage

    # Security-specific
    INDICATES = auto()      # Threat indicator
    MITIGATES = auto()      # Mitigation
    EXPLOITS = auto()       # Exploitation


@dataclass
class Concept:
    """
    Node in the knowledge graph
    """
    concept_id: str
    name: str
    concept_type: ConceptType

    # Content
    description: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Confidence and importance
    confidence: float = 0.5  # 0-1, how confident we are in this concept
    importance: float = 0.5  # 0-1, how important is this concept

    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Source tracking
    source_episodes: Set[str] = field(default_factory=set)
    evidence_count: int = 1

    # Forgetting curve
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    retention_strength: float = 1.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self):
        """Mark concept as accessed"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        # Strengthen retention on access
        self.retention_strength = min(1.0, self.retention_strength + 0.1)

    def decay(self, decay_rate: float = 0.01):
        """Apply forgetting curve decay"""
        elapsed = (datetime.now(timezone.utc) - self.last_accessed).total_seconds()
        # Ebbinghaus forgetting curve
        self.retention_strength *= math.exp(-decay_rate * elapsed / 86400)  # Daily decay

    def get_effective_importance(self) -> float:
        """Get importance weighted by retention"""
        self.decay()
        return self.importance * self.confidence * self.retention_strength

    def reinforce(self, evidence_source: Optional[str] = None):
        """Reinforce concept with additional evidence"""
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        if evidence_source:
            self.source_episodes.add(evidence_source)
        self.touch()

    def is_valid(self, at_time: Optional[datetime] = None) -> bool:
        """Check if concept is valid at given time"""
        check_time = at_time or datetime.now(timezone.utc)

        if self.valid_from and check_time < self.valid_from:
            return False
        if self.valid_until and check_time > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "concept_type": self.concept_type.name,
            "description": self.description,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "importance": self.importance,
            "evidence_count": self.evidence_count,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Relationship:
    """
    Edge in the knowledge graph
    """
    relationship_id: str
    source_id: str
    target_id: str
    relation_type: RelationType

    # Strength and confidence
    strength: float = 0.5  # 0-1, how strong is this relationship
    confidence: float = 0.5  # 0-1, how confident are we

    # Direction
    is_bidirectional: bool = False

    # Temporal
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Evidence
    evidence: List[str] = field(default_factory=list)
    source_episodes: Set[str] = field(default_factory=set)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def reinforce(self, evidence_source: Optional[str] = None):
        """Reinforce relationship with evidence"""
        self.confidence = min(1.0, self.confidence + 0.05)
        self.strength = min(1.0, self.strength + 0.02)
        if evidence_source:
            self.source_episodes.add(evidence_source)

    def to_dict(self) -> Dict:
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.name,
            "strength": self.strength,
            "confidence": self.confidence,
            "is_bidirectional": self.is_bidirectional,
        }


class KnowledgeGraph:
    """
    Core knowledge graph implementation

    Provides:
    - Concept and relationship storage
    - Graph traversal
    - Pattern matching
    - Inference support
    """

    def __init__(self):
        self._concepts: Dict[str, Concept] = {}
        self._relationships: Dict[str, Relationship] = {}

        # Indices for fast lookup
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)  # concept_id -> relationship_ids
        self._incoming: Dict[str, Set[str]] = defaultdict(set)  # concept_id -> relationship_ids
        self._by_type: Dict[ConceptType, Set[str]] = defaultdict(set)
        self._by_name: Dict[str, str] = {}  # name -> concept_id (for quick lookup)

        self._lock = threading.RLock()

    def add_concept(self, concept: Concept) -> str:
        """Add or update concept"""
        with self._lock:
            existing = self._by_name.get(concept.name.lower())
            if existing:
                # Reinforce existing concept
                self._concepts[existing].reinforce()
                return existing

            self._concepts[concept.concept_id] = concept
            self._by_type[concept.concept_type].add(concept.concept_id)
            self._by_name[concept.name.lower()] = concept.concept_id

            return concept.concept_id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add or reinforce relationship"""
        with self._lock:
            # Check for existing relationship
            for rel_id in self._outgoing.get(relationship.source_id, set()):
                rel = self._relationships.get(rel_id)
                if (rel and rel.target_id == relationship.target_id and
                    rel.relation_type == relationship.relation_type):
                    # Reinforce existing
                    rel.reinforce()
                    return rel_id

            self._relationships[relationship.relationship_id] = relationship
            self._outgoing[relationship.source_id].add(relationship.relationship_id)
            self._incoming[relationship.target_id].add(relationship.relationship_id)

            if relationship.is_bidirectional:
                self._outgoing[relationship.target_id].add(relationship.relationship_id)
                self._incoming[relationship.source_id].add(relationship.relationship_id)

            return relationship.relationship_id

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID"""
        with self._lock:
            concept = self._concepts.get(concept_id)
            if concept:
                concept.touch()
            return concept

    def find_concept(self, name: str) -> Optional[Concept]:
        """Find concept by name"""
        with self._lock:
            concept_id = self._by_name.get(name.lower())
            if concept_id:
                return self.get_concept(concept_id)
            return None

    def get_neighbors(self, concept_id: str,
                      relation_types: Optional[Set[RelationType]] = None,
                      direction: str = "both") -> List[Tuple[Concept, Relationship]]:
        """
        Get neighboring concepts

        Args:
            concept_id: Source concept
            relation_types: Filter by relation types
            direction: "outgoing", "incoming", or "both"
        """
        with self._lock:
            neighbors = []

            rel_ids = set()
            if direction in ("outgoing", "both"):
                rel_ids |= self._outgoing.get(concept_id, set())
            if direction in ("incoming", "both"):
                rel_ids |= self._incoming.get(concept_id, set())

            for rel_id in rel_ids:
                rel = self._relationships.get(rel_id)
                if not rel:
                    continue

                if relation_types and rel.relation_type not in relation_types:
                    continue

                # Get the other concept
                other_id = rel.target_id if rel.source_id == concept_id else rel.source_id
                other = self._concepts.get(other_id)

                if other:
                    neighbors.append((other, rel))

            return neighbors

    def traverse(self, start_id: str, max_depth: int = 3,
                 relation_types: Optional[Set[RelationType]] = None) -> Dict[str, Concept]:
        """
        Traverse graph from starting concept

        Returns:
            Dict of concept_id -> Concept for all reachable concepts
        """
        with self._lock:
            visited = {}
            queue = [(start_id, 0)]

            while queue:
                current_id, depth = queue.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                concept = self._concepts.get(current_id)
                if not concept:
                    continue

                visited[current_id] = concept

                # Add neighbors to queue
                for neighbor, rel in self.get_neighbors(current_id, relation_types):
                    if neighbor.concept_id not in visited:
                        queue.append((neighbor.concept_id, depth + 1))

            return visited

    def find_path(self, start_id: str, end_id: str,
                  max_depth: int = 5) -> Optional[List[Tuple[Concept, Relationship]]]:
        """
        Find path between two concepts
        """
        with self._lock:
            if start_id == end_id:
                return []

            visited = {start_id}
            queue = [(start_id, [])]

            while queue:
                current_id, path = queue.pop(0)

                if len(path) >= max_depth:
                    continue

                for neighbor, rel in self.get_neighbors(current_id, direction="outgoing"):
                    if neighbor.concept_id == end_id:
                        return path + [(neighbor, rel)]

                    if neighbor.concept_id not in visited:
                        visited.add(neighbor.concept_id)
                        queue.append((neighbor.concept_id, path + [(neighbor, rel)]))

            return None

    def query_by_type(self, concept_type: ConceptType) -> List[Concept]:
        """Get all concepts of a type"""
        with self._lock:
            return [
                self._concepts[cid]
                for cid in self._by_type.get(concept_type, set())
                if cid in self._concepts
            ]

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        with self._lock:
            return {
                "concept_count": len(self._concepts),
                "relationship_count": len(self._relationships),
                "type_distribution": {
                    t.name: len(ids)
                    for t, ids in self._by_type.items()
                },
            }


class SemanticMemory:
    """
    L3 Semantic Memory - Long-term knowledge storage

    Features:
    - Knowledge graph with concepts and relationships
    - Confidence-based fact storage
    - Forgetting curves for memory management
    - Query and inference capabilities

    Usage:
        sm = SemanticMemory()

        # Add knowledge
        sm.add_fact("APT29", "IS_A", "Threat Actor", confidence=0.9)
        sm.add_fact("APT29", "USES", "Cobalt Strike", confidence=0.8)

        # Query
        facts = sm.query("APT29")

        # Get related concepts
        related = sm.get_related("APT29", depth=2)
    """

    def __init__(self, decay_rate: float = 0.001,
                 min_retention: float = 0.1,
                 forgetting_enabled: bool = True):
        """
        Initialize semantic memory

        Args:
            decay_rate: Rate of forgetting (per day)
            min_retention: Minimum retention before forgetting
            forgetting_enabled: Enable forgetting mechanism
        """
        self.graph = KnowledgeGraph()
        self.decay_rate = decay_rate
        self.min_retention = min_retention
        self.forgetting_enabled = forgetting_enabled

        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "facts_added": 0,
            "queries": 0,
            "forgotten": 0,
        }

        logger.info("SemanticMemory initialized")

    def add_concept(self, name: str, concept_type: ConceptType,
                   description: Optional[str] = None,
                   attributes: Optional[Dict] = None,
                   confidence: float = 0.5,
                   importance: float = 0.5,
                   source_episode: Optional[str] = None) -> Concept:
        """
        Add a concept to memory

        Args:
            name: Concept name
            concept_type: Type of concept
            description: Description
            attributes: Attributes dict
            confidence: Confidence level
            importance: Importance level
            source_episode: Source episode ID

        Returns:
            Created/updated Concept
        """
        with self._lock:
            # Generate ID
            concept_id = hashlib.sha256(
                f"{name}:{concept_type.name}".encode()
            ).hexdigest()[:16]

            # Check if exists
            existing = self.graph.find_concept(name)
            if existing:
                existing.reinforce(source_episode)
                return existing

            concept = Concept(
                concept_id=concept_id,
                name=name,
                concept_type=concept_type,
                description=description,
                attributes=attributes or {},
                confidence=confidence,
                importance=importance,
            )

            if source_episode:
                concept.source_episodes.add(source_episode)

            self.graph.add_concept(concept)
            self.stats["facts_added"] += 1

            return concept

    def add_fact(self, subject: str, predicate: str, obj: str,
                confidence: float = 0.5,
                strength: float = 0.5,
                source_episode: Optional[str] = None,
                metadata: Optional[Dict] = None) -> Tuple[Concept, Relationship, Concept]:
        """
        Add a fact (triple) to memory

        Args:
            subject: Subject name
            predicate: Relationship type
            obj: Object name
            confidence: Confidence in this fact
            strength: Relationship strength
            source_episode: Source episode ID
            metadata: Additional metadata

        Returns:
            Tuple of (subject_concept, relationship, object_concept)
        """
        with self._lock:
            # Parse predicate to relation type
            try:
                relation_type = RelationType[predicate.upper().replace(" ", "_")]
            except KeyError:
                relation_type = RelationType.RELATED_TO

            # Create or get concepts
            subject_concept = self.add_concept(
                subject, ConceptType.ENTITY,
                source_episode=source_episode
            )
            object_concept = self.add_concept(
                obj, ConceptType.ENTITY,
                source_episode=source_episode
            )

            # Create relationship
            rel_id = hashlib.sha256(
                f"{subject_concept.concept_id}:{predicate}:{object_concept.concept_id}".encode()
            ).hexdigest()[:16]

            relationship = Relationship(
                relationship_id=rel_id,
                source_id=subject_concept.concept_id,
                target_id=object_concept.concept_id,
                relation_type=relation_type,
                strength=strength,
                confidence=confidence,
                metadata=metadata or {},
            )

            if source_episode:
                relationship.source_episodes.add(source_episode)

            self.graph.add_relationship(relationship)
            self.stats["facts_added"] += 1

            return subject_concept, relationship, object_concept

    def query(self, name: str) -> Optional[Dict]:
        """
        Query knowledge about a concept

        Args:
            name: Concept name to query

        Returns:
            Dict with concept info and relationships
        """
        with self._lock:
            self.stats["queries"] += 1

            concept = self.graph.find_concept(name)
            if not concept:
                return None

            concept.touch()

            # Get all relationships
            neighbors = self.graph.get_neighbors(concept.concept_id)

            relationships = []
            for neighbor, rel in neighbors:
                relationships.append({
                    "relation": rel.relation_type.name,
                    "target": neighbor.name,
                    "strength": rel.strength,
                    "confidence": rel.confidence,
                })

            return {
                "concept": concept.to_dict(),
                "relationships": relationships,
                "effective_importance": concept.get_effective_importance(),
            }

    def get_related(self, name: str, depth: int = 2,
                   min_confidence: float = 0.3) -> Dict[str, Concept]:
        """
        Get related concepts

        Args:
            name: Starting concept name
            depth: How many hops to traverse
            min_confidence: Minimum confidence filter

        Returns:
            Dict of concept_id -> Concept
        """
        with self._lock:
            concept = self.graph.find_concept(name)
            if not concept:
                return {}

            related = self.graph.traverse(concept.concept_id, max_depth=depth)

            # Filter by confidence
            return {
                cid: c for cid, c in related.items()
                if c.confidence >= min_confidence
            }

    def find_path(self, from_name: str, to_name: str) -> Optional[List[str]]:
        """
        Find relationship path between two concepts

        Returns:
            List of relationship descriptions forming path
        """
        with self._lock:
            from_concept = self.graph.find_concept(from_name)
            to_concept = self.graph.find_concept(to_name)

            if not from_concept or not to_concept:
                return None

            path = self.graph.find_path(from_concept.concept_id, to_concept.concept_id)

            if path is None:
                return None

            return [
                f"--[{rel.relation_type.name}]--> {concept.name}"
                for concept, rel in path
            ]

    def infer(self, query_pattern: Dict) -> List[Dict]:
        """
        Simple inference based on patterns

        Args:
            query_pattern: Pattern to match
                e.g., {"type": "THREAT", "relation": "TARGETS"}

        Returns:
            List of matching facts
        """
        with self._lock:
            results = []

            # Get concepts of specified type
            concept_type = None
            if "type" in query_pattern:
                try:
                    concept_type = ConceptType[query_pattern["type"]]
                except KeyError:
                    pass

            if concept_type:
                concepts = self.graph.query_by_type(concept_type)
            else:
                concepts = list(self.graph._concepts.values())

            # Filter by relation if specified
            if "relation" in query_pattern:
                try:
                    rel_type = RelationType[query_pattern["relation"]]
                    for concept in concepts:
                        neighbors = self.graph.get_neighbors(
                            concept.concept_id,
                            relation_types={rel_type}
                        )
                        for neighbor, rel in neighbors:
                            results.append({
                                "subject": concept.name,
                                "relation": rel.relation_type.name,
                                "object": neighbor.name,
                                "confidence": rel.confidence,
                            })
                except KeyError:
                    pass
            else:
                results = [c.to_dict() for c in concepts]

            return results

    def apply_forgetting(self):
        """Apply forgetting curve to all concepts"""
        if not self.forgetting_enabled:
            return

        with self._lock:
            to_forget = []

            for concept in self.graph._concepts.values():
                concept.decay(self.decay_rate)

                if concept.retention_strength < self.min_retention:
                    to_forget.append(concept.concept_id)

            # Remove forgotten concepts
            for concept_id in to_forget:
                del self.graph._concepts[concept_id]
                self.stats["forgotten"] += 1

            if to_forget:
                logger.info(f"Forgot {len(to_forget)} concepts due to decay")

    def get_consolidation_candidates(self, min_importance: float = 0.7,
                                    min_evidence: int = 2) -> List[Concept]:
        """
        Get concepts that are well-established and important
        These are candidates for being considered "core knowledge"
        """
        with self._lock:
            candidates = []

            for concept in self.graph._concepts.values():
                if (concept.get_effective_importance() >= min_importance and
                    concept.evidence_count >= min_evidence):
                    candidates.append(concept)

            return candidates

    def export_for_sync(self) -> Dict:
        """Export knowledge graph for synchronization"""
        with self._lock:
            return {
                "concepts": {
                    cid: c.to_dict()
                    for cid, c in self.graph._concepts.items()
                },
                "relationships": {
                    rid: r.to_dict()
                    for rid, r in self.graph._relationships.items()
                },
                "stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        with self._lock:
            graph_stats = self.graph.get_stats()
            return {
                **self.stats,
                **graph_stats,
            }


if __name__ == "__main__":
    print("Semantic Memory Self-Test")
    print("=" * 50)

    sm = SemanticMemory()

    print(f"\n[1] Add Facts")
    sm.add_fact("APT29", "IS_A", "Threat Actor", confidence=0.95)
    sm.add_fact("APT29", "USES", "Cobalt Strike", confidence=0.85)
    sm.add_fact("APT29", "TARGETS", "Government", confidence=0.90)
    sm.add_fact("Cobalt Strike", "IS_A", "Malware", confidence=0.99)
    sm.add_fact("Cobalt Strike", "USES", "Beacon Protocol", confidence=0.80)
    print(f"    Added 5 facts")

    print(f"\n[2] Query Concept")
    result = sm.query("APT29")
    if result:
        print(f"    Name: {result['concept']['name']}")
        print(f"    Type: {result['concept']['concept_type']}")
        print(f"    Relationships: {len(result['relationships'])}")
        for rel in result['relationships']:
            print(f"      - {rel['relation']} -> {rel['target']}")

    print(f"\n[3] Get Related")
    related = sm.get_related("APT29", depth=2)
    print(f"    Found {len(related)} related concepts:")
    for cid, concept in related.items():
        print(f"      - {concept.name} ({concept.concept_type.name})")

    print(f"\n[4] Find Path")
    path = sm.find_path("APT29", "Beacon Protocol")
    if path:
        print(f"    Path: APT29 {' '.join(path)}")

    print(f"\n[5] Inference Query")
    results = sm.infer({"type": "ENTITY", "relation": "USES"})
    print(f"    Found {len(results)} results:")
    for r in results[:3]:
        print(f"      - {r['subject']} {r['relation']} {r['object']}")

    print(f"\n[6] Statistics")
    stats = sm.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Semantic Memory test complete")

