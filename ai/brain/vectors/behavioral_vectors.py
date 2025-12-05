#!/usr/bin/env python3
"""
Behavioral Vector Engine for DSMIL Brain

Embeddings for actions, decisions, and outcomes:
- Action pattern representation
- Decision sequence embedding
- Outcome prediction vectors
"""

import hashlib
import threading
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions"""
    QUERY = auto()
    ACCESS = auto()
    MODIFY = auto()
    CREATE = auto()
    DELETE = auto()
    COMMUNICATE = auto()
    AUTHENTICATE = auto()
    EXFILTRATE = auto()


@dataclass
class ActionPattern:
    """A pattern of actions"""
    pattern_id: str
    actions: List[ActionType]

    # Frequency
    occurrence_count: int = 0

    # Assessment
    suspicion_score: float = 0.0

    # Entities
    associated_entities: Set[str] = field(default_factory=set)

    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BehavioralVector:
    """A behavioral embedding"""
    vector_id: str
    vector: List[float]

    # Source
    entity_id: str = ""
    session_id: str = ""

    # Actions represented
    action_sequence: List[ActionType] = field(default_factory=list)

    # Metadata
    outcome: str = ""  # "success", "failure", "suspicious", etc.

    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BehavioralVectorEngine:
    """
    Behavioral Vector Engine

    Creates and manages embeddings for behavioral patterns.

    Usage:
        engine = BehavioralVectorEngine()

        # Record actions
        engine.record_action("entity1", ActionType.ACCESS, "resource1")

        # Get behavioral embedding
        vector = engine.get_entity_embedding("entity1")

        # Find similar behaviors
        similar = engine.find_similar_behaviors(vector)

        # Detect patterns
        patterns = engine.detect_patterns()
    """

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

        self._vectors: Dict[str, BehavioralVector] = {}
        self._entity_actions: Dict[str, List[Tuple[datetime, ActionType, str]]] = defaultdict(list)
        self._patterns: Dict[str, ActionPattern] = {}

        # Action co-occurrence matrix (for embedding)
        self._action_cooccurrence: Dict[Tuple[ActionType, ActionType], int] = defaultdict(int)

        self._lock = threading.RLock()

        logger.info(f"BehavioralVectorEngine initialized (dimensions={dimensions})")

    def record_action(self, entity_id: str, action: ActionType,
                     target: str = "", session_id: str = ""):
        """Record an action by an entity"""
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            self._entity_actions[entity_id].append((timestamp, action, target))

            # Update co-occurrence
            actions = self._entity_actions[entity_id]
            if len(actions) >= 2:
                prev_action = actions[-2][1]
                self._action_cooccurrence[(prev_action, action)] += 1

    def get_entity_embedding(self, entity_id: str,
                            window_size: int = 100) -> Optional[BehavioralVector]:
        """Get behavioral embedding for entity"""
        with self._lock:
            actions = self._entity_actions.get(entity_id, [])
            if not actions:
                return None

            # Get recent actions
            recent = actions[-window_size:]
            action_sequence = [a[1] for a in recent]

            # Build embedding
            vector = self._embed_action_sequence(action_sequence)

            bv = BehavioralVector(
                vector_id=hashlib.sha256(f"bv:{entity_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                vector=vector,
                entity_id=entity_id,
                action_sequence=action_sequence,
            )

            self._vectors[bv.vector_id] = bv
            return bv

    def _embed_action_sequence(self, actions: List[ActionType]) -> List[float]:
        """Create embedding from action sequence"""
        # Action frequency features
        action_counts = defaultdict(int)
        for a in actions:
            action_counts[a] += 1

        total = len(actions) if actions else 1

        # Normalized frequency per action type
        freq_features = [
            action_counts.get(at, 0) / total
            for at in ActionType
        ]

        # Transition features (bigrams)
        transition_counts = defaultdict(int)
        for i in range(len(actions) - 1):
            transition_counts[(actions[i], actions[i+1])] += 1

        transition_features = []
        for a1 in ActionType:
            for a2 in ActionType:
                count = transition_counts.get((a1, a2), 0)
                transition_features.append(count / max(len(actions) - 1, 1))

        # Temporal features
        if actions:
            # Burstiness (how clustered are same actions)
            burst_score = 0
            for i in range(len(actions) - 1):
                if actions[i] == actions[i+1]:
                    burst_score += 1
            burst_score = burst_score / max(len(actions) - 1, 1)
        else:
            burst_score = 0

        # Diversity (unique actions / total)
        diversity = len(set(actions)) / len(ActionType)

        # Suspicion indicators
        suspicion_actions = {ActionType.EXFILTRATE, ActionType.DELETE}
        suspicion_score = sum(1 for a in actions if a in suspicion_actions) / total

        # Combine features
        features = freq_features + transition_features + [burst_score, diversity, suspicion_score]

        # Pad/truncate to dimensions
        if len(features) < self.dimensions:
            features = features + [0.0] * (self.dimensions - len(features))
        else:
            features = features[:self.dimensions]

        return features

    def find_similar_behaviors(self, query_vector: BehavioralVector,
                              k: int = 10) -> List[Tuple[str, float]]:
        """Find entities with similar behavior"""
        with self._lock:
            results = []

            for vid, bv in self._vectors.items():
                if vid == query_vector.vector_id:
                    continue

                similarity = self._cosine_similarity(query_vector.vector, bv.vector)
                results.append((bv.entity_id, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def detect_patterns(self, min_occurrences: int = 3,
                       pattern_length: int = 3) -> List[ActionPattern]:
        """Detect recurring action patterns"""
        with self._lock:
            pattern_counts = defaultdict(lambda: {"count": 0, "entities": set()})

            for entity_id, actions in self._entity_actions.items():
                action_types = [a[1] for a in actions]

                # Find all subsequences of pattern_length
                for i in range(len(action_types) - pattern_length + 1):
                    pattern = tuple(action_types[i:i + pattern_length])
                    pattern_counts[pattern]["count"] += 1
                    pattern_counts[pattern]["entities"].add(entity_id)

            # Filter by min occurrences
            patterns = []
            for pattern, data in pattern_counts.items():
                if data["count"] >= min_occurrences:
                    # Calculate suspicion score
                    suspicious_actions = {ActionType.EXFILTRATE, ActionType.DELETE}
                    suspicion = sum(1 for a in pattern if a in suspicious_actions) / len(pattern)

                    ap = ActionPattern(
                        pattern_id=hashlib.sha256(str(pattern).encode()).hexdigest()[:16],
                        actions=list(pattern),
                        occurrence_count=data["count"],
                        suspicion_score=suspicion,
                        associated_entities=data["entities"],
                    )
                    patterns.append(ap)
                    self._patterns[ap.pattern_id] = ap

            return sorted(patterns, key=lambda p: p.occurrence_count, reverse=True)

    def get_suspicious_entities(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Get entities with suspicious behavior patterns"""
        with self._lock:
            entity_scores = defaultdict(float)

            for pattern in self._patterns.values():
                if pattern.suspicion_score >= threshold:
                    for entity in pattern.associated_entities:
                        entity_scores[entity] = max(
                            entity_scores[entity],
                            pattern.suspicion_score
                        )

            return sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        with self._lock:
            total_actions = sum(len(a) for a in self._entity_actions.values())
            return {
                "entities_tracked": len(self._entity_actions),
                "total_actions": total_actions,
                "behavioral_vectors": len(self._vectors),
                "patterns_detected": len(self._patterns),
            }


if __name__ == "__main__":
    print("Behavioral Vector Engine Self-Test")
    print("=" * 50)

    import random

    engine = BehavioralVectorEngine(dimensions=128)

    print("\n[1] Record Actions")
    # Normal user
    normal_actions = [ActionType.AUTHENTICATE, ActionType.QUERY, ActionType.ACCESS, ActionType.QUERY]
    for action in normal_actions * 5:
        engine.record_action("user-normal", action)

    # Suspicious user
    suspicious_actions = [ActionType.AUTHENTICATE, ActionType.ACCESS, ActionType.EXFILTRATE]
    for action in suspicious_actions * 3:
        engine.record_action("user-suspicious", action)

    # Another normal user
    for _ in range(20):
        action = random.choice([ActionType.QUERY, ActionType.ACCESS, ActionType.MODIFY])
        engine.record_action("user-normal-2", action)

    print("    Recorded actions for 3 entities")

    print("\n[2] Get Behavioral Embeddings")
    emb_normal = engine.get_entity_embedding("user-normal")
    emb_suspicious = engine.get_entity_embedding("user-suspicious")

    print(f"    Normal embedding: first 5 dims = {[f'{v:.3f}' for v in emb_normal.vector[:5]]}")
    print(f"    Suspicious embedding: first 5 dims = {[f'{v:.3f}' for v in emb_suspicious.vector[:5]]}")

    print("\n[3] Find Similar Behaviors")
    similar = engine.find_similar_behaviors(emb_normal, k=3)
    print(f"    Similar to 'user-normal':")
    for entity, sim in similar:
        print(f"      {entity}: {sim:.3f}")

    print("\n[4] Detect Patterns")
    patterns = engine.detect_patterns(min_occurrences=2, pattern_length=3)
    print(f"    Found {len(patterns)} patterns")
    for p in patterns[:3]:
        actions = [a.name for a in p.actions]
        print(f"      {actions}: count={p.occurrence_count}, suspicion={p.suspicion_score:.2f}")

    print("\n[5] Suspicious Entities")
    suspicious = engine.get_suspicious_entities(threshold=0.3)
    print(f"    Suspicious entities:")
    for entity, score in suspicious:
        print(f"      {entity}: {score:.2f}")

    print("\n[6] Statistics")
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Behavioral Vector Engine test complete")

