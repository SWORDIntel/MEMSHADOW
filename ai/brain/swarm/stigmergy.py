#!/usr/bin/env python3
"""
Stigmergic Knowledge Building for DSMIL Brain

Indirect coordination through environmental markers:
- Nodes leave "pheromone" markers on knowledge
- Popular paths strengthen
- Unused paths decay
- Self-organizing knowledge highways
"""

import hashlib
import threading
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PheromoneMarker:
    """A pheromone marker on knowledge"""
    marker_id: str
    knowledge_id: str

    # Pheromone level
    strength: float = 1.0

    # Source
    deposited_by: str = ""

    # Timing
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class KnowledgePath:
    """A path between knowledge nodes"""
    path_id: str
    source: str
    target: str

    # Pheromone
    pheromone_level: float = 0.0

    # Usage
    traversal_count: int = 0
    last_traversed: Optional[datetime] = None


@dataclass
class KnowledgeHighway:
    """A high-traffic knowledge pathway"""
    highway_id: str
    path: List[str]  # Sequence of knowledge IDs

    # Traffic
    total_pheromone: float = 0.0
    average_traffic: float = 0.0

    # Assessment
    importance: float = 0.0


class StigmergicKnowledge:
    """
    Stigmergic Knowledge System

    Enables indirect coordination through knowledge markers.

    Usage:
        stigmergy = StigmergicKnowledge()

        # Access knowledge (deposits pheromone)
        stigmergy.access_knowledge("knowledge_id", "node_id")

        # Traverse path (strengthens path)
        stigmergy.traverse_path("from", "to", "node_id")

        # Find highways (popular paths)
        highways = stigmergy.find_highways()

        # Decay unused paths
        stigmergy.decay_pheromones()
    """

    def __init__(self, decay_rate: float = 0.1,
                deposit_amount: float = 1.0,
                highway_threshold: float = 5.0):
        self.decay_rate = decay_rate
        self.deposit_amount = deposit_amount
        self.highway_threshold = highway_threshold

        self._markers: Dict[str, PheromoneMarker] = {}  # knowledge_id -> marker
        self._paths: Dict[Tuple[str, str], KnowledgePath] = {}  # (from, to) -> path
        self._highways: Dict[str, KnowledgeHighway] = {}

        self._access_history: Dict[str, List[datetime]] = defaultdict(list)

        self._lock = threading.RLock()

        logger.info("StigmergicKnowledge initialized")

    def access_knowledge(self, knowledge_id: str, node_id: str) -> float:
        """
        Node accesses knowledge, depositing pheromone

        Returns current pheromone level
        """
        with self._lock:
            if knowledge_id not in self._markers:
                self._markers[knowledge_id] = PheromoneMarker(
                    marker_id=hashlib.sha256(f"marker:{knowledge_id}".encode()).hexdigest()[:16],
                    knowledge_id=knowledge_id,
                    strength=self.deposit_amount,
                    deposited_by=node_id,
                )
            else:
                marker = self._markers[knowledge_id]
                marker.strength += self.deposit_amount
                marker.last_reinforced = datetime.now(timezone.utc)

            self._access_history[knowledge_id].append(datetime.now(timezone.utc))

            return self._markers[knowledge_id].strength

    def traverse_path(self, from_id: str, to_id: str, node_id: str) -> float:
        """
        Traverse a path between knowledge, strengthening it

        Returns path pheromone level
        """
        with self._lock:
            path_key = (from_id, to_id)

            if path_key not in self._paths:
                self._paths[path_key] = KnowledgePath(
                    path_id=hashlib.sha256(f"path:{from_id}:{to_id}".encode()).hexdigest()[:16],
                    source=from_id,
                    target=to_id,
                    pheromone_level=self.deposit_amount,
                    traversal_count=1,
                    last_traversed=datetime.now(timezone.utc),
                )
            else:
                path = self._paths[path_key]
                path.pheromone_level += self.deposit_amount
                path.traversal_count += 1
                path.last_traversed = datetime.now(timezone.utc)

            # Also mark the knowledge endpoints
            self.access_knowledge(from_id, node_id)
            self.access_knowledge(to_id, node_id)

            return self._paths[path_key].pheromone_level

    def decay_pheromones(self, time_delta: Optional[timedelta] = None):
        """
        Decay all pheromones (simulates evaporation)
        """
        with self._lock:
            decay = 1 - self.decay_rate

            # Decay markers
            for marker in list(self._markers.values()):
                marker.strength *= decay
                if marker.strength < 0.01:
                    del self._markers[marker.knowledge_id]

            # Decay paths
            for path_key, path in list(self._paths.items()):
                path.pheromone_level *= decay
                if path.pheromone_level < 0.01:
                    del self._paths[path_key]

    def get_pheromone_level(self, knowledge_id: str) -> float:
        """Get current pheromone level for knowledge"""
        with self._lock:
            marker = self._markers.get(knowledge_id)
            return marker.strength if marker else 0.0

    def get_path_strength(self, from_id: str, to_id: str) -> float:
        """Get pheromone level for a path"""
        with self._lock:
            path = self._paths.get((from_id, to_id))
            return path.pheromone_level if path else 0.0

    def get_strongest_paths_from(self, knowledge_id: str,
                                limit: int = 5) -> List[KnowledgePath]:
        """Get strongest paths from a knowledge node"""
        with self._lock:
            paths = [
                path for (from_id, to_id), path in self._paths.items()
                if from_id == knowledge_id
            ]
            return sorted(paths, key=lambda p: p.pheromone_level, reverse=True)[:limit]

    def find_highways(self, min_pheromone: Optional[float] = None) -> List[KnowledgeHighway]:
        """
        Find high-traffic knowledge pathways
        """
        with self._lock:
            threshold = min_pheromone or self.highway_threshold

            # Find paths above threshold
            strong_paths = [
                path for path in self._paths.values()
                if path.pheromone_level >= threshold
            ]

            # Build connected highways
            highways = []

            # Simple approach: each strong path is a mini-highway
            for path in strong_paths:
                highway = KnowledgeHighway(
                    highway_id=hashlib.sha256(f"hwy:{path.source}:{path.target}".encode()).hexdigest()[:16],
                    path=[path.source, path.target],
                    total_pheromone=path.pheromone_level,
                    average_traffic=path.traversal_count,
                    importance=path.pheromone_level / threshold,
                )
                highways.append(highway)
                self._highways[highway.highway_id] = highway

            return sorted(highways, key=lambda h: h.importance, reverse=True)

    def get_hot_knowledge(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get most accessed knowledge"""
        with self._lock:
            markers = sorted(
                self._markers.values(),
                key=lambda m: m.strength,
                reverse=True
            )
            return [(m.knowledge_id, m.strength) for m in markers[:limit]]

    def get_cold_knowledge(self, threshold: float = 0.5) -> List[str]:
        """Get rarely accessed knowledge (candidates for pruning)"""
        with self._lock:
            return [
                m.knowledge_id for m in self._markers.values()
                if m.strength < threshold
            ]

    def suggest_next(self, current_knowledge: str) -> Optional[str]:
        """
        Suggest next knowledge to access based on pheromone trails
        """
        with self._lock:
            paths = self.get_strongest_paths_from(current_knowledge, limit=1)
            if paths:
                return paths[0].target
            return None

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            return {
                "active_markers": len(self._markers),
                "active_paths": len(self._paths),
                "highways": len(self._highways),
                "total_pheromone": sum(m.strength for m in self._markers.values()),
                "avg_path_strength": sum(p.pheromone_level for p in self._paths.values()) / max(len(self._paths), 1),
            }


if __name__ == "__main__":
    print("Stigmergic Knowledge Self-Test")
    print("=" * 50)

    stigmergy = StigmergicKnowledge(decay_rate=0.1, highway_threshold=3.0)

    print("\n[1] Access Knowledge")
    knowledge_ids = ["threat_intel", "actor_profiles", "indicators", "ttps", "mitigations"]

    # Simulate access patterns
    import random
    for _ in range(20):
        kid = random.choice(knowledge_ids[:3])  # First 3 accessed more
        stigmergy.access_knowledge(kid, f"node-{random.randint(1,3)}")

    print("    Simulated 20 knowledge accesses")

    print("\n[2] Hot Knowledge")
    hot = stigmergy.get_hot_knowledge(limit=5)
    for kid, strength in hot:
        print(f"    {kid}: {strength:.2f}")

    print("\n[3] Traverse Paths")
    # Create common paths
    for _ in range(10):
        stigmergy.traverse_path("threat_intel", "actor_profiles", "node-1")
    for _ in range(8):
        stigmergy.traverse_path("actor_profiles", "ttps", "node-2")
    for _ in range(3):
        stigmergy.traverse_path("indicators", "mitigations", "node-3")
    print("    Created path traversals")

    print("\n[4] Find Highways")
    highways = stigmergy.find_highways()
    print(f"    Found {len(highways)} highways")
    for hwy in highways:
        print(f"      {hwy.path[0]} -> {hwy.path[1]}: importance={hwy.importance:.2f}")

    print("\n[5] Navigation Suggestion")
    suggestion = stigmergy.suggest_next("threat_intel")
    print(f"    From 'threat_intel', suggest: {suggestion}")

    print("\n[6] Decay Pheromones")
    stigmergy.decay_pheromones()
    print("    Applied decay")
    hot_after = stigmergy.get_hot_knowledge(limit=3)
    for kid, strength in hot_after:
        print(f"    {kid}: {strength:.2f}")

    print("\n[7] Cold Knowledge")
    cold = stigmergy.get_cold_knowledge(threshold=1.0)
    print(f"    Cold knowledge (< 1.0): {cold}")

    print("\n[8] Statistics")
    stats = stigmergy.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Stigmergic Knowledge test complete")

