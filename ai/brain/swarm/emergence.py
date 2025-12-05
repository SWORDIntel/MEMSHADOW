#!/usr/bin/env python3
"""
Emergent Intelligence for DSMIL Brain

Collective intelligence that emerges from distributed nodes:
- No single node has complete picture
- Collective reasoning > individual capability
- Patterns only visible at network scale
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Callable
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class NodeContribution:
    """A single node's contribution to collective reasoning"""
    node_id: str
    fragment: Any  # Partial knowledge/reasoning
    confidence: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkPattern:
    """A pattern visible only at network scale"""
    pattern_id: str
    pattern_type: str

    # Evidence
    contributing_nodes: Set[str] = field(default_factory=set)
    fragments: List[NodeContribution] = field(default_factory=list)

    # Assessment
    confidence: float = 0.0
    significance: float = 0.0

    discovered: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CollectiveReasoning:
    """Result of collective reasoning process"""
    reasoning_id: str
    query: str

    # Contributions
    node_contributions: List[NodeContribution] = field(default_factory=list)
    patterns_discovered: List[NetworkPattern] = field(default_factory=list)

    # Synthesis
    collective_answer: Any = None
    collective_confidence: float = 0.0

    # Comparison
    best_individual_answer: Any = None
    best_individual_confidence: float = 0.0
    emergence_factor: float = 0.0  # How much better collective vs individual


class EmergentIntelligence:
    """
    Emergent Intelligence System

    Enables collective reasoning where the network knows more
    than any individual node.

    Usage:
        emergence = EmergentIntelligence()

        # Nodes contribute fragments
        emergence.contribute("node-1", fragment, confidence=0.6)

        # Collective reasoning
        result = emergence.reason_collectively("What is the threat?")

        # Discover network-scale patterns
        patterns = emergence.discover_patterns()
    """

    def __init__(self, min_nodes_for_emergence: int = 3):
        self.min_nodes_for_emergence = min_nodes_for_emergence

        self._contributions: Dict[str, List[NodeContribution]] = defaultdict(list)
        self._patterns: Dict[str, NetworkPattern] = {}
        self._reasonings: Dict[str, CollectiveReasoning] = {}

        self._lock = threading.RLock()

        logger.info("EmergentIntelligence initialized")

    def contribute(self, node_id: str, fragment: Any,
                  confidence: float = 0.5,
                  topic: str = "general",
                  context: Optional[Dict] = None):
        """
        Node contributes a fragment of knowledge/reasoning
        """
        with self._lock:
            contribution = NodeContribution(
                node_id=node_id,
                fragment=fragment,
                confidence=confidence,
                context=context or {},
            )
            self._contributions[topic].append(contribution)

    def reason_collectively(self, query: str,
                           topic: str = "general") -> CollectiveReasoning:
        """
        Perform collective reasoning across all contributions
        """
        with self._lock:
            contributions = self._contributions.get(topic, [])

            if len(contributions) < self.min_nodes_for_emergence:
                # Not enough nodes for emergence
                return CollectiveReasoning(
                    reasoning_id=hashlib.sha256(query.encode()).hexdigest()[:16],
                    query=query,
                    node_contributions=contributions,
                    collective_answer="Insufficient nodes for collective reasoning",
                )

            # Find best individual answer
            best_individual = max(contributions, key=lambda c: c.confidence)

            # Synthesize collective answer
            collective_answer, collective_confidence = self._synthesize(contributions)

            # Discover patterns
            patterns = self._find_patterns(contributions)

            # Calculate emergence factor
            emergence_factor = (collective_confidence - best_individual.confidence) / max(best_individual.confidence, 0.01)

            reasoning = CollectiveReasoning(
                reasoning_id=hashlib.sha256(f"{query}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                query=query,
                node_contributions=contributions,
                patterns_discovered=patterns,
                collective_answer=collective_answer,
                collective_confidence=collective_confidence,
                best_individual_answer=best_individual.fragment,
                best_individual_confidence=best_individual.confidence,
                emergence_factor=emergence_factor,
            )

            self._reasonings[reasoning.reasoning_id] = reasoning
            return reasoning

    def _synthesize(self, contributions: List[NodeContribution]) -> tuple:
        """Synthesize collective answer from contributions"""
        if not contributions:
            return None, 0.0

        # Weighted aggregation
        total_weight = sum(c.confidence for c in contributions)
        if total_weight == 0:
            return None, 0.0

        # For numeric data
        numeric_fragments = [c.fragment for c in contributions if isinstance(c.fragment, (int, float))]
        if numeric_fragments:
            weighted_sum = sum(
                c.fragment * c.confidence
                for c in contributions
                if isinstance(c.fragment, (int, float))
            )
            answer = weighted_sum / total_weight

            # Confidence increases with agreement
            variance = sum((f - answer) ** 2 for f in numeric_fragments) / len(numeric_fragments)
            agreement = max(0, 1 - variance / max(abs(answer), 1))
            confidence = (sum(c.confidence for c in contributions) / len(contributions)) * (0.5 + 0.5 * agreement)

            return answer, min(confidence, 1.0)

        # For categorical/text data
        fragment_counts = defaultdict(float)
        for c in contributions:
            fragment_str = str(c.fragment)
            fragment_counts[fragment_str] += c.confidence

        if fragment_counts:
            best = max(fragment_counts.items(), key=lambda x: x[1])
            answer = best[0]
            # Confidence based on agreement
            confidence = best[1] / total_weight
            return answer, confidence

        return None, 0.0

    def _find_patterns(self, contributions: List[NodeContribution]) -> List[NetworkPattern]:
        """Find patterns visible only at network scale"""
        patterns = []

        # Group by context keys
        context_groups = defaultdict(list)
        for c in contributions:
            for key, value in c.context.items():
                context_groups[(key, str(value))].append(c)

        # Patterns where multiple nodes see same context
        for (key, value), nodes in context_groups.items():
            if len(nodes) >= self.min_nodes_for_emergence:
                pattern = NetworkPattern(
                    pattern_id=hashlib.sha256(f"pattern:{key}:{value}".encode()).hexdigest()[:16],
                    pattern_type=f"shared_{key}",
                    contributing_nodes=set(c.node_id for c in nodes),
                    fragments=nodes,
                    confidence=sum(c.confidence for c in nodes) / len(nodes),
                    significance=len(nodes) / len(contributions),
                )
                patterns.append(pattern)
                self._patterns[pattern.pattern_id] = pattern

        return patterns

    def discover_patterns(self, min_significance: float = 0.3) -> List[NetworkPattern]:
        """Discover all significant network-scale patterns"""
        with self._lock:
            return [
                p for p in self._patterns.values()
                if p.significance >= min_significance
            ]

    def get_emergence_stats(self) -> Dict:
        """Get emergence statistics"""
        with self._lock:
            reasonings = list(self._reasonings.values())

            if not reasonings:
                return {
                    "total_reasonings": 0,
                    "avg_emergence_factor": 0.0,
                    "patterns_discovered": len(self._patterns),
                }

            emergence_factors = [r.emergence_factor for r in reasonings]

            return {
                "total_reasonings": len(reasonings),
                "avg_emergence_factor": sum(emergence_factors) / len(emergence_factors),
                "max_emergence_factor": max(emergence_factors),
                "patterns_discovered": len(self._patterns),
                "contributing_nodes": len(set(
                    c.node_id
                    for contribs in self._contributions.values()
                    for c in contribs
                )),
            }


if __name__ == "__main__":
    print("Emergent Intelligence Self-Test")
    print("=" * 50)

    emergence = EmergentIntelligence(min_nodes_for_emergence=2)

    print("\n[1] Contribute Fragments")
    import random

    for i in range(5):
        # Each node has partial view
        base_value = 75  # "True" value they're all approximating
        node_fragment = base_value + random.gauss(0, 10)  # Noisy observation

        emergence.contribute(
            node_id=f"node-{i}",
            fragment=node_fragment,
            confidence=random.uniform(0.5, 0.9),
            topic="threat_level",
            context={"sector": "financial" if i % 2 == 0 else "government"},
        )
    print("    5 nodes contributed fragments")

    print("\n[2] Collective Reasoning")
    result = emergence.reason_collectively("What is the threat level?", topic="threat_level")
    print(f"    Collective answer: {result.collective_answer:.2f}")
    print(f"    Collective confidence: {result.collective_confidence:.2f}")
    print(f"    Best individual: {result.best_individual_answer:.2f}")
    print(f"    Best individual confidence: {result.best_individual_confidence:.2f}")
    print(f"    Emergence factor: {result.emergence_factor:.2%}")

    print("\n[3] Discovered Patterns")
    patterns = result.patterns_discovered
    print(f"    Found {len(patterns)} patterns")
    for p in patterns:
        print(f"      - {p.pattern_type}: {len(p.contributing_nodes)} nodes")

    print("\n[4] Network-Scale Patterns")
    all_patterns = emergence.discover_patterns(min_significance=0.2)
    print(f"    Total patterns: {len(all_patterns)}")

    print("\n[5] Emergence Statistics")
    stats = emergence.get_emergence_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2%}")
        else:
            print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Emergent Intelligence test complete")

