#!/usr/bin/env python3
"""
Causality Chain Engine for DSMIL Brain

Event cause-effect tracking:
- Event A at T1 → Event B at T2 relationships
- Multi-hop causal reasoning
- Retroactive analysis (what caused this?)
- Proactive analysis (what will this cause?)
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CausalEvent:
    """An event in a causal chain"""
    event_id: str
    event_type: str
    description: str

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context
    entities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalLink:
    """A causal relationship between events"""
    link_id: str
    cause_event_id: str
    effect_event_id: str

    # Relationship
    strength: float = 1.0  # How strongly A causes B
    confidence: float = 1.0  # How confident in this link
    lag_seconds: float = 0.0  # Time delay between cause and effect

    # Classification
    link_type: str = "causes"  # "causes", "enables", "prevents", "influences"


@dataclass
class CausalChain:
    """A chain of causal events"""
    chain_id: str
    events: List[str] = field(default_factory=list)  # Event IDs in order
    links: List[str] = field(default_factory=list)  # Link IDs

    # Assessment
    total_strength: float = 0.0
    total_confidence: float = 0.0


class CausalityChainEngine:
    """
    Causality Chain Engine

    Tracks cause-effect relationships between events.

    Usage:
        engine = CausalityChainEngine()

        # Record events
        event_a = engine.record_event("attack", "Phishing email sent")
        event_b = engine.record_event("compromise", "Credentials stolen")

        # Link causally
        engine.link_cause_effect(event_a, event_b, strength=0.9)

        # Analyze
        causes = engine.get_causes(event_b)
        effects = engine.get_effects(event_a)
        chain = engine.trace_chain(event_a, event_b)
    """

    def __init__(self):
        self._events: Dict[str, CausalEvent] = {}
        self._links: Dict[str, CausalLink] = {}

        # Indices for fast lookup
        self._causes: Dict[str, Set[str]] = defaultdict(set)  # effect -> cause links
        self._effects: Dict[str, Set[str]] = defaultdict(set)  # cause -> effect links

        self._lock = threading.RLock()

        logger.info("CausalityChainEngine initialized")

    def record_event(self, event_type: str, description: str,
                    timestamp: Optional[datetime] = None,
                    entities: Optional[Set[str]] = None,
                    metadata: Optional[Dict] = None) -> CausalEvent:
        """Record an event"""
        with self._lock:
            event_id = hashlib.sha256(
                f"{event_type}:{description}:{timestamp or datetime.now()}".encode()
            ).hexdigest()[:16]

            event = CausalEvent(
                event_id=event_id,
                event_type=event_type,
                description=description,
                timestamp=timestamp or datetime.now(timezone.utc),
                entities=entities or set(),
                metadata=metadata or {},
            )

            self._events[event_id] = event
            return event

    def link_cause_effect(self, cause: CausalEvent, effect: CausalEvent,
                         strength: float = 1.0,
                         confidence: float = 1.0,
                         link_type: str = "causes") -> CausalLink:
        """Link two events causally"""
        with self._lock:
            link_id = hashlib.sha256(
                f"{cause.event_id}:{effect.event_id}".encode()
            ).hexdigest()[:16]

            lag = (effect.timestamp - cause.timestamp).total_seconds()

            link = CausalLink(
                link_id=link_id,
                cause_event_id=cause.event_id,
                effect_event_id=effect.event_id,
                strength=strength,
                confidence=confidence,
                lag_seconds=lag,
                link_type=link_type,
            )

            self._links[link_id] = link
            self._causes[effect.event_id].add(link_id)
            self._effects[cause.event_id].add(link_id)

            return link

    def get_causes(self, event: CausalEvent,
                  min_confidence: float = 0.0) -> List[Tuple[CausalEvent, CausalLink]]:
        """Get all causes of an event"""
        with self._lock:
            results = []

            for link_id in self._causes.get(event.event_id, set()):
                link = self._links.get(link_id)
                if link and link.confidence >= min_confidence:
                    cause_event = self._events.get(link.cause_event_id)
                    if cause_event:
                        results.append((cause_event, link))

            return sorted(results, key=lambda x: x[1].strength, reverse=True)

    def get_effects(self, event: CausalEvent,
                   min_confidence: float = 0.0) -> List[Tuple[CausalEvent, CausalLink]]:
        """Get all effects of an event"""
        with self._lock:
            results = []

            for link_id in self._effects.get(event.event_id, set()):
                link = self._links.get(link_id)
                if link and link.confidence >= min_confidence:
                    effect_event = self._events.get(link.effect_event_id)
                    if effect_event:
                        results.append((effect_event, link))

            return sorted(results, key=lambda x: x[1].strength, reverse=True)

    def trace_chain(self, start: CausalEvent, end: CausalEvent,
                   max_hops: int = 10) -> Optional[CausalChain]:
        """
        Trace causal chain from start to end

        Uses BFS to find path.
        """
        with self._lock:
            # BFS to find path
            queue = [(start.event_id, [start.event_id], [])]
            visited = {start.event_id}

            while queue:
                current, path, link_path = queue.pop(0)

                if current == end.event_id:
                    # Found path
                    total_strength = 1.0
                    total_confidence = 1.0
                    for link_id in link_path:
                        link = self._links[link_id]
                        total_strength *= link.strength
                        total_confidence *= link.confidence

                    return CausalChain(
                        chain_id=hashlib.sha256(f"chain:{start.event_id}:{end.event_id}".encode()).hexdigest()[:16],
                        events=path,
                        links=link_path,
                        total_strength=total_strength,
                        total_confidence=total_confidence,
                    )

                if len(path) >= max_hops:
                    continue

                # Explore effects
                for link_id in self._effects.get(current, set()):
                    link = self._links[link_id]
                    next_event = link.effect_event_id

                    if next_event not in visited:
                        visited.add(next_event)
                        queue.append((
                            next_event,
                            path + [next_event],
                            link_path + [link_id]
                        ))

            return None  # No path found

    def get_root_causes(self, event: CausalEvent,
                       max_depth: int = 5) -> List[CausalEvent]:
        """Get all root causes (events with no causes)"""
        with self._lock:
            roots = []
            visited = set()

            def trace_back(current_id: str, depth: int):
                if depth > max_depth or current_id in visited:
                    return
                visited.add(current_id)

                cause_links = self._causes.get(current_id, set())

                if not cause_links:
                    # This is a root cause
                    event = self._events.get(current_id)
                    if event:
                        roots.append(event)
                else:
                    for link_id in cause_links:
                        link = self._links[link_id]
                        trace_back(link.cause_event_id, depth + 1)

            trace_back(event.event_id, 0)
            return roots

    def predict_effects(self, event: CausalEvent,
                       max_depth: int = 3) -> List[Tuple[CausalEvent, float]]:
        """
        Predict downstream effects with probability

        Returns list of (event, probability) tuples.
        """
        with self._lock:
            predictions = []
            visited = set()

            def predict_downstream(current_id: str, probability: float, depth: int):
                if depth > max_depth or current_id in visited or probability < 0.01:
                    return
                visited.add(current_id)

                for link_id in self._effects.get(current_id, set()):
                    link = self._links[link_id]
                    effect_event = self._events.get(link.effect_event_id)

                    if effect_event:
                        effect_prob = probability * link.strength * link.confidence
                        predictions.append((effect_event, effect_prob))
                        predict_downstream(link.effect_event_id, effect_prob, depth + 1)

            predict_downstream(event.event_id, 1.0, 0)

            return sorted(predictions, key=lambda x: x[1], reverse=True)

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        with self._lock:
            # Find events with most causes/effects
            max_causes = max(len(v) for v in self._causes.values()) if self._causes else 0
            max_effects = max(len(v) for v in self._effects.values()) if self._effects else 0

            return {
                "total_events": len(self._events),
                "total_links": len(self._links),
                "max_causes": max_causes,
                "max_effects": max_effects,
            }


if __name__ == "__main__":
    print("Causality Chain Engine Self-Test")
    print("=" * 50)

    from datetime import timedelta

    engine = CausalityChainEngine()

    print("\n[1] Record Events")
    base_time = datetime.now(timezone.utc)

    e1 = engine.record_event("reconnaissance", "Port scan detected",
                            timestamp=base_time)
    e2 = engine.record_event("exploit", "Vulnerability exploited",
                            timestamp=base_time + timedelta(hours=1))
    e3 = engine.record_event("persistence", "Backdoor installed",
                            timestamp=base_time + timedelta(hours=2))
    e4 = engine.record_event("exfiltration", "Data exfiltrated",
                            timestamp=base_time + timedelta(hours=5))

    print("    Recorded 4 attack chain events")

    print("\n[2] Link Causally")
    engine.link_cause_effect(e1, e2, strength=0.7, confidence=0.8)
    engine.link_cause_effect(e2, e3, strength=0.9, confidence=0.9)
    engine.link_cause_effect(e3, e4, strength=0.8, confidence=0.85)
    print("    Created causal links")

    print("\n[3] Get Causes")
    causes = engine.get_causes(e3)
    print(f"    Causes of '{e3.description}':")
    for cause_event, link in causes:
        print(f"      {cause_event.description} (strength={link.strength})")

    print("\n[4] Get Effects")
    effects = engine.get_effects(e2)
    print(f"    Effects of '{e2.description}':")
    for effect_event, link in effects:
        print(f"      {effect_event.description} (strength={link.strength})")

    print("\n[5] Trace Chain")
    chain = engine.trace_chain(e1, e4)
    if chain:
        print(f"    Chain from reconnaissance to exfiltration:")
        print(f"      Events: {len(chain.events)}")
        print(f"      Total strength: {chain.total_strength:.3f}")
        print(f"      Total confidence: {chain.total_confidence:.3f}")

    print("\n[6] Get Root Causes")
    roots = engine.get_root_causes(e4)
    print(f"    Root causes of exfiltration:")
    for root in roots:
        print(f"      {root.description}")

    print("\n[7] Predict Effects")
    predictions = engine.predict_effects(e1)
    print(f"    Predicted effects of reconnaissance:")
    for pred_event, prob in predictions[:3]:
        print(f"      {pred_event.description}: {prob:.3f}")

    print("\n[8] Statistics")
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Causality Chain Engine test complete")

