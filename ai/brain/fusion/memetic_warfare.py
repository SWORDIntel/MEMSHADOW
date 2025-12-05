#!/usr/bin/env python3
"""
Memetic Warfare Detection for DSMIL Brain

Detects and tracks influence operations:
- Narrative tracking and evolution
- Bot/troll network detection
- Amplification pattern analysis
- Source tracing for campaigns
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Narrative:
    """A tracked narrative"""
    narrative_id: str
    content: str
    keywords: Set[str] = field(default_factory=set)

    # Tracking
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mentions: int = 0
    sources: Set[str] = field(default_factory=set)

    # Classification
    is_coordinated: bool = False
    threat_score: float = 0.0


@dataclass
class BotNetwork:
    """Detected bot network"""
    network_id: str
    accounts: Set[str] = field(default_factory=set)
    coordination_score: float = 0.0
    narratives_pushed: Set[str] = field(default_factory=set)


@dataclass
class InfluenceOperation:
    """Detected influence operation"""
    operation_id: str
    name: str
    narratives: List[str] = field(default_factory=list)
    bot_networks: List[str] = field(default_factory=list)
    targets: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MemeticWarfareDetector:
    """
    Memetic Warfare Detection System

    Tracks and detects influence operations.

    Usage:
        detector = MemeticWarfareDetector()

        # Track content
        detector.track_content(post_content, source, timestamp)

        # Detect coordinated behavior
        operations = detector.detect_operations()
    """

    def __init__(self, coordination_threshold: float = 0.7):
        self.coordination_threshold = coordination_threshold

        self._narratives: Dict[str, Narrative] = {}
        self._bot_networks: Dict[str, BotNetwork] = {}
        self._operations: Dict[str, InfluenceOperation] = {}

        # Tracking
        self._source_activity: Dict[str, List[datetime]] = defaultdict(list)
        self._source_narratives: Dict[str, Set[str]] = defaultdict(set)

        self._lock = threading.RLock()

        logger.info("MemeticWarfareDetector initialized")

    def track_content(self, content: str, source: str,
                     timestamp: Optional[datetime] = None):
        """Track content for narrative detection"""
        ts = timestamp or datetime.now(timezone.utc)

        with self._lock:
            # Extract keywords (simplified)
            keywords = set(w.lower() for w in content.split() if len(w) > 4)

            # Find or create narrative
            narrative_id = self._find_or_create_narrative(content, keywords)

            if narrative_id:
                self._narratives[narrative_id].mentions += 1
                self._narratives[narrative_id].sources.add(source)
                self._source_narratives[source].add(narrative_id)

            self._source_activity[source].append(ts)

    def _find_or_create_narrative(self, content: str, keywords: Set[str]) -> Optional[str]:
        """Find existing narrative or create new one"""
        # Check for keyword overlap with existing narratives
        for nid, narrative in self._narratives.items():
            overlap = keywords & narrative.keywords
            if len(overlap) >= 3:  # Threshold for same narrative
                narrative.keywords.update(keywords)
                return nid

        # Create new narrative
        narrative_id = hashlib.sha256(content[:100].encode()).hexdigest()[:16]
        self._narratives[narrative_id] = Narrative(
            narrative_id=narrative_id,
            content=content[:200],
            keywords=keywords,
        )
        return narrative_id

    def detect_coordination(self) -> List[BotNetwork]:
        """Detect coordinated behavior"""
        networks = []

        with self._lock:
            # Group sources by shared narratives
            narrative_sources = defaultdict(set)
            for source, narratives in self._source_narratives.items():
                for nid in narratives:
                    narrative_sources[nid].add(source)

            # Find sources with high overlap
            processed = set()
            for sources in narrative_sources.values():
                if len(sources) < 3:
                    continue

                sources_tuple = tuple(sorted(sources))
                if sources_tuple in processed:
                    continue
                processed.add(sources_tuple)

                # Check timing coordination
                coordination = self._calculate_coordination(sources)

                if coordination >= self.coordination_threshold:
                    network_id = hashlib.sha256(str(sources_tuple).encode()).hexdigest()[:16]
                    network = BotNetwork(
                        network_id=network_id,
                        accounts=sources,
                        coordination_score=coordination,
                    )
                    networks.append(network)
                    self._bot_networks[network_id] = network

        return networks

    def _calculate_coordination(self, sources: Set[str]) -> float:
        """Calculate coordination score based on timing"""
        if len(sources) < 2:
            return 0.0

        # Check if sources post at similar times
        all_times = []
        for source in sources:
            all_times.extend(self._source_activity.get(source, []))

        if len(all_times) < 2:
            return 0.0

        all_times.sort()

        # Calculate average interval
        intervals = []
        for i in range(len(all_times) - 1):
            intervals.append((all_times[i+1] - all_times[i]).total_seconds())

        if not intervals:
            return 0.0

        avg_interval = sum(intervals) / len(intervals)

        # Short intervals = high coordination
        if avg_interval < 60:  # Less than 1 minute average
            return 0.9
        elif avg_interval < 300:  # Less than 5 minutes
            return 0.7
        elif avg_interval < 3600:  # Less than 1 hour
            return 0.4

        return 0.2

    def detect_operations(self) -> List[InfluenceOperation]:
        """Detect influence operations"""
        operations = []

        with self._lock:
            # First detect coordination
            networks = self.detect_coordination()

            # Group networks pushing same narratives
            for network in networks:
                narratives_pushed = set()
                for account in network.accounts:
                    narratives_pushed.update(self._source_narratives.get(account, set()))

                network.narratives_pushed = narratives_pushed

                if len(narratives_pushed) >= 2:  # Multi-narrative campaign
                    op_id = hashlib.sha256(f"op:{network.network_id}".encode()).hexdigest()[:16]

                    operation = InfluenceOperation(
                        operation_id=op_id,
                        name=f"Operation {op_id[:8]}",
                        narratives=list(narratives_pushed),
                        bot_networks=[network.network_id],
                        confidence=network.coordination_score,
                    )
                    operations.append(operation)
                    self._operations[op_id] = operation

        return operations

    def get_stats(self) -> Dict:
        """Get detector statistics"""
        with self._lock:
            return {
                "narratives_tracked": len(self._narratives),
                "sources_tracked": len(self._source_activity),
                "bot_networks": len(self._bot_networks),
                "operations_detected": len(self._operations),
            }


if __name__ == "__main__":
    print("Memetic Warfare Detection Self-Test")
    print("=" * 50)

    detector = MemeticWarfareDetector(coordination_threshold=0.3)

    print("\n[1] Track Content")
    # Simulate coordinated posts
    from datetime import timedelta
    base_time = datetime.now(timezone.utc)

    posts = [
        ("The government is hiding the truth about project X", "bot1", base_time),
        ("Truth about project X being hidden by authorities", "bot2", base_time + timedelta(seconds=30)),
        ("Project X truth hidden from public!", "bot3", base_time + timedelta(seconds=45)),
        ("Something completely unrelated", "user1", base_time + timedelta(hours=1)),
        ("Wake up! Project X is real and being covered up", "bot1", base_time + timedelta(minutes=2)),
    ]

    for content, source, ts in posts:
        detector.track_content(content, source, ts)

    print(f"    Tracked {len(posts)} posts")

    print("\n[2] Detect Coordination")
    networks = detector.detect_coordination()
    print(f"    Bot networks detected: {len(networks)}")
    for net in networks:
        print(f"      - {net.network_id}: {len(net.accounts)} accounts, score={net.coordination_score:.2f}")

    print("\n[3] Detect Operations")
    operations = detector.detect_operations()
    print(f"    Operations detected: {len(operations)}")
    for op in operations:
        print(f"      - {op.name}: {len(op.narratives)} narratives, confidence={op.confidence:.2f}")

    print("\n[4] Statistics")
    stats = detector.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Memetic Warfare Detection test complete")

