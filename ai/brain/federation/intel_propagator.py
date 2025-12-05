#!/usr/bin/env python3
"""
Intel Propagator for DSMIL Brain Federation

Manages intelligence propagation across the network:
- Priority-based propagation
- Deduplication
- Relevance filtering per node
- Propagation tracking
- Feedback loops for importance
"""

import hashlib
import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import deque
import json

logger = logging.getLogger(__name__)


class PropagationPriority(Enum):
    """Priority levels for intel propagation"""
    BACKGROUND = 0     # Low priority, batch propagation
    NORMAL = 1         # Standard propagation
    ELEVATED = 2       # Faster propagation
    HIGH = 3           # Priority propagation
    CRITICAL = 4       # Immediate propagation
    EMERGENCY = 5      # Flash traffic - all nodes immediately


class IntelType(Enum):
    """Types of intelligence"""
    THREAT_INDICATOR = auto()   # IOCs, signatures
    BEHAVIORAL = auto()         # Behavioral patterns
    CORRELATION = auto()        # Cross-correlation results
    ALERT = auto()              # Active alerts
    PATTERN = auto()            # Detected patterns
    KNOWLEDGE = auto()          # Knowledge graph updates
    POLICY = auto()             # Policy updates
    CAPABILITY = auto()         # Capability updates


@dataclass
class IntelReport:
    """
    Intelligence report for propagation
    """
    report_id: str
    intel_type: IntelType
    priority: PropagationPriority

    # Content
    content: Dict[str, Any]
    summary: str = ""

    # Source
    source_node: str = ""
    source_confidence: float = 0.5

    # Relevance
    relevant_domains: Set[str] = field(default_factory=set)
    relevant_capabilities: Set[str] = field(default_factory=set)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    # Propagation tracking
    propagated_to: Set[str] = field(default_factory=set)
    acknowledged_by: Set[str] = field(default_factory=set)

    # Deduplication
    content_hash: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:32]

    def is_expired(self) -> bool:
        """Check if report has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_relevant_for(self, node_domains: Set[str],
                        node_capabilities: Set[str]) -> bool:
        """Check if report is relevant for a node"""
        # If no relevance filters, relevant to all
        if not self.relevant_domains and not self.relevant_capabilities:
            return True

        # Check domain overlap
        if self.relevant_domains:
            if not self.relevant_domains & node_domains:
                return False

        # Check capability overlap
        if self.relevant_capabilities:
            if not self.relevant_capabilities & node_capabilities:
                return False

        return True

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "intel_type": self.intel_type.name,
            "priority": self.priority.name,
            "content": self.content,
            "summary": self.summary,
            "source_node": self.source_node,
            "source_confidence": self.source_confidence,
            "created_at": self.created_at.isoformat(),
            "content_hash": self.content_hash,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "IntelReport":
        return cls(
            report_id=data["report_id"],
            intel_type=IntelType[data["intel_type"]],
            priority=PropagationPriority[data["priority"]],
            content=data["content"],
            summary=data.get("summary", ""),
            source_node=data.get("source_node", ""),
            source_confidence=data.get("source_confidence", 0.5),
            content_hash=data.get("content_hash", ""),
            tags=set(data.get("tags", [])),
        )


class PropagationQueue:
    """
    Priority queue for intel propagation
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[PropagationPriority, deque] = {
            p: deque(maxlen=max_size // len(PropagationPriority))
            for p in PropagationPriority
        }
        self._lock = threading.Lock()

    def enqueue(self, report: IntelReport):
        """Add report to queue"""
        with self._lock:
            self._queues[report.priority].append(report)

    def dequeue(self) -> Optional[IntelReport]:
        """Get highest priority report"""
        with self._lock:
            # Check queues in priority order (highest first)
            for priority in sorted(PropagationPriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                if queue:
                    return queue.popleft()
            return None

    def peek(self) -> Optional[IntelReport]:
        """Peek at highest priority report without removing"""
        with self._lock:
            for priority in sorted(PropagationPriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                if queue:
                    return queue[0]
            return None

    def size(self) -> int:
        """Get total queue size"""
        with self._lock:
            return sum(len(q) for q in self._queues.values())

    def size_by_priority(self) -> Dict[str, int]:
        """Get size broken down by priority"""
        with self._lock:
            return {p.name: len(q) for p, q in self._queues.items()}


class IntelPropagator:
    """
    Manages intelligence propagation across the network

    Features:
    - Priority-based propagation scheduling
    - Content deduplication
    - Relevance filtering
    - Propagation tracking
    - Acknowledgment handling

    Usage:
        propagator = IntelPropagator(node_id="hub-001")

        # Create and propagate intel
        report = propagator.create_report(
            intel_type=IntelType.THREAT_INDICATOR,
            priority=PropagationPriority.HIGH,
            content={"ioc": "192.168.1.100", "type": "ip"}
        )

        propagator.propagate(report, target_nodes)

        # Start background propagation
        propagator.start_propagation_worker()
    """

    def __init__(self, node_id: str,
                 dedup_window_hours: int = 24):
        """
        Initialize propagator

        Args:
            node_id: This node's ID
            dedup_window_hours: Hours to keep dedup cache
        """
        self.node_id = node_id
        self.dedup_window = timedelta(hours=dedup_window_hours)

        # Queues
        self._outgoing_queue = PropagationQueue()
        self._pending_acks: Dict[str, IntelReport] = {}

        # Deduplication
        self._seen_hashes: Dict[str, datetime] = {}

        # Node registry (for relevance filtering)
        self._node_info: Dict[str, Dict] = {}  # node_id -> {domains, capabilities}

        # Callbacks
        self.on_intel_created: Optional[Callable[[IntelReport], None]] = None
        self.on_propagation_complete: Optional[Callable[[str, int], None]] = None
        self.send_to_node: Optional[Callable[[str, IntelReport], bool]] = None

        # Background worker
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            "reports_created": 0,
            "reports_propagated": 0,
            "duplicates_filtered": 0,
            "irrelevant_filtered": 0,
            "acks_received": 0,
        }

        self._lock = threading.RLock()

        logger.info(f"IntelPropagator initialized for {node_id}")

    def register_node(self, node_id: str,
                     domains: Set[str],
                     capabilities: Set[str]):
        """Register a node for relevance filtering"""
        with self._lock:
            self._node_info[node_id] = {
                "domains": domains,
                "capabilities": capabilities,
            }

    def unregister_node(self, node_id: str):
        """Unregister a node"""
        with self._lock:
            if node_id in self._node_info:
                del self._node_info[node_id]

    def create_report(self, intel_type: IntelType,
                     priority: PropagationPriority,
                     content: Dict,
                     summary: str = "",
                     relevant_domains: Optional[Set[str]] = None,
                     relevant_capabilities: Optional[Set[str]] = None,
                     expires_hours: Optional[int] = None,
                     tags: Optional[Set[str]] = None) -> IntelReport:
        """
        Create an intel report

        Args:
            intel_type: Type of intelligence
            priority: Propagation priority
            content: Report content
            summary: Human-readable summary
            relevant_domains: Domains this is relevant to
            relevant_capabilities: Capabilities this is relevant to
            expires_hours: Hours until expiration
            tags: Tags for categorization

        Returns:
            IntelReport instance
        """
        report_id = hashlib.sha256(
            f"{self.node_id}:{intel_type.name}:{time.time()}".encode()
        ).hexdigest()[:16]

        expires_at = None
        if expires_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)

        report = IntelReport(
            report_id=report_id,
            intel_type=intel_type,
            priority=priority,
            content=content,
            summary=summary,
            source_node=self.node_id,
            relevant_domains=relevant_domains or set(),
            relevant_capabilities=relevant_capabilities or set(),
            expires_at=expires_at,
            tags=tags or set(),
        )

        self.stats["reports_created"] += 1

        if self.on_intel_created:
            self.on_intel_created(report)

        return report

    def is_duplicate(self, report: IntelReport) -> bool:
        """Check if report is a duplicate"""
        with self._lock:
            # Clean old entries
            cutoff = datetime.now(timezone.utc) - self.dedup_window
            expired = [h for h, t in self._seen_hashes.items() if t < cutoff]
            for h in expired:
                del self._seen_hashes[h]

            # Check if seen
            if report.content_hash in self._seen_hashes:
                self.stats["duplicates_filtered"] += 1
                return True

            # Mark as seen
            self._seen_hashes[report.content_hash] = datetime.now(timezone.utc)
            return False

    def queue_for_propagation(self, report: IntelReport):
        """Queue report for propagation"""
        if not self.is_duplicate(report):
            self._outgoing_queue.enqueue(report)

    def propagate(self, report: IntelReport,
                 target_nodes: Optional[List[str]] = None) -> int:
        """
        Propagate report to nodes

        Args:
            report: Report to propagate
            target_nodes: Specific nodes (None = all registered)

        Returns:
            Number of nodes propagated to
        """
        # Check for duplicate
        if self.is_duplicate(report):
            return 0

        propagated_count = 0

        with self._lock:
            targets = target_nodes or list(self._node_info.keys())

            for node_id in targets:
                if node_id == self.node_id:
                    continue

                if node_id in report.propagated_to:
                    continue

                # Check relevance
                if node_id in self._node_info:
                    node_info = self._node_info[node_id]
                    if not report.is_relevant_for(
                        node_info["domains"],
                        node_info["capabilities"]
                    ):
                        self.stats["irrelevant_filtered"] += 1
                        continue

                # Send (would be actual network call)
                if self.send_to_node:
                    success = self.send_to_node(node_id, report)
                    if success:
                        report.propagated_to.add(node_id)
                        propagated_count += 1
                        self._pending_acks[report.report_id] = report
                else:
                    # Simulate success
                    report.propagated_to.add(node_id)
                    propagated_count += 1

            if propagated_count > 0:
                self.stats["reports_propagated"] += 1

        if self.on_propagation_complete:
            self.on_propagation_complete(report.report_id, propagated_count)

        logger.debug(f"Propagated {report.report_id} to {propagated_count} nodes")
        return propagated_count

    def acknowledge(self, report_id: str, node_id: str):
        """Record acknowledgment from a node"""
        with self._lock:
            if report_id in self._pending_acks:
                self._pending_acks[report_id].acknowledged_by.add(node_id)
                self.stats["acks_received"] += 1

    def broadcast_emergency(self, report: IntelReport):
        """
        Broadcast emergency intel to all nodes immediately

        Bypasses normal queue and filters.
        """
        report.priority = PropagationPriority.EMERGENCY

        with self._lock:
            # Send to ALL nodes, regardless of relevance
            for node_id in self._node_info.keys():
                if node_id != self.node_id:
                    if self.send_to_node:
                        self.send_to_node(node_id, report)
                    report.propagated_to.add(node_id)

        logger.warning(f"EMERGENCY broadcast: {report.report_id}")

    def start_propagation_worker(self, interval: float = 1.0):
        """Start background propagation worker"""
        if self._running:
            return

        self._running = True

        def worker_loop():
            while self._running:
                try:
                    # Process queue
                    report = self._outgoing_queue.dequeue()
                    if report:
                        self.propagate(report)
                except Exception as e:
                    logger.error(f"Propagation worker error: {e}")

                time.sleep(interval)

        self._worker_thread = threading.Thread(target=worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Propagation worker started")

    def stop_propagation_worker(self):
        """Stop propagation worker"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def get_queue_status(self) -> Dict:
        """Get queue status"""
        return {
            "total_queued": self._outgoing_queue.size(),
            "by_priority": self._outgoing_queue.size_by_priority(),
            "pending_acks": len(self._pending_acks),
        }

    def get_stats(self) -> Dict:
        """Get propagator statistics"""
        return {
            **self.stats,
            "registered_nodes": len(self._node_info),
            "dedup_cache_size": len(self._seen_hashes),
            "queue_status": self.get_queue_status(),
        }


if __name__ == "__main__":
    print("Intel Propagator Self-Test")
    print("=" * 50)

    propagator = IntelPropagator("hub-001")

    print(f"\n[1] Register Nodes")
    propagator.register_node("node-001", {"threat_intel"}, {"search"})
    propagator.register_node("node-002", {"network_logs"}, {"correlate"})
    propagator.register_node("node-003", {"threat_intel", "network_logs"}, {"search", "correlate"})
    print(f"    Registered 3 nodes")

    print(f"\n[2] Create Reports")
    report1 = propagator.create_report(
        intel_type=IntelType.THREAT_INDICATOR,
        priority=PropagationPriority.HIGH,
        content={"ioc": "192.168.1.100", "type": "ip", "threat": "malware"},
        summary="Malicious IP detected",
        relevant_domains={"threat_intel"},
        tags={"malware", "ioc"}
    )
    print(f"    Created: {report1.report_id}")

    report2 = propagator.create_report(
        intel_type=IntelType.CORRELATION,
        priority=PropagationPriority.NORMAL,
        content={"correlation": "network-threat", "confidence": 0.85},
        summary="Network-threat correlation found",
        relevant_domains={"network_logs", "threat_intel"}
    )
    print(f"    Created: {report2.report_id}")

    print(f"\n[3] Propagate")
    count1 = propagator.propagate(report1)
    print(f"    Report 1 propagated to {count1} nodes")
    count2 = propagator.propagate(report2)
    print(f"    Report 2 propagated to {count2} nodes")

    print(f"\n[4] Duplicate Detection")
    dup_count = propagator.propagate(report1)
    print(f"    Duplicate propagated to {dup_count} nodes (expected 0)")

    print(f"\n[5] Queue Reports")
    for i in range(5):
        r = propagator.create_report(
            intel_type=IntelType.PATTERN,
            priority=PropagationPriority(i % 3),
            content={"pattern": f"pattern-{i}"}
        )
        propagator.queue_for_propagation(r)
    print(f"    Queue status: {propagator.get_queue_status()}")

    print(f"\n[6] Statistics")
    stats = propagator.get_stats()
    for key, value in stats.items():
        if key != "queue_status":
            print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Intel Propagator test complete")

