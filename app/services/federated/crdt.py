"""
Conflict-Free Replicated Data Types (CRDTs) for Federated Memory
Phase 8.1: Eventually consistent distributed data structures

Implements CRDTs for conflict-free memory synchronization:
- LWW-Element-Set: Last-Write-Wins set
- G-Counter: Grow-only counter
- PN-Counter: Positive-Negative counter
- OR-Set: Observed-Remove set
- LWW-Register: Last-Write-Wins register
"""

from typing import Any, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import structlog

logger = structlog.get_logger()


@dataclass
class VectorClock:
    """
    Vector clock for causality tracking.

    Each node maintains a counter for every other node.
    """
    clock: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str):
        """Increment counter for node"""
        self.clock[node_id] = self.clock.get(node_id, 0) + 1

    def update(self, other: 'VectorClock'):
        """Update with another vector clock (take maximum)"""
        for node_id, count in other.clock.items():
            self.clock[node_id] = max(self.clock.get(node_id, 0), count)

    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before another"""
        # self <= other and self != other
        for node_id, count in self.clock.items():
            if count > other.clock.get(node_id, 0):
                return False

        return self.clock != other.clock

    def concurrent(self, other: 'VectorClock') -> bool:
        """Check if two clocks are concurrent (neither happens before the other)"""
        return not self.happens_before(other) and not other.happens_before(self)


class GCounter:
    """
    Grow-Only Counter CRDT.

    A counter that can only increment. Each node has its own counter.
    The total is the sum of all node counters.

    Use case: Counting memory accesses, views, etc.
    """

    def __init__(self, node_id: str):
        """
        Initialize G-Counter.

        Args:
            node_id: This node's identifier
        """
        self.node_id = node_id
        self.counts: Dict[str, int] = defaultdict(int)

    def increment(self, amount: int = 1):
        """Increment this node's counter"""
        self.counts[self.node_id] += amount

    def value(self) -> int:
        """Get current counter value"""
        return sum(self.counts.values())

    def merge(self, other: 'GCounter'):
        """Merge with another G-Counter (take maximum for each node)"""
        for node_id, count in other.counts.items():
            self.counts[node_id] = max(self.counts[node_id], count)


class PNCounter:
    """
    Positive-Negative Counter CRDT.

    A counter that can increment and decrement.
    Implemented as two G-Counters (positive and negative).

    Use case: Voting, reputation scores, etc.
    """

    def __init__(self, node_id: str):
        """Initialize PN-Counter"""
        self.node_id = node_id
        self.positive = GCounter(node_id)
        self.negative = GCounter(node_id)

    def increment(self, amount: int = 1):
        """Increment counter"""
        self.positive.increment(amount)

    def decrement(self, amount: int = 1):
        """Decrement counter"""
        self.negative.increment(amount)

    def value(self) -> int:
        """Get current counter value"""
        return self.positive.value() - self.negative.value()

    def merge(self, other: 'PNCounter'):
        """Merge with another PN-Counter"""
        self.positive.merge(other.positive)
        self.negative.merge(other.negative)


class LWWRegister:
    """
    Last-Write-Wins Register CRDT.

    Stores a single value with timestamp. Concurrent writes resolved
    by taking the value with the latest timestamp (with tie-breaking).

    Use case: Distributed configuration, user preferences
    """

    def __init__(self, node_id: str):
        """Initialize LWW-Register"""
        self.node_id = node_id
        self.value: Any = None
        self.timestamp: float = 0.0
        self.writer_id: str = node_id

    def set(self, value: Any):
        """Set register value"""
        self.value = value
        self.timestamp = datetime.utcnow().timestamp()
        self.writer_id = self.node_id

    def get(self) -> Any:
        """Get register value"""
        return self.value

    def merge(self, other: 'LWWRegister'):
        """Merge with another LWW-Register (take latest write)"""
        # Compare timestamps
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
            self.writer_id = other.writer_id

        elif other.timestamp == self.timestamp:
            # Tie-break by node ID (deterministic)
            if other.writer_id > self.writer_id:
                self.value = other.value
                self.writer_id = other.writer_id


class LWWElementSet:
    """
    Last-Write-Wins Element Set CRDT.

    A set where adds and removes are timestamped. Latest operation wins.
    Bias towards adds (if timestamps equal, element is in the set).

    Use case: Distributed tag sets, label sets
    """

    def __init__(self, node_id: str):
        """Initialize LWW-Element-Set"""
        self.node_id = node_id

        # element -> (timestamp, operation)
        self.add_set: Dict[Any, Tuple[float, str]] = {}
        self.remove_set: Dict[Any, Tuple[float, str]] = {}

    def add(self, element: Any):
        """Add element to set"""
        timestamp = datetime.utcnow().timestamp()
        self.add_set[element] = (timestamp, self.node_id)

    def remove(self, element: Any):
        """Remove element from set"""
        timestamp = datetime.utcnow().timestamp()
        self.remove_set[element] = (timestamp, self.node_id)

    def contains(self, element: Any) -> bool:
        """Check if element is in set"""
        # Element is in set if:
        # 1. It was added
        # 2. It wasn't removed, or remove timestamp < add timestamp

        if element not in self.add_set:
            return False

        add_ts, add_node = self.add_set[element]

        if element not in self.remove_set:
            return True

        remove_ts, remove_node = self.remove_set[element]

        # If timestamps equal, bias towards add
        if remove_ts < add_ts:
            return True
        elif remove_ts == add_ts:
            return True  # Bias towards add
        else:
            return False

    def elements(self) -> Set[Any]:
        """Get all elements in set"""
        return {elem for elem in self.add_set if self.contains(elem)}

    def merge(self, other: 'LWWElementSet'):
        """Merge with another LWW-Element-Set"""
        # Merge add sets (take max timestamp for each element)
        for element, (timestamp, node_id) in other.add_set.items():
            if element not in self.add_set:
                self.add_set[element] = (timestamp, node_id)
            else:
                self_ts, self_node = self.add_set[element]
                if timestamp > self_ts or (timestamp == self_ts and node_id > self_node):
                    self.add_set[element] = (timestamp, node_id)

        # Merge remove sets
        for element, (timestamp, node_id) in other.remove_set.items():
            if element not in self.remove_set:
                self.remove_set[element] = (timestamp, node_id)
            else:
                self_ts, self_node = self.remove_set[element]
                if timestamp > self_ts or (timestamp == self_ts and node_id > self_node):
                    self.remove_set[element] = (timestamp, node_id)


class ORSet:
    """
    Observed-Remove Set CRDT.

    More sophisticated than LWW-Element-Set. Each add operation gets
    a unique tag. Removes specify which tags to remove.

    Guarantees: If an add is observed before a remove, the element
    will be in the set after merge.

    Use case: Collaborative editing, distributed collections
    """

    def __init__(self, node_id: str):
        """Initialize OR-Set"""
        self.node_id = node_id

        # element -> set of unique tags
        self.elements: Dict[Any, Set[str]] = defaultdict(set)

        self._counter = 0

    def add(self, element: Any) -> str:
        """
        Add element to set.

        Returns:
            Unique tag for this add operation
        """
        # Generate unique tag
        tag = f"{self.node_id}:{self._counter}:{datetime.utcnow().timestamp()}"
        self._counter += 1

        self.elements[element].add(tag)

        return tag

    def remove(self, element: Any):
        """Remove element from set (removes all observed tags)"""
        if element in self.elements:
            self.elements[element].clear()

    def contains(self, element: Any) -> bool:
        """Check if element is in set"""
        return element in self.elements and len(self.elements[element]) > 0

    def get_elements(self) -> Set[Any]:
        """Get all elements in set"""
        return {elem for elem, tags in self.elements.items() if len(tags) > 0}

    def merge(self, other: 'ORSet'):
        """Merge with another OR-Set"""
        # Union of tags for each element
        for element, tags in other.elements.items():
            self.elements[element].update(tags)


class MemoryCRDT:
    """
    Memory CRDT for MEMSHADOW federated system.

    Combines multiple CRDTs for different memory attributes:
    - Access count (G-Counter)
    - Importance score (PN-Counter)
    - Tags (LWW-Element-Set)
    - Metadata (LWW-Register)
    - Related memories (OR-Set)

    Example:
        memory_crdt = MemoryCRDT(node_id="node_001", memory_id="mem_123")

        # Update locally
        memory_crdt.increment_access_count()
        memory_crdt.adjust_importance(+5)
        memory_crdt.add_tag("important")

        # Merge with replica from another node
        memory_crdt.merge(other_memory_crdt)

        # Get current state
        state = memory_crdt.get_state()
    """

    def __init__(self, node_id: str, memory_id: str):
        """
        Initialize Memory CRDT.

        Args:
            node_id: This node's identifier
            memory_id: Memory identifier
        """
        self.node_id = node_id
        self.memory_id = memory_id

        # CRDT components
        self.access_count = GCounter(node_id)
        self.importance_score = PNCounter(node_id)
        self.tags = LWWElementSet(node_id)
        self.metadata = LWWRegister(node_id)
        self.related_memories = ORSet(node_id)

        logger.debug(
            "Memory CRDT initialized",
            memory_id=memory_id,
            node_id=node_id
        )

    def increment_access_count(self, amount: int = 1):
        """Increment access count"""
        self.access_count.increment(amount)

    def get_access_count(self) -> int:
        """Get current access count"""
        return self.access_count.value()

    def adjust_importance(self, delta: int):
        """Adjust importance score"""
        if delta > 0:
            self.importance_score.increment(delta)
        else:
            self.importance_score.decrement(abs(delta))

    def get_importance_score(self) -> int:
        """Get current importance score"""
        return self.importance_score.value()

    def add_tag(self, tag: str):
        """Add tag to memory"""
        self.tags.add(tag)

    def remove_tag(self, tag: str):
        """Remove tag from memory"""
        self.tags.remove(tag)

    def get_tags(self) -> Set[str]:
        """Get all tags"""
        return self.tags.elements()

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set memory metadata"""
        self.metadata.set(metadata)

    def get_metadata(self) -> Dict[str, Any]:
        """Get memory metadata"""
        return self.metadata.get()

    def add_related_memory(self, related_memory_id: str) -> str:
        """
        Add related memory.

        Returns:
            Unique tag for this relation
        """
        return self.related_memories.add(related_memory_id)

    def remove_related_memory(self, related_memory_id: str):
        """Remove related memory"""
        self.related_memories.remove(related_memory_id)

    def get_related_memories(self) -> Set[str]:
        """Get all related memory IDs"""
        return self.related_memories.get_elements()

    def merge(self, other: 'MemoryCRDT'):
        """
        Merge with another Memory CRDT.

        After merge, both replicas converge to the same state.
        """
        if self.memory_id != other.memory_id:
            raise ValueError(
                f"Cannot merge CRDTs for different memories: "
                f"{self.memory_id} != {other.memory_id}"
            )

        # Merge each CRDT component
        self.access_count.merge(other.access_count)
        self.importance_score.merge(other.importance_score)
        self.tags.merge(other.tags)
        self.metadata.merge(other.metadata)
        self.related_memories.merge(other.related_memories)

        logger.debug(
            "Memory CRDT merged",
            memory_id=self.memory_id,
            node_id=self.node_id,
            other_node=other.node_id
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current CRDT state"""
        return {
            "memory_id": self.memory_id,
            "node_id": self.node_id,
            "access_count": self.get_access_count(),
            "importance_score": self.get_importance_score(),
            "tags": list(self.get_tags()),
            "metadata": self.get_metadata(),
            "related_memories": list(self.get_related_memories())
        }

    def __repr__(self) -> str:
        return (
            f"MemoryCRDT(memory_id={self.memory_id}, "
            f"accesses={self.get_access_count()}, "
            f"importance={self.get_importance_score()}, "
            f"tags={len(self.get_tags())})"
        )
