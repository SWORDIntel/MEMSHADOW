"""
Global Workspace
Phase 8.3: Limited-capacity conscious workspace

Implements Global Workspace Theory (Baars, 1988):
- Limited capacity (7±2 items)
- Broadcasting mechanism
- Competition for workspace access
- Temporal decay

The global workspace acts as a "blackboard" where information
becomes "conscious" and is broadcast to all cognitive modules.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import heapq
import structlog

logger = structlog.get_logger()


class ItemPriority(Enum):
    """Priority levels for workspace items"""
    CRITICAL = 100  # Immediate attention required
    HIGH = 75      # Important but not urgent
    NORMAL = 50    # Standard processing
    LOW = 25       # Background processing
    MINIMAL = 10   # Can be deferred


@dataclass
class WorkspaceItem:
    """
    Item in the global workspace.

    Represents a piece of information that has "won" the competition
    for conscious awareness and is being broadcast to all modules.
    """
    item_id: str
    content: Any
    source_module: str

    # Competition metrics
    salience: float  # How attention-grabbing (0.0 to 1.0)
    relevance: float  # How relevant to current goals (0.0 to 1.0)
    novelty: float   # How new/unexpected (0.0 to 1.0)
    priority: ItemPriority

    # Timing
    entered_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    time_to_live_seconds: float = 60.0

    # Broadcasting
    broadcast_count: int = 0
    subscribed_modules: Set[str] = field(default_factory=set)

    # Metadata
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def age_seconds(self) -> float:
        """How long item has been in workspace"""
        return (datetime.utcnow() - self.entered_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if item has exceeded its TTL"""
        return self.age_seconds > self.time_to_live_seconds

    @property
    def activation_level(self) -> float:
        """
        Combined activation level for competition.

        Higher activation = more likely to stay in workspace.
        """
        # Base activation from metrics
        base = (self.salience + self.relevance + self.novelty) / 3.0

        # Priority multiplier
        priority_multiplier = self.priority.value / 100.0

        # Recency bonus (decay over time)
        seconds_since_access = (datetime.utcnow() - self.last_accessed).total_seconds()
        recency_bonus = max(0, 1.0 - (seconds_since_access / 60.0))

        return base * priority_multiplier * (1.0 + recency_bonus)

    def __lt__(self, other: 'WorkspaceItem') -> bool:
        """Comparison for priority queue (higher activation = higher priority)"""
        return self.activation_level > other.activation_level


@dataclass
class WorkspaceState:
    """Current state of the global workspace"""
    capacity: int
    current_items: int
    utilization_percent: float
    total_broadcasts: int
    total_items_processed: int
    avg_item_lifetime_seconds: float


@dataclass
class BroadcastResult:
    """Result of broadcasting workspace contents"""
    broadcast_id: str
    timestamp: datetime
    items_broadcast: List[str]  # Item IDs
    recipient_modules: Set[str]
    broadcast_size_bytes: int


class GlobalWorkspace:
    """
    Global Workspace for conscious processing.

    Implements a limited-capacity workspace where information competes
    for "conscious" awareness. Items in the workspace are broadcast
    to all cognitive modules.

    Based on Global Workspace Theory (Baars, 1988).

    Architecture:
        - Limited capacity (7±2 items by default)
        - Competition-based access
        - Broadcasting to subscribed modules
        - Temporal decay of items
        - Priority-based eviction

    Example:
        workspace = GlobalWorkspace(capacity=7)
        await workspace.start()

        # Add item to workspace
        item = WorkspaceItem(
            item_id="item_001",
            content={"type": "insight", "data": ...},
            source_module="pattern_detector",
            salience=0.8,
            relevance=0.9,
            novelty=0.7,
            priority=ItemPriority.HIGH
        )

        added = await workspace.add_item(item)

        # Broadcast to all modules
        result = await workspace.broadcast()
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's Law: 7±2
        broadcast_interval_seconds: float = 1.0,
        auto_decay: bool = True
    ):
        """
        Initialize global workspace.

        Args:
            capacity: Maximum items in workspace (default 7)
            broadcast_interval_seconds: How often to broadcast
            auto_decay: Automatically remove expired items
        """
        self.capacity = capacity
        self.broadcast_interval = broadcast_interval_seconds
        self.auto_decay = auto_decay

        # Workspace contents (priority queue)
        self.items: Dict[str, WorkspaceItem] = {}
        self._item_heap: List[WorkspaceItem] = []

        # Subscribed modules
        self.modules: Set[str] = set()

        # Statistics
        self.total_broadcasts = 0
        self.total_items_processed = 0
        self.item_lifetimes: List[float] = []

        # Background tasks
        self._broadcast_task = None
        self._decay_task = None

        logger.info(
            "Global workspace initialized",
            capacity=capacity,
            broadcast_interval=broadcast_interval_seconds
        )

    async def start(self):
        """Start workspace background tasks"""
        import asyncio

        logger.info("Starting global workspace")

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        if self.auto_decay:
            self._decay_task = asyncio.create_task(self._decay_loop())

    async def stop(self):
        """Stop workspace background tasks"""
        logger.info("Stopping global workspace")

        if self._broadcast_task:
            self._broadcast_task.cancel()
        if self._decay_task:
            self._decay_task.cancel()

    def subscribe_module(self, module_name: str):
        """Subscribe a module to workspace broadcasts"""
        self.modules.add(module_name)
        logger.debug("Module subscribed", module=module_name)

    def unsubscribe_module(self, module_name: str):
        """Unsubscribe a module"""
        if module_name in self.modules:
            self.modules.remove(module_name)
            logger.debug("Module unsubscribed", module=module_name)

    async def add_item(self, item: WorkspaceItem) -> bool:
        """
        Add item to workspace (if it wins competition).

        Args:
            item: Item to add

        Returns:
            True if item was added, False if rejected
        """
        # If workspace is full, check if new item should replace something
        if len(self.items) >= self.capacity:
            # Find lowest activation item
            lowest_item = min(self.items.values(), key=lambda i: i.activation_level)

            # New item must have higher activation to enter
            if item.activation_level <= lowest_item.activation_level:
                logger.debug(
                    "Item rejected - insufficient activation",
                    item_id=item.item_id,
                    activation=item.activation_level,
                    threshold=lowest_item.activation_level
                )
                return False

            # Evict lowest activation item
            await self.remove_item(lowest_item.item_id, reason="evicted")

        # Add item
        self.items[item.item_id] = item
        heapq.heappush(self._item_heap, item)

        logger.info(
            "Item added to workspace",
            item_id=item.item_id,
            source=item.source_module,
            activation=item.activation_level,
            workspace_utilization=f"{len(self.items)}/{self.capacity}"
        )

        return True

    async def remove_item(self, item_id: str, reason: str = "removed"):
        """Remove item from workspace"""
        if item_id not in self.items:
            return

        item = self.items[item_id]

        # Record lifetime
        lifetime = item.age_seconds
        self.item_lifetimes.append(lifetime)

        # Remove
        del self.items[item_id]

        # Rebuild heap (remove is O(n), so just rebuild)
        self._item_heap = [i for i in self._item_heap if i.item_id != item_id]
        heapq.heapify(self._item_heap)

        logger.debug(
            "Item removed from workspace",
            item_id=item_id,
            reason=reason,
            lifetime_seconds=lifetime
        )

        self.total_items_processed += 1

    async def access_item(self, item_id: str) -> Optional[WorkspaceItem]:
        """
        Access an item in workspace (updates last_accessed).

        Args:
            item_id: Item to access

        Returns:
            Item if found, None otherwise
        """
        if item_id not in self.items:
            return None

        item = self.items[item_id]
        item.last_accessed = datetime.utcnow()

        return item

    async def broadcast(self) -> BroadcastResult:
        """
        Broadcast workspace contents to all subscribed modules.

        Returns:
            Broadcast result
        """
        broadcast_id = str(uuid.uuid4())

        # Get all items
        item_ids = list(self.items.keys())

        # Increment broadcast count
        for item in self.items.values():
            item.broadcast_count += 1

        # Create result
        result = BroadcastResult(
            broadcast_id=broadcast_id,
            timestamp=datetime.utcnow(),
            items_broadcast=item_ids,
            recipient_modules=self.modules.copy(),
            broadcast_size_bytes=self._estimate_broadcast_size()
        )

        self.total_broadcasts += 1

        logger.debug(
            "Workspace broadcast",
            broadcast_id=broadcast_id,
            items_count=len(item_ids),
            recipients=len(self.modules)
        )

        # In production: actually send to modules via message bus

        return result

    async def get_state(self) -> WorkspaceState:
        """Get current workspace state"""
        avg_lifetime = sum(self.item_lifetimes) / len(self.item_lifetimes) \
            if self.item_lifetimes else 0.0

        return WorkspaceState(
            capacity=self.capacity,
            current_items=len(self.items),
            utilization_percent=(len(self.items) / self.capacity) * 100,
            total_broadcasts=self.total_broadcasts,
            total_items_processed=self.total_items_processed,
            avg_item_lifetime_seconds=avg_lifetime
        )

    async def get_items(self) -> List[WorkspaceItem]:
        """Get all items currently in workspace (sorted by activation)"""
        return sorted(self.items.values(), key=lambda i: i.activation_level, reverse=True)

    async def clear(self):
        """Clear all items from workspace"""
        item_ids = list(self.items.keys())
        for item_id in item_ids:
            await self.remove_item(item_id, reason="cleared")

        logger.info("Workspace cleared")

    # Private methods

    async def _broadcast_loop(self):
        """Background loop for periodic broadcasting"""
        import asyncio

        while True:
            try:
                await asyncio.sleep(self.broadcast_interval)

                if self.items:
                    await self.broadcast()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Broadcast loop error", error=str(e))

    async def _decay_loop(self):
        """Background loop for removing expired items"""
        import asyncio

        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Find expired items
                expired = [
                    item_id for item_id, item in self.items.items()
                    if item.is_expired
                ]

                # Remove expired items
                for item_id in expired:
                    await self.remove_item(item_id, reason="expired")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Decay loop error", error=str(e))

    def _estimate_broadcast_size(self) -> int:
        """Estimate broadcast payload size in bytes"""
        # Rough estimate
        return len(self.items) * 1024  # Assume 1KB per item
