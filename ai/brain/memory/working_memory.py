#!/usr/bin/env python3
"""
Working Memory (L1) for DSMIL Brain

High-speed, hardware-adaptive memory scratchpad:
- Auto-detect RAM/VRAM and size accordingly
- Attention-weighted context window
- Real-time relevance scoring with decay
- Cross-node shared memory pool
- Dynamic compaction under pressure

This is the "fast" memory used for active reasoning and correlation.
"""

import os
import sys
import time
import threading
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple, Set
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import OrderedDict
import heapq
import math

logger = logging.getLogger(__name__)


# Try to get system memory info
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryPressure(Enum):
    """Memory pressure levels"""
    LOW = auto()       # <50% utilization
    MEDIUM = auto()    # 50-75% utilization
    HIGH = auto()      # 75-90% utilization
    CRITICAL = auto()  # >90% utilization


class ItemPriority(Enum):
    """Priority levels for memory items"""
    BACKGROUND = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    PINNED = 5  # Never evict


@dataclass
class AttentionWeight:
    """
    Attention weight for memory item

    Combines multiple factors into a single relevance score
    """
    base_relevance: float = 1.0      # Intrinsic importance
    recency_weight: float = 1.0      # Decay with time
    access_frequency: float = 1.0    # How often accessed
    context_match: float = 1.0       # Match to current context

    def compute_score(self) -> float:
        """Compute combined attention score"""
        return (
            self.base_relevance * 0.3 +
            self.recency_weight * 0.3 +
            self.access_frequency * 0.2 +
            self.context_match * 0.2
        )


@dataclass
class WorkingMemoryItem:
    """
    Single item in working memory
    """
    item_id: str
    content: Any
    content_type: str  # "text", "embedding", "structured", "binary"
    size_bytes: int

    # Attention and priority
    attention: AttentionWeight = field(default_factory=AttentionWeight)
    priority: ItemPriority = ItemPriority.NORMAL

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    # Relationships
    related_items: Set[str] = field(default_factory=set)
    source_episode_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self):
        """Mark item as accessed"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        self.attention.access_frequency = min(2.0, 1.0 + math.log1p(self.access_count) / 10)

    def decay(self, decay_rate: float = 0.01):
        """Apply time-based decay to attention"""
        elapsed = (datetime.now(timezone.utc) - self.last_accessed).total_seconds()
        self.attention.recency_weight = math.exp(-decay_rate * elapsed)

    def get_score(self) -> float:
        """Get current attention score"""
        self.decay()
        return self.attention.compute_score() * (self.priority.value + 1)


class HardwareProfile:
    """
    Detects and tracks hardware capabilities for memory sizing
    """

    def __init__(self):
        self.total_ram_bytes = 0
        self.available_ram_bytes = 0
        self.total_vram_bytes = 0
        self.available_vram_bytes = 0
        self.cpu_count = os.cpu_count() or 1
        self.has_gpu = False
        self.gpu_name = ""

        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware"""
        # RAM detection
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            self.total_ram_bytes = mem.total
            self.available_ram_bytes = mem.available
        else:
            # Fallback: try reading /proc/meminfo on Linux
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            self.total_ram_bytes = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            self.available_ram_bytes = int(line.split()[1]) * 1024
            except Exception:
                # Conservative defaults
                self.total_ram_bytes = 4 * 1024**3  # 4 GB
                self.available_ram_bytes = 2 * 1024**3  # 2 GB

        # GPU/VRAM detection
        self._detect_gpu()

        logger.info(f"Hardware: RAM={self.total_ram_bytes/1024**3:.1f}GB, "
                   f"VRAM={self.total_vram_bytes/1024**3:.1f}GB, "
                   f"CPUs={self.cpu_count}")

    def _detect_gpu(self):
        """Detect GPU and VRAM"""
        # Try NVIDIA
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.total_vram_bytes = info.total
            self.available_vram_bytes = info.free
            self.gpu_name = pynvml.nvmlDeviceGetName(handle)
            self.has_gpu = True
            pynvml.nvmlShutdown()
            return
        except Exception:
            pass

        # Try Intel GPU (through OpenVINO or similar)
        try:
            # Check for Intel GPU via sysfs
            intel_gpu_path = "/sys/class/drm/card0/device/vendor"
            if os.path.exists(intel_gpu_path):
                with open(intel_gpu_path) as f:
                    vendor = f.read().strip()
                    if vendor == "0x8086":  # Intel vendor ID
                        self.has_gpu = True
                        self.gpu_name = "Intel Integrated GPU"
                        # Estimate shared VRAM (typically ~50% of available RAM for iGPU)
                        self.total_vram_bytes = min(8 * 1024**3, self.available_ram_bytes // 2)
                        self.available_vram_bytes = self.total_vram_bytes
        except Exception:
            pass

    def refresh(self):
        """Refresh available memory stats"""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            self.available_ram_bytes = mem.available

    def get_recommended_working_memory_size(self, max_fraction: float = 0.15) -> int:
        """
        Get recommended working memory size in bytes

        Args:
            max_fraction: Maximum fraction of available RAM to use

        Returns:
            Recommended size in bytes
        """
        self.refresh()

        # Use fraction of available RAM, with min/max bounds
        base_size = int(self.available_ram_bytes * max_fraction)

        # Minimum 256 MB, maximum 8 GB for working memory
        min_size = 256 * 1024**2
        max_size = 8 * 1024**3

        return max(min_size, min(max_size, base_size))


class WorkingMemory:
    """
    L1 Working Memory - High-speed active memory

    Features:
    - Hardware-adaptive sizing
    - Attention-weighted eviction
    - Context-based retrieval
    - Cross-node synchronization support

    Usage:
        wm = WorkingMemory()

        # Store item
        item_id = wm.store("key-facts", {"fact": "important"}, priority=ItemPriority.HIGH)

        # Retrieve
        item = wm.retrieve(item_id)

        # Search by relevance
        results = wm.search_by_context(current_context, top_k=10)

        # Get pressure status
        pressure = wm.get_pressure()
    """

    def __init__(self, max_size_bytes: Optional[int] = None,
                 auto_compact: bool = True,
                 decay_rate: float = 0.001):
        """
        Initialize working memory

        Args:
            max_size_bytes: Maximum size (auto-detected if None)
            auto_compact: Automatically compact when pressure is high
            decay_rate: Decay rate for attention weights
        """
        self.hardware = HardwareProfile()

        # Size management
        if max_size_bytes is None:
            max_size_bytes = self.hardware.get_recommended_working_memory_size()

        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0

        # Storage
        self._items: Dict[str, WorkingMemoryItem] = {}
        self._lock = threading.RLock()

        # Configuration
        self.auto_compact = auto_compact
        self.decay_rate = decay_rate

        # Statistics
        self.stats = {
            "stores": 0,
            "retrievals": 0,
            "evictions": 0,
            "compactions": 0,
            "hits": 0,
            "misses": 0,
        }

        # Background maintenance
        self._maintenance_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"WorkingMemory initialized: {max_size_bytes/1024**2:.1f} MB max")

    def store(self, key: str, content: Any,
              content_type: str = "structured",
              priority: ItemPriority = ItemPriority.NORMAL,
              metadata: Optional[Dict] = None) -> str:
        """
        Store item in working memory

        Args:
            key: Unique key for the item
            content: Content to store
            content_type: Type of content
            priority: Priority level
            metadata: Additional metadata

        Returns:
            Item ID
        """
        # Estimate size
        size = self._estimate_size(content)

        # Generate ID
        item_id = hashlib.sha256(f"{key}:{time.time()}".encode()).hexdigest()[:16]

        item = WorkingMemoryItem(
            item_id=item_id,
            content=content,
            content_type=content_type,
            size_bytes=size,
            priority=priority,
            metadata=metadata or {},
        )

        with self._lock:
            # Check if we need to make room
            if self.current_size_bytes + size > self.max_size_bytes:
                if self.auto_compact:
                    self._compact(size)
                else:
                    # Evict lowest priority items
                    self._evict(size)

            # Store
            self._items[item_id] = item
            self.current_size_bytes += size
            self.stats["stores"] += 1

        return item_id

    def retrieve(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """
        Retrieve item by ID

        Args:
            item_id: Item identifier

        Returns:
            WorkingMemoryItem or None
        """
        with self._lock:
            item = self._items.get(item_id)

            if item:
                item.touch()
                self.stats["retrievals"] += 1
                self.stats["hits"] += 1
                return item
            else:
                self.stats["misses"] += 1
                return None

    def search_by_context(self, context: Dict[str, Any],
                         top_k: int = 10) -> List[WorkingMemoryItem]:
        """
        Search items by context relevance

        Args:
            context: Current context dictionary
            top_k: Number of results

        Returns:
            List of relevant items
        """
        with self._lock:
            scored_items = []

            for item in self._items.values():
                # Compute context match
                match_score = self._compute_context_match(item, context)
                item.attention.context_match = match_score

                score = item.get_score()
                scored_items.append((score, item))

            # Sort by score descending
            scored_items.sort(key=lambda x: x[0], reverse=True)

            # Touch and return top-k
            results = []
            for score, item in scored_items[:top_k]:
                item.touch()
                results.append(item)

            return results

    def search_by_type(self, content_type: str) -> List[WorkingMemoryItem]:
        """Search items by content type"""
        with self._lock:
            return [
                item for item in self._items.values()
                if item.content_type == content_type
            ]

    def delete(self, item_id: str) -> bool:
        """Delete item from memory"""
        with self._lock:
            if item_id in self._items:
                item = self._items.pop(item_id)
                self.current_size_bytes -= item.size_bytes
                return True
            return False

    def clear(self, keep_pinned: bool = True):
        """Clear all items (optionally keeping pinned)"""
        with self._lock:
            if keep_pinned:
                pinned = {
                    k: v for k, v in self._items.items()
                    if v.priority == ItemPriority.PINNED
                }
                self._items = pinned
                self.current_size_bytes = sum(i.size_bytes for i in pinned.values())
            else:
                self._items.clear()
                self.current_size_bytes = 0

    def get_pressure(self) -> MemoryPressure:
        """Get current memory pressure level"""
        utilization = self.current_size_bytes / self.max_size_bytes

        if utilization < 0.5:
            return MemoryPressure.LOW
        elif utilization < 0.75:
            return MemoryPressure.MEDIUM
        elif utilization < 0.9:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL

    def _estimate_size(self, content: Any) -> int:
        """Estimate memory size of content"""
        if isinstance(content, bytes):
            return len(content)
        elif isinstance(content, str):
            return len(content.encode('utf-8'))
        elif isinstance(content, (list, tuple)):
            return sum(self._estimate_size(item) for item in content) + 64
        elif isinstance(content, dict):
            return sum(
                self._estimate_size(k) + self._estimate_size(v)
                for k, v in content.items()
            ) + 64
        else:
            return sys.getsizeof(content)

    def _compute_context_match(self, item: WorkingMemoryItem,
                               context: Dict[str, Any]) -> float:
        """Compute how well item matches current context"""
        if not context:
            return 0.5

        match_score = 0.0
        total_weight = 0.0

        # Check metadata matches
        for key, value in context.items():
            total_weight += 1.0
            if key in item.metadata:
                if item.metadata[key] == value:
                    match_score += 1.0
                elif isinstance(value, str) and isinstance(item.metadata[key], str):
                    # Partial string match
                    if value.lower() in item.metadata[key].lower():
                        match_score += 0.5

        if total_weight == 0:
            return 0.5

        return match_score / total_weight

    def _compact(self, needed_bytes: int):
        """Compact memory by removing low-priority items"""
        with self._lock:
            # Get items sorted by score (lowest first)
            scored_items = [
                (item.get_score(), item_id, item)
                for item_id, item in self._items.items()
                if item.priority != ItemPriority.PINNED
            ]
            heapq.heapify(scored_items)

            freed = 0
            evicted = []

            while scored_items and freed < needed_bytes:
                score, item_id, item = heapq.heappop(scored_items)
                freed += item.size_bytes
                evicted.append(item_id)

            # Remove evicted items
            for item_id in evicted:
                del self._items[item_id]
                self.stats["evictions"] += 1

            self.current_size_bytes -= freed
            self.stats["compactions"] += 1

            logger.debug(f"Compacted: freed {freed/1024:.1f} KB, evicted {len(evicted)} items")

    def _evict(self, needed_bytes: int):
        """Evict items to free space (alias for compact)"""
        self._compact(needed_bytes)

    def start_maintenance(self, interval: float = 60.0):
        """Start background maintenance thread"""
        if self._running:
            return

        self._running = True

        def maintenance_loop():
            while self._running:
                try:
                    self._run_maintenance()
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
                time.sleep(interval)

        self._maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        self._maintenance_thread.start()

    def stop_maintenance(self):
        """Stop maintenance thread"""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)

    def _run_maintenance(self):
        """Run periodic maintenance"""
        with self._lock:
            # Decay all attention weights
            for item in self._items.values():
                item.decay(self.decay_rate)

            # Check pressure and compact if needed
            pressure = self.get_pressure()
            if pressure == MemoryPressure.CRITICAL:
                target_free = self.max_size_bytes * 0.2  # Free 20%
                self._compact(int(target_free))
            elif pressure == MemoryPressure.HIGH:
                target_free = self.max_size_bytes * 0.1  # Free 10%
                self._compact(int(target_free))

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        with self._lock:
            return {
                **self.stats,
                "max_size_bytes": self.max_size_bytes,
                "current_size_bytes": self.current_size_bytes,
                "utilization": self.current_size_bytes / self.max_size_bytes,
                "item_count": len(self._items),
                "pressure": self.get_pressure().name,
                "hardware": {
                    "total_ram_gb": self.hardware.total_ram_bytes / 1024**3,
                    "available_ram_gb": self.hardware.available_ram_bytes / 1024**3,
                    "has_gpu": self.hardware.has_gpu,
                },
            }

    def get_items_for_consolidation(self,
                                    min_access_count: int = 3,
                                    min_score: float = 0.5) -> List[WorkingMemoryItem]:
        """
        Get items that should be consolidated to episodic memory

        Items that have been frequently accessed and have high scores
        should be promoted to longer-term storage.
        """
        with self._lock:
            candidates = []

            for item in self._items.values():
                if (item.access_count >= min_access_count and
                    item.get_score() >= min_score):
                    candidates.append(item)

            return candidates

    def export_for_sync(self) -> Dict:
        """Export state for synchronization with other nodes"""
        with self._lock:
            return {
                "items": {
                    item_id: {
                        "content": item.content,
                        "content_type": item.content_type,
                        "priority": item.priority.name,
                        "attention_score": item.get_score(),
                        "access_count": item.access_count,
                        "metadata": item.metadata,
                    }
                    for item_id, item in self._items.items()
                },
                "stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def import_from_sync(self, data: Dict, merge: bool = True):
        """Import synchronized state from another node"""
        with self._lock:
            for item_id, item_data in data.get("items", {}).items():
                if merge and item_id in self._items:
                    # Update attention if remote has higher score
                    continue

                # Import new item
                self.store(
                    key=item_id,
                    content=item_data["content"],
                    content_type=item_data["content_type"],
                    priority=ItemPriority[item_data["priority"]],
                    metadata=item_data["metadata"],
                )

    # =========================================================================
    # MEMSHADOW Protocol Integration (v2 32-byte header)
    # =========================================================================

    def get_modified_since(self, timestamp_ns: int) -> List[Dict]:
        """
        Get items modified since given nanosecond timestamp

        Used by MemorySyncManager to create delta sync batches.

        Args:
            timestamp_ns: Nanosecond timestamp

        Returns:
            List of modified item dictionaries
        """
        with self._lock:
            modified = []

            for item_id, item in self._items.items():
                # Convert item timestamp to nanoseconds
                item_ts_ns = int(item.last_accessed.timestamp() * 1e9)

                if item_ts_ns > timestamp_ns:
                    modified.append({
                        "item_id": item_id,
                        "timestamp_ns": item_ts_ns,
                        "content": item.content,
                        "content_type": item.content_type,
                        "priority": item.priority.value,
                        "metadata": item.metadata,
                        "operation": 2,  # SyncOperation.UPDATE
                    })

            return modified

    def get_by_id(self, item_id: str) -> Optional[Dict]:
        """
        Get item by ID as dictionary (for sync protocol)

        Args:
            item_id: Item identifier

        Returns:
            Item as dictionary or None
        """
        with self._lock:
            item = self._items.get(item_id)
            if item:
                return {
                    "item_id": item_id,
                    "timestamp_ns": int(item.last_accessed.timestamp() * 1e9),
                    "content": item.content,
                    "content_type": item.content_type,
                    "priority": item.priority.value,
                    "metadata": item.metadata,
                    "source_node": item.metadata.get("source_node", ""),
                }
            return None

    def add(self, item_id: str, content: Dict):
        """
        Add item from sync operation (compatible with sync protocol)

        Args:
            item_id: Item identifier
            content: Content dictionary
        """
        self.store(
            key=item_id,
            content=content.get("content", content),
            content_type=content.get("content_type", "structured"),
            priority=ItemPriority(content.get("priority", ItemPriority.NORMAL.value)),
            metadata=content.get("metadata", {}),
        )

    def update(self, item_id: str, content: Dict):
        """
        Update existing item (for sync protocol)

        Args:
            item_id: Item identifier
            content: New content dictionary
        """
        with self._lock:
            if item_id in self._items:
                item = self._items[item_id]
                item.content = content.get("content", content)
                if "metadata" in content:
                    item.metadata.update(content["metadata"])
                item.touch()
            else:
                # Item doesn't exist, add it
                self.add(item_id, content)

    def delete(self, item_id: str) -> bool:
        """
        Delete item by ID (for sync protocol)

        Args:
            item_id: Item identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if item_id in self._items:
                item = self._items.pop(item_id)
                self.current_size_bytes -= item.size_bytes
                return True
            return False

    def merge(self, item_id: str, content: Dict):
        """
        Merge content with existing item (for sync protocol)

        Args:
            item_id: Item identifier
            content: Content to merge
        """
        with self._lock:
            if item_id in self._items:
                item = self._items[item_id]
                # Deep merge content
                if isinstance(item.content, dict) and isinstance(content.get("content"), dict):
                    item.content.update(content["content"])
                else:
                    item.content = content.get("content", content)
                if "metadata" in content:
                    item.metadata.update(content["metadata"])
                item.touch()
            else:
                self.add(item_id, content)


if __name__ == "__main__":
    print("Working Memory Self-Test")
    print("=" * 50)

    wm = WorkingMemory()

    print(f"\n[1] Hardware Detection")
    stats = wm.get_stats()
    print(f"    Max size: {stats['max_size_bytes']/1024**2:.1f} MB")
    print(f"    RAM: {stats['hardware']['total_ram_gb']:.1f} GB total")
    print(f"    GPU: {stats['hardware']['has_gpu']}")

    print(f"\n[2] Store Items")
    items = []
    for i in range(100):
        item_id = wm.store(
            f"test-{i}",
            {"data": f"Test data {i}" * 100, "index": i},
            priority=ItemPriority.NORMAL if i < 90 else ItemPriority.HIGH,
            metadata={"category": "test", "index": i},
        )
        items.append(item_id)
    print(f"    Stored 100 items")
    print(f"    Size: {wm.current_size_bytes/1024:.1f} KB")

    print(f"\n[3] Retrieve Item")
    item = wm.retrieve(items[0])
    print(f"    Retrieved: {item.item_id if item else 'None'}")
    print(f"    Access count: {item.access_count if item else 0}")

    print(f"\n[4] Context Search")
    results = wm.search_by_context({"category": "test", "index": 50}, top_k=5)
    print(f"    Found {len(results)} items")
    for r in results[:3]:
        print(f"      - {r.item_id}: score={r.get_score():.3f}")

    print(f"\n[5] Memory Pressure")
    print(f"    Pressure: {wm.get_pressure().name}")
    print(f"    Utilization: {stats['utilization']*100:.1f}%")

    print(f"\n[6] Consolidation Candidates")
    # Access some items multiple times
    for _ in range(5):
        wm.retrieve(items[0])
        wm.retrieve(items[1])

    candidates = wm.get_items_for_consolidation(min_access_count=3)
    print(f"    Candidates: {len(candidates)}")

    print(f"\n[7] Statistics")
    stats = wm.get_stats()
    for key in ["stores", "retrievals", "hits", "misses", "evictions"]:
        print(f"    {key}: {stats[key]}")

    print("\n" + "=" * 50)
    print("Working Memory test complete")

