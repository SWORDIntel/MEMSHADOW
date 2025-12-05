"""
Working Memory (L1 Tier)

Fast, limited-capacity working memory for active data.
Optimized for speed with RAMDISK storage and 256-dimension embeddings.

Based on: HUB_DOCS/MEMSHADOW_INTEGRATION.md
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class WorkingMemoryItem:
    """Item in working memory"""
    item_id: str = field(default_factory=lambda: str(uuid4()))
    data: bytes = b""
    embedding: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time() * 1e9))
    accessed_at: int = field(default_factory=lambda: int(time.time() * 1e9))
    access_count: int = 0
    priority: int = 0


class WorkingMemory:
    """
    L1 Working Memory Tier
    
    Characteristics:
    - Limited capacity (default 1000 items)
    - LRU eviction policy
    - Fast access (< 1ms target)
    - 256-dimension embeddings (compressed)
    - RAMDISK storage backing
    
    Implements MEMSHADOW sync interface.
    """
    
    TIER_NAME = "L1_WORKING"
    MAX_DIMENSION = 256
    
    def __init__(
        self,
        capacity: int = 1000,
        ramdisk_path: Optional[str] = None,
    ):
        self.capacity = capacity
        self.ramdisk_path = ramdisk_path
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, WorkingMemoryItem] = OrderedDict()
        
        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "stores": 0,
        }
        
        logger.info(
            "WorkingMemory initialized",
            capacity=capacity,
            ramdisk=ramdisk_path,
        )
    
    async def store(
        self,
        item_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[bytes] = None,
        priority: int = 0,
    ) -> bool:
        """
        Store item in working memory.
        
        Will evict LRU items if at capacity.
        """
        # Evict if at capacity
        while len(self._cache) >= self.capacity:
            self._evict_lru()
        
        item = WorkingMemoryItem(
            item_id=item_id,
            data=data,
            embedding=embedding,
            metadata=metadata or {},
            priority=priority,
        )
        
        self._cache[item_id] = item
        self._cache.move_to_end(item_id)  # Mark as recently used
        
        self._stats["stores"] += 1
        
        logger.debug("Stored in L1", item_id=item_id, size=len(data))
        return True
    
    async def retrieve(self, item_id: str) -> Optional[bytes]:
        """Retrieve item from working memory"""
        item = self._cache.get(item_id)
        
        if item is None:
            self._stats["misses"] += 1
            return None
        
        # Update access info
        item.accessed_at = int(time.time() * 1e9)
        item.access_count += 1
        self._cache.move_to_end(item_id)  # Mark as recently used
        
        self._stats["hits"] += 1
        return item.data
    
    async def delete(self, item_id: str) -> bool:
        """Delete item from working memory"""
        if item_id in self._cache:
            del self._cache[item_id]
            return True
        return False
    
    async def list_items(self, since_timestamp: Optional[int] = None) -> List[str]:
        """List item IDs, optionally filtered by timestamp"""
        if since_timestamp is None:
            return list(self._cache.keys())
        return [
            k for k, v in self._cache.items()
            if v.created_at >= since_timestamp
        ]
    
    async def get_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item"""
        item = self._cache.get(item_id)
        return item.metadata if item else None
    
    async def get_item(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Get full item object"""
        return self._cache.get(item_id)
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self._cache:
            return
        
        # Get oldest item (first in OrderedDict)
        oldest_id = next(iter(self._cache))
        del self._cache[oldest_id]
        
        self._stats["evictions"] += 1
        logger.debug("Evicted from L1", item_id=oldest_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory tier statistics"""
        return {
            "tier": self.TIER_NAME,
            "capacity": self.capacity,
            "current_size": len(self._cache),
            "utilization": len(self._cache) / self.capacity,
            "hit_rate": self._stats["hits"] / max(self._stats["hits"] + self._stats["misses"], 1),
            **self._stats,
        }
    
    def clear(self):
        """Clear all items from working memory"""
        self._cache.clear()
        logger.info("Working memory cleared")


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "WorkingMemoryItem",
    "WorkingMemory",
]
