"""
Episodic Memory (L2 Tier)

Session and episode storage with full-resolution embeddings.
Stored on NVMe for balance of speed and capacity.

Based on: HUB_DOCS/MEMSHADOW_INTEGRATION.md
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class Episode:
    """A coherent episode/session"""
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    start_time: int = field(default_factory=lambda: int(time.time() * 1e9))
    end_time: Optional[int] = None
    items: Dict[str, bytes] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[bytes] = None  # Episode-level embedding
    
    def add_item(self, item_id: str, data: bytes):
        self.items[item_id] = data
    
    def close(self):
        self.end_time = int(time.time() * 1e9)
    
    @property
    def is_open(self) -> bool:
        return self.end_time is None
    
    @property
    def duration_sec(self) -> float:
        end = self.end_time or int(time.time() * 1e9)
        return (end - self.start_time) / 1e9


class EpisodicMemory:
    """
    L2 Episodic Memory Tier
    
    Characteristics:
    - Episode/session-based organization
    - Full 2048-dimension embeddings
    - NVMe storage backing
    - Automatic episode segmentation
    - Temporal indexing
    
    Implements MEMSHADOW sync interface.
    """
    
    TIER_NAME = "L2_EPISODIC"
    MAX_DIMENSION = 2048
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        episode_timeout_sec: float = 300.0,  # 5 minutes
        max_episodes: int = 10000,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.episode_timeout_sec = episode_timeout_sec
        self.max_episodes = max_episodes
        
        # Episodes: episode_id -> Episode
        self._episodes: Dict[str, Episode] = {}
        
        # Current active episode per session
        self._active_episodes: Dict[str, str] = {}  # session_id -> episode_id
        
        # Item index: item_id -> episode_id
        self._item_index: Dict[str, str] = {}
        
        # Stats
        self._stats = {
            "episodes_created": 0,
            "episodes_closed": 0,
            "items_stored": 0,
            "items_retrieved": 0,
        }
        
        logger.info(
            "EpisodicMemory initialized",
            storage=storage_path,
            timeout_sec=episode_timeout_sec,
        )
    
    async def store(
        self,
        item_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        embedding: Optional[bytes] = None,
    ) -> bool:
        """
        Store item in episodic memory.
        
        Items are organized into episodes by session.
        """
        session_id = session_id or "default"
        
        # Get or create active episode
        episode = await self._get_or_create_episode(session_id)
        
        # Store item in episode
        episode.add_item(item_id, data)
        if metadata:
            episode.metadata[item_id] = metadata
        
        # Update index
        self._item_index[item_id] = episode.episode_id
        
        self._stats["items_stored"] += 1
        
        logger.debug(
            "Stored in L2",
            item_id=item_id,
            episode=episode.episode_id,
            size=len(data),
        )
        
        return True
    
    async def retrieve(self, item_id: str) -> Optional[bytes]:
        """Retrieve item from episodic memory"""
        episode_id = self._item_index.get(item_id)
        if not episode_id:
            return None
        
        episode = self._episodes.get(episode_id)
        if not episode:
            return None
        
        data = episode.items.get(item_id)
        if data:
            self._stats["items_retrieved"] += 1
        
        return data
    
    async def delete(self, item_id: str) -> bool:
        """Delete item from episodic memory"""
        episode_id = self._item_index.pop(item_id, None)
        if not episode_id:
            return False
        
        episode = self._episodes.get(episode_id)
        if episode and item_id in episode.items:
            del episode.items[item_id]
            return True
        
        return False
    
    async def list_items(self, since_timestamp: Optional[int] = None) -> List[str]:
        """List all item IDs"""
        if since_timestamp is None:
            return list(self._item_index.keys())
        
        result = []
        for episode in self._episodes.values():
            if episode.start_time >= since_timestamp:
                result.extend(episode.items.keys())
        return result
    
    async def get_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item"""
        episode_id = self._item_index.get(item_id)
        if not episode_id:
            return None
        
        episode = self._episodes.get(episode_id)
        if not episode:
            return None
        
        return episode.metadata.get(item_id)
    
    async def _get_or_create_episode(self, session_id: str) -> Episode:
        """Get active episode for session or create new one"""
        episode_id = self._active_episodes.get(session_id)
        
        if episode_id:
            episode = self._episodes.get(episode_id)
            if episode and episode.is_open:
                # Check if episode has timed out
                elapsed = (time.time() * 1e9 - episode.start_time) / 1e9
                if elapsed < self.episode_timeout_sec:
                    return episode
                # Close timed-out episode
                episode.close()
                self._stats["episodes_closed"] += 1
        
        # Create new episode
        episode = Episode(session_id=session_id)
        self._episodes[episode.episode_id] = episode
        self._active_episodes[session_id] = episode.episode_id
        
        self._stats["episodes_created"] += 1
        
        # Evict old episodes if at capacity
        while len(self._episodes) > self.max_episodes:
            self._evict_oldest_episode()
        
        logger.debug("Episode created", episode_id=episode.episode_id, session=session_id)
        
        return episode
    
    def _evict_oldest_episode(self):
        """Evict the oldest closed episode"""
        oldest = None
        oldest_time = float("inf")
        
        for ep_id, ep in self._episodes.items():
            if not ep.is_open and ep.start_time < oldest_time:
                oldest = ep_id
                oldest_time = ep.start_time
        
        if oldest:
            episode = self._episodes.pop(oldest)
            # Remove from item index
            for item_id in episode.items.keys():
                self._item_index.pop(item_id, None)
            logger.debug("Episode evicted", episode_id=oldest)
    
    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode"""
        return self._episodes.get(episode_id)
    
    async def get_session_episodes(self, session_id: str) -> List[Episode]:
        """Get all episodes for a session"""
        return [
            ep for ep in self._episodes.values()
            if ep.session_id == session_id
        ]
    
    async def close_episode(self, episode_id: str):
        """Close an episode"""
        episode = self._episodes.get(episode_id)
        if episode and episode.is_open:
            episode.close()
            self._stats["episodes_closed"] += 1
            logger.debug("Episode closed", episode_id=episode_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory tier statistics"""
        open_episodes = sum(1 for ep in self._episodes.values() if ep.is_open)
        
        return {
            "tier": self.TIER_NAME,
            "max_episodes": self.max_episodes,
            "current_episodes": len(self._episodes),
            "open_episodes": open_episodes,
            "total_items": len(self._item_index),
            **self._stats,
        }
    
    def clear(self):
        """Clear all episodes"""
        self._episodes.clear()
        self._active_episodes.clear()
        self._item_index.clear()
        logger.info("Episodic memory cleared")


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "Episode",
    "EpisodicMemory",
]
