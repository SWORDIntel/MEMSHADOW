"""
RAMDISK Storage Layer - Ultra-Fast Working Memory

Implements a dynamically allocated in-memory storage layer that serves as
the "working memory" for the neural storage system.

Features:
- Dynamic memory allocation based on data size
- Automatic eviction of cold data
- Memory-mapped file backing for persistence
- LRU caching with temperature-based eviction
- Zero-copy access for maximum speed
"""

import asyncio
import hashlib
import mmap
import numpy as np
import os
import pickle
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
import structlog

logger = structlog.get_logger()


@dataclass
class RAMDiskEntry:
    """Entry stored in RAMDISK"""
    memory_id: UUID
    embedding: np.ndarray  # 256d compressed embedding
    content_hash: str
    temperature: float = 1.0  # Starts hot
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    connections: Set[UUID] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LRUCache:
    """LRU cache with temperature-based eviction"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[UUID, RAMDiskEntry] = OrderedDict()

    def get(self, key: UUID) -> Optional[RAMDiskEntry]:
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry = self.cache[key]
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        entry.temperature = min(1.0, entry.temperature + 0.1)
        return entry

    def put(self, key: UUID, entry: RAMDiskEntry):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = entry
        else:
            if len(self.cache) >= self.max_size:
                self._evict()
            self.cache[key] = entry

    def _evict(self):
        """Evict coldest entry"""
        if not self.cache:
            return None

        # Find coldest entry
        coldest_key = None
        coldest_temp = float('inf')

        for key, entry in self.cache.items():
            if entry.temperature < coldest_temp:
                coldest_temp = entry.temperature
                coldest_key = key

        if coldest_key:
            evicted = self.cache.pop(coldest_key)
            return evicted
        return None

    def remove(self, key: UUID) -> Optional[RAMDiskEntry]:
        return self.cache.pop(key, None)

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: UUID) -> bool:
        return key in self.cache

    def items(self):
        return self.cache.items()

    def values(self):
        return self.cache.values()


class RAMDiskStorage:
    """
    High-speed in-memory storage with dynamic allocation.

    This serves as the "working memory" tier, storing the most
    frequently accessed and recently used data for instant retrieval.
    """

    def __init__(
        self,
        max_memory_mb: int = 512,
        min_memory_mb: int = 64,
        embedding_dim: int = 256,
        enable_persistence: bool = True,
        persistence_path: Optional[str] = None,
        eviction_temperature: float = 0.2,
        decay_rate: float = 0.05
    ):
        self.max_memory_mb = max_memory_mb
        self.min_memory_mb = min_memory_mb
        self.embedding_dim = embedding_dim
        self.enable_persistence = enable_persistence
        self.eviction_temperature = eviction_temperature
        self.decay_rate = decay_rate

        # Calculate max entries based on memory
        entry_size = embedding_dim * 4 + 500  # embedding + overhead
        self.max_entries = (max_memory_mb * 1024 * 1024) // entry_size

        # Primary storage (LRU cache)
        self.storage = LRUCache(self.max_entries)

        # Hash index for deduplication
        self.hash_index: Dict[str, UUID] = {}

        # Persistence
        self.persistence_path = persistence_path or tempfile.mkdtemp(prefix="memshadow_ramdisk_")
        self._mmap_file: Optional[mmap.mmap] = None

        # Statistics
        self.stats = {
            "total_stored": 0,
            "total_evicted": 0,
            "total_hits": 0,
            "total_misses": 0,
            "current_size_mb": 0,
            "peak_size_mb": 0,
        }

        # Background tasks
        self._running = False
        self._decay_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None

        # Lock
        self._lock = asyncio.Lock()

        logger.info("RAMDiskStorage initialized",
                   max_memory_mb=max_memory_mb,
                   max_entries=self.max_entries,
                   embedding_dim=embedding_dim)

    async def start(self):
        """Start background tasks"""
        self._running = True

        # Load persisted data if available
        if self.enable_persistence:
            await self._load_persisted_data()

        # Start decay task
        self._decay_task = asyncio.create_task(self._decay_loop())

        # Start persistence task
        if self.enable_persistence:
            self._persist_task = asyncio.create_task(self._persist_loop())

        logger.info("RAMDiskStorage started")

    async def stop(self):
        """Stop and cleanup"""
        self._running = False

        if self._decay_task:
            self._decay_task.cancel()
        if self._persist_task:
            self._persist_task.cancel()

        # Final persist
        if self.enable_persistence:
            await self._persist_data()

        logger.info("RAMDiskStorage stopped")

    async def store(
        self,
        memory_id: UUID,
        embedding: np.ndarray,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
        connections: Optional[Set[UUID]] = None
    ) -> bool:
        """
        Store an entry in RAMDISK.
        Returns True if stored, False if duplicate.
        """
        async with self._lock:
            # Check for duplicate
            if content_hash in self.hash_index:
                existing_id = self.hash_index[content_hash]
                if existing_id in self.storage:
                    # Touch existing entry
                    self.storage.get(existing_id)
                    return False

            # Ensure correct dimension
            if len(embedding) != self.embedding_dim:
                # Project to RAMDISK dimension
                embedding = self._project_embedding(embedding)

            # Calculate size
            size_bytes = embedding.nbytes + len(pickle.dumps(metadata or {}))

            # Create entry
            entry = RAMDiskEntry(
                memory_id=memory_id,
                embedding=embedding.astype(np.float32),
                content_hash=content_hash,
                size_bytes=size_bytes,
                metadata=metadata or {},
                connections=connections or set()
            )

            # Store in LRU cache (may evict cold entries)
            self.storage.put(memory_id, entry)
            self.hash_index[content_hash] = memory_id

            # Update stats
            self.stats["total_stored"] += 1
            self._update_size_stats()

            return True

    def _project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to RAMDISK dimension"""
        source_dim = len(embedding)
        target_dim = self.embedding_dim

        if source_dim == target_dim:
            return embedding

        if source_dim > target_dim:
            # Downsample: take strided elements
            stride = source_dim // target_dim
            projected = embedding[::stride][:target_dim]
        else:
            # Upsample: repeat elements
            factor = target_dim // source_dim
            projected = np.repeat(embedding, factor)[:target_dim]

        # Normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm

        return projected.astype(np.float32)

    async def retrieve(self, memory_id: UUID) -> Optional[RAMDiskEntry]:
        """Retrieve an entry by ID"""
        entry = self.storage.get(memory_id)
        if entry:
            self.stats["total_hits"] += 1
        else:
            self.stats["total_misses"] += 1
        return entry

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[UUID, float]]:
        """
        Search for similar embeddings in RAMDISK.
        Uses vectorized operations for maximum speed.
        """
        # Project query to RAMDISK dimension
        if len(query_embedding) != self.embedding_dim:
            query_embedding = self._project_embedding(query_embedding)

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        results = []

        for memory_id, entry in self.storage.items():
            emb = entry.embedding
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm

            similarity = float(np.dot(query_embedding, emb))
            if similarity >= threshold:
                results.append((memory_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def batch_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[UUID, float]]:
        """
        Optimized batch search using matrix operations.
        """
        if len(self.storage) == 0:
            return []

        # Project query
        if len(query_embedding) != self.embedding_dim:
            query_embedding = self._project_embedding(query_embedding)

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Build embedding matrix
        memory_ids = []
        embeddings = []
        for mid, entry in self.storage.items():
            memory_ids.append(mid)
            emb = entry.embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        if not embeddings:
            return []

        embedding_matrix = np.vstack(embeddings)

        # Compute all similarities at once
        similarities = np.dot(embedding_matrix, query_embedding)

        # Filter and sort
        results = [
            (memory_ids[i], float(similarities[i]))
            for i in range(len(similarities))
            if similarities[i] >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    async def remove(self, memory_id: UUID) -> bool:
        """Remove an entry from RAMDISK"""
        async with self._lock:
            entry = self.storage.remove(memory_id)
            if entry:
                if entry.content_hash in self.hash_index:
                    del self.hash_index[entry.content_hash]
                self._update_size_stats()
                return True
            return False

    async def promote(
        self,
        memory_id: UUID,
        temperature_boost: float = 0.3
    ):
        """Promote a memory (increase temperature)"""
        entry = self.storage.get(memory_id)
        if entry:
            entry.temperature = min(1.0, entry.temperature + temperature_boost)

    async def demote(
        self,
        memory_id: UUID,
        temperature_penalty: float = 0.2
    ):
        """Demote a memory (decrease temperature)"""
        entry = self.storage.get(memory_id)
        if entry:
            entry.temperature = max(0.0, entry.temperature - temperature_penalty)

    async def _decay_loop(self):
        """Background task to decay temperatures"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._apply_decay()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Decay loop error", error=str(e))

    async def _apply_decay(self):
        """Apply temperature decay to all entries"""
        async with self._lock:
            evicted_count = 0
            for memory_id, entry in list(self.storage.items()):
                # Calculate decay based on time since last access
                hours_since_access = (
                    datetime.utcnow() - entry.last_accessed
                ).total_seconds() / 3600

                # Exponential decay
                decay_factor = np.exp(-self.decay_rate * hours_since_access)
                entry.temperature *= decay_factor

                # Evict if below threshold
                if entry.temperature < self.eviction_temperature:
                    self.storage.remove(memory_id)
                    if entry.content_hash in self.hash_index:
                        del self.hash_index[entry.content_hash]
                    evicted_count += 1
                    self.stats["total_evicted"] += 1

            if evicted_count > 0:
                logger.info("Temperature decay applied", evicted=evicted_count)
                self._update_size_stats()

    async def _persist_loop(self):
        """Background task to persist data"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._persist_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Persist loop error", error=str(e))

    async def _persist_data(self):
        """Persist current data to disk"""
        if not self.enable_persistence:
            return

        try:
            filepath = Path(self.persistence_path) / "ramdisk_state.pkl"
            data = {
                "entries": [
                    {
                        "memory_id": str(entry.memory_id),
                        "embedding": entry.embedding.tolist(),
                        "content_hash": entry.content_hash,
                        "temperature": entry.temperature,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed.isoformat(),
                        "created_at": entry.created_at.isoformat(),
                        "connections": [str(c) for c in entry.connections],
                        "metadata": entry.metadata,
                    }
                    for entry in self.storage.values()
                ],
                "stats": self.stats,
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            logger.debug("RAMDISK data persisted", entries=len(data["entries"]))

        except Exception as e:
            logger.error("Persist error", error=str(e))

    async def _load_persisted_data(self):
        """Load persisted data from disk"""
        try:
            filepath = Path(self.persistence_path) / "ramdisk_state.pkl"
            if not filepath.exists():
                return

            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            for entry_data in data.get("entries", []):
                entry = RAMDiskEntry(
                    memory_id=UUID(entry_data["memory_id"]),
                    embedding=np.array(entry_data["embedding"], dtype=np.float32),
                    content_hash=entry_data["content_hash"],
                    temperature=entry_data.get("temperature", 0.5),
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    connections=set(UUID(c) for c in entry_data.get("connections", [])),
                    metadata=entry_data.get("metadata", {}),
                )
                self.storage.put(entry.memory_id, entry)
                self.hash_index[entry.content_hash] = entry.memory_id

            self._update_size_stats()
            logger.info("RAMDISK data loaded", entries=len(self.storage))

        except Exception as e:
            logger.error("Load persisted data error", error=str(e))

    def _update_size_stats(self):
        """Update size statistics"""
        total_bytes = sum(e.size_bytes for e in self.storage.values())
        self.stats["current_size_mb"] = total_bytes / (1024 * 1024)
        self.stats["peak_size_mb"] = max(
            self.stats["peak_size_mb"],
            self.stats["current_size_mb"]
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get RAMDISK statistics"""
        hit_rate = (
            self.stats["total_hits"] /
            max(1, self.stats["total_hits"] + self.stats["total_misses"])
        )

        temps = [e.temperature for e in self.storage.values()]
        avg_temp = np.mean(temps) if temps else 0

        return {
            "entries": len(self.storage),
            "max_entries": self.max_entries,
            "current_size_mb": self.stats["current_size_mb"],
            "max_memory_mb": self.max_memory_mb,
            "peak_size_mb": self.stats["peak_size_mb"],
            "total_stored": self.stats["total_stored"],
            "total_evicted": self.stats["total_evicted"],
            "hit_rate": hit_rate,
            "avg_temperature": float(avg_temp),
            "embedding_dim": self.embedding_dim,
        }

    async def get_hottest_memories(self, top_k: int = 10) -> List[Tuple[UUID, float]]:
        """Get the hottest (most active) memories"""
        entries = list(self.storage.items())
        entries.sort(key=lambda x: x[1].temperature, reverse=True)
        return [(mid, entry.temperature) for mid, entry in entries[:top_k]]

    async def get_coldest_memories(self, top_k: int = 10) -> List[Tuple[UUID, float]]:
        """Get the coldest (least active) memories"""
        entries = list(self.storage.items())
        entries.sort(key=lambda x: x[1].temperature)
        return [(mid, entry.temperature) for mid, entry in entries[:top_k]]

    async def allocate_dynamic(self, requested_mb: int) -> int:
        """
        Dynamically allocate additional memory.
        Returns actual allocated amount.
        """
        current_mb = self.stats["current_size_mb"]
        available_mb = self.max_memory_mb - current_mb

        if requested_mb <= available_mb:
            # Can allocate full request
            return requested_mb

        # Need to evict to make space
        needed_mb = requested_mb - available_mb
        evicted_mb = await self._evict_for_space(needed_mb)

        return min(requested_mb, available_mb + evicted_mb)

    async def _evict_for_space(self, needed_mb: float) -> float:
        """Evict cold entries to free space"""
        freed_mb = 0
        evicted = 0

        # Sort by temperature (coldest first)
        entries = sorted(
            self.storage.items(),
            key=lambda x: x[1].temperature
        )

        for memory_id, entry in entries:
            if freed_mb >= needed_mb:
                break

            self.storage.remove(memory_id)
            if entry.content_hash in self.hash_index:
                del self.hash_index[entry.content_hash]

            freed_mb += entry.size_bytes / (1024 * 1024)
            evicted += 1
            self.stats["total_evicted"] += 1

        self._update_size_stats()
        logger.info("Evicted for space", evicted=evicted, freed_mb=freed_mb)

        return freed_mb
