"""
Multi-Tiered Dimensional Database Manager

Implements a hierarchical storage system with different dimensional embeddings:
    Tier 0: Ultra-High Dimensional (4096d) - Maximum semantic fidelity
    Tier 1: High Dimensional (2048d) - Standard production storage
    Tier 2: Medium Dimensional (1024d) - Warm storage with good quality
    Tier 3: Low Dimensional (512d) - Hot compressed storage
    Tier 4: RAMDISK (256d) - Ultra-fast working memory

Each tier maintains embeddings at different resolutions, allowing for:
- Fast approximate searches at lower tiers
- High-fidelity retrieval from higher tiers
- Intelligent promotion/demotion between tiers
"""

import asyncio
import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import structlog

logger = structlog.get_logger()


class StorageTier(IntEnum):
    """Storage tier enumeration with dimensional mapping"""
    ULTRA_HIGH = 0  # 4096d - Archival
    HIGH = 1        # 2048d - Long-term
    MEDIUM = 2      # 1024d - Warm
    LOW = 3         # 512d  - Hot compressed
    RAMDISK = 4     # 256d  - Working memory

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for this tier"""
        dims = {0: 4096, 1: 2048, 2: 1024, 3: 512, 4: 256}
        return dims[self.value]

    @property
    def description(self) -> str:
        """Human-readable tier description"""
        descs = {
            0: "Ultra-High Dimensional (Archival)",
            1: "High Dimensional (Long-term)",
            2: "Medium Dimensional (Warm)",
            3: "Low Dimensional (Hot)",
            4: "RAMDISK (Working Memory)"
        }
        return descs[self.value]


@dataclass
class TieredMemoryRecord:
    """Memory record stored across tiers"""
    memory_id: UUID
    content_hash: str
    embeddings: Dict[StorageTier, np.ndarray] = field(default_factory=dict)
    current_tier: StorageTier = StorageTier.HIGH
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    temperature: float = 0.5  # 0.0 = cold, 1.0 = hot
    connections: List[UUID] = field(default_factory=list)
    connection_weights: Dict[UUID, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DimensionalProjector:
    """
    Projects embeddings between different dimensional spaces.
    Uses learned projection matrices for quality preservation.
    """

    def __init__(self):
        self.projection_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        self._initialize_projections()

    def _initialize_projections(self):
        """Initialize random orthogonal projection matrices"""
        # Generate projection matrices between all tier dimensions
        dimensions = [4096, 2048, 1024, 512, 256]

        for i, from_dim in enumerate(dimensions):
            for to_dim in dimensions[i+1:]:
                # Create semi-orthogonal projection matrix
                matrix = self._create_projection_matrix(from_dim, to_dim)
                self.projection_matrices[(from_dim, to_dim)] = matrix

                # Also create reverse projection (with pseudo-inverse)
                reverse = np.linalg.pinv(matrix)
                self.projection_matrices[(to_dim, from_dim)] = reverse

    def _create_projection_matrix(self, from_dim: int, to_dim: int) -> np.ndarray:
        """Create a quality-preserving projection matrix"""
        # Use random orthogonal projection (preserves distances better)
        random_matrix = np.random.randn(to_dim, from_dim)
        # Orthogonalize using QR decomposition
        q, _ = np.linalg.qr(random_matrix.T)
        return q.T[:to_dim, :]

    def project(self, embedding: np.ndarray, from_dim: int, to_dim: int) -> np.ndarray:
        """Project embedding to target dimension"""
        if from_dim == to_dim:
            return embedding.copy()

        key = (from_dim, to_dim)
        if key not in self.projection_matrices:
            # Generate on-the-fly if needed
            if from_dim > to_dim:
                matrix = self._create_projection_matrix(from_dim, to_dim)
            else:
                matrix = np.linalg.pinv(self._create_projection_matrix(to_dim, from_dim))
            self.projection_matrices[key] = matrix

        projected = np.dot(self.projection_matrices[key], embedding)
        # Normalize to unit vector
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected


class TieredDatabaseManager:
    """
    Manages multi-tiered storage with intelligent data placement.

    Features:
    - Automatic tier selection based on access patterns
    - Lazy projection to lower dimensions
    - Connection tracking across tiers
    - Unified search across all tiers
    """

    def __init__(
        self,
        enable_ultra_high: bool = True,
        max_ramdisk_mb: int = 512,
        auto_migrate: bool = True
    ):
        self.enable_ultra_high = enable_ultra_high
        self.max_ramdisk_mb = max_ramdisk_mb
        self.auto_migrate = auto_migrate

        # Storage indexes per tier
        self.tier_indexes: Dict[StorageTier, Dict[UUID, TieredMemoryRecord]] = {
            tier: {} for tier in StorageTier
        }

        # Global index for fast lookup
        self.global_index: Dict[UUID, TieredMemoryRecord] = {}

        # Content hash index for deduplication
        self.hash_index: Dict[str, UUID] = {}

        # Projector for dimensional transformations
        self.projector = DimensionalProjector()

        # Statistics
        self.stats = {
            "total_memories": 0,
            "tier_counts": {tier: 0 for tier in StorageTier},
            "tier_bytes": {tier: 0 for tier in StorageTier},
            "projections_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info("TieredDatabaseManager initialized",
                   enable_ultra_high=enable_ultra_high,
                   max_ramdisk_mb=max_ramdisk_mb)

    async def store(
        self,
        memory_id: UUID,
        content_hash: str,
        embedding: np.ndarray,
        initial_tier: StorageTier = StorageTier.HIGH,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TieredMemoryRecord:
        """
        Store a memory with embedding at specified tier.
        Automatically generates projections for faster tiers.
        """
        async with self._lock:
            # Check for duplicate by hash
            if content_hash in self.hash_index:
                existing_id = self.hash_index[content_hash]
                existing = self.global_index.get(existing_id)
                if existing:
                    logger.debug("Duplicate detected, returning existing",
                               memory_id=str(memory_id),
                               existing_id=str(existing_id))
                    return existing

            # Ensure embedding is proper numpy array
            embedding = np.asarray(embedding, dtype=np.float32)
            source_dim = len(embedding)

            # Create record
            record = TieredMemoryRecord(
                memory_id=memory_id,
                content_hash=content_hash,
                current_tier=initial_tier,
                metadata=metadata or {}
            )

            # Store embedding at initial tier
            target_dim = initial_tier.dimensions
            if source_dim != target_dim:
                embedding = self.projector.project(embedding, source_dim, target_dim)
                self.stats["projections_performed"] += 1

            record.embeddings[initial_tier] = embedding

            # Generate projections for faster tiers (lazy pre-computation)
            await self._generate_lower_tier_projections(record, initial_tier)

            # Add to indexes
            self.tier_indexes[initial_tier][memory_id] = record
            self.global_index[memory_id] = record
            self.hash_index[content_hash] = memory_id

            # Update stats
            self.stats["total_memories"] += 1
            self.stats["tier_counts"][initial_tier] += 1
            self.stats["tier_bytes"][initial_tier] += embedding.nbytes

            logger.info("Memory stored in tiered DB",
                       memory_id=str(memory_id),
                       tier=initial_tier.description,
                       embedding_dim=target_dim)

            return record

    async def _generate_lower_tier_projections(
        self,
        record: TieredMemoryRecord,
        source_tier: StorageTier
    ):
        """Generate projections for all tiers below source"""
        source_embedding = record.embeddings.get(source_tier)
        if source_embedding is None:
            return

        source_dim = source_tier.dimensions

        for tier in StorageTier:
            if tier.value > source_tier.value:
                target_dim = tier.dimensions
                projected = self.projector.project(source_embedding, source_dim, target_dim)
                record.embeddings[tier] = projected
                self.stats["projections_performed"] += 1

    async def retrieve(
        self,
        memory_id: UUID,
        preferred_tier: Optional[StorageTier] = None
    ) -> Optional[Tuple[np.ndarray, StorageTier]]:
        """
        Retrieve embedding for a memory.
        Returns embedding from preferred tier if available, else closest available.
        """
        record = self.global_index.get(memory_id)
        if not record:
            self.stats["cache_misses"] += 1
            return None

        self.stats["cache_hits"] += 1

        # Update access tracking
        record.access_count += 1
        record.last_accessed = datetime.utcnow()
        record.temperature = self._calculate_temperature(record)

        # Find best available tier
        if preferred_tier and preferred_tier in record.embeddings:
            return record.embeddings[preferred_tier], preferred_tier

        # Return from current tier or nearest available
        for tier in StorageTier:
            if tier in record.embeddings:
                return record.embeddings[tier], tier

        return None

    def _calculate_temperature(self, record: TieredMemoryRecord) -> float:
        """
        Calculate memory temperature based on access patterns.
        Higher = hotter (more frequently accessed)
        """
        now = datetime.utcnow()
        age_hours = (now - record.last_accessed).total_seconds() / 3600

        # Decay factor: temperature decreases over time without access
        decay = np.exp(-age_hours / 24)  # Half-life of ~24 hours

        # Access frequency boost
        access_boost = min(1.0, record.access_count / 100)

        # Connection boost: well-connected memories stay warmer
        connection_boost = min(0.3, len(record.connections) * 0.05)

        temperature = (0.5 * decay) + (0.3 * access_boost) + (0.2 + connection_boost)
        return min(1.0, max(0.0, temperature))

    async def search_tier(
        self,
        query_embedding: np.ndarray,
        tier: StorageTier,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[UUID, float]]:
        """
        Search within a specific tier using cosine similarity.
        Returns list of (memory_id, similarity_score) tuples.
        """
        query_dim = len(query_embedding)
        tier_dim = tier.dimensions

        # Project query to tier dimension if needed
        if query_dim != tier_dim:
            query_embedding = self.projector.project(query_embedding, query_dim, tier_dim)

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        results = []

        for memory_id, record in self.tier_indexes[tier].items():
            if tier not in record.embeddings:
                continue

            emb = record.embeddings[tier]
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm

            similarity = float(np.dot(query_embedding, emb))

            if similarity >= threshold:
                results.append((memory_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def search_all_tiers(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5,
        start_tier: StorageTier = StorageTier.RAMDISK
    ) -> List[Tuple[UUID, float, StorageTier]]:
        """
        Cascade search from fastest tier (RAMDISK) to slowest (ULTRA_HIGH).
        Returns results from fastest tier that has enough matches.
        """
        query_dim = len(query_embedding)
        all_results = []

        # Search from hot to cold
        for tier in reversed(list(StorageTier)):
            if tier.value < start_tier.value:
                continue

            tier_results = await self.search_tier(
                query_embedding, tier, top_k * 2, threshold * 0.9
            )

            for memory_id, score in tier_results:
                all_results.append((memory_id, score, tier))

        # Deduplicate and sort
        seen = set()
        unique_results = []
        for memory_id, score, tier in sorted(all_results, key=lambda x: x[1], reverse=True):
            if memory_id not in seen:
                seen.add(memory_id)
                unique_results.append((memory_id, score, tier))

        return unique_results[:top_k]

    async def add_connection(
        self,
        source_id: UUID,
        target_id: UUID,
        weight: float = 1.0,
        bidirectional: bool = True
    ):
        """Add a connection between two memories (synaptic link)"""
        async with self._lock:
            source = self.global_index.get(source_id)
            target = self.global_index.get(target_id)

            if not source or not target:
                return

            if target_id not in source.connections:
                source.connections.append(target_id)
            source.connection_weights[target_id] = weight

            if bidirectional:
                if source_id not in target.connections:
                    target.connections.append(source_id)
                target.connection_weights[source_id] = weight

    async def get_connections(self, memory_id: UUID) -> List[Tuple[UUID, float]]:
        """Get all connections for a memory with their weights"""
        record = self.global_index.get(memory_id)
        if not record:
            return []

        return [(conn_id, record.connection_weights.get(conn_id, 1.0))
                for conn_id in record.connections]

    async def migrate_tier(
        self,
        memory_id: UUID,
        target_tier: StorageTier
    ) -> bool:
        """Migrate a memory to a different tier"""
        async with self._lock:
            record = self.global_index.get(memory_id)
            if not record:
                return False

            current_tier = record.current_tier

            # Remove from current tier index
            if memory_id in self.tier_indexes[current_tier]:
                del self.tier_indexes[current_tier][memory_id]
                self.stats["tier_counts"][current_tier] -= 1

            # Ensure embedding exists for target tier
            if target_tier not in record.embeddings:
                # Need to project from available tier
                source_tier = None
                for tier in StorageTier:
                    if tier in record.embeddings:
                        source_tier = tier
                        if tier.value <= target_tier.value:
                            break

                if source_tier is not None:
                    source_emb = record.embeddings[source_tier]
                    projected = self.projector.project(
                        source_emb,
                        source_tier.dimensions,
                        target_tier.dimensions
                    )
                    record.embeddings[target_tier] = projected
                    self.stats["projections_performed"] += 1

            # Add to target tier index
            record.current_tier = target_tier
            self.tier_indexes[target_tier][memory_id] = record
            self.stats["tier_counts"][target_tier] += 1

            logger.info("Memory migrated between tiers",
                       memory_id=str(memory_id),
                       from_tier=current_tier.description,
                       to_tier=target_tier.description)

            return True

    async def get_tier_stats(self) -> Dict[str, Any]:
        """Get statistics for all tiers"""
        tier_info = {}
        for tier in StorageTier:
            records = self.tier_indexes[tier]
            if records:
                avg_temp = np.mean([r.temperature for r in records.values()])
                avg_connections = np.mean([len(r.connections) for r in records.values()])
            else:
                avg_temp = 0.0
                avg_connections = 0.0

            tier_info[tier.name] = {
                "dimensions": tier.dimensions,
                "count": len(records),
                "avg_temperature": float(avg_temp),
                "avg_connections": float(avg_connections),
                "bytes": self.stats["tier_bytes"][tier]
            }

        return {
            "tiers": tier_info,
            "total_memories": self.stats["total_memories"],
            "projections_performed": self.stats["projections_performed"],
            "cache_hit_rate": (
                self.stats["cache_hits"] /
                max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
        }

    async def cleanup_cold_memories(
        self,
        temperature_threshold: float = 0.1,
        max_age_days: int = 30
    ) -> int:
        """Archive cold memories to highest tier"""
        migrated = 0
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)

        for tier in [StorageTier.RAMDISK, StorageTier.LOW, StorageTier.MEDIUM]:
            for memory_id, record in list(self.tier_indexes[tier].items()):
                if record.temperature < temperature_threshold and record.last_accessed < cutoff:
                    # Migrate to colder storage
                    target = StorageTier(min(tier.value + 1, StorageTier.ULTRA_HIGH.value))
                    if await self.migrate_tier(memory_id, target):
                        migrated += 1

        logger.info("Cold memory cleanup completed", migrated=migrated)
        return migrated
