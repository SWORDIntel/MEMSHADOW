"""
Cross-Tier Deduplication System

Implements intelligent deduplication across all storage tiers to:
- Detect duplicate memories based on content and semantic similarity
- Merge duplicate connections
- Maintain referential integrity
- Optimize storage utilization

This prevents redundancy while preserving all valuable connections.
"""

import asyncio
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import structlog

from .tiered_database import TieredDatabaseManager, StorageTier, TieredMemoryRecord
from .ramdisk_storage import RAMDiskStorage

logger = structlog.get_logger()


@dataclass
class DuplicateCandidate:
    """Represents a potential duplicate"""
    memory_id: UUID
    duplicate_of: UUID
    similarity: float
    match_type: str  # "exact", "semantic", "near_duplicate"
    tier: StorageTier
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MergeResult:
    """Result of a merge operation"""
    primary_id: UUID
    merged_ids: List[UUID]
    connections_inherited: int
    storage_freed_bytes: int


class CrossTierDeduplicator:
    """
    Detects and handles duplicates across all storage tiers.

    Features:
    - Exact duplicate detection via content hash
    - Semantic duplicate detection via embedding similarity
    - Near-duplicate detection for similar but not identical content
    - Connection merging to preserve relationships
    - Cross-tier deduplication scanning
    """

    def __init__(
        self,
        tiered_db: TieredDatabaseManager,
        ramdisk: RAMDiskStorage,
        exact_threshold: float = 1.0,
        semantic_threshold: float = 0.95,
        near_duplicate_threshold: float = 0.85
    ):
        self.tiered_db = tiered_db
        self.ramdisk = ramdisk
        self.exact_threshold = exact_threshold
        self.semantic_threshold = semantic_threshold
        self.near_duplicate_threshold = near_duplicate_threshold

        # Duplicate tracking
        self.detected_duplicates: List[DuplicateCandidate] = []
        self.merge_history: List[MergeResult] = []

        # Hash collision tracking (same hash but different content - unlikely but possible)
        self.hash_collisions: Set[Tuple[UUID, UUID]] = set()

        # Statistics
        self.stats = {
            "exact_duplicates_found": 0,
            "semantic_duplicates_found": 0,
            "near_duplicates_found": 0,
            "total_merges": 0,
            "connections_preserved": 0,
            "storage_freed_bytes": 0,
            "scans_performed": 0,
        }

        # Lock
        self._lock = asyncio.Lock()

        logger.info("CrossTierDeduplicator initialized",
                   semantic_threshold=semantic_threshold,
                   near_duplicate_threshold=near_duplicate_threshold)

    async def scan_for_duplicates(
        self,
        tier: Optional[StorageTier] = None,
        limit: int = 1000
    ) -> List[DuplicateCandidate]:
        """
        Scan for duplicates within a tier or across all tiers.
        Returns list of detected duplicates.
        """
        self.stats["scans_performed"] += 1
        duplicates = []

        tiers_to_scan = [tier] if tier else list(StorageTier)

        async with self._lock:
            for scan_tier in tiers_to_scan:
                tier_duplicates = await self._scan_tier(scan_tier, limit)
                duplicates.extend(tier_duplicates)

        self.detected_duplicates.extend(duplicates)
        logger.info("Duplicate scan complete",
                   tier=tier.name if tier else "ALL",
                   duplicates_found=len(duplicates))

        return duplicates

    async def _scan_tier(
        self,
        tier: StorageTier,
        limit: int
    ) -> List[DuplicateCandidate]:
        """Scan a single tier for duplicates"""
        duplicates = []
        records = list(self.tiered_db.tier_indexes[tier].values())

        # Track hashes we've seen
        seen_hashes: Dict[str, UUID] = {}

        for record in records[:limit]:
            # Check exact duplicate (hash match)
            if record.content_hash in seen_hashes:
                existing_id = seen_hashes[record.content_hash]
                duplicates.append(DuplicateCandidate(
                    memory_id=record.memory_id,
                    duplicate_of=existing_id,
                    similarity=1.0,
                    match_type="exact",
                    tier=tier
                ))
                self.stats["exact_duplicates_found"] += 1
                continue

            seen_hashes[record.content_hash] = record.memory_id

            # Check semantic duplicates
            if tier in record.embeddings:
                embedding = record.embeddings[tier]
                semantic_dups = await self._find_semantic_duplicates(
                    record.memory_id,
                    embedding,
                    tier,
                    records
                )
                duplicates.extend(semantic_dups)

        return duplicates

    async def _find_semantic_duplicates(
        self,
        memory_id: UUID,
        embedding: np.ndarray,
        tier: StorageTier,
        records: List[TieredMemoryRecord]
    ) -> List[DuplicateCandidate]:
        """Find semantically similar memories"""
        duplicates = []

        # Normalize query embedding
        query_norm = np.linalg.norm(embedding)
        if query_norm > 0:
            embedding = embedding / query_norm

        for record in records:
            if record.memory_id == memory_id:
                continue

            if tier not in record.embeddings:
                continue

            other_emb = record.embeddings[tier]
            other_norm = np.linalg.norm(other_emb)
            if other_norm > 0:
                other_emb = other_emb / other_norm

            similarity = float(np.dot(embedding, other_emb))

            if similarity >= self.semantic_threshold:
                # High semantic similarity - likely duplicate
                duplicates.append(DuplicateCandidate(
                    memory_id=memory_id,
                    duplicate_of=record.memory_id,
                    similarity=similarity,
                    match_type="semantic",
                    tier=tier
                ))
                self.stats["semantic_duplicates_found"] += 1

            elif similarity >= self.near_duplicate_threshold:
                # Near duplicate - similar but not identical
                duplicates.append(DuplicateCandidate(
                    memory_id=memory_id,
                    duplicate_of=record.memory_id,
                    similarity=similarity,
                    match_type="near_duplicate",
                    tier=tier
                ))
                self.stats["near_duplicates_found"] += 1

        return duplicates

    async def cross_tier_scan(self) -> List[DuplicateCandidate]:
        """
        Scan for duplicates across different tiers.
        A memory might exist in multiple tiers - this finds those cases.
        """
        cross_tier_dups = []
        all_hashes: Dict[str, List[Tuple[UUID, StorageTier]]] = defaultdict(list)

        # Collect all hashes across tiers
        for tier in StorageTier:
            for record in self.tiered_db.tier_indexes[tier].values():
                all_hashes[record.content_hash].append((record.memory_id, tier))

        # Find cross-tier duplicates
        for content_hash, occurrences in all_hashes.items():
            if len(occurrences) > 1:
                # Same hash in multiple tiers
                primary_id, primary_tier = occurrences[0]

                for dup_id, dup_tier in occurrences[1:]:
                    if dup_id != primary_id:
                        cross_tier_dups.append(DuplicateCandidate(
                            memory_id=dup_id,
                            duplicate_of=primary_id,
                            similarity=1.0,
                            match_type="exact",
                            tier=dup_tier
                        ))

        logger.info("Cross-tier scan complete", duplicates=len(cross_tier_dups))
        return cross_tier_dups

    async def merge_duplicates(
        self,
        primary_id: UUID,
        duplicate_ids: List[UUID],
        preserve_all_connections: bool = True
    ) -> Optional[MergeResult]:
        """
        Merge duplicate memories into primary, preserving connections.

        Args:
            primary_id: The memory to keep
            duplicate_ids: Memories to merge into primary
            preserve_all_connections: If True, inherit all connections from duplicates
        """
        async with self._lock:
            primary = self.tiered_db.global_index.get(primary_id)
            if not primary:
                logger.warning("Primary memory not found for merge", primary_id=str(primary_id))
                return None

            connections_inherited = 0
            storage_freed = 0
            successfully_merged = []

            for dup_id in duplicate_ids:
                duplicate = self.tiered_db.global_index.get(dup_id)
                if not duplicate:
                    continue

                # Inherit connections
                if preserve_all_connections:
                    for conn_id in duplicate.connections:
                        if conn_id != primary_id and conn_id not in primary.connections:
                            primary.connections.append(conn_id)
                            # Also get the weight
                            weight = duplicate.connection_weights.get(conn_id, 0.5)
                            primary.connection_weights[conn_id] = max(
                                primary.connection_weights.get(conn_id, 0),
                                weight
                            )
                            connections_inherited += 1

                    # Update connection weights (average if both have)
                    for conn_id, weight in duplicate.connection_weights.items():
                        if conn_id in primary.connection_weights:
                            primary.connection_weights[conn_id] = (
                                primary.connection_weights[conn_id] + weight
                            ) / 2

                # Calculate freed storage
                for tier, emb in duplicate.embeddings.items():
                    storage_freed += emb.nbytes

                # Remove duplicate from all tiers
                await self._remove_duplicate(dup_id)
                successfully_merged.append(dup_id)

            # Update statistics
            self.stats["total_merges"] += 1
            self.stats["connections_preserved"] += connections_inherited
            self.stats["storage_freed_bytes"] += storage_freed

            result = MergeResult(
                primary_id=primary_id,
                merged_ids=successfully_merged,
                connections_inherited=connections_inherited,
                storage_freed_bytes=storage_freed
            )
            self.merge_history.append(result)

            logger.info("Duplicates merged",
                       primary=str(primary_id),
                       merged=len(successfully_merged),
                       connections=connections_inherited)

            return result

    async def _remove_duplicate(self, memory_id: UUID):
        """Remove a duplicate memory from all storage"""
        record = self.tiered_db.global_index.get(memory_id)
        if not record:
            return

        # Remove from tier indexes
        for tier in StorageTier:
            if memory_id in self.tiered_db.tier_indexes[tier]:
                del self.tiered_db.tier_indexes[tier][memory_id]

        # Remove from global index
        if memory_id in self.tiered_db.global_index:
            del self.tiered_db.global_index[memory_id]

        # Remove from hash index
        if record.content_hash in self.tiered_db.hash_index:
            if self.tiered_db.hash_index[record.content_hash] == memory_id:
                del self.tiered_db.hash_index[record.content_hash]

        # Remove from RAMDISK if present
        await self.ramdisk.remove(memory_id)

        # Update other memories' connections to point to nothing
        # (they'll be cleaned up on access)

    async def auto_deduplicate(
        self,
        batch_size: int = 100,
        merge_semantic: bool = True,
        merge_near: bool = False
    ) -> Dict[str, int]:
        """
        Automatically detect and merge duplicates.

        Args:
            batch_size: Number of duplicates to process
            merge_semantic: Also merge semantic duplicates
            merge_near: Also merge near-duplicates (risky, may lose info)
        """
        results = {
            "exact_merged": 0,
            "semantic_merged": 0,
            "near_merged": 0,
            "total_merged": 0,
        }

        # Scan for duplicates
        duplicates = await self.scan_for_duplicates(limit=batch_size * 2)

        # Group by primary
        groups: Dict[UUID, List[DuplicateCandidate]] = defaultdict(list)
        for dup in duplicates:
            groups[dup.duplicate_of].append(dup)

        # Process each group
        processed = 0
        for primary_id, dups in groups.items():
            if processed >= batch_size:
                break

            # Filter by match type
            exact_dups = [d.memory_id for d in dups if d.match_type == "exact"]
            semantic_dups = [d.memory_id for d in dups if d.match_type == "semantic"]
            near_dups = [d.memory_id for d in dups if d.match_type == "near_duplicate"]

            # Merge exact duplicates always
            if exact_dups:
                result = await self.merge_duplicates(primary_id, exact_dups)
                if result:
                    results["exact_merged"] += len(result.merged_ids)

            # Optionally merge semantic duplicates
            if merge_semantic and semantic_dups:
                result = await self.merge_duplicates(primary_id, semantic_dups)
                if result:
                    results["semantic_merged"] += len(result.merged_ids)

            # Optionally merge near-duplicates
            if merge_near and near_dups:
                result = await self.merge_duplicates(primary_id, near_dups)
                if result:
                    results["near_merged"] += len(result.merged_ids)

            processed += 1

        results["total_merged"] = (
            results["exact_merged"] +
            results["semantic_merged"] +
            results["near_merged"]
        )

        logger.info("Auto-deduplication complete", **results)
        return results

    async def find_similar_across_tiers(
        self,
        embedding: np.ndarray,
        source_tier: StorageTier,
        threshold: float = 0.8
    ) -> Dict[StorageTier, List[Tuple[UUID, float]]]:
        """
        Find similar memories across all tiers for a given embedding.
        Useful for detecting potential duplicates before storage.
        """
        similar_by_tier = {}

        for tier in StorageTier:
            similar = await self.tiered_db.search_tier(
                embedding, tier, top_k=10, threshold=threshold
            )
            if similar:
                similar_by_tier[tier] = similar

        return similar_by_tier

    async def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            "exact_duplicates_found": self.stats["exact_duplicates_found"],
            "semantic_duplicates_found": self.stats["semantic_duplicates_found"],
            "near_duplicates_found": self.stats["near_duplicates_found"],
            "total_merges": self.stats["total_merges"],
            "connections_preserved": self.stats["connections_preserved"],
            "storage_freed_mb": self.stats["storage_freed_bytes"] / (1024 * 1024),
            "scans_performed": self.stats["scans_performed"],
            "pending_duplicates": len(self.detected_duplicates),
            "hash_collisions": len(self.hash_collisions),
        }

    async def estimate_duplicate_ratio(
        self,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Estimate the duplicate ratio in the database.
        Uses sampling for efficiency on large databases.
        """
        total_records = sum(
            len(self.tiered_db.tier_indexes[tier])
            for tier in StorageTier
        )

        if total_records == 0:
            return {"exact_ratio": 0, "semantic_ratio": 0, "near_ratio": 0}

        # Sample from each tier proportionally
        sample_dups = await self.scan_for_duplicates(limit=sample_size)

        exact_count = sum(1 for d in sample_dups if d.match_type == "exact")
        semantic_count = sum(1 for d in sample_dups if d.match_type == "semantic")
        near_count = sum(1 for d in sample_dups if d.match_type == "near_duplicate")

        sample_total = len(sample_dups) or 1

        return {
            "exact_ratio": exact_count / sample_total,
            "semantic_ratio": semantic_count / sample_total,
            "near_ratio": near_count / sample_total,
            "total_records": total_records,
            "sample_size": sample_size,
        }

    async def clean_orphaned_connections(self) -> int:
        """
        Clean up connections that point to non-existent memories.
        These can occur after merges or deletions.
        """
        cleaned = 0

        for record in self.tiered_db.global_index.values():
            valid_connections = []
            for conn_id in record.connections:
                if conn_id in self.tiered_db.global_index:
                    valid_connections.append(conn_id)
                else:
                    cleaned += 1

            record.connections = valid_connections

            # Also clean connection weights
            valid_weights = {
                cid: w for cid, w in record.connection_weights.items()
                if cid in self.tiered_db.global_index
            }
            record.connection_weights = valid_weights

        if cleaned > 0:
            logger.info("Orphaned connections cleaned", count=cleaned)

        return cleaned
