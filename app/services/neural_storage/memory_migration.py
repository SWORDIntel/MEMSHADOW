"""
Memory Migration Manager - Hot/Cold Storage Management

Implements intelligent migration of memories between storage tiers
based on access patterns, temperature, and importance.

Features:
- Automatic promotion of hot memories to faster tiers
- Automatic demotion of cold memories to archival tiers
- Batch migration for efficiency
- Priority-based migration scheduling
- Connection-aware migration (keep related memories together)
"""

import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import structlog

from .tiered_database import TieredDatabaseManager, StorageTier, TieredMemoryRecord
from .ramdisk_storage import RAMDiskStorage
from .dynamic_cpu_manager import DynamicCPUManager, WorkloadIntensity

logger = structlog.get_logger()


class MigrationDirection(Enum):
    """Direction of migration"""
    PROMOTE = "promote"   # Move to faster tier
    DEMOTE = "demote"     # Move to slower tier


@dataclass
class MigrationTask:
    """Represents a pending migration task"""
    memory_id: UUID
    source_tier: StorageTier
    target_tier: StorageTier
    direction: MigrationDirection
    priority: float = 0.5  # 0.0 to 1.0
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class MigrationPolicy:
    """Policy configuration for migrations"""
    # Temperature thresholds
    promote_threshold: float = 0.8  # Promote if temp > this
    demote_threshold: float = 0.2   # Demote if temp < this

    # Time thresholds
    min_age_for_demotion_hours: float = 24.0
    max_idle_hours_before_archive: float = 168.0  # 1 week

    # Batch settings
    batch_size: int = 100
    max_migrations_per_cycle: int = 500

    # Connection settings
    keep_connections_together: bool = True
    connection_proximity_tiers: int = 1  # Max tier difference for connected memories


class MemoryMigrationManager:
    """
    Manages migration of memories between tiers based on access patterns.

    The manager continuously monitors memory temperatures and access patterns,
    automatically promoting hot memories to faster tiers and demoting cold
    memories to archival storage.
    """

    def __init__(
        self,
        tiered_db: TieredDatabaseManager,
        ramdisk: RAMDiskStorage,
        cpu_manager: DynamicCPUManager,
        policy: Optional[MigrationPolicy] = None
    ):
        self.tiered_db = tiered_db
        self.ramdisk = ramdisk
        self.cpu_manager = cpu_manager
        self.policy = policy or MigrationPolicy()

        # Migration queue
        self.pending_migrations: List[MigrationTask] = []
        self.in_progress: Set[UUID] = set()

        # Statistics
        self.stats = {
            "total_promotions": 0,
            "total_demotions": 0,
            "total_to_ramdisk": 0,
            "total_from_ramdisk": 0,
            "failed_migrations": 0,
            "batches_processed": 0,
        }

        # Background task
        self._running = False
        self._migration_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None

        # Lock
        self._lock = asyncio.Lock()

        logger.info("MemoryMigrationManager initialized",
                   promote_threshold=self.policy.promote_threshold,
                   demote_threshold=self.policy.demote_threshold)

    async def start(self):
        """Start the migration manager"""
        self._running = True
        self._migration_task = asyncio.create_task(self._migration_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("MemoryMigrationManager started")

    async def stop(self):
        """Stop the migration manager"""
        self._running = False
        if self._migration_task:
            self._migration_task.cancel()
        if self._analysis_task:
            self._analysis_task.cancel()
        logger.info("MemoryMigrationManager stopped")

    async def _migration_loop(self):
        """Background loop to process migrations"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._process_pending_migrations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Migration loop error", error=str(e))

    async def _analysis_loop(self):
        """Background loop to analyze and schedule migrations"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Analyze every minute
                await self._analyze_and_schedule()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Analysis loop error", error=str(e))

    async def _analyze_and_schedule(self):
        """Analyze all memories and schedule necessary migrations"""
        async with self._lock:
            now = datetime.utcnow()

            # Analyze each tier
            for tier in StorageTier:
                for memory_id, record in self.tiered_db.tier_indexes[tier].items():
                    if memory_id in self.in_progress:
                        continue

                    # Calculate current temperature
                    record.temperature = self.tiered_db._calculate_temperature(record)

                    # Check for promotion (move to faster tier)
                    if self._should_promote(record, tier):
                        target = self._get_promotion_target(tier)
                        if target is not None:
                            await self._schedule_migration(
                                memory_id, tier, target,
                                MigrationDirection.PROMOTE,
                                record.temperature,
                                "hot_memory"
                            )

                    # Check for demotion (move to slower tier)
                    elif self._should_demote(record, tier, now):
                        target = self._get_demotion_target(tier)
                        if target is not None:
                            await self._schedule_migration(
                                memory_id, tier, target,
                                MigrationDirection.DEMOTE,
                                1.0 - record.temperature,
                                "cold_memory"
                            )

            logger.debug("Migration analysis complete",
                        pending=len(self.pending_migrations))

    def _should_promote(self, record: TieredMemoryRecord, current_tier: StorageTier) -> bool:
        """Determine if a memory should be promoted"""
        if current_tier == StorageTier.RAMDISK:
            return False  # Already at fastest tier

        return record.temperature > self.policy.promote_threshold

    def _should_demote(
        self,
        record: TieredMemoryRecord,
        current_tier: StorageTier,
        now: datetime
    ) -> bool:
        """Determine if a memory should be demoted"""
        if current_tier == StorageTier.ULTRA_HIGH:
            return False  # Already at coldest tier

        # Check temperature
        if record.temperature > self.policy.demote_threshold:
            return False

        # Check minimum age
        age_hours = (now - record.created_at).total_seconds() / 3600
        if age_hours < self.policy.min_age_for_demotion_hours:
            return False

        # Check idle time
        idle_hours = (now - record.last_accessed).total_seconds() / 3600
        if idle_hours > self.policy.max_idle_hours_before_archive:
            return True  # Force archive for very idle memories

        return record.temperature < self.policy.demote_threshold

    def _get_promotion_target(self, current_tier: StorageTier) -> Optional[StorageTier]:
        """Get the target tier for promotion"""
        if current_tier == StorageTier.RAMDISK:
            return None
        return StorageTier(current_tier.value + 1)  # Higher value = faster

    def _get_demotion_target(self, current_tier: StorageTier) -> Optional[StorageTier]:
        """Get the target tier for demotion"""
        if current_tier == StorageTier.ULTRA_HIGH:
            return None
        return StorageTier(current_tier.value - 1)  # Lower value = slower

    async def _schedule_migration(
        self,
        memory_id: UUID,
        source_tier: StorageTier,
        target_tier: StorageTier,
        direction: MigrationDirection,
        priority: float,
        reason: str
    ):
        """Schedule a migration task"""
        # Check if already scheduled
        for task in self.pending_migrations:
            if task.memory_id == memory_id:
                return

        task = MigrationTask(
            memory_id=memory_id,
            source_tier=source_tier,
            target_tier=target_tier,
            direction=direction,
            priority=priority,
            reason=reason
        )

        self.pending_migrations.append(task)

        # Sort by priority (highest first)
        self.pending_migrations.sort(key=lambda t: t.priority, reverse=True)

    async def _process_pending_migrations(self):
        """Process pending migration tasks"""
        if not self.pending_migrations:
            return

        # Get batch of high-priority migrations
        batch_size = min(
            self.policy.batch_size,
            len(self.pending_migrations)
        )
        batch = self.pending_migrations[:batch_size]
        self.pending_migrations = self.pending_migrations[batch_size:]

        # Request CPU burst for batch processing
        if batch_size > 10:
            await self.cpu_manager.request_burst(5.0)

        # Process batch
        success = 0
        failed = 0

        for task in batch:
            try:
                self.in_progress.add(task.memory_id)

                if task.direction == MigrationDirection.PROMOTE:
                    result = await self._execute_promotion(task)
                else:
                    result = await self._execute_demotion(task)

                if result:
                    success += 1
                else:
                    failed += 1
                    task.attempts += 1
                    if task.attempts < task.max_attempts:
                        # Re-queue with lower priority
                        task.priority *= 0.5
                        self.pending_migrations.append(task)

            except Exception as e:
                logger.error("Migration failed",
                           memory_id=str(task.memory_id),
                           error=str(e))
                failed += 1
                self.stats["failed_migrations"] += 1

            finally:
                self.in_progress.discard(task.memory_id)

        self.stats["batches_processed"] += 1
        logger.info("Migration batch processed",
                   success=success, failed=failed)

    async def _execute_promotion(self, task: MigrationTask) -> bool:
        """Execute a promotion migration"""
        memory_id = task.memory_id
        target = task.target_tier

        # Handle promotion to RAMDISK
        if target == StorageTier.RAMDISK:
            return await self._promote_to_ramdisk(memory_id, task.source_tier)

        # Normal tier promotion
        result = await self.tiered_db.migrate_tier(memory_id, target)

        if result:
            self.stats["total_promotions"] += 1

        return result

    async def _execute_demotion(self, task: MigrationTask) -> bool:
        """Execute a demotion migration"""
        memory_id = task.memory_id

        # Handle demotion from RAMDISK
        if task.source_tier == StorageTier.RAMDISK:
            return await self._demote_from_ramdisk(memory_id, task.target_tier)

        # Normal tier demotion
        result = await self.tiered_db.migrate_tier(memory_id, task.target_tier)

        if result:
            self.stats["total_demotions"] += 1

        return result

    async def _promote_to_ramdisk(
        self,
        memory_id: UUID,
        source_tier: StorageTier
    ) -> bool:
        """Promote a memory to RAMDISK"""
        record = self.tiered_db.global_index.get(memory_id)
        if not record:
            return False

        # Get embedding from tiered DB
        result = await self.tiered_db.retrieve(memory_id, preferred_tier=source_tier)
        if not result:
            return False

        embedding, _ = result

        # Store in RAMDISK
        stored = await self.ramdisk.store(
            memory_id=memory_id,
            embedding=embedding,
            content_hash=record.content_hash,
            metadata=record.metadata,
            connections=set(record.connections)
        )

        if stored:
            # Also update record's current tier
            record.current_tier = StorageTier.RAMDISK
            self.stats["total_to_ramdisk"] += 1
            logger.debug("Promoted to RAMDISK", memory_id=str(memory_id))

        return stored

    async def _demote_from_ramdisk(
        self,
        memory_id: UUID,
        target_tier: StorageTier
    ) -> bool:
        """Demote a memory from RAMDISK to a slower tier"""
        # Get from RAMDISK
        entry = await self.ramdisk.retrieve(memory_id)
        if not entry:
            return False

        # Update the tiered DB record
        record = self.tiered_db.global_index.get(memory_id)
        if record:
            record.current_tier = target_tier

            # Ensure we have embedding for target tier
            if target_tier not in record.embeddings:
                await self.tiered_db._generate_lower_tier_projections(record, StorageTier.RAMDISK)

            # Move to target tier index
            await self.tiered_db.migrate_tier(memory_id, target_tier)

        # Remove from RAMDISK
        await self.ramdisk.remove(memory_id)

        self.stats["total_from_ramdisk"] += 1
        logger.debug("Demoted from RAMDISK",
                    memory_id=str(memory_id),
                    target=target_tier.description)

        return True

    async def force_promote(
        self,
        memory_id: UUID,
        target_tier: StorageTier = StorageTier.RAMDISK
    ) -> bool:
        """Force immediate promotion of a memory"""
        record = self.tiered_db.global_index.get(memory_id)
        if not record:
            return False

        source_tier = record.current_tier

        # Skip if already at target or faster
        if source_tier.value >= target_tier.value:
            return True

        if target_tier == StorageTier.RAMDISK:
            return await self._promote_to_ramdisk(memory_id, source_tier)
        else:
            return await self.tiered_db.migrate_tier(memory_id, target_tier)

    async def force_demote(
        self,
        memory_id: UUID,
        target_tier: StorageTier = StorageTier.ULTRA_HIGH
    ) -> bool:
        """Force immediate demotion of a memory"""
        record = self.tiered_db.global_index.get(memory_id)
        if not record:
            return False

        source_tier = record.current_tier

        # Skip if already at target or slower
        if source_tier.value <= target_tier.value:
            return True

        if source_tier == StorageTier.RAMDISK:
            return await self._demote_from_ramdisk(memory_id, target_tier)
        else:
            return await self.tiered_db.migrate_tier(memory_id, target_tier)

    async def migrate_connected_group(
        self,
        seed_memory_id: UUID,
        target_tier: StorageTier
    ) -> int:
        """Migrate a memory and all strongly connected memories together"""
        record = self.tiered_db.global_index.get(seed_memory_id)
        if not record:
            return 0

        # Collect connected memories
        to_migrate = {seed_memory_id}
        for conn_id in record.connections:
            conn_record = self.tiered_db.global_index.get(conn_id)
            if conn_record:
                # Only include if within proximity policy
                tier_diff = abs(conn_record.current_tier.value - target_tier.value)
                if tier_diff <= self.policy.connection_proximity_tiers:
                    to_migrate.add(conn_id)

        # Migrate all
        migrated = 0
        for memory_id in to_migrate:
            if target_tier == StorageTier.RAMDISK:
                result = await self._promote_to_ramdisk(
                    memory_id,
                    self.tiered_db.global_index[memory_id].current_tier
                )
            else:
                result = await self.tiered_db.migrate_tier(memory_id, target_tier)

            if result:
                migrated += 1

        logger.info("Connected group migrated",
                   seed_id=str(seed_memory_id),
                   total=len(to_migrate),
                   migrated=migrated,
                   target=target_tier.description)

        return migrated

    async def get_migration_stats(self) -> Dict[str, Any]:
        """Get migration statistics"""
        return {
            "pending_migrations": len(self.pending_migrations),
            "in_progress": len(self.in_progress),
            "total_promotions": self.stats["total_promotions"],
            "total_demotions": self.stats["total_demotions"],
            "total_to_ramdisk": self.stats["total_to_ramdisk"],
            "total_from_ramdisk": self.stats["total_from_ramdisk"],
            "failed_migrations": self.stats["failed_migrations"],
            "batches_processed": self.stats["batches_processed"],
            "policy": {
                "promote_threshold": self.policy.promote_threshold,
                "demote_threshold": self.policy.demote_threshold,
                "batch_size": self.policy.batch_size,
            }
        }

    async def get_tier_distribution(self) -> Dict[str, int]:
        """Get count of memories per tier"""
        distribution = {}
        for tier in StorageTier:
            distribution[tier.name] = len(self.tiered_db.tier_indexes[tier])

        # Add RAMDISK count
        ramdisk_stats = await self.ramdisk.get_stats()
        distribution["RAMDISK_ENTRIES"] = ramdisk_stats["entries"]

        return distribution
