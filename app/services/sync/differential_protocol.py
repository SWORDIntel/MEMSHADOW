"""
Differential Sync Protocol
Phase 4: Distributed Architecture - Efficient delta-based synchronization
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
import hashlib
import json
import structlog

logger = structlog.get_logger()

@dataclass
class SyncDelta:
    """Represents a single change delta"""
    operation: str  # "create", "update", "delete"
    resource_type: str  # "memory", "enrichment", "kg_node", etc.
    resource_id: str
    data: Optional[Dict] = None
    version: int = 1
    timestamp: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.checksum and self.data:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute checksum for data integrity"""
        if not self.data:
            return ""
        serialized = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

@dataclass
class SyncBatch:
    """Batch of sync deltas"""
    deltas: List[SyncDelta]
    batch_id: str
    created_at: str
    compressed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "compressed": self.compressed,
            "deltas": [
                {
                    "operation": d.operation,
                    "resource_type": d.resource_type,
                    "resource_id": d.resource_id,
                    "data": d.data,
                    "version": d.version,
                    "timestamp": d.timestamp,
                    "checksum": d.checksum
                }
                for d in self.deltas
            ]
        }

class DifferentialSyncProtocol:
    """
    Implements efficient differential synchronization protocol.
    
    Features:
    - Delta-based sync (only changes)
    - Conflict detection and resolution
    - Compression for bandwidth efficiency
    - Batch operations
    - Checksum validation
    """
    
    def __init__(self):
        self.version_tracker = {}  # resource_id -> version
        self.change_log = []  # Log of all changes
        self.sync_checkpoints = {}  # device_id -> last_sync_checkpoint
    
    async def create_delta(
        self,
        operation: str,
        resource_type: str,
        resource_id: str,
        data: Optional[Dict] = None,
        version: Optional[int] = None
    ) -> SyncDelta:
        """
        Create a sync delta for a change.
        
        Args:
            operation: Type of operation (create, update, delete)
            resource_type: Type of resource
            resource_id: Resource identifier
            data: Resource data (for create/update)
            version: Version number (auto-incremented if not provided)
        
        Returns:
            SyncDelta object
        """
        # Get or create version
        if version is None:
            current_version = self.version_tracker.get(resource_id, 0)
            version = current_version + 1
            self.version_tracker[resource_id] = version
        
        delta = SyncDelta(
            operation=operation,
            resource_type=resource_type,
            resource_id=resource_id,
            data=data,
            version=version
        )
        
        # Log the change
        self.change_log.append(delta)
        
        logger.debug("Delta created",
                    operation=operation,
                    resource_id=resource_id,
                    version=version)
        
        return delta
    
    async def create_batch(
        self,
        deltas: List[SyncDelta],
        compress: bool = False
    ) -> SyncBatch:
        """
        Create a batch of deltas for efficient transmission.
        
        Args:
            deltas: List of sync deltas
            compress: Whether to compress the batch
        
        Returns:
            SyncBatch object
        """
        import uuid
        
        batch = SyncBatch(
            deltas=deltas,
            batch_id=str(uuid.uuid4()),
            created_at=datetime.utcnow().isoformat(),
            compressed=compress
        )
        
        logger.info("Sync batch created",
                   batch_id=batch.batch_id,
                   delta_count=len(deltas),
                   compressed=compress)
        
        return batch
    
    async def compute_diff(
        self,
        local_state: Dict[str, Any],
        remote_state: Dict[str, Any]
    ) -> List[SyncDelta]:
        """
        Compute differences between local and remote states.
        
        Returns list of deltas to sync remote to match local.
        """
        deltas = []
        
        local_ids = set(local_state.keys())
        remote_ids = set(remote_state.keys())
        
        # New items (in local, not in remote)
        for item_id in local_ids - remote_ids:
            delta = await self.create_delta(
                operation="create",
                resource_type="memory",
                resource_id=item_id,
                data=local_state[item_id]
            )
            deltas.append(delta)
        
        # Deleted items (in remote, not in local)
        for item_id in remote_ids - local_ids:
            delta = await self.create_delta(
                operation="delete",
                resource_type="memory",
                resource_id=item_id
            )
            deltas.append(delta)
        
        # Updated items (in both, but different)
        for item_id in local_ids & remote_ids:
            local_item = local_state[item_id]
            remote_item = remote_state[item_id]
            
            # Compare checksums
            local_checksum = self._compute_checksum(local_item)
            remote_checksum = self._compute_checksum(remote_item)
            
            if local_checksum != remote_checksum:
                # Item has changed
                delta = await self.create_delta(
                    operation="update",
                    resource_type="memory",
                    resource_id=item_id,
                    data=local_item
                )
                deltas.append(delta)
        
        logger.info("Diff computed",
                   creates=len([d for d in deltas if d.operation == "create"]),
                   updates=len([d for d in deltas if d.operation == "update"]),
                   deletes=len([d for d in deltas if d.operation == "delete"]))
        
        return deltas
    
    async def apply_delta(
        self,
        delta: SyncDelta,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a delta to current state.
        
        Returns updated state.
        """
        logger.debug("Applying delta",
                    operation=delta.operation,
                    resource_id=delta.resource_id)
        
        if delta.operation == "create":
            if delta.resource_id in current_state:
                # Conflict: item already exists
                logger.warning("Create conflict", resource_id=delta.resource_id)
                # Could implement conflict resolution here
            current_state[delta.resource_id] = delta.data
        
        elif delta.operation == "update":
            if delta.resource_id not in current_state:
                logger.warning("Update on non-existent item", resource_id=delta.resource_id)
                current_state[delta.resource_id] = delta.data
            else:
                # Merge update
                current_state[delta.resource_id].update(delta.data)
        
        elif delta.operation == "delete":
            if delta.resource_id in current_state:
                del current_state[delta.resource_id]
            else:
                logger.warning("Delete on non-existent item", resource_id=delta.resource_id)
        
        return current_state
    
    async def apply_batch(
        self,
        batch: SyncBatch,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a batch of deltas to current state.
        
        Returns updated state.
        """
        logger.info("Applying sync batch",
                   batch_id=batch.batch_id,
                   delta_count=len(batch.deltas))
        
        for delta in batch.deltas:
            # Verify checksum
            if delta.data and delta.checksum:
                computed = self._compute_checksum(delta.data)
                if computed != delta.checksum:
                    logger.error("Checksum mismatch",
                               resource_id=delta.resource_id,
                               expected=delta.checksum,
                               computed=computed)
                    continue
            
            current_state = await self.apply_delta(delta, current_state)
        
        return current_state
    
    async def get_changes_since(
        self,
        checkpoint: datetime,
        resource_types: Optional[List[str]] = None
    ) -> List[SyncDelta]:
        """
        Get all changes since a checkpoint.
        
        Args:
            checkpoint: Timestamp to get changes since
            resource_types: Filter by resource types
        
        Returns:
            List of deltas since checkpoint
        """
        changes = []
        
        for delta in self.change_log:
            delta_time = datetime.fromisoformat(delta.timestamp)
            
            if delta_time > checkpoint:
                if resource_types is None or delta.resource_type in resource_types:
                    changes.append(delta)
        
        logger.info("Changes retrieved",
                   since=checkpoint.isoformat(),
                   count=len(changes))
        
        return changes
    
    async def create_checkpoint(self, device_id: str) -> str:
        """
        Create a sync checkpoint for a device.
        
        Returns checkpoint ID.
        """
        checkpoint = datetime.utcnow()
        checkpoint_id = f"{device_id}_{checkpoint.isoformat()}"
        
        self.sync_checkpoints[device_id] = {
            "checkpoint_id": checkpoint_id,
            "timestamp": checkpoint.isoformat(),
            "version_snapshot": dict(self.version_tracker)
        }
        
        logger.info("Checkpoint created",
                   device_id=device_id,
                   checkpoint_id=checkpoint_id)
        
        return checkpoint_id
    
    async def get_bandwidth_estimate(self, batch: SyncBatch) -> Dict[str, Any]:
        """
        Estimate bandwidth required for a sync batch.
        
        Returns size estimates.
        """
        # Serialize batch
        batch_data = batch.to_dict()
        serialized = json.dumps(batch_data)
        
        uncompressed_size = len(serialized.encode())
        
        # Estimate compression ratio (typically 60-80% reduction)
        compressed_size = int(uncompressed_size * 0.3)
        
        return {
            "uncompressed_bytes": uncompressed_size,
            "uncompressed_kb": uncompressed_size / 1024,
            "compressed_bytes": compressed_size,
            "compressed_kb": compressed_size / 1024,
            "compression_ratio": 0.3,
            "delta_count": len(batch.deltas)
        }
    
    def _compute_checksum(self, data: Dict) -> str:
        """Compute checksum for data"""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        operation_counts = {}
        resource_counts = {}
        
        for delta in self.change_log:
            operation_counts[delta.operation] = operation_counts.get(delta.operation, 0) + 1
            resource_counts[delta.resource_type] = resource_counts.get(delta.resource_type, 0) + 1
        
        return {
            "total_changes": len(self.change_log),
            "tracked_resources": len(self.version_tracker),
            "checkpoints": len(self.sync_checkpoints),
            "operation_breakdown": operation_counts,
            "resource_breakdown": resource_counts
        }


# Global instance
differential_sync = DifferentialSyncProtocol()
