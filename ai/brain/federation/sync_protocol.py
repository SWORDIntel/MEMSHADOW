#!/usr/bin/env python3
"""
Sync Protocol for DSMIL Brain Federation

Handles state synchronization between nodes:
- Delta-based synchronization
- Conflict resolution
- Version tracking
- Eventual consistency
"""

import hashlib
import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import json

logger = logging.getLogger(__name__)


class SyncState(Enum):
    """State of synchronization"""
    IDLE = auto()
    SYNCING = auto()
    CONFLICT = auto()
    FAILED = auto()
    COMPLETE = auto()


class ConflictResolution(Enum):
    """Strategies for resolving conflicts"""
    LATEST_WINS = auto()      # Most recent timestamp wins
    HUB_AUTHORITY = auto()    # Hub version always wins
    MERGE = auto()            # Attempt to merge changes
    HIGHER_CONFIDENCE = auto() # Higher confidence wins


@dataclass
class SyncVersion:
    """Version information for sync"""
    version: int
    timestamp: datetime
    node_id: str
    checksum: str  # Hash of content at this version


@dataclass
class DeltaChange:
    """A single change in a delta"""
    change_id: str
    operation: str  # "add", "update", "delete"
    path: str       # Path to changed item (e.g., "memory/semantic/concepts/abc123")
    value: Any      # New value (None for delete)
    old_value: Any  # Previous value (for conflict detection)
    timestamp: datetime
    node_id: str


@dataclass
class DeltaSync:
    """
    Delta synchronization package

    Contains changes between two versions
    """
    delta_id: str
    source_node: str
    target_node: str

    # Version info
    from_version: int
    to_version: int

    # Changes
    changes: List[DeltaChange] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Checksum
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute checksum of delta"""
        content = json.dumps({
            "from": self.from_version,
            "to": self.to_version,
            "changes": [
                {"id": c.change_id, "op": c.operation, "path": c.path}
                for c in self.changes
            ]
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            "delta_id": self.delta_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "changes": [
                {
                    "change_id": c.change_id,
                    "operation": c.operation,
                    "path": c.path,
                    "value": c.value,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in self.changes
            ],
            "checksum": self.checksum,
        }


@dataclass
class SyncConflict:
    """Detected conflict during sync"""
    path: str
    local_value: Any
    remote_value: Any
    local_timestamp: datetime
    remote_timestamp: datetime
    resolution: Optional[Any] = None
    resolved_by: Optional[ConflictResolution] = None


class VersionTracker:
    """
    Tracks versions and changes for synchronization
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._current_version = 0
        self._change_log: List[DeltaChange] = []
        self._version_history: Dict[int, SyncVersion] = {}
        self._lock = threading.RLock()

        # Track peer versions
        self._peer_versions: Dict[str, int] = {}

    def increment_version(self, checksum: str) -> int:
        """Increment version and record"""
        with self._lock:
            self._current_version += 1

            self._version_history[self._current_version] = SyncVersion(
                version=self._current_version,
                timestamp=datetime.now(timezone.utc),
                node_id=self.node_id,
                checksum=checksum,
            )

            return self._current_version

    def record_change(self, operation: str, path: str,
                     value: Any, old_value: Any = None) -> DeltaChange:
        """Record a change"""
        change_id = hashlib.sha256(
            f"{self.node_id}:{path}:{time.time()}".encode()
        ).hexdigest()[:16]

        change = DeltaChange(
            change_id=change_id,
            operation=operation,
            path=path,
            value=value,
            old_value=old_value,
            timestamp=datetime.now(timezone.utc),
            node_id=self.node_id,
        )

        with self._lock:
            self._change_log.append(change)

        return change

    def get_changes_since(self, version: int) -> List[DeltaChange]:
        """Get all changes since a version"""
        with self._lock:
            # In real implementation, changes would be indexed by version
            # For now, return changes that are newer
            cutoff_time = self._version_history.get(version)
            if not cutoff_time:
                return list(self._change_log)

            return [
                c for c in self._change_log
                if c.timestamp > cutoff_time.timestamp
            ]

    def get_current_version(self) -> int:
        """Get current version number"""
        return self._current_version

    def set_peer_version(self, peer_id: str, version: int):
        """Record a peer's known version"""
        self._peer_versions[peer_id] = version

    def get_peer_version(self, peer_id: str) -> int:
        """Get a peer's known version"""
        return self._peer_versions.get(peer_id, 0)


class SyncProtocol:
    """
    Synchronization protocol for distributed state

    Handles:
    - Delta generation and application
    - Conflict detection and resolution
    - Version tracking
    - Peer synchronization

    Usage:
        sync = SyncProtocol(node_id="node-001", is_hub=False)

        # Record changes
        sync.record_change("add", "memory/concepts/abc", {"name": "threat"})

        # Generate delta for peer
        delta = sync.generate_delta("peer-001")

        # Apply received delta
        conflicts = sync.apply_delta(received_delta)
    """

    def __init__(self, node_id: str,
                 is_hub: bool = False,
                 conflict_resolution: ConflictResolution = ConflictResolution.LATEST_WINS):
        """
        Initialize sync protocol

        Args:
            node_id: This node's ID
            is_hub: True if this is the central hub
            conflict_resolution: Default conflict resolution strategy
        """
        self.node_id = node_id
        self.is_hub = is_hub
        self.conflict_resolution = conflict_resolution

        # Version tracking
        self._version_tracker = VersionTracker(node_id)

        # Local state (simplified - in reality would interface with memory/graph)
        self._local_state: Dict[str, Any] = {}

        # Sync state
        self._sync_state = SyncState.IDLE

        # Conflict tracking
        self._unresolved_conflicts: List[SyncConflict] = []

        # Callbacks
        self.on_state_change: Optional[Callable[[str, Any], None]] = None
        self.on_conflict: Optional[Callable[[SyncConflict], None]] = None

        # Statistics
        self.stats = {
            "deltas_generated": 0,
            "deltas_applied": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
        }

        self._lock = threading.RLock()

        logger.info(f"SyncProtocol initialized for {node_id} (hub={is_hub})")

    def record_change(self, operation: str, path: str,
                     value: Any, old_value: Any = None):
        """
        Record a local change

        Args:
            operation: "add", "update", or "delete"
            path: Path to changed item
            value: New value
            old_value: Previous value (optional, for conflict detection)
        """
        with self._lock:
            # Record in version tracker
            self._version_tracker.record_change(operation, path, value, old_value)

            # Update local state
            if operation == "delete":
                if path in self._local_state:
                    del self._local_state[path]
            else:
                self._local_state[path] = value

            # Trigger callback
            if self.on_state_change:
                self.on_state_change(path, value)

    def generate_delta(self, target_node: str) -> DeltaSync:
        """
        Generate delta for a target node

        Args:
            target_node: Node to generate delta for

        Returns:
            DeltaSync with changes since peer's last known version
        """
        with self._lock:
            peer_version = self._version_tracker.get_peer_version(target_node)
            current_version = self._version_tracker.get_current_version()

            changes = self._version_tracker.get_changes_since(peer_version)

            delta_id = hashlib.sha256(
                f"{self.node_id}:{target_node}:{time.time()}".encode()
            ).hexdigest()[:16]

            delta = DeltaSync(
                delta_id=delta_id,
                source_node=self.node_id,
                target_node=target_node,
                from_version=peer_version,
                to_version=current_version,
                changes=changes,
            )

            self.stats["deltas_generated"] += 1

            return delta

    def apply_delta(self, delta: DeltaSync) -> List[SyncConflict]:
        """
        Apply received delta

        Args:
            delta: Delta to apply

        Returns:
            List of conflicts (if any)
        """
        self._sync_state = SyncState.SYNCING
        conflicts = []

        with self._lock:
            for change in delta.changes:
                # Check for conflict
                conflict = self._check_conflict(change)

                if conflict:
                    self.stats["conflicts_detected"] += 1

                    # Try to resolve
                    resolved = self._resolve_conflict(conflict)

                    if resolved:
                        self.stats["conflicts_resolved"] += 1
                        # Apply resolved value
                        self._apply_change(change.path, conflict.resolution)
                    else:
                        conflicts.append(conflict)

                        if self.on_conflict:
                            self.on_conflict(conflict)
                else:
                    # No conflict - apply directly
                    self._apply_change(change.path, change.value)

            # Update peer version
            self._version_tracker.set_peer_version(
                delta.source_node,
                delta.to_version
            )

            self.stats["deltas_applied"] += 1

        if conflicts:
            self._sync_state = SyncState.CONFLICT
            self._unresolved_conflicts.extend(conflicts)
        else:
            self._sync_state = SyncState.COMPLETE

        return conflicts

    def _check_conflict(self, change: DeltaChange) -> Optional[SyncConflict]:
        """Check if change conflicts with local state"""
        path = change.path

        if path not in self._local_state:
            return None

        local_value = self._local_state[path]

        # Check if local value differs from expected old value
        if change.old_value is not None and local_value != change.old_value:
            return SyncConflict(
                path=path,
                local_value=local_value,
                remote_value=change.value,
                local_timestamp=datetime.now(timezone.utc),  # Would track actual
                remote_timestamp=change.timestamp,
            )

        return None

    def _resolve_conflict(self, conflict: SyncConflict) -> bool:
        """
        Attempt to resolve a conflict

        Returns:
            True if resolved
        """
        if self.conflict_resolution == ConflictResolution.HUB_AUTHORITY:
            if self.is_hub:
                conflict.resolution = conflict.local_value
            else:
                conflict.resolution = conflict.remote_value
            conflict.resolved_by = ConflictResolution.HUB_AUTHORITY
            return True

        elif self.conflict_resolution == ConflictResolution.LATEST_WINS:
            if conflict.local_timestamp > conflict.remote_timestamp:
                conflict.resolution = conflict.local_value
            else:
                conflict.resolution = conflict.remote_value
            conflict.resolved_by = ConflictResolution.LATEST_WINS
            return True

        elif self.conflict_resolution == ConflictResolution.MERGE:
            merged = self._attempt_merge(conflict.local_value, conflict.remote_value)
            if merged is not None:
                conflict.resolution = merged
                conflict.resolved_by = ConflictResolution.MERGE
                return True
            return False

        elif self.conflict_resolution == ConflictResolution.HIGHER_CONFIDENCE:
            # Would need confidence scores on values
            # Fall back to latest wins
            if conflict.local_timestamp > conflict.remote_timestamp:
                conflict.resolution = conflict.local_value
            else:
                conflict.resolution = conflict.remote_value
            conflict.resolved_by = ConflictResolution.HIGHER_CONFIDENCE
            return True

        return False

    def _attempt_merge(self, local: Any, remote: Any) -> Optional[Any]:
        """Attempt to merge two values"""
        # Handle dict merge
        if isinstance(local, dict) and isinstance(remote, dict):
            merged = {**local}
            for key, value in remote.items():
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    # Nested conflict - can't merge
                    return None
            return merged

        # Handle set merge
        if isinstance(local, set) and isinstance(remote, set):
            return local | remote

        # Handle list merge (append)
        if isinstance(local, list) and isinstance(remote, list):
            merged = list(local)
            for item in remote:
                if item not in merged:
                    merged.append(item)
            return merged

        # Can't merge other types
        return None

    def _apply_change(self, path: str, value: Any):
        """Apply a change to local state"""
        if value is None:
            if path in self._local_state:
                del self._local_state[path]
        else:
            self._local_state[path] = value

    def get_unresolved_conflicts(self) -> List[SyncConflict]:
        """Get list of unresolved conflicts"""
        return list(self._unresolved_conflicts)

    def resolve_conflict_manually(self, path: str, value: Any) -> bool:
        """
        Manually resolve a conflict

        Args:
            path: Path of conflicted item
            value: Value to use

        Returns:
            True if conflict was found and resolved
        """
        with self._lock:
            for conflict in self._unresolved_conflicts:
                if conflict.path == path:
                    conflict.resolution = value
                    conflict.resolved_by = None  # Manual
                    self._apply_change(path, value)
                    self._unresolved_conflicts.remove(conflict)
                    return True
            return False

    def get_sync_status(self) -> Dict:
        """Get synchronization status"""
        return {
            "node_id": self.node_id,
            "is_hub": self.is_hub,
            "state": self._sync_state.name,
            "current_version": self._version_tracker.get_current_version(),
            "unresolved_conflicts": len(self._unresolved_conflicts),
            "local_items": len(self._local_state),
            "conflict_resolution": self.conflict_resolution.name,
            "stats": self.stats,
        }

    def export_state(self) -> Dict:
        """Export full state for sync"""
        return {
            "node_id": self.node_id,
            "version": self._version_tracker.get_current_version(),
            "state": self._local_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def import_state(self, state: Dict):
        """Import full state (full sync)"""
        with self._lock:
            self._local_state = state.get("state", {})
            # Reset version tracker
            self._version_tracker._current_version = state.get("version", 0)


if __name__ == "__main__":
    print("Sync Protocol Self-Test")
    print("=" * 50)

    # Create hub and node sync protocols
    hub_sync = SyncProtocol("hub-001", is_hub=True)
    node_sync = SyncProtocol("node-001", is_hub=False)

    print(f"\n[1] Record Changes on Hub")
    hub_sync.record_change("add", "concepts/threat1", {"name": "APT29", "type": "threat"})
    hub_sync.record_change("add", "concepts/threat2", {"name": "Cobalt Strike", "type": "malware"})
    hub_sync.record_change("update", "concepts/threat1", {"name": "APT29", "type": "threat", "severity": "high"})
    print(f"    Hub version: {hub_sync._version_tracker.get_current_version()}")
    print(f"    Hub items: {len(hub_sync._local_state)}")

    print(f"\n[2] Generate Delta")
    delta = hub_sync.generate_delta("node-001")
    print(f"    Delta ID: {delta.delta_id}")
    print(f"    Changes: {len(delta.changes)}")
    print(f"    Version range: {delta.from_version} -> {delta.to_version}")

    print(f"\n[3] Apply Delta on Node")
    conflicts = node_sync.apply_delta(delta)
    print(f"    Conflicts: {len(conflicts)}")
    print(f"    Node items after sync: {len(node_sync._local_state)}")

    print(f"\n[4] Create Conflict")
    # Both modify same item
    hub_sync.record_change("update", "concepts/threat1", {"name": "APT29", "priority": "critical"})
    node_sync.record_change("update", "concepts/threat1", {"name": "APT29", "priority": "high"})

    # Node gets hub's delta
    delta2 = hub_sync.generate_delta("node-001")
    conflicts2 = node_sync.apply_delta(delta2)
    print(f"    Conflicts detected: {len(conflicts2)}")

    if conflicts2:
        for c in conflicts2:
            print(f"      Path: {c.path}")
            print(f"      Local: {c.local_value}")
            print(f"      Remote: {c.remote_value}")
            print(f"      Resolved: {c.resolution}")

    print(f"\n[5] Sync Status")
    print("    Hub:")
    for key, value in hub_sync.get_sync_status().items():
        if key != "stats":
            print(f"      {key}: {value}")
    print("    Node:")
    for key, value in node_sync.get_sync_status().items():
        if key != "stats":
            print(f"      {key}: {value}")

    print("\n" + "=" * 50)
    print("Sync Protocol test complete")

