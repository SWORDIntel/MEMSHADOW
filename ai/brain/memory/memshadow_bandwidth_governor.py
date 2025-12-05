#!/usr/bin/env python3
"""
MEMSHADOW Bandwidth & Sync Governor

Implements bandwidth management and sync control for MEMSHADOW traffic,
consistent with the 62 GB / 64 GB/s constraints from the Memory Bandwidth
Optimization guide.

Key Features:
- Per-layer, per-category rolling counters (bytes/sec, messages/sec)
- Current vs configured budget tracking
- Degradation modes (summarize, compress, batch, reduce frequency)
- Priority-aware accept/drop/degrade decisions

Usage:
    governor = MemshadowBandwidthGovernor()
    
    # Check if frame should be accepted
    decision = governor.should_accept(frame, category, layer)
    if decision == AcceptDecision.ACCEPT:
        process_frame(frame)
    elif decision == AcceptDecision.DEGRADE:
        process_degraded_frame(frame)
    else:
        drop_frame(frame)
    
    # Get sync mode recommendation
    mode = governor.choose_sync_mode(category, layer)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

from .memshadow_layer_mapping import (
    DSMILLayer,
    MemshadowCategory,
    LAYER_BANDWIDTH_BUDGETS_GBPS,
    get_target_layers_for_category,
    get_category_priority_weight,
    can_degrade_category,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AcceptDecision",
    "DegradationMode",
    "SyncMode",
    "BandwidthSample",
    "LayerCategoryStats",
    "GovernorConfig",
    "MemshadowBandwidthGovernor",
    "get_bandwidth_governor",
]


class AcceptDecision(Enum):
    """Decision for whether to accept a frame."""
    ACCEPT = auto()      # Accept and process normally
    DEGRADE = auto()     # Accept but apply degradation (summarize, compress)
    DROP = auto()        # Drop the frame (over budget)
    DEFER = auto()       # Defer processing to next sync window


class DegradationMode(Enum):
    """Active degradation modes."""
    NONE = auto()           # Normal operation
    SUMMARIZE = auto()      # Aggregate events instead of full payloads
    COMPRESS = auto()       # Force compression
    BATCH_INCREASE = auto() # Increase batch sizes
    REDUCE_FREQUENCY = auto()  # Reduce sync frequency
    CRITICAL_ONLY = auto()  # Only process CRITICAL/EMERGENCY


class SyncMode(Enum):
    """Sync mode for memory operations."""
    FULL = auto()        # Full synchronization
    DELTA = auto()       # Delta/incremental sync only
    SUMMARY = auto()     # Summary sync (metadata only)
    DISABLED = auto()    # Sync disabled for this layer/category


class Priority(IntEnum):
    """Message priority levels (matches dsmil_protocol.Priority)."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class BandwidthSample:
    """A single bandwidth measurement sample."""
    timestamp_ns: int
    bytes_count: int
    message_count: int = 1


@dataclass
class LayerCategoryStats:
    """
    Rolling statistics for a specific layer+category combination.
    
    Tracks bytes/sec and messages/sec over a configurable window.
    """
    layer: DSMILLayer
    category: MemshadowCategory
    window_seconds: float = 60.0
    max_samples: int = 1000
    
    # Internal state
    _samples: Deque[BandwidthSample] = field(default_factory=lambda: deque(maxlen=1000))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Counters
    total_bytes_accepted: int = 0
    total_bytes_dropped: int = 0
    total_messages_accepted: int = 0
    total_messages_dropped: int = 0
    total_messages_degraded: int = 0
    degradation_events: int = 0
    
    def record_sample(self, bytes_count: int, message_count: int = 1) -> None:
        """Record a bandwidth sample."""
        sample = BandwidthSample(
            timestamp_ns=time.time_ns(),
            bytes_count=bytes_count,
            message_count=message_count,
        )
        with self._lock:
            self._samples.append(sample)
            self.total_bytes_accepted += bytes_count
            self.total_messages_accepted += message_count
    
    def record_dropped(self, bytes_count: int, message_count: int = 1) -> None:
        """Record a dropped frame."""
        with self._lock:
            self.total_bytes_dropped += bytes_count
            self.total_messages_dropped += message_count
    
    def record_degraded(self, message_count: int = 1) -> None:
        """Record a degraded frame."""
        with self._lock:
            self.total_messages_degraded += message_count
    
    def record_degradation_event(self) -> None:
        """Record entry into degradation mode."""
        with self._lock:
            self.degradation_events += 1
    
    def get_bytes_per_second(self) -> float:
        """Calculate rolling bytes per second."""
        return self._calculate_rate("bytes")
    
    def get_messages_per_second(self) -> float:
        """Calculate rolling messages per second."""
        return self._calculate_rate("messages")
    
    def _calculate_rate(self, metric: str) -> float:
        """Calculate rate over the rolling window."""
        now_ns = time.time_ns()
        window_ns = int(self.window_seconds * 1e9)
        cutoff_ns = now_ns - window_ns
        
        with self._lock:
            # Filter to window
            total = 0
            earliest_ns = now_ns
            for sample in self._samples:
                if sample.timestamp_ns >= cutoff_ns:
                    if metric == "bytes":
                        total += sample.bytes_count
                    else:
                        total += sample.message_count
                    earliest_ns = min(earliest_ns, sample.timestamp_ns)
            
            if earliest_ns >= now_ns:
                return 0.0
            
            duration_sec = (now_ns - earliest_ns) / 1e9
            if duration_sec <= 0:
                return 0.0
            
            return total / duration_sec
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary."""
        return {
            "layer": self.layer.name,
            "category": self.category.name,
            "bytes_per_second": self.get_bytes_per_second(),
            "messages_per_second": self.get_messages_per_second(),
            "total_bytes_accepted": self.total_bytes_accepted,
            "total_bytes_dropped": self.total_bytes_dropped,
            "total_messages_accepted": self.total_messages_accepted,
            "total_messages_dropped": self.total_messages_dropped,
            "total_messages_degraded": self.total_messages_degraded,
            "degradation_events": self.degradation_events,
        }
    
    def clear_samples(self) -> None:
        """Clear sample history (for testing)."""
        with self._lock:
            self._samples.clear()


@dataclass
class GovernorConfig:
    """
    Configuration for the bandwidth governor.
    
    Loaded from config/memshadow_config.py with these additions.
    """
    # Overall system constraints
    total_bandwidth_bytes_per_sec: int = 64 * 1024**3  # 64 GB/s
    memshadow_bandwidth_percent: float = 0.15  # 15% of total for MEMSHADOW
    
    # Per-layer budget overrides (None = use calculated from layer mapping)
    layer_budgets_bytes_per_sec: Dict[DSMILLayer, int] = field(default_factory=dict)
    
    # Per-category budget overrides
    category_budgets_bytes_per_sec: Dict[MemshadowCategory, int] = field(default_factory=dict)
    
    # Degradation thresholds (% of budget)
    degrade_threshold_percent: float = 0.80  # Start degrading at 80%
    drop_threshold_percent: float = 0.95     # Start dropping at 95%
    
    # Priority cutoff for degradation (never degrade these)
    never_degrade_priority: int = Priority.CRITICAL  # CRITICAL and EMERGENCY never degraded
    never_drop_priority: int = Priority.EMERGENCY    # EMERGENCY never dropped
    
    # Sync mode defaults
    default_sync_mode: SyncMode = SyncMode.DELTA
    degraded_sync_mode: SyncMode = SyncMode.SUMMARY
    
    # Compression settings
    force_compression_threshold_bytes: int = 512  # Force compression above this size
    
    # Batch settings
    normal_batch_size: int = 100
    degraded_batch_size: int = 500
    
    # Frequency reduction
    normal_sync_interval_ms: int = 1000
    degraded_sync_interval_ms: int = 5000
    
    # Rolling window for rate calculation
    stats_window_seconds: float = 60.0
    
    # Monitor-only mode (observe but don't enforce)
    monitor_only: bool = False
    
    def get_layer_budget(self, layer: DSMILLayer) -> int:
        """Get effective budget for a layer in bytes/sec."""
        if layer in self.layer_budgets_bytes_per_sec:
            return self.layer_budgets_bytes_per_sec[layer]
        
        # Calculate from overall budget and layer weights
        memshadow_budget = int(self.total_bandwidth_bytes_per_sec * self.memshadow_bandwidth_percent)
        layer_share = LAYER_BANDWIDTH_BUDGETS_GBPS.get(layer, 1.0)
        total_layer_budget = sum(LAYER_BANDWIDTH_BUDGETS_GBPS.values())
        
        return int(memshadow_budget * (layer_share / total_layer_budget))
    
    def get_category_budget(self, category: MemshadowCategory) -> int:
        """Get effective budget for a category in bytes/sec."""
        if category in self.category_budgets_bytes_per_sec:
            return self.category_budgets_bytes_per_sec[category]
        
        # Calculate from priority weight
        memshadow_budget = int(self.total_bandwidth_bytes_per_sec * self.memshadow_bandwidth_percent)
        weight = get_category_priority_weight(category)
        total_weight = sum(get_category_priority_weight(c) for c in MemshadowCategory if c != MemshadowCategory.UNKNOWN)
        
        return int(memshadow_budget * (weight / total_weight)) if total_weight > 0 else memshadow_budget // 5


class MemshadowBandwidthGovernor:
    """
    Central bandwidth and sync governor for MEMSHADOW traffic.
    
    Tracks bandwidth usage across layers and categories, makes accept/drop/degrade
    decisions, and recommends sync modes based on current load.
    """
    
    def __init__(self, config: Optional[GovernorConfig] = None):
        """
        Initialize the governor.
        
        Args:
            config: Governor configuration (uses defaults if None)
        """
        self._config = config or GovernorConfig()
        self._lock = threading.RLock()
        
        # Per-layer, per-category statistics
        self._stats: Dict[Tuple[DSMILLayer, MemshadowCategory], LayerCategoryStats] = {}
        
        # Current degradation modes
        self._degradation_modes: Dict[Tuple[DSMILLayer, MemshadowCategory], DegradationMode] = defaultdict(
            lambda: DegradationMode.NONE
        )
        
        # Callbacks for Layer 8 hooks (security stack)
        self._layer8_hooks: List[callable] = []
        
        logger.info(
            "MemshadowBandwidthGovernor initialized: "
            f"memshadow_budget={self._config.memshadow_bandwidth_percent*100:.1f}%, "
            f"degrade_at={self._config.degrade_threshold_percent*100:.0f}%, "
            f"drop_at={self._config.drop_threshold_percent*100:.0f}%, "
            f"monitor_only={self._config.monitor_only}"
        )
    
    @property
    def config(self) -> GovernorConfig:
        """Get current configuration."""
        return self._config
    
    def update_config(self, config: GovernorConfig) -> None:
        """Update governor configuration."""
        with self._lock:
            self._config = config
            logger.info("Governor config updated")
    
    # =========================================================================
    # Accept/Drop/Degrade Decisions
    # =========================================================================
    
    def should_accept(
        self,
        payload_bytes: int,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
        priority: int = Priority.NORMAL,
        record_sample: bool = True,
    ) -> AcceptDecision:
        """
        Decide whether to accept, degrade, or drop a frame.
        
        Args:
            payload_bytes: Size of the payload in bytes
            category: MEMSHADOW category
            layer: Target layer (auto-detected from category if None)
            priority: Message priority (0-4)
            record_sample: Whether to record this in statistics
            
        Returns:
            AcceptDecision indicating how to handle the frame
        """
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        
        # Never drop EMERGENCY
        if priority >= self._config.never_drop_priority:
            if record_sample:
                self._record_accepted(target_layer, cat, payload_bytes)
            return AcceptDecision.ACCEPT
        
        # Check budget utilization
        utilization = self._get_utilization(target_layer, cat)
        
        # Monitor-only mode: always accept but track
        if self._config.monitor_only:
            if record_sample:
                self._record_accepted(target_layer, cat, payload_bytes)
            return AcceptDecision.ACCEPT
        
        # Under degrade threshold: accept normally
        if utilization < self._config.degrade_threshold_percent:
            if record_sample:
                self._record_accepted(target_layer, cat, payload_bytes)
            return AcceptDecision.ACCEPT
        
        # Between degrade and drop thresholds
        if utilization < self._config.drop_threshold_percent:
            # Never degrade CRITICAL
            if priority >= self._config.never_degrade_priority:
                if record_sample:
                    self._record_accepted(target_layer, cat, payload_bytes)
                return AcceptDecision.ACCEPT
            
            # Check if category allows degradation
            if not can_degrade_category(cat):
                if record_sample:
                    self._record_accepted(target_layer, cat, payload_bytes)
                return AcceptDecision.ACCEPT
            
            # Enter degradation mode
            self._enter_degradation_mode(target_layer, cat)
            if record_sample:
                self._record_degraded(target_layer, cat, payload_bytes)
            return AcceptDecision.DEGRADE
        
        # Over drop threshold
        # Never drop high priority
        if priority >= self._config.never_degrade_priority:
            if record_sample:
                self._record_accepted(target_layer, cat, payload_bytes)
            return AcceptDecision.ACCEPT
        
        # Drop low-priority traffic
        if priority <= Priority.NORMAL:
            if record_sample:
                self._record_dropped(target_layer, cat, payload_bytes)
            logger.debug(
                f"Dropping frame: layer={target_layer.name}, cat={cat.name}, "
                f"size={payload_bytes}, util={utilization:.1%}"
            )
            return AcceptDecision.DROP
        
        # Degrade HIGH priority
        self._enter_degradation_mode(target_layer, cat)
        if record_sample:
            self._record_degraded(target_layer, cat, payload_bytes)
        return AcceptDecision.DEGRADE
    
    # =========================================================================
    # Sync Mode Selection
    # =========================================================================
    
    def choose_sync_mode(
        self,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
    ) -> SyncMode:
        """
        Choose the appropriate sync mode for a category/layer.
        
        Args:
            category: MEMSHADOW category
            layer: Target layer (auto-detected if None)
            
        Returns:
            Recommended SyncMode
        """
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        
        key = (target_layer, cat)
        degradation = self._degradation_modes.get(key, DegradationMode.NONE)
        
        if degradation == DegradationMode.NONE:
            return self._config.default_sync_mode
        
        if degradation == DegradationMode.CRITICAL_ONLY:
            return SyncMode.DISABLED
        
        return self._config.degraded_sync_mode
    
    def get_sync_interval_ms(
        self,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
    ) -> int:
        """Get recommended sync interval in milliseconds."""
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        
        key = (target_layer, cat)
        degradation = self._degradation_modes.get(key, DegradationMode.NONE)
        
        if degradation in (DegradationMode.REDUCE_FREQUENCY, DegradationMode.CRITICAL_ONLY):
            return self._config.degraded_sync_interval_ms
        
        return self._config.normal_sync_interval_ms
    
    def get_batch_size(
        self,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
    ) -> int:
        """Get recommended batch size."""
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        
        key = (target_layer, cat)
        degradation = self._degradation_modes.get(key, DegradationMode.NONE)
        
        if degradation == DegradationMode.BATCH_INCREASE:
            return self._config.degraded_batch_size
        
        return self._config.normal_batch_size
    
    def should_compress(self, payload_bytes: int) -> bool:
        """Check if payload should be compressed."""
        return payload_bytes >= self._config.force_compression_threshold_bytes
    
    # =========================================================================
    # Degradation Mode Management
    # =========================================================================
    
    def get_degradation_mode(
        self,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
    ) -> DegradationMode:
        """Get current degradation mode for a layer/category."""
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        return self._degradation_modes.get((target_layer, cat), DegradationMode.NONE)
    
    def _enter_degradation_mode(
        self,
        layer: DSMILLayer,
        category: MemshadowCategory,
    ) -> None:
        """Enter or escalate degradation mode."""
        key = (layer, category)
        current = self._degradation_modes.get(key, DegradationMode.NONE)
        
        # Escalation order
        escalation = [
            DegradationMode.NONE,
            DegradationMode.COMPRESS,
            DegradationMode.BATCH_INCREASE,
            DegradationMode.SUMMARIZE,
            DegradationMode.REDUCE_FREQUENCY,
            DegradationMode.CRITICAL_ONLY,
        ]
        
        current_idx = escalation.index(current) if current in escalation else 0
        next_idx = min(current_idx + 1, len(escalation) - 1)
        
        if escalation[next_idx] != current:
            self._degradation_modes[key] = escalation[next_idx]
            stats = self._get_or_create_stats(layer, category)
            stats.record_degradation_event()
            logger.warning(
                f"Degradation mode escalated: {layer.name}/{category.name} "
                f"-> {escalation[next_idx].name}"
            )
    
    def exit_degradation_mode(
        self,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
    ) -> None:
        """Exit degradation mode (e.g., when load decreases)."""
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        
        key = (target_layer, cat)
        if key in self._degradation_modes:
            old_mode = self._degradation_modes[key]
            self._degradation_modes[key] = DegradationMode.NONE
            logger.info(
                f"Exited degradation mode: {target_layer.name}/{cat.name} "
                f"(was {old_mode.name})"
            )
    
    def check_and_update_degradation_modes(self) -> Dict[Tuple[DSMILLayer, MemshadowCategory], DegradationMode]:
        """
        Check utilization and update degradation modes accordingly.
        
        Should be called periodically (e.g., every stats window).
        
        Returns:
            Current degradation modes
        """
        with self._lock:
            for key, stats in self._stats.items():
                layer, category = key
                utilization = self._get_utilization(layer, category)
                
                if utilization < self._config.degrade_threshold_percent * 0.5:
                    # Well under threshold, can exit degradation
                    if self._degradation_modes.get(key, DegradationMode.NONE) != DegradationMode.NONE:
                        self.exit_degradation_mode(category, layer)
            
            return dict(self._degradation_modes)
    
    # =========================================================================
    # Layer 8 Hooks
    # =========================================================================
    
    def register_layer8_hook(self, callback: callable) -> None:
        """
        Register a callback for Layer 8 security events.
        
        Callback signature: callback(category, priority, payload_bytes, metadata)
        """
        self._layer8_hooks.append(callback)
    
    def trigger_layer8_hooks(
        self,
        category: MemshadowCategory,
        priority: int,
        payload_bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Trigger Layer 8 hooks if conditions are met."""
        from .memshadow_layer_mapping import should_trigger_layer8_hook
        
        if not should_trigger_layer8_hook(category, priority):
            return
        
        for hook in self._layer8_hooks:
            try:
                hook(category, priority, payload_bytes, metadata or {})
            except Exception as e:
                logger.warning(f"Layer 8 hook failed: {e}")
    
    # =========================================================================
    # Statistics & Reporting
    # =========================================================================
    
    def get_stats(
        self,
        category: Optional[MemshadowCategory | str | int] = None,
        layer: Optional[DSMILLayer] = None,
    ) -> Dict[str, Any]:
        """
        Get bandwidth statistics.
        
        Args:
            category: Filter to specific category (None = all)
            layer: Filter to specific layer (None = all)
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            result: Dict[str, Any] = {
                "config": {
                    "memshadow_bandwidth_percent": self._config.memshadow_bandwidth_percent,
                    "degrade_threshold_percent": self._config.degrade_threshold_percent,
                    "drop_threshold_percent": self._config.drop_threshold_percent,
                    "monitor_only": self._config.monitor_only,
                },
                "layer_category_stats": [],
                "degradation_modes": {},
            }
            
            cat_filter = self._normalize_category(category) if category else None
            
            for key, stats in self._stats.items():
                stat_layer, stat_cat = key
                
                if cat_filter and stat_cat != cat_filter:
                    continue
                if layer and stat_layer != layer:
                    continue
                
                stat_dict = stats.get_stats_dict()
                stat_dict["budget_bytes_per_sec"] = self._config.get_layer_budget(stat_layer)
                stat_dict["utilization_percent"] = self._get_utilization(stat_layer, stat_cat) * 100
                result["layer_category_stats"].append(stat_dict)
            
            for key, mode in self._degradation_modes.items():
                if mode != DegradationMode.NONE:
                    layer_name, cat_name = key[0].name, key[1].name
                    result["degradation_modes"][f"{layer_name}/{cat_name}"] = mode.name
            
            return result
    
    def get_utilization(
        self,
        category: MemshadowCategory | str | int,
        layer: Optional[DSMILLayer] = None,
    ) -> float:
        """
        Get current utilization as a fraction (0.0 to 1.0+).
        
        Args:
            category: MEMSHADOW category
            layer: Target layer
            
        Returns:
            Utilization as fraction of budget
        """
        cat = self._normalize_category(category)
        target_layer = layer or self._get_primary_layer(cat)
        return self._get_utilization(target_layer, cat)
    
    def reset_stats(self) -> None:
        """Reset all statistics (for testing)."""
        with self._lock:
            for stats in self._stats.values():
                stats.clear_samples()
            self._degradation_modes.clear()
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _normalize_category(self, category: MemshadowCategory | str | int) -> MemshadowCategory:
        """Normalize category to MemshadowCategory enum."""
        if isinstance(category, MemshadowCategory):
            return category
        if isinstance(category, str):
            return MemshadowCategory.from_string(category)
        if isinstance(category, int):
            if category <= 5:
                return MemshadowCategory(category)
            return MemshadowCategory.from_msg_type_value(category)
        return MemshadowCategory.UNKNOWN
    
    def _get_primary_layer(self, category: MemshadowCategory) -> DSMILLayer:
        """Get the primary layer for a category."""
        layers = get_target_layers_for_category(category, include_secondary=False)
        if layers:
            return min(layers, key=lambda l: l.value)  # Lowest layer number
        return DSMILLayer.PRIMARY_AI_MEMORY  # Default
    
    def _get_or_create_stats(
        self,
        layer: DSMILLayer,
        category: MemshadowCategory,
    ) -> LayerCategoryStats:
        """Get or create statistics for a layer/category pair."""
        key = (layer, category)
        if key not in self._stats:
            self._stats[key] = LayerCategoryStats(
                layer=layer,
                category=category,
                window_seconds=self._config.stats_window_seconds,
            )
        return self._stats[key]
    
    def _get_utilization(
        self,
        layer: DSMILLayer,
        category: MemshadowCategory,
    ) -> float:
        """Get utilization as fraction of budget."""
        key = (layer, category)
        if key not in self._stats:
            return 0.0
        
        stats = self._stats[key]
        current_bps = stats.get_bytes_per_second()
        budget_bps = self._config.get_layer_budget(layer)
        
        if budget_bps <= 0:
            return 0.0
        
        return current_bps / budget_bps
    
    def _record_accepted(
        self,
        layer: DSMILLayer,
        category: MemshadowCategory,
        payload_bytes: int,
    ) -> None:
        """Record an accepted frame."""
        stats = self._get_or_create_stats(layer, category)
        stats.record_sample(payload_bytes)
    
    def _record_dropped(
        self,
        layer: DSMILLayer,
        category: MemshadowCategory,
        payload_bytes: int,
    ) -> None:
        """Record a dropped frame."""
        stats = self._get_or_create_stats(layer, category)
        stats.record_dropped(payload_bytes)
    
    def _record_degraded(
        self,
        layer: DSMILLayer,
        category: MemshadowCategory,
        payload_bytes: int,
    ) -> None:
        """Record a degraded frame."""
        stats = self._get_or_create_stats(layer, category)
        stats.record_sample(payload_bytes)
        stats.record_degraded()


# =============================================================================
# Module-Level Singleton
# =============================================================================

_GOVERNOR: Optional[MemshadowBandwidthGovernor] = None
_GOVERNOR_LOCK = threading.Lock()


def get_bandwidth_governor(config: Optional[GovernorConfig] = None) -> MemshadowBandwidthGovernor:
    """
    Get the singleton bandwidth governor instance.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        Shared MemshadowBandwidthGovernor instance
    """
    global _GOVERNOR
    
    with _GOVERNOR_LOCK:
        if _GOVERNOR is None:
            _GOVERNOR = MemshadowBandwidthGovernor(config)
        elif config is not None:
            _GOVERNOR.update_config(config)
        return _GOVERNOR


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("MEMSHADOW Bandwidth Governor Self-Test")
    print("=" * 60)
    
    # Create governor with test config
    config = GovernorConfig(
        memshadow_bandwidth_percent=0.15,
        degrade_threshold_percent=0.80,
        drop_threshold_percent=0.95,
        stats_window_seconds=10.0,
    )
    governor = MemshadowBandwidthGovernor(config)
    
    print("\n[1] Configuration:")
    print(f"    MEMSHADOW Budget: {config.memshadow_bandwidth_percent*100:.1f}%")
    print(f"    Degrade Threshold: {config.degrade_threshold_percent*100:.0f}%")
    print(f"    Drop Threshold: {config.drop_threshold_percent*100:.0f}%")
    
    print("\n[2] Per-Layer Budgets:")
    for layer in DSMILLayer:
        budget = config.get_layer_budget(layer)
        print(f"    {layer.name}: {budget / 1e9:.2f} GB/s")
    
    print("\n[3] Accept/Drop Decisions:")
    test_cases = [
        (1000, MemshadowCategory.PSYCH, Priority.NORMAL),
        (1000, MemshadowCategory.THREAT, Priority.HIGH),
        (1000, MemshadowCategory.MEMORY, Priority.CRITICAL),
        (1000, MemshadowCategory.IMPROVEMENT, Priority.LOW),
    ]
    
    for payload, cat, priority in test_cases:
        decision = governor.should_accept(payload, cat, priority=priority)
        print(f"    {cat.name}@{priority.name}: {decision.name}")
    
    print("\n[4] Sync Mode Recommendations:")
    for cat in [MemshadowCategory.PSYCH, MemshadowCategory.THREAT, MemshadowCategory.MEMORY]:
        mode = governor.choose_sync_mode(cat)
        interval = governor.get_sync_interval_ms(cat)
        batch = governor.get_batch_size(cat)
        print(f"    {cat.name}: mode={mode.name}, interval={interval}ms, batch={batch}")
    
    print("\n[5] Simulated Load Test:")
    # Simulate traffic to trigger degradation
    for i in range(100):
        # Simulate 100 MB messages
        governor.should_accept(100 * 1024 * 1024, MemshadowCategory.PSYCH, priority=Priority.NORMAL)
    
    stats = governor.get_stats(MemshadowCategory.PSYCH)
    print(f"    PSYCH Stats: {len(stats['layer_category_stats'])} layer(s) tracked")
    for stat in stats["layer_category_stats"]:
        print(f"        {stat['layer']}: {stat['bytes_per_second']/1e6:.1f} MB/s "
              f"({stat['utilization_percent']:.1f}% util)")
    
    if stats["degradation_modes"]:
        print(f"    Degradation Modes: {stats['degradation_modes']}")
    
    print("\n" + "=" * 60)
    print("MEMSHADOW Bandwidth Governor test complete")
