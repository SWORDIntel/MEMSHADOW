#!/usr/bin/env python3
"""
Lightweight in-memory metrics registry for MEMSHADOW components.

The registry is intentionally simple: counters and latency samples are tracked
in-memory and can be queried via :func:`get_memshadow_metrics_registry`.

Extended for layer-aware bandwidth governance:
- Per-layer, per-category byte and frame counters
- Dropped/degraded frame tracking with reason codes
- Degradation mode active flags
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from statistics import mean
from typing import Any, Deque, Dict, Optional, Set, Tuple


class MemshadowMetricsRegistry:
    """Thread-safe metrics collector with layer/category awareness."""

    # Original counter keys
    _COUNTER_KEYS = (
        "memshadow_batches_sent",
        "memshadow_batches_received",
        "memshadow_conflicts_detected",
        "memshadow_psych_events_ingested",
        "memshadow_psych_messages",
        "memshadow_threat_messages",
        "memshadow_memory_messages",
        "memshadow_federation_messages",
        "memshadow_improvement_messages",
        "memshadow_unknown_messages",
        "memshadow_unknown_msg_type",
        "memshadow_parse_errors",
    )
    
    # NEW: Extended counter keys for bandwidth governance
    _EXTENDED_COUNTER_KEYS = (
        # Accepted frames
        "memshadow_frames_accepted",
        "memshadow_bytes_accepted",
        # Dropped frames
        "memshadow_frames_dropped",
        "memshadow_bytes_dropped",
        # Degraded frames
        "memshadow_frames_degraded",
        "memshadow_bytes_degraded",
        # Degradation mode events
        "memshadow_degradation_events",
        # Layer 8 hook triggers
        "memshadow_layer8_hooks_triggered",
    )
    
    # Drop reason codes
    DROP_REASON_BANDWIDTH = "bandwidth_guard"
    DROP_REASON_CONFIG_DISABLED = "config_disabled"
    DROP_REASON_PRIORITY_CUTOFF = "priority_cutoff"
    DROP_REASON_PARSE_ERROR = "parse_error"

    def __init__(self, latency_window: int = 512):
        self._lock = threading.Lock()
        
        # Original counters
        self._counters: Dict[str, int] = {key: 0 for key in self._COUNTER_KEYS}
        
        # Extended counters
        for key in self._EXTENDED_COUNTER_KEYS:
            self._counters[key] = 0
        
        # Latency samples
        self._latencies_ms: Deque[float] = deque(maxlen=latency_window)
        
        # NEW: Per-layer, per-category counters
        # Key: (layer_id, category_name) -> {"bytes": int, "frames": int}
        self._layer_category_bytes: Dict[Tuple[int, str], int] = defaultdict(int)
        self._layer_category_frames: Dict[Tuple[int, str], int] = defaultdict(int)
        
        # NEW: Dropped frames per reason
        self._dropped_per_reason: Dict[str, int] = defaultdict(int)
        
        # NEW: Degradation mode active flags
        # Key: (layer_id, category_name) -> bool
        self._degradation_active: Dict[Tuple[int, str], bool] = defaultdict(bool)
        
        # NEW: Timestamp of last update for each layer/category (for rate calculation)
        self._last_update_ns: Dict[Tuple[int, str], int] = defaultdict(lambda: time.time_ns())

    def increment(self, key: str, value: int = 1) -> None:
        """Increment a counter by value."""
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe_latency(self, duration_ms: float) -> None:
        """Record a latency sample."""
        with self._lock:
            self._latencies_ms.append(duration_ms)
    
    # =========================================================================
    # NEW: Layer/Category Tracking
    # =========================================================================
    
    def record_layer_category_bytes(
        self,
        layer_id: int,
        category: str,
        byte_count: int,
        frame_count: int = 1,
    ) -> None:
        """
        Record bytes/frames for a specific layer and category.
        
        Args:
            layer_id: DSMIL layer ID (2-9)
            category: MEMSHADOW category name (psych, threat, memory, etc.)
            byte_count: Number of bytes processed
            frame_count: Number of frames/messages
        """
        key = (layer_id, category.lower())
        with self._lock:
            self._layer_category_bytes[key] += byte_count
            self._layer_category_frames[key] += frame_count
            self._last_update_ns[key] = time.time_ns()
            # Also update global counters
            self._counters["memshadow_bytes_accepted"] += byte_count
            self._counters["memshadow_frames_accepted"] += frame_count
    
    def record_dropped_frame(
        self,
        layer_id: int,
        category: str,
        byte_count: int,
        reason: str = DROP_REASON_BANDWIDTH,
    ) -> None:
        """
        Record a dropped frame.
        
        Args:
            layer_id: DSMIL layer ID
            category: MEMSHADOW category name
            byte_count: Size of dropped frame
            reason: Reason code for drop
        """
        with self._lock:
            self._counters["memshadow_frames_dropped"] += 1
            self._counters["memshadow_bytes_dropped"] += byte_count
            self._dropped_per_reason[reason] += 1
    
    def record_degraded_frame(
        self,
        layer_id: int,
        category: str,
        byte_count: int,
    ) -> None:
        """
        Record a frame processed in degraded mode.
        
        Args:
            layer_id: DSMIL layer ID
            category: MEMSHADOW category name
            byte_count: Size of frame
        """
        key = (layer_id, category.lower())
        with self._lock:
            self._counters["memshadow_frames_degraded"] += 1
            self._counters["memshadow_bytes_degraded"] += byte_count
            # Still count as accepted for throughput
            self._layer_category_bytes[key] += byte_count
            self._layer_category_frames[key] += 1
    
    def record_degradation_event(self, layer_id: int, category: str, active: bool) -> None:
        """
        Record entry/exit from degradation mode.
        
        Args:
            layer_id: DSMIL layer ID
            category: MEMSHADOW category name
            active: True if entering, False if exiting degradation
        """
        key = (layer_id, category.lower())
        with self._lock:
            was_active = self._degradation_active[key]
            self._degradation_active[key] = active
            if active and not was_active:
                self._counters["memshadow_degradation_events"] += 1
    
    def record_layer8_hook(self) -> None:
        """Record a Layer 8 security hook trigger."""
        with self._lock:
            self._counters["memshadow_layer8_hooks_triggered"] += 1
    
    def get_layer_category_stats(
        self,
        layer_id: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get per-layer, per-category statistics.
        
        Args:
            layer_id: Filter to specific layer (None = all)
            category: Filter to specific category (None = all)
            
        Returns:
            Dict mapping "layer_id/category" to stats dict
        """
        with self._lock:
            result = {}
            seen_keys: Set[Tuple[int, str]] = set()
            seen_keys.update(self._layer_category_bytes.keys())
            seen_keys.update(self._layer_category_frames.keys())
            
            for key in seen_keys:
                key_layer, key_cat = key
                
                if layer_id is not None and key_layer != layer_id:
                    continue
                if category is not None and key_cat != category.lower():
                    continue
                
                stat_key = f"{key_layer}/{key_cat}"
                result[stat_key] = {
                    "layer_id": key_layer,
                    "category": key_cat,
                    "bytes_total": self._layer_category_bytes[key],
                    "frames_total": self._layer_category_frames[key],
                    "degradation_active": self._degradation_active[key],
                }
            
            return result
    
    def get_dropped_by_reason(self) -> Dict[str, int]:
        """Get dropped frame counts by reason."""
        with self._lock:
            return dict(self._dropped_per_reason)
    
    def get_active_degradation_modes(self) -> Dict[str, bool]:
        """Get currently active degradation modes."""
        with self._lock:
            return {
                f"{k[0]}/{k[1]}": v
                for k, v in self._degradation_active.items()
                if v
            }

    def snapshot(self) -> Dict[str, Any]:
        """Get complete metrics snapshot."""
        with self._lock:
            # Latency stats
            latency_stats = {}
            if self._latencies_ms:
                samples = list(self._latencies_ms)
                sorted_samples = sorted(samples)
                latency_stats = {
                    "memshadow_sync_latency_ms_avg": mean(samples),
                    "memshadow_sync_latency_ms_p50": sorted_samples[int(0.50 * (len(samples) - 1))],
                    "memshadow_sync_latency_ms_p95": sorted_samples[int(0.95 * (len(samples) - 1))],
                    "memshadow_sync_latency_ms_p99": sorted_samples[int(0.99 * (len(samples) - 1))] if len(samples) > 10 else sorted_samples[-1],
                    "memshadow_sync_latency_ms_samples": len(samples),
                }
            
            # Build complete snapshot
            result = {
                **self._counters,
                **latency_stats,
                "dropped_by_reason": dict(self._dropped_per_reason),
                "layer_category_stats": self.get_layer_category_stats(),
                "active_degradation_modes": self.get_active_degradation_modes(),
            }
            
            return result

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            for key in self._counters:
                self._counters[key] = 0
            self._latencies_ms.clear()
            self._layer_category_bytes.clear()
            self._layer_category_frames.clear()
            self._dropped_per_reason.clear()
            self._degradation_active.clear()
            self._last_update_ns.clear()
    
    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-compatible metrics string
        """
        lines = []
        snapshot = self.snapshot()
        
        # Simple counters
        for key, value in snapshot.items():
            if isinstance(value, (int, float)) and not key.startswith("memshadow_sync_latency"):
                lines.append(f"# TYPE {key} counter")
                lines.append(f"{key} {value}")
        
        # Latency histograms
        if "memshadow_sync_latency_ms_avg" in snapshot:
            lines.append("# TYPE memshadow_sync_latency_ms summary")
            lines.append(f"memshadow_sync_latency_ms{{quantile=\"0.5\"}} {snapshot.get('memshadow_sync_latency_ms_p50', 0)}")
            lines.append(f"memshadow_sync_latency_ms{{quantile=\"0.95\"}} {snapshot.get('memshadow_sync_latency_ms_p95', 0)}")
            lines.append(f"memshadow_sync_latency_ms{{quantile=\"0.99\"}} {snapshot.get('memshadow_sync_latency_ms_p99', 0)}")
            lines.append(f"memshadow_sync_latency_ms_sum {snapshot.get('memshadow_sync_latency_ms_avg', 0) * snapshot.get('memshadow_sync_latency_ms_samples', 0)}")
            lines.append(f"memshadow_sync_latency_ms_count {snapshot.get('memshadow_sync_latency_ms_samples', 0)}")
        
        # Per-layer stats
        for key, stats in snapshot.get("layer_category_stats", {}).items():
            layer_id = stats["layer_id"]
            category = stats["category"]
            labels = f'layer="{layer_id}",category="{category}"'
            lines.append(f"memshadow_bytes_per_layer_category{{{labels}}} {stats['bytes_total']}")
            lines.append(f"memshadow_frames_per_layer_category{{{labels}}} {stats['frames_total']}")
            if stats.get("degradation_active"):
                lines.append(f"memshadow_degradation_active{{{labels}}} 1")
        
        # Dropped by reason
        for reason, count in snapshot.get("dropped_by_reason", {}).items():
            lines.append(f"memshadow_dropped_frames{{reason=\"{reason}\"}} {count}")
        
        return "\n".join(lines)


_REGISTRY = MemshadowMetricsRegistry()


def get_memshadow_metrics_registry() -> MemshadowMetricsRegistry:
    """Return the shared metrics registry singleton."""
    return _REGISTRY
