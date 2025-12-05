#!/usr/bin/env python3
"""
Lightweight in-memory metrics registry for MEMSHADOW components.

The registry is intentionally simple: counters and latency samples are tracked
in-memory and can be queried via :func:`get_memshadow_metrics_registry`.
"""

from __future__ import annotations

import threading
from collections import deque
from statistics import mean
from typing import Any, Deque, Dict


class MemshadowMetricsRegistry:
    """Thread-safe metrics collector."""

    _COUNTER_KEYS = (
        "memshadow_batches_sent",
        "memshadow_batches_received",
        "memshadow_conflicts_detected",
        "memshadow_psych_events_ingested",
    )

    def __init__(self, latency_window: int = 512):
        self._counters: Dict[str, int] = {key: 0 for key in self._COUNTER_KEYS}
        self._latencies_ms: Deque[float] = deque(maxlen=latency_window)
        self._lock = threading.Lock()

    def increment(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe_latency(self, duration_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(duration_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latency_stats = {}
            if self._latencies_ms:
                samples = list(self._latencies_ms)
                latency_stats = {
                    "memshadow_sync_latency_ms_avg": mean(samples),
                    "memshadow_sync_latency_ms_p95": sorted(samples)[int(0.95 * (len(samples) - 1))],
                    "memshadow_sync_latency_ms_samples": len(samples),
                }

            return {**self._counters, **latency_stats}

    def reset(self) -> None:
        with self._lock:
            for key in self._counters:
                self._counters[key] = 0
            self._latencies_ms.clear()


_REGISTRY = MemshadowMetricsRegistry()


def get_memshadow_metrics_registry() -> MemshadowMetricsRegistry:
    """Return the shared metrics registry singleton."""
    return _REGISTRY
