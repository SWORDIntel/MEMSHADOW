#!/usr/bin/env python3
"""
Self-Improvement Tracker for DSMIL Brain Federation

Tracks local performance metrics, detects statistically significant improvements,
and packages improvements for propagation across the mesh network.

Uses hybrid propagation: critical updates go direct P2P, normal updates via hub.
"""

import time
import threading
import logging
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from statistics import mean, stdev
import json

from .improvement_types import (
    ImprovementPackage, ImprovementType, ImprovementPriority,
    CompatibilityLevel, PerformanceMetrics, ModelWeightDelta,
    ConfigTuning, LearnedPattern, ImprovementAnnouncement
)

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: datetime
    metrics: PerformanceMetrics
    context: Dict[str, Any] = field(default_factory=dict)


class ImprovementTracker:
    """
    Tracks local improvements and packages them for propagation

    Monitors performance metrics over time, detects improvements,
    and creates improvement packages ready for mesh propagation.
    """

    def __init__(self, node_id: str, min_improvement_threshold: float = 5.0,
                 statistical_significance_level: float = 0.95,
                 min_samples_for_detection: int = 10):
        """
        Initialize improvement tracker

        Args:
            node_id: Unique identifier for this node
            min_improvement_threshold: Minimum improvement % to consider (default 5%)
            statistical_significance_level: Confidence level for statistical tests (0.95 = 95%)
            min_samples_for_detection: Minimum samples needed before detecting improvement
        """
        self.node_id = node_id
        self.min_improvement_threshold = min_improvement_threshold
        self.statistical_significance_level = statistical_significance_level
        self.min_samples_for_detection = min_samples_for_detection

        # Metric history (rolling window)
        self._metric_history: Dict[str, deque] = {}  # component_name -> deque of MetricSnapshot
        self._history_lock = threading.RLock()

        # Baseline metrics (last known good state)
        self._baselines: Dict[str, PerformanceMetrics] = {}

        # Tracked improvements (pending propagation)
        self._pending_improvements: Dict[str, ImprovementPackage] = {}
        self._improvements_lock = threading.RLock()

        # Callbacks
        self.on_improvement_detected: Optional[Callable[[ImprovementPackage], None]] = None

        # Statistics
        self.stats = {
            "improvements_detected": 0,
            "improvements_propagated": 0,
            "improvements_applied": 0,
            "improvements_rejected": 0,
        }

        logger.info(f"ImprovementTracker initialized for node {node_id}")

    def record_metrics(self, component_name: str, metrics: PerformanceMetrics,
                      context: Optional[Dict[str, Any]] = None):
        """
        Record performance metrics for a component

        Args:
            component_name: Name of the component (e.g., "risk_assessment", "memory_l1")
            metrics: Current performance metrics
            context: Additional context (model version, config hash, etc.)
        """
        snapshot = MetricSnapshot(
            timestamp=datetime.now(timezone.utc),
            metrics=metrics,
            context=context or {}
        )

        with self._history_lock:
            if component_name not in self._metric_history:
                self._metric_history[component_name] = deque(maxlen=1000)

            self._metric_history[component_name].append(snapshot)

            # Update baseline if this is better than current baseline
            if component_name not in self._baselines:
                self._baselines[component_name] = metrics
            else:
                baseline = self._baselines[component_name]
                if metrics.improvement_percentage(baseline) > 0:
                    # New baseline established
                    self._baselines[component_name] = metrics

    def detect_improvement(self, component_name: str,
                          improvement_type: ImprovementType,
                          improvement_data: Any,
                          requires_version: Optional[str] = None,
                          requires_architecture: Optional[str] = None,
                          requires_capabilities: Optional[List[str]] = None) -> Optional[ImprovementPackage]:
        """
        Detect if there's been a statistically significant improvement

        Args:
            component_name: Component that improved
            improvement_type: Type of improvement (weights, config, pattern)
            improvement_data: The actual improvement data
            requires_version: Version requirement (if any)
            requires_architecture: Architecture requirement (if any)
            requires_capabilities: Required capabilities (if any)

        Returns:
            ImprovementPackage if improvement detected, None otherwise
        """
        with self._history_lock:
            if component_name not in self._metric_history:
                logger.debug(f"No metric history for {component_name}")
                return None

            history = self._metric_history[component_name]
            if len(history) < self.min_samples_for_detection:
                logger.debug(f"Not enough samples for {component_name}: {len(history)} < {self.min_samples_for_detection}")
                return None

            # Get baseline (from before improvement)
            baseline = self._baselines.get(component_name)
            if not baseline:
                baseline = history[0].metrics

            # Get recent metrics (after improvement)
            recent_snapshots = list(history)[-self.min_samples_for_detection:]
            recent_metrics = [s.metrics for s in recent_snapshots]

            # Calculate average recent metrics
            avg_recent = PerformanceMetrics(
                accuracy=mean([m.accuracy for m in recent_metrics]),
                precision=mean([m.precision for m in recent_metrics]),
                recall=mean([m.recall for m in recent_metrics]),
                f1_score=mean([m.f1_score for m in recent_metrics]),
                latency_ms=mean([m.latency_ms for m in recent_metrics]),
                throughput=mean([m.throughput for m in recent_metrics]),
                confidence=mean([m.confidence for m in recent_metrics]),
                error_rate=mean([m.error_rate for m in recent_metrics]),
            )

            # Calculate improvement percentage
            improvement_pct = avg_recent.improvement_percentage(baseline)

            if improvement_pct < self.min_improvement_threshold:
                logger.debug(f"Improvement {improvement_pct:.2f}% below threshold {self.min_improvement_threshold}%")
                return None

            # Statistical significance test (simple t-test approximation)
            if not self._is_statistically_significant(history, baseline, avg_recent):
                logger.debug(f"Improvement not statistically significant")
                return None

            # Determine priority based on improvement magnitude
            if improvement_pct >= 15.0:
                priority = ImprovementPriority.CRITICAL
            elif improvement_pct >= 10.0:
                priority = ImprovementPriority.HIGH
            elif improvement_pct >= 7.0:
                priority = ImprovementPriority.NORMAL
            else:
                priority = ImprovementPriority.LOW

            # Determine compatibility level
            compatibility = CompatibilityLevel.UNIVERSAL
            if requires_version or requires_architecture or requires_capabilities:
                if requires_architecture:
                    compatibility = CompatibilityLevel.ARCHITECTURE_SPECIFIC
                elif requires_version:
                    compatibility = CompatibilityLevel.VERSION_SPECIFIC
                else:
                    compatibility = CompatibilityLevel.CUSTOM

            # Create improvement package
            improvement_id = str(uuid.uuid4())

            package = ImprovementPackage(
                improvement_id=improvement_id,
                improvement_type=improvement_type,
                priority=priority,
                compatibility=compatibility,
                source_node_id=self.node_id,
                baseline_metrics=baseline,
                improved_metrics=avg_recent,
                improvement_percentage=improvement_pct,
                requires_version=requires_version,
                requires_architecture=requires_architecture,
                requires_capabilities=requires_capabilities or [],
            )

            # Attach improvement data
            if improvement_type == ImprovementType.MODEL_WEIGHTS:
                package.weight_delta = improvement_data
            elif improvement_type == ImprovementType.CONFIG_TUNING:
                package.config_tuning = improvement_data
            elif improvement_type == ImprovementType.LEARNED_PATTERN:
                package.learned_pattern = improvement_data
            else:
                package.raw_data = improvement_data.pack() if hasattr(improvement_data, 'pack') else json.dumps(improvement_data).encode()
                package.raw_data_type = improvement_type.name

            # Compute validation hash
            package.validation_hash = package.compute_hash()

            # Store pending improvement
            with self._improvements_lock:
                self._pending_improvements[improvement_id] = package

            self.stats["improvements_detected"] += 1

            logger.info(f"Improvement detected: {improvement_id} ({improvement_pct:.2f}% improvement, priority={priority.name})")

            # Notify callback
            if self.on_improvement_detected:
                try:
                    self.on_improvement_detected(package)
                except Exception as e:
                    logger.error(f"Error in improvement callback: {e}")

            return package

    def _is_statistically_significant(self, history: deque, baseline: PerformanceMetrics,
                                     improved: PerformanceMetrics) -> bool:
        """
        Check if improvement is statistically significant

        Uses a simple approach: compare means with standard deviation check.
        In production, would use proper t-test or Mann-Whitney U test.
        """
        if len(history) < self.min_samples_for_detection:
            return False

        # Get accuracy values from history
        accuracies = [s.metrics.accuracy for s in history]

        if len(accuracies) < 2:
            return False

        # Simple check: improved mean should be > baseline + some margin
        baseline_acc = baseline.accuracy
        improved_acc = improved.accuracy

        if improved_acc <= baseline_acc:
            return False

        # Check if improvement is beyond noise (using standard deviation)
        try:
            acc_std = stdev(accuracies)
            improvement_margin = improved_acc - baseline_acc

            # Improvement should be at least 2 standard deviations above baseline
            # (simplified statistical test)
            if improvement_margin > 2 * acc_std:
                return True
        except:
            # If we can't compute std, use simple threshold
            pass

        # Fallback: improvement must be substantial
        return (improved_acc - baseline_acc) > 0.01  # At least 1% absolute improvement

    def get_pending_improvements(self) -> List[ImprovementPackage]:
        """Get all pending improvements ready for propagation"""
        with self._improvements_lock:
            return list(self._pending_improvements.values())

    def get_improvement(self, improvement_id: str) -> Optional[ImprovementPackage]:
        """Get a specific improvement by ID"""
        with self._improvements_lock:
            return self._pending_improvements.get(improvement_id)

    def mark_propagated(self, improvement_id: str):
        """Mark an improvement as propagated"""
        with self._improvements_lock:
            if improvement_id in self._pending_improvements:
                del self._pending_improvements[improvement_id]
                self.stats["improvements_propagated"] += 1

    def record_improvement_applied(self, improvement_id: str, success: bool,
                                  new_metrics: Optional[PerformanceMetrics] = None):
        """Record that an improvement was applied (from another node)"""
        if success:
            self.stats["improvements_applied"] += 1
            if new_metrics:
                # Update baseline if this is better
                # (would need component_name to update correct baseline)
                logger.info(f"Improvement {improvement_id} applied successfully")
        else:
            self.stats["improvements_rejected"] += 1

    def create_announcement(self, improvement: ImprovementPackage) -> ImprovementAnnouncement:
        """Create an announcement for an improvement"""
        return ImprovementAnnouncement(
            improvement_id=improvement.improvement_id,
            improvement_type=improvement.improvement_type,
            priority=improvement.priority,
            source_node_id=improvement.source_node_id,
            improvement_percentage=improvement.improvement_percentage,
            size_bytes=improvement.get_size_bytes(),
            created_at=improvement.created_at,
            compatibility=improvement.compatibility,
            requires_version=improvement.requires_version,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        with self._history_lock:
            components_tracked = len(self._metric_history)
            total_samples = sum(len(h) for h in self._metric_history.values())

        with self._improvements_lock:
            pending_count = len(self._pending_improvements)

        return {
            **self.stats,
            "components_tracked": components_tracked,
            "total_samples": total_samples,
            "pending_improvements": pending_count,
            "min_improvement_threshold": self.min_improvement_threshold,
        }

