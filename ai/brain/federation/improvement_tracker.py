"""
Improvement Tracker

Tracks local performance metrics and detects statistically significant improvements.

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import structlog

from .improvement_types import (
    ImprovementPackage,
    ImprovementType,
    ImprovementPriority,
    ImprovementMetrics,
)

logger = structlog.get_logger()


@dataclass
class MetricWindow:
    """Sliding window for metric collection"""
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float):
        self.values.append(value)
        self.timestamps.append(time.time())
    
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0
    
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0
    
    def min(self) -> float:
        return min(self.values) if self.values else 0.0
    
    def max(self) -> float:
        return max(self.values) if self.values else 0.0


class ImprovementTracker:
    """
    Tracks performance metrics and detects significant improvements.
    
    Features:
    - Performance metric tracking (accuracy, latency, confidence)
    - Statistical significance detection
    - Improvement packaging
    - Compatibility checking
    - Effectiveness measurement
    """
    
    def __init__(
        self,
        node_id: str,
        node_version: str = "1.0.0",
        significance_threshold: float = 0.05,  # 5% improvement threshold
        min_samples: int = 10,
    ):
        self.node_id = node_id
        self.node_version = node_version
        self.significance_threshold = significance_threshold
        self.min_samples = min_samples
        
        # Metric windows: metric_name -> MetricWindow
        self._metrics: Dict[str, MetricWindow] = defaultdict(MetricWindow)
        
        # Baseline metrics (before improvements)
        self._baselines: Dict[str, float] = {}
        
        # Applied improvements
        self._applied: Dict[str, ImprovementPackage] = {}
        
        # Improvement effectiveness tracking
        self._effectiveness: Dict[str, ImprovementMetrics] = {}
        
        # Pending improvements (detected but not yet packaged)
        self._pending_improvements: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_improvement_detected: Optional[Callable] = None
        
        logger.info("ImprovementTracker initialized", node_id=node_id)
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric value"""
        self._metrics[name].add(value)
        
        # Check for significant improvement
        if len(self._metrics[name].values) >= self.min_samples:
            self._check_for_improvement(name)
    
    def set_baseline(self, name: str, value: float):
        """Set baseline value for a metric"""
        self._baselines[name] = value
        logger.debug("Baseline set", metric=name, value=value)
    
    def get_current_value(self, name: str) -> float:
        """Get current (mean) value for a metric"""
        return self._metrics[name].mean()
    
    def _check_for_improvement(self, metric_name: str):
        """Check if metric shows significant improvement over baseline"""
        baseline = self._baselines.get(metric_name)
        if baseline is None:
            # First run: set baseline
            self._baselines[metric_name] = self._metrics[metric_name].mean()
            return
        
        current = self._metrics[metric_name].mean()
        
        # Calculate improvement percentage
        if baseline > 0:
            improvement_pct = ((current - baseline) / baseline) * 100
        else:
            improvement_pct = 0
        
        # Check significance (depends on metric type)
        # For latency, lower is better, so negate
        if "latency" in metric_name.lower():
            improvement_pct = -improvement_pct
        
        if improvement_pct >= self.significance_threshold * 100:
            self._register_improvement(metric_name, improvement_pct, baseline, current)
    
    def _register_improvement(
        self,
        metric_name: str,
        improvement_pct: float,
        baseline: float,
        current: float,
    ):
        """Register a detected improvement"""
        improvement_info = {
            "metric": metric_name,
            "improvement_pct": improvement_pct,
            "baseline": baseline,
            "current": current,
            "detected_at": datetime.utcnow(),
        }
        
        self._pending_improvements.append(improvement_info)
        
        logger.info(
            "Improvement detected",
            metric=metric_name,
            improvement_pct=f"{improvement_pct:.2f}%",
            baseline=f"{baseline:.4f}",
            current=f"{current:.4f}",
        )
        
        # Notify callback if set
        if self._on_improvement_detected:
            self._on_improvement_detected(improvement_info)
    
    def package_improvement(
        self,
        improvement_type: ImprovementType,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImprovementPackage:
        """
        Package a detected improvement for propagation.
        
        Args:
            improvement_type: Type of improvement
            data: Improvement data (weights, patterns, etc.)
            metadata: Additional metadata
        
        Returns:
            ImprovementPackage ready for propagation
        """
        # Calculate overall gain from pending improvements
        total_gain = sum(
            p["improvement_pct"] for p in self._pending_improvements
        ) / max(len(self._pending_improvements), 1)
        
        # Determine priority based on gain
        if total_gain > 20:
            priority = ImprovementPriority.CRITICAL
        elif total_gain > 10:
            priority = ImprovementPriority.NORMAL
        else:
            priority = ImprovementPriority.MINOR
        
        package = ImprovementPackage(
            improvement_type=improvement_type,
            source_node=self.node_id,
            version=self.node_version,
            gain_percent=total_gain,
            priority=priority,
            data=data,
            metadata=metadata or {},
            required_version=self.node_version,
        )
        
        # Clear pending improvements
        self._pending_improvements.clear()
        
        logger.info(
            "Improvement packaged",
            id=package.improvement_id,
            type=improvement_type.value,
            gain=f"{total_gain:.2f}%",
            priority=priority.value,
        )
        
        return package
    
    def check_compatibility(self, package: ImprovementPackage) -> bool:
        """Check if an incoming improvement is compatible"""
        return package.is_compatible(self.node_version)
    
    def apply_improvement(self, package: ImprovementPackage) -> bool:
        """
        Record that an improvement was applied.
        
        Starts effectiveness tracking.
        """
        if package.improvement_id in self._applied:
            logger.warning("Improvement already applied", id=package.improvement_id)
            return False
        
        self._applied[package.improvement_id] = package
        
        # Capture current metrics as "before"
        metrics = ImprovementMetrics(
            improvement_id=package.improvement_id,
            source_node=package.source_node,
            accuracy_before=self.get_current_value("accuracy"),
            latency_before_ms=self.get_current_value("latency"),
            confidence_before=self.get_current_value("confidence"),
        )
        
        self._effectiveness[package.improvement_id] = metrics
        
        logger.info("Improvement applied", id=package.improvement_id)
        return True
    
    def measure_effectiveness(self, improvement_id: str) -> Optional[ImprovementMetrics]:
        """
        Measure effectiveness of an applied improvement.
        
        Should be called after some time has passed.
        """
        if improvement_id not in self._effectiveness:
            return None
        
        metrics = self._effectiveness[improvement_id]
        
        # Capture current metrics as "after"
        metrics.accuracy_after = self.get_current_value("accuracy")
        metrics.latency_after_ms = self.get_current_value("latency")
        metrics.confidence_after = self.get_current_value("confidence")
        
        # Calculate effectiveness
        metrics.calculate_effectiveness()
        
        logger.info(
            "Effectiveness measured",
            id=improvement_id,
            effectiveness=f"{metrics.effectiveness:.4f}",
        )
        
        return metrics
    
    def on_improvement_detected(self, callback: Callable):
        """Set callback for when improvement is detected"""
        self._on_improvement_detected = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return {
            "node_id": self.node_id,
            "metrics_tracked": len(self._metrics),
            "baselines_set": len(self._baselines),
            "improvements_applied": len(self._applied),
            "pending_improvements": len(self._pending_improvements),
            "current_metrics": {
                name: {
                    "mean": window.mean(),
                    "stdev": window.stdev(),
                    "samples": len(window.values),
                }
                for name, window in self._metrics.items()
            },
        }


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "MetricWindow",
    "ImprovementTracker",
]
