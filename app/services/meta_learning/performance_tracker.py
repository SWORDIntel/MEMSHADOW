"""
Performance Tracker
Phase 8.2: Monitor and baseline performance metrics

Tracks system performance across multiple dimensions:
- Retrieval accuracy and latency
- Memory encoding efficiency
- Cache hit rates
- API call optimization
- User satisfaction proxies

Incorporates patterns from LAT5150DRVMIL autonomous self-improvement
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import structlog

logger = structlog.get_logger()


class MetricCategory(Enum):
    """Performance metric categories"""
    RETRIEVAL = "retrieval"  # Memory retrieval performance
    ENCODING = "encoding"  # Memory encoding performance
    CACHE = "cache"  # Cache efficiency
    API = "api"  # API call optimization
    LATENCY = "latency"  # Response time
    ACCURACY = "accuracy"  # Correctness metrics
    RESOURCE = "resource"  # CPU, memory, disk usage


@dataclass
class PerformanceMetric:
    """
    Single performance measurement.

    Tracks a specific metric value with context and timestamp.
    """
    metric_name: str
    category: MetricCategory
    value: float

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Baseline comparison
    baseline: Optional[float] = None

    def improvement_percentage(self) -> Optional[float]:
        """Calculate improvement vs baseline"""
        if self.baseline and self.baseline > 0:
            return ((self.value - self.baseline) / self.baseline) * 100
        return None

    def is_improvement(self) -> Optional[bool]:
        """Check if this is an improvement over baseline"""
        improvement = self.improvement_percentage()
        if improvement is None:
            return None

        # For latency/resource metrics, lower is better
        if self.category in [MetricCategory.LATENCY, MetricCategory.RESOURCE]:
            return improvement < 0  # Negative means improvement (lower value)

        # For accuracy/cache hit rate, higher is better
        return improvement > 0


@dataclass
class Baseline:
    """Performance baseline for a metric"""
    metric_name: str
    baseline_value: float

    # How baseline was established
    method: str  # "average", "median", "percentile_95", "best"

    # Data used to compute baseline
    num_samples: int
    sample_period_days: int

    # Timing
    established_at: datetime = field(default_factory=datetime.utcnow)

    # Validity
    expires_at: Optional[datetime] = None


@dataclass
class Bottleneck:
    """Detected performance bottleneck"""
    bottleneck_id: str
    component: str  # Which component is slow
    severity: str  # "low", "medium", "high", "critical"

    # Measurements
    current_value: float
    expected_value: float
    delta_percentage: float

    # Context
    description: str
    evidence: Dict[str, Any]

    # Recommendations
    suggested_fixes: List[str] = field(default_factory=list)

    detected_at: datetime = field(default_factory=datetime.utcnow)


class PerformanceTracker:
    """
    Performance Tracker for MEMSHADOW.

    Monitors performance metrics, establishes baselines, and
    detects bottlenecks.

    Features:
    - Real-time metric collection
    - Automatic baseline establishment
    - Anomaly detection
    - Bottleneck identification
    - Trend analysis

    Example:
        tracker = PerformanceTracker()

        # Record metric
        tracker.record_metric(
            metric_name="retrieval_latency_ms",
            category=MetricCategory.LATENCY,
            value=42.5,
            context={"query": "python async", "results": 10}
        )

        # Establish baseline
        tracker.establish_baseline(
            metric_name="retrieval_latency_ms",
            method="percentile_95"
        )

        # Check for bottlenecks
        bottlenecks = tracker.detect_bottlenecks()
    """

    def __init__(
        self,
        baseline_sample_size: int = 1000,
        baseline_validity_days: int = 7,
        anomaly_threshold_std: float = 3.0  # Standard deviations
    ):
        """
        Initialize performance tracker.

        Args:
            baseline_sample_size: Number of samples to establish baseline
            baseline_validity_days: How long baselines are valid
            anomaly_threshold_std: Threshold for anomaly detection (in std devs)
        """
        self.baseline_sample_size = baseline_sample_size
        self.baseline_validity_days = baseline_validity_days
        self.anomaly_threshold_std = anomaly_threshold_std

        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)  # Keep last 10k measurements
        )

        # Baselines
        self.baselines: Dict[str, Baseline] = {}

        # Detected bottlenecks
        self.bottlenecks: List[Bottleneck] = []

        # Anomalies
        self.anomalies: List[PerformanceMetric] = []

        logger.info(
            "Performance tracker initialized",
            baseline_sample_size=baseline_sample_size,
            baseline_validity_days=baseline_validity_days
        )

    def record_metric(
        self,
        metric_name: str,
        category: MetricCategory,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetric:
        """
        Record a performance metric.

        Args:
            metric_name: Name of the metric
            category: Metric category
            value: Measured value
            context: Optional context information

        Returns:
            Created PerformanceMetric
        """
        # Get baseline if exists
        baseline_value = None
        if metric_name in self.baselines:
            baseline = self.baselines[metric_name]

            # Check if baseline is still valid
            if baseline.expires_at and datetime.utcnow() > baseline.expires_at:
                logger.warning(f"Baseline for {metric_name} has expired")
            else:
                baseline_value = baseline.baseline_value

        # Create metric
        metric = PerformanceMetric(
            metric_name=metric_name,
            category=category,
            value=value,
            context=context or {},
            baseline=baseline_value
        )

        # Store metric
        self.metrics[metric_name].append(metric)

        # Check for anomalies
        if self._is_anomaly(metric):
            self.anomalies.append(metric)
            logger.warning(
                "Anomaly detected",
                metric=metric_name,
                value=value,
                baseline=baseline_value
            )

        return metric

    def establish_baseline(
        self,
        metric_name: str,
        method: str = "percentile_95",
        sample_size: Optional[int] = None
    ) -> Optional[Baseline]:
        """
        Establish performance baseline for a metric.

        Args:
            metric_name: Name of the metric
            method: How to compute baseline ("average", "median", "percentile_95", "best")
            sample_size: Number of samples to use (default: config value)

        Returns:
            Established baseline or None if insufficient data
        """
        if metric_name not in self.metrics:
            logger.warning(f"No data for metric: {metric_name}")
            return None

        metric_history = list(self.metrics[metric_name])

        sample_size = sample_size or self.baseline_sample_size

        if len(metric_history) < sample_size:
            logger.warning(
                f"Insufficient data for baseline",
                metric=metric_name,
                have=len(metric_history),
                need=sample_size
            )
            return None

        # Get recent samples
        recent_samples = metric_history[-sample_size:]
        values = [m.value for m in recent_samples]

        # Compute baseline based on method
        if method == "average":
            baseline_value = statistics.mean(values)
        elif method == "median":
            baseline_value = statistics.median(values)
        elif method == "percentile_95":
            # 95th percentile (for latency: value where 95% are faster)
            sorted_values = sorted(values)
            idx = int(len(sorted_values) * 0.95)
            baseline_value = sorted_values[idx]
        elif method == "best":
            # Best recorded value
            metric_category = recent_samples[0].category

            if metric_category in [MetricCategory.LATENCY, MetricCategory.RESOURCE]:
                baseline_value = min(values)  # Lower is better
            else:
                baseline_value = max(values)  # Higher is better
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        # Calculate sample period
        oldest = recent_samples[0].timestamp
        newest = recent_samples[-1].timestamp
        sample_period_days = (newest - oldest).days

        # Create baseline
        baseline = Baseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            method=method,
            num_samples=len(recent_samples),
            sample_period_days=sample_period_days,
            expires_at=datetime.utcnow() + timedelta(days=self.baseline_validity_days)
        )

        self.baselines[metric_name] = baseline

        logger.info(
            "Baseline established",
            metric=metric_name,
            value=baseline_value,
            method=method,
            samples=len(recent_samples)
        )

        return baseline

    def detect_bottlenecks(
        self,
        lookback_minutes: int = 60
    ) -> List[Bottleneck]:
        """
        Detect performance bottlenecks.

        Analyzes recent metrics to identify components performing
        significantly worse than baseline.

        Args:
            lookback_minutes: How far back to analyze

        Returns:
            List of detected bottlenecks
        """
        bottlenecks = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)

        for metric_name, metric_history in self.metrics.items():
            # Get baseline
            if metric_name not in self.baselines:
                continue

            baseline = self.baselines[metric_name]

            # Get recent measurements
            recent = [
                m for m in metric_history
                if m.timestamp > cutoff_time
            ]

            if not recent:
                continue

            # Calculate current average
            current_avg = statistics.mean([m.value for m in recent])

            # Check for significant deviation
            delta_pct = ((current_avg - baseline.baseline_value) / baseline.baseline_value) * 100

            # Determine if this is a bottleneck
            is_bottleneck = False
            severity = "low"

            # For latency/resource metrics, higher is worse
            if recent[0].category in [MetricCategory.LATENCY, MetricCategory.RESOURCE]:
                if delta_pct > 50:  # 50% slower
                    is_bottleneck = True
                    severity = "critical" if delta_pct > 200 else "high" if delta_pct > 100 else "medium"

            # For accuracy/cache metrics, lower is worse
            else:
                if delta_pct < -20:  # 20% drop
                    is_bottleneck = True
                    severity = "critical" if delta_pct < -50 else "high" if delta_pct < -35 else "medium"

            if is_bottleneck:
                bottleneck = Bottleneck(
                    bottleneck_id=f"{metric_name}_{datetime.utcnow().timestamp()}",
                    component=metric_name,
                    severity=severity,
                    current_value=current_avg,
                    expected_value=baseline.baseline_value,
                    delta_percentage=delta_pct,
                    description=f"{metric_name} performing {abs(delta_pct):.1f}% worse than baseline",
                    evidence={
                        "recent_samples": len(recent),
                        "baseline_method": baseline.method,
                        "lookback_minutes": lookback_minutes
                    },
                    suggested_fixes=self._suggest_fixes(metric_name, delta_pct)
                )

                bottlenecks.append(bottleneck)

        # Store detected bottlenecks
        self.bottlenecks.extend(bottlenecks)

        if bottlenecks:
            logger.warning(
                f"Detected {len(bottlenecks)} bottlenecks",
                critical=sum(1 for b in bottlenecks if b.severity == "critical"),
                high=sum(1 for b in bottlenecks if b.severity == "high")
            )

        return bottlenecks

    def get_metric_summary(
        self,
        metric_name: str,
        lookback_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.

        Args:
            metric_name: Name of the metric
            lookback_minutes: Time window

        Returns:
            Summary statistics
        """
        if metric_name not in self.metrics:
            return {"error": "Metric not found"}

        cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        recent = [
            m for m in self.metrics[metric_name]
            if m.timestamp > cutoff_time
        ]

        if not recent:
            return {"error": "No recent data"}

        values = [m.value for m in recent]

        summary = {
            "metric_name": metric_name,
            "category": recent[0].category.value,
            "samples": len(recent),
            "current": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0
        }

        # Add baseline info
        if metric_name in self.baselines:
            baseline = self.baselines[metric_name]
            summary["baseline"] = baseline.baseline_value
            summary["vs_baseline_pct"] = ((summary["current"] - baseline.baseline_value) / baseline.baseline_value) * 100

        return summary

    def _is_anomaly(self, metric: PerformanceMetric) -> bool:
        """Check if a metric is an anomaly"""
        metric_name = metric.metric_name

        if len(self.metrics[metric_name]) < 30:  # Need history
            return False

        # Get recent values
        recent_values = [m.value for m in list(self.metrics[metric_name])[-100:]]

        # Calculate mean and std dev
        mean = statistics.mean(recent_values)
        stdev = statistics.stdev(recent_values)

        if stdev == 0:
            return False

        # Check if current value is beyond threshold
        z_score = abs((metric.value - mean) / stdev)

        return z_score > self.anomaly_threshold_std

    def _suggest_fixes(self, metric_name: str, delta_pct: float) -> List[str]:
        """Suggest fixes for a bottleneck"""
        fixes = []

        if "latency" in metric_name.lower():
            fixes = [
                "Check for network issues or increased API latency",
                "Verify database query performance",
                "Review recent code changes that may impact performance",
                "Consider increasing cache size or TTL"
            ]

        elif "cache" in metric_name.lower():
            fixes = [
                "Increase cache size (current may be too small)",
                "Adjust cache eviction policy",
                "Check if cache is being invalidated too frequently",
                "Review cache key generation logic"
            ]

        elif "accuracy" in metric_name.lower():
            fixes = [
                "Retrain model with recent data",
                "Check for data drift in input distribution",
                "Review feature engineering pipeline",
                "Increase model capacity or complexity"
            ]

        return fixes
