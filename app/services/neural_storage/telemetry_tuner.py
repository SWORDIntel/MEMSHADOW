"""
Telemetry & Auto-Tuner - Self-Optimizing Memory Fabric

Tracks:
- Hit/miss by tier
- Average path length in activation
- Migration churn
- Per-AI memory usage & recall latency

Uses bandit/RL heuristic to adapt cache sizes, temperature thresholds, decay rates.
"""

import asyncio
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import structlog

logger = structlog.get_logger()


class MetricType(str, Enum):
    """Types of metrics tracked"""
    HIT_RATE = "hit_rate"
    MISS_RATE = "miss_rate"
    LATENCY = "latency"
    PATH_LENGTH = "path_length"
    MIGRATION_CHURN = "migration_churn"
    MEMORY_USAGE = "memory_usage"
    ACTIVATION_DEPTH = "activation_depth"
    CONNECTION_DENSITY = "connection_density"


@dataclass
class MetricSample:
    """A single metric sample"""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tier: Optional[str] = None
    agent_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TuningParameter:
    """A tunable parameter"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    step_size: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    update_count: int = 0


@dataclass
class TuningAction:
    """An action taken by the auto-tuner"""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expected_improvement: float = 0.0


class MultiArmedBandit:
    """
    Thompson Sampling bandit for parameter tuning.

    Each arm represents a parameter value bucket.
    Uses Beta distribution for exploration/exploitation balance.
    """

    def __init__(self, num_arms: int = 10):
        self.num_arms = num_arms
        self.successes = np.ones(num_arms)  # Beta prior
        self.failures = np.ones(num_arms)

    def select_arm(self) -> int:
        """Select arm using Thompson Sampling"""
        samples = np.random.beta(self.successes, self.failures)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """Update arm statistics based on reward (0-1)"""
        if reward > 0.5:
            self.successes[arm] += reward
        else:
            self.failures[arm] += (1 - reward)

    def get_best_arm(self) -> int:
        """Get arm with highest expected value"""
        expected = self.successes / (self.successes + self.failures)
        return int(np.argmax(expected))


class TelemetryCollector:
    """
    Collects and aggregates telemetry from all components.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Metric storage (circular buffers)
        self.metrics: Dict[MetricType, deque] = {
            mt: deque(maxlen=window_size) for mt in MetricType
        }

        # Per-tier metrics
        self.tier_metrics: Dict[str, Dict[MetricType, deque]] = defaultdict(
            lambda: {mt: deque(maxlen=window_size) for mt in MetricType}
        )

        # Per-agent metrics
        self.agent_metrics: Dict[str, Dict[MetricType, deque]] = defaultdict(
            lambda: {mt: deque(maxlen=window_size) for mt in MetricType}
        )

        # Aggregated stats
        self.aggregated: Dict[str, float] = {}
        self.last_aggregation = datetime.utcnow()

    def record(self, sample: MetricSample):
        """Record a metric sample"""
        self.metrics[sample.metric_type].append(sample)

        if sample.tier:
            self.tier_metrics[sample.tier][sample.metric_type].append(sample)

        if sample.agent_id:
            self.agent_metrics[sample.agent_id][sample.metric_type].append(sample)

    def record_hit(self, tier: str, latency_ms: float, agent_id: Optional[str] = None):
        """Record a cache hit"""
        self.record(MetricSample(
            metric_type=MetricType.HIT_RATE,
            value=1.0,
            tier=tier,
            agent_id=agent_id
        ))
        self.record(MetricSample(
            metric_type=MetricType.LATENCY,
            value=latency_ms,
            tier=tier,
            agent_id=agent_id
        ))

    def record_miss(self, tier: str, agent_id: Optional[str] = None):
        """Record a cache miss"""
        self.record(MetricSample(
            metric_type=MetricType.MISS_RATE,
            value=1.0,
            tier=tier,
            agent_id=agent_id
        ))

    def record_activation(self, depth: int, path_length: float, agent_id: Optional[str] = None):
        """Record activation statistics"""
        self.record(MetricSample(
            metric_type=MetricType.ACTIVATION_DEPTH,
            value=float(depth),
            agent_id=agent_id
        ))
        self.record(MetricSample(
            metric_type=MetricType.PATH_LENGTH,
            value=path_length,
            agent_id=agent_id
        ))

    def record_migration(self, source_tier: str, target_tier: str):
        """Record a tier migration"""
        self.record(MetricSample(
            metric_type=MetricType.MIGRATION_CHURN,
            value=1.0,
            tier=f"{source_tier}->{target_tier}"
        ))

    def get_hit_rate(self, tier: Optional[str] = None, window_minutes: int = 5) -> float:
        """Calculate hit rate for tier or overall"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

        if tier:
            hits = sum(1 for s in self.tier_metrics[tier][MetricType.HIT_RATE]
                      if s.timestamp > cutoff)
            misses = sum(1 for s in self.tier_metrics[tier][MetricType.MISS_RATE]
                        if s.timestamp > cutoff)
        else:
            hits = sum(1 for s in self.metrics[MetricType.HIT_RATE]
                      if s.timestamp > cutoff)
            misses = sum(1 for s in self.metrics[MetricType.MISS_RATE]
                        if s.timestamp > cutoff)

        total = hits + misses
        return hits / total if total > 0 else 0.0

    def get_avg_latency(self, tier: Optional[str] = None, window_minutes: int = 5) -> float:
        """Get average latency"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

        if tier:
            samples = [s.value for s in self.tier_metrics[tier][MetricType.LATENCY]
                      if s.timestamp > cutoff]
        else:
            samples = [s.value for s in self.metrics[MetricType.LATENCY]
                      if s.timestamp > cutoff]

        return float(np.mean(samples)) if samples else 0.0

    def get_migration_churn(self, window_minutes: int = 5) -> int:
        """Get migration count in window"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        return sum(1 for s in self.metrics[MetricType.MIGRATION_CHURN]
                  if s.timestamp > cutoff)

    def get_agent_usage(self, agent_id: str) -> Dict[str, float]:
        """Get usage stats for an agent"""
        agent_data = self.agent_metrics.get(agent_id, {})

        return {
            "hit_rate": self._calc_rate(agent_data.get(MetricType.HIT_RATE, []),
                                       agent_data.get(MetricType.MISS_RATE, [])),
            "avg_latency": self._calc_avg(agent_data.get(MetricType.LATENCY, [])),
            "avg_path_length": self._calc_avg(agent_data.get(MetricType.PATH_LENGTH, [])),
            "total_requests": len(agent_data.get(MetricType.HIT_RATE, [])) +
                            len(agent_data.get(MetricType.MISS_RATE, [])),
        }

    def _calc_rate(self, hits: deque, misses: deque) -> float:
        total = len(hits) + len(misses)
        return len(hits) / total if total > 0 else 0.0

    def _calc_avg(self, samples: deque) -> float:
        if not samples:
            return 0.0
        return float(np.mean([s.value for s in samples]))

    def aggregate(self) -> Dict[str, Any]:
        """Aggregate all metrics"""
        self.aggregated = {
            "overall_hit_rate": self.get_hit_rate(),
            "overall_avg_latency_ms": self.get_avg_latency(),
            "migration_churn_5min": self.get_migration_churn(),
            "tier_stats": {},
            "agent_stats": {},
        }

        # Per-tier
        for tier in self.tier_metrics.keys():
            self.aggregated["tier_stats"][tier] = {
                "hit_rate": self.get_hit_rate(tier),
                "avg_latency_ms": self.get_avg_latency(tier),
            }

        # Per-agent
        for agent_id in self.agent_metrics.keys():
            self.aggregated["agent_stats"][agent_id] = self.get_agent_usage(agent_id)

        self.last_aggregation = datetime.utcnow()
        return self.aggregated


class AutoTuner:
    """
    Auto-tuner using bandit/RL heuristics to optimize parameters.

    Tracks performance and adjusts:
    - Cache sizes
    - Temperature thresholds
    - Decay rates
    - Activation parameters
    """

    def __init__(self, telemetry: TelemetryCollector):
        self.telemetry = telemetry

        # Tunable parameters
        self.parameters: Dict[str, TuningParameter] = {}
        self._setup_default_parameters()

        # Bandits for each parameter
        self.bandits: Dict[str, MultiArmedBandit] = {}

        # Action history
        self.action_history: List[TuningAction] = []
        self.max_history = 1000

        # Performance baseline
        self.baseline_metrics: Dict[str, float] = {}

        # Tuning state
        self.tuning_enabled = True
        self.min_samples_for_tuning = 100
        self.tuning_interval_seconds = 300

        logger.info("AutoTuner initialized")

    def _setup_default_parameters(self):
        """Setup default tunable parameters"""
        defaults = [
            TuningParameter(
                name="promote_temperature",
                current_value=0.8,
                min_value=0.5,
                max_value=0.95,
                default_value=0.8,
                step_size=0.05
            ),
            TuningParameter(
                name="demote_temperature",
                current_value=0.2,
                min_value=0.05,
                max_value=0.4,
                default_value=0.2,
                step_size=0.05
            ),
            TuningParameter(
                name="activation_decay",
                current_value=0.7,
                min_value=0.3,
                max_value=0.9,
                default_value=0.7,
                step_size=0.05
            ),
            TuningParameter(
                name="connection_decay_rate",
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
                default_value=0.01,
                step_size=0.005
            ),
            TuningParameter(
                name="ramdisk_size_factor",
                current_value=1.0,
                min_value=0.5,
                max_value=2.0,
                default_value=1.0,
                step_size=0.1
            ),
            TuningParameter(
                name="ann_top_k_factor",
                current_value=1.0,
                min_value=0.5,
                max_value=2.0,
                default_value=1.0,
                step_size=0.1
            ),
        ]

        for param in defaults:
            self.parameters[param.name] = param
            # Create bandit with arms for value range
            num_arms = int((param.max_value - param.min_value) / param.step_size) + 1
            self.bandits[param.name] = MultiArmedBandit(num_arms)

    def get_parameter(self, name: str) -> float:
        """Get current parameter value"""
        if name in self.parameters:
            return self.parameters[name].current_value
        return 0.0

    def set_parameter(self, name: str, value: float, reason: str = "manual"):
        """Set a parameter value"""
        if name not in self.parameters:
            return

        param = self.parameters[name]
        old_value = param.current_value

        # Clamp to valid range
        value = max(param.min_value, min(param.max_value, value))
        param.current_value = value
        param.last_updated = datetime.utcnow()
        param.update_count += 1

        # Log action
        action = TuningAction(
            parameter=name,
            old_value=old_value,
            new_value=value,
            reason=reason
        )
        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

        logger.info("Parameter updated",
                   parameter=name,
                   old_value=old_value,
                   new_value=value,
                   reason=reason)

    async def tune(self) -> List[TuningAction]:
        """
        Run tuning cycle.

        Returns list of actions taken.
        """
        if not self.tuning_enabled:
            return []

        actions = []

        # Aggregate current metrics
        metrics = self.telemetry.aggregate()

        # Calculate reward based on metrics
        reward = self._calculate_reward(metrics)

        # Update bandits with reward
        for name, bandit in self.bandits.items():
            param = self.parameters[name]
            current_arm = self._value_to_arm(param)
            bandit.update(current_arm, reward)

        # Decide whether to explore or exploit
        explore = np.random.random() < 0.1  # 10% exploration

        for name, bandit in self.bandits.items():
            param = self.parameters[name]

            if explore:
                # Thompson sampling
                new_arm = bandit.select_arm()
            else:
                # Best known arm
                new_arm = bandit.get_best_arm()

            new_value = self._arm_to_value(param, new_arm)

            if abs(new_value - param.current_value) > param.step_size / 2:
                old_value = param.current_value
                self.set_parameter(name, new_value, reason="auto_tune")
                actions.append(TuningAction(
                    parameter=name,
                    old_value=old_value,
                    new_value=new_value,
                    reason="bandit_selection",
                    expected_improvement=reward
                ))

        return actions

    def _calculate_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate reward signal from metrics.

        Higher hit rate + lower latency + lower churn = higher reward
        """
        hit_rate = metrics.get("overall_hit_rate", 0)
        latency = metrics.get("overall_avg_latency_ms", 100)
        churn = metrics.get("migration_churn_5min", 0)

        # Normalize components (0-1)
        hit_reward = hit_rate
        latency_reward = 1.0 / (1.0 + latency / 100)  # 100ms baseline
        churn_penalty = 1.0 / (1.0 + churn / 10)  # 10 migrations baseline

        # Combined reward
        reward = (
            0.5 * hit_reward +
            0.3 * latency_reward +
            0.2 * churn_penalty
        )

        return min(1.0, max(0.0, reward))

    def _value_to_arm(self, param: TuningParameter) -> int:
        """Convert parameter value to bandit arm index"""
        range_size = param.max_value - param.min_value
        normalized = (param.current_value - param.min_value) / range_size
        num_arms = self.bandits[param.name].num_arms
        return min(num_arms - 1, int(normalized * num_arms))

    def _arm_to_value(self, param: TuningParameter, arm: int) -> float:
        """Convert bandit arm index to parameter value"""
        num_arms = self.bandits[param.name].num_arms
        normalized = arm / (num_arms - 1)
        value = param.min_value + normalized * (param.max_value - param.min_value)
        # Round to step size
        return round(value / param.step_size) * param.step_size

    def get_stats(self) -> Dict[str, Any]:
        """Get tuner statistics"""
        return {
            "tuning_enabled": self.tuning_enabled,
            "parameters": {
                name: {
                    "current": p.current_value,
                    "min": p.min_value,
                    "max": p.max_value,
                    "updates": p.update_count,
                }
                for name, p in self.parameters.items()
            },
            "recent_actions": [
                {
                    "parameter": a.parameter,
                    "old": a.old_value,
                    "new": a.new_value,
                    "reason": a.reason,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self.action_history[-10:]
            ],
        }

    def reset_to_defaults(self):
        """Reset all parameters to defaults"""
        for name, param in self.parameters.items():
            self.set_parameter(name, param.default_value, reason="reset")

        # Reset bandits
        for name in self.bandits:
            num_arms = self.bandits[name].num_arms
            self.bandits[name] = MultiArmedBandit(num_arms)

        logger.info("Auto-tuner reset to defaults")


class TelemetryAutoTuner:
    """
    Combined telemetry and auto-tuner system.
    """

    def __init__(self, window_size: int = 1000):
        self.telemetry = TelemetryCollector(window_size)
        self.tuner = AutoTuner(self.telemetry)

        self._running = False
        self._tune_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the tuning loop"""
        self._running = True
        self._tune_task = asyncio.create_task(self._tune_loop())
        logger.info("TelemetryAutoTuner started")

    async def stop(self):
        """Stop the tuning loop"""
        self._running = False
        if self._tune_task:
            self._tune_task.cancel()
        logger.info("TelemetryAutoTuner stopped")

    async def _tune_loop(self):
        """Background tuning loop"""
        while self._running:
            try:
                await asyncio.sleep(self.tuner.tuning_interval_seconds)
                await self.tuner.tune()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Tuning loop error", error=str(e))

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined telemetry and tuner stats"""
        return {
            "telemetry": self.telemetry.aggregate(),
            "tuner": self.tuner.get_stats(),
        }
