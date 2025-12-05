#!/usr/bin/env python3
"""
Pattern-of-Life Engine for DSMIL Brain

Behavioral analysis and prediction:
- Entity behavioral baseline establishment
- Deviation scoring and alerting
- Activity prediction (location/action at time T)
- Relationship inference via co-occurrence
- Habit pattern extraction
"""

import math
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Types of activities"""
    LOGIN = auto()
    LOGOUT = auto()
    FILE_ACCESS = auto()
    NETWORK = auto()
    EMAIL = auto()
    APPLICATION = auto()
    PHYSICAL_ACCESS = auto()
    COMMUNICATION = auto()
    TRAVEL = auto()
    TRANSACTION = auto()


class DeviationType(Enum):
    """Types of behavioral deviations"""
    TIMING = auto()      # Activity at unusual time
    FREQUENCY = auto()   # Unusual frequency
    LOCATION = auto()    # Unusual location
    PATTERN = auto()     # Unusual pattern
    VOLUME = auto()      # Unusual volume
    RELATIONSHIP = auto() # Unusual relationship


class AlertSeverity(Enum):
    """Severity of deviation alerts"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Activity:
    """A single activity observation"""
    activity_id: str
    entity_id: str
    activity_type: ActivityType
    timestamp: datetime

    # Details
    location: Optional[str] = None
    target: Optional[str] = None  # Target of activity (file, person, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Derived
    hour_of_day: int = field(init=False)
    day_of_week: int = field(init=False)

    def __post_init__(self):
        self.hour_of_day = self.timestamp.hour
        self.day_of_week = self.timestamp.weekday()


@dataclass
class BehavioralBaseline:
    """Baseline behavior model for an entity"""
    entity_id: str

    # Temporal patterns
    active_hours: Dict[int, float] = field(default_factory=dict)  # hour -> frequency
    active_days: Dict[int, float] = field(default_factory=dict)   # day -> frequency

    # Activity patterns
    activity_frequencies: Dict[ActivityType, float] = field(default_factory=dict)
    activity_times: Dict[ActivityType, List[int]] = field(default_factory=dict)  # type -> typical hours

    # Location patterns
    common_locations: Dict[str, float] = field(default_factory=dict)

    # Relationship patterns
    common_contacts: Dict[str, float] = field(default_factory=dict)
    common_targets: Dict[str, float] = field(default_factory=dict)

    # Volume patterns
    daily_activity_count: float = 0.0
    daily_activity_std: float = 0.0

    # Metadata
    observation_count: int = 0
    first_observed: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    confidence: float = 0.0

    def update_confidence(self):
        """Update confidence based on observation count"""
        # Confidence grows with more observations, plateaus at ~100
        self.confidence = min(0.95, 1 - math.exp(-self.observation_count / 50))


@dataclass
class DeviationAlert:
    """Alert for behavioral deviation"""
    alert_id: str
    entity_id: str
    deviation_type: DeviationType
    severity: AlertSeverity

    # Details
    description: str
    observed_value: Any
    expected_value: Any
    deviation_score: float  # Standard deviations from norm

    # Context
    activity: Optional[Activity] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Investigation
    investigated: bool = False
    resolution: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "entity_id": self.entity_id,
            "deviation_type": self.deviation_type.name,
            "severity": self.severity.name,
            "description": self.description,
            "deviation_score": self.deviation_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ActivityPrediction:
    """Prediction of future activity"""
    prediction_id: str
    entity_id: str

    # Predicted activity
    predicted_type: ActivityType
    predicted_time: datetime
    predicted_location: Optional[str] = None

    # Confidence
    probability: float = 0.0
    confidence: float = 0.0

    # Basis
    based_on_pattern: str = ""


@dataclass
class EntityProfile:
    """Complete profile for an entity"""
    entity_id: str
    entity_type: str  # "user", "device", "service", etc.
    name: str

    # Baseline
    baseline: BehavioralBaseline = field(default_factory=lambda: BehavioralBaseline(""))

    # Activities
    recent_activities: List[Activity] = field(default_factory=list)
    activity_count: int = 0

    # Relationships
    relationships: Dict[str, float] = field(default_factory=dict)  # entity_id -> strength

    # Risk
    risk_score: float = 0.0
    alert_count: int = 0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternOfLifeEngine:
    """
    Pattern-of-Life Analysis Engine

    Learns behavioral patterns and detects anomalies:
    - Establishes baselines from activity observations
    - Detects deviations in real-time
    - Predicts future activities
    - Infers relationships

    Usage:
        engine = PatternOfLifeEngine()

        # Record activities
        engine.record_activity(activity)

        # Check for deviations
        alerts = engine.check_deviations(entity_id)

        # Predict next activity
        prediction = engine.predict_next_activity(entity_id)

        # Get entity profile
        profile = engine.get_profile(entity_id)
    """

    def __init__(self, deviation_threshold: float = 2.0,
                 min_observations: int = 10):
        """
        Initialize engine

        Args:
            deviation_threshold: Standard deviations for alert
            min_observations: Minimum observations before alerting
        """
        self.deviation_threshold = deviation_threshold
        self.min_observations = min_observations

        self._profiles: Dict[str, EntityProfile] = {}
        self._activities: List[Activity] = []
        self._alerts: List[DeviationAlert] = []

        # Co-occurrence matrix for relationship inference
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        self._lock = threading.RLock()

        logger.info("PatternOfLifeEngine initialized")

    def record_activity(self, activity: Activity) -> List[DeviationAlert]:
        """
        Record an activity and check for deviations

        Args:
            activity: Activity to record

        Returns:
            List of any deviation alerts
        """
        alerts = []

        with self._lock:
            # Get or create profile
            if activity.entity_id not in self._profiles:
                self._profiles[activity.entity_id] = EntityProfile(
                    entity_id=activity.entity_id,
                    entity_type="unknown",
                    name=activity.entity_id,
                )
                self._profiles[activity.entity_id].baseline.entity_id = activity.entity_id

            profile = self._profiles[activity.entity_id]
            baseline = profile.baseline

            # Store activity
            self._activities.append(activity)
            profile.recent_activities.append(activity)
            profile.activity_count += 1
            profile.last_active = activity.timestamp

            # Keep only recent activities in profile
            if len(profile.recent_activities) > 1000:
                profile.recent_activities = profile.recent_activities[-1000:]

            # Update baseline first
            self._update_baseline(profile, activity)

            # Check for deviations (only if enough observations)
            if baseline.observation_count >= self.min_observations:
                alerts = self._check_activity_deviation(profile, activity)

            # Update co-occurrence for relationships
            self._update_cooccurrence(activity)

        return alerts

    def _update_baseline(self, profile: EntityProfile, activity: Activity):
        """Update baseline from new activity"""
        baseline = profile.baseline

        # First observation
        if baseline.first_observed is None:
            baseline.first_observed = activity.timestamp
        baseline.last_updated = activity.timestamp
        baseline.observation_count += 1

        # Update temporal patterns
        hour = activity.hour_of_day
        baseline.active_hours[hour] = baseline.active_hours.get(hour, 0) + 1

        day = activity.day_of_week
        baseline.active_days[day] = baseline.active_days.get(day, 0) + 1

        # Update activity type patterns
        atype = activity.activity_type
        baseline.activity_frequencies[atype] = baseline.activity_frequencies.get(atype, 0) + 1

        if atype not in baseline.activity_times:
            baseline.activity_times[atype] = []
        baseline.activity_times[atype].append(hour)

        # Keep only recent times
        if len(baseline.activity_times[atype]) > 100:
            baseline.activity_times[atype] = baseline.activity_times[atype][-100:]

        # Update location patterns
        if activity.location:
            baseline.common_locations[activity.location] = \
                baseline.common_locations.get(activity.location, 0) + 1

        # Update target patterns
        if activity.target:
            baseline.common_targets[activity.target] = \
                baseline.common_targets.get(activity.target, 0) + 1

        # Update daily volume
        # Simplified: would need proper day aggregation
        baseline.daily_activity_count = profile.activity_count / max(1,
            (activity.timestamp - baseline.first_observed).days + 1)

        baseline.update_confidence()

    def _check_activity_deviation(self, profile: EntityProfile,
                                  activity: Activity) -> List[DeviationAlert]:
        """Check if activity deviates from baseline"""
        alerts = []
        baseline = profile.baseline

        # Timing deviation
        hour = activity.hour_of_day
        hour_freq = baseline.active_hours.get(hour, 0) / baseline.observation_count
        if hour_freq < 0.05:  # Less than 5% of activity at this hour
            alerts.append(self._create_alert(
                profile.entity_id,
                DeviationType.TIMING,
                f"Activity at unusual hour ({hour}:00)",
                hour,
                list(baseline.active_hours.keys())[:3],
                1.0 / max(0.01, hour_freq)
            ))

        # Location deviation
        if activity.location:
            loc_freq = baseline.common_locations.get(activity.location, 0) / baseline.observation_count
            if activity.location not in baseline.common_locations or loc_freq < 0.02:
                alerts.append(self._create_alert(
                    profile.entity_id,
                    DeviationType.LOCATION,
                    f"Activity at unusual location ({activity.location})",
                    activity.location,
                    list(baseline.common_locations.keys())[:3],
                    1.0 / max(0.01, loc_freq)
                ))

        # Activity type deviation
        atype = activity.activity_type
        type_freq = baseline.activity_frequencies.get(atype, 0) / baseline.observation_count
        if type_freq < 0.01:  # Rare activity type
            alerts.append(self._create_alert(
                profile.entity_id,
                DeviationType.PATTERN,
                f"Unusual activity type ({atype.name})",
                atype.name,
                [t.name for t in list(baseline.activity_frequencies.keys())[:3]],
                1.0 / max(0.01, type_freq)
            ))

        # Set severity based on deviation score
        for alert in alerts:
            if alert.deviation_score > 10:
                alert.severity = AlertSeverity.CRITICAL
            elif alert.deviation_score > 5:
                alert.severity = AlertSeverity.HIGH
            elif alert.deviation_score > 2:
                alert.severity = AlertSeverity.MEDIUM
            else:
                alert.severity = AlertSeverity.LOW

            alert.activity = activity

        # Store alerts
        self._alerts.extend(alerts)
        profile.alert_count += len(alerts)

        # Update risk score
        profile.risk_score = min(1.0, profile.alert_count / 100)

        return alerts

    def _create_alert(self, entity_id: str, deviation_type: DeviationType,
                     description: str, observed: Any, expected: Any,
                     score: float) -> DeviationAlert:
        """Create a deviation alert"""
        alert_id = hashlib.sha256(
            f"{entity_id}:{deviation_type.name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        return DeviationAlert(
            alert_id=alert_id,
            entity_id=entity_id,
            deviation_type=deviation_type,
            severity=AlertSeverity.MEDIUM,  # Will be updated
            description=description,
            observed_value=observed,
            expected_value=expected,
            deviation_score=min(100, score),
        )

    def _update_cooccurrence(self, activity: Activity):
        """Update co-occurrence for relationship inference"""
        # Track entities that interact with same targets
        if activity.target:
            # Would track other entities accessing same target
            pass

    def predict_next_activity(self, entity_id: str,
                              time_horizon_hours: int = 24) -> List[ActivityPrediction]:
        """
        Predict entity's next activities

        Args:
            entity_id: Entity to predict for
            time_horizon_hours: How far ahead to predict

        Returns:
            List of predictions
        """
        predictions = []

        with self._lock:
            if entity_id not in self._profiles:
                return predictions

            profile = self._profiles[entity_id]
            baseline = profile.baseline

            if baseline.observation_count < self.min_observations:
                return predictions

            now = datetime.now(timezone.utc)

            # Predict based on temporal patterns
            for hour_offset in range(time_horizon_hours):
                pred_time = now + timedelta(hours=hour_offset)
                pred_hour = pred_time.hour
                pred_day = pred_time.weekday()

                # Check if typically active at this time
                hour_prob = baseline.active_hours.get(pred_hour, 0) / baseline.observation_count
                day_prob = baseline.active_days.get(pred_day, 0) / baseline.observation_count

                combined_prob = hour_prob * day_prob * 4  # Normalize

                if combined_prob > 0.1:
                    # Predict most likely activity type
                    best_type = max(
                        baseline.activity_frequencies.keys(),
                        key=lambda t: baseline.activity_frequencies[t],
                        default=ActivityType.LOGIN
                    )

                    # Predict most likely location
                    best_location = max(
                        baseline.common_locations.keys(),
                        key=lambda l: baseline.common_locations[l],
                        default=None
                    ) if baseline.common_locations else None

                    pred_id = hashlib.sha256(
                        f"{entity_id}:{pred_time.isoformat()}".encode()
                    ).hexdigest()[:16]

                    predictions.append(ActivityPrediction(
                        prediction_id=pred_id,
                        entity_id=entity_id,
                        predicted_type=best_type,
                        predicted_time=pred_time,
                        predicted_location=best_location,
                        probability=combined_prob,
                        confidence=baseline.confidence,
                        based_on_pattern=f"Historical activity at hour {pred_hour}",
                    ))

        # Sort by probability
        predictions.sort(key=lambda p: p.probability, reverse=True)

        return predictions[:10]  # Top 10 predictions

    def infer_relationships(self, entity_id: str) -> Dict[str, float]:
        """
        Infer relationships based on co-occurrence patterns

        Returns:
            Dict of related_entity_id -> relationship_strength
        """
        with self._lock:
            if entity_id not in self._profiles:
                return {}

            profile = self._profiles[entity_id]
            baseline = profile.baseline

            # Use common contacts/targets as proxy for relationships
            relationships = {}

            # From explicit contacts
            for contact, count in baseline.common_contacts.items():
                relationships[contact] = count / max(1, baseline.observation_count)

            # From shared target access
            for target, count in baseline.common_targets.items():
                # Find other entities accessing same target
                for other_id, other_profile in self._profiles.items():
                    if other_id != entity_id:
                        other_count = other_profile.baseline.common_targets.get(target, 0)
                        if other_count > 0:
                            # Jaccard-like similarity
                            similarity = min(count, other_count) / max(count, other_count)
                            relationships[other_id] = max(
                                relationships.get(other_id, 0),
                                similarity * 0.5
                            )

            return relationships

    def get_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """Get entity profile"""
        with self._lock:
            profile = self._profiles.get(entity_id)
            if profile:
                # Update relationships
                profile.relationships = self.infer_relationships(entity_id)
            return profile

    def get_alerts(self, entity_id: Optional[str] = None,
                  since: Optional[datetime] = None) -> List[DeviationAlert]:
        """Get alerts, optionally filtered"""
        with self._lock:
            alerts = self._alerts

            if entity_id:
                alerts = [a for a in alerts if a.entity_id == entity_id]

            if since:
                alerts = [a for a in alerts if a.timestamp >= since]

            return alerts

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        with self._lock:
            return {
                "entities_tracked": len(self._profiles),
                "total_activities": len(self._activities),
                "total_alerts": len(self._alerts),
                "high_risk_entities": sum(
                    1 for p in self._profiles.values()
                    if p.risk_score > 0.5
                ),
            }


if __name__ == "__main__":
    print("Pattern-of-Life Self-Test")
    print("=" * 50)

    engine = PatternOfLifeEngine(min_observations=5)

    print("\n[1] Record Activities")
    # Simulate a user's normal pattern
    base_time = datetime.now(timezone.utc)

    activities = []
    for day in range(10):
        for hour in [9, 10, 11, 14, 15, 16]:  # Normal work hours
            activities.append(Activity(
                activity_id=f"act-{day}-{hour}",
                entity_id="user-001",
                activity_type=ActivityType.LOGIN if hour == 9 else ActivityType.FILE_ACCESS,
                timestamp=base_time + timedelta(days=day, hours=hour),
                location="office",
            ))

    all_alerts = []
    for act in activities:
        alerts = engine.record_activity(act)
        all_alerts.extend(alerts)

    print(f"    Recorded {len(activities)} normal activities")
    print(f"    Alerts during normal: {len(all_alerts)}")

    print("\n[2] Introduce Anomalous Activity")
    anomalous = Activity(
        activity_id="act-anomaly-1",
        entity_id="user-001",
        activity_type=ActivityType.FILE_ACCESS,
        timestamp=base_time + timedelta(days=11, hours=3),  # 3 AM!
        location="remote",
    )
    alerts = engine.record_activity(anomalous)
    print(f"    Anomalous activity at 3 AM from remote location")
    print(f"    Alerts generated: {len(alerts)}")
    for alert in alerts:
        print(f"      - {alert.deviation_type.name}: {alert.description}")
        print(f"        Severity: {alert.severity.name}, Score: {alert.deviation_score:.1f}")

    print("\n[3] Get Profile")
    profile = engine.get_profile("user-001")
    if profile:
        print(f"    Entity: {profile.entity_id}")
        print(f"    Activity count: {profile.activity_count}")
        print(f"    Risk score: {profile.risk_score:.2f}")
        print(f"    Baseline confidence: {profile.baseline.confidence:.2f}")
        print(f"    Common locations: {list(profile.baseline.common_locations.keys())}")

    print("\n[4] Predict Next Activity")
    predictions = engine.predict_next_activity("user-001", time_horizon_hours=24)
    print(f"    Predictions: {len(predictions)}")
    for pred in predictions[:3]:
        print(f"      - {pred.predicted_type.name} at {pred.predicted_time.hour}:00")
        print(f"        Location: {pred.predicted_location}, P={pred.probability:.2f}")

    print("\n[5] Statistics")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Pattern-of-Life test complete")

