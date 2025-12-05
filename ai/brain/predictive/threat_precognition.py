#!/usr/bin/env python3
"""
Threat Precognition Engine for DSMIL Brain

Predicts future threats before they materialize:
- Bayesian attack graph modeling
- Adversary capability trajectory analysis
- Behavioral drift detection ("pre-crime")
- Confidence-scored threat forecasts with timelines
- Probability distributions over future states
"""

import math
import random
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class ThreatCategory(Enum):
    """Categories of threats"""
    MALWARE = auto()
    APT = auto()
    INSIDER = auto()
    DDOS = auto()
    DATA_BREACH = auto()
    RANSOMWARE = auto()
    SUPPLY_CHAIN = auto()
    ZERO_DAY = auto()
    SOCIAL_ENGINEERING = auto()
    PHYSICAL = auto()


@dataclass
class ThreatIndicator:
    """An observed threat indicator"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, behavior, etc.
    value: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackNode:
    """Node in an attack graph"""
    node_id: str
    name: str
    description: str

    # Probability
    prior_probability: float = 0.0  # Base probability
    posterior_probability: float = 0.0  # Updated probability

    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)  # Required prior nodes

    # Impact
    impact_score: float = 0.0

    # Evidence
    observed_indicators: List[str] = field(default_factory=list)


@dataclass
class AttackEdge:
    """Edge in an attack graph"""
    source_id: str
    target_id: str
    probability: float  # Probability of transition
    technique: str  # MITRE ATT&CK technique
    conditions: List[str] = field(default_factory=list)


class AttackGraph:
    """
    Bayesian Attack Graph for threat modeling

    Models potential attack paths with probabilistic transitions.
    Updates probabilities based on observed indicators.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._nodes: Dict[str, AttackNode] = {}
        self._edges: List[AttackEdge] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_node(self, node: AttackNode):
        """Add node to attack graph"""
        with self._lock:
            self._nodes[node.node_id] = node

    def add_edge(self, edge: AttackEdge):
        """Add edge to attack graph"""
        with self._lock:
            self._edges.append(edge)
            self._adjacency[edge.source_id].append(edge.target_id)
            self._reverse_adjacency[edge.target_id].append(edge.source_id)

    def get_attack_paths(self, start_node: str, end_node: str,
                         max_depth: int = 10) -> List[List[str]]:
        """Find all attack paths between two nodes"""
        paths = []

        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return

            for next_node in self._adjacency.get(current, []):
                if next_node not in path:
                    path.append(next_node)
                    dfs(next_node, target, path, depth + 1)
                    path.pop()

        with self._lock:
            dfs(start_node, end_node, [start_node], 0)

        return paths

    def calculate_path_probability(self, path: List[str]) -> float:
        """Calculate probability of an attack path"""
        if len(path) < 2:
            return 0.0

        probability = 1.0

        with self._lock:
            for i in range(len(path) - 1):
                source, target = path[i], path[i + 1]

                # Find edge
                edge_prob = 0.0
                for edge in self._edges:
                    if edge.source_id == source and edge.target_id == target:
                        edge_prob = edge.probability
                        break

                probability *= edge_prob

        return probability

    def update_probabilities(self, observed_node: str, confidence: float):
        """
        Update node probabilities using Bayesian inference
        when evidence is observed
        """
        with self._lock:
            if observed_node not in self._nodes:
                return

            node = self._nodes[observed_node]

            # Update observed node
            node.posterior_probability = min(1.0, node.prior_probability + confidence * (1 - node.prior_probability))

            # Propagate forward (subsequent nodes more likely)
            visited = set()
            queue = [(observed_node, confidence)]

            while queue:
                current_id, current_conf = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                for next_id in self._adjacency.get(current_id, []):
                    if next_id in self._nodes:
                        next_node = self._nodes[next_id]
                        # Decay confidence as we propagate
                        propagated_conf = current_conf * 0.7
                        next_node.posterior_probability = min(
                            1.0,
                            next_node.posterior_probability + propagated_conf * 0.3
                        )
                        queue.append((next_id, propagated_conf))

    def get_highest_risk_paths(self, top_k: int = 5) -> List[Tuple[List[str], float, float]]:
        """
        Get highest risk attack paths

        Returns:
            List of (path, probability, impact) tuples
        """
        # Find terminal nodes (high impact targets)
        terminal_nodes = [
            nid for nid, node in self._nodes.items()
            if node.impact_score > 0.5 and not self._adjacency.get(nid)
        ]

        # Find initial nodes (entry points)
        initial_nodes = [
            nid for nid in self._nodes
            if not self._reverse_adjacency.get(nid)
        ]

        all_paths = []

        for start in initial_nodes:
            for end in terminal_nodes:
                paths = self.get_attack_paths(start, end)
                for path in paths:
                    prob = self.calculate_path_probability(path)
                    impact = self._nodes[end].impact_score if end in self._nodes else 0
                    risk = prob * impact
                    all_paths.append((path, prob, impact, risk))

        # Sort by risk descending
        all_paths.sort(key=lambda x: x[3], reverse=True)

        return [(p[0], p[1], p[2]) for p in all_paths[:top_k]]


@dataclass
class AdversaryTrajectory:
    """Tracks an adversary's capability evolution over time"""
    adversary_id: str
    name: str

    # Capabilities over time
    capability_history: List[Tuple[datetime, Set[str]]] = field(default_factory=list)

    # Targets over time
    target_history: List[Tuple[datetime, Set[str]]] = field(default_factory=list)

    # Techniques evolution
    technique_adoption: Dict[str, datetime] = field(default_factory=dict)

    # Predicted future capabilities
    predicted_capabilities: Set[str] = field(default_factory=set)
    prediction_confidence: float = 0.0

    def add_capability_observation(self, capabilities: Set[str], timestamp: Optional[datetime] = None):
        """Record observed capabilities at a point in time"""
        ts = timestamp or datetime.now(timezone.utc)
        self.capability_history.append((ts, capabilities))

        # Track new technique adoption
        for cap in capabilities:
            if cap not in self.technique_adoption:
                self.technique_adoption[cap] = ts

    def predict_future_capabilities(self, months_ahead: int = 6) -> Set[str]:
        """Predict capabilities the adversary will likely develop"""
        if len(self.capability_history) < 2:
            return set()

        # Analyze capability growth rate
        recent = self.capability_history[-1][1] if self.capability_history else set()

        # Simple prediction: adversaries tend to expand adjacent capabilities
        # In real implementation, would use ML model
        predicted = set(recent)

        # Add likely expansions based on patterns
        capability_families = {
            "phishing": {"spearphishing", "whaling", "vishing"},
            "malware": {"ransomware", "trojan", "backdoor", "rootkit"},
            "lateral_movement": {"pass_the_hash", "psexec", "wmi"},
            "exfiltration": {"dns_tunnel", "https_exfil", "steganography"},
        }

        for cap in recent:
            for family, members in capability_families.items():
                if cap in members:
                    # Predict they'll develop other capabilities in family
                    predicted |= members

        self.predicted_capabilities = predicted - recent
        self.prediction_confidence = min(0.8, len(self.capability_history) * 0.1)

        return self.predicted_capabilities


@dataclass
class ThreatForecast:
    """A threat forecast with confidence and timeline"""
    forecast_id: str
    threat_category: ThreatCategory
    severity: ThreatSeverity

    # Probability
    probability: float
    confidence: float

    # Timeline
    earliest: datetime
    most_likely: datetime
    latest: datetime

    # Details
    description: str
    attack_vectors: List[str] = field(default_factory=list)
    target_assets: List[str] = field(default_factory=list)

    # Indicators that led to forecast
    supporting_indicators: List[str] = field(default_factory=list)

    # Mitigations
    recommended_mitigations: List[str] = field(default_factory=list)

    # Tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "forecast_id": self.forecast_id,
            "threat_category": self.threat_category.name,
            "severity": self.severity.name,
            "probability": self.probability,
            "confidence": self.confidence,
            "timeline": {
                "earliest": self.earliest.isoformat(),
                "most_likely": self.most_likely.isoformat(),
                "latest": self.latest.isoformat(),
            },
            "description": self.description,
            "attack_vectors": self.attack_vectors,
            "target_assets": self.target_assets,
        }


class ThreatPrecognition:
    """
    Threat Precognition Engine

    Predicts future threats using:
    - Attack graph analysis
    - Adversary trajectory modeling
    - Indicator correlation
    - Behavioral drift detection

    Usage:
        precog = ThreatPrecognition()

        # Add indicators
        precog.add_indicator(indicator)

        # Get forecasts
        forecasts = precog.generate_forecasts()

        # Query specific threat probability
        prob = precog.probability_of_threat("ransomware", days=30)
    """

    def __init__(self):
        self._attack_graphs: Dict[str, AttackGraph] = {}
        self._adversaries: Dict[str, AdversaryTrajectory] = {}
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._forecasts: Dict[str, ThreatForecast] = {}

        # Behavioral baselines for drift detection
        self._baselines: Dict[str, Dict[str, float]] = {}

        # Correlation matrix
        self._indicator_correlations: Dict[Tuple[str, str], float] = {}

        self._lock = threading.RLock()

        # Initialize default attack graph
        self._init_default_attack_graph()

        logger.info("ThreatPrecognition engine initialized")

    def _init_default_attack_graph(self):
        """Initialize a default attack graph based on common patterns"""
        graph = AttackGraph("default")

        # Add nodes (simplified kill chain)
        nodes = [
            AttackNode("recon", "Reconnaissance", "Initial recon", prior_probability=0.3),
            AttackNode("weaponize", "Weaponization", "Create payload", prior_probability=0.1),
            AttackNode("deliver", "Delivery", "Deliver payload", prior_probability=0.1),
            AttackNode("exploit", "Exploitation", "Exploit vulnerability", prior_probability=0.05),
            AttackNode("install", "Installation", "Install malware", prior_probability=0.03),
            AttackNode("c2", "Command & Control", "Establish C2", prior_probability=0.02),
            AttackNode("action", "Actions on Objectives", "Achieve goal", prior_probability=0.01, impact_score=1.0),
        ]

        for node in nodes:
            graph.add_node(node)

        # Add edges
        edges = [
            AttackEdge("recon", "weaponize", 0.6, "T1595"),
            AttackEdge("weaponize", "deliver", 0.7, "T1566"),
            AttackEdge("deliver", "exploit", 0.4, "T1203"),
            AttackEdge("exploit", "install", 0.5, "T1059"),
            AttackEdge("install", "c2", 0.6, "T1071"),
            AttackEdge("c2", "action", 0.7, "T1486"),
        ]

        for edge in edges:
            graph.add_edge(edge)

        self._attack_graphs["default"] = graph

    def add_attack_graph(self, name: str, graph: AttackGraph):
        """Add a custom attack graph"""
        with self._lock:
            self._attack_graphs[name] = graph

    def add_indicator(self, indicator: ThreatIndicator):
        """Add a threat indicator"""
        with self._lock:
            self._indicators[indicator.indicator_id] = indicator

            # Update attack graph probabilities
            self._update_from_indicator(indicator)

    def _update_from_indicator(self, indicator: ThreatIndicator):
        """Update attack graphs based on new indicator"""
        # Map indicator types to attack graph nodes
        indicator_node_map = {
            "scan": "recon",
            "phishing": "deliver",
            "exploit": "exploit",
            "malware": "install",
            "c2_beacon": "c2",
        }

        node_id = indicator_node_map.get(indicator.indicator_type)
        if node_id:
            for graph in self._attack_graphs.values():
                graph.update_probabilities(node_id, indicator.confidence)

    def add_adversary(self, adversary: AdversaryTrajectory):
        """Track an adversary"""
        with self._lock:
            self._adversaries[adversary.adversary_id] = adversary

    def detect_behavioral_drift(self, entity_id: str,
                                current_behavior: Dict[str, float]) -> Dict[str, float]:
        """
        Detect behavioral drift from baseline

        Returns dict of metric -> drift_score
        """
        with self._lock:
            if entity_id not in self._baselines:
                # First observation becomes baseline
                self._baselines[entity_id] = current_behavior.copy()
                return {}

            baseline = self._baselines[entity_id]
            drift = {}

            for metric, value in current_behavior.items():
                if metric in baseline:
                    baseline_value = baseline[metric]
                    if baseline_value > 0:
                        drift_score = abs(value - baseline_value) / baseline_value
                        if drift_score > 0.2:  # 20% threshold
                            drift[metric] = drift_score

            return drift

    def generate_forecasts(self, time_horizon_days: int = 30) -> List[ThreatForecast]:
        """
        Generate threat forecasts based on current intelligence

        Returns:
            List of ThreatForecast objects
        """
        forecasts = []
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(days=time_horizon_days)

        with self._lock:
            # Analyze attack graphs
            for graph_name, graph in self._attack_graphs.items():
                high_risk_paths = graph.get_highest_risk_paths()

                for path, probability, impact in high_risk_paths:
                    if probability > 0.1:  # Only forecast if probability > 10%
                        forecast_id = hashlib.sha256(
                            f"{graph_name}:{':'.join(path)}".encode()
                        ).hexdigest()[:16]

                        # Determine category based on final node
                        category = ThreatCategory.MALWARE  # Default
                        if "ransomware" in path[-1].lower():
                            category = ThreatCategory.RANSOMWARE
                        elif "breach" in path[-1].lower():
                            category = ThreatCategory.DATA_BREACH

                        # Determine severity
                        if impact > 0.8:
                            severity = ThreatSeverity.CRITICAL
                        elif impact > 0.6:
                            severity = ThreatSeverity.HIGH
                        elif impact > 0.4:
                            severity = ThreatSeverity.MEDIUM
                        else:
                            severity = ThreatSeverity.LOW

                        # Calculate timeline
                        days_to_likely = int((1 - probability) * time_horizon_days)

                        forecast = ThreatForecast(
                            forecast_id=forecast_id,
                            threat_category=category,
                            severity=severity,
                            probability=probability,
                            confidence=min(0.9, probability + 0.2),
                            earliest=now + timedelta(days=max(1, days_to_likely - 7)),
                            most_likely=now + timedelta(days=days_to_likely),
                            latest=horizon,
                            description=f"Potential attack via path: {' -> '.join(path)}",
                            attack_vectors=path,
                        )

                        forecasts.append(forecast)
                        self._forecasts[forecast_id] = forecast

            # Analyze adversary trajectories
            for adv_id, adversary in self._adversaries.items():
                predicted = adversary.predict_future_capabilities()

                if predicted and adversary.prediction_confidence > 0.3:
                    forecast_id = hashlib.sha256(
                        f"adv:{adv_id}:{','.join(sorted(predicted))}".encode()
                    ).hexdigest()[:16]

                    forecast = ThreatForecast(
                        forecast_id=forecast_id,
                        threat_category=ThreatCategory.APT,
                        severity=ThreatSeverity.HIGH,
                        probability=adversary.prediction_confidence,
                        confidence=adversary.prediction_confidence,
                        earliest=now + timedelta(days=30),
                        most_likely=now + timedelta(days=90),
                        latest=now + timedelta(days=180),
                        description=f"Adversary {adversary.name} likely to develop: {', '.join(predicted)}",
                    )

                    forecasts.append(forecast)

        # Sort by probability * severity
        forecasts.sort(key=lambda f: f.probability * f.severity.value, reverse=True)

        return forecasts

    def probability_of_threat(self, threat_type: str, days: int = 30) -> float:
        """
        Calculate probability of a specific threat type

        Args:
            threat_type: Type of threat
            days: Time horizon in days

        Returns:
            Probability (0-1)
        """
        # Generate forecasts and filter
        forecasts = self.generate_forecasts(days)

        relevant = [
            f for f in forecasts
            if threat_type.lower() in f.description.lower() or
               threat_type.upper() == f.threat_category.name
        ]

        if not relevant:
            return 0.0

        # Combine probabilities (assuming independence)
        combined = 1.0
        for f in relevant:
            combined *= (1 - f.probability)

        return 1 - combined

    def get_forecasts(self) -> List[ThreatForecast]:
        """Get all current forecasts"""
        return list(self._forecasts.values())

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "attack_graphs": len(self._attack_graphs),
            "adversaries_tracked": len(self._adversaries),
            "indicators": len(self._indicators),
            "active_forecasts": len(self._forecasts),
            "behavioral_baselines": len(self._baselines),
        }


if __name__ == "__main__":
    print("Threat Precognition Self-Test")
    print("=" * 50)

    precog = ThreatPrecognition()

    print("\n[1] Add Indicators")
    indicators = [
        ThreatIndicator("ind-1", "scan", "192.168.1.100", 0.7,
                       datetime.now(timezone.utc), datetime.now(timezone.utc), "firewall"),
        ThreatIndicator("ind-2", "phishing", "evil.com", 0.8,
                       datetime.now(timezone.utc), datetime.now(timezone.utc), "email"),
        ThreatIndicator("ind-3", "c2_beacon", "beacon.bad", 0.9,
                       datetime.now(timezone.utc), datetime.now(timezone.utc), "network"),
    ]
    for ind in indicators:
        precog.add_indicator(ind)
    print(f"    Added {len(indicators)} indicators")

    print("\n[2] Track Adversary")
    adversary = AdversaryTrajectory("APT-TEST", "Test APT")
    adversary.add_capability_observation({"phishing", "malware"})
    adversary.add_capability_observation({"phishing", "malware", "ransomware"})
    precog.add_adversary(adversary)
    print(f"    Tracking adversary: {adversary.name}")

    print("\n[3] Generate Forecasts")
    forecasts = precog.generate_forecasts(days=30)
    print(f"    Generated {len(forecasts)} forecasts")
    for f in forecasts[:3]:
        print(f"    - {f.threat_category.name}: P={f.probability:.2f}, Severity={f.severity.name}")

    print("\n[4] Probability Queries")
    prob_malware = precog.probability_of_threat("malware", 30)
    prob_ransomware = precog.probability_of_threat("ransomware", 30)
    print(f"    P(Malware|30d): {prob_malware:.2%}")
    print(f"    P(Ransomware|30d): {prob_ransomware:.2%}")

    print("\n[5] Behavioral Drift")
    drift = precog.detect_behavioral_drift("user-001", {
        "login_frequency": 10,
        "data_access": 100,
    })
    drift2 = precog.detect_behavioral_drift("user-001", {
        "login_frequency": 50,  # 5x increase
        "data_access": 120,     # 20% increase
    })
    print(f"    Drift detected: {drift2}")

    print("\n[6] Statistics")
    stats = precog.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Threat Precognition test complete")

