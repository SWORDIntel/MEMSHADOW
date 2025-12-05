#!/usr/bin/env python3
"""
T-Cell Agents for DSMIL Brain Digital Immune System

Autonomous anomaly hunting agents:
- Roaming analysis agents
- Unknown threat detection
- Adaptive response generation
- Swarm coordination
"""

import hashlib
import secrets
import threading
import logging
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Callable
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """T-Cell agent states"""
    IDLE = auto()
    HUNTING = auto()
    ANALYZING = auto()
    ALERTING = auto()
    RESPONDING = auto()
    DORMANT = auto()


class AnomalyType(Enum):
    """Types of anomalies detected"""
    STATISTICAL = auto()
    BEHAVIORAL = auto()
    TEMPORAL = auto()
    STRUCTURAL = auto()
    SEMANTIC = auto()


@dataclass
class AnomalyHunt:
    """Result of an anomaly hunt"""
    hunt_id: str
    agent_id: str

    # Target
    target_domain: str
    target_data: Any

    # Detection
    anomaly_type: Optional[AnomalyType] = None
    anomaly_score: float = 0.0
    is_anomalous: bool = False

    # Details
    description: str = ""
    evidence: List[Dict] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0


@dataclass
class TCellAgent:
    """
    A single T-Cell agent for anomaly hunting
    """
    agent_id: str
    state: AgentState = AgentState.IDLE

    # Specialization
    specialization: str = "general"  # Domain of expertise
    detection_threshold: float = 0.7

    # Statistics
    hunts_completed: int = 0
    anomalies_found: int = 0
    false_positives: int = 0

    # Configuration
    sensitivity: float = 0.5  # 0-1, higher = more sensitive

    # State
    current_hunt: Optional[AnomalyHunt] = None
    last_hunt: Optional[datetime] = None

    def hunt(self, data: Any, domain: str = "general") -> AnomalyHunt:
        """
        Hunt for anomalies in data

        Args:
            data: Data to analyze
            domain: Domain context

        Returns:
            AnomalyHunt result
        """
        self.state = AgentState.HUNTING
        start_time = time.time()

        hunt = AnomalyHunt(
            hunt_id=hashlib.sha256(f"{self.agent_id}:{time.time()}".encode()).hexdigest()[:16],
            agent_id=self.agent_id,
            target_domain=domain,
            target_data=data,
        )

        self.current_hunt = hunt

        # Perform analysis
        self.state = AgentState.ANALYZING

        anomaly_score = self._analyze(data, domain)
        hunt.anomaly_score = anomaly_score
        hunt.is_anomalous = anomaly_score >= self.detection_threshold

        if hunt.is_anomalous:
            hunt.anomaly_type = self._classify_anomaly(data)
            hunt.description = f"Anomaly detected (score: {anomaly_score:.2f})"
            self.anomalies_found += 1

        # Complete hunt
        hunt.completed_at = datetime.now(timezone.utc)
        hunt.duration_ms = (time.time() - start_time) * 1000

        self.hunts_completed += 1
        self.last_hunt = hunt.completed_at
        self.current_hunt = None
        self.state = AgentState.IDLE

        return hunt

    def _analyze(self, data: Any, domain: str) -> float:
        """
        Analyze data for anomalies

        Returns anomaly score 0-1
        """
        score = 0.0

        if isinstance(data, str):
            # Text analysis
            # Check for unusual patterns
            if len(data) > 10000:
                score += 0.2

            # Check for suspicious content
            suspicious = ["eval(", "exec(", "system(", "base64_decode", "shell_exec"]
            for s in suspicious:
                if s in data:
                    score += 0.3

            # Check entropy
            if len(data) > 100:
                entropy = self._calculate_entropy(data)
                if entropy > 5.5:  # High entropy might indicate encryption/obfuscation
                    score += 0.2

        elif isinstance(data, dict):
            # Structure analysis
            # Check for unexpected keys
            if "_system" in data or "__" in str(data.keys()):
                score += 0.3

        elif isinstance(data, (list, tuple)):
            # Check for anomalous patterns in sequences
            if len(data) > 1000:
                score += 0.1

        # Apply sensitivity
        score *= (0.5 + self.sensitivity)

        return min(1.0, score)

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        from collections import Counter
        import math

        counts = Counter(text)
        total = len(text)

        entropy = 0.0
        for count in counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def _classify_anomaly(self, data: Any) -> AnomalyType:
        """Classify the type of anomaly"""
        if isinstance(data, str):
            if "eval" in data or "exec" in data:
                return AnomalyType.STRUCTURAL
            return AnomalyType.SEMANTIC
        elif isinstance(data, dict):
            return AnomalyType.STRUCTURAL
        elif isinstance(data, (list, tuple)):
            return AnomalyType.STATISTICAL
        return AnomalyType.BEHAVIORAL

    def adjust_sensitivity(self, feedback: bool):
        """
        Adjust sensitivity based on feedback

        Args:
            feedback: True if detection was valid, False if false positive
        """
        if feedback:
            # Increase sensitivity slightly
            self.sensitivity = min(1.0, self.sensitivity + 0.05)
        else:
            # Decrease sensitivity
            self.sensitivity = max(0.1, self.sensitivity - 0.1)
            self.false_positives += 1


class TCellSwarm:
    """
    Swarm of T-Cell agents for distributed anomaly hunting

    Usage:
        swarm = TCellSwarm(num_agents=10)

        # Hunt for anomalies
        hunts = swarm.hunt_all(data_list)

        # Get anomalies
        anomalies = swarm.get_anomalies()
    """

    def __init__(self, num_agents: int = 5,
                 specializations: Optional[List[str]] = None):
        """
        Initialize T-Cell swarm

        Args:
            num_agents: Number of agents
            specializations: List of specializations
        """
        self._agents: Dict[str, TCellAgent] = {}
        self._hunts: List[AnomalyHunt] = []
        self._lock = threading.RLock()

        # Create agents
        specs = specializations or ["general", "network", "code", "behavior", "data"]

        for i in range(num_agents):
            agent_id = f"tcell-{secrets.token_hex(4)}"
            spec = specs[i % len(specs)]

            agent = TCellAgent(
                agent_id=agent_id,
                specialization=spec,
                sensitivity=0.3 + random.random() * 0.4,  # Random sensitivity
            )
            self._agents[agent_id] = agent

        # Callbacks
        self.on_anomaly: Optional[Callable[[AnomalyHunt], None]] = None

        logger.info(f"TCellSwarm initialized with {num_agents} agents")

    def hunt(self, data: Any, domain: str = "general") -> List[AnomalyHunt]:
        """
        Hunt for anomalies with all agents

        Args:
            data: Data to analyze
            domain: Domain context

        Returns:
            List of hunt results
        """
        results = []

        with self._lock:
            for agent in self._agents.values():
                if agent.state == AgentState.IDLE:
                    hunt = agent.hunt(data, domain)
                    results.append(hunt)
                    self._hunts.append(hunt)

                    if hunt.is_anomalous and self.on_anomaly:
                        self.on_anomaly(hunt)

        return results

    def hunt_parallel(self, data_items: List[Tuple[Any, str]]) -> List[AnomalyHunt]:
        """
        Hunt multiple items in parallel

        Args:
            data_items: List of (data, domain) tuples

        Returns:
            List of hunt results
        """
        import concurrent.futures

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._agents)) as executor:
            futures = []

            agents = list(self._agents.values())
            for i, (data, domain) in enumerate(data_items):
                agent = agents[i % len(agents)]
                future = executor.submit(agent.hunt, data, domain)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    hunt = future.result()
                    results.append(hunt)

                    with self._lock:
                        self._hunts.append(hunt)

                        if hunt.is_anomalous and self.on_anomaly:
                            self.on_anomaly(hunt)
                except Exception as e:
                    logger.error(f"Hunt failed: {e}")

        return results

    def get_anomalies(self, min_score: float = 0.7) -> List[AnomalyHunt]:
        """Get all detected anomalies"""
        with self._lock:
            return [h for h in self._hunts if h.is_anomalous and h.anomaly_score >= min_score]

    def provide_feedback(self, hunt_id: str, is_valid: bool):
        """Provide feedback on a detection"""
        with self._lock:
            for hunt in self._hunts:
                if hunt.hunt_id == hunt_id:
                    agent = self._agents.get(hunt.agent_id)
                    if agent:
                        agent.adjust_sensitivity(is_valid)
                    break

    def get_agent(self, agent_id: str) -> Optional[TCellAgent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)

    def get_stats(self) -> Dict:
        """Get swarm statistics"""
        with self._lock:
            total_hunts = sum(a.hunts_completed for a in self._agents.values())
            total_anomalies = sum(a.anomalies_found for a in self._agents.values())
            total_fp = sum(a.false_positives for a in self._agents.values())

            return {
                "agents": len(self._agents),
                "total_hunts": total_hunts,
                "total_anomalies": total_anomalies,
                "false_positives": total_fp,
                "precision": (total_anomalies - total_fp) / max(1, total_anomalies),
            }


if __name__ == "__main__":
    print("T-Cell Agents Self-Test")
    print("=" * 50)

    swarm = TCellSwarm(num_agents=5)

    # Callback
    def on_anomaly(hunt):
        print(f"    [ANOMALY] {hunt.anomaly_type.name}: {hunt.anomaly_score:.2f}")

    swarm.on_anomaly = on_anomaly

    print("\n[1] Hunt Normal Data")
    normal_data = "This is perfectly normal text content."
    hunts = swarm.hunt(normal_data, "text")
    anomalous = [h for h in hunts if h.is_anomalous]
    print(f"    Hunts: {len(hunts)}, Anomalies: {len(anomalous)}")

    print("\n[2] Hunt Suspicious Data")
    suspicious_data = """
    <?php
    $code = base64_decode($_POST['cmd']);
    eval($code);
    system('rm -rf /');
    shell_exec($_GET['x']);
    ?>
    """ * 50  # Make it long

    hunts = swarm.hunt(suspicious_data, "code")
    anomalous = [h for h in hunts if h.is_anomalous]
    print(f"    Hunts: {len(hunts)}, Anomalies: {len(anomalous)}")

    print("\n[3] Parallel Hunt")
    items = [
        ("Normal text", "text"),
        ("eval(malicious_code)", "code"),
        ("SELECT * FROM users WHERE 1=1", "sql"),
        ("<script>alert('xss')</script>", "html"),
    ]

    hunts = swarm.hunt_parallel(items)
    print(f"    Parallel hunts: {len(hunts)}")
    for hunt in hunts:
        status = "ANOMALY" if hunt.is_anomalous else "clean"
        print(f"      - {hunt.target_domain}: {status} ({hunt.anomaly_score:.2f})")

    print("\n[4] All Anomalies")
    all_anomalies = swarm.get_anomalies(min_score=0.5)
    print(f"    Total anomalies detected: {len(all_anomalies)}")

    print("\n[5] Statistics")
    stats = swarm.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("T-Cell Agents test complete")

