#!/usr/bin/env python3
"""
Self-Healing Infrastructure for DSMIL Brain

Auto-recovery and resilience:
- Node loss detection and compensation
- Knowledge redistribution
- Capability replication
- Automatic failover
"""

import hashlib
import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node health status"""
    HEALTHY = auto()
    DEGRADED = auto()
    UNREACHABLE = auto()
    FAILED = auto()
    RECOVERING = auto()


class RecoveryType(Enum):
    """Types of recovery actions"""
    REDISTRIBUTE_KNOWLEDGE = auto()
    REPLICATE_CAPABILITY = auto()
    FAILOVER = auto()
    RESTART_SERVICE = auto()
    SCALE_OUT = auto()


@dataclass
class NodeHealth:
    """Health status of a node"""
    node_id: str
    status: NodeStatus = NodeStatus.HEALTHY

    # Metrics
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    knowledge_shards: Set[str] = field(default_factory=set)

    # History
    failures: int = 0
    last_failure: Optional[datetime] = None


@dataclass
class FailoverEvent:
    """A failover event"""
    event_id: str
    failed_node: str
    backup_node: str
    capabilities_transferred: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False


@dataclass
class RecoveryAction:
    """A recovery action"""
    action_id: str
    action_type: RecoveryType
    target_node: str

    # Details
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Status
    started: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed: Optional[datetime] = None
    success: bool = False
    error: str = ""


class SelfHealingInfrastructure:
    """
    Self-Healing Infrastructure System

    Monitors node health and automatically recovers from failures.

    Usage:
        healer = SelfHealingInfrastructure()

        # Register nodes
        healer.register_node("node-1", capabilities={"vector_db", "inference"})

        # Update health
        healer.update_health("node-1", heartbeat=True, response_time=50)

        # Check and heal
        actions = healer.check_and_heal()
    """

    def __init__(self, heartbeat_timeout: float = 30.0,
                failure_threshold: int = 3):
        self.heartbeat_timeout = heartbeat_timeout
        self.failure_threshold = failure_threshold

        self._nodes: Dict[str, NodeHealth] = {}
        self._failovers: Dict[str, FailoverEvent] = {}
        self._recovery_actions: Dict[str, RecoveryAction] = {}

        # Capability to node mapping
        self._capability_nodes: Dict[str, Set[str]] = defaultdict(set)

        # Knowledge shard to node mapping
        self._shard_nodes: Dict[str, Set[str]] = defaultdict(set)

        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info("SelfHealingInfrastructure initialized")

    def register_node(self, node_id: str,
                     capabilities: Optional[Set[str]] = None,
                     knowledge_shards: Optional[Set[str]] = None):
        """Register a node"""
        with self._lock:
            health = NodeHealth(
                node_id=node_id,
                capabilities=capabilities or set(),
                knowledge_shards=knowledge_shards or set(),
            )
            self._nodes[node_id] = health

            # Update mappings
            for cap in health.capabilities:
                self._capability_nodes[cap].add(node_id)

            for shard in health.knowledge_shards:
                self._shard_nodes[shard].add(node_id)

            logger.info(f"Registered node {node_id}")

    def update_health(self, node_id: str,
                     heartbeat: bool = True,
                     response_time: Optional[float] = None,
                     error_rate: Optional[float] = None,
                     cpu_usage: Optional[float] = None,
                     memory_usage: Optional[float] = None):
        """Update node health metrics"""
        with self._lock:
            if node_id not in self._nodes:
                return

            health = self._nodes[node_id]

            if heartbeat:
                health.last_heartbeat = datetime.now(timezone.utc)

            if response_time is not None:
                health.response_time_ms = response_time

            if error_rate is not None:
                health.error_rate = error_rate

            if cpu_usage is not None:
                health.cpu_usage = cpu_usage

            if memory_usage is not None:
                health.memory_usage = memory_usage

            # Update status
            health.status = self._assess_status(health)

    def _assess_status(self, health: NodeHealth) -> NodeStatus:
        """Assess node status from metrics"""
        now = datetime.now(timezone.utc)
        heartbeat_age = (now - health.last_heartbeat).total_seconds()

        if heartbeat_age > self.heartbeat_timeout * 2:
            return NodeStatus.FAILED
        elif heartbeat_age > self.heartbeat_timeout:
            return NodeStatus.UNREACHABLE
        elif health.error_rate > 0.5 or health.cpu_usage > 0.9 or health.memory_usage > 0.9:
            return NodeStatus.DEGRADED

        return NodeStatus.HEALTHY

    def check_and_heal(self) -> List[RecoveryAction]:
        """Check all nodes and initiate healing actions"""
        actions = []

        with self._lock:
            for node_id, health in self._nodes.items():
                health.status = self._assess_status(health)

                if health.status == NodeStatus.FAILED:
                    actions.extend(self._handle_failure(node_id))

                elif health.status == NodeStatus.DEGRADED:
                    actions.extend(self._handle_degradation(node_id))

        return actions

    def _handle_failure(self, node_id: str) -> List[RecoveryAction]:
        """Handle node failure"""
        actions = []
        health = self._nodes[node_id]

        health.failures += 1
        health.last_failure = datetime.now(timezone.utc)

        logger.warning(f"Node {node_id} failed (failures: {health.failures})")

        # Redistribute knowledge shards
        for shard in health.knowledge_shards:
            backup_nodes = self._shard_nodes[shard] - {node_id}
            if not backup_nodes:
                # Need to replicate shard
                actions.append(self._create_recovery_action(
                    RecoveryType.REDISTRIBUTE_KNOWLEDGE,
                    node_id,
                    {"shard": shard, "action": "replicate"},
                ))

        # Failover capabilities
        for cap in health.capabilities:
            backup_nodes = self._capability_nodes[cap] - {node_id}
            if backup_nodes:
                backup = next(iter(backup_nodes))
                failover = FailoverEvent(
                    event_id=hashlib.sha256(f"fo:{node_id}:{cap}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                    failed_node=node_id,
                    backup_node=backup,
                    capabilities_transferred={cap},
                    success=True,
                )
                self._failovers[failover.event_id] = failover
                logger.info(f"Failover {cap}: {node_id} -> {backup}")
            else:
                # Need to scale out
                actions.append(self._create_recovery_action(
                    RecoveryType.SCALE_OUT,
                    node_id,
                    {"capability": cap},
                ))

        return actions

    def _handle_degradation(self, node_id: str) -> List[RecoveryAction]:
        """Handle node degradation"""
        actions = []
        health = self._nodes[node_id]

        logger.info(f"Node {node_id} degraded (cpu={health.cpu_usage:.1%}, mem={health.memory_usage:.1%})")

        # Try restarting services
        if health.error_rate > 0.3:
            actions.append(self._create_recovery_action(
                RecoveryType.RESTART_SERVICE,
                node_id,
                {"reason": "high_error_rate"},
            ))

        return actions

    def _create_recovery_action(self, action_type: RecoveryType,
                               target_node: str,
                               parameters: Dict) -> RecoveryAction:
        """Create a recovery action"""
        action = RecoveryAction(
            action_id=hashlib.sha256(f"ra:{action_type.name}:{target_node}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            action_type=action_type,
            target_node=target_node,
            parameters=parameters,
        )
        self._recovery_actions[action.action_id] = action
        return action

    def execute_recovery(self, action: RecoveryAction,
                        executor: Optional[Callable] = None) -> bool:
        """Execute a recovery action"""
        with self._lock:
            logger.info(f"Executing recovery: {action.action_type.name} on {action.target_node}")

            try:
                if executor:
                    success = executor(action)
                else:
                    # Simulate recovery
                    success = True

                action.completed = datetime.now(timezone.utc)
                action.success = success

                if success and action.target_node in self._nodes:
                    # Reset failure count on successful recovery
                    self._nodes[action.target_node].status = NodeStatus.RECOVERING

                return success

            except Exception as e:
                action.error = str(e)
                action.success = False
                return False

    def get_healthy_nodes(self) -> List[str]:
        """Get list of healthy nodes"""
        with self._lock:
            return [
                nid for nid, health in self._nodes.items()
                if health.status == NodeStatus.HEALTHY
            ]

    def get_nodes_for_capability(self, capability: str) -> List[str]:
        """Get nodes that have a capability"""
        with self._lock:
            nodes = self._capability_nodes.get(capability, set())
            # Filter to healthy nodes
            return [
                nid for nid in nodes
                if self._nodes.get(nid, NodeHealth(node_id=nid)).status in
                   (NodeStatus.HEALTHY, NodeStatus.DEGRADED)
            ]

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            status_counts = defaultdict(int)
            for health in self._nodes.values():
                status_counts[health.status.name] += 1

            return {
                "total_nodes": len(self._nodes),
                "status": dict(status_counts),
                "failovers": len(self._failovers),
                "recovery_actions": len(self._recovery_actions),
                "successful_recoveries": len([a for a in self._recovery_actions.values() if a.success]),
            }


if __name__ == "__main__":
    print("Self-Healing Infrastructure Self-Test")
    print("=" * 50)

    healer = SelfHealingInfrastructure(heartbeat_timeout=5.0)

    print("\n[1] Register Nodes")
    healer.register_node("node-1", capabilities={"vector_db", "inference"}, knowledge_shards={"shard-a"})
    healer.register_node("node-2", capabilities={"vector_db", "fusion"}, knowledge_shards={"shard-a", "shard-b"})
    healer.register_node("node-3", capabilities={"inference"}, knowledge_shards={"shard-b"})
    print("    Registered 3 nodes")

    print("\n[2] Update Health - Normal")
    healer.update_health("node-1", heartbeat=True, response_time=50, cpu_usage=0.3)
    healer.update_health("node-2", heartbeat=True, response_time=45, cpu_usage=0.4)
    healer.update_health("node-3", heartbeat=True, response_time=55, cpu_usage=0.35)
    print("    All nodes healthy")

    print("\n[3] Simulate Degradation")
    healer.update_health("node-2", cpu_usage=0.95, error_rate=0.4)
    actions = healer.check_and_heal()
    print(f"    Recovery actions generated: {len(actions)}")
    for action in actions:
        print(f"      - {action.action_type.name} on {action.target_node}")

    print("\n[4] Simulate Failure (no heartbeat)")
    # Simulate timeout by backdating heartbeat
    healer._nodes["node-1"].last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=60)
    actions = healer.check_and_heal()
    print(f"    Recovery actions generated: {len(actions)}")
    for action in actions:
        print(f"      - {action.action_type.name}: {action.parameters}")

    print("\n[5] Execute Recovery")
    for action in actions:
        success = healer.execute_recovery(action)
        print(f"    {action.action_type.name}: {'success' if success else 'failed'}")

    print("\n[6] Get Nodes for Capability")
    vector_nodes = healer.get_nodes_for_capability("vector_db")
    print(f"    Nodes with vector_db: {vector_nodes}")

    print("\n[7] Statistics")
    stats = healer.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Self-Healing Infrastructure test complete")

