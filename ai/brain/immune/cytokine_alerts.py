#!/usr/bin/env python3
"""
Cytokine Alert System for DSMIL Brain

Threat alerts propagating through the network:
- Severity-based propagation speed
- Cascade activation
- Network-wide immune response coordination
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Set
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class ThreatAlert:
    """A threat alert for propagation"""
    alert_id: str
    severity: AlertSeverity
    threat_type: str
    description: str

    source_node: str = ""
    affected_nodes: Set[str] = field(default_factory=set)
    propagated_to: Set[str] = field(default_factory=set)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_by: Set[str] = field(default_factory=set)


@dataclass
class AlertCascade:
    """A cascade of related alerts"""
    cascade_id: str
    root_alert_id: str
    alerts: List[str] = field(default_factory=list)
    nodes_affected: Set[str] = field(default_factory=set)
    severity_escalation: bool = False


class CytokineSystem:
    """
    Cytokine Alert System

    Propagates threat alerts across the network.

    Usage:
        cytokine = CytokineSystem(node_id="hub-001")

        # Create alert
        alert = cytokine.create_alert(AlertSeverity.HIGH, "Malware", "APT detected")

        # Propagate
        cytokine.propagate(alert, target_nodes)
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._alerts: Dict[str, ThreatAlert] = {}
        self._cascades: Dict[str, AlertCascade] = {}
        self._lock = threading.RLock()

        self.on_alert: Optional[Callable[[ThreatAlert], None]] = None
        self.send_to_node: Optional[Callable[[str, ThreatAlert], bool]] = None

        logger.info(f"CytokineSystem initialized for {node_id}")

    def create_alert(self, severity: AlertSeverity, threat_type: str,
                    description: str) -> ThreatAlert:
        """Create a new threat alert"""
        alert_id = hashlib.sha256(f"{self.node_id}:{threat_type}:{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        alert = ThreatAlert(
            alert_id=alert_id,
            severity=severity,
            threat_type=threat_type,
            description=description,
            source_node=self.node_id,
        )

        with self._lock:
            self._alerts[alert_id] = alert

        if self.on_alert:
            self.on_alert(alert)

        return alert

    def propagate(self, alert: ThreatAlert, target_nodes: List[str]) -> int:
        """
        Propagate alert to target nodes

        Returns number of nodes notified
        """
        notified = 0

        for node_id in target_nodes:
            if node_id == self.node_id:
                continue
            if node_id in alert.propagated_to:
                continue

            if self.send_to_node:
                if self.send_to_node(node_id, alert):
                    alert.propagated_to.add(node_id)
                    notified += 1
            else:
                alert.propagated_to.add(node_id)
                notified += 1

        return notified

    def acknowledge(self, alert_id: str, node_id: str):
        """Acknowledge receipt of alert"""
        with self._lock:
            if alert_id in self._alerts:
                self._alerts[alert_id].acknowledged_by.add(node_id)

    def get_active_alerts(self, min_severity: AlertSeverity = AlertSeverity.LOW) -> List[ThreatAlert]:
        """Get active alerts at or above severity"""
        with self._lock:
            return [a for a in self._alerts.values() if a.severity.value >= min_severity.value]

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            return {
                "total_alerts": len(self._alerts),
                "active_cascades": len(self._cascades),
            }


if __name__ == "__main__":
    print("Cytokine System Self-Test")
    print("=" * 50)

    cytokine = CytokineSystem("hub-001")

    print("\n[1] Create Alert")
    alert = cytokine.create_alert(
        AlertSeverity.HIGH,
        "Malware.APT",
        "Advanced persistent threat detected in network segment"
    )
    print(f"    Alert ID: {alert.alert_id}")
    print(f"    Severity: {alert.severity.name}")

    print("\n[2] Propagate")
    nodes = ["node-001", "node-002", "node-003"]
    notified = cytokine.propagate(alert, nodes)
    print(f"    Notified: {notified} nodes")

    print("\n[3] Statistics")
    stats = cytokine.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Cytokine System test complete")

