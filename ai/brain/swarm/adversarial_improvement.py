#!/usr/bin/env python3
"""
Adversarial Self-Improvement for DSMIL Brain

Nodes probe each other for weaknesses:
- Discovered vulnerabilities auto-patched
- Attack patterns added to threat library
- Darwinian security hardening
"""

import hashlib
import threading
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Callable
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ProbeType(Enum):
    """Types of adversarial probes"""
    INJECTION = auto()      # Data injection attacks
    EXTRACTION = auto()     # Information extraction
    DENIAL = auto()         # Resource exhaustion
    MANIPULATION = auto()   # Data manipulation
    BYPASS = auto()         # Authentication bypass
    TIMING = auto()         # Timing attacks


class ProbeSeverity(Enum):
    """Severity of discovered vulnerability"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProbeResult:
    """Result of an adversarial probe"""
    probe_id: str
    probe_type: ProbeType
    target_node: str
    probing_node: str

    # Result
    vulnerability_found: bool = False
    description: str = ""
    severity: ProbeSeverity = ProbeSeverity.LOW

    # Reproduction
    probe_payload: Any = None
    response: Any = None

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VulnerabilityPatch:
    """A patch for a discovered vulnerability"""
    patch_id: str
    probe_result_id: str

    # Patch details
    patch_type: str
    description: str

    # Application
    nodes_patched: Set[str] = field(default_factory=set)

    # Effectiveness
    verified: bool = False
    verification_timestamp: Optional[datetime] = None


@dataclass
class ThreatPattern:
    """An attack pattern learned from probing"""
    pattern_id: str
    probe_type: ProbeType

    # Pattern
    signature: str
    indicators: List[str] = field(default_factory=list)

    # Statistics
    times_seen: int = 0
    times_blocked: int = 0

    added: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdversarialImprover:
    """
    Adversarial Self-Improvement System

    Nodes continuously probe each other to find and fix weaknesses.

    Usage:
        improver = AdversarialImprover()

        # Register node for probing
        improver.register_node("node-1", probe_handler)

        # Run probe campaign
        results = improver.probe_all_nodes()

        # Generate patches
        patches = improver.generate_patches(results)

        # Apply patches
        improver.apply_patches(patches)
    """

    def __init__(self, probe_intensity: float = 0.5):
        self.probe_intensity = probe_intensity  # 0-1, higher = more probes

        self._nodes: Dict[str, Dict] = {}  # node_id -> {handler, defenses}
        self._probe_results: Dict[str, ProbeResult] = {}
        self._patches: Dict[str, VulnerabilityPatch] = {}
        self._threat_patterns: Dict[str, ThreatPattern] = {}

        self._lock = threading.RLock()

        logger.info("AdversarialImprover initialized")

    def register_node(self, node_id: str,
                     probe_handler: Callable[[ProbeType, Any], tuple],
                     defenses: Optional[Set[str]] = None):
        """
        Register a node for adversarial testing

        probe_handler should return (vulnerable: bool, response: Any)
        """
        with self._lock:
            self._nodes[node_id] = {
                "handler": probe_handler,
                "defenses": defenses or set(),
            }

    def probe_node(self, target_node: str, probing_node: str,
                  probe_type: ProbeType,
                  payload: Any = None) -> ProbeResult:
        """
        Probe a specific node for vulnerabilities
        """
        with self._lock:
            if target_node not in self._nodes:
                return ProbeResult(
                    probe_id=hashlib.sha256(f"probe:{target_node}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                    probe_type=probe_type,
                    target_node=target_node,
                    probing_node=probing_node,
                    description="Target node not registered",
                )

            handler = self._nodes[target_node]["handler"]
            defenses = self._nodes[target_node]["defenses"]

            # Check if defense exists
            defense_name = f"defense_{probe_type.name.lower()}"
            has_defense = defense_name in defenses

            try:
                # Execute probe
                vulnerable, response = handler(probe_type, payload)

                # Defense might have caught it
                if has_defense and vulnerable:
                    vulnerable = random.random() > 0.3  # 70% chance defense works

                severity = self._assess_severity(probe_type, vulnerable)

                result = ProbeResult(
                    probe_id=hashlib.sha256(f"probe:{target_node}:{probe_type.name}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                    probe_type=probe_type,
                    target_node=target_node,
                    probing_node=probing_node,
                    vulnerability_found=vulnerable,
                    description=f"Probed {target_node} with {probe_type.name}",
                    severity=severity if vulnerable else ProbeSeverity.LOW,
                    probe_payload=payload,
                    response=response,
                )

            except Exception as e:
                result = ProbeResult(
                    probe_id=hashlib.sha256(f"probe:{target_node}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                    probe_type=probe_type,
                    target_node=target_node,
                    probing_node=probing_node,
                    description=f"Probe failed: {str(e)}",
                )

            self._probe_results[result.probe_id] = result

            # Add to threat patterns
            if result.vulnerability_found:
                self._add_threat_pattern(probe_type, payload)

            return result

    def _assess_severity(self, probe_type: ProbeType, vulnerable: bool) -> ProbeSeverity:
        """Assess vulnerability severity"""
        if not vulnerable:
            return ProbeSeverity.LOW

        severity_map = {
            ProbeType.INJECTION: ProbeSeverity.CRITICAL,
            ProbeType.EXTRACTION: ProbeSeverity.HIGH,
            ProbeType.DENIAL: ProbeSeverity.MEDIUM,
            ProbeType.MANIPULATION: ProbeSeverity.HIGH,
            ProbeType.BYPASS: ProbeSeverity.CRITICAL,
            ProbeType.TIMING: ProbeSeverity.LOW,
        }
        return severity_map.get(probe_type, ProbeSeverity.MEDIUM)

    def _add_threat_pattern(self, probe_type: ProbeType, payload: Any):
        """Add discovered attack pattern to library"""
        pattern_key = f"{probe_type.name}:{hash(str(payload)) % 10000}"

        if pattern_key not in self._threat_patterns:
            self._threat_patterns[pattern_key] = ThreatPattern(
                pattern_id=hashlib.sha256(pattern_key.encode()).hexdigest()[:16],
                probe_type=probe_type,
                signature=str(payload)[:100] if payload else "generic",
                times_seen=1,
            )
        else:
            self._threat_patterns[pattern_key].times_seen += 1

    def probe_all_nodes(self, probe_types: Optional[List[ProbeType]] = None) -> List[ProbeResult]:
        """
        Run probing campaign across all nodes
        """
        with self._lock:
            results = []
            types_to_probe = probe_types or list(ProbeType)
            nodes = list(self._nodes.keys())

            for target in nodes:
                for prober in nodes:
                    if target == prober:
                        continue

                    # Probabilistic probing based on intensity
                    if random.random() > self.probe_intensity:
                        continue

                    probe_type = random.choice(types_to_probe)
                    result = self.probe_node(target, prober, probe_type)
                    results.append(result)

            return results

    def generate_patches(self, results: Optional[List[ProbeResult]] = None) -> List[VulnerabilityPatch]:
        """
        Generate patches for discovered vulnerabilities
        """
        with self._lock:
            vulnerabilities = results or [
                r for r in self._probe_results.values()
                if r.vulnerability_found
            ]

            patches = []
            for vuln in vulnerabilities:
                patch = VulnerabilityPatch(
                    patch_id=hashlib.sha256(f"patch:{vuln.probe_id}".encode()).hexdigest()[:16],
                    probe_result_id=vuln.probe_id,
                    patch_type=f"fix_{vuln.probe_type.name.lower()}",
                    description=f"Patch for {vuln.probe_type.name} vulnerability on {vuln.target_node}",
                )
                patches.append(patch)
                self._patches[patch.patch_id] = patch

            return patches

    def apply_patches(self, patches: List[VulnerabilityPatch]) -> Dict[str, int]:
        """
        Apply patches to nodes
        """
        with self._lock:
            results = {"applied": 0, "failed": 0}

            for patch in patches:
                vuln = self._probe_results.get(patch.probe_result_id)
                if not vuln:
                    results["failed"] += 1
                    continue

                # Apply defense to target node
                target = vuln.target_node
                if target in self._nodes:
                    defense_name = f"defense_{vuln.probe_type.name.lower()}"
                    self._nodes[target]["defenses"].add(defense_name)
                    patch.nodes_patched.add(target)
                    patch.verified = True
                    patch.verification_timestamp = datetime.now(timezone.utc)
                    results["applied"] += 1
                else:
                    results["failed"] += 1

            return results

    def get_vulnerability_summary(self) -> Dict:
        """Get summary of discovered vulnerabilities"""
        with self._lock:
            vulnerabilities = [r for r in self._probe_results.values() if r.vulnerability_found]

            by_type = {}
            by_severity = {}
            by_node = {}

            for v in vulnerabilities:
                by_type[v.probe_type.name] = by_type.get(v.probe_type.name, 0) + 1
                by_severity[v.severity.name] = by_severity.get(v.severity.name, 0) + 1
                by_node[v.target_node] = by_node.get(v.target_node, 0) + 1

            return {
                "total_vulnerabilities": len(vulnerabilities),
                "by_type": by_type,
                "by_severity": by_severity,
                "by_node": by_node,
                "patches_generated": len(self._patches),
                "patches_applied": len([p for p in self._patches.values() if p.verified]),
            }

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            return {
                "nodes_registered": len(self._nodes),
                "probes_executed": len(self._probe_results),
                "vulnerabilities_found": len([r for r in self._probe_results.values() if r.vulnerability_found]),
                "patches_created": len(self._patches),
                "threat_patterns": len(self._threat_patterns),
            }


# Example probe handler for testing
def example_probe_handler(probe_type: ProbeType, payload: Any) -> tuple:
    """Example handler that simulates vulnerability"""
    # Simulate some vulnerabilities
    vulnerable = random.random() < 0.3  # 30% chance of vulnerability
    response = {"status": "probed", "type": probe_type.name}
    return vulnerable, response


if __name__ == "__main__":
    print("Adversarial Self-Improvement Self-Test")
    print("=" * 50)

    improver = AdversarialImprover(probe_intensity=0.8)

    print("\n[1] Register Nodes")
    for i in range(4):
        improver.register_node(
            f"node-{i}",
            example_probe_handler,
            defenses={"defense_timing"} if i % 2 == 0 else set()
        )
    print("    Registered 4 nodes")

    print("\n[2] Run Probe Campaign")
    results = improver.probe_all_nodes()
    print(f"    Executed {len(results)} probes")

    vulnerabilities = [r for r in results if r.vulnerability_found]
    print(f"    Found {len(vulnerabilities)} vulnerabilities")

    for v in vulnerabilities[:3]:
        print(f"      - {v.target_node}: {v.probe_type.name} ({v.severity.name})")

    print("\n[3] Generate Patches")
    patches = improver.generate_patches(vulnerabilities)
    print(f"    Generated {len(patches)} patches")

    print("\n[4] Apply Patches")
    apply_results = improver.apply_patches(patches)
    print(f"    Applied: {apply_results['applied']}, Failed: {apply_results['failed']}")

    print("\n[5] Vulnerability Summary")
    summary = improver.get_vulnerability_summary()
    print(f"    Total: {summary['total_vulnerabilities']}")
    print(f"    By Type: {summary['by_type']}")
    print(f"    By Severity: {summary['by_severity']}")

    print("\n[6] Re-probe (should find fewer vulnerabilities)")
    results2 = improver.probe_all_nodes()
    vulnerabilities2 = [r for r in results2 if r.vulnerability_found]
    print(f"    Found {len(vulnerabilities2)} vulnerabilities (was {len(vulnerabilities)})")

    print("\n[7] Statistics")
    stats = improver.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Adversarial Self-Improvement test complete")

