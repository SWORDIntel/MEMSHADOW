#!/usr/bin/env python3
"""
Neural Plasticity for DSMIL Brain

Self-restructuring under stress:
- Frequently used pathways strengthen
- Unused connections prune
- New threat = new specialized circuits
- Damage recovery through redundancy
"""

import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class PathwayStrength:
    """Strength of a neural pathway"""
    pathway_id: str
    source: str
    target: str
    strength: float = 0.5
    usage_count: int = 0
    last_used: Optional[datetime] = None

    def use(self):
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
        self.strength = min(1.0, self.strength + 0.05)

    def decay(self, rate: float = 0.01):
        self.strength = max(0.0, self.strength - rate)


@dataclass
class DamageRecovery:
    """Record of damage recovery"""
    recovery_id: str
    damaged_component: str
    backup_component: str
    recovery_time_ms: float
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NeuralPlasticity:
    """
    Neural Plasticity System

    Enables self-restructuring and adaptation.

    Usage:
        plasticity = NeuralPlasticity()

        # Strengthen pathway on use
        plasticity.use_pathway("input", "analysis")

        # Prune weak pathways
        plasticity.prune_weak_pathways()

        # Recover from damage
        plasticity.recover("damaged_node", "backup_node")
    """

    def __init__(self, decay_rate: float = 0.01, prune_threshold: float = 0.1):
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold

        self._pathways: Dict[str, PathwayStrength] = {}
        self._recoveries: List[DamageRecovery] = []
        self._lock = threading.RLock()

        logger.info("NeuralPlasticity initialized")

    def create_pathway(self, source: str, target: str, initial_strength: float = 0.5) -> PathwayStrength:
        """Create a new pathway"""
        pathway_id = f"{source}->{target}"

        pathway = PathwayStrength(
            pathway_id=pathway_id,
            source=source,
            target=target,
            strength=initial_strength,
        )

        with self._lock:
            self._pathways[pathway_id] = pathway

        return pathway

    def use_pathway(self, source: str, target: str) -> PathwayStrength:
        """Record pathway usage (strengthens it)"""
        pathway_id = f"{source}->{target}"

        with self._lock:
            if pathway_id not in self._pathways:
                self.create_pathway(source, target)

            pathway = self._pathways[pathway_id]
            pathway.use()
            return pathway

    def apply_decay(self):
        """Apply decay to all pathways"""
        with self._lock:
            for pathway in self._pathways.values():
                pathway.decay(self.decay_rate)

    def prune_weak_pathways(self) -> int:
        """Remove pathways below threshold"""
        pruned = 0

        with self._lock:
            to_remove = [
                pid for pid, p in self._pathways.items()
                if p.strength < self.prune_threshold
            ]

            for pid in to_remove:
                del self._pathways[pid]
                pruned += 1

        return pruned

    def recover(self, damaged: str, backup: str) -> DamageRecovery:
        """Recover from damage using backup"""
        import time
        import hashlib

        start = time.time()

        # Reroute pathways through backup
        with self._lock:
            for pathway in list(self._pathways.values()):
                if pathway.source == damaged:
                    new_id = f"{backup}->{pathway.target}"
                    self._pathways[new_id] = PathwayStrength(
                        pathway_id=new_id,
                        source=backup,
                        target=pathway.target,
                        strength=pathway.strength * 0.8,  # Slight degradation
                    )
                elif pathway.target == damaged:
                    new_id = f"{pathway.source}->{backup}"
                    self._pathways[new_id] = PathwayStrength(
                        pathway_id=new_id,
                        source=pathway.source,
                        target=backup,
                        strength=pathway.strength * 0.8,
                    )

        recovery = DamageRecovery(
            recovery_id=hashlib.sha256(f"{damaged}:{backup}:{time.time()}".encode()).hexdigest()[:16],
            damaged_component=damaged,
            backup_component=backup,
            recovery_time_ms=(time.time() - start) * 1000,
            success=True,
        )

        self._recoveries.append(recovery)
        return recovery

    def get_stats(self) -> Dict:
        """Get plasticity statistics"""
        with self._lock:
            avg_strength = sum(p.strength for p in self._pathways.values()) / max(1, len(self._pathways))
            return {
                "total_pathways": len(self._pathways),
                "avg_strength": avg_strength,
                "recoveries": len(self._recoveries),
            }


if __name__ == "__main__":
    print("Neural Plasticity Self-Test")
    print("=" * 50)

    plasticity = NeuralPlasticity()

    print("\n[1] Create Pathways")
    plasticity.create_pathway("input", "analysis")
    plasticity.create_pathway("analysis", "output")
    plasticity.create_pathway("input", "memory")
    print(f"    Created 3 pathways")

    print("\n[2] Use Pathways")
    for _ in range(10):
        plasticity.use_pathway("input", "analysis")
    plasticity.use_pathway("analysis", "output")

    print("\n[3] Apply Decay")
    plasticity.apply_decay()

    print("\n[4] Damage Recovery")
    recovery = plasticity.recover("analysis", "backup_analysis")
    print(f"    Recovery time: {recovery.recovery_time_ms:.2f}ms")

    print("\n[5] Statistics")
    stats = plasticity.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Neural Plasticity test complete")

