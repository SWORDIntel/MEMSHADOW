#!/usr/bin/env python3
"""
Meta-Intelligence Self Assessment for DSMIL Brain

Know what we know and what we don't:
- Well understood domains
- Poorly understood domains
- Blind spots
- Systematic biases
- Source reliability
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BlindSpot:
    """An identified blind spot in knowledge"""
    spot_id: str
    domain: str
    description: str

    # Assessment
    impact: float = 0.5  # How much this affects our intelligence
    detectability: float = 0.5  # How hard to detect we have this gap

    # Tracking
    discovered: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_addressed: bool = False


@dataclass
class BiasIndicator:
    """A detected systematic bias"""
    bias_id: str
    bias_type: str  # confirmation, selection, recency, etc.
    description: str

    # Evidence
    affected_domains: Set[str] = field(default_factory=set)
    evidence: List[str] = field(default_factory=list)

    # Assessment
    severity: float = 0.5  # How much this distorts our intelligence
    confidence: float = 0.5  # How sure we are this bias exists


@dataclass
class SourceTrust:
    """Trust assessment for an intelligence source"""
    source_id: str
    reliability: float = 0.5  # Historical accuracy
    access_quality: float = 0.5  # Quality of access to information
    bias_risk: float = 0.0  # Risk of biased reporting
    timeliness: float = 0.5  # How current their info is


@dataclass
class KnowledgeAssessment:
    """Overall assessment of knowledge quality"""
    assessment_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Domain assessments
    well_understood: Dict[str, float] = field(default_factory=dict)
    poorly_understood: Dict[str, float] = field(default_factory=dict)

    # Issues
    blind_spots: List[BlindSpot] = field(default_factory=list)
    biases: List[BiasIndicator] = field(default_factory=list)

    # Sources
    source_reliability: Dict[str, SourceTrust] = field(default_factory=dict)

    # Summary
    overall_confidence: float = 0.5


class MetaIntelligence:
    """
    Meta-Intelligence System

    Assesses the quality and completeness of our intelligence.

    Usage:
        meta = MetaIntelligence()

        # Register knowledge domains
        meta.register_domain("cyber_threats", confidence=0.8)
        meta.register_domain("geopolitics", confidence=0.4)

        # Register sources
        meta.register_source("sigint", reliability=0.9)

        # Assess
        assessment = meta.assess_knowledge_quality()
    """

    def __init__(self):
        self._domains: Dict[str, float] = {}  # domain -> confidence
        self._domain_counts: Dict[str, int] = defaultdict(int)  # domain -> fact count
        self._sources: Dict[str, SourceTrust] = {}
        self._blind_spots: Dict[str, BlindSpot] = {}
        self._biases: Dict[str, BiasIndicator] = {}

        self._assessments: Dict[str, KnowledgeAssessment] = {}

        self._lock = threading.RLock()

        logger.info("MetaIntelligence initialized")

    def register_domain(self, domain: str, confidence: float = 0.5,
                       fact_count: int = 0):
        """Register a knowledge domain"""
        with self._lock:
            self._domains[domain] = confidence
            self._domain_counts[domain] = fact_count

    def register_source(self, source_id: str,
                       reliability: float = 0.5,
                       access_quality: float = 0.5,
                       bias_risk: float = 0.0,
                       timeliness: float = 0.5):
        """Register an intelligence source"""
        with self._lock:
            self._sources[source_id] = SourceTrust(
                source_id=source_id,
                reliability=reliability,
                access_quality=access_quality,
                bias_risk=bias_risk,
                timeliness=timeliness,
            )

    def identify_blind_spot(self, domain: str, description: str,
                           impact: float = 0.5) -> BlindSpot:
        """Identify a blind spot"""
        with self._lock:
            spot = BlindSpot(
                spot_id=hashlib.sha256(f"blind:{domain}:{description}".encode()).hexdigest()[:16],
                domain=domain,
                description=description,
                impact=impact,
            )
            self._blind_spots[spot.spot_id] = spot
            return spot

    def identify_bias(self, bias_type: str, description: str,
                     affected_domains: Optional[Set[str]] = None,
                     severity: float = 0.5) -> BiasIndicator:
        """Identify a systematic bias"""
        with self._lock:
            bias = BiasIndicator(
                bias_id=hashlib.sha256(f"bias:{bias_type}:{description}".encode()).hexdigest()[:16],
                bias_type=bias_type,
                description=description,
                affected_domains=affected_domains or set(),
                severity=severity,
            )
            self._biases[bias.bias_id] = bias
            return bias

    def assess_knowledge_quality(self) -> KnowledgeAssessment:
        """
        Perform comprehensive knowledge quality assessment
        """
        with self._lock:
            # Classify domains by understanding
            well_understood = {}
            poorly_understood = {}

            for domain, confidence in self._domains.items():
                if confidence >= 0.7:
                    well_understood[domain] = confidence
                elif confidence < 0.4:
                    poorly_understood[domain] = confidence

            # Calculate overall confidence
            if self._domains:
                overall = sum(self._domains.values()) / len(self._domains)
            else:
                overall = 0.5

            # Adjust for blind spots and biases
            overall -= len(self._blind_spots) * 0.05
            overall -= sum(b.severity * 0.1 for b in self._biases.values())
            overall = max(0.0, min(1.0, overall))

            assessment = KnowledgeAssessment(
                assessment_id=hashlib.sha256(f"assess:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                well_understood=well_understood,
                poorly_understood=poorly_understood,
                blind_spots=list(self._blind_spots.values()),
                biases=list(self._biases.values()),
                source_reliability=dict(self._sources),
                overall_confidence=overall,
            )

            self._assessments[assessment.assessment_id] = assessment
            return assessment

    def get_high_confidence_domains(self, threshold: float = 0.7) -> List[str]:
        """Get domains where we have high confidence"""
        with self._lock:
            return [d for d, c in self._domains.items() if c >= threshold]

    def get_low_confidence_domains(self, threshold: float = 0.4) -> List[str]:
        """Get domains where we have low confidence"""
        with self._lock:
            return [d for d, c in self._domains.items() if c < threshold]

    def get_unreliable_sources(self, threshold: float = 0.4) -> List[str]:
        """Get sources below reliability threshold"""
        with self._lock:
            return [s for s, t in self._sources.items() if t.reliability < threshold]

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            return {
                "domains_tracked": len(self._domains),
                "sources_tracked": len(self._sources),
                "blind_spots": len(self._blind_spots),
                "biases_identified": len(self._biases),
                "assessments_performed": len(self._assessments),
            }


if __name__ == "__main__":
    print("Meta-Intelligence Self-Assessment Self-Test")
    print("=" * 50)

    meta = MetaIntelligence()

    print("\n[1] Register Domains")
    meta.register_domain("cyber_threats", confidence=0.85, fact_count=1500)
    meta.register_domain("nation_state_actors", confidence=0.7, fact_count=800)
    meta.register_domain("insider_threats", confidence=0.35, fact_count=150)
    meta.register_domain("supply_chain", confidence=0.25, fact_count=50)
    print("    Registered 4 domains")

    print("\n[2] Register Sources")
    meta.register_source("sigint", reliability=0.9, access_quality=0.85)
    meta.register_source("osint", reliability=0.6, access_quality=0.95)
    meta.register_source("humint", reliability=0.7, access_quality=0.4, bias_risk=0.3)
    print("    Registered 3 sources")

    print("\n[3] Identify Blind Spots")
    spot = meta.identify_blind_spot(
        domain="supply_chain",
        description="Limited visibility into 3rd tier suppliers",
        impact=0.7,
    )
    print(f"    Blind spot: {spot.description}")

    print("\n[4] Identify Biases")
    bias = meta.identify_bias(
        bias_type="recency",
        description="Over-weighting recent threats vs historical patterns",
        affected_domains={"cyber_threats", "nation_state_actors"},
        severity=0.4,
    )
    print(f"    Bias: {bias.description}")

    print("\n[5] Assess Knowledge Quality")
    assessment = meta.assess_knowledge_quality()
    print(f"    Assessment ID: {assessment.assessment_id}")
    print(f"    Overall Confidence: {assessment.overall_confidence:.2f}")
    print(f"    Well Understood: {list(assessment.well_understood.keys())}")
    print(f"    Poorly Understood: {list(assessment.poorly_understood.keys())}")
    print(f"    Blind Spots: {len(assessment.blind_spots)}")
    print(f"    Biases: {len(assessment.biases)}")

    print("\n[6] Query Functions")
    print(f"    High confidence domains: {meta.get_high_confidence_domains()}")
    print(f"    Low confidence domains: {meta.get_low_confidence_domains()}")
    print(f"    Unreliable sources: {meta.get_unreliable_sources()}")

    print("\n[7] Statistics")
    stats = meta.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Meta-Intelligence Self-Assessment test complete")

