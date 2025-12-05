#!/usr/bin/env python3
"""
Multi-INT Fusion Engine for DSMIL Brain

Combines all intelligence disciplines:
- SIGINT: Signals intelligence
- OSINT: Open source
- HUMINT: Human reports
- GEOINT: Geospatial
- MASINT: Measurement/signature
- FININT: Financial
- SOCINT: Social media
- TECHINT: Technical
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)


class IntelType(Enum):
    """Intelligence disciplines"""
    SIGINT = auto()   # Signals
    OSINT = auto()    # Open Source
    HUMINT = auto()   # Human
    GEOINT = auto()   # Geospatial
    MASINT = auto()   # Measurement
    FININT = auto()   # Financial
    SOCINT = auto()   # Social Media
    TECHINT = auto()  # Technical
    CYBERINT = auto() # Cyber


@dataclass
class IntelSource:
    """An intelligence source"""
    source_id: str
    source_type: IntelType
    reliability: float = 0.5  # A-F rating as 0-1
    credibility: float = 0.5  # 1-6 rating as 0-1
    name: str = ""


@dataclass
class IntelReport:
    """A single intelligence report"""
    report_id: str
    intel_type: IntelType
    source: IntelSource

    # Content
    content: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    # Confidence
    confidence: float = 0.5

    # Entities
    entities: Set[str] = field(default_factory=set)

    # Classification
    classification: str = "UNCLASSIFIED"

    # Timestamps
    collection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    report_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FusedIntelligence:
    """Fused intelligence product"""
    fusion_id: str

    # Source reports
    source_reports: List[str] = field(default_factory=list)
    int_types_used: Set[IntelType] = field(default_factory=set)

    # Fused content
    assessment: str = ""
    key_findings: List[str] = field(default_factory=list)
    entities: Set[str] = field(default_factory=set)

    # Confidence
    overall_confidence: float = 0.0
    corroboration_score: float = 0.0

    # Contradictions
    contradictions: List[Dict] = field(default_factory=list)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MultiINTFusion:
    """
    Multi-INT Fusion Engine

    Combines intelligence from multiple disciplines.

    Usage:
        fusion = MultiINTFusion()

        # Add reports
        fusion.add_report(sigint_report)
        fusion.add_report(osint_report)

        # Fuse
        product = fusion.fuse(entity="target_entity")
    """

    def __init__(self):
        self._reports: Dict[str, IntelReport] = {}
        self._by_type: Dict[IntelType, List[str]] = defaultdict(list)
        self._by_entity: Dict[str, List[str]] = defaultdict(list)
        self._fused: Dict[str, FusedIntelligence] = {}
        self._lock = threading.RLock()

        logger.info("MultiINTFusion initialized")

    def add_report(self, report: IntelReport):
        """Add intelligence report"""
        with self._lock:
            self._reports[report.report_id] = report
            self._by_type[report.intel_type].append(report.report_id)

            for entity in report.entities:
                self._by_entity[entity].append(report.report_id)

    def fuse(self, entity: Optional[str] = None,
            int_types: Optional[Set[IntelType]] = None) -> FusedIntelligence:
        """
        Fuse intelligence

        Args:
            entity: Focus on specific entity
            int_types: Filter by INT types

        Returns:
            FusedIntelligence product
        """
        with self._lock:
            # Get relevant reports
            if entity:
                report_ids = set(self._by_entity.get(entity, []))
            else:
                report_ids = set(self._reports.keys())

            if int_types:
                type_ids = set()
                for t in int_types:
                    type_ids.update(self._by_type.get(t, []))
                report_ids &= type_ids

            reports = [self._reports[rid] for rid in report_ids if rid in self._reports]

            if not reports:
                return FusedIntelligence(
                    fusion_id=hashlib.sha256(f"empty:{datetime.now().isoformat()}".encode()).hexdigest()[:16]
                )

            # Calculate corroboration
            entity_mentions = defaultdict(list)
            for r in reports:
                for e in r.entities:
                    entity_mentions[e].append(r)

            # Fuse findings
            all_entities = set()
            findings = []
            contradictions = []
            total_confidence = 0.0

            for r in reports:
                all_entities.update(r.entities)
                if r.summary:
                    findings.append(f"[{r.intel_type.name}] {r.summary}")
                total_confidence += r.confidence * r.source.reliability

            avg_confidence = total_confidence / len(reports) if reports else 0

            # Corroboration: how many sources mention same entities
            corroboration = 0.0
            if entity_mentions:
                multi_source = sum(1 for e, rs in entity_mentions.items() if len(rs) > 1)
                corroboration = multi_source / len(entity_mentions)

            fusion = FusedIntelligence(
                fusion_id=hashlib.sha256(f"fusion:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                source_reports=list(report_ids),
                int_types_used=set(r.intel_type for r in reports),
                assessment=f"Fused {len(reports)} reports from {len(set(r.intel_type for r in reports))} INT types",
                key_findings=findings[:10],
                entities=all_entities,
                overall_confidence=avg_confidence,
                corroboration_score=corroboration,
                contradictions=contradictions,
            )

            self._fused[fusion.fusion_id] = fusion
            return fusion

    def get_stats(self) -> Dict:
        """Get fusion statistics"""
        with self._lock:
            return {
                "total_reports": len(self._reports),
                "reports_by_type": {t.name: len(ids) for t, ids in self._by_type.items()},
                "entities_tracked": len(self._by_entity),
                "fusions_created": len(self._fused),
            }


if __name__ == "__main__":
    print("Multi-INT Fusion Self-Test")
    print("=" * 50)

    fusion = MultiINTFusion()

    print("\n[1] Add Reports")
    reports = [
        IntelReport(
            report_id="sig-001",
            intel_type=IntelType.SIGINT,
            source=IntelSource("nsa-1", IntelType.SIGINT, 0.9, 0.9),
            summary="Communications intercept indicates APT activity",
            entities={"APT29", "target_network"},
            confidence=0.8,
        ),
        IntelReport(
            report_id="os-001",
            intel_type=IntelType.OSINT,
            source=IntelSource("twitter", IntelType.OSINT, 0.5, 0.6),
            summary="Social media mentions of APT29 campaign",
            entities={"APT29", "malware_family"},
            confidence=0.5,
        ),
        IntelReport(
            report_id="tech-001",
            intel_type=IntelType.TECHINT,
            source=IntelSource("malware_lab", IntelType.TECHINT, 0.85, 0.9),
            summary="Malware analysis confirms APT29 TTPs",
            entities={"APT29", "malware_family", "c2_server"},
            confidence=0.9,
        ),
    ]

    for r in reports:
        fusion.add_report(r)
    print(f"    Added {len(reports)} reports")

    print("\n[2] Fuse Intelligence")
    product = fusion.fuse(entity="APT29")
    print(f"    Fusion ID: {product.fusion_id}")
    print(f"    INT types: {[t.name for t in product.int_types_used]}")
    print(f"    Confidence: {product.overall_confidence:.2f}")
    print(f"    Corroboration: {product.corroboration_score:.2f}")
    print(f"    Entities: {product.entities}")

    print("\n[3] Statistics")
    stats = fusion.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Multi-INT Fusion test complete")

