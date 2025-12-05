#!/usr/bin/env python3
"""
Automated Asset Identification for DSMIL Brain

Identifies potential intelligence sources:
- Access pattern analysis
- Vulnerability indicators
- Relationship mapping
- Approach strategy generation
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Types of vulnerability indicators"""
    FINANCIAL = auto()
    IDEOLOGICAL = auto()
    COERCION = auto()
    EGO = auto()
    DISGRUNTLED = auto()


class AccessLevel(Enum):
    """Access levels"""
    NONE = 0
    PERIPHERAL = 1
    MODERATE = 2
    SIGNIFICANT = 3
    CRITICAL = 4


@dataclass
class VulnerabilityIndicator:
    """A vulnerability indicator"""
    indicator_id: str
    vuln_type: VulnerabilityType
    description: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)


@dataclass
class PotentialAsset:
    """A potential intelligence asset"""
    asset_id: str
    entity_id: str

    # Access assessment
    access_level: AccessLevel = AccessLevel.NONE
    access_targets: Set[str] = field(default_factory=set)

    # Vulnerabilities
    vulnerabilities: List[VulnerabilityIndicator] = field(default_factory=list)

    # Assessment
    recruitment_potential: float = 0.0  # 0-1
    risk_score: float = 0.0  # 0-1
    value_score: float = 0.0  # 0-1

    # Status
    identified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_assessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ApproachStrategy:
    """Strategy for approaching a potential asset"""
    strategy_id: str
    asset_id: str

    # Approach details
    approach_type: str
    steps: List[str] = field(default_factory=list)

    # Assessment
    success_probability: float = 0.0
    risk_level: float = 0.0
    time_estimate: float = 0.0  # days

    # Resources
    required_resources: List[str] = field(default_factory=list)


class AssetIdentifier:
    """
    Automated Asset Identification System

    Analyzes entities to identify potential intelligence sources.

    Usage:
        identifier = AssetIdentifier()

        # Analyze entity
        asset = identifier.analyze_entity(entity_data)

        # Generate approach strategy
        strategy = identifier.generate_approach_strategy(asset)
    """

    def __init__(self):
        self._assets: Dict[str, PotentialAsset] = {}
        self._strategies: Dict[str, ApproachStrategy] = {}
        self._lock = threading.RLock()

        logger.info("AssetIdentifier initialized")

    def analyze_entity(self, entity_id: str,
                      entity_data: Dict[str, Any]) -> PotentialAsset:
        """
        Analyze entity for asset potential
        """
        with self._lock:
            # Assess access
            access_level = self._assess_access(entity_data)
            access_targets = set(entity_data.get("has_access_to", []))

            # Identify vulnerabilities
            vulnerabilities = self._identify_vulnerabilities(entity_data)

            # Calculate scores
            value_score = self._calculate_value(access_level, access_targets)
            risk_score = self._calculate_risk(entity_data)
            recruitment_potential = self._calculate_recruitment_potential(
                vulnerabilities, value_score, risk_score
            )

            asset = PotentialAsset(
                asset_id=hashlib.sha256(f"asset:{entity_id}".encode()).hexdigest()[:16],
                entity_id=entity_id,
                access_level=access_level,
                access_targets=access_targets,
                vulnerabilities=vulnerabilities,
                recruitment_potential=recruitment_potential,
                risk_score=risk_score,
                value_score=value_score,
            )

            self._assets[asset.asset_id] = asset
            return asset

    def _assess_access(self, entity_data: Dict) -> AccessLevel:
        """Assess entity's access level"""
        access_indicators = entity_data.get("access_indicators", {})

        if access_indicators.get("critical_systems"):
            return AccessLevel.CRITICAL
        elif access_indicators.get("significant_data"):
            return AccessLevel.SIGNIFICANT
        elif access_indicators.get("internal_access"):
            return AccessLevel.MODERATE
        elif access_indicators.get("peripheral"):
            return AccessLevel.PERIPHERAL

        return AccessLevel.NONE

    def _identify_vulnerabilities(self, entity_data: Dict) -> List[VulnerabilityIndicator]:
        """Identify vulnerability indicators"""
        vulnerabilities = []
        indicators = entity_data.get("vulnerability_indicators", {})

        if indicators.get("financial_stress"):
            vulnerabilities.append(VulnerabilityIndicator(
                indicator_id=hashlib.sha256(b"fin").hexdigest()[:16],
                vuln_type=VulnerabilityType.FINANCIAL,
                description="Financial stress indicators",
                confidence=indicators.get("financial_confidence", 0.5),
                evidence=indicators.get("financial_evidence", []),
            ))

        if indicators.get("ideological_alignment"):
            vulnerabilities.append(VulnerabilityIndicator(
                indicator_id=hashlib.sha256(b"ideo").hexdigest()[:16],
                vuln_type=VulnerabilityType.IDEOLOGICAL,
                description="Ideological alignment potential",
                confidence=indicators.get("ideological_confidence", 0.5),
            ))

        if indicators.get("disgruntled"):
            vulnerabilities.append(VulnerabilityIndicator(
                indicator_id=hashlib.sha256(b"disgr").hexdigest()[:16],
                vuln_type=VulnerabilityType.DISGRUNTLED,
                description="Shows signs of dissatisfaction",
                confidence=indicators.get("disgruntled_confidence", 0.5),
            ))

        if indicators.get("ego_driven"):
            vulnerabilities.append(VulnerabilityIndicator(
                indicator_id=hashlib.sha256(b"ego").hexdigest()[:16],
                vuln_type=VulnerabilityType.EGO,
                description="Ego/recognition motivated",
                confidence=indicators.get("ego_confidence", 0.5),
            ))

        return vulnerabilities

    def _calculate_value(self, access_level: AccessLevel,
                        access_targets: Set[str]) -> float:
        """Calculate asset value score"""
        access_score = access_level.value / AccessLevel.CRITICAL.value
        target_score = min(len(access_targets) / 10.0, 1.0)
        return (access_score * 0.7 + target_score * 0.3)

    def _calculate_risk(self, entity_data: Dict) -> float:
        """Calculate risk score"""
        risk_factors = entity_data.get("risk_factors", {})

        risk = 0.0
        if risk_factors.get("counter_intelligence_awareness"):
            risk += 0.3
        if risk_factors.get("loyalty_indicators"):
            risk += 0.2
        if risk_factors.get("surveillance_likely"):
            risk += 0.3
        if risk_factors.get("unstable"):
            risk += 0.2

        return min(risk, 1.0)

    def _calculate_recruitment_potential(self, vulnerabilities: List[VulnerabilityIndicator],
                                        value_score: float,
                                        risk_score: float) -> float:
        """Calculate recruitment potential"""
        if not vulnerabilities:
            return 0.1

        vuln_score = sum(v.confidence for v in vulnerabilities) / len(vulnerabilities)

        # High value + high vulnerability + low risk = high potential
        return (value_score * 0.4 + vuln_score * 0.4 + (1 - risk_score) * 0.2)

    def generate_approach_strategy(self, asset: PotentialAsset) -> ApproachStrategy:
        """Generate approach strategy for asset"""
        with self._lock:
            # Determine best approach based on vulnerabilities
            primary_vuln = None
            if asset.vulnerabilities:
                primary_vuln = max(asset.vulnerabilities, key=lambda v: v.confidence)

            approach_type, steps = self._design_approach(primary_vuln, asset)

            strategy = ApproachStrategy(
                strategy_id=hashlib.sha256(f"strat:{asset.asset_id}".encode()).hexdigest()[:16],
                asset_id=asset.asset_id,
                approach_type=approach_type,
                steps=steps,
                success_probability=asset.recruitment_potential * 0.8,
                risk_level=asset.risk_score,
                time_estimate=self._estimate_time(approach_type),
            )

            self._strategies[strategy.strategy_id] = strategy
            return strategy

    def _design_approach(self, primary_vuln: Optional[VulnerabilityIndicator],
                        asset: PotentialAsset) -> tuple:
        """Design approach based on vulnerability"""
        if not primary_vuln:
            return "standard", [
                "Initial contact establishment",
                "Build rapport over time",
                "Assess receptiveness",
                "Gradual escalation",
            ]

        if primary_vuln.vuln_type == VulnerabilityType.FINANCIAL:
            return "financial", [
                "Identify financial pressure points",
                "Create opportunity for financial relief",
                "Establish transactional relationship",
                "Gradual intelligence requirements",
            ]

        elif primary_vuln.vuln_type == VulnerabilityType.IDEOLOGICAL:
            return "ideological", [
                "Establish shared ideological interest",
                "Build trust through common cause",
                "Frame cooperation as serving shared goals",
                "Request specific assistance",
            ]

        elif primary_vuln.vuln_type == VulnerabilityType.DISGRUNTLED:
            return "sympathetic", [
                "Establish empathetic contact",
                "Validate grievances",
                "Offer alternative recognition",
                "Channel frustration productively",
            ]

        elif primary_vuln.vuln_type == VulnerabilityType.EGO:
            return "flattery", [
                "Establish admiration-based contact",
                "Recognize expertise and accomplishments",
                "Request guidance as expert",
                "Escalate information requests",
            ]

        return "standard", ["Standard approach"]

    def _estimate_time(self, approach_type: str) -> float:
        """Estimate approach time in days"""
        estimates = {
            "standard": 180,
            "financial": 60,
            "ideological": 120,
            "sympathetic": 90,
            "flattery": 45,
        }
        return estimates.get(approach_type, 90)

    def get_high_potential_assets(self, threshold: float = 0.6) -> List[PotentialAsset]:
        """Get assets with high recruitment potential"""
        with self._lock:
            return [a for a in self._assets.values() if a.recruitment_potential >= threshold]

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            high_potential = len([a for a in self._assets.values() if a.recruitment_potential >= 0.6])
            return {
                "total_assets": len(self._assets),
                "high_potential": high_potential,
                "strategies_generated": len(self._strategies),
            }


if __name__ == "__main__":
    print("Asset Identification Self-Test")
    print("=" * 50)

    identifier = AssetIdentifier()

    print("\n[1] Analyze Entity")
    entity_data = {
        "access_indicators": {
            "significant_data": True,
            "internal_access": True,
        },
        "has_access_to": ["database_a", "system_b", "network_c"],
        "vulnerability_indicators": {
            "financial_stress": True,
            "financial_confidence": 0.7,
            "disgruntled": True,
            "disgruntled_confidence": 0.6,
        },
        "risk_factors": {
            "surveillance_likely": False,
            "unstable": False,
        },
    }

    asset = identifier.analyze_entity("target_001", entity_data)
    print(f"    Asset ID: {asset.asset_id}")
    print(f"    Access Level: {asset.access_level.name}")
    print(f"    Value Score: {asset.value_score:.2f}")
    print(f"    Risk Score: {asset.risk_score:.2f}")
    print(f"    Recruitment Potential: {asset.recruitment_potential:.2f}")
    print(f"    Vulnerabilities: {len(asset.vulnerabilities)}")
    for v in asset.vulnerabilities:
        print(f"      - {v.vuln_type.name}: {v.confidence:.2f}")

    print("\n[2] Generate Approach Strategy")
    strategy = identifier.generate_approach_strategy(asset)
    print(f"    Strategy ID: {strategy.strategy_id}")
    print(f"    Approach Type: {strategy.approach_type}")
    print(f"    Success Probability: {strategy.success_probability:.2f}")
    print(f"    Time Estimate: {strategy.time_estimate} days")
    print(f"    Steps:")
    for i, step in enumerate(strategy.steps, 1):
        print(f"      {i}. {step}")

    print("\n[3] High Potential Assets")
    high_potential = identifier.get_high_potential_assets(threshold=0.4)
    print(f"    Found {len(high_potential)} high potential assets")

    print("\n[4] Statistics")
    stats = identifier.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Asset Identification test complete")

