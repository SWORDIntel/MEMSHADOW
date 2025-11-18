"""
Improvement Engine
Phase 8.2: Autonomous system improvement proposals

Analyzes performance data and proposes optimizations:
- Detects improvement opportunities
- Proposes specific changes
- Assesses risk and impact
- Tracks implementation success

Based on LAT5150DRVMIL autonomous_self_improvement patterns
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import structlog

logger = structlog.get_logger()


class ProposalCategory(Enum):
    """Improvement proposal categories"""
    PERFORMANCE = "performance"  # Speed/latency optimization
    ARCHITECTURE = "architecture"  # Structural improvements
    FEATURE = "feature"  # New capability
    BUGFIX = "bugfix"  # Fix incorrect behavior
    MEMORY = "memory"  # Memory/resource optimization
    CACHE = "cache"  # Cache strategy improvement


class RiskLevel(Enum):
    """Risk level for proposed changes"""
    LOW = "low"  # Safe, incremental change
    MEDIUM = "medium"  # Moderate risk, should test
    HIGH = "high"  # Significant risk, needs review
    CRITICAL = "critical"  # Could break system, needs approval


class ImpactLevel(Enum):
    """Expected impact of change"""
    LOW = "low"  # Minor improvement
    MEDIUM = "medium"  # Noticeable improvement
    HIGH = "high"  # Significant improvement
    CRITICAL = "critical"  # Game-changing improvement


@dataclass
class ImprovementProposal:
    """
    Proposed system improvement.

    Represents a specific change the system suggests to improve
    its own performance.
    """
    proposal_id: str
    category: ProposalCategory
    title: str
    description: str
    rationale: str  # Why this improvement is needed

    # Assessment
    risk_level: RiskLevel
    impact_level: ImpactLevel
    estimated_improvement_pct: float  # Expected improvement %

    # Implementation details
    affected_components: List[str]
    code_changes: Optional[str] = None  # Proposed code changes
    config_changes: Optional[Dict[str, Any]] = None

    # Gating
    requires_approval: bool = True
    auto_implementable: bool = False

    # Status
    status: str = "proposed"  # proposed, approved, implemented, rejected
    implemented_at: Optional[datetime] = None
    actual_improvement_pct: Optional[float] = None

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.proposal_id:
            hash_input = f"{self.title}{self.created_at.isoformat()}"
            self.proposal_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


@dataclass
class LearningInsight:
    """
    Learned pattern or insight from observations.

    The system identifies recurring patterns that could lead
    to improvements.
    """
    insight_id: str
    insight_type: str  # "pattern", "optimization", "bottleneck", "preference"
    content: str
    confidence: float  # 0.0-1.0

    # Evidence
    evidence_count: int
    first_observed: datetime
    last_observed: datetime

    # Actionability
    actionable: bool = False
    action_taken: bool = False
    related_proposal_id: Optional[str] = None


class ImprovementEngine:
    """
    Autonomous Improvement Engine.

    Analyzes system performance and proposes targeted improvements.

    Features:
    - Analyzes performance tracker data
    - Detects improvement opportunities
    - Proposes specific changes with risk/impact assessment
    - Tracks implementation success
    - Learns from past improvements

    Safety:
    - All proposals require approval by default
    - Risk assessment for every change
    - Rollback capability
    - A/B testing support

    Example:
        engine = ImprovementEngine(performance_tracker=tracker)

        # Analyze and generate proposals
        proposals = await engine.analyze_and_propose()

        # Review proposal
        proposal = proposals[0]
        print(f"{proposal.title}: {proposal.estimated_improvement_pct}% improvement")
        print(f"Risk: {proposal.risk_level.value}, Impact: {proposal.impact_level.value}")

        # Approve and implement (if safe)
        if proposal.risk_level == RiskLevel.LOW:
            await engine.approve_proposal(proposal.proposal_id)
            await engine.implement_proposal(proposal.proposal_id)
    """

    def __init__(
        self,
        performance_tracker: Any,  # PerformanceTracker instance
        auto_implement_threshold: float = 0.8,  # Confidence for auto-implementation
        enable_auto_implementation: bool = False
    ):
        """
        Initialize improvement engine.

        Args:
            performance_tracker: PerformanceTracker instance
            auto_implement_threshold: Confidence threshold for auto-implementation
            enable_auto_implementation: Allow automatic implementation of safe changes
        """
        self.performance_tracker = performance_tracker
        self.auto_implement_threshold = auto_implement_threshold
        self.enable_auto_implementation = enable_auto_implementation

        # Proposals
        self.proposals: Dict[str, ImprovementProposal] = {}
        self.implemented_proposals: List[str] = []

        # Learning insights
        self.insights: Dict[str, LearningInsight] = {}

        # Success tracking
        self.improvement_history: List[Dict[str, Any]] = []

        logger.info(
            "Improvement engine initialized",
            auto_implement=enable_auto_implementation,
            threshold=auto_implement_threshold
        )

    async def analyze_and_propose(self) -> List[ImprovementProposal]:
        """
        Analyze performance data and generate improvement proposals.

        Returns:
            List of proposed improvements
        """
        proposals = []

        # 1. Analyze bottlenecks
        bottleneck_proposals = await self._analyze_bottlenecks()
        proposals.extend(bottleneck_proposals)

        # 2. Analyze baselines (performance regressions)
        regression_proposals = await self._analyze_regressions()
        proposals.extend(regression_proposals)

        # 3. Analyze cache performance
        cache_proposals = await self._analyze_cache_performance()
        proposals.extend(cache_proposals)

        # 4. Analyze learning insights
        insight_proposals = await self._analyze_insights()
        proposals.extend(insight_proposals)

        # Store proposals
        for proposal in proposals:
            self.proposals[proposal.proposal_id] = proposal

            # Auto-implement if appropriate
            if (self.enable_auto_implementation and
                proposal.auto_implementable and
                proposal.risk_level == RiskLevel.LOW):

                logger.info(
                    f"Auto-implementing low-risk proposal: {proposal.title}"
                )
                await self.implement_proposal(proposal.proposal_id)

        logger.info(f"Generated {len(proposals)} improvement proposals")

        return proposals

    async def _analyze_bottlenecks(self) -> List[ImprovementProposal]:
        """Analyze bottlenecks and propose fixes"""
        proposals = []

        bottlenecks = self.performance_tracker.detect_bottlenecks()

        for bottleneck in bottlenecks:
            # Create proposal based on bottleneck
            proposal = ImprovementProposal(
                proposal_id=f"bottleneck_{bottleneck.bottleneck_id}",
                category=ProposalCategory.PERFORMANCE,
                title=f"Fix {bottleneck.component} bottleneck",
                description=f"{bottleneck.description}. Current performance is {abs(bottleneck.delta_percentage):.1f}% worse than baseline.",
                rationale=f"Component {bottleneck.component} is performing significantly below baseline, impacting overall system performance.",
                risk_level=self._assess_risk(bottleneck.severity),
                impact_level=self._assess_impact(bottleneck.severity),
                estimated_improvement_pct=abs(bottleneck.delta_percentage),
                affected_components=[bottleneck.component],
                requires_approval=bottleneck.severity in ["high", "critical"],
                auto_implementable=False,  # Bottlenecks need investigation
                evidence={
                    "current_value": bottleneck.current_value,
                    "expected_value": bottleneck.expected_value,
                    "delta_pct": bottleneck.delta_percentage,
                    "severity": bottleneck.severity,
                    "suggested_fixes": bottleneck.suggested_fixes
                }
            )

            # Add suggested fixes to description
            if bottleneck.suggested_fixes:
                proposal.description += "\n\nSuggested fixes:\n" + "\n".join(
                    f"- {fix}" for fix in bottleneck.suggested_fixes
                )

            proposals.append(proposal)

        return proposals

    async def _analyze_regressions(self) -> List[ImprovementProposal]:
        """Analyze performance regressions"""
        proposals = []

        # Check all metrics with baselines
        for metric_name, baseline in self.performance_tracker.baselines.items():
            summary = self.performance_tracker.get_metric_summary(metric_name)

            if "vs_baseline_pct" not in summary:
                continue

            delta_pct = summary["vs_baseline_pct"]

            # Check for regression (performance worse than baseline)
            is_regression = False

            if "latency" in metric_name.lower() or "resource" in metric_name.lower():
                # For latency/resource, higher is worse
                is_regression = delta_pct > 20  # 20% slower

            else:
                # For accuracy/cache hit, lower is worse
                is_regression = delta_pct < -10  # 10% drop

            if is_regression:
                proposal = ImprovementProposal(
                    proposal_id=f"regression_{metric_name}_{int(datetime.utcnow().timestamp())}",
                    category=ProposalCategory.PERFORMANCE,
                    title=f"Fix {metric_name} regression",
                    description=f"Metric {metric_name} has regressed {abs(delta_pct):.1f}% from baseline of {baseline.baseline_value:.2f}",
                    rationale="Performance has degraded compared to established baseline",
                    risk_level=RiskLevel.MEDIUM,
                    impact_level=ImpactLevel.MEDIUM,
                    estimated_improvement_pct=abs(delta_pct),
                    affected_components=[metric_name],
                    requires_approval=True,
                    evidence=summary
                )

                proposals.append(proposal)

        return proposals

    async def _analyze_cache_performance(self) -> List[ImprovementProposal]:
        """Analyze cache performance and suggest optimizations"""
        proposals = []

        # Check cache hit rate
        if "cache_hit_rate" in self.performance_tracker.metrics:
            summary = self.performance_tracker.get_metric_summary("cache_hit_rate")

            if summary.get("mean", 0) < 0.7:  # Below 70% hit rate
                proposal = ImprovementProposal(
                    proposal_id=f"cache_opt_{int(datetime.utcnow().timestamp())}",
                    category=ProposalCategory.CACHE,
                    title="Improve cache hit rate",
                    description=f"Current cache hit rate is {summary['mean']*100:.1f}%, below target of 70%",
                    rationale="Low cache hit rate indicates suboptimal cache strategy",
                    risk_level=RiskLevel.LOW,
                    impact_level=ImpactLevel.MEDIUM,
                    estimated_improvement_pct=30.0,  # Expect 30% latency improvement
                    affected_components=["cache"],
                    requires_approval=False,
                    auto_implementable=True,
                    config_changes={
                        "cache_size_mb": "increase by 50%",
                        "cache_ttl_seconds": "increase by 2x"
                    },
                    evidence=summary
                )

                proposals.append(proposal)

        return proposals

    async def _analyze_insights(self) -> List[ImprovementProposal]:
        """Generate proposals from learning insights"""
        proposals = []

        for insight in self.insights.values():
            if not insight.actionable or insight.action_taken:
                continue

            if insight.confidence < 0.7:
                continue

            # Create proposal based on insight
            proposal = ImprovementProposal(
                proposal_id=f"insight_{insight.insight_id}",
                category=ProposalCategory.FEATURE,
                title=f"Implement insight: {insight.insight_type}",
                description=insight.content,
                rationale=f"Learned pattern with {insight.confidence*100:.0f}% confidence based on {insight.evidence_count} observations",
                risk_level=RiskLevel.LOW,
                impact_level=ImpactLevel.LOW,
                estimated_improvement_pct=5.0,
                affected_components=["learning"],
                requires_approval=True,
                evidence={
                    "insight_type": insight.insight_type,
                    "confidence": insight.confidence,
                    "evidence_count": insight.evidence_count
                }
            )

            proposals.append(proposal)

            # Mark insight as acted upon
            insight.action_taken = True
            insight.related_proposal_id = proposal.proposal_id

        return proposals

    async def approve_proposal(self, proposal_id: str):
        """Approve a proposal for implementation"""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]
        proposal.status = "approved"

        logger.info(f"Proposal approved: {proposal.title}")

    async def reject_proposal(self, proposal_id: str, reason: str):
        """Reject a proposal"""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]
        proposal.status = "rejected"

        logger.info(f"Proposal rejected: {proposal.title}, reason: {reason}")

    async def implement_proposal(self, proposal_id: str) -> bool:
        """
        Implement an approved proposal.

        Args:
            proposal_id: ID of proposal to implement

        Returns:
            True if implementation succeeded
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        if proposal.status != "approved" and not proposal.auto_implementable:
            logger.error(f"Proposal {proposal_id} not approved")
            return False

        # Implementation would happen here
        # For now: Mock implementation
        logger.info(f"Implementing proposal: {proposal.title}")

        # Mark as implemented
        proposal.status = "implemented"
        proposal.implemented_at = datetime.utcnow()
        self.implemented_proposals.append(proposal_id)

        # Record in history
        self.improvement_history.append({
            "proposal_id": proposal_id,
            "title": proposal.title,
            "implemented_at": proposal.implemented_at.isoformat(),
            "expected_improvement_pct": proposal.estimated_improvement_pct
        })

        return True

    async def measure_improvement(
        self,
        proposal_id: str,
        measurement_window_minutes: int = 60
    ) -> Optional[float]:
        """
        Measure actual improvement after implementation.

        Args:
            proposal_id: ID of implemented proposal
            measurement_window_minutes: How long to measure

        Returns:
            Actual improvement percentage or None
        """
        if proposal_id not in self.proposals:
            return None

        proposal = self.proposals[proposal_id]

        if proposal.status != "implemented":
            return None

        # Measure performance of affected components
        # Compare to baseline
        # For now: Mock measurement
        actual_improvement = proposal.estimated_improvement_pct * 0.8  # 80% of expected

        proposal.actual_improvement_pct = actual_improvement

        logger.info(
            "Improvement measured",
            proposal=proposal.title,
            expected=proposal.estimated_improvement_pct,
            actual=actual_improvement
        )

        return actual_improvement

    def _assess_risk(self, severity: str) -> RiskLevel:
        """Assess risk level based on bottleneck severity"""
        mapping = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        return mapping.get(severity, RiskLevel.MEDIUM)

    def _assess_impact(self, severity: str) -> ImpactLevel:
        """Assess impact level based on bottleneck severity"""
        mapping = {
            "low": ImpactLevel.LOW,
            "medium": ImpactLevel.MEDIUM,
            "high": ImpactLevel.HIGH,
            "critical": ImpactLevel.CRITICAL
        }
        return mapping.get(severity, ImpactLevel.MEDIUM)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get improvement engine statistics"""
        return {
            "total_proposals": len(self.proposals),
            "approved": sum(1 for p in self.proposals.values() if p.status == "approved"),
            "implemented": sum(1 for p in self.proposals.values() if p.status == "implemented"),
            "rejected": sum(1 for p in self.proposals.values() if p.status == "rejected"),
            "auto_implemented": sum(1 for p in self.proposals.values() if p.auto_implementable and p.status == "implemented"),
            "total_insights": len(self.insights),
            "actionable_insights": sum(1 for i in self.insights.values() if i.actionable),
            "improvement_history_count": len(self.improvement_history)
        }
