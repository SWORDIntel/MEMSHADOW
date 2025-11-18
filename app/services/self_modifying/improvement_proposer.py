"""
Improvement Proposer
Phase 8.4: LLM-powered code improvement suggestions

Generates improvement proposals for code using:
- Code metrics analysis
- Pattern matching
- LLM-powered suggestions (when available)
- Best practices database

All proposals include:
- Risk assessment
- Estimated benefit
- Test requirements
- Rollback plan
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import structlog

from app.services.self_modifying.code_introspector import CodeMetrics, CodeComplexity

logger = structlog.get_logger()


class ImprovementCategory(Enum):
    """Categories of code improvements"""
    PERFORMANCE = "performance"        # Speed/efficiency
    READABILITY = "readability"        # Code clarity
    MAINTAINABILITY = "maintainability" # Easier to maintain
    SECURITY = "security"              # Security fixes
    BUG_FIX = "bug_fix"               # Fix bugs
    REFACTORING = "refactoring"        # Restructure code
    TESTING = "testing"                # Add/improve tests
    DOCUMENTATION = "documentation"    # Improve docs


@dataclass
class CodeImprovement:
    """
    Proposed code improvement.

    Represents a suggested modification to the codebase.
    """
    improvement_id: str
    category: ImprovementCategory

    # Target
    target_file: str
    target_function: Optional[str] = None
    target_lines: Optional[range] = None

    # Description
    title: str = ""
    description: str = ""
    rationale: str = ""

    # Proposed change
    current_code: str = ""
    proposed_code: str = ""

    # Assessment
    risk_level: str = "medium"  # low, medium, high, critical
    estimated_benefit: str = "medium"  # low, medium, high
    confidence: float = 0.5  # 0.0 to 1.0

    # Requirements
    requires_tests: bool = True
    requires_human_review: bool = False
    requires_backup: bool = True

    # Validation
    test_plan: List[str] = field(default_factory=list)
    rollback_plan: str = ""

    # Metadata
    proposed_at: datetime = field(default_factory=datetime.utcnow)
    proposed_by: str = "system"

    @property
    def should_auto_apply(self) -> bool:
        """
        Should this improvement be automatically applied?

        Only safe, low-risk improvements qualify.
        """
        return (
            self.risk_level == "low" and
            self.confidence > 0.8 and
            not self.requires_human_review
        )


class ImprovementProposer:
    """
    Improvement Proposer for code optimization.

    Analyzes code and generates improvement proposals using:
    - Static analysis (metrics)
    - Pattern detection
    - LLM suggestions (when available)
    - Best practices

    All proposals are assessed for risk and benefit.

    Example:
        proposer = ImprovementProposer()

        # Analyze function and get proposals
        proposals = await proposer.propose_improvements(
            code_metrics=metrics,
            source_code=source
        )

        # Filter safe auto-apply proposals
        auto_apply = [p for p in proposals if p.should_auto_apply]
    """

    def __init__(
        self,
        enable_llm: bool = False,
        llm_model: Optional[str] = None
    ):
        """
        Initialize improvement proposer.

        Args:
            enable_llm: Use LLM for suggestions
            llm_model: LLM model to use (if enabled)
        """
        self.enable_llm = enable_llm
        self.llm_model = llm_model

        # Proposal history
        self.proposals: Dict[str, CodeImprovement] = {}

        logger.info(
            "Improvement proposer initialized",
            llm_enabled=enable_llm
        )

    async def propose_improvements(
        self,
        code_metrics: CodeMetrics,
        source_code: str
    ) -> List[CodeImprovement]:
        """
        Propose improvements for code.

        Args:
            code_metrics: Metrics for the code
            source_code: The source code

        Returns:
            List of proposed improvements
        """
        proposals = []

        # 1. Metrics-based proposals
        metric_proposals = await self._propose_from_metrics(
            code_metrics, source_code
        )
        proposals.extend(metric_proposals)

        # 2. Pattern-based proposals
        pattern_proposals = await self._propose_from_patterns(
            source_code, code_metrics
        )
        proposals.extend(pattern_proposals)

        # 3. LLM-based proposals (if enabled)
        if self.enable_llm:
            llm_proposals = await self._propose_from_llm(
                code_metrics, source_code
            )
            proposals.extend(llm_proposals)

        # Store proposals
        for proposal in proposals:
            self.proposals[proposal.improvement_id] = proposal

        logger.info(
            "Improvements proposed",
            function=code_metrics.function_name,
            proposals_count=len(proposals)
        )

        return proposals

    async def _propose_from_metrics(
        self,
        metrics: CodeMetrics,
        source: str
    ) -> List[CodeImprovement]:
        """Generate proposals based on code metrics"""
        proposals = []

        # Missing docstring
        if not metrics.has_docstring:
            proposals.append(CodeImprovement(
                improvement_id=f"docstring_{metrics.function_name}",
                category=ImprovementCategory.DOCUMENTATION,
                target_file=metrics.file_path,
                target_function=metrics.function_name,
                title="Add docstring",
                description=f"Add docstring to {metrics.function_name}",
                rationale="Docstrings improve code understandability",
                current_code=source,
                proposed_code=self._generate_docstring_addition(source, metrics),
                risk_level="low",
                estimated_benefit="medium",
                confidence=0.9,
                requires_human_review=False,
                test_plan=["Verify docstring exists", "Check docstring format"],
                rollback_plan="Remove added docstring"
            ))

        # Missing type hints
        if not metrics.has_type_hints:
            proposals.append(CodeImprovement(
                improvement_id=f"types_{metrics.function_name}",
                category=ImprovementCategory.READABILITY,
                target_file=metrics.file_path,
                target_function=metrics.function_name,
                title="Add type hints",
                description=f"Add type hints to {metrics.function_name}",
                rationale="Type hints improve code safety and IDE support",
                current_code=source,
                proposed_code="# Type hints would be added here",
                risk_level="low",
                estimated_benefit="medium",
                confidence=0.8,
                requires_human_review=True,  # Types need verification
                test_plan=["Run mypy", "Verify type correctness"],
                rollback_plan="Remove type hints"
            ))

        # High complexity
        if metrics.cyclomatic_complexity > 10:
            proposals.append(CodeImprovement(
                improvement_id=f"complexity_{metrics.function_name}",
                category=ImprovementCategory.REFACTORING,
                target_file=metrics.file_path,
                target_function=metrics.function_name,
                title="Reduce complexity",
                description=f"Refactor {metrics.function_name} to reduce complexity",
                rationale=f"Cyclomatic complexity is {metrics.cyclomatic_complexity} (target: <10)",
                current_code=source,
                proposed_code="# Refactored code would go here",
                risk_level="high",  # Refactoring is risky
                estimated_benefit="high",
                confidence=0.6,
                requires_human_review=True,
                requires_tests=True,
                test_plan=[
                    "Comprehensive unit tests",
                    "Integration tests",
                    "Regression tests"
                ],
                rollback_plan="Revert to original implementation"
            ))

        return proposals

    async def _propose_from_patterns(
        self,
        source: str,
        metrics: CodeMetrics
    ) -> List[CodeImprovement]:
        """Detect anti-patterns and propose fixes"""
        proposals = []

        # Detect bare except
        if "except:" in source:
            proposals.append(CodeImprovement(
                improvement_id=f"except_{metrics.function_name}",
                category=ImprovementCategory.BUG_FIX,
                target_file=metrics.file_path,
                target_function=metrics.function_name,
                title="Replace bare except",
                description="Replace 'except:' with specific exception types",
                rationale="Bare except catches all exceptions, including KeyboardInterrupt",
                risk_level="low",
                estimated_benefit="medium",
                confidence=0.9,
                test_plan=["Verify exception handling still works"],
                rollback_plan="Revert to bare except"
            ))

        # Detect mutable default arguments
        if "def " in source and "=[]" in source or "={}" in source:
            proposals.append(CodeImprovement(
                improvement_id=f"mutable_default_{metrics.function_name}",
                category=ImprovementCategory.BUG_FIX,
                target_file=metrics.file_path,
                target_function=metrics.function_name,
                title="Fix mutable default argument",
                description="Replace mutable default argument with None",
                rationale="Mutable defaults can cause unexpected behavior",
                risk_level="medium",
                estimated_benefit="high",
                confidence=0.85,
                requires_tests=True,
                test_plan=["Test function with multiple calls"],
                rollback_plan="Revert to mutable default"
            ))

        return proposals

    async def _propose_from_llm(
        self,
        metrics: CodeMetrics,
        source: str
    ) -> List[CodeImprovement]:
        """
        Use LLM to suggest improvements.

        In production: would call Claude/GPT for suggestions.
        For now: returns mock proposals.
        """
        # Mock LLM suggestion
        if metrics.complexity_level == CodeComplexity.HIGH:
            return [CodeImprovement(
                improvement_id=f"llm_{metrics.function_name}",
                category=ImprovementCategory.PERFORMANCE,
                target_file=metrics.file_path,
                target_function=metrics.function_name,
                title="Optimize algorithm",
                description="LLM suggested algorithmic optimization",
                rationale="Analysis suggests O(n²) can be reduced to O(n log n)",
                risk_level="high",
                estimated_benefit="high",
                confidence=0.7,
                requires_human_review=True,
                requires_tests=True,
                test_plan=[
                    "Performance benchmarks",
                    "Correctness tests",
                    "Edge case tests"
                ],
                rollback_plan="Revert to original algorithm"
            )]

        return []

    def _generate_docstring_addition(
        self,
        source: str,
        metrics: CodeMetrics
    ) -> str:
        """Generate source with added docstring"""
        # Simple mock - in production would be more sophisticated
        lines = source.split('\n')

        # Find function definition
        for i, line in enumerate(lines):
            if f"def {metrics.function_name}" in line or \
               f"async def {metrics.function_name}" in line:
                # Insert docstring after function def
                indent = len(line) - len(line.lstrip())
                docstring = f'{" " * (indent + 4)}"""TODO: Add description"""'

                lines.insert(i + 1, docstring)
                break

        return '\n'.join(lines)

    async def assess_proposal(
        self,
        proposal: CodeImprovement
    ) -> Dict[str, Any]:
        """
        Assess a proposal's risk and benefit.

        Args:
            proposal: Proposal to assess

        Returns:
            Assessment results
        """
        # Risk factors
        risk_factors = []

        if proposal.category in [ImprovementCategory.REFACTORING, ImprovementCategory.PERFORMANCE]:
            risk_factors.append("Changes program logic")

        if not proposal.test_plan:
            risk_factors.append("No test plan")

        if proposal.confidence < 0.7:
            risk_factors.append("Low confidence")

        # Benefit factors
        benefit_factors = []

        if proposal.category == ImprovementCategory.SECURITY:
            benefit_factors.append("Improves security")

        if proposal.category == ImprovementCategory.PERFORMANCE:
            benefit_factors.append("Improves performance")

        if proposal.category == ImprovementCategory.BUG_FIX:
            benefit_factors.append("Fixes bugs")

        return {
            "proposal_id": proposal.improvement_id,
            "risk_level": proposal.risk_level,
            "risk_factors": risk_factors,
            "estimated_benefit": proposal.estimated_benefit,
            "benefit_factors": benefit_factors,
            "should_auto_apply": proposal.should_auto_apply,
            "requires_human_review": proposal.requires_human_review
        }
