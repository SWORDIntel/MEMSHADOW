"""
Self-Modifying Architecture
Phase 8.4: Safe autonomous code modification

Implements self-modification capabilities with safety-first design:
- Code introspection and analysis
- LLM-powered improvement proposals
- Automated test generation
- Safe modification with rollback
- Performance validation

Safety Features:
- Gradual privilege escalation (read-only → read-write)
- Mandatory test coverage for modifications
- Automatic rollback on failure
- Human approval for critical changes
- Comprehensive audit logging

⚠️ WARNING: Self-modifying code is inherently risky. This implementation
prioritizes safety and includes multiple safeguards.
"""

from app.services.self_modifying.code_introspector import (
    CodeIntrospector,
    CodeMetrics,
    CodeComplexity
)

from app.services.self_modifying.improvement_proposer import (
    ImprovementProposer,
    CodeImprovement,
    ImprovementCategory
)

from app.services.self_modifying.test_generator import (
    TestGenerator,
    GeneratedTest,
    TestCoverage
)

from app.services.self_modifying.safe_modifier import (
    SafeModifier,
    ModificationResult,
    SafetyLevel
)

from app.services.self_modifying.self_modifying_engine import (
    SelfModifyingEngine,
    ModificationRequest,
    ModificationStatus
)

__all__ = [
    # Introspection
    "CodeIntrospector",
    "CodeMetrics",
    "CodeComplexity",

    # Improvement
    "ImprovementProposer",
    "CodeImprovement",
    "ImprovementCategory",

    # Testing
    "TestGenerator",
    "GeneratedTest",
    "TestCoverage",

    # Safe Modification
    "SafeModifier",
    "ModificationResult",
    "SafetyLevel",

    # Main Engine
    "SelfModifyingEngine",
    "ModificationRequest",
    "ModificationStatus",
]
