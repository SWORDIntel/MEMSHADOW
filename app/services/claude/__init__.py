"""
MEMSHADOW Claude Integration Services
Phase 5: Claude Deep Integration

This package provides specialized services for Claude AI interactions:
- Memory adapter for turn-by-turn conversation tracking
- Code memory system with dependency tracking
- Session continuity for resuming work
- Intelligent context injection
- Project-level memory organization
"""

from app.services.claude.claude_adapter import (
    ClaudeMemoryAdapter,
    ClaudeTurn,
    ClaudeArtifact,
    TurnType,
    ArtifactType,
    claude_adapter
)

from app.services.claude.code_memory import (
    CodeMemorySystem,
    CodeArtifact,
    CodeLanguage,
    code_memory
)

from app.services.claude.session_continuity import (
    SessionContinuityBridge,
    SessionCheckpoint,
    SessionStatus,
    session_continuity
)

from app.services.claude.context_injection import (
    IntelligentContextInjection,
    ContextSource,
    QueryIntent,
    context_injector
)

from app.services.claude.project_memory import (
    ProjectMemoryOrganizer,
    Project,
    ProjectMilestone,
    ProjectStatus,
    MilestoneStatus,
    project_organizer
)

__all__ = [
    # Claude Memory Adapter
    "ClaudeMemoryAdapter",
    "ClaudeTurn",
    "ClaudeArtifact",
    "TurnType",
    "ArtifactType",
    "claude_adapter",

    # Code Memory System
    "CodeMemorySystem",
    "CodeArtifact",
    "CodeLanguage",
    "code_memory",

    # Session Continuity
    "SessionContinuityBridge",
    "SessionCheckpoint",
    "SessionStatus",
    "session_continuity",

    # Context Injection
    "IntelligentContextInjection",
    "ContextSource",
    "QueryIntent",
    "context_injector",

    # Project Memory
    "ProjectMemoryOrganizer",
    "Project",
    "ProjectMilestone",
    "ProjectStatus",
    "MilestoneStatus",
    "project_organizer",
]
