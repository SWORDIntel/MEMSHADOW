"""
Session Continuity Bridge
Phase 5: Claude Deep Integration - Resume Claude sessions with context

Features:
- Session checkpointing
- Context generation for resumption
- Key decision tracking
- Objective tracking
- Next steps recommendation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import structlog
from dataclasses import dataclass, field
import uuid

logger = structlog.get_logger()


class SessionStatus(str, Enum):
    """Session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class SessionCheckpoint:
    """Session checkpoint for continuity"""
    checkpoint_id: str
    session_id: str
    project_id: str
    created_at: datetime

    # Context
    summary: str  # What has been accomplished
    key_decisions: List[str]  # Important decisions made
    current_objective: str  # What we're working on
    next_steps: List[str]  # Recommended next actions

    # State
    conversation_id: str
    turn_count: int
    total_tokens: int

    # Artifacts
    code_artifacts: List[str]  # Artifact IDs
    documents: List[str]  # Document IDs

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionContinuityBridge:
    """
    Manages session continuity for Claude interactions.

    Features:
    - Create checkpoints at key moments
    - Generate resumption context
    - Track session objectives and progress
    - Recommend next steps
    - Link related sessions

    Example:
        bridge = SessionContinuityBridge()

        # Create checkpoint
        checkpoint = await bridge.create_checkpoint(
            session_id="sess_123",
            project_id="proj_456",
            summary="Implemented user authentication with JWT",
            key_decisions=[
                "Using bcrypt for password hashing",
                "JWT expiry set to 7 days"
            ],
            current_objective="Add password reset functionality",
            next_steps=[
                "Create password reset token generation",
                "Implement email sending",
                "Add reset form to frontend"
            ],
            conversation_id="conv_789"
        )

        # Later, resume session
        context = await bridge.generate_resumption_context(
            checkpoint["checkpoint_id"]
        )
    """

    def __init__(self):
        # In-memory storage (would be database in production)
        self.checkpoints: Dict[str, SessionCheckpoint] = {}
        self.sessions: Dict[str, List[str]] = {}  # session_id -> [checkpoint_ids]
        self.project_sessions: Dict[str, List[str]] = {}  # project_id -> [session_ids]

        logger.info("Session continuity bridge initialized")

    async def create_checkpoint(
        self,
        session_id: str,
        project_id: str,
        summary: str,
        key_decisions: List[str],
        current_objective: str,
        next_steps: List[str],
        conversation_id: str,
        turn_count: int = 0,
        total_tokens: int = 0,
        code_artifacts: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create session checkpoint.

        Args:
            session_id: Session ID
            project_id: Project ID
            summary: What has been accomplished
            key_decisions: Important decisions made
            current_objective: Current work objective
            next_steps: Recommended next actions
            conversation_id: Conversation ID
            turn_count: Number of conversation turns
            total_tokens: Total tokens used
            code_artifacts: Code artifact IDs
            documents: Document IDs
            metadata: Additional metadata

        Returns:
            Checkpoint metadata
        """
        checkpoint_id = str(uuid.uuid4())

        checkpoint = SessionCheckpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            project_id=project_id,
            created_at=datetime.utcnow(),
            summary=summary,
            key_decisions=key_decisions,
            current_objective=current_objective,
            next_steps=next_steps,
            conversation_id=conversation_id,
            turn_count=turn_count,
            total_tokens=total_tokens,
            code_artifacts=code_artifacts or [],
            documents=documents or [],
            metadata=metadata or {}
        )

        self.checkpoints[checkpoint_id] = checkpoint

        # Update indexes
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(checkpoint_id)

        if project_id not in self.project_sessions:
            self.project_sessions[project_id] = []
        if session_id not in self.project_sessions[project_id]:
            self.project_sessions[project_id].append(session_id)

        logger.info(
            "Checkpoint created",
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            project_id=project_id,
            turn_count=turn_count
        )

        return {
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "project_id": project_id,
            "created_at": checkpoint.created_at.isoformat(),
            "summary": summary,
            "next_steps": next_steps
        }

    async def generate_resumption_context(
        self,
        checkpoint_id: str,
        include_code_snippets: bool = True,
        max_snippet_lines: int = 20
    ) -> str:
        """
        Generate context for resuming session.

        Creates Claude-friendly XML context for seamless resumption.

        Args:
            checkpoint_id: Checkpoint ID
            include_code_snippets: Include code snippet previews
            max_snippet_lines: Max lines per snippet

        Returns:
            Formatted resumption context
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint = self.checkpoints[checkpoint_id]

        context_parts = [
            "<session_resume>",
            "  <project>",
            f"    <project_id>{checkpoint.project_id}</project_id>",
            f"    <session_id>{checkpoint.session_id}</session_id>",
            "  </project>",
            "",
            "  <progress>",
            "    <summary>",
            f"      {checkpoint.summary}",
            "    </summary>",
            "",
            "    <key_decisions>"
        ]

        for i, decision in enumerate(checkpoint.key_decisions, 1):
            context_parts.append(f"      {i}. {decision}")

        context_parts.extend([
            "    </key_decisions>",
            "  </progress>",
            "",
            "  <current_state>",
            "    <objective>",
            f"      {checkpoint.current_objective}",
            "    </objective>",
            "",
            "    <next_steps>"
        ])

        for i, step in enumerate(checkpoint.next_steps, 1):
            context_parts.append(f"      {i}. {step}")

        context_parts.extend([
            "    </next_steps>",
            "  </current_state>"
        ])

        # Add code artifacts if requested
        if include_code_snippets and checkpoint.code_artifacts:
            context_parts.append("")
            context_parts.append("  <code_artifacts>")

            for artifact_id in checkpoint.code_artifacts[:5]:  # Limit to 5 artifacts
                # In production, would fetch actual artifact
                context_parts.append(f"    <artifact id=\"{artifact_id}\">")
                context_parts.append("      [Code snippet would be included here]")
                context_parts.append("    </artifact>")

            if len(checkpoint.code_artifacts) > 5:
                context_parts.append(f"    <!-- {len(checkpoint.code_artifacts) - 5} more artifacts available -->")

            context_parts.append("  </code_artifacts>")

        context_parts.extend([
            "",
            "  <statistics>",
            f"    <turns>{checkpoint.turn_count}</turns>",
            f"    <tokens_used>{checkpoint.total_tokens}</tokens_used>",
            f"    <checkpoint_time>{checkpoint.created_at.isoformat()}</checkpoint_time>",
            "  </statistics>",
            "</session_resume>"
        ])

        return "\\n".join(context_parts)

    async def get_session_history(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all checkpoints for a session.

        Args:
            session_id: Session ID

        Returns:
            List of checkpoints
        """
        checkpoint_ids = self.sessions.get(session_id, [])

        checkpoints = [
            {
                "checkpoint_id": cp_id,
                "created_at": self.checkpoints[cp_id].created_at.isoformat(),
                "summary": self.checkpoints[cp_id].summary,
                "current_objective": self.checkpoints[cp_id].current_objective,
                "turn_count": self.checkpoints[cp_id].turn_count,
                "total_tokens": self.checkpoints[cp_id].total_tokens
            }
            for cp_id in checkpoint_ids
        ]

        return sorted(checkpoints, key=lambda x: x["created_at"])

    async def get_project_sessions(
        self,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a project.

        Args:
            project_id: Project ID

        Returns:
            List of sessions with latest checkpoint
        """
        session_ids = self.project_sessions.get(project_id, [])

        sessions = []
        for session_id in session_ids:
            checkpoint_ids = self.sessions.get(session_id, [])
            if not checkpoint_ids:
                continue

            # Get latest checkpoint
            latest_checkpoint_id = checkpoint_ids[-1]
            latest_checkpoint = self.checkpoints[latest_checkpoint_id]

            sessions.append({
                "session_id": session_id,
                "checkpoint_count": len(checkpoint_ids),
                "latest_checkpoint": {
                    "checkpoint_id": latest_checkpoint_id,
                    "summary": latest_checkpoint.summary,
                    "current_objective": latest_checkpoint.current_objective,
                    "created_at": latest_checkpoint.created_at.isoformat()
                },
                "total_turns": sum(
                    self.checkpoints[cp_id].turn_count
                    for cp_id in checkpoint_ids
                ),
                "total_tokens": sum(
                    self.checkpoints[cp_id].total_tokens
                    for cp_id in checkpoint_ids
                )
            })

        return sorted(sessions, key=lambda x: x["latest_checkpoint"]["created_at"], reverse=True)

    async def suggest_next_session(
        self,
        project_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest what to work on next based on latest checkpoint.

        Args:
            project_id: Project ID

        Returns:
            Suggested next session context
        """
        sessions = await self.get_project_sessions(project_id)

        if not sessions:
            return None

        latest_session = sessions[0]
        checkpoint_ids = self.sessions[latest_session["session_id"]]
        latest_checkpoint = self.checkpoints[checkpoint_ids[-1]]

        return {
            "suggested_objective": latest_checkpoint.current_objective,
            "next_steps": latest_checkpoint.next_steps,
            "previous_session_id": latest_session["session_id"],
            "resume_checkpoint_id": checkpoint_ids[-1],
            "context_preview": await self.generate_resumption_context(
                checkpoint_ids[-1],
                include_code_snippets=False
            )
        }

    async def mark_objective_completed(
        self,
        checkpoint_id: str,
        completion_notes: str
    ):
        """
        Mark objective as completed.

        Args:
            checkpoint_id: Checkpoint ID
            completion_notes: Completion notes
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint = self.checkpoints[checkpoint_id]
        checkpoint.metadata["completed"] = True
        checkpoint.metadata["completion_notes"] = completion_notes
        checkpoint.metadata["completed_at"] = datetime.utcnow().isoformat()

        logger.info(
            "Objective marked completed",
            checkpoint_id=checkpoint_id,
            objective=checkpoint.current_objective
        )

    async def update_next_steps(
        self,
        checkpoint_id: str,
        next_steps: List[str]
    ):
        """
        Update next steps for checkpoint.

        Args:
            checkpoint_id: Checkpoint ID
            next_steps: Updated next steps
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint = self.checkpoints[checkpoint_id]
        checkpoint.next_steps = next_steps
        checkpoint.metadata["next_steps_updated_at"] = datetime.utcnow().isoformat()

        logger.info(
            "Next steps updated",
            checkpoint_id=checkpoint_id,
            steps_count=len(next_steps)
        )


# Global instance
session_continuity = SessionContinuityBridge()
