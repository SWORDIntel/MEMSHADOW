"""
Claude Memory Adapter
Phase 5: Claude Deep Integration - Specialized adapter for Claude interactions

Captures Claude-specific interaction patterns:
- Turn-by-turn conversation tracking
- Artifact extraction (code blocks, documents)
- Context window optimization
- Token estimation for Claude models
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import re
import structlog
from dataclasses import dataclass, field
import uuid

logger = structlog.get_logger()


class TurnType(str, Enum):
    """Claude turn types"""
    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    SYSTEM_PROMPT = "system_prompt"
    THINKING = "thinking"  # Claude's internal thinking
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class ArtifactType(str, Enum):
    """Types of artifacts in Claude conversations"""
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    DATA = "data"
    DIAGRAM = "diagram"


@dataclass
class ClaudeTurn:
    """Individual turn in Claude conversation"""
    turn_id: str
    turn_type: TurnType
    content: str
    timestamp: datetime
    token_count: int = 0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeArtifact:
    """Artifact extracted from Claude interaction"""
    artifact_id: str
    artifact_type: ArtifactType
    language: Optional[str]  # For code artifacts
    content: str
    title: Optional[str] = None
    description: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.utcnow)


class ClaudeMemoryAdapter:
    """
    Specialized adapter for capturing Claude interactions.

    Features:
    - Turn-by-turn conversation tracking
    - Artifact extraction from responses
    - Code block detection and language identification
    - Claude-specific token estimation
    - Context window optimization
    - Thinking block capture (when available)

    Example:
        adapter = ClaudeMemoryAdapter()

        # Capture user message
        user_turn = await adapter.capture_user_message(
            content="Write a Python function for fibonacci",
            project_id="proj_123"
        )

        # Capture Claude response
        response_turn = await adapter.capture_assistant_response(
            content="Here's a Fibonacci function:\\n```python\\ndef fib(n)...```",
            conversation_id=user_turn["conversation_id"]
        )

        # Extract artifacts
        artifacts = await adapter.extract_artifacts(response_turn["content"])
    """

    def __init__(
        self,
        max_context_window: int = 200000,  # Claude 3.5 Sonnet context window
        estimate_tokens: bool = True
    ):
        self.max_context_window = max_context_window
        self.estimate_tokens = estimate_tokens

        # Conversation storage (in-memory, would be DB in production)
        self.conversations: Dict[str, List[ClaudeTurn]] = {}

        # Artifact patterns
        self.code_block_pattern = re.compile(
            r'```(\w+)?\n(.*?)```',
            re.DOTALL
        )

        logger.info(
            "Claude memory adapter initialized",
            max_context_window=max_context_window
        )

    async def capture_user_message(
        self,
        content: str,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture user message to Claude.

        Args:
            content: User message content
            project_id: Associated project ID
            conversation_id: Existing conversation or create new
            metadata: Additional metadata

        Returns:
            Turn metadata
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []

        turn = ClaudeTurn(
            turn_id=str(uuid.uuid4()),
            turn_type=TurnType.USER_MESSAGE,
            content=content,
            timestamp=datetime.utcnow(),
            token_count=await self._estimate_tokens(content) if self.estimate_tokens else 0,
            metadata={
                **(metadata or {}),
                "project_id": project_id,
                "conversation_id": conversation_id
            }
        )

        self.conversations[conversation_id].append(turn)

        logger.info(
            "User message captured",
            turn_id=turn.turn_id,
            conversation_id=conversation_id,
            token_count=turn.token_count,
            project_id=project_id
        )

        return {
            "turn_id": turn.turn_id,
            "conversation_id": conversation_id,
            "turn_type": turn.turn_type,
            "token_count": turn.token_count,
            "timestamp": turn.timestamp.isoformat()
        }

    async def capture_assistant_response(
        self,
        content: str,
        conversation_id: str,
        thinking: Optional[str] = None,
        tool_uses: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture Claude's response.

        Args:
            content: Assistant response content
            conversation_id: Conversation ID
            thinking: Claude's thinking (if available)
            tool_uses: Tool use information
            metadata: Additional metadata

        Returns:
            Turn metadata with extracted artifacts
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Extract artifacts from response
        artifacts = await self.extract_artifacts(content)

        turn = ClaudeTurn(
            turn_id=str(uuid.uuid4()),
            turn_type=TurnType.ASSISTANT_RESPONSE,
            content=content,
            timestamp=datetime.utcnow(),
            token_count=await self._estimate_tokens(content) if self.estimate_tokens else 0,
            artifacts=artifacts,
            metadata={
                **(metadata or {}),
                "thinking": thinking,
                "tool_uses": tool_uses or [],
                "artifact_count": len(artifacts)
            }
        )

        # Add thinking as separate turn if present
        if thinking:
            thinking_turn = ClaudeTurn(
                turn_id=str(uuid.uuid4()),
                turn_type=TurnType.THINKING,
                content=thinking,
                timestamp=turn.timestamp,
                token_count=await self._estimate_tokens(thinking) if self.estimate_tokens else 0
            )
            self.conversations[conversation_id].append(thinking_turn)

        self.conversations[conversation_id].append(turn)

        logger.info(
            "Assistant response captured",
            turn_id=turn.turn_id,
            conversation_id=conversation_id,
            token_count=turn.token_count,
            artifact_count=len(artifacts),
            has_thinking=thinking is not None
        )

        return {
            "turn_id": turn.turn_id,
            "conversation_id": conversation_id,
            "turn_type": turn.turn_type,
            "token_count": turn.token_count,
            "artifacts": artifacts,
            "timestamp": turn.timestamp.isoformat()
        }

    async def extract_artifacts(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract artifacts (code blocks, etc.) from content.

        Args:
            content: Content to extract artifacts from

        Returns:
            List of extracted artifacts
        """
        artifacts = []

        # Extract code blocks
        code_blocks = self.code_block_pattern.findall(content)

        for i, (language, code) in enumerate(code_blocks):
            artifact = ClaudeArtifact(
                artifact_id=str(uuid.uuid4()),
                artifact_type=ArtifactType.CODE,
                language=language if language else "plaintext",
                content=code.strip(),
                title=f"Code Block {i+1}" if not language else f"{language.title()} Code"
            )

            artifacts.append({
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "language": artifact.language,
                "content": artifact.content,
                "title": artifact.title,
                "extracted_at": artifact.extracted_at.isoformat()
            })

        logger.debug(
            "Artifacts extracted",
            count=len(artifacts),
            types=[a["artifact_type"] for a in artifacts]
        )

        return artifacts

    async def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude models.

        Uses character-based approximation:
        - ~4 characters per token (English text)
        - Code is denser: ~3.5 characters per token

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Check if content is code-heavy
        code_ratio = len(self.code_block_pattern.findall(text)) / max(len(text.split('\\n')), 1)

        if code_ratio > 0.3:  # Code-heavy content
            chars_per_token = 3.5
        else:  # Regular text
            chars_per_token = 4.0

        estimated_tokens = int(len(text) / chars_per_token)

        return estimated_tokens

    async def get_conversation_history(
        self,
        conversation_id: str,
        max_tokens: Optional[int] = None,
        include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history with optional token limiting.

        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens to return (most recent first)
            include_thinking: Include thinking blocks

        Returns:
            List of turns
        """
        if conversation_id not in self.conversations:
            return []

        turns = self.conversations[conversation_id]

        # Filter out thinking if not requested
        if not include_thinking:
            turns = [t for t in turns if t.turn_type != TurnType.THINKING]

        # Apply token limit (most recent first)
        if max_tokens:
            limited_turns = []
            current_tokens = 0

            for turn in reversed(turns):
                if current_tokens + turn.token_count > max_tokens:
                    break

                limited_turns.insert(0, turn)
                current_tokens += turn.token_count

            turns = limited_turns

        return [
            {
                "turn_id": t.turn_id,
                "turn_type": t.turn_type,
                "content": t.content,
                "token_count": t.token_count,
                "timestamp": t.timestamp.isoformat(),
                "artifacts": t.artifacts,
                "metadata": t.metadata
            }
            for t in turns
        ]

    async def format_for_context_injection(
        self,
        conversation_id: str,
        max_tokens: int = 10000,
        include_artifacts: bool = True
    ) -> str:
        """
        Format conversation history for context injection.

        Returns Claude-friendly XML-style formatting:
        <conversation_history>
          <turn type="user">...</turn>
          <turn type="assistant">...</turn>
        </conversation_history>

        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens for context
            include_artifacts: Include artifact references

        Returns:
            Formatted context string
        """
        turns = await self.get_conversation_history(
            conversation_id,
            max_tokens=max_tokens
        )

        context_parts = ["<conversation_history>"]

        for turn in turns:
            turn_type = turn["turn_type"].replace("_", " ")
            context_parts.append(f'  <turn type="{turn_type}" timestamp="{turn["timestamp"]}">')
            context_parts.append(f"    {turn['content']}")

            if include_artifacts and turn.get("artifacts"):
                context_parts.append("    <artifacts>")
                for artifact in turn["artifacts"]:
                    context_parts.append(
                        f'      <artifact type="{artifact["artifact_type"]}" '
                        f'language="{artifact.get("language", "")}">'
                    )
                    context_parts.append(f'        {artifact.get("title", "Artifact")}')
                    context_parts.append("      </artifact>")
                context_parts.append("    </artifacts>")

            context_parts.append("  </turn>")

        context_parts.append("</conversation_history>")

        return "\\n".join(context_parts)

    async def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics for conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation statistics
        """
        if conversation_id not in self.conversations:
            return {}

        turns = self.conversations[conversation_id]

        total_tokens = sum(t.token_count for t in turns)
        total_artifacts = sum(len(t.artifacts) for t in turns)

        user_turns = [t for t in turns if t.turn_type == TurnType.USER_MESSAGE]
        assistant_turns = [t for t in turns if t.turn_type == TurnType.ASSISTANT_RESPONSE]

        return {
            "conversation_id": conversation_id,
            "total_turns": len(turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "total_tokens": total_tokens,
            "total_artifacts": total_artifacts,
            "context_window_usage": (total_tokens / self.max_context_window * 100),
            "started_at": turns[0].timestamp.isoformat() if turns else None,
            "last_updated": turns[-1].timestamp.isoformat() if turns else None
        }


# Global instance
claude_adapter = ClaudeMemoryAdapter()
