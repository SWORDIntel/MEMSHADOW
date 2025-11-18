"""
Intelligent Context Injection
Phase 5: Claude Deep Integration - Dynamic context generation for Claude

Features:
- Intent-based context selection
- Token-aware context optimization
- Claude-friendly formatting (XML tags)
- Relevance ranking
- Multi-source context aggregation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


class QueryIntent(str, Enum):
    """Detected query intents"""
    CODE_GENERATION = "code_generation"
    CODE_DEBUGGING = "code_debugging"
    CODE_REVIEW = "code_review"
    EXPLANATION = "explanation"
    RESEARCH = "research"
    CONTINUATION = "continuation"
    GENERAL = "general"


@dataclass
class ContextSource:
    """Source of context information"""
    source_type: str  # memory, code, session, project
    content: str
    relevance_score: float
    token_count: int
    metadata: Dict[str, Any]


class IntelligentContextInjection:
    """
    Dynamically generates optimized context for Claude queries.

    Features:
    - Detects query intent
    - Ranks relevant context sources
    - Optimizes for token budget
    - Formats using Claude-friendly XML
    - Aggregates multi-source context

    Example:
        context_injector = IntelligentContextInjection()

        # Generate context for query
        context = await context_injector.generate_context(
            query="How do I fix the authentication bug?",
            project_id="proj_123",
            max_tokens=5000
        )

        # Use in Claude prompt
        prompt = f\"\"\"{context}

        User question: {query}
        \"\"\"
    """

    def __init__(
        self,
        max_context_tokens: int = 20000,
        default_max_tokens: int = 5000
    ):
        self.max_context_tokens = max_context_tokens
        self.default_max_tokens = default_max_tokens

        # Intent detection patterns (simplified)
        self.intent_keywords = {
            QueryIntent.CODE_GENERATION: [
                "create", "generate", "write", "build", "implement", "make"
            ],
            QueryIntent.CODE_DEBUGGING: [
                "fix", "bug", "error", "issue", "debug", "problem", "broken"
            ],
            QueryIntent.CODE_REVIEW: [
                "review", "optimize", "improve", "refactor", "better"
            ],
            QueryIntent.EXPLANATION: [
                "explain", "how does", "what is", "why", "understand"
            ],
            QueryIntent.RESEARCH: [
                "find", "search", "look for", "research", "investigate"
            ],
            QueryIntent.CONTINUATION: [
                "continue", "resume", "next", "more", "keep going"
            ]
        }

        logger.info(
            "Intelligent context injection initialized",
            max_context_tokens=max_context_tokens
        )

    async def generate_context(
        self,
        query: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        include_code: bool = True,
        include_session_history: bool = True
    ) -> str:
        """
        Generate optimized context for query.

        Args:
            query: User query
            project_id: Optional project ID
            session_id: Optional session ID
            max_tokens: Maximum tokens for context
            include_code: Include code artifacts
            include_session_history: Include session history

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.default_max_tokens

        # Detect query intent
        intent = await self._detect_intent(query)

        logger.info(
            "Generating context",
            intent=intent,
            max_tokens=max_tokens,
            project_id=project_id
        )

        # Gather context sources
        context_sources = await self._gather_context_sources(
            query=query,
            intent=intent,
            project_id=project_id,
            session_id=session_id,
            include_code=include_code,
            include_session_history=include_session_history
        )

        # Rank and filter by relevance
        ranked_sources = sorted(
            context_sources,
            key=lambda x: x.relevance_score,
            reverse=True
        )

        # Select sources within token budget
        selected_sources = await self._select_within_budget(
            ranked_sources,
            max_tokens
        )

        # Format context
        formatted_context = await self._format_context(
            selected_sources,
            intent
        )

        logger.info(
            "Context generated",
            sources_count=len(selected_sources),
            estimated_tokens=sum(s.token_count for s in selected_sources)
        )

        return formatted_context

    async def _detect_intent(self, query: str) -> QueryIntent:
        """
        Detect query intent from keywords.

        Args:
            query: User query

        Returns:
            Detected intent
        """
        query_lower = query.lower()

        # Count keyword matches for each intent
        intent_scores = {}

        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score

        # Return highest scoring intent or GENERAL
        if intent_scores:
            detected_intent = max(intent_scores, key=intent_scores.get)
            logger.debug(
                "Intent detected",
                intent=detected_intent,
                score=intent_scores[detected_intent]
            )
            return detected_intent

        return QueryIntent.GENERAL

    async def _gather_context_sources(
        self,
        query: str,
        intent: QueryIntent,
        project_id: Optional[str],
        session_id: Optional[str],
        include_code: bool,
        include_session_history: bool
    ) -> List[ContextSource]:
        """
        Gather relevant context from multiple sources.

        Args:
            query: User query
            intent: Detected intent
            project_id: Project ID
            session_id: Session ID
            include_code: Include code sources
            include_session_history: Include session history

        Returns:
            List of context sources
        """
        sources = []

        # In production, would query actual services
        # For now, create mock sources

        # Mock memory search results
        if project_id:
            sources.append(ContextSource(
                source_type="memory",
                content=f"Relevant memory about {query[:50]}...",
                relevance_score=0.85,
                token_count=200,
                metadata={"project_id": project_id}
            ))

        # Mock code artifacts for code-related intents
        if include_code and intent in [
            QueryIntent.CODE_GENERATION,
            QueryIntent.CODE_DEBUGGING,
            QueryIntent.CODE_REVIEW
        ]:
            sources.append(ContextSource(
                source_type="code",
                content="def related_function():\\n    # Relevant code snippet\\n    pass",
                relevance_score=0.90,
                token_count=150,
                metadata={"language": "python", "file_path": "utils.py"}
            ))

        # Mock session history
        if include_session_history and session_id:
            sources.append(ContextSource(
                source_type="session",
                content="Previous conversation summary...",
                relevance_score=0.75,
                token_count=300,
                metadata={"session_id": session_id}
            ))

        # Mock project context
        if project_id:
            sources.append(ContextSource(
                source_type="project",
                content=f"Project overview for {project_id}",
                relevance_score=0.60,
                token_count=100,
                metadata={"project_id": project_id}
            ))

        return sources

    async def _select_within_budget(
        self,
        sources: List[ContextSource],
        max_tokens: int
    ) -> List[ContextSource]:
        """
        Select sources that fit within token budget.

        Args:
            sources: Ranked context sources
            max_tokens: Maximum tokens

        Returns:
            Selected sources
        """
        selected = []
        current_tokens = 0

        for source in sources:
            if current_tokens + source.token_count <= max_tokens:
                selected.append(source)
                current_tokens += source.token_count
            else:
                # Check if we can fit a truncated version
                remaining = max_tokens - current_tokens
                if remaining > 100:  # Minimum useful content
                    # Would truncate content here in production
                    logger.debug(
                        "Truncating source to fit budget",
                        source_type=source.source_type,
                        original_tokens=source.token_count,
                        remaining_tokens=remaining
                    )
                break

        return selected

    async def _format_context(
        self,
        sources: List[ContextSource],
        intent: QueryIntent
    ) -> str:
        """
        Format context sources using Claude-friendly XML.

        Args:
            sources: Selected context sources
            intent: Query intent

        Returns:
            Formatted context
        """
        if not sources:
            return ""

        context_parts = [
            "<context>",
            f"  <query_intent>{intent}</query_intent>",
            ""
        ]

        # Group sources by type
        sources_by_type: Dict[str, List[ContextSource]] = {}
        for source in sources:
            if source.source_type not in sources_by_type:
                sources_by_type[source.source_type] = []
            sources_by_type[source.source_type].append(source)

        # Format each source type
        for source_type, type_sources in sources_by_type.items():
            context_parts.append(f"  <{source_type}_context>")

            for source in type_sources:
                context_parts.append(f'    <item relevance="{source.relevance_score:.2f}">')

                # Add metadata
                if source.metadata:
                    context_parts.append("      <metadata>")
                    for key, value in source.metadata.items():
                        context_parts.append(f"        <{key}>{value}</{key}>")
                    context_parts.append("      </metadata>")

                # Add content
                context_parts.append("      <content>")
                # Indent content lines
                for line in source.content.split("\\n"):
                    context_parts.append(f"        {line}")
                context_parts.append("      </content>")

                context_parts.append("    </item>")

            context_parts.append(f"  </{source_type}_context>")
            context_parts.append("")

        context_parts.append("</context>")

        return "\\n".join(context_parts)

    async def format_for_continuation(
        self,
        session_id: str,
        max_tokens: int = 3000
    ) -> str:
        """
        Format context specifically for session continuation.

        Args:
            session_id: Session ID
            max_tokens: Maximum tokens

        Returns:
            Continuation context
        """
        # In production, would fetch actual session checkpoint
        context = f"""<session_continuation>
  <session_id>{session_id}</session_id>
  <last_interaction>
    <summary>Working on implementing user authentication</summary>
    <objective>Add password reset functionality</objective>
    <progress>
      - Completed JWT-based authentication
      - Implemented bcrypt password hashing
      - Created login/logout endpoints
    </progress>
    <next_steps>
      1. Create password reset token generation
      2. Implement email sending service
      3. Add password reset form to frontend
    </next_steps>
  </last_interaction>
</session_continuation>"""

        return context

    async def optimize_for_claude(
        self,
        context: str,
        target_model: str = "claude-3-5-sonnet"
    ) -> Dict[str, Any]:
        """
        Optimize context for specific Claude model.

        Args:
            context: Raw context
            target_model: Target Claude model

        Returns:
            Optimized context with metadata
        """
        # Model-specific optimizations
        model_preferences = {
            "claude-3-5-sonnet": {
                "prefers_xml": True,
                "max_context": 200000,
                "optimal_context": 20000
            },
            "claude-3-opus": {
                "prefers_xml": True,
                "max_context": 200000,
                "optimal_context": 30000
            },
            "claude-3-haiku": {
                "prefers_xml": True,
                "max_context": 200000,
                "optimal_context": 10000
            }
        }

        preferences = model_preferences.get(
            target_model,
            model_preferences["claude-3-5-sonnet"]
        )

        return {
            "optimized_context": context,  # Would apply optimizations in production
            "model": target_model,
            "estimated_tokens": len(context) // 4,  # Rough estimate
            "preferences": preferences,
            "uses_xml_formatting": preferences["prefers_xml"]
        }


# Global instance
context_injector = IntelligentContextInjection()
