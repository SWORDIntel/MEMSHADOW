"""
Anthropic Claude Wrapper with MEMSHADOW Memory
Drop-in replacement for anthropic.Anthropic with automatic memory persistence
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Anthropic:
    """
    Anthropic-compatible wrapper that automatically stores and retrieves memories.

    This class wraps the Anthropic Messages API and automatically:
    1. Stores conversation messages in MEMSHADOW
    2. Retrieves relevant memories to inject as context
    3. Provides seamless memory persistence across sessions

    Example:
        >>> from memshadow_sdk import Anthropic
        >>> client = Anthropic(
        ...     api_key="your-anthropic-key",
        ...     memshadow_url="http://localhost:8000/api/v1",
        ...     memshadow_api_key="your-memshadow-key",
        ...     user_id="user_123"
        ... )
        >>> response = client.chat([
        ...     {"role": "user", "content": "My favorite color is blue"}
        ... ])
        >>> # Later session...
        >>> response = client.chat([
        ...     {"role": "user", "content": "What's my favorite color?"}
        ... ])
        >>> # Returns: "Your favorite color is blue"
    """

    def __init__(
        self,
        api_key: str,
        memshadow_url: str,
        memshadow_api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        user_id: Optional[str] = None,
        auto_inject_context: bool = True,
        context_limit: int = 5
    ):
        """
        Initialize Anthropic wrapper with MEMSHADOW integration.

        Args:
            api_key: Anthropic API key
            memshadow_url: MEMSHADOW API URL
            memshadow_api_key: MEMSHADOW authentication token
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
            user_id: Optional user ID for multi-user scenarios
            auto_inject_context: Whether to automatically inject memory context
            context_limit: Maximum memories to inject as context
        """
        try:
            import anthropic
            self.anthropic = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        from memshadow_sdk.client import MemshadowClient

        self.memshadow = MemshadowClient(
            api_url=memshadow_url,
            api_key=memshadow_api_key,
            user_id=user_id
        )

        self.model = model
        self.user_id = user_id
        self.auto_inject_context = auto_inject_context
        self.context_limit = context_limit

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a messages API request with automatic memory handling.

        This method:
        1. Retrieves relevant memories based on the user's message
        2. Injects memory context into the system prompt
        3. Calls Anthropic Messages API
        4. Stores the conversation in MEMSHADOW

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system: System prompt
            **kwargs: Additional arguments passed to Anthropic API

        Returns:
            Anthropic API response
        """
        model = model or self.model

        # Extract user message for context retrieval
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages and self.auto_inject_context:
            last_user_msg = user_messages[-1]["content"]

            # Retrieve relevant memories
            try:
                memories = self.memshadow.retrieve(
                    query=last_user_msg,
                    limit=self.context_limit
                )

                # Inject memory context into system prompt
                if memories:
                    memory_context = self._build_memory_context(memories)

                    if system:
                        system = f"{system}\n\n{memory_context}"
                    else:
                        system = memory_context

                    logger.debug(f"Injected {len(memories)} memories as context")

            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        # Call Anthropic API
        try:
            response = self.anthropic.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                **kwargs
            )

            # Store conversation in MEMSHADOW
            self._store_conversation(messages, response)

            return response

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build memory context for system prompt."""
        context_lines = [
            "Here is relevant context from previous conversations:",
            ""
        ]

        for i, memory in enumerate(memories, 1):
            context_lines.append(f"{i}. {memory['content']}")

        return "\n".join(context_lines)

    def _store_conversation(
        self,
        messages: List[Dict[str, str]],
        response: Any
    ):
        """Store conversation messages in MEMSHADOW."""
        try:
            # Store user messages
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg["content"]
                    if isinstance(content, list):
                        # Handle multimodal content
                        text_parts = [
                            block.get("text", "")
                            for block in content
                            if block.get("type") == "text"
                        ]
                        content = " ".join(text_parts)

                    self.memshadow.ingest(
                        content=content,
                        extra_data={
                            "role": "user",
                            "model": self.model,
                            "source": "anthropic_wrapper"
                        }
                    )

            # Store assistant response
            if hasattr(response, "content") and len(response.content) > 0:
                assistant_msg = response.content[0].text
                self.memshadow.ingest(
                    content=assistant_msg,
                    extra_data={
                        "role": "assistant",
                        "model": self.model,
                        "source": "anthropic_wrapper",
                        "stop_reason": response.stop_reason
                    }
                )

            logger.debug("Conversation stored in MEMSHADOW")

        except Exception as e:
            logger.warning(f"Failed to store conversation: {e}")

    def ingest_manual(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Manually ingest content into MEMSHADOW.

        Useful for storing important facts outside of conversation flow.

        Args:
            content: Content to store
            metadata: Optional metadata
        """
        return self.memshadow.ingest(content, extra_data=metadata)

    def retrieve_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Manually retrieve memories.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of memory objects
        """
        return self.memshadow.retrieve(query, limit=limit)
