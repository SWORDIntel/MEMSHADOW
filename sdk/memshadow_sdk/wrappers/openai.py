"""
OpenAI Wrapper with MEMSHADOW Memory
Drop-in replacement for openai.ChatCompletion with automatic memory persistence
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OpenAI:
    """
    OpenAI-compatible wrapper that automatically stores and retrieves memories.

    This class wraps the OpenAI ChatCompletion API and automatically:
    1. Stores conversation messages in MEMSHADOW
    2. Retrieves relevant memories to inject as context
    3. Provides seamless memory persistence across sessions

    Example:
        >>> from memshadow_sdk import OpenAI
        >>> client = OpenAI(
        ...     api_key="your-openai-key",
        ...     memshadow_url="http://localhost:8000/api/v1",
        ...     memshadow_api_key="your-memshadow-key",
        ...     user_id="user_123"
        ... )
        >>> response = client.chat([
        ...     {"role": "user", "content": "My name is Alice"}
        ... ])
        >>> # Later session...
        >>> response = client.chat([
        ...     {"role": "user", "content": "What's my name?"}
        ... ])
        >>> # Returns: "Your name is Alice"
    """

    def __init__(
        self,
        api_key: str,
        memshadow_url: str,
        memshadow_api_key: str,
        model: str = "gpt-4",
        user_id: Optional[str] = None,
        auto_inject_context: bool = True,
        context_limit: int = 5
    ):
        """
        Initialize OpenAI wrapper with MEMSHADOW integration.

        Args:
            api_key: OpenAI API key
            memshadow_url: MEMSHADOW API URL
            memshadow_api_key: MEMSHADOW authentication token
            model: OpenAI model to use (default: gpt-4)
            user_id: Optional user ID for multi-user scenarios
            auto_inject_context: Whether to automatically inject memory context
            context_limit: Maximum memories to inject as context
        """
        try:
            import openai
            self.openai = openai
            self.openai.api_key = api_key
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
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
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request with automatic memory handling.

        This method:
        1. Retrieves relevant memories based on the user's message
        2. Injects memory context into the conversation
        3. Calls OpenAI API
        4. Stores the conversation in MEMSHADOW

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            OpenAI API response
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

                # Inject memory context
                if memories:
                    context_msg = self._build_context_message(memories)
                    messages = [context_msg] + messages

                    logger.debug(f"Injected {len(memories)} memories as context")

            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        # Call OpenAI API
        try:
            response = self.openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Store conversation in MEMSHADOW
            self._store_conversation(messages, response)

            return response

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _build_context_message(self, memories: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build a system message with memory context."""
        context_lines = [
            "Here is relevant context from previous conversations:",
            ""
        ]

        for i, memory in enumerate(memories, 1):
            context_lines.append(f"{i}. {memory['content']}")

        context = "\n".join(context_lines)

        return {
            "role": "system",
            "content": context
        }

    def _store_conversation(
        self,
        messages: List[Dict[str, str]],
        response: Dict[str, Any]
    ):
        """Store conversation messages in MEMSHADOW."""
        try:
            # Store user messages
            for msg in messages:
                if msg.get("role") == "user":
                    self.memshadow.ingest(
                        content=msg["content"],
                        extra_data={
                            "role": "user",
                            "model": self.model,
                            "source": "openai_wrapper"
                        }
                    )

            # Store assistant response
            if "choices" in response and len(response["choices"]) > 0:
                assistant_msg = response["choices"][0]["message"]["content"]
                self.memshadow.ingest(
                    content=assistant_msg,
                    extra_data={
                        "role": "assistant",
                        "model": self.model,
                        "source": "openai_wrapper"
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
