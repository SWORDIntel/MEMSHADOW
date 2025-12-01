"""
MEMSHADOW Client SDK

Simple client for integrating custom AI systems with MEMSHADOW memory.

Usage:
    from memshadow_client import MemshadowClient

    memory = MemshadowClient("http://localhost:8000", api_key="your-key")

    # Store memories
    memory.store("User prefers dark mode")

    # Retrieve relevant context
    context = memory.recall("What are the user's preferences?")

    # Use in your AI prompt
    prompt = f"Context: {context}\n\nUser: {user_input}"
"""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class Memory:
    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: str


class MemshadowClient:
    """Client for custom AI integration with MEMSHADOW"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def store(
        self,
        content: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store a memory. Returns memory ID.

        Args:
            content: The memory content
            tags: Optional tags for categorization
            metadata: Optional additional metadata
        """
        payload = {
            "content": content,
            "extra_data": metadata or {}
        }
        if tags:
            payload["extra_data"]["tags"] = tags

        resp = self._session.post(
            f"{self.base_url}/api/v1/memory/ingest",
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json().get("id", "")

    def recall(
        self,
        query: str,
        limit: int = 5,
        as_text: bool = True
    ) -> str | List[Memory]:
        """
        Recall memories relevant to query.

        Args:
            query: Search query
            limit: Max results
            as_text: Return formatted text (True) or Memory objects (False)
        """
        resp = self._session.post(
            f"{self.base_url}/api/v1/memory/retrieve",
            json={"query": query},
            params={"limit": limit},
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        memories = [
            Memory(
                id=m.get("id", ""),
                content=m.get("content", ""),
                metadata=m.get("extra_data", {}),
                created_at=m.get("created_at", "")
            )
            for m in data
        ]

        if as_text:
            if not memories:
                return ""
            return "\n---\n".join(m.content for m in memories)

        return memories

    def chat(
        self,
        messages: List[Dict[str, str]],
        use_memory: bool = True,
        memory_limit: int = 5,
        ingest_response: bool = True
    ) -> str:
        """
        Chat via OpenAI-compatible endpoint with memory augmentation.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            use_memory: Inject relevant memories into context
            memory_limit: Max memories to inject
            ingest_response: Store conversation in memory
        """
        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "memshadow",
                "messages": messages,
                "use_memory": use_memory,
                "memory_limit": memory_limit,
                "ingest_response": ingest_response
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def embed(self, texts: str | List[str]) -> List[List[float]]:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]

        resp = self._session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts},
            timeout=self.timeout
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        resp = self._session.delete(
            f"{self.base_url}/api/v1/memory/{memory_id}",
            timeout=self.timeout
        )
        return resp.status_code == 204

    def capabilities(self) -> Dict[str, Any]:
        """Get MEMSHADOW capabilities"""
        resp = self._session.get(
            f"{self.base_url}/v1/capabilities",
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()


# =============================================================================
# Helper for wrapping any LLM with MEMSHADOW memory
# =============================================================================

class MemoryAugmentedLLM:
    """
    Wrap any LLM function with MEMSHADOW memory.

    Usage:
        def my_llm(prompt: str) -> str:
            # Your custom LLM call (ollama, llama.cpp, transformers, etc.)
            return response

        augmented = MemoryAugmentedLLM(my_llm, "http://localhost:8000", "api-key")
        response = augmented.chat("What did we discuss yesterday?")
    """

    def __init__(
        self,
        llm_fn,
        memshadow_url: str = "http://localhost:8000",
        api_key: str = "",
        auto_store: bool = True,
        memory_limit: int = 5
    ):
        """
        Args:
            llm_fn: Function that takes prompt string and returns response string
            memshadow_url: MEMSHADOW API URL
            api_key: API key
            auto_store: Automatically store conversations in memory
            memory_limit: Max memories to inject per query
        """
        self.llm = llm_fn
        self.memory = MemshadowClient(memshadow_url, api_key)
        self.auto_store = auto_store
        self.memory_limit = memory_limit

    def chat(self, user_input: str, system_prompt: str = "") -> str:
        """
        Chat with memory-augmented context.
        """
        # Recall relevant memories
        context = self.memory.recall(user_input, limit=self.memory_limit)

        # Build prompt with context
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if context:
            parts.append(f"Relevant context from memory:\n{context}")
        parts.append(f"User: {user_input}")

        prompt = "\n\n".join(parts)

        # Call LLM
        response = self.llm(prompt)

        # Store in memory
        if self.auto_store:
            self.memory.store(
                f"User: {user_input}\nAssistant: {response}",
                tags=["conversation"]
            )

        return response


# =============================================================================
# Example integrations
# =============================================================================

def example_ollama_integration():
    """Example: Integrate with Ollama"""
    import requests

    def ollama_llm(prompt: str) -> str:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return resp.json()["response"]

    ai = MemoryAugmentedLLM(ollama_llm, "http://localhost:8000", "your-api-key")
    return ai.chat("Hello, what do you remember about me?")


def example_transformers_integration():
    """Example: Integrate with HuggingFace Transformers"""
    # from transformers import pipeline
    # generator = pipeline("text-generation", model="microsoft/phi-2")

    def transformers_llm(prompt: str) -> str:
        # result = generator(prompt, max_length=500)
        # return result[0]["generated_text"]
        return "Example response"

    ai = MemoryAugmentedLLM(transformers_llm, "http://localhost:8000")
    return ai.chat("What are my preferences?")


def example_openai_integration():
    """Example: Integrate with OpenAI API"""
    # from openai import OpenAI
    # client = OpenAI()

    def openai_llm(prompt: str) -> str:
        # resp = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return resp.choices[0].message.content
        return "Example response"

    ai = MemoryAugmentedLLM(openai_llm, "http://localhost:8000")
    return ai.chat("Summarize our past conversations")


if __name__ == "__main__":
    # Quick test
    client = MemshadowClient()
    print("Capabilities:", client.capabilities())
