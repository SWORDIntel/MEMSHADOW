"""
OpenAI-compatible API endpoints for MEMSHADOW.

This module provides OpenAI spec-compliant endpoints for:
- Chat completions (with memory augmentation)
- Model listing
- Embeddings generation

Use these endpoints to connect local AI tools (ollama, llama.cpp, etc.) to MEMSHADOW.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from datetime import datetime
import time
import uuid
import structlog
import httpx

from app.api.dependencies import get_current_active_user, get_db
from app.models.user import User
from app.services.memory_service import MemoryService
from app.services.embedding_service import EmbeddingService
from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = structlog.get_logger()


# =============================================================================
# OpenAI-compatible Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "memshadow"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # MEMSHADOW extensions
    use_memory: Optional[bool] = True
    memory_limit: Optional[int] = 5
    ingest_response: Optional[bool] = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "memshadow-embed"
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "memshadow"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Configuration
# =============================================================================

# Local AI backend configuration (ollama, llama.cpp, vllm, etc.)
LOCAL_AI_URL = getattr(settings, 'LOCAL_AI_URL', 'http://localhost:11434/v1')
LOCAL_AI_MODEL = getattr(settings, 'LOCAL_AI_MODEL', 'llama3.2')


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models (OpenAI spec: GET /v1/models)
    """
    created = int(datetime.utcnow().timestamp())

    models = [
        ModelInfo(id="memshadow", created=created, owned_by="memshadow"),
        ModelInfo(id="memshadow-embed", created=created, owned_by="memshadow"),
        ModelInfo(id="memshadow-memory", created=created, owned_by="memshadow"),
    ]

    # Try to fetch models from local AI backend
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LOCAL_AI_URL}/models")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", data.get("models", [])):
                    model_id = m.get("id") or m.get("name")
                    if model_id:
                        models.append(ModelInfo(
                            id=f"local:{model_id}",
                            created=m.get("created", created),
                            owned_by="local"
                        ))
    except Exception:
        pass  # Local AI not available

    return ModelsResponse(data=models)


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    Get model info (OpenAI spec: GET /v1/models/{model_id})
    """
    return ModelInfo(
        id=model_id,
        created=int(datetime.utcnow().timestamp()),
        owned_by="memshadow" if model_id.startswith("memshadow") else "local"
    )


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Chat completions with memory augmentation (OpenAI spec: POST /v1/chat/completions)

    MEMSHADOW extensions:
    - use_memory: Retrieve relevant memories to augment context
    - memory_limit: Max memories to inject
    - ingest_response: Store the conversation in memory
    """
    start_time = time.time()
    memory_service = MemoryService(db)

    # Extract user message for memory lookup
    user_messages = [m for m in request.messages if m.role == "user"]
    last_user_msg = user_messages[-1].content if user_messages else ""

    augmented_messages = list(request.messages)
    memories_used = []

    # Retrieve relevant memories
    if request.use_memory and last_user_msg:
        try:
            memories = await memory_service.search_memories(
                user_id=current_user.id,
                query=last_user_msg,
                limit=request.memory_limit
            )

            if memories:
                memory_context = "\n\n".join([
                    f"[Memory {i+1}]: {m.content}"
                    for i, m in enumerate(memories)
                ])

                # Inject memory context as system message
                memory_system = ChatMessage(
                    role="system",
                    content=f"Relevant context from memory:\n{memory_context}"
                )
                augmented_messages.insert(0, memory_system)
                memories_used = [str(m.id) for m in memories]

        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

    # Forward to local AI backend
    response_content = ""
    model_used = request.model

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Determine backend URL
            if request.model.startswith("local:"):
                model_used = request.model.replace("local:", "")
                backend_url = f"{LOCAL_AI_URL}/chat/completions"
            else:
                model_used = LOCAL_AI_MODEL
                backend_url = f"{LOCAL_AI_URL}/chat/completions"

            payload = {
                "model": model_used,
                "messages": [m.model_dump() for m in augmented_messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False
            }

            resp = await client.post(backend_url, json=payload)

            if resp.status_code == 200:
                data = resp.json()
                response_content = data["choices"][0]["message"]["content"]
            else:
                # Fallback response if local AI unavailable
                response_content = (
                    f"[MEMSHADOW] Local AI backend unavailable. "
                    f"Retrieved {len(memories_used)} relevant memories for your query."
                )

    except Exception as e:
        logger.warning(f"Local AI call failed: {e}")
        response_content = (
            f"[MEMSHADOW] Could not reach local AI at {LOCAL_AI_URL}. "
            f"Configure LOCAL_AI_URL in settings. "
            f"Retrieved {len(memories_used)} memories."
        )

    # Ingest conversation into memory
    if request.ingest_response and last_user_msg:
        try:
            # Store user message
            await memory_service.create_memory(
                user_id=current_user.id,
                content=f"User: {last_user_msg}",
                extra_data={"type": "conversation", "role": "user"}
            )
            # Store assistant response
            if response_content and not response_content.startswith("[MEMSHADOW]"):
                await memory_service.create_memory(
                    user_id=current_user.id,
                    content=f"Assistant: {response_content}",
                    extra_data={"type": "conversation", "role": "assistant"}
                )
        except Exception as e:
            logger.warning(f"Memory ingestion failed: {e}")

    # Calculate token usage (approximate)
    prompt_tokens = sum(len(m.content.split()) * 1.3 for m in augmented_messages)
    completion_tokens = len(response_content.split()) * 1.3

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_used,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens)
        )
    )


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate embeddings (OpenAI spec: POST /v1/embeddings)
    """
    embedding_service = EmbeddingService()

    inputs = request.input if isinstance(request.input, list) else [request.input]

    embeddings = await embedding_service.generate_batch_embeddings(inputs)

    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]

    total_tokens = sum(len(text.split()) for text in inputs)

    return EmbeddingResponse(
        data=data,
        model=settings.EMBEDDING_MODEL,
        usage=Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens
        )
    )


@router.get("/capabilities")
async def get_capabilities():
    """
    MEMSHADOW-specific endpoint: List system capabilities
    """
    return {
        "name": "MEMSHADOW",
        "version": settings.VERSION,
        "capabilities": {
            "memory": {
                "semantic_search": True,
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "embedding_model": settings.EMBEDDING_MODEL,
                "corpus_import": ["chatgpt", "claude", "json", "jsonl", "csv", "markdown", "text", "zip"],
                "autoscan_enabled": True,
                "autoscan_interval": "hourly"
            },
            "api": {
                "openai_compatible": True,
                "endpoints": [
                    "/v1/models",
                    "/v1/chat/completions",
                    "/v1/embeddings"
                ]
            },
            "local_ai": {
                "url": LOCAL_AI_URL,
                "default_model": LOCAL_AI_MODEL
            }
        }
    }
