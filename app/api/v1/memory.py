from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from uuid import UUID

from app.api.dependencies import get_current_active_user, get_db
from app.services.corpus_importer import import_corpus_from_upload, ImportResult, CorpusImporter
from app.schemas.memory import (
    MemoryCreate,
    MemoryResponse,
    MemorySearch,
    MemoryUpdate
)
from app.models.user import User
from app.services.memory_service import MemoryService
from app.workers.tasks import generate_embedding_task

router = APIRouter()
logger = structlog.get_logger()

@router.post("/ingest", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def ingest_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_in: MemoryCreate,
    current_user: User = Depends(get_current_active_user),
) -> MemoryResponse:
    """
    Ingest a new memory into the system.
    Embedding generation is handled by a Celery worker.
    """
    memory_service = MemoryService(db)

    try:
        memory = await memory_service.create_memory(
            user_id=current_user.id,
            content=memory_in.content,
            extra_data=memory_in.extra_data
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Dispatch embedding generation to Celery worker
    generate_embedding_task.delay(memory_id=str(memory.id), content=memory_in.content)

    logger.info(
        "Memory ingested, embedding task dispatched",
        user_id=str(current_user.id),
        memory_id=str(memory.id)
    )

    return memory

@router.post("/retrieve", response_model=List[MemoryResponse])
async def retrieve_memories(
    *,
    db: AsyncSession = Depends(get_db),
    search: MemorySearch,
    current_user: User = Depends(get_current_active_user),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[MemoryResponse]:
    """
    Retrieve memories using semantic search.
    """
    memory_service = MemoryService(db)

    memories = await memory_service.search_memories(
        user_id=current_user.id,
        query=search.query,
        filters=search.filters,
        limit=limit,
        offset=offset
    )

    logger.info(
        "Memories retrieved",
        user_id=str(current_user.id),
        count=len(memories)
    )

    return memories

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_id: UUID,
    current_user: User = Depends(get_current_active_user)
) -> MemoryResponse:
    """
    Get a specific memory by ID.
    """
    memory_service = MemoryService(db)

    memory = await memory_service.get_memory(
        memory_id=memory_id,
        user_id=current_user.id
    )

    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    return memory

@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_id: UUID,
    memory_update: MemoryUpdate,
    current_user: User = Depends(get_current_active_user),
) -> MemoryResponse:
    """
    Update an existing memory.
    """
    memory_service = MemoryService(db)

    memory = await memory_service.update_memory(
        memory_id=memory_id,
        user_id=current_user.id,
        updates=memory_update.model_dump(exclude_unset=True)
    )

    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    # Re-generate embedding if content changed
    if memory_update.content:
        generate_embedding_task.delay(memory_id=str(memory.id), content=memory_update.content)
        logger.info("Re-embedding task dispatched for updated memory", memory_id=str(memory.id))

    return memory

@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_id: UUID,
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Delete a memory.
    """
    memory_service = MemoryService(db)

    success = await memory_service.delete_memory(
        memory_id=memory_id,
        user_id=current_user.id
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    logger.info("Memory deleted", user_id=str(current_user.id), memory_id=str(memory_id))
    return None


@router.post("/import", response_model=ImportResult)
async def import_corpus(
    *,
    file: UploadFile = File(...),
    format: str = Form("auto"),
    current_user: User = Depends(get_current_active_user)
) -> ImportResult:
    """
    Import a chat corpus to pre-seed memory system.

    Supported formats: chatgpt, claude, json, jsonl, csv, markdown, text, auto
    """
    content = await file.read()

    result = await import_corpus_from_upload(
        user_id=str(current_user.id),
        content=content,
        filename=file.filename or "upload.txt",
        format=format
    )

    logger.info(
        "Corpus import completed",
        user_id=str(current_user.id),
        total=result.total_messages,
        imported=result.imported_messages
    )

    return result


@router.post("/import/directory", response_model=ImportResult)
async def import_corpus_directory(
    *,
    directory: str = Form(...),
    recursive: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
) -> ImportResult:
    """
    Import all corpus files from a directory (for manually uploaded .zip extracts).

    Place your corpus files in the directory, then trigger this endpoint.
    Supports: .json, .jsonl, .csv, .md, .txt, .zip
    """
    importer = CorpusImporter(user_id=str(current_user.id))
    result = await importer.import_directory(directory, recursive=recursive)

    logger.info(
        "Directory import completed",
        user_id=str(current_user.id),
        directory=directory,
        total=result.total_messages,
        imported=result.imported_messages
    )

    return result