from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from uuid import UUID

from app.api.dependencies import get_current_active_user, get_db
from app.schemas.memory import (
    MemoryCreate,
    MemoryResponse,
    MemorySearch,
    MemoryUpdate,
    DocumentUploadResponse
)
from app.models.user import User
from app.services.memory_service import MemoryService
from app.services.document_service import document_processor, DocumentProcessingError
from app.workers.tasks import generate_embedding_task, process_document_task

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


@router.post("/ingest/document", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    *,
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
) -> DocumentUploadResponse:
    """
    Ingest a document file (PDF, DOCX, PPTX, XLSX, etc.) into the system.

    The document is processed asynchronously:
    1. File is validated and uploaded
    2. Document processing task is queued (extraction, OCR, chunking)
    3. Individual chunks are ingested as memories
    4. Embeddings are generated for each chunk

    Supported formats:
    - PDF (.pdf)
    - Word (.docx)
    - PowerPoint (.pptx)
    - Excel (.xlsx)
    - HTML (.html)
    - Markdown (.md)
    - Plain text (.txt)
    - Images (PNG, JPEG, TIFF) with OCR

    MCP Integration: This endpoint can be called via MCP server tools
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )

        # Read file content
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )

        # Validate file size (50MB max)
        max_size = 50 * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {max_size} bytes"
            )

        logger.info(
            "Document upload received",
            filename=file.filename,
            content_type=file.content_type,
            size=len(file_content),
            user_id=str(current_user.id)
        )

        # Queue document processing task
        task = process_document_task.delay(
            file_content=file_content,
            filename=file.filename,
            user_id=str(current_user.id)
        )

        return DocumentUploadResponse(
            task_id=task.id,
            filename=file.filename,
            file_size=len(file_content),
            status="processing",
            message="Document queued for processing. Use task_id to check status."
        )

    except DocumentProcessingError as e:
        logger.error("Document processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Document upload failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document upload"
        )


@router.get("/document/status/{task_id}")
async def get_document_processing_status(
    *,
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Check the status of a document processing task.

    MCP Integration: Poll this endpoint to check document processing status
    """
    from app.workers.celery_app import celery_app

    task = celery_app.AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": task.state,
    }

    if task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
    elif task.state == "PROGRESS":
        response["progress"] = task.info

    return response