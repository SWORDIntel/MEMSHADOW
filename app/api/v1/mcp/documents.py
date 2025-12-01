"""
MCP Endpoints for Document Processing

MCP-compatible endpoints for document upload, processing status, and retrieval.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from app.api.dependencies import get_current_active_user, get_db
from app.models.user import User
from app.services.document_service import document_processor
from app.workers.tasks import process_document_task

router = APIRouter()


@router.post("/upload")
async def mcp_upload_document(
    *,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    MCP Tool: Upload and process a document

    This endpoint is designed for MCP servers to upload documents for processing.
    Returns a task ID that can be used to check processing status.

    Args:
        file: Document file to upload

    Returns:
        Task information including task_id for status checking
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    file_content = await file.read()

    task = process_document_task.delay(
        file_content=file_content,
        filename=file.filename,
        user_id=str(current_user.id)
    )

    return {
        "tool": "upload_document",
        "status": "processing",
        "task_id": task.id,
        "filename": file.filename,
        "file_size": len(file_content),
        "message": "Document queued for processing. Use check_document_status to monitor progress."
    }


@router.get("/status/{task_id}")
async def mcp_check_document_status(
    *,
    task_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    MCP Tool: Check document processing status

    Args:
        task_id: Task ID from upload_document

    Returns:
        Task status and results if complete
    """
    from app.workers.celery_app import celery_app

    task = celery_app.AsyncResult(task_id)

    response = {
        "tool": "check_document_status",
        "task_id": task_id,
        "status": task.state
    }

    if task.state == "SUCCESS":
        response["result"] = task.result
        response["message"] = "Document processed successfully"
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
        response["message"] = "Document processing failed"
    elif task.state == "PROGRESS":
        response["progress"] = task.info
        response["message"] = "Document processing in progress"
    else:
        response["message"] = f"Task status: {task.state}"

    return response
