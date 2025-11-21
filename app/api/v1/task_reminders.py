"""
Task Reminder API Endpoints
RESTful API for managing task reminders and notifications
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from uuid import UUID

from app.api.dependencies import get_current_active_user, get_db
from app.schemas.task_reminder import (
    TaskReminderCreate,
    TaskReminderResponse,
    TaskReminderUpdate,
    TaskReminderStats
)
from app.models.user import User
from app.models.task_reminder import ReminderStatus, ReminderPriority
from app.services.task_reminder_service import TaskReminderService

router = APIRouter()
logger = structlog.get_logger()


@router.post("/", response_model=TaskReminderResponse, status_code=status.HTTP_201_CREATED)
async def create_task_reminder(
    *,
    db: AsyncSession = Depends(get_db),
    reminder_in: TaskReminderCreate,
    current_user: User = Depends(get_current_active_user),
) -> TaskReminderResponse:
    """
    Create a new task reminder.

    The reminder will be sent to the user at the specified reminder_date.
    """
    service = TaskReminderService(db)

    reminder = await service.create_reminder(
        user_id=current_user.id,
        title=reminder_in.title,
        description=reminder_in.description,
        reminder_date=reminder_in.reminder_date,
        due_date=reminder_in.due_date,
        priority=reminder_in.priority,
        associated_memory_id=reminder_in.associated_memory_id,
        extra_data=reminder_in.extra_data
    )

    logger.info(
        "Task reminder created",
        user_id=str(current_user.id),
        reminder_id=str(reminder.id)
    )

    return TaskReminderResponse.from_orm_with_computed(reminder)


@router.get("/", response_model=List[TaskReminderResponse])
async def list_task_reminders(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    status_filter: Optional[ReminderStatus] = Query(None, alias="status"),
    priority_filter: Optional[ReminderPriority] = Query(None, alias="priority"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[TaskReminderResponse]:
    """
    List task reminders for the current user.

    Supports filtering by status and priority.
    """
    service = TaskReminderService(db)

    reminders = await service.list_reminders(
        user_id=current_user.id,
        status=status_filter,
        priority=priority_filter,
        limit=limit,
        offset=offset
    )

    logger.info(
        "Task reminders listed",
        user_id=str(current_user.id),
        count=len(reminders)
    )

    return [TaskReminderResponse.from_orm_with_computed(r) for r in reminders]


@router.get("/upcoming", response_model=List[TaskReminderResponse])
async def get_upcoming_reminders(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    hours: int = Query(24, ge=1, le=168, description="Hours ahead to look")
) -> List[TaskReminderResponse]:
    """
    Get upcoming reminders for the next N hours.
    """
    service = TaskReminderService(db)

    reminders = await service.get_upcoming_reminders(
        user_id=current_user.id,
        hours_ahead=hours
    )

    return [TaskReminderResponse.from_orm_with_computed(r) for r in reminders]


@router.get("/overdue", response_model=List[TaskReminderResponse])
async def get_overdue_reminders(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[TaskReminderResponse]:
    """
    Get all overdue tasks for the current user.
    """
    service = TaskReminderService(db)

    reminders = await service.get_overdue_reminders(user_id=current_user.id)

    return [TaskReminderResponse.from_orm_with_computed(r) for r in reminders]


@router.get("/stats", response_model=TaskReminderStats)
async def get_reminder_stats(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> TaskReminderStats:
    """
    Get statistics about task reminders for the current user.
    """
    service = TaskReminderService(db)

    stats = await service.get_reminder_stats(user_id=current_user.id)

    return TaskReminderStats(**stats)


@router.get("/{reminder_id}", response_model=TaskReminderResponse)
async def get_task_reminder(
    *,
    db: AsyncSession = Depends(get_db),
    reminder_id: UUID,
    current_user: User = Depends(get_current_active_user)
) -> TaskReminderResponse:
    """
    Get a specific task reminder by ID.
    """
    service = TaskReminderService(db)

    reminder = await service.get_reminder(
        reminder_id=reminder_id,
        user_id=current_user.id
    )

    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task reminder not found"
        )

    return TaskReminderResponse.from_orm_with_computed(reminder)


@router.patch("/{reminder_id}", response_model=TaskReminderResponse)
async def update_task_reminder(
    *,
    db: AsyncSession = Depends(get_db),
    reminder_id: UUID,
    reminder_update: TaskReminderUpdate,
    current_user: User = Depends(get_current_active_user)
) -> TaskReminderResponse:
    """
    Update a task reminder.
    """
    service = TaskReminderService(db)

    # Convert to dict and remove None values
    updates = reminder_update.dict(exclude_unset=True)

    reminder = await service.update_reminder(
        reminder_id=reminder_id,
        user_id=current_user.id,
        updates=updates
    )

    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task reminder not found"
        )

    logger.info(
        "Task reminder updated",
        user_id=str(current_user.id),
        reminder_id=str(reminder_id)
    )

    return TaskReminderResponse.from_orm_with_computed(reminder)


@router.post("/{reminder_id}/complete", response_model=TaskReminderResponse)
async def complete_task_reminder(
    *,
    db: AsyncSession = Depends(get_db),
    reminder_id: UUID,
    current_user: User = Depends(get_current_active_user)
) -> TaskReminderResponse:
    """
    Mark a task reminder as completed.
    """
    service = TaskReminderService(db)

    reminder = await service.mark_as_completed(
        reminder_id=reminder_id,
        user_id=current_user.id
    )

    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task reminder not found"
        )

    logger.info(
        "Task reminder completed",
        user_id=str(current_user.id),
        reminder_id=str(reminder_id)
    )

    return TaskReminderResponse.from_orm_with_computed(reminder)


@router.post("/{reminder_id}/cancel", response_model=TaskReminderResponse)
async def cancel_task_reminder(
    *,
    db: AsyncSession = Depends(get_db),
    reminder_id: UUID,
    current_user: User = Depends(get_current_active_user)
) -> TaskReminderResponse:
    """
    Cancel a task reminder.
    """
    service = TaskReminderService(db)

    reminder = await service.cancel_reminder(
        reminder_id=reminder_id,
        user_id=current_user.id
    )

    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task reminder not found"
        )

    logger.info(
        "Task reminder cancelled",
        user_id=str(current_user.id),
        reminder_id=str(reminder_id)
    )

    return TaskReminderResponse.from_orm_with_computed(reminder)


@router.delete("/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task_reminder(
    *,
    db: AsyncSession = Depends(get_db),
    reminder_id: UUID,
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Delete a task reminder.
    """
    service = TaskReminderService(db)

    deleted = await service.delete_reminder(
        reminder_id=reminder_id,
        user_id=current_user.id
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task reminder not found"
        )

    logger.info(
        "Task reminder deleted",
        user_id=str(current_user.id),
        reminder_id=str(reminder_id)
    )
