"""
Task Reminder Service
Manages scheduled task reminders and notifications
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import structlog

from app.models.task_reminder import TaskReminder, ReminderStatus, ReminderPriority

logger = structlog.get_logger()


class TaskReminderService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_reminder(
        self,
        user_id: UUID,
        title: str,
        reminder_date: datetime,
        description: Optional[str] = None,
        due_date: Optional[datetime] = None,
        priority: ReminderPriority = ReminderPriority.MEDIUM,
        associated_memory_id: Optional[UUID] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> TaskReminder:
        """Create a new task reminder"""
        reminder = TaskReminder(
            user_id=user_id,
            title=title,
            description=description,
            reminder_date=reminder_date,
            due_date=due_date,
            priority=priority,
            associated_memory_id=associated_memory_id,
            extra_data=extra_data or {},
            status=ReminderStatus.PENDING
        )

        self.db.add(reminder)
        await self.db.commit()
        await self.db.refresh(reminder)

        logger.info("Task reminder created",
                   reminder_id=str(reminder.id),
                   user_id=str(user_id),
                   title=title,
                   reminder_date=reminder_date.isoformat())

        return reminder

    async def get_reminder(
        self,
        reminder_id: UUID,
        user_id: UUID
    ) -> Optional[TaskReminder]:
        """Get a specific reminder"""
        stmt = select(TaskReminder).where(
            and_(
                TaskReminder.id == reminder_id,
                TaskReminder.user_id == user_id
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_reminders(
        self,
        user_id: UUID,
        status: Optional[ReminderStatus] = None,
        priority: Optional[ReminderPriority] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[TaskReminder]:
        """List reminders for a user with optional filters"""
        conditions = [TaskReminder.user_id == user_id]

        if status:
            conditions.append(TaskReminder.status == status)

        if priority:
            conditions.append(TaskReminder.priority == priority)

        stmt = select(TaskReminder).where(
            and_(*conditions)
        ).order_by(
            TaskReminder.reminder_date.asc()
        ).limit(limit).offset(offset)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_reminders(
        self,
        user_id: Optional[UUID] = None
    ) -> List[TaskReminder]:
        """Get all pending reminders that should be sent now"""
        conditions = [
            TaskReminder.status == ReminderStatus.PENDING,
            TaskReminder.reminder_date <= datetime.utcnow()
        ]

        if user_id:
            conditions.append(TaskReminder.user_id == user_id)

        stmt = select(TaskReminder).where(and_(*conditions))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_overdue_reminders(
        self,
        user_id: UUID
    ) -> List[TaskReminder]:
        """Get all overdue tasks for a user"""
        stmt = select(TaskReminder).where(
            and_(
                TaskReminder.user_id == user_id,
                TaskReminder.due_date < datetime.utcnow(),
                TaskReminder.status.in_([ReminderStatus.PENDING, ReminderStatus.REMINDED])
            )
        ).order_by(TaskReminder.due_date.asc())

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_reminder(
        self,
        reminder_id: UUID,
        user_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[TaskReminder]:
        """Update an existing reminder"""
        reminder = await self.get_reminder(reminder_id, user_id)
        if not reminder:
            return None

        # Update allowed fields
        allowed_fields = {
            'title', 'description', 'reminder_date', 'due_date',
            'priority', 'status', 'extra_data'
        }

        for field, value in updates.items():
            if field in allowed_fields and value is not None:
                setattr(reminder, field, value)

        # Update completed_at if status changed to completed
        if updates.get('status') == ReminderStatus.COMPLETED and not reminder.completed_at:
            reminder.completed_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(reminder)

        logger.info("Task reminder updated",
                   reminder_id=str(reminder_id),
                   updates=list(updates.keys()))

        return reminder

    async def mark_as_reminded(
        self,
        reminder_id: UUID
    ) -> Optional[TaskReminder]:
        """Mark a reminder as sent"""
        stmt = select(TaskReminder).where(TaskReminder.id == reminder_id)
        result = await self.db.execute(stmt)
        reminder = result.scalar_one_or_none()

        if not reminder:
            return None

        reminder.status = ReminderStatus.REMINDED
        reminder.reminded_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(reminder)

        logger.info("Reminder marked as sent",
                   reminder_id=str(reminder_id))

        return reminder

    async def mark_as_completed(
        self,
        reminder_id: UUID,
        user_id: UUID
    ) -> Optional[TaskReminder]:
        """Mark a task as completed"""
        reminder = await self.get_reminder(reminder_id, user_id)
        if not reminder:
            return None

        reminder.status = ReminderStatus.COMPLETED
        reminder.completed_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(reminder)

        logger.info("Task marked as completed",
                   reminder_id=str(reminder_id))

        return reminder

    async def cancel_reminder(
        self,
        reminder_id: UUID,
        user_id: UUID
    ) -> Optional[TaskReminder]:
        """Cancel a reminder"""
        reminder = await self.get_reminder(reminder_id, user_id)
        if not reminder:
            return None

        reminder.status = ReminderStatus.CANCELLED

        await self.db.commit()
        await self.db.refresh(reminder)

        logger.info("Reminder cancelled",
                   reminder_id=str(reminder_id))

        return reminder

    async def delete_reminder(
        self,
        reminder_id: UUID,
        user_id: UUID
    ) -> bool:
        """Delete a reminder"""
        reminder = await self.get_reminder(reminder_id, user_id)
        if not reminder:
            return False

        await self.db.delete(reminder)
        await self.db.commit()

        logger.info("Reminder deleted",
                   reminder_id=str(reminder_id))

        return True

    async def get_upcoming_reminders(
        self,
        user_id: UUID,
        hours_ahead: int = 24
    ) -> List[TaskReminder]:
        """Get reminders due in the next N hours"""
        future_time = datetime.utcnow() + timedelta(hours=hours_ahead)

        stmt = select(TaskReminder).where(
            and_(
                TaskReminder.user_id == user_id,
                TaskReminder.status == ReminderStatus.PENDING,
                TaskReminder.reminder_date <= future_time,
                TaskReminder.reminder_date >= datetime.utcnow()
            )
        ).order_by(TaskReminder.reminder_date.asc())

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_reminder_stats(
        self,
        user_id: UUID
    ) -> Dict[str, int]:
        """Get statistics about reminders for a user"""
        from sqlalchemy import func

        # Count by status
        status_counts = {}
        for status in ReminderStatus:
            stmt = select(func.count(TaskReminder.id)).where(
                and_(
                    TaskReminder.user_id == user_id,
                    TaskReminder.status == status
                )
            )
            result = await self.db.execute(stmt)
            status_counts[status.value] = result.scalar() or 0

        # Count overdue
        stmt = select(func.count(TaskReminder.id)).where(
            and_(
                TaskReminder.user_id == user_id,
                TaskReminder.due_date < datetime.utcnow(),
                TaskReminder.status.in_([ReminderStatus.PENDING, ReminderStatus.REMINDED])
            )
        )
        result = await self.db.execute(stmt)
        overdue_count = result.scalar() or 0

        return {
            **status_counts,
            "overdue": overdue_count
        }
