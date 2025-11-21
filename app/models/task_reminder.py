"""
Task Reminder Model
Scheduled memory notifications and task tracking
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey,
    Index, Enum
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.postgres import Base


class ReminderStatus(str, PyEnum):
    """Status of a task reminder"""
    PENDING = "pending"
    REMINDED = "reminded"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReminderPriority(str, PyEnum):
    """Priority level of a task reminder"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskReminder(Base):
    __tablename__ = "task_reminders"
    __table_args__ = (
        Index("idx_reminder_user_status", "user_id", "status"),
        Index("idx_reminder_due_date", "due_date"),
        Index("idx_reminder_reminder_date", "reminder_date"),
        Index("idx_reminder_memory", "associated_memory_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Task details
    title = Column(String(500), nullable=False)
    description = Column(Text)

    # Timing
    due_date = Column(DateTime)  # When the task is due
    reminder_date = Column(DateTime, nullable=False)  # When to send the reminder

    # Status and priority
    status = Column(
        Enum(ReminderStatus),
        nullable=False,
        default=ReminderStatus.PENDING
    )
    priority = Column(
        Enum(ReminderPriority),
        nullable=False,
        default=ReminderPriority.MEDIUM
    )

    # Optional association with a memory
    associated_memory_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memories.id"),
        nullable=True
    )

    # Additional metadata
    extra_data = Column(JSONB, nullable=False, default={})

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    reminded_at = Column(DateTime)  # When the reminder was actually sent
    completed_at = Column(DateTime)  # When the task was marked complete

    def __repr__(self):
        return f"<TaskReminder(id={self.id}, title='{self.title}', status={self.status})>"

    @property
    def is_overdue(self) -> bool:
        """Check if the task is overdue"""
        if not self.due_date:
            return False
        return datetime.utcnow() > self.due_date and self.status not in [
            ReminderStatus.COMPLETED,
            ReminderStatus.CANCELLED
        ]

    @property
    def should_remind(self) -> bool:
        """Check if the reminder should be sent now"""
        return (
            self.status == ReminderStatus.PENDING and
            datetime.utcnow() >= self.reminder_date
        )
