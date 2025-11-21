"""
Task Reminder Schemas
Pydantic schemas for task reminder API requests and responses
"""

from pydantic import BaseModel, UUID4, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.task_reminder import ReminderStatus, ReminderPriority


# Shared properties
class TaskReminderBase(BaseModel):
    title: str = Field(..., max_length=500, description="Title of the task reminder")
    description: Optional[str] = Field(None, description="Detailed description of the task")
    reminder_date: datetime = Field(..., description="When to send the reminder")
    due_date: Optional[datetime] = Field(None, description="When the task is due")
    priority: ReminderPriority = Field(ReminderPriority.MEDIUM, description="Priority level")
    associated_memory_id: Optional[UUID4] = Field(None, description="Associated memory ID")
    extra_data: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Properties to receive via API on creation
class TaskReminderCreate(TaskReminderBase):
    pass


# Properties to receive via API on update
class TaskReminderUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    reminder_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    priority: Optional[ReminderPriority] = None
    status: Optional[ReminderStatus] = None
    extra_data: Optional[Dict[str, Any]] = None


# Properties shared in database
class TaskReminderInDBBase(TaskReminderBase):
    id: UUID4
    user_id: UUID4
    status: ReminderStatus
    created_at: datetime
    updated_at: datetime
    reminded_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Additional properties to return via API
class TaskReminderResponse(TaskReminderInDBBase):
    is_overdue: bool = Field(..., description="Whether the task is overdue")
    should_remind: bool = Field(..., description="Whether the reminder should be sent now")

    @classmethod
    def from_orm_with_computed(cls, reminder):
        """Create response with computed properties"""
        data = {
            **{k: getattr(reminder, k) for k in TaskReminderInDBBase.__fields__},
            "is_overdue": reminder.is_overdue,
            "should_remind": reminder.should_remind
        }
        return cls(**data)


# Statistics response
class TaskReminderStats(BaseModel):
    pending: int = 0
    reminded: int = 0
    completed: int = 0
    cancelled: int = 0
    overdue: int = 0
