import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey,
    Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from pgvector.sqlalchemy import Vector

from app.db.postgres import Base

class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_content_hash", "content_hash"),
        Index("idx_extra_data_gin", "extra_data", postgresql_using="gin"),
        CheckConstraint("char_length(content) >= 1", name="check_content_not_empty"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA256 hash
    embedding = Column(Vector(768))  # Embedding vector
    extra_data = Column(JSONB, nullable=False, default={})

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Computed properties
    @hybrid_property
    def age_days(self):
        """Age of memory in days"""
        return (datetime.utcnow() - self.created_at).days

    @hybrid_property
    def access_frequency(self):
        """How frequently this memory is accessed"""
        if not self.extra_data or not self.extra_data.get("access_count"):
            return 0
        days_old = max(1, self.age_days)
        return self.extra_data["access_count"] / days_old