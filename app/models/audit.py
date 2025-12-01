import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Text,
    Float,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB

from app.db.postgres import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=True)
    session_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    ip_address = Column(INET)
    user_agent = Column(Text)
    outcome = Column(String(20))  # e.g., SUCCESS, FAILURE, ERROR
    details = Column(JSONB)
    risk_score = Column(Float)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_audit_user_timestamp", "user_id", "timestamp"),
        Index("ix_audit_type_timestamp", "event_type", "timestamp"),
    )

    def is_security_relevant(self) -> bool:
        """
        Determines if the event is security-relevant based on its type or outcome.
        This can be expanded with more sophisticated rules.
        """
        security_event_types = {"AUTH", "SECURITY", "ADMIN"}
        if self.event_type in security_event_types:
            return True
        if self.outcome and self.outcome.upper() == "FAILURE":
            return True
        return False