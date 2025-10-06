import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    LargeBinary,
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB

from app.db.postgres import Base

class Lure(Base):
    __tablename__ = "lures"
    __table_args__ = {"schema": "chimera_deception"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lure_type = Column(String(50), nullable=False)
    encrypted_content = Column(LargeBinary, nullable=False)
    deployment_metadata = Column(JSONB, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

class TriggerEvent(Base):
    __tablename__ = "trigger_events"
    __table_args__ = {"schema": "chimera_deception"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lure_id = Column(UUID(as_uuid=True), ForeignKey("chimera_deception.lures.id"), nullable=False)
    trigger_type = Column(String(50), nullable=False)
    session_id = Column(UUID(as_uuid=True), index=True)
    source_ip = Column(INET)
    user_agent = Column(Text)
    trigger_metadata = Column(JSONB)
    severity = Column(String(20))
    handled = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=False, default=datetime.utcnow)