"""
DSMILSYSTEM-aligned Memory Model

This module defines the memory model aligned with DSMILSYSTEM architecture:
- Layer semantics (2-9)
- Device semantics (0-103)
- Clearance tokens and ROE metadata
- Multi-tier storage (hot/warm/cold)
- Event correlation IDs
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Text, DateTime, Integer, ForeignKey,
    Index, CheckConstraint, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from pgvector.sqlalchemy import Vector
import enum

from app.db.postgres import Base


class MemoryTier(str, enum.Enum):
    """Memory storage tier"""
    HOT = "hot"      # Redis Streams
    WARM = "warm"    # SQLite (tmpfs)
    COLD = "cold"    # PostgreSQL


class Memory(Base):
    """
    DSMILSYSTEM-aligned Memory model.
    
    All memories are tagged with:
    - layer_id: Layer 2-9
    - device_id: Device 0-103
    - clearance_token: Access control token
    """
    __tablename__ = "memories_dsmil"
    __table_args__ = (
        Index("idx_layer_device", "layer_id", "device_id"),
        Index("idx_clearance_token", "clearance_token"),
        Index("idx_correlation_id", "correlation_id"),
        Index("idx_tier", "tier"),
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_content_hash", "content_hash"),
        Index("idx_extra_data_gin", "extra_data", postgresql_using="gin"),
        Index("idx_tags_gin", "tags", postgresql_using="gin"),
        CheckConstraint("layer_id >= 2 AND layer_id <= 9", name="check_layer_range"),
        CheckConstraint("device_id >= 0 AND device_id <= 103", name="check_device_range"),
        CheckConstraint("char_length(content) >= 1", name="check_content_not_empty"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # DSMILSYSTEM semantics
    layer_id = Column(Integer, nullable=False)  # Layers 2-9
    device_id = Column(Integer, nullable=False)  # Devices 0-103
    clearance_token = Column(String(128), nullable=False)  # Clearance token
    
    # User context (maintained for backward compatibility)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA256 hash
    embedding = Column(Vector(2048))  # 2048d INT8 quantized embedding
    tags = Column(JSONB, nullable=False, default=[])  # List of tags
    
    # Metadata
    extra_data = Column(JSONB, nullable=False, default={})
    roe_metadata = Column(JSONB, nullable=False, default={})  # Rules of Engagement
    correlation_id = Column(String(128))  # Event correlation ID
    
    # Storage tier
    tier = Column(SQLEnum(MemoryTier), nullable=False, default=MemoryTier.COLD)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    promoted_at = Column(DateTime)  # When promoted to warm/cold
    demoted_at = Column(DateTime)  # When demoted from hot/warm

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "layer_id": self.layer_id,
            "device_id": self.device_id,
            "clearance_token": self.clearance_token,
            "user_id": str(self.user_id),
            "content": self.content,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "extra_data": self.extra_data,
            "roe_metadata": self.roe_metadata,
            "correlation_id": self.correlation_id,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
        }
