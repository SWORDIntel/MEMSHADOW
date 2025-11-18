"""
Knowledge Graph Database Models
Phase 3: Intelligence Layer - Persistent storage for knowledge graph
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, ForeignKey, Index, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from app.db.postgres import Base

class KGNode(Base):
    """Knowledge Graph Node model"""
    __tablename__ = "kg_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(255), unique=True, nullable=False, index=True)
    node_type = Column(String(50), nullable=False, index=True)
    label = Column(String(255), nullable=False)
    properties = Column(JSONB, default={})
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_kg_node_type_user', 'node_type', 'user_id'),
        Index('idx_kg_node_label', 'label'),
    )

class KGEdge(Base):
    """Knowledge Graph Edge model"""
    __tablename__ = "kg_edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node_id = Column(String(255), nullable=False, index=True)
    target_node_id = Column(String(255), nullable=False, index=True)
    edge_type = Column(String(50), nullable=False, index=True)
    weight = Column(Float, default=1.0)
    properties = Column(JSONB, default={})
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_kg_edge_source_target', 'source_node_id', 'target_node_id'),
        Index('idx_kg_edge_type_user', 'edge_type', 'user_id'),
    )

class MemoryEnrichment(Base):
    """Memory Enrichment metadata"""
    __tablename__ = "memory_enrichments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id"), nullable=False, index=True)
    
    # NLP Enrichment
    entities = Column(JSONB, default=[])  # List of extracted entities
    keywords = Column(JSONB, default=[])  # List of keywords with scores
    sentiment = Column(JSONB, default={})  # Sentiment analysis results
    language = Column(String(10), default="en")
    summary = Column(Text)
    
    # Relationships
    relationships = Column(JSONB, default=[])  # Extracted relationships
    
    # LLM Enrichment
    llm_summary = Column(Text)
    insights = Column(JSONB, default=[])
    questions = Column(JSONB, default=[])
    classifications = Column(JSONB, default={})
    
    # Metadata
    enrichment_version = Column(Integer, default=1)
    enriched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_enrichment_memory', 'memory_id'),
    )

class AccessPattern(Base):
    """User access patterns for predictive retrieval"""
    __tablename__ = "access_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id"), nullable=False, index=True)
    
    # Access context
    query = Column(Text)
    context = Column(JSONB, default={})
    
    # Temporal information
    accessed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    
    # Session information
    session_id = Column(UUID(as_uuid=True), index=True)
    
    __table_args__ = (
        Index('idx_access_user_time', 'user_id', 'accessed_at'),
        Index('idx_access_memory_time', 'memory_id', 'accessed_at'),
        Index('idx_access_temporal', 'user_id', 'hour_of_day', 'day_of_week'),
    )
