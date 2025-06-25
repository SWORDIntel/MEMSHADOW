# MEMSHADOW Phase 1: Foundation Implementation Guide

## Executive Summary

This document provides a comprehensive implementation guide for Phase 1 of Project MEMSHADOW, covering weeks 1-8 of development. Phase 1 establishes the core infrastructure, API framework, data persistence layers, and essential security components that will serve as the foundation for all subsequent phases.

---

## Table of Contents

1. [Project Setup & Development Environment](#1-project-setup--development-environment)
2. [Core API Implementation](#2-core-api-implementation)
3. [Database Infrastructure](#3-database-infrastructure)
4. [Ingestion/Retrieval Pipeline](#4-ingestionretrieval-pipeline)
5. [SDAP Backup System](#5-sdap-backup-system)
6. [Initial MFA/A Implementation](#6-initial-mfaa-implementation)
7. [Testing Strategy](#7-testing-strategy)
8. [Deployment & Operations](#8-deployment--operations)

---

## 1. Project Setup & Development Environment

### 1.1 Repository Structure

```bash
memshadow/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── security-scan.yml
│   └── ISSUE_TEMPLATE/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── middleware.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── memory.py
│   │       └── health.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── postgres.py
│   │   ├── chromadb.py
│   │   └── redis.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── user.py
│   │   └── auth.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── user.py
│   │   └── auth.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── memory_service.py
│   │   ├── embedding_service.py
│   │   ├── auth_service.py
│   │   └── mfa_service.py
│   └── workers/
│       ├── __init__.py
│       ├── celery_app.py
│       └── tasks.py
├── scripts/
│   ├── sdap/
│   │   ├── sdap_backup.sh
│   │   └── sdap_restore.sh
│   └── setup/
│       ├── init_db.py
│       └── create_admin.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   └── docker-compose.yml
├── migrations/
│   └── alembic/
├── docs/
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── .env.example
├── pyproject.toml
├── README.md
└── Makefile
```

### 1.2 Development Environment Setup

```bash
# Makefile
.PHONY: install dev test clean

PYTHON := python3.11
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements/dev.txt
	$(PYTHON_VENV) -m pre_commit install

dev:
	docker-compose -f docker/docker-compose.yml up -d postgres redis chromadb
	$(PYTHON_VENV) scripts/setup/init_db.py
	$(PYTHON_VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=app --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	docker-compose -f docker/docker-compose.yml down -v
```

### 1.3 Configuration Management

```python
# app/core/config.py
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
import secrets

class Settings(BaseSettings):
    # API Configuration
    PROJECT_NAME: str = "MEMSHADOW"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DATABASE_URL: Optional[PostgresDsn] = None
    
    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.data.get("POSTGRES_USER"),
            password=values.data.get("POSTGRES_PASSWORD"),
            host=values.data.get("POSTGRES_SERVER"),
            path=f"{values.data.get('POSTGRES_DB') or ''}",
        )
    
    # Redis Configuration
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"
    
    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "memshadow_memories"
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Security Configuration
    ALGORITHM: str = "HS256"
    BCRYPT_ROUNDS: int = 12
    
    # MFA Configuration
    MFA_ISSUER: str = "MEMSHADOW"
    FIDO2_RP_ID: str = "localhost"
    FIDO2_RP_NAME: str = "MEMSHADOW"
    
    # SDAP Configuration
    SDAP_BACKUP_PATH: str = "/var/backups/memshadow"
    SDAP_ARCHIVE_SERVER: str = "backup.memshadow.internal"
    SDAP_GPG_KEY_ID: str = ""
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_CACHE_TTL: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

---

## 2. Core API Implementation

### 2.1 FastAPI Application Structure

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from app.api.v1 import auth, memory, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.db import postgres, chromadb, redis
from app.api.middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware
)

# Setup structured logging
setup_logging()
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting MEMSHADOW API", version=settings.VERSION)
    
    # Initialize database connections
    await postgres.init_db()
    await chromadb.init_client()
    await redis.init_pool()
    
    # Warm up caches
    logger.info("Warming up caches")
    # TODO: Implement cache warming
    
    yield
    
    # Shutdown
    logger.info("Shutting down MEMSHADOW API")
    await postgres.close_db()
    await chromadb.close_client()
    await redis.close_pool()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(memory.router, prefix=f"{settings.API_V1_STR}/memory", tags=["memory"])
```

### 2.2 API Endpoints Implementation

```python
# app/api/v1/memory.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.api.dependencies import get_current_user, get_db
from app.schemas.memory import (
    MemoryCreate,
    MemoryResponse,
    MemorySearch,
    MemoryUpdate
)
from app.models.user import User
from app.services.memory_service import MemoryService

router = APIRouter()
logger = structlog.get_logger()

@router.post("/ingest", response_model=MemoryResponse)
async def ingest_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_in: MemoryCreate,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
) -> MemoryResponse:
    """Ingest a new memory into the system"""
    memory_service = MemoryService(db)
    
    # Create memory record
    memory = await memory_service.create_memory(
        user_id=current_user.id,
        content=memory_in.content,
        metadata=memory_in.metadata
    )
    
    # Queue embedding generation
    background_tasks.add_task(
        memory_service.generate_embedding,
        memory_id=memory.id,
        content=memory_in.content
    )
    
    logger.info(
        "Memory ingested",
        user_id=str(current_user.id),
        memory_id=str(memory.id)
    )
    
    return MemoryResponse.from_orm(memory)

@router.post("/retrieve", response_model=List[MemoryResponse])
async def retrieve_memories(
    *,
    db: AsyncSession = Depends(get_db),
    search: MemorySearch,
    current_user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[MemoryResponse]:
    """Retrieve memories using semantic search"""
    memory_service = MemoryService(db)
    
    memories = await memory_service.search_memories(
        user_id=current_user.id,
        query=search.query,
        filters=search.filters,
        limit=limit,
        offset=offset
    )
    
    logger.info(
        "Memories retrieved",
        user_id=str(current_user.id),
        count=len(memories)
    )
    
    return [MemoryResponse.from_orm(m) for m in memories]

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_id: str,
    current_user: User = Depends(get_current_user)
) -> MemoryResponse:
    """Get a specific memory by ID"""
    memory_service = MemoryService(db)
    
    memory = await memory_service.get_memory(
        memory_id=memory_id,
        user_id=current_user.id
    )
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return MemoryResponse.from_orm(memory)

@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_id: str,
    memory_update: MemoryUpdate,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
) -> MemoryResponse:
    """Update an existing memory"""
    memory_service = MemoryService(db)
    
    memory = await memory_service.update_memory(
        memory_id=memory_id,
        user_id=current_user.id,
        updates=memory_update.dict(exclude_unset=True)
    )
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Re-generate embedding if content changed
    if memory_update.content:
        background_tasks.add_task(
            memory_service.generate_embedding,
            memory_id=memory.id,
            content=memory_update.content
        )
    
    return MemoryResponse.from_orm(memory)

@router.delete("/{memory_id}")
async def delete_memory(
    *,
    db: AsyncSession = Depends(get_db),
    memory_id: str,
    current_user: User = Depends(get_current_user)
) -> dict:
    """Delete a memory"""
    memory_service = MemoryService(db)
    
    success = await memory_service.delete_memory(
        memory_id=memory_id,
        user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"status": "deleted", "memory_id": memory_id}
```

### 2.3 Middleware Implementation

```python
# app/api/middleware.py
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        
        # Add to request state
        request.state.request_id = request_id
        
        # Add to logger context
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Clear context
        structlog.contextvars.clear_contextvars()
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        now = time.time()
        
        # Clean old entries
        self.clients = {
            ip: times for ip, times in self.clients.items()
            if any(t > now - self.period for t in times)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            requests = [t for t in self.clients[client_ip] if t > now - self.period]
            if len(requests) >= self.calls:
                logger.warning("Rate limit exceeded", client_ip=client_ip)
                return Response(content="Rate limit exceeded", status_code=429)
            self.clients[client_ip] = requests + [now]
        else:
            self.clients[client_ip] = [now]
        
        return await call_next(request)
```

---

## 3. Database Infrastructure

### 3.1 PostgreSQL Setup

```python
# app/db/postgres.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Create async engine
engine = create_async_engine(
    str(settings.DATABASE_URL),
    poolclass=NullPool,  # Use NullPool for async
    echo=False,
    future=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

async def init_db():
    """Initialize database"""
    try:
        async with engine.begin() as conn:
            # Install extensions
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise

async def close_db():
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### 3.2 Database Models

```python
# app/models/memory.py
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, 
    Index, CheckConstraint, ARRAY, Float
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
        Index("idx_metadata_gin", "metadata", postgresql_using="gin"),
        CheckConstraint("char_length(content) >= 1", name="check_content_not_empty"),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA256 hash
    embedding = Column(Vector(768))  # Embedding vector
    metadata = Column(JSONB, nullable=False, default={})
    
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
        if not self.metadata.get("access_count"):
            return 0
        days_old = max(1, self.age_days)
        return self.metadata["access_count"] / days_old

# app/models/user.py
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.postgres import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # MFA fields
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String)  # Encrypted
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
```

### 3.3 ChromaDB Integration

```python
# app/db/chromadb.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import structlog

from app.core.config import settings

logger = structlog.get_logger()

class ChromaDBClient:
    def __init__(self):
        self.client = None
        self.collection = None
    
    async def init_client(self):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB client initialized", 
                       collection=settings.CHROMA_COLLECTION)
        except Exception as e:
            logger.error("ChromaDB initialization failed", error=str(e))
            raise
    
    async def close_client(self):
        """Close ChromaDB client"""
        # ChromaDB HTTP client doesn't need explicit closing
        logger.info("ChromaDB client closed")
    
    async def add_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add embedding to ChromaDB"""
        try:
            self.collection.add(
                embeddings=[embedding],
                ids=[memory_id],
                metadatas=[metadata]
            )
            logger.debug("Embedding added", memory_id=memory_id)
        except Exception as e:
            logger.error("Failed to add embedding", 
                        memory_id=memory_id, 
                        error=str(e))
            raise
    
    async def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar embeddings"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            logger.debug("Similarity search completed", 
                        n_results=len(results['ids'][0]))
            
            return results
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            raise
    
    async def delete_embedding(self, memory_id: str):
        """Delete embedding from ChromaDB"""
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug("Embedding deleted", memory_id=memory_id)
        except Exception as e:
            logger.error("Failed to delete embedding", 
                        memory_id=memory_id, 
                        error=str(e))
            raise

# Global client instance
chroma_client = ChromaDBClient()

async def init_client():
    await chroma_client.init_client()

async def close_client():
    await chroma_client.close_client()
```

### 3.4 Redis Integration

```python
# app/db/redis.py
import redis.asyncio as redis
from typing import Optional, Any
import json
import structlog

from app.core.config import settings

logger = structlog.get_logger()

class RedisClient:
    def __init__(self):
        self.pool = None
    
    async def init_pool(self):
        """Initialize Redis connection pool"""
        try:
            self.pool = redis.ConnectionPool.from_url(
                str(settings.REDIS_URL),
                max_connections=50,
                decode_responses=True
            )
            
            # Test connection
            async with redis.Redis(connection_pool=self.pool) as conn:
                await conn.ping()
            
            logger.info("Redis pool initialized")
        except Exception as e:
            logger.error("Redis initialization failed", error=str(e))
            raise
    
    async def close_pool(self):
        """Close Redis connection pool"""
        if self.pool:
            await self.pool.disconnect()
            logger.info("Redis pool closed")
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client from pool"""
        return redis.Redis(connection_pool=self.pool)
    
    # Cache operations
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.get_client() as conn:
            value = await conn.get(key)
            if value:
                return json.loads(value)
            return None
    
    async def cache_set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ):
        """Set value in cache"""
        async with self.get_client() as conn:
            serialized = json.dumps(value)
            if ttl:
                await conn.setex(key, ttl, serialized)
            else:
                await conn.set(key, serialized)
    
    async def cache_delete(self, key: str):
        """Delete value from cache"""
        async with self.get_client() as conn:
            await conn.delete(key)
    
    # Rate limiting
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> bool:
        """Check if rate limit is exceeded"""
        async with self.get_client() as conn:
            pipe = conn.pipeline()
            now = int(time.time())
            window_start = now - window
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Count requests in window
            pipe.zcard(key)
            # Set expiry
            pipe.expire(key, window + 1)
            
            results = await pipe.execute()
            request_count = results[2]
            
            return request_count <= limit

# Global client instance
redis_client = RedisClient()

async def init_pool():
    await redis_client.init_pool()

async def close_pool():
    await redis_client.close_pool()
```

---

## 4. Ingestion/Retrieval Pipeline

### 4.1 Memory Service Implementation

```python
# app/services/memory_service.py
import hashlib
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import structlog

from app.models.memory import Memory
from app.db.chromadb import chroma_client
from app.db.redis import redis_client
from app.services.embedding_service import EmbeddingService
from app.workers.tasks import generate_embedding_task

logger = structlog.get_logger()

class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService()
    
    async def create_memory(
        self,
        user_id: UUID,
        content: str,
        metadata: Dict[str, Any]
    ) -> Memory:
        """Create a new memory"""
        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check for duplicate
        existing = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.content_hash == content_hash
                )
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Duplicate memory content")
        
        # Create memory record
        memory = Memory(
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            metadata=metadata
        )
        
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)
        
        logger.info("Memory created", 
                   memory_id=str(memory.id),
                   user_id=str(user_id))
        
        return memory
    
    async def generate_embedding(self, memory_id: str, content: str):
        """Generate and store embedding for memory"""
        try:
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content)
            
            # Update memory with embedding
            memory = await self.db.get(Memory, memory_id)
            if memory:
                memory.embedding = embedding
                await self.db.commit()
            
            # Store in ChromaDB
            await chroma_client.add_embedding(
                memory_id=memory_id,
                embedding=embedding,
                metadata={
                    "user_id": str(memory.user_id),
                    "created_at": memory.created_at.isoformat()
                }
            )
            
            # Cache embedding
            cache_key = f"embedding:{memory_id}"
            await redis_client.cache_set(
                cache_key, 
                embedding, 
                ttl=settings.EMBEDDING_CACHE_TTL
            )
            
            logger.info("Embedding generated", memory_id=memory_id)
            
        except Exception as e:
            logger.error("Embedding generation failed", 
                        memory_id=memory_id,
                        error=str(e))
            raise
    
    async def search_memories(
        self,
        user_id: UUID,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Memory]:
        """Search memories using semantic similarity"""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Search in ChromaDB
        where_clause = {"user_id": str(user_id)}
        if filters:
            where_clause.update(filters)
        
        results = await chroma_client.search_similar(
            query_embedding=query_embedding,
            n_results=limit + offset,
            where=where_clause
        )
        
        # Get memory IDs from results
        memory_ids = results['ids'][0][offset:offset + limit]
        
        if not memory_ids:
            return []
        
        # Fetch memories from database
        stmt = select(Memory).where(Memory.id.in_(memory_ids))
        result = await self.db.execute(stmt)
        memories = result.scalars().all()
        
        # Sort by relevance score
        memory_dict = {str(m.id): m for m in memories}
        sorted_memories = [
            memory_dict[mid] for mid in memory_ids 
            if mid in memory_dict
        ]
        
        # Update access timestamps
        for memory in sorted_memories:
            memory.accessed_at = datetime.utcnow()
            metadata = memory.metadata or {}
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            memory.metadata = metadata
        
        await self.db.commit()
        
        logger.info("Memories searched", 
                   user_id=str(user_id),
                   query=query[:50],
                   results=len(sorted_memories))
        
        return sorted_memories
    
    async def get_memory(
        self, 
        memory_id: str, 
        user_id: UUID
    ) -> Optional[Memory]:
        """Get a specific memory"""
        stmt = select(Memory).where(
            and_(
                Memory.id == memory_id,
                Memory.user_id == user_id
            )
        )
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()
        
        if memory:
            # Update access timestamp
            memory.accessed_at = datetime.utcnow()
            metadata = memory.metadata or {}
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            memory.metadata = metadata
            await self.db.commit()
        
        return memory
    
    async def update_memory(
        self,
        memory_id: str,
        user_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Memory]:
        """Update an existing memory"""
        memory = await self.get_memory(memory_id, user_id)
        if not memory:
            return None
        
        # Update fields
        for field, value in updates.items():
            if hasattr(memory, field):
                setattr(memory, field, value)
        
        # Update content hash if content changed
        if "content" in updates:
            memory.content_hash = hashlib.sha256(
                updates["content"].encode()
            ).hexdigest()
        
        memory.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(memory)
        
        logger.info("Memory updated", memory_id=memory_id)
        
        return memory
    
    async def delete_memory(
        self, 
        memory_id: str, 
        user_id: UUID
    ) -> bool:
        """Delete a memory"""
        memory = await self.get_memory(memory_id, user_id)
        if not memory:
            return False
        
        # Delete from ChromaDB
        await chroma_client.delete_embedding(memory_id)
        
        # Delete from cache
        cache_key = f"embedding:{memory_id}"
        await redis_client.cache_delete(cache_key)
        
        # Delete from database
        await self.db.delete(memory)
        await self.db.commit()
        
        logger.info("Memory deleted", 
                   memory_id=memory_id,
                   user_id=str(user_id))
        
        return True
```

### 4.2 Embedding Service

```python
# app/services/embedding_service.py
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import structlog

from app.core.config import settings
from app.db.redis import redis_client

logger = structlog.get_logger()

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            # Detect available device
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Load model
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=self.device
            )
            
            # Warm up model
            _ = self.model.encode("warmup", convert_to_numpy=True)
            
            logger.info("Embedding model initialized", 
                       model=settings.EMBEDDING_MODEL,
                       device=str(self.device))
            
        except Exception as e:
            logger.error("Failed to initialize embedding model", 
                        error=str(e))
            raise
    
    async def generate_embedding(
        self, 
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for text"""
        # Check cache
        if use_cache:
            cache_key = f"embedding:text:{hashlib.md5(text.encode()).hexdigest()}"
            cached = await redis_client.cache_get(cache_key)
            if cached:
                logger.debug("Embedding cache hit")
                return cached
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            # Cache result
            if use_cache:
                await redis_client.cache_set(
                    cache_key,
                    embedding_list,
                    ttl=settings.EMBEDDING_CACHE_TTL
                )
            
            return embedding_list
            
        except Exception as e:
            logger.error("Embedding generation failed", 
                        error=str(e),
                        text_length=len(text))
            raise
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error("Batch embedding generation failed", 
                        error=str(e),
                        batch_size=len(texts))
            raise
```

### 4.3 Celery Worker Tasks

```python
# app/workers/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "memshadow",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# app/workers/tasks.py
from celery import current_task
from typing import Dict, Any
import structlog

from app.workers.celery_app import celery_app
from app.services.memory_service import MemoryService
from app.services.embedding_service import EmbeddingService

logger = structlog.get_logger()

@celery_app.task(bind=True, max_retries=3)
def generate_embedding_task(
    self,
    memory_id: str,
    content: str,
    user_id: str
) -> Dict[str, Any]:
    """Generate embedding for memory (background task)"""
    try:
        logger.info("Starting embedding generation", 
                   memory_id=memory_id,
                   task_id=current_task.request.id)
        
        # Initialize services
        embedding_service = EmbeddingService()
        
        # Generate embedding
        embedding = embedding_service.generate_embedding(content)
        
        # Store in ChromaDB and update memory
        # Note: This is simplified - in production, you'd need proper
        # database session management for Celery tasks
        
        logger.info("Embedding generation completed", 
                   memory_id=memory_id)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "embedding_dimension": len(embedding)
        }
        
    except Exception as e:
        logger.error("Embedding generation failed", 
                    memory_id=memory_id,
                    error=str(e))
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

@celery_app.task
def cleanup_old_memories_task() -> Dict[str, Any]:
    """Cleanup old memories based on TTL"""
    # Implementation for memory cleanup
    pass

@celery_app.task
def optimize_embeddings_task() -> Dict[str, Any]:
    """Optimize ChromaDB indices"""
    # Implementation for index optimization
    pass
```

---

## 5. SDAP Backup System

### 5.1 SDAP Backup Script

```bash
#!/bin/bash
# scripts/sdap/sdap_backup.sh

set -euo pipefail

# Load configuration
source /etc/memshadow/sdap.conf

# Variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="memshadow_backup_${TIMESTAMP}"
BACKUP_DIR="${SDAP_BACKUP_PATH}/${BACKUP_NAME}"
LOG_FILE="/var/log/memshadow/sdap_backup_${TIMESTAMP}.log"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Create backup directory
log "Starting SDAP backup process"
mkdir -p "${BACKUP_DIR}" || error_exit "Failed to create backup directory"

# Backup PostgreSQL
log "Backing up PostgreSQL database"
PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
    -h "${POSTGRES_HOST}" \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    -f "${BACKUP_DIR}/postgres_dump.sql" \
    --verbose \
    --no-owner \
    --no-acl \
    || error_exit "PostgreSQL backup failed"

# Backup ChromaDB data
log "Backing up ChromaDB data"
if [ -d "${CHROMA_PERSIST_DIR}" ]; then
    tar -czf "${BACKUP_DIR}/chromadb_data.tar.gz" \
        -C "${CHROMA_PERSIST_DIR}" . \
        || error_exit "ChromaDB backup failed"
else
    log "WARNING: ChromaDB persist directory not found"
fi

# Create metadata file
log "Creating backup metadata"
cat > "${BACKUP_DIR}/metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "version": "${MEMSHADOW_VERSION}",
    "components": {
        "postgresql": {
            "version": "$(psql --version | awk '{print $3}')",
            "database": "${POSTGRES_DB}",
            "size": "$(du -sh ${BACKUP_DIR}/postgres_dump.sql | cut -f1)"
        },
        "chromadb": {
            "size": "$(du -sh ${BACKUP_DIR}/chromadb_data.tar.gz 2>/dev/null | cut -f1 || echo 'N/A')"
        }
    },
    "host": "$(hostname)",
    "checksum": ""
}
EOF

# Calculate checksums
log "Calculating checksums"
cd "${BACKUP_DIR}"
sha256sum * > checksums.sha256

# Create archive
log "Creating compressed archive"
cd "${SDAP_BACKUP_PATH}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/" \
    || error_exit "Archive creation failed"

# Encrypt archive
log "Encrypting archive"
gpg --encrypt \
    --armor \
    --recipient "${SDAP_GPG_KEY_ID}" \
    --cipher-algo AES256 \
    --output "${BACKUP_NAME}.tar.gz.asc" \
    "${BACKUP_NAME}.tar.gz" \
    || error_exit "Encryption failed"

# Transfer to archive server
log "Transferring to archive server"
scp -i "${SDAP_SSH_KEY}" \
    -o StrictHostKeyChecking=yes \
    -o ConnectTimeout=30 \
    "${BACKUP_NAME}.tar.gz.asc" \
    "sdap_receiver@${SDAP_ARCHIVE_SERVER}:/incoming/" \
    || error_exit "Transfer failed"

# Verify transfer
log "Verifying transfer"
REMOTE_CHECKSUM=$(ssh -i "${SDAP_SSH_KEY}" \
    "sdap_receiver@${SDAP_ARCHIVE_SERVER}" \
    "sha256sum /incoming/${BACKUP_NAME}.tar.gz.asc" | awk '{print $1}')

LOCAL_CHECKSUM=$(sha256sum "${BACKUP_NAME}.tar.gz.asc" | awk '{print $1}')

if [ "${LOCAL_CHECKSUM}" != "${REMOTE_CHECKSUM}" ]; then
    error_exit "Checksum verification failed"
fi

# Cleanup
log "Cleaning up local files"
rm -rf "${BACKUP_DIR}"
rm -f "${BACKUP_NAME}.tar.gz"
rm -f "${BACKUP_NAME}.tar.gz.asc"

# Update last backup timestamp
echo "${TIMESTAMP}" > /var/lib/memshadow/last_sdap_backup

log "SDAP backup completed successfully"

# Send notification
if [ -n "${SDAP_NOTIFICATION_WEBHOOK}" ]; then
    curl -X POST "${SDAP_NOTIFICATION_WEBHOOK}" \
        -H "Content-Type: application/json" \
        -d "{\"status\": \"success\", \"backup\": \"${BACKUP_NAME}\", \"size\": \"$(du -sh ${BACKUP_NAME}.tar.gz.asc | cut -f1)\"}"
fi
```

### 5.2 SDAP Configuration

```bash
# /etc/memshadow/sdap.conf

# Database Configuration
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"
POSTGRES_USER="memshadow"
POSTGRES_PASSWORD=""  # Loaded from secure storage
POSTGRES_DB="memshadow"

# ChromaDB Configuration
CHROMA_PERSIST_DIR="/var/lib/chromadb"

# Backup Configuration
SDAP_BACKUP_PATH="/var/backups/memshadow"
SDAP_ARCHIVE_SERVER="backup.memshadow.internal"
SDAP_SSH_KEY="/etc/memshadow/keys/sdap_rsa"
SDAP_GPG_KEY_ID="MEMSHADOW_BACKUP_2024"

# Notification Configuration
SDAP_NOTIFICATION_WEBHOOK=""

# Application Configuration
MEMSHADOW_VERSION="1.0.0"
```

### 5.3 Systemd Timer for SDAP

```ini
# /etc/systemd/system/sdap-backup.service
[Unit]
Description=MEMSHADOW SDAP Backup Service
After=network.target postgresql.service

[Service]
Type=oneshot
User=memshadow
Group=memshadow
ExecStart=/opt/memshadow/scripts/sdap/sdap_backup.sh
StandardOutput=journal
StandardError=journal
Environment="PATH=/usr/local/bin:/usr/bin:/bin"

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/backups/memshadow /var/log/memshadow /var/lib/memshadow

# /etc/systemd/system/sdap-backup.timer
[Unit]
Description=MEMSHADOW SDAP Backup Timer
Requires=sdap-backup.service

[Timer]
OnCalendar=daily
OnCalendar=03:00
RandomizedDelaySec=30m
Persistent=true

[Install]
WantedBy=timers.target
```

---

## 6. Initial MFA/A Implementation

### 6.1 FIDO2/WebAuthn Service

```python
# app/services/mfa_service.py
from typing import Dict, Any, Optional, List
from fido2.server import Fido2Server
from fido2.webauthn import PublicKeyCredentialRpEntity, PublicKeyCredentialUserEntity
from fido2.client import ClientData
from fido2.attestation import Attestation
from fido2 import cbor
import base64
import structlog

from app.core.config import settings
from app.models.auth import WebAuthnCredential

logger = structlog.get_logger()

class MFAService:
    def __init__(self):
        self.rp = PublicKeyCredentialRpEntity(
            id=settings.FIDO2_RP_ID,
            name=settings.FIDO2_RP_NAME
        )
        self.server = Fido2Server(self.rp)
    
    async def begin_registration(
        self, 
        user_id: str, 
        username: str,
        existing_credentials: List[WebAuthnCredential] = None
    ) -> Dict[str, Any]:
        """Begin FIDO2 registration"""
        user = PublicKeyCredentialUserEntity(
            id=user_id.encode(),
            name=username,
            display_name=username
        )
        
        # Exclude existing credentials
        exclude_credentials = []
        if existing_credentials:
            exclude_credentials = [
                {
                    "id": base64.b64decode(cred.credential_id),
                    "type": "public-key"
                }
                for cred in existing_credentials
            ]
        
        # Generate registration options
        options, state = self.server.register_begin(
            user=user,
            credentials=exclude_credentials,
            user_verification="preferred",
            authenticator_attachment="cross-platform"
        )
        
        # Store state in session/cache
        await self._store_challenge_state(user_id, "registration", state)
        
        # Convert to JSON-serializable format
        return {
            "publicKey": {
                "challenge": base64.b64encode(options["publicKey"]["challenge"]).decode(),
                "rp": options["publicKey"]["rp"],
                "user": {
                    "id": base64.b64encode(options["publicKey"]["user"]["id"]).decode(),
                    "name": options["publicKey"]["user"]["name"],
                    "displayName": options["publicKey"]["user"]["displayName"]
                },
                "pubKeyCredParams": options["publicKey"]["pubKeyCredParams"],
                "excludeCredentials": [
                    {
                        "id": base64.b64encode(cred["id"]).decode(),
                        "type": cred["type"]
                    }
                    for cred in options["publicKey"].get("excludeCredentials", [])
                ],
                "authenticatorSelection": options["publicKey"]["authenticatorSelection"],
                "attestation": options["publicKey"]["attestation"],
                "extensions": options["publicKey"].get("extensions", {})
            }
        }
    
    async def complete_registration(
        self,
        user_id: str,
        credential: Dict[str, Any]
    ) -> WebAuthnCredential:
        """Complete FIDO2 registration"""
        # Retrieve state
        state = await self._get_challenge_state(user_id, "registration")
        if not state:
            raise ValueError("Registration state not found")
        
        # Decode credential response
        client_data = ClientData(base64.b64decode(credential["response"]["clientDataJSON"]))
        attestation_object = base64.b64decode(credential["response"]["attestationObject"])
        
        # Verify registration
        auth_data = self.server.register_complete(
            state,
            client_data,
            attestation_object
        )
        
        # Create credential record
        credential_record = WebAuthnCredential(
            user_id=user_id,
            credential_id=base64.b64encode(auth_data.credential_data.credential_id).decode(),
            public_key=base64.b64encode(auth_data.credential_data.public_key).decode(),
            sign_count=auth_data.counter,
            aaguid=auth_data.credential_data.aaguid.hex() if auth_data.credential_data.aaguid else None,
            fmt=attestation_object.get("fmt", "none")
        )
        
        logger.info("FIDO2 registration completed", user_id=user_id)
        
        return credential_record
    
    async def begin_authentication(
        self,
        user_id: str,
        credentials: List[WebAuthnCredential]
    ) -> Dict[str, Any]:
        """Begin FIDO2 authentication"""
        if not credentials:
            raise ValueError("No credentials found for user")
        
        # Convert credentials for authentication
        allowed_credentials = [
            {
                "id": base64.b64decode(cred.credential_id),
                "type": "public-key"
            }
            for cred in credentials
        ]
        
        # Generate authentication options
        options, state = self.server.authenticate_begin(
            credentials=allowed_credentials,
            user_verification="preferred"
        )
        
        # Store state
        await self._store_challenge_state(user_id, "authentication", state)
        
        # Convert to JSON-serializable format
        return {
            "publicKey": {
                "challenge": base64.b64encode(options["publicKey"]["challenge"]).decode(),
                "allowCredentials": [
                    {
                        "id": base64.b64encode(cred["id"]).decode(),
                        "type": cred["type"]
                    }
                    for cred in options["publicKey"]["allowCredentials"]
                ],
                "userVerification": options["publicKey"]["userVerification"],
                "extensions": options["publicKey"].get("extensions", {})
            }
        }
    
    async def complete_authentication(
        self,
        user_id: str,
        credential_id: str,
        credential_response: Dict[str, Any]
    ) -> bool:
        """Complete FIDO2 authentication"""
        # Retrieve state
        state = await self._get_challenge_state(user_id, "authentication")
        if not state:
            raise ValueError("Authentication state not found")
        
        # Get credential from database
        credential = await self._get_credential(user_id, credential_id)
        if not credential:
            raise ValueError("Credential not found")
        
        # Decode response
        client_data = ClientData(base64.b64decode(credential_response["clientDataJSON"]))
        auth_data = base64.b64decode(credential_response["authenticatorData"])
        signature = base64.b64decode(credential_response["signature"])
        
        # Verify authentication
        self.server.authenticate_complete(
            state,
            [credential],
            credential_id,
            client_data,
            auth_data,
            signature
        )
        
        # Update sign count
        await self._update_sign_count(credential_id, auth_data.counter)
        
        logger.info("FIDO2 authentication completed", 
                   user_id=user_id,
                   credential_id=credential_id)
        
        return True
    
    async def _store_challenge_state(
        self, 
        user_id: str, 
        operation: str, 
        state: Any
    ):
        """Store challenge state in Redis"""
        key = f"fido2:{operation}:{user_id}"
        await redis_client.cache_set(key, state, ttl=300)  # 5 minutes
    
    async def _get_challenge_state(
        self, 
        user_id: str, 
        operation: str
    ) -> Optional[Any]:
        """Retrieve challenge state from Redis"""
        key = f"fido2:{operation}:{user_id}"
        state = await redis_client.cache_get(key)
        if state:
            await redis_client.cache_delete(key)  # One-time use
        return state
```

### 6.2 Behavioral Biometrics (Initial)

```python
# app/services/behavioral_biometrics.py
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timedelta
import structlog

from app.db.redis import redis_client

logger = structlog.get_logger()

class BehavioralBiometricsService:
    def __init__(self):
        self.baseline_window = 50  # Number of commands for baseline
        self.deviation_threshold = 2.5  # Standard deviations
    
    async def record_telemetry(
        self, 
        session_id: str, 
        telemetry: Dict[str, Any]
    ):
        """Record user behavior telemetry"""
        key = f"telemetry:{session_id}"
        
        # Add timestamp
        telemetry["timestamp"] = datetime.utcnow().isoformat()
        
        # Store in Redis list
        await redis_client.lpush(key, telemetry)
        await redis_client.expire(key, 3600)  # 1 hour TTL
        
        # Trim to keep only recent data
        await redis_client.ltrim(key, 0, 999)
    
    async def analyze_behavior(
        self, 
        session_id: str
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns"""
        key = f"telemetry:{session_id}"
        
        # Get telemetry data
        telemetry_data = await redis_client.lrange(key, 0, -1)
        if len(telemetry_data) < 10:
            return {"score": 0, "confidence": "low"}
        
        # Extract features
        features = self._extract_features(telemetry_data)
        
        # Compare with baseline
        baseline = await self._get_baseline(session_id)
        if not baseline:
            # Build baseline
            await self._build_baseline(session_id, features)
            return {"score": 0, "confidence": "building"}
        
        # Calculate deviation score
        score = self._calculate_deviation_score(features, baseline)
        
        # Determine confidence
        confidence = "high" if len(telemetry_data) > 50 else "medium"
        
        return {
            "score": score,
            "confidence": confidence,
            "suspicious": score > self.deviation_threshold
        }
    
    def _extract_features(
        self, 
        telemetry_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extract behavioral features from telemetry"""
        features = {
            "command_velocity": 0,
            "typing_speed": 0,
            "error_rate": 0,
            "command_complexity": 0
        }
        
        if len(telemetry_data) < 2:
            return features
        
        # Calculate inter-command intervals
        intervals = []
        for i in range(1, len(telemetry_data)):
            t1 = datetime.fromisoformat(telemetry_data[i-1]["timestamp"])
            t2 = datetime.fromisoformat(telemetry_data[i]["timestamp"])
            interval = (t2 - t1).total_seconds()
            intervals.append(interval)
        
        if intervals:
            features["command_velocity"] = np.mean(intervals)
        
        # Calculate typing speed (characters per second)
        typing_speeds = [
            t.get("typing_speed", 0) 
            for t in telemetry_data 
            if "typing_speed" in t
        ]
        if typing_speeds:
            features["typing_speed"] = np.mean(typing_speeds)
        
        # Calculate error rate
        errors = sum(1 for t in telemetry_data if t.get("error", False))
        features["error_rate"] = errors / len(telemetry_data)
        
        # Calculate command complexity
        complexities = [
            len(t.get("command", "").split()) 
            for t in telemetry_data 
            if "command" in t
        ]
        if complexities:
            features["command_complexity"] = np.mean(complexities)
        
        return features
    
    async def
```python
    async def _get_baseline(
        self, 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get behavioral baseline for session"""
        key = f"baseline:{session_id}"
        return await redis_client.cache_get(key)
    
    async def _build_baseline(
        self, 
        session_id: str, 
        features: Dict[str, float]
    ):
        """Build behavioral baseline"""
        key = f"baseline:{session_id}"
        baseline = {
            "features": features,
            "std_devs": {k: 0.1 for k in features},  # Initial std dev
            "sample_count": 1,
            "created_at": datetime.utcnow().isoformat()
        }
        await redis_client.cache_set(key, baseline, ttl=86400)  # 24 hours
    
    def _calculate_deviation_score(
        self,
        current_features: Dict[str, float],
        baseline: Dict[str, Any]
    ) -> float:
        """Calculate deviation from baseline behavior"""
        baseline_features = baseline["features"]
        std_devs = baseline["std_devs"]
        
        deviations = []
        for feature, value in current_features.items():
            if feature in baseline_features and std_devs[feature] > 0:
                z_score = abs(value - baseline_features[feature]) / std_devs[feature]
                deviations.append(z_score)
        
        return np.mean(deviations) if deviations else 0
```

### 6.3 Authentication API Endpoints

```python
# app/api/v1/auth.py
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.api.dependencies import get_db
from app.schemas.auth import (
    UserCreate, UserResponse, Token, 
    WebAuthnRegistrationBegin, WebAuthnRegistrationComplete,
    WebAuthnAuthenticationBegin, WebAuthnAuthenticationComplete
)
from app.services.auth_service import AuthService
from app.services.mfa_service import MFAService
from app.core.security import create_access_token

router = APIRouter()
logger = structlog.get_logger()

@router.post("/register", response_model=UserResponse)
async def register(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserCreate
) -> UserResponse:
    """Register a new user"""
    auth_service = AuthService(db)
    
    # Check if user exists
    if await auth_service.get_user_by_email(user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = await auth_service.create_user(
        email=user_in.email,
        username=user_in.username,
        password=user_in.password
    )
    
    logger.info("User registered", user_id=str(user.id))
    
    return UserResponse.from_orm(user)

@router.post("/login", response_model=Token)
async def login(
    *,
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """Login with username/password"""
    auth_service = AuthService(db)
    
    # Authenticate user
    user = await auth_service.authenticate_user(
        username=form_data.username,
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username}
    )
    
    logger.info("User logged in", user_id=str(user.id))
    
    return Token(access_token=access_token, token_type="bearer")

@router.post("/webauthn/register/begin")
async def webauthn_register_begin(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Begin WebAuthn registration"""
    mfa_service = MFAService()
    auth_service = AuthService(db)
    
    # Get existing credentials
    credentials = await auth_service.get_user_credentials(current_user.id)
    
    # Begin registration
    options = await mfa_service.begin_registration(
        user_id=str(current_user.id),
        username=current_user.username,
        existing_credentials=credentials
    )
    
    return options

@router.post("/webauthn/register/complete")
async def webauthn_register_complete(
    *,
    db: AsyncSession = Depends(get_db),
    credential_data: WebAuthnRegistrationComplete,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Complete WebAuthn registration"""
    mfa_service = MFAService()
    auth_service = AuthService(db)
    
    # Complete registration
    credential = await mfa_service.complete_registration(
        user_id=str(current_user.id),
        credential=credential_data.dict()
    )
    
    # Save credential
    await auth_service.save_credential(credential)
    
    # Enable MFA for user
    await auth_service.enable_mfa(current_user.id)
    
    logger.info("WebAuthn registration completed", 
               user_id=str(current_user.id))
    
    return {"status": "success", "credential_id": credential.credential_id}

@router.post("/webauthn/authenticate/begin")
async def webauthn_authenticate_begin(
    *,
    db: AsyncSession = Depends(get_db),
    username: str
) -> Dict[str, Any]:
    """Begin WebAuthn authentication"""
    auth_service = AuthService(db)
    mfa_service = MFAService()
    
    # Get user
    user = await auth_service.get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get credentials
    credentials = await auth_service.get_user_credentials(user.id)
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No credentials registered"
        )
    
    # Begin authentication
    options = await mfa_service.begin_authentication(
        user_id=str(user.id),
        credentials=credentials
    )
    
    return options

@router.post("/webauthn/authenticate/complete", response_model=Token)
async def webauthn_authenticate_complete(
    *,
    db: AsyncSession = Depends(get_db),
    auth_data: WebAuthnAuthenticationComplete
) -> Token:
    """Complete WebAuthn authentication"""
    auth_service = AuthService(db)
    mfa_service = MFAService()
    
    # Get user
    user = await auth_service.get_user_by_username(auth_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Complete authentication
    success = await mfa_service.complete_authentication(
        user_id=str(user.id),
        credential_id=auth_data.credential_id,
        credential_response=auth_data.credential_response
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": str(user.id), 
            "username": user.username,
            "mfa_verified": True
        }
    )
    
    logger.info("WebAuthn authentication completed", 
               user_id=str(user.id))
    
    return Token(access_token=access_token, token_type="bearer")
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/unit/test_memory_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from app.services.memory_service import MemoryService
from app.models.memory import Memory

@pytest.fixture
def memory_service():
    mock_db = AsyncMock()
    return MemoryService(mock_db)

@pytest.mark.asyncio
async def test_create_memory_success(memory_service):
    # Arrange
    user_id = uuid4()
    content = "Test memory content"
    metadata = {"tag": "test"}
    
    memory_service.db.execute = AsyncMock(return_value=Mock(scalar_one_or_none=Mock(return_value=None)))
    memory_service.db.add = Mock()
    memory_service.db.commit = AsyncMock()
    memory_service.db.refresh = AsyncMock()
    
    # Act
    result = await memory_service.create_memory(user_id, content, metadata)
    
    # Assert
    assert result is not None
    memory_service.db.add.assert_called_once()
    memory_service.db.commit.assert_called_once()

@pytest.mark.asyncio
async def test_create_memory_duplicate(memory_service):
    # Arrange
    user_id = uuid4()
    content = "Duplicate content"
    metadata = {}
    
    existing_memory = Memory(id=uuid4(), user_id=user_id, content=content)
    memory_service.db.execute = AsyncMock(
        return_value=Mock(scalar_one_or_none=Mock(return_value=existing_memory))
    )
    
    # Act & Assert
    with pytest.raises(ValueError, match="Duplicate memory content"):
        await memory_service.create_memory(user_id, content, metadata)

# tests/unit/test_embedding_service.py
import pytest
from unittest.mock import Mock, patch

from app.services.embedding_service import EmbeddingService

@pytest.fixture
def embedding_service():
    with patch('app.services.embedding_service.SentenceTransformer'):
        service = EmbeddingService()
        service.model = Mock()
        return service

@pytest.mark.asyncio
async def test_generate_embedding(embedding_service):
    # Arrange
    text = "Test text for embedding"
    expected_embedding = [0.1, 0.2, 0.3]
    
    embedding_service.model.encode.return_value = expected_embedding
    
    # Act
    result = await embedding_service.generate_embedding(text, use_cache=False)
    
    # Assert
    assert result == expected_embedding
    embedding_service.model.encode.assert_called_once_with(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
```

### 7.2 Integration Tests

```python
# tests/integration/test_api_memory.py
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models.user import User
from tests.utils import create_test_user, create_test_token

@pytest.mark.asyncio
async def test_ingest_memory_endpoint(
    async_client: AsyncClient,
    db_session: AsyncSession
):
    # Create test user
    user = await create_test_user(db_session)
    token = create_test_token(user.id)
    
    # Test data
    memory_data = {
        "content": "Integration test memory",
        "metadata": {"source": "test", "importance": "high"}
    }
    
    # Make request
    response = await async_client.post(
        "/api/v1/memory/ingest",
        json=memory_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == memory_data["content"]
    assert data["metadata"] == memory_data["metadata"]
    assert "id" in data
    assert "created_at" in data

@pytest.mark.asyncio
async def test_retrieve_memories_endpoint(
    async_client: AsyncClient,
    db_session: AsyncSession
):
    # Create test user and memories
    user = await create_test_user(db_session)
    token = create_test_token(user.id)
    
    # Ingest test memories
    memories = [
        {"content": "Python programming tutorial", "metadata": {"topic": "programming"}},
        {"content": "Machine learning basics", "metadata": {"topic": "ml"}},
        {"content": "FastAPI development guide", "metadata": {"topic": "programming"}}
    ]
    
    for memory in memories:
        await async_client.post(
            "/api/v1/memory/ingest",
            json=memory,
            headers={"Authorization": f"Bearer {token}"}
        )
    
    # Search memories
    search_data = {
        "query": "programming",
        "filters": {"topic": "programming"}
    }
    
    response = await async_client.post(
        "/api/v1/memory/retrieve",
        json=search_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    assert all("programming" in m["content"].lower() or 
              m["metadata"]["topic"] == "programming" 
              for m in data)
```

### 7.3 End-to-End Tests

```python
# tests/e2e/test_full_workflow.py
import pytest
from httpx import AsyncClient
import asyncio

@pytest.mark.asyncio
async def test_complete_memory_workflow(async_client: AsyncClient):
    # 1. Register user
    register_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "SecurePassword123!"
    }
    
    response = await async_client.post("/api/v1/auth/register", json=register_data)
    assert response.status_code == 200
    user = response.json()
    
    # 2. Login
    login_data = {
        "username": "testuser",
        "password": "SecurePassword123!"
    }
    
    response = await async_client.post(
        "/api/v1/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # 3. Ingest memories
    memories = [
        {
            "content": "Project MEMSHADOW is an AI memory persistence system",
            "metadata": {"project": "memshadow", "type": "definition"}
        },
        {
            "content": "MEMSHADOW uses ChromaDB for vector storage",
            "metadata": {"project": "memshadow", "type": "technical"}
        },
        {
            "content": "The system implements FIDO2 authentication",
            "metadata": {"project": "memshadow", "type": "security"}
        }
    ]
    
    headers = {"Authorization": f"Bearer {token}"}
    memory_ids = []
    
    for memory in memories:
        response = await async_client.post(
            "/api/v1/memory/ingest",
            json=memory,
            headers=headers
        )
        assert response.status_code == 200
        memory_ids.append(response.json()["id"])
    
    # Wait for embeddings to be generated
    await asyncio.sleep(2)
    
    # 4. Search memories
    search_query = {
        "query": "What is MEMSHADOW?",
        "filters": {"project": "memshadow"}
    }
    
    response = await async_client.post(
        "/api/v1/memory/retrieve",
        json=search_query,
        headers=headers
    )
    
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert any("AI memory persistence" in r["content"] for r in results)
    
    # 5. Update a memory
    update_data = {
        "content": "Project MEMSHADOW is an advanced AI memory persistence platform",
        "metadata": {"project": "memshadow", "type": "definition", "version": "2"}
    }
    
    response = await async_client.patch(
        f"/api/v1/memory/{memory_ids[0]}",
        json=update_data,
        headers=headers
    )
    
    assert response.status_code == 200
    updated = response.json()
    assert "advanced" in updated["content"]
    assert updated["metadata"]["version"] == "2"
    
    # 6. Delete a memory
    response = await async_client.delete(
        f"/api/v1/memory/{memory_ids[2]}",
        headers=headers
    )
    
    assert response.status_code == 200
    
    # 7. Verify deletion
    response = await async_client.get(
        f"/api/v1/memory/{memory_ids[2]}",
        headers=headers
    )
    
    assert response.status_code == 404
```

---

## 8. Deployment & Operations

### 8.1 Docker Configuration

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/prod.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r prod.txt

FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY app/ ./app/
COPY alembic.ini .
COPY migrations/ ./migrations/

# Create non-root user
RUN useradd -m -u 1000 memshadow && chown -R memshadow:memshadow /app
USER memshadow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: memshadow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: memshadow
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U memshadow"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  chromadb:
    image: chromadb/chroma:latest
    environment:
      ANONYMIZED_TELEMETRY: "false"
      ALLOW_RESET: "false"
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    environment:
      DATABASE_URL: postgresql+asyncpg://memshadow:${POSTGRES_PASSWORD}@postgres:5432/memshadow
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      CHROMA_HOST: chromadb
      CHROMA_PORT: 8000
    ports:
      - "8001:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      chromadb:
        condition: service_started
    volumes:
      - ../logs:/app/logs

  worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    environment:
      DATABASE_URL: postgresql+asyncpg://memshadow:${POSTGRES_PASSWORD}@postgres:5432/memshadow
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      CHROMA_HOST: chromadb
      CHROMA_PORT: 8000
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      chromadb:
        condition: service_started
    command: celery -A app.workers.celery_app worker --loglevel=info

  flower:
    image: mher/flower:latest
    environment:
      CELERY_BROKER_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      FLOWER_PORT: 5555
    ports:
      - "5555:5555"
    depends_on:
      - redis

volumes:
  postgres_data:
  redis_data:
  chroma_data:
```

### 8.3 Production Deployment Checklist

```markdown
# MEMSHADOW Phase 1 Deployment Checklist

## Pre-Deployment
- [ ] All tests passing (unit, integration, e2e)
- [ ] Security scan completed (Bandit, pip-audit)
- [ ] Docker images built and scanned
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Database migrations tested

## Infrastructure
- [ ] PostgreSQL deployed with pgvector extension
- [ ] Redis deployed with authentication
- [ ] ChromaDB deployed and configured
- [ ] Load balancer configured
- [ ] Monitoring stack deployed (Prometheus, Grafana)
- [ ] Log aggregation configured

## Security
- [ ] Firewall rules configured
- [ ] Database access restricted
- [ ] API rate limiting enabled
- [ ] CORS properly configured
- [ ] Secrets stored in secure vault
- [ ] SDAP backup configured and tested

## Application
- [ ] API deployed with multiple replicas
- [ ] Celery workers deployed
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Error tracking enabled

## Post-Deployment
- [ ] Smoke tests executed
- [ ] Performance benchmarks run
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Incident response plan in place
```

### 8.4 Monitoring Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'memshadow-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

# Alerting rules
rule_files:
  - 'alerts.yml'

# alerts.yml
groups:
  - name: memshadow
    rules:
      - alert: HighMemoryIngestionLatency
        expr: histogram_quantile(0.95, memory_ingestion_duration_seconds_bucket) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory ingestion latency"
          description: "95th percentile latency is {{ $value }}s"

      - alert: LowEmbeddingCacheHitRate
        expr: embedding_cache_hit_rate < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low embedding cache hit rate"
          description: "Cache hit rate is {{ $value }}"

      - alert: ChimeraTriggered
        expr: increase(chimera_trigger_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "CHIMERA deception triggered"
          description: "{{ $value }} triggers in last 5 minutes"
```

---

## Summary

This Phase 1 implementation guide provides a solid foundation for Project MEMSHADOW with:

1. **Robust API Framework**: FastAPI with comprehensive middleware and security
2. **Scalable Data Layer**: PostgreSQL with pgvector, ChromaDB for embeddings, Redis for caching
3. **Secure Architecture**: MFA/A with FIDO2, behavioral biometrics foundation
4. **Reliable Operations**: SDAP backups, monitoring, comprehensive testing
5. **Production-Ready**: Docker deployment, health checks, operational procedures

The implementation focuses on establishing core functionality while maintaining flexibility for future enhancements in subsequent phases. All components are designed with security, scalability, and maintainability as primary concerns.
