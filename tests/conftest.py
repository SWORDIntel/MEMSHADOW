import sys
from unittest.mock import MagicMock

# Mock the heavy sentence_transformers library before it's imported by the app.
# This allows tests to run without needing torch/sentence-transformers installed,
# as the actual embedding service is mocked in the unit/integration tests anyway.
sys.modules['sentence_transformers'] = MagicMock()

import asyncio
from typing import AsyncGenerator, Generator

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.main import app
from app.db.postgres import Base, get_db
from app.db import chromadb
from app.core.config import settings
from app.workers.celery_app import celery_app as a_celery_app

@pytest.fixture(scope='session')
def celery_config():
    return {
        'broker_url': settings.CELERY_BROKER_URL,
        'result_backend': settings.CELERY_RESULT_BACKEND,
        'task_always_eager': False,
    }

@pytest.fixture(scope='session')
def celery_app(celery_config):
    a_celery_app.conf.update(celery_config)
    return a_celery_app

# Create a new async engine for the test database
engine_test = create_async_engine(str(settings.DATABASE_URL), poolclass=NullPool)
TestingSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine_test, expire_on_commit=False
)

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

from sqlalchemy import text

@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """
    Set up the test database for the session.
    Creates all tables before tests run and drops the schema after.
    """
    # Initialize ChromaDB client
    await chromadb.init_client()

    async with engine_test.begin() as conn:
        # Create extensions and tables
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Teardown: Drop the entire public schema and recreate it for a clean slate.
    async with engine_test.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE;"))
        await conn.execute(text("CREATE SCHEMA public;"))

    # Close ChromaDB client
    await chromadb.close_client()

async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Override dependency to use the test database session."""
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Fixture to get a test database session."""
    async with TestingSessionLocal() as session:
        yield session

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Fixture for an async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client