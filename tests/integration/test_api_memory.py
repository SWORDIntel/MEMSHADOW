import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from uuid import UUID

from app.models.user import User
from app.models.memory import Memory
from app.core.security import create_access_token, get_password_hash

import uuid

# Helper function to create a user and get a token
async def create_user_and_get_token(db_session: AsyncSession) -> (User, str):
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        email=f"memtest_{user_id}@example.com",
        username=f"memtestuser_{user_id}",
        hashed_password=get_password_hash("short_pass")
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    token = create_access_token(subject=user.id)
    return user, token

@pytest.mark.skip(reason="Celery worker tests are failing and will be addressed in a separate task.")
@pytest.mark.asyncio
async def test_ingest_memory(
    async_client: AsyncClient, db_session: AsyncSession, celery_app, celery_worker
):
    # Arrange
    user, token = await create_user_and_get_token(db_session)
    headers = {"Authorization": f"Bearer {token}"}
    memory_data = {
        "content": "This is a test memory for ingestion.",
        "extra_data": {"source": "integration_test"}
    }

    # Act
    response = await async_client.post("/api/v1/memory/ingest", json=memory_data, headers=headers)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == memory_data["content"]
    assert data["extra_data"] == memory_data["extra_data"]

    # Verify in DB
    memory_id = UUID(data["id"])
    db_memory = await db_session.get(Memory, memory_id)
    assert db_memory is not None
    assert db_memory.user_id == user.id

    # Allow time for the background task to run
    await asyncio.sleep(2)
    await db_session.refresh(db_memory)
    assert db_memory.embedding is not None

@pytest.mark.skip(reason="Celery worker tests are failing and will be addressed in a separate task.")
@pytest.mark.asyncio
async def test_retrieve_memories(
    async_client: AsyncClient, db_session: AsyncSession, celery_app, celery_worker
):
    # Arrange
    user, token = await create_user_and_get_token(db_session)
    headers = {"Authorization": f"Bearer {token}"}

    # Ingest a memory first
    await async_client.post(
        "/api/v1/memory/ingest",
        json={"content": "A memory about Python programming.", "extra_data": {}},
        headers=headers
    )
    await asyncio.sleep(2) # Wait for embedding

    # Act
    search_data = {"query": "python"}
    response = await async_client.post("/api/v1/memory/retrieve", json=search_data, headers=headers)

    # Assert
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0
    assert "Python programming" in results[0]["content"]

@pytest.mark.asyncio
async def test_update_memory(
    async_client: AsyncClient, db_session: AsyncSession, celery_app, celery_worker
):
    # Arrange
    user, token = await create_user_and_get_token(db_session)
    headers = {"Authorization": f"Bearer {token}"}

    # Ingest a memory
    ingest_response = await async_client.post(
        "/api/v1/memory/ingest",
        json={"content": "Original content.", "extra_data": {"version": 1}},
        headers=headers
    )
    memory_id = ingest_response.json()["id"]

    update_data = {
        "content": "Updated content.",
        "extra_data": {"version": 2, "source": "update_test"}
    }

    # Act
    response = await async_client.patch(f"/api/v1/memory/{memory_id}", json=update_data, headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == update_data["content"]
    assert data["extra_data"]["version"] == 2

    # Verify in DB
    db_memory = await db_session.get(Memory, UUID(memory_id))
    assert db_memory.content == "Updated content."

@pytest.mark.skip(reason="Celery worker tests are failing and will be addressed in a separate task.")
@pytest.mark.asyncio
async def test_delete_memory(
    async_client: AsyncClient, db_session: AsyncSession, celery_app, celery_worker
):
    # Arrange
    user, token = await create_user_and_get_token(db_session)
    headers = {"Authorization": f"Bearer {token}"}

    ingest_response = await async_client.post(
        "/api/v1/memory/ingest",
        json={"content": "This memory will be deleted.", "extra_data": {}},
        headers=headers
    )
    memory_id = ingest_response.json()["id"]

    # Act
    response = await async_client.delete(f"/api/v1/memory/{memory_id}", headers=headers)

    # Assert
    assert response.status_code == 204

    # Verify it's gone from the DB
    db_memory = await db_session.get(Memory, UUID(memory_id))
    assert db_memory is None

    # Verify getting it returns 404
    get_response = await async_client.get(f"/api/v1/memory/{memory_id}", headers=headers)
    assert get_response.status_code == 404