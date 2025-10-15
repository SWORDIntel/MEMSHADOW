import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User

@pytest.mark.asyncio
async def test_register_new_user(async_client: AsyncClient, db_session: AsyncSession):
    # Arrange
    register_data = {
        "email": "testuser1@example.com",
        "username": "testuser1",
        "password": "short_pass"
    }

    # Act
    response = await async_client.post("/api/v1/auth/register", json=register_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == register_data["email"]
    assert data["username"] == register_data["username"]
    assert "id" in data

    # Verify user was actually created in the DB
    user_in_db = await db_session.get(User, data["id"])
    assert user_in_db is not None
    assert user_in_db.email == register_data["email"]

@pytest.mark.asyncio
async def test_register_duplicate_user_error(async_client: AsyncClient, db_session: AsyncSession):
    # Arrange: First, create a user
    register_data = {
        "email": "duplicate2@example.com",
        "username": "duplicateuser2",
        "password": "short_pass"
    }
    await async_client.post("/api/v1/auth/register", json=register_data)

    # Act: Try to register the same user again
    response = await async_client.post("/api/v1/auth/register", json=register_data)

    # Assert
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

@pytest.mark.asyncio
async def test_login_success(async_client: AsyncClient, db_session: AsyncSession):
    # Arrange: Create a user to login with
    username = "loginuser3"
    password = "short_pass"
    register_data = {
        "email": "login3@example.com",
        "username": username,
        "password": password
    }
    await async_client.post("/api/v1/auth/register", json=register_data)

    login_data = {
        "username": username,
        "password": password,
    }

    # Act
    response = await async_client.post(
        "/api/v1/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_failure_wrong_password(async_client: AsyncClient, db_session: AsyncSession):
    # Arrange: Create a user
    username = "wrongpassuser4"
    password = "short_pass"
    register_data = {
        "email": "wrongpass4@example.com",
        "username": username,
        "password": password
    }
    await async_client.post("/api/v1/auth/register", json=register_data)

    login_data = {
        "username": username,
        "password": "wrong_password",
    }

    # Act
    response = await async_client.post(
        "/api/v1/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Assert
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_current_user(async_client: AsyncClient, db_session: AsyncSession):
    # Arrange: Register and login to get a token
    username = "me_user5"
    password = "short_pass"
    register_data = {
        "email": "me5@example.com",
        "username": username,
        "password": password
    }
    await async_client.post("/api/v1/auth/register", json=register_data)

    login_response = await async_client.post(
        "/api/v1/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Act
    response = await async_client.get("/api/v1/auth/me", headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == username
    assert data["email"] == "me5@example.com"