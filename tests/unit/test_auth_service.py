import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.auth_service import AuthService
from app.models.user import User

@pytest.fixture
def mock_db_session():
    """Fixture for a mocked async database session."""
    return AsyncMock(spec=AsyncSession)

@pytest.fixture
def auth_service(mock_db_session):
    """Fixture for AuthService with a mocked database session."""
    return AuthService(mock_db_session)

@pytest.mark.asyncio
async def test_get_user_by_email(auth_service, mock_db_session):
    # Arrange
    test_email = "test@example.com"
    mock_user = User(id=uuid4(), email=test_email, username="testuser", hashed_password="hashedpassword")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db_session.execute.return_value = mock_result

    # Act
    user = await auth_service.get_user_by_email(test_email)

    # Assert
    assert user is not None
    assert user.email == test_email
    mock_db_session.execute.assert_called_once()

@pytest.mark.asyncio
async def test_create_user(auth_service, mock_db_session):
    # Arrange
    test_email = "newuser@example.com"
    test_username = "newuser"
    test_password = "password123"

    # Act
    with patch("app.services.auth_service.get_password_hash") as mock_get_hash:
        mock_get_hash.return_value = "hashed_password_abc"
        new_user = await auth_service.create_user(
            email=test_email, username=test_username, password=test_password
        )

    # Assert
    assert new_user.email == test_email
    assert new_user.username == test_username
    assert new_user.hashed_password == "hashed_password_abc"
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

@pytest.mark.asyncio
async def test_authenticate_user_success(auth_service, mock_db_session):
    # Arrange
    test_username = "authuser"
    test_password = "correctpassword"
    hashed_password = "correct_hashed_password"
    mock_user = User(id=uuid4(), username=test_username, hashed_password=hashed_password)

    # Mock get_user_by_username to return our mock user
    auth_service.get_user_by_username = AsyncMock(return_value=mock_user)

    # Act
    with patch("app.services.auth_service.verify_password") as mock_verify_password:
        mock_verify_password.return_value = True
        authenticated_user = await auth_service.authenticate_user(
            username=test_username, password=test_password
        )

    # Assert
    assert authenticated_user is not None
    assert authenticated_user.username == test_username
    auth_service.get_user_by_username.assert_called_once_with(test_username)
    mock_verify_password.assert_called_once_with(test_password, hashed_password)

@pytest.mark.asyncio
async def test_authenticate_user_failure_wrong_password(auth_service, mock_db_session):
    # Arrange
    test_username = "authuser"
    test_password = "wrongpassword"
    hashed_password = "correct_hashed_password"
    mock_user = User(id=uuid4(), username=test_username, hashed_password=hashed_password)

    auth_service.get_user_by_username = AsyncMock(return_value=mock_user)

    # Act
    with patch("app.services.auth_service.verify_password") as mock_verify_password:
        mock_verify_password.return_value = False
        authenticated_user = await auth_service.authenticate_user(
            username=test_username, password=test_password
        )

    # Assert
    assert authenticated_user is None
    auth_service.get_user_by_username.assert_called_once_with(test_username)
    mock_verify_password.assert_called_once_with(test_password, hashed_password)

@pytest.mark.asyncio
async def test_authenticate_user_failure_no_user(auth_service, mock_db_session):
    # Arrange
    test_username = "nonexistentuser"
    test_password = "anypassword"

    auth_service.get_user_by_username = AsyncMock(return_value=None)

    # Act
    authenticated_user = await auth_service.authenticate_user(
        username=test_username, password=test_password
    )

    # Assert
    assert authenticated_user is None
    auth_service.get_user_by_username.assert_called_once_with(test_username)