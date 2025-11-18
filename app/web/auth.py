"""
Authentication and Authorization
JWT-based authentication for MEMSHADOW web interface
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import structlog

logger = structlog.get_logger()

# TODO: Load from config
SECRET_KEY = "CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET"
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

security = HTTPBearer()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiry duration

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    logger.info("Access token created", subject=data.get("sub"))

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except jwt.JWTError as e:
        logger.warning("Invalid token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authenticate_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    Authenticate user from Bearer token.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    return payload


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.

    TODO: Implement proper password hashing (bcrypt/argon2)

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    # TODO: Use proper password hashing
    # For demo purposes only - DO NOT use in production!
    return plain_password == hashed_password


def get_password_hash(password: str) -> str:
    """
    Hash password.

    TODO: Implement proper password hashing (bcrypt/argon2)

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    # TODO: Use proper password hashing
    # For demo purposes only - DO NOT use in production!
    return password


class User:
    """User model"""

    def __init__(self, username: str, hashed_password: str, role: str = "user"):
        self.username = username
        self.hashed_password = hashed_password
        self.role = role


# In-memory user database (TODO: Replace with proper database)
USERS_DB: Dict[str, User] = {
    "admin": User("admin", "admin", role="admin"),  # CHANGE IN PRODUCTION!
    "user": User("user", "user", role="user")  # CHANGE IN PRODUCTION!
}


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username/password.

    Args:
        username: Username
        password: Password

    Returns:
        User object if authenticated, None otherwise
    """
    user = USERS_DB.get(username)

    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    logger.info("User authenticated", username=username)

    return user
