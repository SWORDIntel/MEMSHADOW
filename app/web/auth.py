"""
Authentication and Authorization
JWT-based authentication for MEMSHADOW web interface
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from passlib.context import CryptContext
import structlog

logger = structlog.get_logger()

# Load from environment variables
SECRET_KEY = os.getenv("WEB_SECRET_KEY", "INSECURE_DEFAULT_CHANGE_ME_IN_PRODUCTION")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
TOKEN_EXPIRY_HOURS = int(os.getenv("WEB_TOKEN_EXPIRY_HOURS", "24"))

# Password hashing context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

security = HTTPBearer()

# Warn if using default secret
if SECRET_KEY == "INSECURE_DEFAULT_CHANGE_ME_IN_PRODUCTION":
    logger.critical(
        "SECURITY WARNING: Using default SECRET_KEY! Set WEB_SECRET_KEY environment variable!"
    )


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
    Verify password against hash using bcrypt.

    Args:
        plain_password: Plain text password
        hashed_password: Bcrypt hashed password

    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Bcrypt hashed password
    """
    return pwd_context.hash(password)


class User:
    """User model"""

    def __init__(self, username: str, hashed_password: str, role: str = "user"):
        self.username = username
        self.hashed_password = hashed_password
        self.role = role


# In-memory user database (TODO: Replace with proper database)
# Load admin credentials from environment
_admin_username = os.getenv("WEB_ADMIN_USERNAME", "admin")
_admin_password = os.getenv("WEB_ADMIN_PASSWORD", "admin")

# Hash the admin password if it's not already hashed
# Bcrypt hashes start with $2b$ or $2a$
if not _admin_password.startswith("$2"):
    logger.warning(
        "Admin password not hashed in environment, hashing now. "
        "For better security, store pre-hashed passwords."
    )
    _admin_password = get_password_hash(_admin_password)

USERS_DB: Dict[str, User] = {
    _admin_username: User(_admin_username, _admin_password, role="admin"),
}

# Warn if using default credentials
if _admin_username == "admin" and os.getenv("WEB_ADMIN_PASSWORD", "admin") == "admin":
    logger.critical(
        "SECURITY WARNING: Using default admin credentials! "
        "Set WEB_ADMIN_USERNAME and WEB_ADMIN_PASSWORD environment variables!"
    )


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
