from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel, UUID4
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.core.config import settings
from app.db.postgres import get_db
from app.models.user import User
from app.services.auth_service import AuthService
from app.schemas.auth import TokenPayload

logger = structlog.get_logger()

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

async def get_current_user(
    db: AsyncSession = Depends(get_db), token: str = Depends(reusable_oauth2)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(sub=payload.get("sub"))
    except (JWTError, ValueError):
        logger.warning("JWTError or ValueError while decoding token")
        raise credentials_exception

    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(user_id=token_data.sub)
    if user is None:
        logger.warning("User not found from token", user_id=token_data.sub)
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="The user doesn't have enough privileges"
        )
    return current_user