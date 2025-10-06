from typing import Optional, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.models.auth import WebAuthnCredential
from app.schemas.user import UserCreate

class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_user(self, email: str, username: str, password: str) -> User:
        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = await self.get_user_by_username(username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def get_user_credentials(self, user_id: UUID) -> List[WebAuthnCredential]:
        result = await self.db.execute(
            select(WebAuthnCredential).where(WebAuthnCredential.user_id == user_id)
        )
        return result.scalars().all()

    async def save_credential(self, credential: WebAuthnCredential) -> None:
        self.db.add(credential)
        await self.db.commit()

    async def enable_mfa(self, user_id: UUID) -> None:
        user = await self.db.get(User, user_id)
        if user:
            user.mfa_enabled = True
            await self.db.commit()

    async def get_credential_by_id(self, credential_id: str) -> Optional[WebAuthnCredential]:
        result = await self.db.execute(
            select(WebAuthnCredential).where(WebAuthnCredential.credential_id == credential_id)
        )
        return result.scalar_one_or_none()

    async def update_credential_sign_count(self, credential_id: str, sign_count: int) -> None:
        credential = await self.get_credential_by_id(credential_id)
        if credential:
            credential.sign_count = sign_count
            await self.db.commit()