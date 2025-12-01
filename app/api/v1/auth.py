from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import structlog

from app.api.dependencies import get_db, get_current_active_user
from app.schemas.user import UserCreate, UserResponse
from app.schemas.auth import Token
from app.services.auth_service import AuthService
from app.services.mfa_service import MFAService
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.auth import WebAuthnRegistrationComplete, WebAuthnAuthenticationComplete

router = APIRouter()
logger = structlog.get_logger()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserCreate
) -> Any:
    """
    Register a new user.
    """
    auth_service = AuthService(db)
    try:
        user = await auth_service.create_user(
            email=user_in.email, username=user_in.username, password=user_in.password
        )
    except IntegrityError:
        await db.rollback()
        # This generic message is better as it doesn't reveal which field (email/username) is taken.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email or username already exists.",
        )

    logger.info("User registered", user_id=str(user.id))
    return user

@router.post("/login", response_model=Token)
async def login(
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    auth_service = AuthService(db)
    user = await auth_service.authenticate_user(
        username=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # If MFA is enabled, we just return a temporary token or a message
    if user.mfa_enabled:
        # For simplicity, we'll just indicate MFA is required.
        # A more robust solution would use a temporary token.
        raise HTTPException(status_code=403, detail="MFA required")

    access_token = create_access_token(
        subject=user.id
    )
    logger.info("User logged in", user_id=str(user.id))
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user

@router.post("/webauthn/register/begin", status_code=status.HTTP_200_OK)
async def webauthn_register_begin(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Begin WebAuthn registration"""
    mfa_service = MFAService(db)
    options = await mfa_service.begin_registration(
        user_id=str(current_user.id),
        username=current_user.username,
    )
    return options

@router.post("/webauthn/register/complete", status_code=status.HTTP_200_OK)
async def webauthn_register_complete(
    *,
    db: AsyncSession = Depends(get_db),
    credential_data: WebAuthnRegistrationComplete,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, str]:
    """Complete WebAuthn registration"""
    mfa_service = MFAService(db)
    try:
        credential = await mfa_service.complete_registration(
            user_id=str(current_user.id),
            response=credential_data.model_dump()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "success", "credential_id": credential.credential_id}

@router.post("/webauthn/authenticate/begin/{username}", status_code=status.HTTP_200_OK)
async def webauthn_authenticate_begin(
    *,
    username: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Begin WebAuthn authentication"""
    mfa_service = MFAService(db)
    try:
        options = await mfa_service.begin_authentication(username)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return options

@router.post("/webauthn/authenticate/complete", response_model=Token)
async def webauthn_authenticate_complete(
    *,
    db: AsyncSession = Depends(get_db),
    auth_data: WebAuthnAuthenticationComplete
) -> Token:
    """Complete WebAuthn authentication and return a JWT."""
    mfa_service = MFAService(db)
    try:
        user_id = await mfa_service.complete_authentication(
            username=auth_data.username,
            response=auth_data.model_dump()
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    access_token = create_access_token(subject=user_id)
    logger.info("User logged in with WebAuthn", user_id=user_id)
    return Token(access_token=access_token, token_type="bearer")