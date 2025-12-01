from typing import Dict, Any, Optional, List
from fido2.server import Fido2Server
from fido2.webauthn import (
    PublicKeyCredentialRpEntity,
    PublicKeyCredentialUserEntity,
    AttestedCredentialData,
)
from fido2.utils import websafe_decode, websafe_encode
import structlog

from app.core.config import settings
from app.models.auth import WebAuthnCredential
from app.db.redis import redis_client
from app.services.auth_service import AuthService

logger = structlog.get_logger()

class MFAService:
    def __init__(self, db_session):
        rp = PublicKeyCredentialRpEntity(settings.FIDO2_RP_ID, settings.FIDO2_RP_NAME)
        self.server = Fido2Server(rp)
        self.auth_service = AuthService(db_session)

    async def begin_registration(
        self, user_id: str, username: str
    ) -> Dict[str, Any]:
        user_entity = PublicKeyCredentialUserEntity(
            id=user_id.encode(), name=username, display_name=username
        )

        existing_credentials = await self.auth_service.get_user_credentials(user_id)
        exclude_credentials = [
            {"type": "public-key", "id": websafe_decode(cred.credential_id)}
            for cred in existing_credentials
        ]

        options, state = self.server.register_begin(
            user=user_entity,
            credentials=exclude_credentials,
            user_verification="preferred",
            authenticator_attachment="cross-platform",
        )

        await self._store_challenge_state(user_id, "registration", state)
        return dict(options)

    async def complete_registration(
        self, user_id: str, response: Dict[str, Any]
    ) -> WebAuthnCredential:
        state = await self._get_challenge_state(user_id, "registration")
        if not state:
            raise ValueError("Registration state not found or expired.")

        auth_data = self.server.register_complete(
            state,
            websafe_decode(response["response"]["clientDataJSON"]),
            websafe_decode(response["response"]["attestationObject"]),
        )

        credential_record = WebAuthnCredential(
            user_id=user_id,
            credential_id=websafe_encode(auth_data.credential_data.credential_id),
            public_key=auth_data.credential_data.public_key,
            sign_count=auth_data.counter,
            aaguid=auth_data.credential_data.aaguid.hex() if auth_data.credential_data.aaguid else None,
        )

        await self.auth_service.save_credential(credential_record)
        await self.auth_service.enable_mfa(user_id)

        logger.info("FIDO2 registration completed", user_id=user_id)
        return credential_record

    async def begin_authentication(self, username: str) -> Dict[str, Any]:
        user = await self.auth_service.get_user_by_username(username)
        if not user:
            raise ValueError("User not found")

        credentials = await self.auth_service.get_user_credentials(user.id)
        if not credentials:
            raise ValueError("No credentials found for user")

        allowed_credentials = [
             {"type": "public-key", "id": websafe_decode(cred.credential_id)}
             for cred in credentials
        ]

        options, state = self.server.authenticate_begin(
            credentials=allowed_credentials, user_verification="preferred"
        )

        await self._store_challenge_state(user.id, "authentication", state)
        return dict(options)

    async def complete_authentication(
        self, username: str, response: Dict[str, Any]
    ) -> str:
        user = await self.auth_service.get_user_by_username(username)
        if not user:
            raise ValueError("User not found")

        state = await self._get_challenge_state(user.id, "authentication")
        if not state:
            raise ValueError("Authentication state not found or expired.")

        db_credentials = await self.auth_service.get_user_credentials(user.id)

        stored_credentials = []
        for db_cred in db_credentials:
            stored_credentials.append(
                AttestedCredentialData.create(
                    websafe_decode(db_cred.credential_id), db_cred.public_key
                )
            )

        auth_data = self.server.authenticate_complete(
            state,
            stored_credentials,
            websafe_decode(response["rawId"]),
            websafe_decode(response["response"]["clientDataJSON"]),
            websafe_decode(response["response"]["authenticatorData"]),
            websafe_decode(response["response"]["signature"]),
        )

        credential_to_update = await self.auth_service.get_credential_by_id(response['rawId'])
        if credential_to_update:
            await self.auth_service.update_credential_sign_count(credential_to_update.id, auth_data.counter)

        logger.info("FIDO2 authentication completed", user_id=str(user.id))
        return str(user.id)

    async def _store_challenge_state(self, user_id: str, operation: str, state: Any):
        key = f"fido2:{operation}:{user_id}"
        await redis_client.cache_set(key, state, ttl=300)

    async def _get_challenge_state(self, user_id: str, operation: str) -> Optional[Any]:
        key = f"fido2:{operation}:{user_id}"
        state = await redis_client.cache_get(key)
        if state:
            await redis_client.cache_delete(key)
            return state
        return None