from pydantic import BaseModel, UUID4
from typing import Optional, Dict, Any

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenPayload(BaseModel):
    sub: Optional[UUID4] = None

# Schemas for WebAuthn, will be used in a later step
class WebAuthnRegistrationComplete(BaseModel):
    id: str
    rawId: str
    response: Dict[str, Any]
    type: str

class WebAuthnAuthenticationComplete(BaseModel):
    id: str
    rawId: str
    response: Dict[str, Any]
    type: str
    username: str