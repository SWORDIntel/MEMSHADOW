import uuid
from sqlalchemy import Column, String, Integer, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
from app.db.postgres import Base

class WebAuthnCredential(Base):
    __tablename__ = "webauthn_credentials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    credential_id = Column(String, unique=True, nullable=False)
    public_key = Column(LargeBinary, nullable=False)
    sign_count = Column(Integer, default=0)
    aaguid = Column(String)
    fmt = Column(String)