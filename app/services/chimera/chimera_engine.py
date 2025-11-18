import uuid
from datetime import datetime
from typing import Dict, Any, Protocol, Optional
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
import structlog

from app.models.chimera import Lure as LureModel
from app.core.encryption import field_encryption

logger = structlog.get_logger()

# --- Lure Data Models ---

class CanaryToken(BaseModel):
    id: str = Field(default_factory=lambda: f"canary_{uuid.uuid4()}")
    trigger_urls: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HoneypotMemory(BaseModel):
    id: str = Field(default_factory=lambda: f"honeypot_{uuid.uuid4()}")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    triggers: list[str] = Field(default_factory=list)

Lure = CanaryToken | HoneypotMemory

# --- Lure Generator Protocols ---

class LureGenerator(Protocol):
    async def generate(self, context: Dict[str, Any]) -> Lure:
        ...

# --- Lure Generator Implementations ---

class CanaryTokenGenerator:
    """Generates canary tokens to detect unauthorized access."""
    async def generate(self, context: Dict[str, Any]) -> CanaryToken:
        token_id = uuid.uuid4()
        token = CanaryToken(
            trigger_urls=[
                f"/api/v1/memory/{uuid.uuid4()}",  # Fake memory endpoint
                f"/api/v1/internal/debug/{token_id}" # Fake debug endpoint
            ],
            metadata={
                "deployed_at": datetime.utcnow().isoformat(),
                "context": context,
                "trigger_action": "alert_and_trace"
            }
        )
        return token

class HoneypotMemoryGenerator:
    """Generates believable but fake sensitive content to attract attackers."""
    async def _generate_believable_content(self, context: Dict[str, Any]) -> str:
        # In a real system, this could use an LLM to generate fake data.
        theme = context.get("theme", "generic secret")
        return f"--- FAKE CONFIDENTIAL DATA ---\nTheme: {theme}\nAPI_KEY: fk_{uuid.uuid4()}\n--- END FAKE DATA ---"

    async def generate(self, context: Dict[str, Any]) -> HoneypotMemory:
        fake_content = await self._generate_believable_content(context)
        return HoneypotMemory(
            content=fake_content,
            metadata={
                "tags": ["confidential", "api_keys", "passwords"],
                "access_pattern": "high_value_target",
                "context": context,
                "deployed_at": datetime.utcnow().isoformat()
            },
            triggers=["access", "export", "modify"]
        )

# --- Core Deception Engine ---

class ChimeraEngine:
    """
    The core engine for deploying and managing deception lures.
    """
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        self.lure_generators: Dict[str, LureGenerator] = {
            'canary_token': CanaryTokenGenerator(),
            'honeypot_memory': HoneypotMemoryGenerator(),
        }

    async def deploy_lure(self, lure_type: str, context: Dict[str, Any]) -> Lure:
        """
        Deploys a deception lure of a specified type into the system.
        """
        if lure_type not in self.lure_generators:
            raise ValueError(f"Unknown lure type: {lure_type}")

        generator = self.lure_generators[lure_type]
        lure = await generator.generate(context)

        # Store the lure in the chimera_deception database
        if self.db:
            await self._store_lure_secure(lure)

        # Configure monitoring
        await self._configure_triggers(lure)

        logger.info("CHIMERA lure deployed", lure_id=lure.id, lure_type=lure_type)
        return lure

    async def _store_lure_secure(self, lure: Lure):
        """
        Stores the lure's details in the segregated CHIMERA database.
        Content is encrypted before storage.
        """
        if not self.db:
            logger.warning("No database session provided, skipping lure storage")
            return

        try:
            # Serialize lure content
            lure_content = json.dumps(lure.model_dump())

            # Encrypt the content
            encrypted_content = field_encryption.encrypt_field(lure_content).encode()

            # Create database model
            lure_db = LureModel(
                id=uuid.UUID(lure.id.split('_')[1]) if '_' in lure.id else uuid.uuid4(),
                lure_type=type(lure).__name__,
                encrypted_content=encrypted_content,
                deployment_metadata=lure.metadata,
                is_active=True
            )

            self.db.add(lure_db)
            await self.db.commit()

            logger.info("Lure stored in database", lure_id=str(lure_db.id))
        except Exception as e:
            logger.error("Failed to store lure", error=str(e), exc_info=True)
            await self.db.rollback()
            raise

    async def _configure_triggers(self, lure: Lure):
        """
        Configures the necessary monitoring to detect interaction with the lure.
        """
        # In a real implementation:
        # - For a canary token, this might involve setting up specific API route handlers
        # - For a honeypot memory, it might involve adding its ID to a Redis watch list
        logger.debug("Configuring triggers for lure", lure_id=lure.id)
        # This is intentionally minimal for now
        pass

    async def get_active_lures(self) -> list[LureModel]:
        """Retrieve all active lures from the database"""
        if not self.db:
            logger.warning("No database session provided")
            return []

        stmt = select(LureModel).where(LureModel.is_active == True)
        result = await self.db.execute(stmt)
        lures = result.scalars().all()

        return list(lures)

    async def deactivate_lure(self, lure_id: uuid.UUID) -> bool:
        """Deactivate a specific lure"""
        if not self.db:
            logger.warning("No database session provided")
            return False

        stmt = select(LureModel).where(LureModel.id == lure_id)
        result = await self.db.execute(stmt)
        lure = result.scalar_one_or_none()

        if not lure:
            return False

        lure.is_active = False
        await self.db.commit()

        logger.info("Lure deactivated", lure_id=str(lure_id))
        return True