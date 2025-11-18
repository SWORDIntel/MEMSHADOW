import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.chimera import TriggerEvent as TriggerEventModel
from app.db.redis import redis_client

logger = structlog.get_logger()

# --- Trigger Event Data Model ---

class TriggerEventData(BaseModel):
    """
    Represents the data associated with a triggered lure.
    """
    lure_id: uuid.UUID
    trigger_type: str
    session_id: uuid.UUID
    source_ip: str
    user_agent: str
    trigger_metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

SeverityLevel = Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

# --- Trigger Handler ---

class TriggerHandler:
    """
    Handles events triggered by CHIMERA lures and orchestrates the response.
    """
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db

    def _assess_severity(self, trigger_event: TriggerEventData) -> SeverityLevel:
        """
        Assesses the severity of a trigger event based on its context.
        In a real system, this would use a sophisticated rules engine.
        """
        if "honeypot" in trigger_event.context.get("lure_type", ""):
            return "CRITICAL"
        if trigger_event.trigger_type == "export":
            return "HIGH"
        if trigger_event.trigger_type == "modify":
            return "HIGH"
        if trigger_event.trigger_type == "access":
            return "MEDIUM"
        return "LOW"

    async def _isolate_session(self, session_id: uuid.UUID):
        """Isolates a user session by adding to Redis denylist."""
        try:
            # Add session to denylist in Redis
            denylist_key = f"session:denylist:{session_id}"
            await redis_client.cache_set(denylist_key, "isolated", ttl=86400)  # 24 hours
            logger.warning("Session isolated due to security trigger", session_id=str(session_id))
        except Exception as e:
            logger.error("Failed to isolate session", session_id=str(session_id), error=str(e))

    async def _alert_security_team(self, trigger_event: TriggerEventData, severity: SeverityLevel):
        """Sends a detailed alert to the security team."""
        # In a production system, this would integrate with:
        # - Slack/Discord webhooks
        # - PagerDuty
        # - Email alerts
        # - SIEM systems
        logger.critical(
            "CHIMERA TRIGGER ALERT",
            severity=severity,
            lure_id=str(trigger_event.lure_id),
            session_id=str(trigger_event.session_id),
            source_ip=trigger_event.source_ip,
            trigger_type=trigger_event.trigger_type
        )

    async def _initiate_forensics(self, trigger_event: TriggerEventData):
        """Kicks off a forensic data collection process for the session."""
        try:
            # Mark session for forensic analysis
            forensics_key = f"forensics:session:{trigger_event.session_id}"
            forensics_data = {
                "lure_id": str(trigger_event.lure_id),
                "trigger_time": datetime.utcnow().isoformat(),
                "source_ip": trigger_event.source_ip,
                "user_agent": trigger_event.user_agent
            }
            await redis_client.cache_set(forensics_key, forensics_data, ttl=604800)  # 7 days
            logger.info("Forensics initiated", session_id=str(trigger_event.session_id))
        except Exception as e:
            logger.error("Failed to initiate forensics", error=str(e))

    async def _enable_session_recording(self, session_id: uuid.UUID):
        """Enables enhanced, detailed logging for a suspicious session."""
        try:
            recording_key = f"session:recording:{session_id}"
            await redis_client.cache_set(recording_key, "enabled", ttl=3600)  # 1 hour
            logger.info("Enhanced session recording enabled", session_id=str(session_id))
        except Exception as e:
            logger.error("Failed to enable session recording", error=str(e))

    async def _deploy_additional_lures(self, context: Dict[str, Any]):
        """Deploys more lures to gather further intelligence on the actor."""
        # This would integrate with ChimeraEngine to deploy contextual lures
        logger.info("Additional lure deployment requested", context=context)
        # TODO: Integrate with ChimeraEngine to actually deploy lures

    async def _log_trigger_event_to_db(self, trigger_event: TriggerEventData, severity: SeverityLevel):
        """
        Logs the complete trigger event to the chimera_deception.trigger_events table.
        """
        if not self.db:
            logger.warning("No database session provided, logging to file only")
            return

        try:
            event_db = TriggerEventModel(
                lure_id=trigger_event.lure_id,
                trigger_type=trigger_event.trigger_type,
                session_id=trigger_event.session_id,
                source_ip=trigger_event.source_ip,
                user_agent=trigger_event.user_agent,
                trigger_metadata=trigger_event.trigger_metadata,
                severity=severity,
                handled=False
            )

            self.db.add(event_db)
            await self.db.commit()

            logger.info("Trigger event logged to database", event_id=str(event_db.id))
        except Exception as e:
            logger.error("Failed to log trigger event to database", error=str(e))
            await self.db.rollback()

    async def handle_trigger(self, trigger_event: TriggerEventData):
        """
        The main entry point for handling a trigger. It assesses severity and
        orchestrates the appropriate defensive and logging actions.
        """
        severity = self._assess_severity(trigger_event)

        logger.warning(
            "CHIMERA trigger activated",
            lure_id=str(trigger_event.lure_id),
            severity=severity,
            trigger_type=trigger_event.trigger_type
        )

        # Always log the event to the database for a complete audit trail.
        await self._log_trigger_event_to_db(trigger_event, severity)

        if severity == "CRITICAL":
            await self._alert_security_team(trigger_event, severity)
            await self._isolate_session(trigger_event.session_id)
            await self._initiate_forensics(trigger_event)
        elif severity == "HIGH":
            await self._alert_security_team(trigger_event, severity)
            await self._enable_session_recording(trigger_event.session_id)
            await self._deploy_additional_lures(trigger_event.context)
        elif severity == "MEDIUM":
            await self._alert_security_team(trigger_event, severity)
        # For INFO or LOW severity, logging to the DB is sufficient
