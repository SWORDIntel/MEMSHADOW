import structlog
from typing import Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditEvent

# Configure a specific logger for audit events
audit_logger = structlog.get_logger("audit")

class AuditLogger:
    """
    Handles the creation and logging of audit events.
    """

    async def log_event(
        self,
        db: AsyncSession,
        event_type: str,
        action: str,
        outcome: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
    ):
        """
        Creates an audit event, saves it to the database, and logs it to a file.

        :param db: The SQLAlchemy async session.
        :param event_type: The category of the event (e.g., 'AUTH', 'ACCESS').
        :param action: The specific action performed (e.g., 'USER_LOGIN', 'CREATE_MEMORY').
        :param outcome: The result of the action (e.g., 'SUCCESS', 'FAILURE').
        :param user_id: The ID of the user performing the action.
        :param session_id: The session ID associated with the event.
        :param resource_type: The type of resource being acted upon (e.g., 'Memory').
        :param resource_id: The ID of the resource.
        :param ip_address: The source IP address of the request.
        :param user_agent: The user agent of the client.
        :param details: A JSONB field for any additional context.
        :param risk_score: A calculated risk score for the event.
        """
        # Create the database model instance
        db_event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome=outcome,
            details=details,
            risk_score=risk_score,
        )

        # Log to structured file logger
        audit_logger.info(
            action,
            event_type=event_type,
            user_id=str(user_id) if user_id else None,
            outcome=outcome,
            ip=ip_address,
        )

        # Save to database
        db.add(db_event)
        await db.commit()
        await db.refresh(db_event)

        # In a real-world scenario, you might also trigger alerts here
        # if db_event.is_security_relevant():
        #     await self._trigger_security_alert(db_event)

# Global instance to be used as a dependency
audit_log_service = AuditLogger()