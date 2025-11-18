import pytest
from uuid import uuid4
from app.services.chimera.chimera_engine import ChimeraEngine, CanaryToken, HoneypotMemory
from app.services.chimera.trigger_handler import TriggerHandler, TriggerEventData
from app.models.chimera import Lure as LureModel, TriggerEvent
from sqlalchemy import select

@pytest.mark.asyncio
class TestChimeraEngine:
    """Test CHIMERA deception engine"""

    async def test_deploy_canary_token(self, db_session):
        """Test deploying a canary token lure"""
        engine = ChimeraEngine(db=db_session)
        
        lure = await engine.deploy_lure(
            "canary_token",
            {"theme": "api_endpoints"}
        )
        
        assert lure is not None
        assert isinstance(lure, CanaryToken)
        assert len(lure.trigger_urls) > 0
        assert "deployed_at" in lure.metadata

    async def test_deploy_honeypot_memory(self, db_session):
        """Test deploying a honeypot memory lure"""
        engine = ChimeraEngine(db=db_session)
        
        lure = await engine.deploy_lure(
            "honeypot_memory",
            {"theme": "api_keys"}
        )
        
        assert lure is not None
        assert isinstance(lure, HoneypotMemory)
        assert "FAKE" in lure.content
        assert "triggers" in lure.model_dump()

    async def test_lure_database_storage(self, db_session):
        """Test that lures are stored in database with encryption"""
        engine = ChimeraEngine(db=db_session)
        
        await engine.deploy_lure(
            "canary_token",
            {"theme": "test"}
        )
        
        # Query the chimera schema
        stmt = select(LureModel).where(LureModel.is_active == True)
        result = await db_session.execute(stmt)
        lures = result.scalars().all()
        
        assert len(lures) > 0
        lure = lures[0]
        assert lure.encrypted_content is not None
        assert lure.lure_type in ["CanaryToken", "HoneypotMemory"]

    async def test_get_active_lures(self, db_session):
        """Test retrieving active lures"""
        engine = ChimeraEngine(db=db_session)
        
        # Deploy multiple lures
        await engine.deploy_lure("canary_token", {"theme": "test1"})
        await engine.deploy_lure("honeypot_memory", {"theme": "test2"})
        
        # Retrieve active lures
        active_lures = await engine.get_active_lures()
        
        assert len(active_lures) >= 2

    async def test_deactivate_lure(self, db_session):
        """Test deactivating a lure"""
        engine = ChimeraEngine(db=db_session)
        
        # Deploy a lure
        await engine.deploy_lure("canary_token", {"theme": "test"})
        
        # Get the lure
        active_lures = await engine.get_active_lures()
        assert len(active_lures) > 0
        
        lure_id = active_lures[0].id
        
        # Deactivate it
        success = await engine.deactivate_lure(lure_id)
        assert success is True
        
        # Verify it's deactivated
        stmt = select(LureModel).where(LureModel.id == lure_id)
        result = await db_session.execute(stmt)
        lure = result.scalar_one()
        assert lure.is_active is False


@pytest.mark.asyncio
class TestTriggerHandler:
    """Test CHIMERA trigger handler"""

    async def test_assess_severity_critical(self):
        """Test severity assessment for CRITICAL events"""
        handler = TriggerHandler()
        
        event = TriggerEventData(
            lure_id=uuid4(),
            trigger_type="access",
            session_id=uuid4(),
            source_ip="192.168.1.1",
            user_agent="TestAgent",
            context={"lure_type": "honeypot_memory"}
        )
        
        severity = handler._assess_severity(event)
        assert severity == "CRITICAL"

    async def test_assess_severity_high(self):
        """Test severity assessment for HIGH events"""
        handler = TriggerHandler()
        
        event = TriggerEventData(
            lure_id=uuid4(),
            trigger_type="export",
            session_id=uuid4(),
            source_ip="192.168.1.1",
            user_agent="TestAgent",
            context={}
        )
        
        severity = handler._assess_severity(event)
        assert severity == "HIGH"

    async def test_log_trigger_event(self, db_session):
        """Test logging trigger events to database"""
        handler = TriggerHandler(db=db_session)
        
        event = TriggerEventData(
            lure_id=uuid4(),
            trigger_type="access",
            session_id=uuid4(),
            source_ip="192.168.1.1",
            user_agent="TestAgent",
            trigger_metadata={"test": "data"},
            context={}
        )
        
        await handler._log_trigger_event_to_db(event, "MEDIUM")
        
        # Verify event was logged
        stmt = select(TriggerEvent)
        result = await db_session.execute(stmt)
        events = result.scalars().all()
        
        assert len(events) > 0
        logged_event = events[0]
        assert logged_event.severity == "MEDIUM"
        assert logged_event.trigger_type == "access"

    async def test_handle_trigger_workflow(self, db_session):
        """Test complete trigger handling workflow"""
        handler = TriggerHandler(db=db_session)
        
        event = TriggerEventData(
            lure_id=uuid4(),
            trigger_type="modify",
            session_id=uuid4(),
            source_ip="192.168.1.1",
            user_agent="TestAgent",
            context={}
        )
        
        # Handle the trigger
        await handler.handle_trigger(event)
        
        # Verify event was logged
        stmt = select(TriggerEvent).where(
            TriggerEvent.lure_id == event.lure_id
        )
        result = await db_session.execute(stmt)
        logged_event = result.scalar_one_or_none()
        
        assert logged_event is not None
        assert logged_event.severity in ["HIGH", "CRITICAL"]
