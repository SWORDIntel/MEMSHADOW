"""
Unit tests for Workflow Engine
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

import sys
sys.path.insert(0, '/home/user/MEMSHADOW')

from app.services.workflow_engine import (
    WorkflowEngine,
    EnumeratePhase,
    PlanPhase,
    ExecutePhase
)
from app.services.swarm.blackboard import Blackboard


@pytest.fixture
def mock_blackboard():
    """Create mock blackboard"""
    with patch('app.services.workflow_engine.Blackboard') as mock:
        blackboard = Mock()
        blackboard.get_shared_state = Mock(return_value=[])
        blackboard.update_shared_state = Mock()
        blackboard.publish_task = Mock()
        blackboard.get_report = Mock(return_value=None)
        blackboard.get_list = Mock(return_value=[])
        blackboard.append_to_list = Mock()
        mock.return_value = blackboard
        yield blackboard


@pytest.mark.asyncio
async def test_enumerate_phase_initialization(mock_blackboard):
    """Test ENUMERATE phase initialization"""
    phase = EnumeratePhase(mock_blackboard, ["192.168.1.0/24"])

    assert phase.name == "ENUMERATE"
    assert phase.targets == ["192.168.1.0/24"]
    assert phase.blackboard == mock_blackboard


@pytest.mark.asyncio
async def test_enumerate_phase_execution(mock_blackboard):
    """Test ENUMERATE phase execution"""
    phase = EnumeratePhase(mock_blackboard, ["192.168.1.0/24"])

    # Mock collect results to return immediately
    with patch.object(phase, '_collect_results', return_value=None):
        result = await phase.execute()

        assert result["phase"] == "ENUMERATE"
        assert result["status"] == "completed"
        assert "start_time" in result
        assert "end_time" in result
        assert "summary" in result


@pytest.mark.asyncio
async def test_plan_phase_initialization(mock_blackboard):
    """Test PLAN phase initialization"""
    enumerate_results = {
        "discoveries": {
            "hosts": [{"ip": "192.168.1.1", "ports": [80, 443]}]
        }
    }

    phase = PlanPhase(mock_blackboard, enumerate_results)

    assert phase.name == "PLAN"
    assert phase.enumerate_results == enumerate_results


@pytest.mark.asyncio
async def test_plan_phase_execution(mock_blackboard):
    """Test PLAN phase execution"""
    enumerate_results = {
        "discoveries": {
            "hosts": [
                {"ip": "192.168.1.1", "ports": [80, 443], "services": ["http", "https"]}
            ]
        }
    }

    phase = PlanPhase(mock_blackboard, enumerate_results)

    with patch.object(phase, '_collect_web_scan_results', return_value=None):
        result = await phase.execute()

        assert result["phase"] == "PLAN"
        assert result["status"] == "completed"
        assert "attack_chains" in result
        assert len(result["attack_chains"]) > 0


@pytest.mark.asyncio
async def test_execute_phase_manual_mode(mock_blackboard):
    """Test EXECUTE phase with manual confirmation (should skip)"""
    plan_results = {
        "priority_targets": [
            {"target": "192.168.1.1", "attack_vector": "web_application"}
        ]
    }

    phase = ExecutePhase(mock_blackboard, plan_results, auto_execute=False)

    result = await phase.execute()

    assert result["status"] == "skipped"
    assert "manual confirmation required" in result["message"].lower()


@pytest.mark.asyncio
async def test_execute_phase_auto_mode(mock_blackboard):
    """Test EXECUTE phase with auto-execute enabled"""
    plan_results = {
        "priority_targets": [
            {"target": "192.168.1.1", "attack_vector": "web_application"}
        ]
    }

    phase = ExecutePhase(mock_blackboard, plan_results, auto_execute=True)

    with patch.object(phase, '_collect_exploitation_results', return_value=None):
        result = await phase.execute()

        assert result["phase"] == "EXECUTE"
        assert result["status"] == "completed"
        assert "exploits_attempted" in result


@pytest.mark.asyncio
async def test_workflow_engine_initialization():
    """Test workflow engine initialization"""
    engine = WorkflowEngine(["192.168.1.0/24"], auto_execute=False)

    assert engine.targets == ["192.168.1.0/24"]
    assert engine.auto_execute is False
    assert "enumerate" in engine.results
    assert "plan" in engine.results
    assert "execute" in engine.results


@pytest.mark.asyncio
async def test_workflow_engine_full_run():
    """Test complete workflow engine run"""
    engine = WorkflowEngine(["192.168.1.0/24"], auto_execute=False)

    with patch('app.services.workflow_engine.Blackboard'):
        with patch.object(EnumeratePhase, 'execute', return_value={
            "phase": "ENUMERATE",
            "status": "completed",
            "discoveries": {"hosts": []},
            "summary": {"hosts_found": 0}
        }), \
        patch.object(PlanPhase, 'execute', return_value={
            "phase": "PLAN",
            "status": "completed",
            "attack_chains": [],
            "summary": {"attack_chains_generated": 0}
        }), \
        patch.object(ExecutePhase, 'execute', return_value={
            "phase": "EXECUTE",
            "status": "skipped",
            "summary": {"exploits_attempted": 0}
        }):

            result = await engine.run_workflow()

            assert result["workflow"] == "ENUMERATE > PLAN > EXECUTE"
            assert "summary" in result
            assert result["targets"] == ["192.168.1.0/24"]


def test_plan_phase_priority_calculation(mock_blackboard):
    """Test target priority calculation"""
    enumerate_results = {"discoveries": {"hosts": []}}
    phase = PlanPhase(mock_blackboard, enumerate_results)

    # Test priority calculation
    host1 = {"ports": [80, 443], "services": ["http", "ssh"]}
    host2 = {"ports": [22], "services": ["ssh"]}

    priority1 = phase._calculate_priority(host1)
    priority2 = phase._calculate_priority(host2)

    # Host with more ports and services should have higher priority
    assert priority1 > priority2


def test_plan_phase_attack_vector_determination(mock_blackboard):
    """Test attack vector determination"""
    enumerate_results = {"discoveries": {"hosts": []}}
    phase = PlanPhase(mock_blackboard, enumerate_results)

    # Web services
    host_web = {"services": ["http", "apache"]}
    assert phase._determine_attack_vector(host_web) == "web_application"

    # SSH
    host_ssh = {"services": ["ssh"]}
    assert phase._determine_attack_vector(host_ssh) == "ssh_bruteforce"

    # SMB
    host_smb = {"services": ["smb", "445"]}
    assert phase._determine_attack_vector(host_smb) == "smb_exploitation"

    # Generic
    host_generic = {"services": ["unknown"]}
    assert phase._determine_attack_vector(host_generic) == "generic_network"
