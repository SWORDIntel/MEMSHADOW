"""
Unit tests for SWARM Blackboard
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from app.services.swarm.blackboard import Blackboard


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('app.db.redis.get_client') as mock:
        client = Mock()
        mock.return_value = client
        yield client


@pytest.fixture
def blackboard(mock_redis):
    """Create Blackboard instance with mocked Redis"""
    return Blackboard()


def test_publish_task(blackboard, mock_redis):
    """Test publishing a task to agent queue"""
    task = {
        'task_id': 'task_001',
        'agent_type': 'recon',
        'params': {'target': '192.168.1.0/24'}
    }

    blackboard.publish_task('recon', task)

    # Verify Redis lpush was called
    assert mock_redis.lpush.called
    call_args = mock_redis.lpush.call_args[0]
    assert call_args[0] == 'swarm:tasks:recon'
    assert json.loads(call_args[1]) == task


def test_get_task(blackboard, mock_redis):
    """Test getting a task from queue"""
    task_data = {
        'task_id': 'task_002',
        'agent_type': 'crawler',
        'params': {'url': 'http://example.com'}
    }

    mock_redis.brpop.return_value = ('swarm:tasks:crawler', json.dumps(task_data))

    task = blackboard.get_task('crawler', timeout=5)

    assert task == task_data
    assert mock_redis.brpop.called


def test_get_task_timeout(blackboard, mock_redis):
    """Test task retrieval timeout"""
    mock_redis.brpop.return_value = None

    task = blackboard.get_task('recon', timeout=1)

    assert task is None


def test_publish_report(blackboard, mock_redis):
    """Test publishing an agent report"""
    report = {
        'agent_id': 'recon_001',
        'task_id': 'task_001',
        'status': 'completed',
        'data': {'hosts_found': 5}
    }

    blackboard.publish_report(report)

    assert mock_redis.lpush.called
    call_args = mock_redis.lpush.call_args[0]
    assert call_args[0] == 'swarm:reports'
    assert json.loads(call_args[1]) == report


def test_get_report(blackboard, mock_redis):
    """Test retrieving a report"""
    report_data = {
        'agent_id': 'apimapper_001',
        'status': 'completed',
        'data': {'endpoints': ['/api/users', '/api/posts']}
    }

    mock_redis.brpop.return_value = ('swarm:reports', json.dumps(report_data))

    report = blackboard.get_report(timeout=5)

    assert report == report_data


def test_set_mission_data(blackboard, mock_redis):
    """Test setting mission data"""
    mission_id = 'mission_001'
    data = {
        'status': 'running',
        'stages_completed': 2
    }

    blackboard.set_mission_data(mission_id, data)

    assert mock_redis.set.called
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == f'swarm:mission:{mission_id}'
    assert json.loads(call_args[1]) == data


def test_get_mission_data(blackboard, mock_redis):
    """Test getting mission data"""
    mission_id = 'mission_002'
    data = {'status': 'completed', 'result': 'success'}

    mock_redis.get.return_value = json.dumps(data)

    result = blackboard.get_mission_data(mission_id)

    assert result == data
    assert mock_redis.get.called


def test_update_shared_state(blackboard, mock_redis):
    """Test updating shared state"""
    key = 'known_hosts'
    value = ['192.168.1.1', '192.168.1.2']

    blackboard.update_shared_state(key, value)

    assert mock_redis.set.called
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == f'swarm:state:{key}'
    assert json.loads(call_args[1]) == value


def test_get_shared_state(blackboard, mock_redis):
    """Test getting shared state"""
    key = 'discovered_endpoints'
    value = ['/api/v1/users', '/api/v1/auth']

    mock_redis.get.return_value = json.dumps(value)

    result = blackboard.get_shared_state(key)

    assert result == value


def test_append_to_list(blackboard, mock_redis):
    """Test appending to a list in blackboard"""
    key = 'vulnerabilities'
    item = {'type': 'SQLi', 'severity': 'high'}

    blackboard.append_to_list(key, item)

    assert mock_redis.rpush.called
    call_args = mock_redis.rpush.call_args[0]
    assert call_args[0] == f'swarm:list:{key}'
    assert json.loads(call_args[1]) == item


def test_get_list(blackboard, mock_redis):
    """Test getting a list from blackboard"""
    key = 'alerts'
    items = [
        {'severity': 'critical', 'message': 'Alert 1'},
        {'severity': 'high', 'message': 'Alert 2'}
    ]

    mock_redis.lrange.return_value = [json.dumps(item) for item in items]

    result = blackboard.get_list(key)

    assert result == items
    assert mock_redis.lrange.called


def test_get_task_count(blackboard, mock_redis):
    """Test getting pending task count"""
    mock_redis.llen.return_value = 5

    count = blackboard.get_task_count('recon')

    assert count == 5
    assert mock_redis.llen.called


def test_clear_tasks(blackboard, mock_redis):
    """Test clearing task queue"""
    blackboard.clear_tasks('crawler')

    assert mock_redis.delete.called
    assert mock_redis.delete.call_args[0][0] == 'swarm:tasks:crawler'


def test_clear_reports(blackboard, mock_redis):
    """Test clearing reports queue"""
    blackboard.clear_reports()

    assert mock_redis.delete.called
    assert mock_redis.delete.call_args[0][0] == 'swarm:reports'
