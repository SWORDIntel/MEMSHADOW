"""
Unit tests for WiFi Agent
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Mock the base agent import
import sys
sys.path.insert(0, '/home/user/MEMSHADOW')

from swarm_agents.agent_wifi import WiFiAgent


@pytest.fixture
def wifi_agent():
    """Create WiFi agent instance"""
    return WiFiAgent(agent_id="wifi_test_001")


@pytest.mark.asyncio
async def test_wifi_agent_initialization(wifi_agent):
    """Test WiFi agent initialization"""
    assert wifi_agent.agent_type == "wifi"
    assert wifi_agent.agent_id == "wifi_test_001"
    assert wifi_agent.interface is None
    assert wifi_agent.mon_interface is None


@pytest.mark.asyncio
async def test_scan_networks_task(wifi_agent):
    """Test network scanning task"""
    task = {
        "task_type": "scan",
        "interface": "wlan0",
        "scan_time": 10
    }

    with patch.object(wifi_agent, '_enable_monitor_mode', return_value="wlan0mon"), \
         patch.object(wifi_agent, '_parse_airodump_csv', return_value=[
             {"bssid": "AA:BB:CC:DD:EE:FF", "essid": "TestNetwork", "power": -50}
         ]):

        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_subprocess.return_value = mock_proc

            result = await wifi_agent.handle_task(task)

            assert result["status"] == "success"
            assert "networks_found" in result
            assert result["networks_found"] == 1


@pytest.mark.asyncio
async def test_capture_handshake_task(wifi_agent):
    """Test handshake capture task"""
    task = {
        "task_type": "capture",
        "interface": "wlan0",
        "bssid": "AA:BB:CC:DD:EE:FF",
        "channel": 6,
        "timeout": 60
    }

    with patch.object(wifi_agent, '_enable_monitor_mode', return_value="wlan0mon"), \
         patch.object(wifi_agent, '_set_channel'), \
         patch.object(wifi_agent, '_verify_handshake', return_value=True):

        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.terminate = Mock()
            mock_subprocess.return_value = mock_proc

            result = await wifi_agent.handle_task(task)

            assert result["status"] == "success"
            assert result["handshake_captured"] is True
            assert result["bssid"] == "AA:BB:CC:DD:EE:FF"


@pytest.mark.asyncio
async def test_crack_handshake_cpu(wifi_agent):
    """Test password cracking with CPU"""
    task = {
        "task_type": "crack",
        "capture_file": "/tmp/test.cap",
        "wordlist": "/tmp/wordlist.txt",
        "bssid": "AA:BB:CC:DD:EE:FF",
        "device": "CPU"
    }

    with patch('asyncio.create_subprocess_exec') as mock_subprocess:
        mock_proc = AsyncMock()
        # Simulate password found
        mock_proc.communicate = AsyncMock(return_value=(
            b"KEY FOUND! [ password123 ]",
            b""
        ))
        mock_subprocess.return_value = mock_proc

        result = await wifi_agent.handle_task(task)

        assert result["status"] == "success"
        assert result["password_found"] is True
        assert result["password"] == "password123"


@pytest.mark.asyncio
async def test_enumerate_clients_task(wifi_agent):
    """Test client enumeration task"""
    task = {
        "task_type": "enumerate",
        "interface": "wlan0",
        "scan_time": 30
    }

    with patch.object(wifi_agent, '_enable_monitor_mode', return_value="wlan0mon"), \
         patch.object(wifi_agent, '_parse_airodump_clients', return_value=[
             {"station_mac": "11:22:33:44:55:66", "bssid": "AA:BB:CC:DD:EE:FF"}
         ]):

        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_subprocess.return_value = mock_proc

            result = await wifi_agent.handle_task(task)

            assert result["status"] == "success"
            assert "clients_found" in result
            assert result["clients_found"] == 1


@pytest.mark.asyncio
async def test_deauth_attack_task(wifi_agent):
    """Test deauth attack task"""
    task = {
        "task_type": "deauth",
        "interface": "wlan0",
        "bssid": "AA:BB:CC:DD:EE:FF",
        "count": 10
    }

    with patch.object(wifi_agent, '_enable_monitor_mode', return_value="wlan0mon"):
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"Sent 10 packets", b""))
            mock_subprocess.return_value = mock_proc

            result = await wifi_agent.handle_task(task)

            assert result["status"] == "success"
            assert result["deauth_sent"] == 10
            assert result["bssid"] == "AA:BB:CC:DD:EE:FF"


@pytest.mark.asyncio
async def test_unknown_task_type(wifi_agent):
    """Test handling of unknown task type"""
    task = {
        "task_type": "invalid_task"
    }

    result = await wifi_agent.handle_task(task)

    assert "error" in result
    assert "Unknown task type" in result["error"]
