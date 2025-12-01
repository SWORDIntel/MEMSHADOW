"""
Unit tests for WebScan Agent
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

import sys
sys.path.insert(0, '/home/user/MEMSHADOW')

from swarm_agents.agent_webscan import WebScanAgent


@pytest.fixture
def webscan_agent():
    """Create WebScan agent instance"""
    return WebScanAgent(agent_id="webscan_test_001")


@pytest.mark.asyncio
async def test_webscan_agent_initialization(webscan_agent):
    """Test WebScan agent initialization"""
    assert webscan_agent.agent_type == "webscan"
    assert webscan_agent.agent_id == "webscan_test_001"
    assert len(webscan_agent.visited_urls) == 0
    assert len(webscan_agent.discovered_forms) == 0
    assert len(webscan_agent.vulnerabilities) == 0


@pytest.mark.asyncio
async def test_crawl_website_task(webscan_agent):
    """Test website crawling task"""
    task = {
        "task_type": "crawl",
        "url": "http://example.com",
        "max_depth": 2,
        "max_pages": 10
    }

    # Mock Selenium
    with patch('swarm_agents.agent_webscan.SELENIUM_AVAILABLE', True), \
         patch('swarm_agents.agent_webscan.webdriver.Chrome') as mock_chrome:

        mock_driver = MagicMock()
        mock_driver.find_elements.return_value = []
        mock_chrome.return_value = mock_driver

        with patch.object(webscan_agent, '_recursive_crawl'), \
             patch.object(webscan_agent, '_detect_technologies', return_value={"WordPress": "unknown"}):

            webscan_agent.visited_urls.add("http://example.com")
            result = await webscan_agent.handle_task(task)

            assert result["status"] == "success"
            assert "pages_crawled" in result
            assert "technologies" in result


@pytest.mark.asyncio
async def test_comprehensive_scan_task(webscan_agent):
    """Test comprehensive vulnerability scan"""
    task = {
        "task_type": "scan",
        "url": "http://example.com",
        "scan_depth": "quick",
        "test_categories": ["xss", "sqli"]
    }

    with patch.object(webscan_agent, '_crawl_website', return_value={
        "status": "success",
        "urls_discovered": ["http://example.com/page1"],
        "forms": [],
        "technologies": {"Apache": "2.4"}
    }), \
    patch.object(webscan_agent, '_test_xss'), \
    patch.object(webscan_agent, '_test_sqli'), \
    patch.object(webscan_agent, '_cve_lookup', return_value={"status": "success"}):

        result = await webscan_agent.handle_task(task)

        assert result["status"] == "success"
        assert "vulnerabilities_found" in result
        assert "risk_score" in result


@pytest.mark.asyncio
async def test_fuzzing_task(webscan_agent):
    """Test endpoint fuzzing"""
    task = {
        "task_type": "fuzzing",
        "url": "http://example.com/search",
        "parameter": "q",
        "payload_type": "xss"
    }

    result = await webscan_agent.handle_task(task)

    assert result["status"] == "success"
    assert "payloads_tested" in result
    assert result["payloads_tested"] > 0


@pytest.mark.asyncio
async def test_cve_lookup_task(webscan_agent):
    """Test CVE lookup"""
    task = {
        "task_type": "cve_lookup",
        "technologies": {
            "Apache": "2.4.49",
            "PHP": "7.4.0"
        }
    }

    result = await webscan_agent.handle_task(task)

    assert result["status"] == "success"
    assert "technologies_analyzed" in result
    assert result["technologies_analyzed"] == 2


@pytest.mark.asyncio
async def test_xss_vulnerability_detection(webscan_agent):
    """Test XSS vulnerability detection"""
    webscan_agent.discovered_forms = [
        {
            "url": "http://example.com/form",
            "action": "/submit",
            "method": "POST",
            "inputs": [{"name": "username", "type": "text"}]
        }
    ]

    await webscan_agent._test_xss()

    assert len(webscan_agent.vulnerabilities) > 0
    assert webscan_agent.vulnerabilities[0]["type"] == "XSS"
    assert webscan_agent.vulnerabilities[0]["severity"] == "high"


@pytest.mark.asyncio
async def test_sqli_vulnerability_detection(webscan_agent):
    """Test SQL injection detection"""
    webscan_agent.discovered_forms = [
        {
            "url": "http://example.com/login",
            "action": "/auth",
            "method": "POST",
            "inputs": [{"name": "username", "type": "text"}]
        }
    ]

    await webscan_agent._test_sqli()

    assert len(webscan_agent.vulnerabilities) > 0
    assert webscan_agent.vulnerabilities[0]["type"] == "SQLi"
    assert webscan_agent.vulnerabilities[0]["severity"] == "critical"


def test_risk_calculation(webscan_agent):
    """Test risk score calculation"""
    vulnerabilities = [
        {"cvss": 9.8},
        {"cvss": 7.5},
        {"cvss": 5.0}
    ]

    score = webscan_agent._calculate_risk_score(vulnerabilities)

    assert score == pytest.approx(7.43, 0.1)


def test_risk_level_determination(webscan_agent):
    """Test risk level determination"""
    assert webscan_agent._risk_level(10.0) == "CRITICAL"
    assert webscan_agent._risk_level(8.0) == "HIGH"
    assert webscan_agent._risk_level(5.0) == "MEDIUM"
    assert webscan_agent._risk_level(2.0) == "LOW"


def test_payload_generation(webscan_agent):
    """Test payload generation by type"""
    xss_payloads = webscan_agent._get_payloads_by_type("xss")
    assert len(xss_payloads) > 0
    assert any("script" in p.lower() for p in xss_payloads)

    sqli_payloads = webscan_agent._get_payloads_by_type("sqli")
    assert len(sqli_payloads) > 0
    assert any("or" in p.lower() for p in sqli_payloads)


@pytest.mark.asyncio
async def test_unknown_task_type(webscan_agent):
    """Test handling of unknown task type"""
    task = {
        "task_type": "invalid_task"
    }

    result = await webscan_agent.handle_task(task)

    assert "error" in result
    assert "Unknown task type" in result["error"]
