"""
SPINBUSTER Dashboard API

Real-time security operations dashboard for MEMSHADOW SWARM and HYDRA.

Provides:
- Live mission status
- Agent health monitoring
- Security alerts
- Threat intelligence visualization
- IoC tracking
- Vulnerability management
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio

from app.api.dependencies import get_current_active_user
from app.models.user import User
from app.services.sentinel.monitor import sentinel_monitor
from app.services.swarm.blackboard import Blackboard
from app.services.fuzzy_vector_intel import fuzzy_vector_intel
from app.services.vanta_blackwidow.ioc_identifier import ioc_identifier

router = APIRouter()
blackboard = Blackboard()


@router.get("/dashboard/overview")
async def get_dashboard_overview(
    *,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get main dashboard overview

    Returns:
        Dashboard data with metrics, alerts, and status
    """
    # Get Sentinel metrics
    metrics = await sentinel_monitor.get_dashboard_metrics()

    # Get active missions
    active_missions = blackboard.get_all_keys("mission:*")

    # Get recent IoCs
    recent_iocs = blackboard.get_list("iocs", start=0, end=49)  # Last 50

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": {
            "operational": True,
            "swarm_active": len(active_missions) > 0,
            "agents_deployed": metrics['system_health']['agents']
        },
        "metrics": metrics,
        "active_missions": len(active_missions),
        "recent_iocs": recent_iocs[:10],  # Top 10
        "threat_level": _calculate_threat_level(metrics)
    }


@router.get("/dashboard/missions")
async def get_missions_status(
    *,
    current_user: User = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """
    Get status of all missions

    Returns:
        List of mission statuses
    """
    mission_keys = blackboard.get_all_keys("mission:*")

    missions = []
    for key in mission_keys:
        mission_data = blackboard.get(key)
        if mission_data:
            missions.append(mission_data)

    return missions


@router.get("/dashboard/agents")
async def get_agents_status(
    *,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get status of all SWARM agents

    Returns:
        Agent status information
    """
    agent_types = ['recon', 'apimapper', 'authtest', 'bgp', 'blockchain', 'crawler']

    agents_status = {}

    for agent_type in agent_types:
        pending_tasks = blackboard.get_task_count(agent_type)

        agents_status[agent_type] = {
            "type": agent_type,
            "status": "active" if pending_tasks >= 0 else "offline",
            "pending_tasks": pending_tasks,
            "last_active": datetime.utcnow().isoformat()  # Would track actual activity
        }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "agents": agents_status,
        "total_agents": len(agent_types),
        "active_agents": sum(1 for a in agents_status.values() if a['status'] == 'active')
    }


@router.get("/dashboard/alerts")
async def get_security_alerts(
    *,
    time_window_hours: int = 24,
    severity: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """
    Get security alerts

    Args:
        time_window_hours: Time window for alerts
        severity: Filter by severity

    Returns:
        List of alerts
    """
    alerts = await sentinel_monitor.get_active_alerts(time_window_hours)

    if severity:
        alerts = [a for a in alerts if a.get('severity') == severity]

    return alerts


@router.get("/dashboard/iocs")
async def get_ioc_dashboard(
    *,
    limit: int = 100,
    threat_level: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get IoC dashboard data

    Args:
        limit: Maximum number of IoCs
        threat_level: Filter by threat level

    Returns:
        IoC dashboard data
    """
    # Get IoCs from blackboard
    all_iocs = blackboard.get_list("iocs", start=0, end=limit-1)

    # Filter by threat level if specified
    if threat_level:
        all_iocs = [ioc for ioc in all_iocs if ioc.get('threat_level') == threat_level]

    # Aggregate statistics
    stats = {
        'total': len(all_iocs),
        'by_type': {},
        'by_threat_level': {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'unknown': 0
        }
    }

    for ioc in all_iocs:
        # Count by type
        ioc_type = ioc.get('type', 'unknown')
        stats['by_type'][ioc_type] = stats['by_type'].get(ioc_type, 0) + 1

        # Count by threat level
        threat = ioc.get('threat_level', 'unknown')
        if threat in stats['by_threat_level']:
            stats['by_threat_level'][threat] += 1

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": stats,
        "iocs": all_iocs[:50],  # Return top 50
        "timeline": _generate_ioc_timeline(all_iocs)
    }


@router.get("/dashboard/vulnerabilities")
async def get_vulnerability_dashboard(
    *,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get vulnerability dashboard data

    Returns:
        Vulnerability dashboard information
    """
    # Get vulnerabilities from blackboard
    vulns = blackboard.get_list("vulnerabilities", start=0, end=99)

    stats = {
        'total': len(vulns),
        'by_severity': {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        },
        'by_type': {}
    }

    for vuln in vulns:
        # Count by severity
        severity = vuln.get('severity', 'unknown')
        if severity in stats['by_severity']:
            stats['by_severity'][severity] += 1

        # Count by type
        vuln_type = vuln.get('type', 'unknown')
        stats['by_type'][vuln_type] = stats['by_type'].get(vuln_type, 0) + 1

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": stats,
        "vulnerabilities": vulns[:20],  # Top 20
        "recommendations": _generate_vuln_recommendations(vulns)
    }


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates

    Provides live updates for:
    - Mission progress
    - Agent status
    - New alerts
    - IoC discoveries
    """
    await websocket.accept()

    try:
        while True:
            # Get latest dashboard data
            overview = await get_dashboard_overview(current_user=None)  # Needs auth handling

            # Send update
            await websocket.send_json(overview)

            # Wait before next update
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass


@router.get("/dashboard/threat-map")
async def get_threat_map(
    *,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get geographic threat map data

    Returns:
        Threat map with IP geolocation
    """
    # Get IP IoCs
    ip_iocs = blackboard.get_list("iocs:ipv4", start=0, end=99)

    # In production, use IP geolocation service
    threat_map = {
        "timestamp": datetime.utcnow().isoformat(),
        "threats_by_country": {},  # Would map IPs to countries
        "total_threats": len(ip_iocs),
        "high_risk_regions": []
    }

    return threat_map


def _calculate_threat_level(metrics: Dict[str, Any]) -> str:
    """
    Calculate overall threat level

    Args:
        metrics: Dashboard metrics

    Returns:
        Threat level (critical/high/medium/low)
    """
    alerts = metrics.get('alerts', {})

    if alerts.get('critical', 0) > 0:
        return 'critical'
    elif alerts.get('high', 0) >= 3:
        return 'high'
    elif alerts.get('medium', 0) >= 5:
        return 'medium'

    return 'low'


def _generate_ioc_timeline(iocs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate IoC timeline for visualization

    Args:
        iocs: List of IoCs

    Returns:
        Timeline data
    """
    # Group IoCs by hour
    timeline = {}

    for ioc in iocs:
        timestamp = ioc.get('timestamp', datetime.utcnow().isoformat())

        try:
            dt = datetime.fromisoformat(timestamp)
            hour_key = dt.strftime('%Y-%m-%d %H:00')

            if hour_key not in timeline:
                timeline[hour_key] = {'count': 0, 'critical': 0}

            timeline[hour_key]['count'] += 1

            if ioc.get('threat_level') == 'critical':
                timeline[hour_key]['critical'] += 1

        except:
            pass

    # Convert to list
    return [
        {'timestamp': k, **v}
        for k, v in sorted(timeline.items())
    ]


def _generate_vuln_recommendations(vulns: List[Dict[str, Any]]) -> List[str]:
    """
    Generate vulnerability remediation recommendations

    Args:
        vulns: List of vulnerabilities

    Returns:
        List of recommendations
    """
    recommendations = []

    # Count by type
    types = {}
    for vuln in vulns:
        vuln_type = vuln.get('type', 'unknown')
        types[vuln_type] = types.get(vuln_type, 0) + 1

    # Generate recommendations
    if types.get('sql_injection', 0) > 0:
        recommendations.append("Implement parameterized queries to prevent SQL injection")

    if types.get('xss', 0) > 0:
        recommendations.append("Enable Content Security Policy and output encoding")

    if types.get('auth_bypass', 0) > 0:
        recommendations.append("Review and strengthen authentication mechanisms")

    if not recommendations:
        recommendations.append("Continue monitoring for new vulnerabilities")

    return recommendations
