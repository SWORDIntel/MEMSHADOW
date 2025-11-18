"""
TEMPEST-Style Dashboard API (FLUSTERCUCKER-inspired)
Military-grade interface for MEMSHADOW operations

CLASSIFICATION: UNCLASSIFIED
For authorized security research and defensive analysis only
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import structlog
import asyncio

from app.services.swarm.coordinator import Coordinator
from app.services.swarm.blackboard import Blackboard
from app.services.swarm.mission import Mission
from app.services.sentinel.monitor import sentinel_monitor
from app.services.vanta_blackwidow.tempest_logger import tempest_logger

logger = structlog.get_logger()

router = APIRouter()

# Global state
active_missions: Dict[str, Coordinator] = {}
ws_clients: List[WebSocket] = []


class TEMPESTFormatter:
    """
    TEMPEST military-style formatting for dashboard output
    """

    @staticmethod
    def classification_banner(level: str = "UNCLASSIFIED") -> str:
        """Generate classification banner"""
        border = "=" * 80
        return f"{border}\n{level.center(80)}\n{border}"

    @staticmethod
    def zulu_time() -> str:
        """Military Zulu time format"""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y%m%d %H%M%SZ")

    @staticmethod
    def risk_indicator(cvss: float) -> str:
        """NATO-style threat indicator"""
        if cvss >= 9.0:
            return "🔴 CRITICAL"
        elif cvss >= 7.0:
            return "🟠 HIGH"
        elif cvss >= 4.0:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"

    @staticmethod
    def format_mission_status(mission: Dict[str, Any]) -> Dict[str, Any]:
        """Format mission status in TEMPEST style"""
        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "mission_id": mission.get("mission_id"),
            "mission_name": mission.get("mission_name"),
            "status": mission.get("status", "PENDING").upper(),
            "stages_completed": mission.get("stages_completed", 0),
            "stages_total": mission.get("stages_total", 0),
            "progress_pct": (
                mission.get("stages_completed", 0) /
                max(mission.get("stages_total", 1), 1) * 100
            ),
            "vulnerabilities_found": mission.get("vulnerabilities_found", 0),
            "iocs_detected": mission.get("iocs_detected", 0),
            "agents_deployed": mission.get("agents_deployed", 0)
        }


@router.get("/dashboard/status")
async def get_dashboard_status():
    """
    Get comprehensive dashboard status
    TEMPEST-formatted system overview
    """
    try:
        # Get active missions
        missions_status = []
        for mission_id, coordinator in active_missions.items():
            mission_data = coordinator.blackboard.get_mission_data(mission_id) or {}
            missions_status.append(TEMPESTFormatter.format_mission_status(mission_data))

        # Get sentinel metrics
        sentinel_metrics = await sentinel_monitor.get_dashboard_metrics()

        # Get recent alerts
        recent_alerts = await sentinel_monitor.get_active_alerts(time_window_hours=24)

        # System health
        blackboard = Blackboard()
        agent_health = {
            "recon": blackboard.get_task_count("recon"),
            "crawler": blackboard.get_task_count("crawler"),
            "apimapper": blackboard.get_task_count("apimapper"),
            "authtest": blackboard.get_task_count("authtest"),
            "bgp": blackboard.get_task_count("bgp"),
            "blockchain": blackboard.get_task_count("blockchain")
        }

        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "system_status": "OPERATIONAL",
            "missions": {
                "active": len([m for m in missions_status if m["status"] == "RUNNING"]),
                "completed": len([m for m in missions_status if m["status"] == "COMPLETED"]),
                "failed": len([m for m in missions_status if m["status"] == "FAILED"]),
                "total": len(missions_status),
                "details": missions_status
            },
            "alerts": {
                "total": len(recent_alerts),
                "critical": len([a for a in recent_alerts if a.get("severity") == "critical"]),
                "high": len([a for a in recent_alerts if a.get("severity") == "high"]),
                "medium": len([a for a in recent_alerts if a.get("severity") == "medium"]),
                "low": len([a for a in recent_alerts if a.get("severity") == "low"]),
                "recent": recent_alerts[:10]
            },
            "agents": {
                "deployed": len([a for a in agent_health.values() if a >= 0]),
                "task_queues": agent_health,
                "total_pending_tasks": sum(agent_health.values())
            },
            "sentinel": sentinel_metrics
        }

    except Exception as e:
        logger.error("Dashboard status error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/mission/start")
async def start_mission(mission_file: str):
    """
    Start a new mission
    ENUMERATE phase of ENUMERATE>PLAN>EXECUTE workflow
    """
    try:
        # Load mission
        mission = Mission.load_from_yaml(mission_file)

        # Create coordinator
        coordinator = Coordinator(mission)
        active_missions[mission.mission_id] = coordinator

        # Log mission start
        tempest_logger.audit(
            event_type="mission_start",
            action="initiate_mission",
            resource=mission.mission_id,
            status="started",
            details={
                "mission_name": mission.mission_name,
                "stages": len(mission.stages),
                "classification": "UNCLASSIFIED"
            },
            severity="info"
        )

        # Start mission execution in background
        asyncio.create_task(execute_mission_background(coordinator, mission.mission_id))

        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "status": "MISSION_INITIATED",
            "mission_id": mission.mission_id,
            "mission_name": mission.mission_name,
            "message": "ENUMERATE phase started - agents deploying"
        }

    except Exception as e:
        logger.error("Mission start error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def execute_mission_background(coordinator: Coordinator, mission_id: str):
    """
    Execute mission in background
    PLAN and EXECUTE phases
    """
    try:
        # Execute mission
        result = await coordinator.execute_mission(timeout=3600)

        # Store final result
        coordinator.blackboard.set_mission_data(mission_id, {
            "status": "completed",
            "result": result,
            "completion_time": TEMPESTFormatter.zulu_time()
        })

        # Log completion
        tempest_logger.audit(
            event_type="mission_complete",
            action="complete_mission",
            resource=mission_id,
            status="success",
            details=result,
            severity="info"
        )

        # Notify websocket clients
        await broadcast_mission_update(mission_id, "COMPLETED", result)

    except Exception as e:
        logger.error("Mission execution error", mission_id=mission_id, error=str(e))

        # Log failure
        tempest_logger.audit(
            event_type="mission_failed",
            action="mission_error",
            resource=mission_id,
            status="failed",
            details={"error": str(e)},
            severity="error"
        )

        # Notify websocket clients
        await broadcast_mission_update(mission_id, "FAILED", {"error": str(e)})


@router.get("/dashboard/mission/{mission_id}")
async def get_mission_status(mission_id: str):
    """
    Get detailed mission status
    Real-time PLAN phase monitoring
    """
    try:
        if mission_id not in active_missions:
            raise HTTPException(status_code=404, detail="Mission not found")

        coordinator = active_missions[mission_id]
        mission_data = coordinator.blackboard.get_mission_data(mission_id) or {}

        # Get stage details
        stages = []
        for stage in coordinator.mission.stages:
            stage_data = {
                "stage_id": stage.stage_id,
                "description": stage.description,
                "status": "completed" if stage.stage_id in coordinator.completed_stages else "pending",
                "tasks": len(stage.tasks),
                "success_criteria": stage.success_criteria
            }
            stages.append(stage_data)

        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "mission": TEMPESTFormatter.format_mission_status(mission_data),
            "stages": stages,
            "vulnerabilities": mission_data.get("vulnerabilities", []),
            "iocs": mission_data.get("iocs", []),
            "recommendations": mission_data.get("recommendations", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mission status error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/intel/iocs")
async def get_iocs(severity: Optional[str] = None, limit: int = 100):
    """
    Get Indicators of Compromise
    Intelligence gathering for PLAN phase
    """
    try:
        blackboard = Blackboard()
        all_iocs = blackboard.get_list("iocs")

        # Filter by severity if specified
        if severity:
            all_iocs = [ioc for ioc in all_iocs if ioc.get("threat_level") == severity.lower()]

        # Limit results
        all_iocs = all_iocs[:limit]

        # Format with TEMPEST style
        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "total_iocs": len(all_iocs),
            "iocs": all_iocs,
            "threat_summary": {
                "critical": len([i for i in all_iocs if i.get("threat_level") == "critical"]),
                "high": len([i for i in all_iocs if i.get("threat_level") == "high"]),
                "medium": len([i for i in all_iocs if i.get("threat_level") == "medium"]),
                "low": len([i for i in all_iocs if i.get("threat_level") == "low"])
            }
        }

    except Exception as e:
        logger.error("IOC retrieval error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/intel/vulnerabilities")
async def get_vulnerabilities(min_cvss: Optional[float] = None, limit: int = 100):
    """
    Get discovered vulnerabilities
    Tactical intelligence for EXECUTE phase
    """
    try:
        blackboard = Blackboard()
        all_vulns = blackboard.get_list("vulnerabilities")

        # Filter by CVSS if specified
        if min_cvss is not None:
            all_vulns = [v for v in all_vulns if v.get("cvss", 0) >= min_cvss]

        # Limit results
        all_vulns = all_vulns[:limit]

        # Add risk indicators
        for vuln in all_vulns:
            vuln["risk_indicator"] = TEMPESTFormatter.risk_indicator(vuln.get("cvss", 0))

        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "total_vulnerabilities": len(all_vulns),
            "vulnerabilities": all_vulns,
            "severity_distribution": {
                "critical": len([v for v in all_vulns if v.get("cvss", 0) >= 9.0]),
                "high": len([v for v in all_vulns if 7.0 <= v.get("cvss", 0) < 9.0]),
                "medium": len([v for v in all_vulns if 4.0 <= v.get("cvss", 0) < 7.0]),
                "low": len([v for v in all_vulns if v.get("cvss", 0) < 4.0])
            }
        }

    except Exception as e:
        logger.error("Vulnerability retrieval error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/dashboard/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time dashboard updates
    Live monitoring during EXECUTE phase
    """
    await websocket.accept()
    ws_clients.append(websocket)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "message": "Dashboard WebSocket connected"
        })

        # Keep connection alive
        while True:
            # Receive heartbeat
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping
                await websocket.send_json({
                    "type": "ping",
                    "timestamp_zulu": TEMPESTFormatter.zulu_time()
                })

    except WebSocketDisconnect:
        ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        if websocket in ws_clients:
            ws_clients.remove(websocket)


async def broadcast_mission_update(mission_id: str, status: str, data: Dict[str, Any]):
    """
    Broadcast mission update to all connected WebSocket clients
    """
    message = {
        "type": "mission_update",
        "classification": "UNCLASSIFIED",
        "timestamp_zulu": TEMPESTFormatter.zulu_time(),
        "mission_id": mission_id,
        "status": status,
        "data": data
    }

    # Send to all connected clients
    disconnected = []
    for client in ws_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error("Failed to send to websocket client", error=str(e))
            disconnected.append(client)

    # Remove disconnected clients
    for client in disconnected:
        ws_clients.remove(client)


@router.post("/dashboard/mission/{mission_id}/stop")
async def stop_mission(mission_id: str):
    """
    Stop a running mission
    Emergency abort for EXECUTE phase
    """
    try:
        if mission_id not in active_missions:
            raise HTTPException(status_code=404, detail="Mission not found")

        coordinator = active_missions[mission_id]

        # Update mission status
        coordinator.blackboard.set_mission_data(mission_id, {
            "status": "stopped",
            "stop_time": TEMPESTFormatter.zulu_time(),
            "stopped_by": "operator"
        })

        # Log stop
        tempest_logger.audit(
            event_type="mission_stopped",
            action="operator_abort",
            resource=mission_id,
            status="stopped",
            details={"reason": "Manual operator abort"},
            severity="warning"
        )

        # Remove from active missions
        del active_missions[mission_id]

        return {
            "classification": "UNCLASSIFIED",
            "timestamp_zulu": TEMPESTFormatter.zulu_time(),
            "status": "MISSION_ABORTED",
            "mission_id": mission_id,
            "message": "Mission stopped by operator"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mission stop error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
