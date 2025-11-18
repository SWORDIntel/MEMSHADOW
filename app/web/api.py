"""
MEMSHADOW Web API
FastAPI application providing comprehensive REST API for all components
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pydantic import BaseModel, Field
import structlog

from app.web.config import ConfigManager, SystemConfig
from app.web.auth import authenticate_token, create_access_token, authenticate_user
import os

logger = structlog.get_logger()

# Initialize FastAPI
app = FastAPI(
    title="MEMSHADOW Web Interface",
    description="Comprehensive web UI for MEMSHADOW memory system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Load CORS origins from environment
cors_origins_str = os.getenv("WEB_CORS_ORIGINS", "http://localhost:8000")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

logger.info("CORS configured", allowed_origins=cors_origins)

# CORS middleware with proper whitelisting
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Whitelist from environment
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security
security = HTTPBearer()

# Templates and static files
templates = Jinja2Templates(directory="app/web/templates")
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

# Configuration manager
config_manager = ConfigManager()

# Global instances (initialized on startup)
federated_coordinator = None
maml_adapter = None
consciousness_integrator = None
self_modifying_engine = None


# ============================================================================
# Pydantic Models
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class StatusResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float


class FederatedUpdateRequest(BaseModel):
    update_data: Dict[str, Any]
    privacy_budget: float = Field(0.1, ge=0.0, le=1.0)


class MAMLTaskRequest(BaseModel):
    task_id: str
    task_name: str
    support_examples: List[Dict[str, Any]]
    query_examples: List[Dict[str, Any]]
    num_adaptation_steps: Optional[int] = None


class ConsciousProcessingRequest(BaseModel):
    input_items: List[Dict[str, Any]]
    goal_context: Dict[str, Any]
    processing_mode: str = "hybrid"  # automatic, controlled, hybrid


class ImprovementRequest(BaseModel):
    function_name: str
    source_code: str
    categories: List[str]
    auto_apply: bool = False


class ConfigUpdateRequest(BaseModel):
    section: str
    updates: Dict[str, Any]


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """Login and receive access token"""
    # Authenticate user
    user = authenticate_user(request.username, request.password)

    if not user:
        logger.warning("Failed login attempt", username=request.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    token = create_access_token({
        "sub": user.username,
        "role": user.role
    })

    logger.info("User logged in successfully", username=request.username)

    return TokenResponse(access_token=token)


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Logout (invalidate token)"""
    # TODO: Implement token blacklisting
    logger.info("User logged out")
    return {"message": "Logged out successfully"}


# ============================================================================
# System Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get system status"""
    import time

    return StatusResponse(
        status="operational",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=time.time()  # TODO: Track actual uptime
    )


@app.get("/api/stats", tags=["System"])
async def get_stats():
    """Get comprehensive system statistics"""
    stats = {
        "federated": {},
        "meta_learning": {},
        "consciousness": {},
        "self_modifying": {}
    }

    # Get stats from each component
    try:
        if federated_coordinator:
            stats["federated"] = await federated_coordinator.get_federation_stats()
    except Exception as e:
        logger.error("Failed to get federated stats", error=str(e))

    try:
        if consciousness_integrator:
            stats["consciousness"] = await consciousness_integrator.get_consciousness_state()
    except Exception as e:
        logger.error("Failed to get consciousness stats", error=str(e))

    try:
        if self_modifying_engine:
            stats["self_modifying"] = await self_modifying_engine.get_improvement_status()
    except Exception as e:
        logger.error("Failed to get self-modifying stats", error=str(e))

    return stats


# ============================================================================
# Configuration Endpoints
# ============================================================================

@app.get("/api/config", tags=["Configuration"])
async def get_config():
    """Get current system configuration"""
    return config_manager.get_all_config()


@app.get("/api/config/{section}", tags=["Configuration"])
async def get_config_section(section: str):
    """Get configuration for specific section"""
    config = config_manager.get_config(section)
    if not config:
        raise HTTPException(status_code=404, detail=f"Config section '{section}' not found")
    return config


@app.put("/api/config/{section}", tags=["Configuration"])
async def update_config_section(section: str, request: ConfigUpdateRequest):
    """Update configuration section"""
    try:
        config_manager.update_config(section, request.updates)
        logger.info("Configuration updated", section=section)
        return {"message": f"Config section '{section}' updated successfully"}
    except Exception as e:
        logger.error("Failed to update config", section=section, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/save", tags=["Configuration"])
async def save_config():
    """Save current configuration to disk"""
    try:
        config_manager.save_config()
        return {"message": "Configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")


@app.post("/api/config/reload", tags=["Configuration"])
async def reload_config():
    """Reload configuration from disk"""
    try:
        config_manager.load_config()
        return {"message": "Configuration reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {e}")


# ============================================================================
# Federated Learning Endpoints (Phase 8.1)
# ============================================================================

@app.get("/federated", response_class=HTMLResponse, tags=["UI"])
async def federated_page(request: Request):
    """Federated learning management page"""
    return templates.TemplateResponse("federated.html", {"request": request})


@app.post("/api/federated/start", tags=["Federated Learning"])
async def start_federated():
    """Start federated coordinator"""
    global federated_coordinator

    try:
        from app.services.federated import FederatedCoordinator

        config = config_manager.get_config("federated")
        federated_coordinator = FederatedCoordinator(
            node_id=config.get("node_id", "memshadow_001"),
            privacy_budget=config.get("privacy_budget", 1.0)
        )

        await federated_coordinator.start()

        logger.info("Federated coordinator started")
        return {"message": "Federated coordinator started successfully"}

    except Exception as e:
        logger.error("Failed to start federated coordinator", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/federated/stop", tags=["Federated Learning"])
async def stop_federated():
    """Stop federated coordinator"""
    global federated_coordinator

    if not federated_coordinator:
        raise HTTPException(status_code=400, detail="Federated coordinator not running")

    try:
        await federated_coordinator.stop()
        federated_coordinator = None

        logger.info("Federated coordinator stopped")
        return {"message": "Federated coordinator stopped successfully"}

    except Exception as e:
        logger.error("Failed to stop federated coordinator", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/federated/join", tags=["Federated Learning"])
async def join_federation(peers: List[str]):
    """Join federated network"""
    if not federated_coordinator:
        raise HTTPException(status_code=400, detail="Federated coordinator not started")

    try:
        await federated_coordinator.join_federation(peers)
        logger.info("Joined federation", peers=peers)
        return {"message": f"Joined federation with {len(peers)} peers"}

    except Exception as e:
        logger.error("Failed to join federation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/federated/update", tags=["Federated Learning"])
async def share_federated_update(request: FederatedUpdateRequest):
    """Share update with federation"""
    if not federated_coordinator:
        raise HTTPException(status_code=400, detail="Federated coordinator not started")

    try:
        update_id = await federated_coordinator.share_update(
            update_data=request.update_data,
            privacy_budget=request.privacy_budget
        )

        logger.info("Update shared", update_id=update_id)
        return {"update_id": update_id, "message": "Update shared successfully"}

    except Exception as e:
        logger.error("Failed to share update", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/federated/stats", tags=["Federated Learning"])
async def get_federated_stats():
    """Get federated learning statistics"""
    if not federated_coordinator:
        return {"status": "not_running"}

    try:
        stats = await federated_coordinator.get_federation_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get federated stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/federated/peers", tags=["Federated Learning"])
async def get_federated_peers():
    """Get connected peers"""
    if not federated_coordinator:
        return {"peers": []}

    return {"peers": list(federated_coordinator.peers.keys())}


# ============================================================================
# Meta-Learning Endpoints (Phase 8.2)
# ============================================================================

@app.get("/meta-learning", response_class=HTMLResponse, tags=["UI"])
async def meta_learning_page(request: Request):
    """Meta-learning management page"""
    return templates.TemplateResponse("meta_learning.html", {"request": request})


@app.post("/api/meta-learning/adapt", tags=["Meta-Learning"])
async def adapt_to_task(request: MAMLTaskRequest):
    """Adapt to new task using MAML"""
    global maml_adapter

    try:
        from app.services.meta_learning import MAMLMemoryAdapter, MemoryTask

        if not maml_adapter:
            maml_adapter = MAMLMemoryAdapter()

        # Create task
        task = MemoryTask(
            task_id=request.task_id,
            task_name=request.task_name,
            support_memories=request.support_examples,
            query_memories=request.query_examples
        )

        # Adapt
        result = await maml_adapter.adapt_to_task(
            task,
            num_adaptation_steps=request.num_adaptation_steps
        )

        logger.info("Adapted to task", task_id=request.task_id)

        return {
            "task_id": result.task_id,
            "pre_accuracy": result.pre_adaptation_accuracy,
            "post_accuracy": result.post_adaptation_accuracy,
            "improvement": result.improvement,
            "adaptation_time_ms": result.adaptation_time_ms
        }

    except Exception as e:
        logger.error("Failed to adapt to task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta-learning/performance", tags=["Meta-Learning"])
async def get_performance_metrics():
    """Get performance tracking metrics"""
    try:
        from app.services.meta_learning import PerformanceTracker

        tracker = PerformanceTracker()

        # Get all metrics
        metrics = {}
        for metric_name, metric_list in tracker.metrics.items():
            if metric_list:
                latest = metric_list[-1]
                metrics[metric_name] = {
                    "value": latest.value,
                    "category": latest.category.value,
                    "timestamp": latest.timestamp.isoformat()
                }

        return {"metrics": metrics}

    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta-learning/proposals", tags=["Meta-Learning"])
async def get_improvement_proposals():
    """Get current improvement proposals"""
    try:
        from app.services.meta_learning import ImprovementEngine, PerformanceTracker

        tracker = PerformanceTracker()
        engine = ImprovementEngine(tracker=tracker)

        proposals = await engine.analyze_and_propose()

        return {
            "proposals": [
                {
                    "id": p.proposal_id,
                    "title": p.title,
                    "type": p.improvement_type.value,
                    "risk": p.risk_level.value,
                    "impact": p.estimated_impact,
                    "description": p.description
                }
                for p in proposals
            ]
        }

    except Exception as e:
        logger.error("Failed to get proposals", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Consciousness Endpoints (Phase 8.3)
# ============================================================================

@app.get("/consciousness", response_class=HTMLResponse, tags=["UI"])
async def consciousness_page(request: Request):
    """Consciousness architecture page"""
    return templates.TemplateResponse("consciousness.html", {"request": request})


@app.post("/api/consciousness/start", tags=["Consciousness"])
async def start_consciousness():
    """Start consciousness integrator"""
    global consciousness_integrator

    try:
        from app.services.consciousness import ConsciousnessIntegrator

        config = config_manager.get_config("consciousness")
        consciousness_integrator = ConsciousnessIntegrator(
            workspace_capacity=config.get("workspace_capacity", 7),
            num_attention_heads=config.get("attention_heads", 8),
            enable_metacognition=config.get("enable_metacognition", True)
        )

        await consciousness_integrator.start()

        logger.info("Consciousness integrator started")
        return {"message": "Consciousness integrator started successfully"}

    except Exception as e:
        logger.error("Failed to start consciousness", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/consciousness/stop", tags=["Consciousness"])
async def stop_consciousness():
    """Stop consciousness integrator"""
    global consciousness_integrator

    if not consciousness_integrator:
        raise HTTPException(status_code=400, detail="Consciousness integrator not running")

    try:
        await consciousness_integrator.stop()
        consciousness_integrator = None

        logger.info("Consciousness integrator stopped")
        return {"message": "Consciousness integrator stopped successfully"}

    except Exception as e:
        logger.error("Failed to stop consciousness", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/consciousness/process", tags=["Consciousness"])
async def process_consciously(request: ConsciousProcessingRequest):
    """Process items through conscious pipeline"""
    if not consciousness_integrator:
        raise HTTPException(status_code=400, detail="Consciousness integrator not started")

    try:
        from app.services.consciousness import ProcessingMode

        # Map mode string to enum
        mode_map = {
            "automatic": ProcessingMode.AUTOMATIC,
            "controlled": ProcessingMode.CONTROLLED,
            "hybrid": ProcessingMode.HYBRID
        }
        mode = mode_map.get(request.processing_mode, ProcessingMode.HYBRID)

        decision = await consciousness_integrator.process_consciously(
            input_items=request.input_items,
            goal_context=request.goal_context,
            mode=mode
        )

        return {
            "decision_id": decision.decision_id,
            "action": decision.action,
            "confidence": decision.confidence,
            "should_defer": decision.should_defer,
            "processing_mode": decision.processing_mode.value,
            "processing_time_ms": decision.processing_time_ms,
            "workspace_items": decision.workspace_items_considered,
            "attended_items": decision.attended_items
        }

    except Exception as e:
        logger.error("Failed to process consciously", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/consciousness/state", tags=["Consciousness"])
async def get_consciousness_state():
    """Get current consciousness state"""
    if not consciousness_integrator:
        return {"status": "not_running"}

    try:
        state = await consciousness_integrator.get_consciousness_state()
        return state
    except Exception as e:
        logger.error("Failed to get consciousness state", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/consciousness/workspace", tags=["Consciousness"])
async def get_workspace_items():
    """Get items currently in global workspace"""
    if not consciousness_integrator:
        return {"items": []}

    try:
        items = await consciousness_integrator.workspace.get_items()

        return {
            "items": [
                {
                    "id": item.item_id,
                    "source": item.source_module,
                    "salience": item.salience,
                    "relevance": item.relevance,
                    "novelty": item.novelty,
                    "priority": item.priority.value,
                    "activation": item.activation_level,
                    "age_seconds": item.age_seconds
                }
                for item in items
            ]
        }
    except Exception as e:
        logger.error("Failed to get workspace items", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Self-Modifying Endpoints (Phase 8.4)
# ============================================================================

@app.get("/self-modifying", response_class=HTMLResponse, tags=["UI"])
async def self_modifying_page(request: Request):
    """Self-modification management page"""
    return templates.TemplateResponse("self_modifying.html", {"request": request})


@app.post("/api/self-modifying/start", tags=["Self-Modifying"])
async def start_self_modifying():
    """Start self-modifying engine"""
    global self_modifying_engine

    try:
        from app.services.self_modifying import SelfModifyingEngine, SafetyLevel

        config = config_manager.get_config("self_modifying")

        # Map safety level string to enum
        safety_map = {
            "read_only": SafetyLevel.READ_ONLY,
            "documentation": SafetyLevel.DOCUMENTATION,
            "low_risk": SafetyLevel.LOW_RISK,
            "medium_risk": SafetyLevel.MEDIUM_RISK,
            "full_access": SafetyLevel.FULL_ACCESS
        }
        safety_level = safety_map.get(
            config.get("safety_level", "read_only"),
            SafetyLevel.READ_ONLY
        )

        self_modifying_engine = SelfModifyingEngine(
            safety_level=safety_level,
            enable_auto_apply=config.get("enable_auto_apply", False)
        )

        await self_modifying_engine.start()

        logger.info("Self-modifying engine started", safety_level=safety_level.value)
        return {
            "message": "Self-modifying engine started successfully",
            "safety_level": safety_level.value
        }

    except Exception as e:
        logger.error("Failed to start self-modifying engine", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/self-modifying/stop", tags=["Self-Modifying"])
async def stop_self_modifying():
    """Stop self-modifying engine"""
    global self_modifying_engine

    if not self_modifying_engine:
        raise HTTPException(status_code=400, detail="Self-modifying engine not running")

    try:
        await self_modifying_engine.stop()
        self_modifying_engine = None

        logger.info("Self-modifying engine stopped")
        return {"message": "Self-modifying engine stopped successfully"}

    except Exception as e:
        logger.error("Failed to stop self-modifying engine", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/self-modifying/improve", tags=["Self-Modifying"])
async def improve_function(request: ImprovementRequest):
    """Request function improvement"""
    if not self_modifying_engine:
        raise HTTPException(status_code=400, detail="Self-modifying engine not started")

    try:
        from app.services.self_modifying import ImprovementCategory
        import ast

        # Map category strings to enums
        category_map = {
            "performance": ImprovementCategory.PERFORMANCE,
            "readability": ImprovementCategory.READABILITY,
            "maintainability": ImprovementCategory.MAINTAINABILITY,
            "security": ImprovementCategory.SECURITY,
            "bug_fix": ImprovementCategory.BUG_FIX,
            "refactoring": ImprovementCategory.REFACTORING,
            "testing": ImprovementCategory.TESTING,
            "documentation": ImprovementCategory.DOCUMENTATION
        }

        categories = [category_map[cat] for cat in request.categories if cat in category_map]

        # SECURITY: Validate code using AST instead of exec()
        # Parse the source code to extract function for analysis
        try:
            tree = ast.parse(request.source_code)
        except SyntaxError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Python syntax: {str(e)}"
            )

        # Find the function definition in the AST
        function_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == request.function_name:
                function_def = node
                break

        if not function_def:
            raise HTTPException(
                status_code=400,
                detail=f"Function '{request.function_name}' not found in source code"
            )

        # Pass the source code and function name to the engine for analysis
        # The engine will perform AST-based analysis without executing code
        result = await self_modifying_engine.analyze_function_source(
            source_code=request.source_code,
            function_name=request.function_name,
            categories=categories,
            auto_apply=request.auto_apply
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to improve function", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/self-modifying/status", tags=["Self-Modifying"])
async def get_self_modifying_status():
    """Get self-modification status"""
    if not self_modifying_engine:
        return {"status": "not_running"}

    try:
        status = await self_modifying_engine.get_improvement_status()
        return status
    except Exception as e:
        logger.error("Failed to get self-modifying status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("MEMSHADOW Web Interface starting up")

    # Load configuration
    config_manager.load_config()

    # Auto-start components if configured
    config = config_manager.get_all_config()

    if config.get("auto_start", {}).get("federated", False):
        try:
            await start_federated()
        except Exception as e:
            logger.error("Failed to auto-start federated", error=str(e))

    if config.get("auto_start", {}).get("consciousness", False):
        try:
            await start_consciousness()
        except Exception as e:
            logger.error("Failed to auto-start consciousness", error=str(e))

    logger.info("MEMSHADOW Web Interface ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("MEMSHADOW Web Interface shutting down")

    # Stop all components
    global federated_coordinator, consciousness_integrator, self_modifying_engine

    if federated_coordinator:
        try:
            await federated_coordinator.stop()
        except Exception as e:
            logger.error("Error stopping federated coordinator", error=str(e))

    if consciousness_integrator:
        try:
            await consciousness_integrator.stop()
        except Exception as e:
            logger.error("Error stopping consciousness integrator", error=str(e))

    if self_modifying_engine:
        try:
            await self_modifying_engine.stop()
        except Exception as e:
            logger.error("Error stopping self-modifying engine", error=str(e))

    logger.info("MEMSHADOW Web Interface shutdown complete")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
