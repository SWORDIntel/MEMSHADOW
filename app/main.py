from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import structlog
import os


from app.api.v1 import auth, memory, health, openai_compat, task_reminders

# DSMILSYSTEM integration (optional, feature-flagged)
try:
    from app.api.v1 import memory_dsmil
    DSMILSYSTEM_ENABLED = True
except ImportError:
    DSMILSYSTEM_ENABLED = False
    memory_dsmil = None

from app.core.config import settings
from app.core.logging import setup_logging
from app.db import postgres, chromadb, redis
from app.api.middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware
)
from app.middleware.step_up_auth import StepUpAuthMiddleware

# Setup structured logging
setup_logging()
logger = structlog.get_logger()

# Feature flags for optional integrations
# KP14 integration is DISABLED by default - MEMSHADOW works standalone
# Enable via config or environment variable
_kp14_env = os.environ.get('MEMSHADOW_ENABLE_KP14', '').lower()
if _kp14_env:
    ENABLE_KP14_INTEGRATION = _kp14_env == 'true'
else:
    ENABLE_KP14_INTEGRATION = settings.ENABLE_KP14_INTEGRATION

# Conditionally import KP14 router
kp14_router = None
if ENABLE_KP14_INTEGRATION:
    try:
        from app.api.v1 import kp14
        kp14_router = kp14.router
        logger.info("KP14 integration enabled")
    except ImportError as e:
        logger.warning(f"KP14 integration not available: {e}")
        ENABLE_KP14_INTEGRATION = False
else:
    logger.info("KP14 integration disabled (ENABLE_KP14_INTEGRATION=false)")

# Conditionally import mesh client (only when KP14 integration enabled)
# Mesh client allows MEMSHADOW to act as a spoke node receiving hub broadcasts
mesh_client = None
MESH_CLIENT_AVAILABLE = False
if ENABLE_KP14_INTEGRATION:
    try:
        from app.services.mesh_client import init_mesh_client, MESH_AVAILABLE
        if MESH_AVAILABLE:
            mesh_client = init_mesh_client(
                node_id=f"memshadow-{os.environ.get('HOSTNAME', 'default')}",
                mesh_port=settings.KP14_MESH_PORT,
                enabled=True,
            )
            MESH_CLIENT_AVAILABLE = True
            logger.info("Mesh client initialized for hub-spoke communication")
        else:
            logger.info("Mesh library not available - running without mesh connectivity")
    except ImportError as e:
        logger.warning(f"Mesh client not available: {e}")
else:
    logger.debug("Mesh client skipped (KP14 integration disabled)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting MEMSHADOW API", version=settings.VERSION)

    # Initialize database connections
    await postgres.init_db()
    await chromadb.init_client()
    await redis.init_pool()

    # Start mesh client if enabled (hub-spoke communication)
    if mesh_client and MESH_CLIENT_AVAILABLE:
        try:
            await mesh_client.start()
            logger.info("Mesh client started - acting as spoke node")
        except Exception as e:
            # Graceful degradation - continue without mesh
            logger.warning(f"Mesh client failed to start (continuing in standalone mode): {e}")

    logger.info("Application startup complete.")
    yield

    # Shutdown
    logger.info("Shutting down MEMSHADOW API")

    # Stop mesh client if running
    if mesh_client and MESH_CLIENT_AVAILABLE:
        try:
            await mesh_client.stop()
            logger.info("Mesh client stopped")
        except Exception as e:
            logger.warning(f"Error stopping mesh client: {e}")

    await postgres.close_db()
    await chromadb.close_client()
    await redis.close_pool()
    logger.info("Application shutdown complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(StepUpAuthMiddleware)
# app.add_middleware(RateLimitMiddleware) # Rate limiting can be noisy in dev, enable as needed
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include core routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(memory.router, prefix=f"{settings.API_V1_STR}/memory", tags=["memory"])
app.include_router(openai_compat.router, prefix="/v1", tags=["openai"])
app.include_router(task_reminders.router, prefix=f"{settings.API_V1_STR}/reminders", tags=["reminders"])

# Include DSMILSYSTEM memory API (if enabled)
if DSMILSYSTEM_ENABLED and memory_dsmil:
    app.include_router(memory_dsmil.router, tags=["dsmilsystem"])

# Include optional KP14 integration router
if kp14_router:
    app.include_router(kp14_router, prefix=f"{settings.API_V1_STR}", tags=["kp14"])


@app.get("/")
def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.VERSION}


@app.get("/integrations")
def list_integrations():
    """List available integrations and their status."""
    integrations = {
        "kp14": {
            "enabled": ENABLE_KP14_INTEGRATION,
            "available": kp14_router is not None,
        },
        "mesh_client": {
            "enabled": MESH_CLIENT_AVAILABLE,
            "available": mesh_client is not None,
            "status": mesh_client.get_status() if mesh_client else None,
        },
    }
    return integrations
