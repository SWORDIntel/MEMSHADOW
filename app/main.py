from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from app.api.v1 import auth, memory, health, spinbuster, tempest_dashboard, c2
from app.api.v1.mcp import mcp_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.metrics import get_metrics
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting MEMSHADOW API", version=settings.VERSION)

    # Initialize database connections
    await postgres.init_db()
    await chromadb.init_client()
    await redis.init_pool()

    logger.info("Application startup complete.")
    yield

    # Shutdown
    logger.info("Shutting down MEMSHADOW API")
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

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(memory.router, prefix=f"{settings.API_V1_STR}/memory", tags=["memory"])
app.include_router(mcp_router, prefix=f"{settings.API_V1_STR}")  # MCP endpoints
app.include_router(tempest_dashboard.router, prefix=f"{settings.API_V1_STR}/tempest", tags=["tempest"])  # TEMPEST Dashboard (FLUSTERCUCKER-inspired)
app.include_router(c2.router, prefix=f"{settings.API_V1_STR}", tags=["c2"])  # C2 Framework (DavBest-inspired)
app.include_router(spinbuster.router, prefix=f"{settings.API_V1_STR}/spinbuster", tags=["spinbuster"])  # SPINBUSTER Dashboard (legacy)

@app.get("/")
def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.VERSION}

@app.get(f"{settings.API_V1_STR}/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return await get_metrics()