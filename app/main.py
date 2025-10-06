from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from app.api.v1 import auth, memory, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.db import postgres, chromadb, redis
from app.api.middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware
)

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

@app.get("/")
def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.VERSION}