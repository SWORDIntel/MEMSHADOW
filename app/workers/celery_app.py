import asyncio
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from app.core.config import settings
from app.db import postgres, chromadb, redis

celery_app = Celery(
    "memshadow",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    beat_schedule={
        "corpus-autoscan": {
            "task": "app.workers.tasks.autoscan_corpus_directory",
            "schedule": 3600,  # Every hour
        },
    },
)

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize database connections for the worker process."""
    asyncio.run(postgres.init_db())
    asyncio.run(chromadb.init_client())
    asyncio.run(redis.init_pool())

@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Close database connections for the worker process."""
    asyncio.run(postgres.close_db())
    asyncio.run(chromadb.close_client())
    asyncio.run(redis.close_pool())