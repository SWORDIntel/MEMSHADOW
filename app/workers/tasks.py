from uuid import UUID
import asyncio
import structlog
from celery import current_task

from app.workers.celery_app import celery_app
from app.db.postgres import AsyncSessionLocal
from app.services.memory_service import MemoryService

logger = structlog.get_logger()

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def generate_embedding_task(self, memory_id: str, content: str):
    """
    Celery task to generate and store embedding for a memory.
    This runs asynchronously in a Celery worker.
    """
    logger.info("Starting embedding generation task", memory_id=memory_id, task_id=current_task.request.id)

    async def async_generate_and_store():
        session = AsyncSessionLocal()
        try:
            memory_service = MemoryService(session)
            await memory_service.generate_and_store_embedding(
                memory_id=UUID(memory_id),
                content=content
            )
            await session.commit()
            logger.info("Embedding generation task completed successfully", memory_id=memory_id)
        except Exception as e:
            logger.error(
                "Error during embedding generation in task",
                memory_id=memory_id,
                error=str(e),
                exc_info=True
            )
            await session.rollback()
            # Retry the task with exponential backoff
            raise self.retry(exc=e)
        finally:
            await session.close()

    # Run the async function in a new event loop
    # This is necessary because Celery 4.x workers are synchronous.
    asyncio.run(async_generate_and_store())

    return {"status": "success", "memory_id": memory_id}