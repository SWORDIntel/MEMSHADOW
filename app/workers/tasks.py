from uuid import UUID
import asyncio
import structlog
from pathlib import Path
from celery import current_task

from app.workers.celery_app import celery_app
from app.db.postgres import AsyncSessionLocal
from app.services.memory_service import MemoryService
from app.core.config import settings

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


@celery_app.task(bind=True)
def autoscan_corpus_directory(self):
    """
    Hourly task to scan corpus directory for new files and import them.
    Files are moved to 'processed' subdirectory after import.
    """
    corpus_dir = getattr(settings, 'CORPUS_IMPORT_DIR', '/data/corpus')
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        logger.debug("Corpus directory does not exist, skipping", path=corpus_dir)
        return {"status": "skipped", "reason": "directory_not_found"}

    async def async_scan():
        from app.services.corpus_importer import CorpusImporter
        import shutil

        processed_dir = corpus_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        supported_extensions = ['.json', '.jsonl', '.csv', '.md', '.txt', '.zip']
        files_found = list(corpus_path.glob('*'))
        files_to_process = [f for f in files_found if f.is_file() and f.suffix.lower() in supported_extensions]

        if not files_to_process:
            logger.debug("No new corpus files to import")
            return {"status": "success", "files_processed": 0}

        logger.info(f"Autoscan found {len(files_to_process)} files to import")

        total_imported = 0
        for file_path in files_to_process:
            try:
                # Use system user for autoscan imports
                importer = CorpusImporter(user_id="system")
                result = await importer.import_file(str(file_path))

                total_imported += result.imported_messages
                logger.info(
                    "Autoscan imported file",
                    file=file_path.name,
                    imported=result.imported_messages
                )

                # Move to processed directory
                shutil.move(str(file_path), str(processed_dir / file_path.name))

            except Exception as e:
                logger.error(f"Autoscan failed for {file_path.name}: {e}")

        return {"status": "success", "files_processed": len(files_to_process), "messages_imported": total_imported}

    return asyncio.run(async_scan())