from uuid import UUID
import asyncio
import structlog
from pathlib import Path
from celery import current_task

from app.workers.celery_app import celery_app
from app.db.postgres import AsyncSessionLocal
from app.services.memory_service import MemoryService
from app.services.task_reminder_service import TaskReminderService
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


@celery_app.task(bind=True)
def check_and_send_reminders(self):
    """
    Periodic task to check for due reminders and process them.
    Runs every 5 minutes to check for pending reminders.
    """
    logger.info("Starting reminder check task", task_id=current_task.request.id)

    async def async_check_reminders():
        session = AsyncSessionLocal()
        try:
            service = TaskReminderService(session)

            # Get all pending reminders that should be sent now
            pending_reminders = await service.get_pending_reminders()

            if not pending_reminders:
                logger.debug("No pending reminders to process")
                return {"status": "success", "reminders_sent": 0}

            logger.info(f"Found {len(pending_reminders)} reminders to process")

            sent_count = 0
            for reminder in pending_reminders:
                try:
                    # In a real implementation, you would send actual notifications here
                    # For example:
                    # - Send email via SMTP
                    # - Send push notification
                    # - Create in-app notification
                    # - Send webhook

                    # For now, we'll just log and mark as reminded
                    logger.info(
                        "Reminder notification sent",
                        reminder_id=str(reminder.id),
                        user_id=str(reminder.user_id),
                        title=reminder.title,
                        priority=reminder.priority.value
                    )

                    # Mark reminder as sent
                    await service.mark_as_reminded(reminder.id)
                    sent_count += 1

                    # Store notification in extra_data for retrieval via API
                    extra_data = reminder.extra_data or {}
                    extra_data["notification_sent"] = True
                    extra_data["notification_method"] = "log"  # Would be 'email', 'push', etc.
                    await session.commit()

                except Exception as e:
                    logger.error(
                        "Failed to process reminder",
                        reminder_id=str(reminder.id),
                        error=str(e),
                        exc_info=True
                    )
                    await session.rollback()

            await session.commit()
            logger.info("Reminder check task completed", reminders_sent=sent_count)

            return {"status": "success", "reminders_sent": sent_count}

        except Exception as e:
            logger.error(
                "Error during reminder check task",
                error=str(e),
                exc_info=True
            )
            await session.rollback()
            raise
        finally:
            await session.close()

    return asyncio.run(async_check_reminders())