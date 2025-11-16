from uuid import UUID
import asyncio
import structlog
from celery import current_task
from typing import Dict, Any

from app.workers.celery_app import celery_app
from app.db.postgres import AsyncSessionLocal
from app.services.memory_service import MemoryService
from app.services.document_service import document_processor, DocumentProcessingError

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


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_task(self, file_content: bytes, filename: str, user_id: str) -> Dict[str, Any]:
    """
    Celery task to process a document file and create memories from it.

    This task:
    1. Processes the document (extraction, OCR, chunking)
    2. Creates individual memory records for each chunk
    3. Triggers embedding generation for each memory

    Args:
        file_content: Binary content of the document
        filename: Original filename
        user_id: ID of the user who uploaded the document

    Returns:
        Dictionary with processing results
    """
    logger.info(
        "Starting document processing task",
        filename=filename,
        user_id=user_id,
        task_id=current_task.request.id,
        file_size=len(file_content)
    )

    async def async_process_document():
        session = AsyncSessionLocal()
        memory_ids = []

        try:
            # Update task state to PROGRESS
            self.update_state(
                state='PROGRESS',
                meta={'status': 'Processing document', 'progress': 10}
            )

            # Process document
            result = await document_processor.process_document(
                file_content=file_content,
                filename=filename,
                user_id=user_id
            )

            self.update_state(
                state='PROGRESS',
                meta={'status': 'Document processed, creating memories', 'progress': 40}
            )

            # Extract document metadata
            doc_metadata = result.get('document_metadata', {})
            doc_structure = result.get('structure', {})

            # Create memory service
            memory_service = MemoryService(session)

            # Determine chunking strategy based on document type
            chunks = result.get('chunks', [])

            if not chunks and result.get('text'):
                # If no chunks provided, create one chunk with all text
                chunks = [{'text': result['text'], 'index': 0}]

            total_chunks = len(chunks)
            logger.info(f"Creating {total_chunks} memory chunks from document", filename=filename)

            # Create a memory for each chunk
            for idx, chunk in enumerate(chunks):
                try:
                    chunk_text = chunk.get('text', '')

                    if not chunk_text or len(chunk_text.strip()) < 10:
                        # Skip very small or empty chunks
                        continue

                    # Prepare extra_data with document metadata
                    extra_data = {
                        'source': 'document',
                        'document_filename': filename,
                        'document_type': doc_structure.get('type'),
                        'document_metadata': doc_metadata,
                        'chunk_index': idx,
                        'total_chunks': total_chunks,
                    }

                    # Add chunk-specific metadata
                    if 'page' in chunk:
                        extra_data['page_number'] = chunk['page']
                    if 'slide_number' in chunk:
                        extra_data['slide_number'] = chunk['slide_number']
                    if 'sheet_name' in chunk:
                        extra_data['sheet_name'] = chunk.get('name', chunk.get('sheet_name'))

                    # Create memory
                    memory = await memory_service.create_memory(
                        user_id=UUID(user_id),
                        content=chunk_text,
                        extra_data=extra_data
                    )

                    memory_ids.append(str(memory.id))

                    # Dispatch embedding generation task for this chunk
                    generate_embedding_task.delay(
                        memory_id=str(memory.id),
                        content=chunk_text
                    )

                    # Update progress
                    progress = 40 + int((idx + 1) / total_chunks * 50)
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'status': f'Created memory {idx + 1}/{total_chunks}',
                            'progress': progress
                        }
                    )

                except Exception as chunk_error:
                    logger.error(
                        "Failed to create memory for chunk",
                        filename=filename,
                        chunk_index=idx,
                        error=str(chunk_error)
                    )
                    # Continue processing other chunks

            await session.commit()

            # Handle images if present
            images = result.get('images', [])
            if images:
                logger.info(f"Document contains {len(images)} images", filename=filename)
                # Optionally create memories for image OCR text
                for img_idx, img in enumerate(images):
                    ocr_text = img.get('ocr_text', '')
                    if ocr_text and len(ocr_text.strip()) > 20:
                        try:
                            extra_data = {
                                'source': 'document_image',
                                'document_filename': filename,
                                'image_index': img_idx,
                                'image_page': img.get('page'),
                                'image_format': img.get('format'),
                            }

                            memory = await memory_service.create_memory(
                                user_id=UUID(user_id),
                                content=f"[Image OCR] {ocr_text}",
                                extra_data=extra_data
                            )

                            memory_ids.append(str(memory.id))

                            generate_embedding_task.delay(
                                memory_id=str(memory.id),
                                content=ocr_text
                            )

                        except Exception as img_error:
                            logger.error(
                                "Failed to create memory for image OCR",
                                filename=filename,
                                image_index=img_idx,
                                error=str(img_error)
                            )

            logger.info(
                "Document processing task completed successfully",
                filename=filename,
                user_id=user_id,
                memories_created=len(memory_ids)
            )

            return {
                "status": "success",
                "filename": filename,
                "user_id": user_id,
                "memories_created": len(memory_ids),
                "memory_ids": memory_ids,
                "document_type": doc_structure.get('type'),
                "total_chunks": total_chunks,
                "metadata": doc_metadata
            }

        except DocumentProcessingError as e:
            logger.error(
                "Document processing failed",
                filename=filename,
                error=str(e),
                exc_info=True
            )
            await session.rollback()
            raise

        except Exception as e:
            logger.error(
                "Unexpected error during document processing",
                filename=filename,
                error=str(e),
                exc_info=True
            )
            await session.rollback()
            # Retry the task
            raise self.retry(exc=e)

        finally:
            await session.close()

    # Run the async function
    result = asyncio.run(async_process_document())
    return result