#!/usr/bin/env python3
"""
MEMSHADOW Embedding Migration Script

Migrates existing embeddings from old dimension (768d) to new dimension (2048d).
This script:
1. Fetches all existing memories from PostgreSQL
2. Regenerates embeddings using the new model/dimension
3. Updates ChromaDB with new embeddings
4. Preserves all metadata and content

Usage:
    python scripts/migrate_embeddings_to_2048d.py [--dry-run] [--batch-size 100]

Options:
    --dry-run       Show what would be migrated without making changes
    --batch-size    Number of embeddings to process per batch (default: 100)
    --force         Skip confirmation prompt
"""

import os
import sys
import asyncio
import argparse
from typing import List
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment before importing app modules
from dotenv import load_dotenv
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import structlog

from app.core.config import settings
from app.models.memory import Memory
from app.services.embedding_service import EmbeddingService
from app.db.chromadb import chroma_client

logger = structlog.get_logger()


class EmbeddingMigrator:
    """Handles migration of embeddings to new dimensions"""

    def __init__(self, dry_run: bool = False, batch_size: int = 100):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.embedding_service = None
        self.db_session = None
        self.stats = {
            "total_memories": 0,
            "migrated": 0,
            "failed": 0,
            "skipped": 0
        }

    async def initialize(self):
        """Initialize database connections and services"""
        logger.info("Initializing migration service")

        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        model_info = self.embedding_service.get_model_info()
        logger.info("Embedding model initialized", **model_info)

        # Initialize ChromaDB
        if not self.dry_run:
            await chroma_client.init_client()
            logger.info("ChromaDB client initialized")

        # Initialize PostgreSQL connection
        engine = create_async_engine(str(settings.DATABASE_URL))
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        self.db_session = async_session()

    async def get_all_memories(self) -> List[Memory]:
        """Fetch all memories from PostgreSQL"""
        logger.info("Fetching all memories from database")

        async with self.db_session as session:
            result = await session.execute(
                select(Memory).order_by(Memory.created_at)
            )
            memories = result.scalars().all()

        self.stats["total_memories"] = len(memories)
        logger.info(f"Found {len(memories)} memories to process")

        return memories

    async def migrate_memory(self, memory: Memory) -> bool:
        """
        Migrate a single memory's embedding

        Args:
            memory: Memory object to migrate

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate new embedding with current configuration
            logger.debug(
                "Generating new embedding",
                memory_id=str(memory.id),
                content_length=len(memory.content)
            )

            new_embedding = await self.embedding_service.generate_embedding(
                text=memory.content,
                use_cache=False  # Don't use cache during migration
            )

            if self.dry_run:
                logger.info(
                    "[DRY RUN] Would update embedding",
                    memory_id=str(memory.id),
                    old_dimension="unknown",
                    new_dimension=len(new_embedding)
                )
                return True

            # Update ChromaDB with new embedding
            metadata = {
                "user_id": str(memory.user_id),
                "created_at": memory.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "migrated": True,
                "migration_date": datetime.utcnow().isoformat()
            }

            # Delete old embedding if exists
            try:
                await chroma_client.delete_embedding(str(memory.id))
            except Exception:
                pass  # Ignore if doesn't exist

            # Add new embedding
            await chroma_client.add_embedding(
                memory_id=str(memory.id),
                embedding=new_embedding,
                metadata=metadata
            )

            logger.debug(
                "Successfully migrated embedding",
                memory_id=str(memory.id),
                dimension=len(new_embedding)
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to migrate embedding",
                memory_id=str(memory.id),
                error=str(e)
            )
            return False

    async def migrate_batch(self, memories: List[Memory]):
        """Migrate a batch of memories"""
        logger.info(f"Processing batch of {len(memories)} memories")

        for i, memory in enumerate(memories):
            success = await self.migrate_memory(memory)

            if success:
                self.stats["migrated"] += 1
            else:
                self.stats["failed"] += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(memories)} "
                    f"(Success: {self.stats['migrated']}, Failed: {self.stats['failed']})"
                )

    async def run_migration(self):
        """Execute the full migration process"""
        start_time = datetime.utcnow()

        logger.info("=" * 60)
        logger.info("MEMSHADOW Embedding Migration")
        logger.info("=" * 60)
        logger.info(f"Target dimension: {settings.EMBEDDING_DIMENSION}")
        logger.info(f"Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"Backend: {settings.EMBEDDING_BACKEND}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 60)

        try:
            # Initialize services
            await self.initialize()

            # Get all memories
            memories = await self.get_all_memories()

            if not memories:
                logger.warning("No memories found to migrate")
                return

            # Process in batches
            for i in range(0, len(memories), self.batch_size):
                batch = memories[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(memories) + self.batch_size - 1) // self.batch_size

                logger.info(f"\nProcessing batch {batch_num}/{total_batches}")
                await self.migrate_batch(batch)

            # Print summary
            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info("\n" + "=" * 60)
            logger.info("Migration Summary")
            logger.info("=" * 60)
            logger.info(f"Total memories: {self.stats['total_memories']}")
            logger.info(f"Successfully migrated: {self.stats['migrated']}")
            logger.info(f"Failed: {self.stats['failed']}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Average: {duration / max(self.stats['total_memories'], 1):.2f} sec/memory")
            logger.info("=" * 60)

            if self.dry_run:
                logger.info("\n⚠️  This was a DRY RUN - no changes were made")
                logger.info("Run without --dry-run to perform the actual migration")

        except Exception as e:
            logger.error("Migration failed", error=str(e))
            raise

        finally:
            # Cleanup
            if self.db_session:
                await self.db_session.close()
            if not self.dry_run and chroma_client:
                await chroma_client.close_client()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate MEMSHADOW embeddings to new dimensions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of embeddings to process per batch"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Confirmation prompt
    if not args.dry_run and not args.force:
        print("\n⚠️  WARNING: This will regenerate ALL embeddings in the system")
        print(f"Target dimension: {settings.EMBEDDING_DIMENSION}")
        print(f"Model: {settings.EMBEDDING_MODEL}")
        print(f"Backend: {settings.EMBEDDING_BACKEND}")
        response = input("\nContinue? (yes/no): ")

        if response.lower() != "yes":
            print("Migration cancelled")
            return

    # Run migration
    migrator = EmbeddingMigrator(
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )

    await migrator.run_migration()


if __name__ == "__main__":
    asyncio.run(main())
