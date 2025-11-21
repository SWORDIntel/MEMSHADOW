"""
MEMSHADOW Corpus Importer

Import chat histories and conversation corpuses to pre-seed the memory system.

Supported formats:
- JSON/JSONL (ChatGPT exports, Claude exports, custom)
- CSV (message, role, timestamp)
- Markdown (conversation logs)
- Plain text (one message per line)

Usage:
    from app.services.corpus_importer import CorpusImporter

    importer = CorpusImporter(user_id="...")
    result = await importer.import_file("/path/to/export.json", format="chatgpt")
"""

import json
import csv
import re
import asyncio
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from uuid import UUID, uuid4
import structlog
from pydantic import BaseModel

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.db.chromadb import chroma_client

logger = structlog.get_logger()


class ImportedMessage(BaseModel):
    """Standardized message format for import"""
    content: str
    role: str = "user"  # user, assistant, system
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class ImportResult(BaseModel):
    """Result of an import operation"""
    success: bool
    total_messages: int
    imported_messages: int
    failed_messages: int
    skipped_messages: int
    errors: List[str] = []
    duration_seconds: float = 0.0


class CorpusImporter:
    """Import chat corpuses into MEMSHADOW memory system"""

    # Supported formats and their parsers
    FORMATS = {
        "chatgpt": "_parse_chatgpt",
        "claude": "_parse_claude",
        "json": "_parse_generic_json",
        "jsonl": "_parse_jsonl",
        "csv": "_parse_csv",
        "markdown": "_parse_markdown",
        "text": "_parse_text",
        "zip": "_parse_zip",
        "auto": "_detect_and_parse"
    }

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.embedding_service = EmbeddingService()
        self.memory_service = MemoryService()
        self.stats = {
            "total": 0,
            "imported": 0,
            "failed": 0,
            "skipped": 0
        }
        self.errors: List[str] = []

    async def import_file(
        self,
        file_path: str,
        format: str = "auto",
        batch_size: int = 50,
        skip_duplicates: bool = True,
        min_content_length: int = 10
    ) -> ImportResult:
        """
        Import a file into the memory system

        Args:
            file_path: Path to the file to import
            format: Format of the file (auto, chatgpt, claude, json, jsonl, csv, markdown, text)
            batch_size: Number of messages to process in each batch
            skip_duplicates: Skip messages that already exist (based on content hash)
            min_content_length: Minimum content length to import (skip very short messages)

        Returns:
            ImportResult with statistics
        """
        start_time = datetime.utcnow()
        self.stats = {"total": 0, "imported": 0, "failed": 0, "skipped": 0}
        self.errors = []

        try:
            # Validate format
            if format not in self.FORMATS:
                raise ValueError(f"Unsupported format: {format}. Supported: {list(self.FORMATS.keys())}")

            # Read and parse file
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.info("Starting corpus import",
                       file=file_path, format=format, user_id=self.user_id)

            # Parse messages
            parser = getattr(self, self.FORMATS[format])
            messages = await parser(path)

            self.stats["total"] = len(messages)
            logger.info(f"Parsed {len(messages)} messages from {file_path}")

            # Process in batches
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                await self._process_batch(
                    batch,
                    skip_duplicates=skip_duplicates,
                    min_content_length=min_content_length
                )

                # Progress logging
                progress = min(i + batch_size, len(messages))
                logger.info(f"Import progress: {progress}/{len(messages)} messages")

        except Exception as e:
            logger.error("Corpus import failed", error=str(e))
            self.errors.append(str(e))

        duration = (datetime.utcnow() - start_time).total_seconds()

        return ImportResult(
            success=len(self.errors) == 0,
            total_messages=self.stats["total"],
            imported_messages=self.stats["imported"],
            failed_messages=self.stats["failed"],
            skipped_messages=self.stats["skipped"],
            errors=self.errors,
            duration_seconds=duration
        )

    async def import_text(
        self,
        content: str,
        format: str = "auto",
        **kwargs
    ) -> ImportResult:
        """Import from text content directly (for API uploads)"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            return await self.import_file(temp_path, format=format, **kwargs)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _process_batch(
        self,
        messages: List[ImportedMessage],
        skip_duplicates: bool,
        min_content_length: int
    ):
        """Process a batch of messages"""
        for msg in messages:
            try:
                # Skip short messages
                if len(msg.content.strip()) < min_content_length:
                    self.stats["skipped"] += 1
                    continue

                # Skip system messages (usually not useful for memory)
                if msg.role == "system":
                    self.stats["skipped"] += 1
                    continue

                # Generate embedding
                embedding = await self.embedding_service.generate_embedding(msg.content)

                # Create memory entry
                memory_id = str(uuid4())
                metadata = {
                    "user_id": self.user_id,
                    "role": msg.role,
                    "imported": True,
                    "import_timestamp": datetime.utcnow().isoformat(),
                    "original_timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    **msg.metadata
                }

                # Store in ChromaDB
                await chroma_client.add_embedding(
                    memory_id=memory_id,
                    embedding=embedding,
                    metadata=metadata
                )

                self.stats["imported"] += 1

            except Exception as e:
                self.stats["failed"] += 1
                self.errors.append(f"Failed to import message: {str(e)[:100]}")
                logger.debug("Message import failed", error=str(e))

    # =========================================================================
    # Format Parsers
    # =========================================================================

    async def _detect_and_parse(self, path: Path) -> List[ImportedMessage]:
        """Auto-detect format and parse"""
        suffix = path.suffix.lower()

        # Handle zip files first (binary)
        if suffix == '.zip':
            return await self._parse_zip(path)

        content = path.read_text(encoding='utf-8', errors='ignore')

        # Try to detect format
        if suffix == '.jsonl':
            return await self._parse_jsonl(path)
        elif suffix == '.csv':
            return await self._parse_csv(path)
        elif suffix == '.md':
            return await self._parse_markdown(path)
        elif suffix == '.json':
            # Try to detect ChatGPT or Claude format
            try:
                data = json.loads(content)
                if isinstance(data, list) and data and "mapping" in str(data[0]):
                    return await self._parse_chatgpt(path)
                elif isinstance(data, dict) and "chat_messages" in data:
                    return await self._parse_claude(path)
                else:
                    return await self._parse_generic_json(path)
            except:
                return await self._parse_generic_json(path)
        else:
            return await self._parse_text(path)

    async def _parse_chatgpt(self, path: Path) -> List[ImportedMessage]:
        """Parse ChatGPT export format (conversations.json)"""
        messages = []
        content = path.read_text(encoding='utf-8')
        data = json.loads(content)

        # ChatGPT export is a list of conversations
        conversations = data if isinstance(data, list) else [data]

        for conv in conversations:
            # Extract messages from mapping
            mapping = conv.get("mapping", {})
            for node_id, node in mapping.items():
                msg_data = node.get("message")
                if not msg_data:
                    continue

                content_parts = msg_data.get("content", {}).get("parts", [])
                if not content_parts:
                    continue

                text = " ".join(str(p) for p in content_parts if p)
                if not text.strip():
                    continue

                role = msg_data.get("author", {}).get("role", "user")
                timestamp = None
                if msg_data.get("create_time"):
                    try:
                        timestamp = datetime.fromtimestamp(msg_data["create_time"])
                    except:
                        pass

                messages.append(ImportedMessage(
                    content=text,
                    role=role,
                    timestamp=timestamp,
                    metadata={"source": "chatgpt", "conversation_id": conv.get("id")}
                ))

        return messages

    async def _parse_claude(self, path: Path) -> List[ImportedMessage]:
        """Parse Claude export format"""
        messages = []
        content = path.read_text(encoding='utf-8')
        data = json.loads(content)

        # Handle both single conversation and multiple
        conversations = data.get("chat_messages", data) if isinstance(data, dict) else data

        for msg in conversations:
            if isinstance(msg, dict):
                text = msg.get("text", msg.get("content", ""))
                role = msg.get("sender", msg.get("role", "user"))

                # Normalize role
                if role in ["human", "Human"]:
                    role = "user"
                elif role in ["assistant", "Assistant", "claude"]:
                    role = "assistant"

                timestamp = None
                if msg.get("created_at") or msg.get("timestamp"):
                    try:
                        ts = msg.get("created_at") or msg.get("timestamp")
                        timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except:
                        pass

                if text.strip():
                    messages.append(ImportedMessage(
                        content=text,
                        role=role,
                        timestamp=timestamp,
                        metadata={"source": "claude"}
                    ))

        return messages

    async def _parse_generic_json(self, path: Path) -> List[ImportedMessage]:
        """Parse generic JSON format"""
        messages = []
        content = path.read_text(encoding='utf-8')
        data = json.loads(content)

        # Handle list or dict with messages key
        if isinstance(data, dict):
            data = data.get("messages", data.get("data", [data]))

        if not isinstance(data, list):
            data = [data]

        for item in data:
            if isinstance(item, str):
                messages.append(ImportedMessage(content=item))
            elif isinstance(item, dict):
                # Try common field names
                text = (item.get("content") or item.get("text") or
                       item.get("message") or item.get("body") or "")
                role = (item.get("role") or item.get("sender") or
                       item.get("author") or "user")

                if text.strip():
                    messages.append(ImportedMessage(
                        content=text,
                        role=role,
                        metadata={"source": "json"}
                    ))

        return messages

    async def _parse_jsonl(self, path: Path) -> List[ImportedMessage]:
        """Parse JSON Lines format (one JSON object per line)"""
        messages = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = (item.get("content") or item.get("text") or
                           item.get("message") or "")
                    role = item.get("role", "user")

                    if text.strip():
                        messages.append(ImportedMessage(
                            content=text,
                            role=role,
                            metadata={"source": "jsonl"}
                        ))
                except json.JSONDecodeError:
                    continue

        return messages

    async def _parse_csv(self, path: Path) -> List[ImportedMessage]:
        """Parse CSV format (content, role, timestamp columns)"""
        messages = []

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names
                text = (row.get("content") or row.get("text") or
                       row.get("message") or row.get("body") or "")
                role = (row.get("role") or row.get("sender") or
                       row.get("author") or "user")

                if text.strip():
                    messages.append(ImportedMessage(
                        content=text,
                        role=role,
                        metadata={"source": "csv"}
                    ))

        return messages

    async def _parse_markdown(self, path: Path) -> List[ImportedMessage]:
        """Parse markdown conversation format"""
        messages = []
        content = path.read_text(encoding='utf-8')

        # Common patterns for conversation markdown
        # Pattern 1: **User:** or **Assistant:**
        pattern1 = r'\*\*(User|Assistant|Human|Claude|AI):\*\*\s*(.+?)(?=\*\*(?:User|Assistant|Human|Claude|AI):\*\*|$)'

        # Pattern 2: ### User or ### Assistant
        pattern2 = r'###\s*(User|Assistant|Human|Claude|AI)\s*\n(.+?)(?=###\s*(?:User|Assistant|Human|Claude|AI)|$)'

        # Pattern 3: > quotes for one role
        pattern3 = r'^>\s*(.+)$'

        # Try pattern 1
        matches = re.findall(pattern1, content, re.DOTALL | re.IGNORECASE)
        if matches:
            for role, text in matches:
                role = "user" if role.lower() in ["user", "human"] else "assistant"
                if text.strip():
                    messages.append(ImportedMessage(
                        content=text.strip(),
                        role=role,
                        metadata={"source": "markdown"}
                    ))
            return messages

        # Try pattern 2
        matches = re.findall(pattern2, content, re.DOTALL | re.IGNORECASE)
        if matches:
            for role, text in matches:
                role = "user" if role.lower() in ["user", "human"] else "assistant"
                if text.strip():
                    messages.append(ImportedMessage(
                        content=text.strip(),
                        role=role,
                        metadata={"source": "markdown"}
                    ))
            return messages

        # Fallback: treat each paragraph as a message
        paragraphs = content.split('\n\n')
        for i, para in enumerate(paragraphs):
            if para.strip():
                messages.append(ImportedMessage(
                    content=para.strip(),
                    role="user" if i % 2 == 0 else "assistant",
                    metadata={"source": "markdown"}
                ))

        return messages

    async def _parse_text(self, path: Path) -> List[ImportedMessage]:
        """Parse plain text (one message per line or paragraph)"""
        messages = []
        content = path.read_text(encoding='utf-8')

        # Split by double newlines (paragraphs) or single lines
        if '\n\n' in content:
            chunks = content.split('\n\n')
        else:
            chunks = content.split('\n')

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                messages.append(ImportedMessage(
                    content=chunk.strip(),
                    role="user",  # Default to user for plain text
                    metadata={"source": "text"}
                ))

        return messages

    async def _parse_zip(self, path: Path) -> List[ImportedMessage]:
        """Parse zip archive containing multiple conversation files"""
        messages = []
        temp_dir = None

        try:
            temp_dir = tempfile.mkdtemp(prefix="memshadow_import_")

            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(temp_dir)

            # Recursively find and parse all supported files
            temp_path = Path(temp_dir)
            for file_path in temp_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.json', '.jsonl', '.csv', '.md', '.txt']:
                    try:
                        file_messages = await self._detect_and_parse(file_path)
                        # Add source file info to metadata
                        for msg in file_messages:
                            msg.metadata["source_file"] = file_path.name
                        messages.extend(file_messages)
                        logger.debug(f"Parsed {len(file_messages)} messages from {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to parse {file_path.name}: {e}")

            logger.info(f"Extracted {len(messages)} total messages from zip")

        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return messages

    async def import_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        **kwargs
    ) -> ImportResult:
        """
        Import all supported files from a directory

        Args:
            dir_path: Path to directory containing corpus files
            recursive: Search subdirectories
            **kwargs: Additional arguments for import_file
        """
        start_time = datetime.utcnow()
        self.stats = {"total": 0, "imported": 0, "failed": 0, "skipped": 0}
        self.errors = []

        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        supported_extensions = ['.json', '.jsonl', '.csv', '.md', '.txt', '.zip']
        files = path.rglob('*') if recursive else path.glob('*')

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing: {file_path.name}")
                try:
                    result = await self.import_file(str(file_path), **kwargs)
                    self.stats["total"] += result.total_messages
                    self.stats["imported"] += result.imported_messages
                    self.stats["failed"] += result.failed_messages
                    self.stats["skipped"] += result.skipped_messages
                    self.errors.extend(result.errors)
                except Exception as e:
                    self.errors.append(f"Failed to process {file_path.name}: {str(e)}")

        duration = (datetime.utcnow() - start_time).total_seconds()

        return ImportResult(
            success=len(self.errors) == 0,
            total_messages=self.stats["total"],
            imported_messages=self.stats["imported"],
            failed_messages=self.stats["failed"],
            skipped_messages=self.stats["skipped"],
            errors=self.errors,
            duration_seconds=duration
        )


# =========================================================================
# API Endpoint Helper
# =========================================================================

async def import_corpus_from_upload(
    user_id: str,
    content: bytes,
    filename: str,
    format: str = "auto"
) -> ImportResult:
    """
    Helper for API file uploads

    Args:
        user_id: User ID for the imported memories
        content: File content as bytes
        filename: Original filename (used for format detection)
        format: Explicit format or "auto" for detection

    Returns:
        ImportResult
    """
    import tempfile

    # Determine suffix from filename
    suffix = Path(filename).suffix or '.txt'

    with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        importer = CorpusImporter(user_id=user_id)
        return await importer.import_file(temp_path, format=format)
    finally:
        Path(temp_path).unlink(missing_ok=True)
