#!/usr/bin/env python3
"""
File Ingest Plugin for DSMIL Brain

Ingests local files with format detection and metadata extraction.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.ingest_framework import IngestPlugin, IngestResult


class FileIngestPlugin(IngestPlugin):
    """
    File Ingest Plugin

    Ingests files from local filesystem.
    """

    @property
    def name(self) -> str:
        return "file_ingest"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Ingest files from local filesystem"

    @property
    def supported_types(self) -> List[str]:
        return ["file", "directory"]

    def __init__(self):
        self._max_file_size = 100 * 1024 * 1024  # 100MB default
        self._allowed_extensions: Optional[set] = None

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._max_file_size = config.get("max_file_size", self._max_file_size)
        if "allowed_extensions" in config:
            self._allowed_extensions = set(config["allowed_extensions"])
        return True

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        """
        Ingest a file or directory

        Args:
            source: File path or Path object
            recursive: For directories, recurse into subdirs
        """
        path = Path(source) if isinstance(source, str) else source

        if not path.exists():
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[f"Path does not exist: {path}"],
            )

        if path.is_file():
            return self._ingest_file(path)
        elif path.is_dir():
            return self._ingest_directory(path, kwargs.get("recursive", False))

        return IngestResult(
            success=False,
            plugin_name=self.name,
            errors=["Unknown path type"],
        )

    def _ingest_file(self, path: Path) -> IngestResult:
        """Ingest a single file"""
        try:
            # Check extension
            if self._allowed_extensions:
                if path.suffix.lower() not in self._allowed_extensions:
                    return IngestResult(
                        success=False,
                        plugin_name=self.name,
                        errors=[f"Extension not allowed: {path.suffix}"],
                    )

            # Check size
            size = path.stat().st_size
            if size > self._max_file_size:
                return IngestResult(
                    success=False,
                    plugin_name=self.name,
                    errors=[f"File too large: {size} bytes"],
                )

            # Read file
            with open(path, "rb") as f:
                data = f.read()

            # Get metadata
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(str(path))

            metadata = {
                "path": str(path.absolute()),
                "name": path.name,
                "extension": path.suffix,
                "size": size,
                "mime_type": mime_type,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "checksum": hashlib.sha256(data).hexdigest(),
            }

            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=data,
                metadata=metadata,
                items_ingested=1,
                bytes_processed=size,
            )

        except Exception as e:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(e)],
            )

    def _ingest_directory(self, path: Path, recursive: bool) -> IngestResult:
        """Ingest all files in directory"""
        all_data = []
        all_metadata = []
        total_bytes = 0
        errors = []

        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue

            result = self._ingest_file(file_path)
            if result.success:
                all_data.append(result.data)
                all_metadata.append(result.metadata)
                total_bytes += result.bytes_processed
            else:
                errors.extend(result.errors)

        return IngestResult(
            success=len(all_data) > 0,
            plugin_name=self.name,
            data=all_data,
            metadata={
                "files": all_metadata,
                "directory": str(path.absolute()),
                "recursive": recursive,
            },
            items_ingested=len(all_data),
            bytes_processed=total_bytes,
            errors=errors,
        )


# Export plugin class
Plugin = FileIngestPlugin

