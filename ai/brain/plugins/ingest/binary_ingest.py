#!/usr/bin/env python3
"""
Binary Ingest Plugin for DSMIL Brain

Ingests raw binary data with analysis and chunking.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.ingest_framework import IngestPlugin, IngestResult
from data.raw_binary_handler import RawBinaryHandler, BinaryTypeHint


class BinaryIngestPlugin(IngestPlugin):
    """
    Binary Ingest Plugin

    Ingests raw binary data with type detection and analysis.
    """

    @property
    def name(self) -> str:
        return "binary_ingest"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Ingest and analyze raw binary data"

    @property
    def supported_types(self) -> List[str]:
        return ["binary", "bytes", "executable", "malware"]

    def __init__(self):
        self._handler = RawBinaryHandler()
        self._chunk_size = 65536

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._chunk_size = config.get("chunk_size", self._chunk_size)
        return True

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        """
        Ingest binary data

        Args:
            source: bytes, file path, or file-like object
            chunk: Whether to return chunked data
        """
        try:
            # Get bytes
            if isinstance(source, bytes):
                data = source
            elif isinstance(source, (str, Path)):
                with open(source, "rb") as f:
                    data = f.read()
            elif hasattr(source, "read"):
                data = source.read()
            else:
                return IngestResult(
                    success=False,
                    plugin_name=self.name,
                    errors=["Unsupported source type"],
                )

            # Analyze
            metadata_obj = self._handler.analyze(data)

            # Build result metadata
            metadata = {
                "size": metadata_obj.size,
                "checksum": metadata_obj.checksum,
                "type_hint": metadata_obj.type_hint.name,
                "entropy": metadata_obj.entropy,
                "magic_bytes": metadata_obj.magic_bytes.hex()[:32],
                "likely_encrypted": self._handler.is_likely_encrypted(data),
                "likely_compressed": self._handler.is_likely_compressed(data),
            }

            # Chunk if requested
            if kwargs.get("chunk", False):
                chunks = self._handler.chunk(data, self._chunk_size)
                result_data = {
                    "chunks": [
                        {
                            "id": c.chunk_id,
                            "sequence": c.sequence,
                            "data": c.data,
                            "checksum": c.checksum,
                        }
                        for c in chunks
                    ],
                    "total_chunks": len(chunks),
                }
            else:
                result_data = data

            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=result_data,
                metadata=metadata,
                items_ingested=1,
                bytes_processed=len(data),
            )

        except Exception as e:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(e)],
            )


# Export plugin class
Plugin = BinaryIngestPlugin

