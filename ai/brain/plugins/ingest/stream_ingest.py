#!/usr/bin/env python3
"""
Stream Ingest Plugin for DSMIL Brain

Ingests streaming/chunked data sources.
"""

import hashlib
import queue
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Callable
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.ingest_framework import IngestPlugin, IngestResult


class StreamIngestPlugin(IngestPlugin):
    """
    Stream Ingest Plugin

    Ingests data from streaming sources.
    """

    @property
    def name(self) -> str:
        return "stream_ingest"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Ingest streaming/chunked data"

    @property
    def supported_types(self) -> List[str]:
        return ["stream", "generator", "iterator", "chunked"]

    def __init__(self):
        self._buffer_size = 1024 * 1024  # 1MB buffer
        self._chunk_timeout = 30.0
        self._active_streams: Dict[str, queue.Queue] = {}

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._buffer_size = config.get("buffer_size", self._buffer_size)
        self._chunk_timeout = config.get("chunk_timeout", self._chunk_timeout)
        return True

    def shutdown(self):
        """Stop all active streams"""
        for stream_id in list(self._active_streams.keys()):
            self.stop_stream(stream_id)

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        """
        Ingest from stream

        Args:
            source: Iterator, generator, or file-like object
            max_chunks: Maximum chunks to read (default: unlimited)
            callback: Optional callback for each chunk
        """
        try:
            max_chunks = kwargs.get("max_chunks", float("inf"))
            callback = kwargs.get("callback")

            # Get iterator
            if hasattr(source, "__iter__"):
                iterator = iter(source)
            elif hasattr(source, "read"):
                iterator = self._file_to_chunks(source)
            else:
                return IngestResult(
                    success=False,
                    plugin_name=self.name,
                    errors=["Source must be iterable or file-like"],
                )

            # Read chunks
            chunks = []
            total_bytes = 0
            chunk_count = 0

            for chunk in iterator:
                if chunk_count >= max_chunks:
                    break

                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")

                chunks.append(chunk)
                total_bytes += len(chunk)
                chunk_count += 1

                if callback:
                    callback(chunk, chunk_count)

            # Build result
            all_data = b"".join(chunks)

            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=all_data,
                metadata={
                    "chunks": chunk_count,
                    "total_bytes": total_bytes,
                    "checksum": hashlib.sha256(all_data).hexdigest(),
                },
                items_ingested=chunk_count,
                bytes_processed=total_bytes,
            )

        except Exception as e:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(e)],
            )

    def _file_to_chunks(self, file_obj, chunk_size: int = 65536) -> Iterator[bytes]:
        """Convert file-like object to chunk iterator"""
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def start_stream(self, stream_id: str,
                    source_factory: Callable[[], Iterator]) -> bool:
        """
        Start a background stream

        Args:
            stream_id: Unique ID for this stream
            source_factory: Callable that returns iterator
        """
        if stream_id in self._active_streams:
            return False

        data_queue: queue.Queue = queue.Queue(maxsize=100)
        self._active_streams[stream_id] = data_queue

        def stream_worker():
            try:
                source = source_factory()
                for chunk in source:
                    if stream_id not in self._active_streams:
                        break
                    try:
                        data_queue.put(chunk, timeout=self._chunk_timeout)
                    except queue.Full:
                        pass  # Drop if queue full
            finally:
                data_queue.put(None)  # Signal end

        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()

        return True

    def read_stream(self, stream_id: str, timeout: float = 5.0) -> Optional[bytes]:
        """Read next chunk from stream"""
        if stream_id not in self._active_streams:
            return None

        try:
            chunk = self._active_streams[stream_id].get(timeout=timeout)
            if chunk is None:
                # Stream ended
                self.stop_stream(stream_id)
            return chunk
        except queue.Empty:
            return b""

    def stop_stream(self, stream_id: str):
        """Stop a background stream"""
        self._active_streams.pop(stream_id, None)


# Export plugin class
Plugin = StreamIngestPlugin

