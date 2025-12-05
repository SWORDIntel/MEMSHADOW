#!/usr/bin/env python3
"""
Data Format Handlers for DSMIL Brain

Handles various data formats:
- Text (UTF-8 documents)
- Structured (JSON/YAML/XML)
- Vector (pre-computed embeddings)
- Graph (serialized knowledge graph)
- Stream (chunked data)
"""

import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Iterator
from datetime import datetime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats"""
    TEXT = auto()
    STRUCTURED = auto()
    BINARY_RAW = auto()
    BINARY_CONTAINER = auto()
    VECTOR = auto()
    GRAPH = auto()
    STREAM = auto()


@dataclass
class ParsedData:
    """Result of parsing data"""
    format: DataFormat
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    parsed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FormatHandler(ABC):
    """Base class for format handlers"""

    @property
    @abstractmethod
    def format(self) -> DataFormat:
        """Return the format this handler supports"""
        pass

    @abstractmethod
    def parse(self, data: bytes, **kwargs) -> ParsedData:
        """Parse data into structured form"""
        pass

    @abstractmethod
    def serialize(self, content: Any, **kwargs) -> bytes:
        """Serialize content to bytes"""
        pass

    def checksum(self, data: bytes) -> str:
        """Calculate checksum"""
        return hashlib.sha256(data).hexdigest()


class TextHandler(FormatHandler):
    """Handler for text data"""

    @property
    def format(self) -> DataFormat:
        return DataFormat.TEXT

    def parse(self, data: bytes, encoding: str = "utf-8", **kwargs) -> ParsedData:
        try:
            text = data.decode(encoding)
            return ParsedData(
                format=DataFormat.TEXT,
                content=text,
                metadata={
                    "encoding": encoding,
                    "length": len(text),
                    "lines": text.count("\n") + 1,
                },
                checksum=self.checksum(data),
            )
        except UnicodeDecodeError as e:
            return ParsedData(
                format=DataFormat.TEXT,
                content=None,
                metadata={"error": str(e)},
            )

    def serialize(self, content: str, encoding: str = "utf-8", **kwargs) -> bytes:
        return content.encode(encoding)


class StructuredHandler(FormatHandler):
    """Handler for structured data (JSON, YAML)"""

    @property
    def format(self) -> DataFormat:
        return DataFormat.STRUCTURED

    def parse(self, data: bytes, format_hint: str = "json", **kwargs) -> ParsedData:
        try:
            text = data.decode("utf-8")

            if format_hint == "json":
                content = json.loads(text)
            elif format_hint == "yaml":
                # Would use pyyaml
                content = json.loads(text)  # Fallback
            else:
                content = json.loads(text)

            return ParsedData(
                format=DataFormat.STRUCTURED,
                content=content,
                metadata={
                    "source_format": format_hint,
                    "type": type(content).__name__,
                },
                checksum=self.checksum(data),
            )
        except Exception as e:
            return ParsedData(
                format=DataFormat.STRUCTURED,
                content=None,
                metadata={"error": str(e)},
            )

    def serialize(self, content: Any, format_hint: str = "json", **kwargs) -> bytes:
        if format_hint == "json":
            return json.dumps(content, indent=2).encode("utf-8")
        return json.dumps(content).encode("utf-8")


class VectorHandler(FormatHandler):
    """Handler for vector data"""

    @property
    def format(self) -> DataFormat:
        return DataFormat.VECTOR

    def parse(self, data: bytes, **kwargs) -> ParsedData:
        import struct

        try:
            # Assume float32 packed format
            num_floats = len(data) // 4
            vector = list(struct.unpack(f'{num_floats}f', data[:num_floats * 4]))

            return ParsedData(
                format=DataFormat.VECTOR,
                content=vector,
                metadata={
                    "dimensions": len(vector),
                    "dtype": "float32",
                },
                checksum=self.checksum(data),
            )
        except Exception as e:
            return ParsedData(
                format=DataFormat.VECTOR,
                content=None,
                metadata={"error": str(e)},
            )

    def serialize(self, content: List[float], **kwargs) -> bytes:
        import struct
        return struct.pack(f'{len(content)}f', *content)


class GraphHandler(FormatHandler):
    """Handler for knowledge graph data"""

    @property
    def format(self) -> DataFormat:
        return DataFormat.GRAPH

    def parse(self, data: bytes, **kwargs) -> ParsedData:
        try:
            # Expect JSON serialized graph
            text = data.decode("utf-8")
            graph_data = json.loads(text)

            # Validate structure
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            return ParsedData(
                format=DataFormat.GRAPH,
                content=graph_data,
                metadata={
                    "nodes": len(nodes),
                    "edges": len(edges),
                },
                checksum=self.checksum(data),
            )
        except Exception as e:
            return ParsedData(
                format=DataFormat.GRAPH,
                content=None,
                metadata={"error": str(e)},
            )

    def serialize(self, content: Dict, **kwargs) -> bytes:
        return json.dumps(content).encode("utf-8")


class StreamHandler(FormatHandler):
    """Handler for streaming data"""

    @property
    def format(self) -> DataFormat:
        return DataFormat.STREAM

    def parse(self, data: bytes, chunk_size: int = 4096, **kwargs) -> ParsedData:
        """Parse returns chunk generator"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])

        return ParsedData(
            format=DataFormat.STREAM,
            content=chunks,
            metadata={
                "total_size": len(data),
                "chunk_size": chunk_size,
                "num_chunks": len(chunks),
            },
            checksum=self.checksum(data),
        )

    def serialize(self, content: List[bytes], **kwargs) -> bytes:
        return b"".join(content)

    def iter_chunks(self, data: bytes, chunk_size: int = 4096) -> Iterator[bytes]:
        """Iterate over data in chunks"""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]


# Format registry
FORMAT_HANDLERS: Dict[DataFormat, FormatHandler] = {
    DataFormat.TEXT: TextHandler(),
    DataFormat.STRUCTURED: StructuredHandler(),
    DataFormat.VECTOR: VectorHandler(),
    DataFormat.GRAPH: GraphHandler(),
    DataFormat.STREAM: StreamHandler(),
}


def get_handler(format: DataFormat) -> Optional[FormatHandler]:
    """Get handler for format"""
    return FORMAT_HANDLERS.get(format)


def detect_format(data: bytes) -> DataFormat:
    """Attempt to detect data format"""
    # Try JSON
    try:
        text = data.decode("utf-8")
        json.loads(text)
        return DataFormat.STRUCTURED
    except:
        pass

    # Try text
    try:
        data.decode("utf-8")
        return DataFormat.TEXT
    except:
        pass

    # Default to binary
    return DataFormat.BINARY_RAW


if __name__ == "__main__":
    print("Data Formats Self-Test")
    print("=" * 50)

    print("\n[1] Text Handler")
    text_handler = TextHandler()
    text_data = b"Hello, DSMIL Brain!\nLine 2"
    parsed = text_handler.parse(text_data)
    print(f"    Content: {parsed.content}")
    print(f"    Metadata: {parsed.metadata}")

    print("\n[2] Structured Handler")
    struct_handler = StructuredHandler()
    json_data = b'{"entity": "APT29", "confidence": 0.85}'
    parsed = struct_handler.parse(json_data)
    print(f"    Content: {parsed.content}")
    print(f"    Metadata: {parsed.metadata}")

    print("\n[3] Vector Handler")
    import struct
    vector_handler = VectorHandler()
    vector = [0.1, 0.2, 0.3, 0.4]
    vector_data = struct.pack(f'{len(vector)}f', *vector)
    parsed = vector_handler.parse(vector_data)
    print(f"    Content: {parsed.content}")
    print(f"    Metadata: {parsed.metadata}")

    print("\n[4] Graph Handler")
    graph_handler = GraphHandler()
    graph_data = b'{"nodes": [{"id": "A"}, {"id": "B"}], "edges": [{"from": "A", "to": "B"}]}'
    parsed = graph_handler.parse(graph_data)
    print(f"    Nodes: {parsed.metadata['nodes']}, Edges: {parsed.metadata['edges']}")

    print("\n[5] Stream Handler")
    stream_handler = StreamHandler()
    large_data = b"X" * 10000
    parsed = stream_handler.parse(large_data, chunk_size=4096)
    print(f"    Chunks: {parsed.metadata['num_chunks']}")

    print("\n[6] Format Detection")
    test_cases = [
        (b'{"key": "value"}', "JSON"),
        (b"Plain text", "Text"),
        (b"\x00\x01\x02\x03", "Binary"),
    ]
    for data, expected in test_cases:
        detected = detect_format(data)
        print(f"    {expected} -> {detected.name}")

    print("\n" + "=" * 50)
    print("Data Formats test complete")

