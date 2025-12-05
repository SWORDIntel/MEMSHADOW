#!/usr/bin/env python3
"""
DSMIL Brain Data Layer

Data format handling and binary containers:
- Format handlers for text, structured, binary, vector, graph, stream
- DSMIL Binary Container (DSMIL-BC) format
- Raw binary handling
"""

from .formats import (
    DataFormat,
    FormatHandler,
    TextHandler,
    StructuredHandler,
    VectorHandler,
    GraphHandler,
    StreamHandler,
)

from .binary_container import (
    DSMILBinaryContainer,
    ContainerFlags,
    ContainerHeader,
)

from .raw_binary_handler import (
    RawBinaryHandler,
    BinaryTypeHint,
    BinaryChunk,
)

__all__ = [
    "DataFormat", "FormatHandler", "TextHandler", "StructuredHandler",
    "VectorHandler", "GraphHandler", "StreamHandler",
    "DSMILBinaryContainer", "ContainerFlags", "ContainerHeader",
    "RawBinaryHandler", "BinaryTypeHint", "BinaryChunk",
]

