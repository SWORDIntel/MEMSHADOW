# Brain Ingest Plugins

Plugins for ingesting data into the DSMIL Brain memory system.

## MEMSHADOW Ingest Plugin

Parses MEMSHADOW protocol v2 binary messages and extracts psychological events.

### Usage

```python
from ai.brain.plugins.ingest.memshadow_ingest import MemshadowIngestPlugin

plugin = MemshadowIngestPlugin()

# Parse MEMSHADOW message
parsed = plugin.parse_memshadow_message(binary_data)

# Extract psych events
events = plugin.extract_psych_events(binary_data)
```

### Message Format

Expects MEMSHADOW v2 format:
- 32-byte header (`MemshadowHeader`)
- 64-byte psych events (`PsychEvent`)
- Optional batch of multiple events

### Integration

Used by:
- `ai/brain/api/shrink_endpoint.py` - HTTP endpoint for SHRINK
- `external/intel/shrink/shrink/kernel_receiver.py` - Kernel module receiver

### Output Format

Parsed events are converted to Brain ingest format:

```python
{
    "type": "psych_event",
    "session_id": 12345,
    "timestamp_ns": 1234567890,
    "scores": {
        "acute_stress": 0.5,
        "machiavellianism": 0.3,
        "narcissism": 0.4,
        "psychopathy": 0.2,
        "burnout": 0.1,
        "espionage": 0.05
    },
    "confidence": 0.85,
    "event_type": "SCORE_UPDATE"
}
```

## Binary Ingest Plugin

Legacy binary ingest plugin for DSMIL Binary Container format.

See `binary_ingest.py` for details.

## Plugin Framework

All ingest plugins implement the `IngestPlugin` interface:

```python
class IngestPlugin:
    def ingest(self, data: bytes, metadata: Dict) -> List[Dict]:
        """Ingest data and return structured records"""
        pass
```

## Related Documentation

- [MEMSHADOW Protocol](../../../../libs/memshadow-protocol/README.md)
- [Brain API](../../api/README.md)
- [Ingest Framework](ingest_framework.py)
