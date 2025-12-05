# Brain API

HTTP endpoints for external integrations with the DSMIL Brain.

## SHRINK Intel Endpoint

### Endpoint

`POST /api/v1/ingest/shrink`

### Purpose

Receive psychological intelligence data from SHRINK kernel module or userspace components.

### Request Formats

#### JSON Format

```json
{
    "type": "psych_assessment",
    "scores": {
        "dark_triad": {
            "machiavellianism": 0.5,
            "narcissism": 0.4,
            "psychopathy": 0.3
        },
        "risk": {
            "espionage": 0.2,
            "burnout": 0.1
        }
    },
    "timestamp": 1234567890.0
}
```

#### Binary MEMSHADOW Format

Raw binary data with MEMSHADOW v2 header (32 bytes) + psych event (64 bytes).

### Response

```json
{
    "status": "success",
    "items_ingested": 1,
    "stored_in_tiers": ["L1", "L2"]
}
```

### Usage

#### Flask

```python
from ai.brain.api.shrink_endpoint import create_flask_endpoint

app = Flask(__name__)
shrink_endpoint = create_flask_endpoint()
app.register_blueprint(shrink_endpoint, url_prefix='/api/v1')
```

#### FastAPI

```python
from ai.brain.api.shrink_endpoint import create_fastapi_router
from fastapi import FastAPI

app = FastAPI()
router = create_fastapi_router()
app.include_router(router, prefix='/api/v1')
```

### Integration Flow

```
SHRINK Kernel Module
    │ (Netlink, MEMSHADOW binary)
    ▼
kernel_receiver.py
    │ (HTTP POST)
    ▼
/api/v1/ingest/shrink
    │ (MEMSHADOW ingest plugin)
    ▼
Brain Memory Tiers
    │ (Significant updates)
    ▼
Hub Orchestrator
    │ (Mesh broadcast)
    ▼
Other Nodes
```

## Implementation

### ShrinkIntelEndpoint

```python
class ShrinkIntelEndpoint:
    def ingest_shrink_intel(self, data: bytes, content_type: str) -> Dict
    def handle_binary_protocol(self, data: bytes) -> Dict
    def handle_json_protocol(self, data: bytes) -> Dict
```

### Error Handling

- Invalid format: Returns 400 Bad Request
- Parse error: Returns 422 Unprocessable Entity
- Storage error: Returns 500 Internal Server Error

## Security

- Authentication: TBD (API keys, certificates)
- Rate limiting: TBD
- Input validation: Validates MEMSHADOW header magic and structure

## Related Documentation

- [MEMSHADOW Protocol](../../../libs/memshadow-protocol/README.md)
- [MEMSHADOW Ingest Plugin](../plugins/ingest/README.md)
- [Hub Orchestrator](../federation/README.md)
