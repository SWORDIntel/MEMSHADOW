# MEMSHADOW Python SDK

A memlayer-inspired Python client library for MEMSHADOW memory persistence with drop-in LLM wrappers.

## Features

- **Simple API Client**: Direct access to MEMSHADOW memory and reminder APIs
- **LLM Wrappers**: Drop-in replacements for OpenAI and Anthropic clients with automatic memory
- **Automatic Context Injection**: Seamlessly retrieve and inject relevant memories
- **Task Reminders**: Schedule and manage task reminders through the API
- **Production Ready**: Built on requests with proper error handling

## Installation

### Basic Installation

```bash
pip install memshadow-sdk
```

### With LLM Support

```bash
# For OpenAI support
pip install memshadow-sdk[openai]

# For Anthropic support
pip install memshadow-sdk[anthropic]

# For both
pip install memshadow-sdk[all]
```

### For Development

```bash
pip install memshadow-sdk[dev]
```

## Quick Start

### 1. Basic Client Usage

```python
from memshadow_sdk import MemshadowClient

# Initialize client
client = MemshadowClient(
    api_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

# Ingest a memory
memory = client.ingest("I love Python programming")

# Retrieve memories
results = client.retrieve("programming languages", limit=5)
for result in results:
    print(result['content'])
```

### 2. OpenAI Wrapper

```python
from memshadow_sdk import OpenAI

# Initialize wrapper
client = OpenAI(
    api_key="your-openai-key",
    memshadow_url="http://localhost:8000/api/v1",
    memshadow_api_key="your-memshadow-key",
    user_id="user_123"
)

# First conversation - store a fact
response = client.chat([
    {"role": "user", "content": "My name is Alice and I work at TechCorp"}
])

# Later conversation - memory is automatically retrieved
response = client.chat([
    {"role": "user", "content": "Where do I work?"}
])
# Returns: "You work at TechCorp."
```

### 3. Anthropic Claude Wrapper

```python
from memshadow_sdk import Anthropic

# Initialize wrapper
client = Anthropic(
    api_key="your-anthropic-key",
    memshadow_url="http://localhost:8000/api/v1",
    memshadow_api_key="your-memshadow-key",
    user_id="user_123"
)

# Conversation with automatic memory
response = client.chat([
    {"role": "user", "content": "My favorite color is blue"}
])

# Later session - memory persists
response = client.chat([
    {"role": "user", "content": "What's my favorite color?"}
])
# Returns: "Your favorite color is blue"
```

### 4. Task Reminders

```python
from datetime import datetime, timedelta

# Create a reminder
reminder = client.create_reminder(
    title="Review pull request",
    reminder_date=datetime.now() + timedelta(hours=2),
    due_date=datetime.now() + timedelta(days=1),
    priority="high",
    description="Review the new authentication feature PR"
)

# List reminders
reminders = client.list_reminders(status="pending", priority="high")

# Complete a reminder
client.complete_reminder(reminder['id'])
```

## Advanced Usage

### Manual Memory Management

```python
# Manually ingest important facts
client.ingest_manual(
    "Project deadline is December 15th",
    metadata={"type": "deadline", "project": "MEMSHADOW"}
)

# Manually retrieve specific memories
memories = client.retrieve_memories("deadlines", limit=10)
```

### Custom Context Control

```python
from memshadow_sdk import OpenAI

# Initialize with custom context settings
client = OpenAI(
    api_key="your-openai-key",
    memshadow_url="http://localhost:8000/api/v1",
    memshadow_api_key="your-memshadow-key",
    auto_inject_context=True,   # Enable automatic context injection
    context_limit=10            # Inject up to 10 memories
)

# Disable context injection for specific requests
# (manually control what context to include)
response = client.chat(
    messages=[{"role": "user", "content": "What's 2+2?"}],
    # Custom implementation would go here
)
```

## API Reference

### MemshadowClient

```python
class MemshadowClient:
    def __init__(
        api_url: str,
        api_key: str,
        user_id: Optional[str] = None,
        timeout: int = 30
    )

    def ingest(content: str, extra_data: Optional[Dict] = None) -> Dict
    def retrieve(query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Dict]
    def create_reminder(title: str, reminder_date: datetime, ...) -> Dict
    def list_reminders(status: Optional[str] = None, ...) -> List[Dict]
    def complete_reminder(reminder_id: str) -> Dict
```

### OpenAI Wrapper

```python
class OpenAI:
    def __init__(
        api_key: str,
        memshadow_url: str,
        memshadow_api_key: str,
        model: str = "gpt-4",
        user_id: Optional[str] = None,
        auto_inject_context: bool = True,
        context_limit: int = 5
    )

    def chat(messages: List[Dict], model: Optional[str] = None, ...) -> Dict
    def ingest_manual(content: str, metadata: Optional[Dict] = None) -> Dict
    def retrieve_memories(query: str, limit: int = 10) -> List[Dict]
```

### Anthropic Wrapper

```python
class Anthropic:
    def __init__(
        api_key: str,
        memshadow_url: str,
        memshadow_api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        user_id: Optional[str] = None,
        auto_inject_context: bool = True,
        context_limit: int = 5
    )

    def chat(messages: List[Dict], model: Optional[str] = None, ...) -> Dict
    def ingest_manual(content: str, metadata: Optional[Dict] = None) -> Dict
    def retrieve_memories(query: str, limit: int = 10) -> List[Dict]
```

## Configuration

### Environment Variables

```bash
# Set default MEMSHADOW configuration
export MEMSHADOW_API_URL="http://localhost:8000/api/v1"
export MEMSHADOW_API_KEY="your-api-key"
export MEMSHADOW_USER_ID="user_123"
```

### Operation Modes

MEMSHADOW supports three operation modes (configured server-side):

- **LOCAL**: Full enrichment, all features (highest quality, slowest)
- **ONLINE**: Balanced speed/features (moderate enrichment)
- **LIGHTWEIGHT**: Minimal processing (fastest, basic features only)

## Examples

See the `/examples` directory for complete working examples:

- `basic_usage.py` - Basic client operations
- `openai_wrapper.py` - OpenAI integration example
- `anthropic_wrapper.py` - Anthropic Claude integration
- `reminders.py` - Task reminder management

## Error Handling

```python
from memshadow_sdk import MemshadowClient
import requests

client = MemshadowClient(
    api_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

try:
    memory = client.ingest("Important information")
except requests.HTTPError as e:
    if e.response.status_code == 409:
        print("Duplicate memory content")
    elif e.response.status_code == 401:
        print("Authentication failed")
    else:
        print(f"API error: {e}")
except requests.RequestException as e:
    print(f"Network error: {e}")
```

## Development

### Running Tests

```bash
cd sdk/
pytest
```

### Building Package

```bash
python setup.py sdist bdist_wheel
```

### Installing Locally

```bash
pip install -e .
```

## Comparison with Memlayer

| Feature | Memlayer | MEMSHADOW SDK |
|---------|----------|---------------|
| **Architecture** | File-based storage | Enterprise PostgreSQL + ChromaDB |
| **Embedding Dimension** | Up to 1536d | 2048d with projection |
| **Operation Modes** | LOCAL/ONLINE/LIGHTWEIGHT | ✅ All three modes |
| **Knowledge Graph** | NetworkX (in-memory) | PostgreSQL-backed persistent |
| **Task Reminders** | ✅ Built-in | ✅ Full API support |
| **Multi-user** | Basic | Enterprise with encryption |
| **Scalability** | Single process | Celery workers + Redis |
| **Security** | Basic | Audit logs, MFA, encryption |

## License

MIT License - see LICENSE file for details

## Support

- GitHub Issues: https://github.com/SWORDIntel/MEMSHADOW/issues
- Documentation: https://github.com/SWORDIntel/MEMSHADOW
- Email: contact@memshadow.dev

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
