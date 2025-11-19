# MEMSHADOW SWARM - Autonomous Red Team Orchestrator

## Overview

SWARM (Security Workflow Autonomous Red-team Module) is an autonomous agent swarm system for distributed security testing and vulnerability analysis. It builds on PROJECT HYDRA Phase 3 to create a self-coordinating red team capability.

## Architecture

### Components

1. **Coordinator Node** - Central intelligence that:
   - Loads and manages missions
   - Dispatches tasks to agents
   - Collects and processes reports
   - Maintains blackboard state
   - Generates mission reports

2. **Blackboard** - Redis-based shared knowledge base for:
   - Asynchronous agent communication
   - Task queuing and distribution
   - Report aggregation
   - Discovered data storage

3. **Autonomous Agents**:
   - **agent-recon**: Network enumeration and service discovery
   - **agent-apimapper**: API endpoint mapping and discovery
   - **agent-authtest**: Authentication/authorization testing
   - **agent-bgp**: BGP hijack detection and route analysis
   - **agent-blockchain**: Cross-chain fraud detection

4. **Arena Environment** - Isolated Docker network for:
   - MEMSHADOW staging deployment (target)
   - SWARM infrastructure
   - Log aggregation
   - Network isolation

## Features

### Document Processing
- **Multi-format support**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, Images
- **OCR capabilities**: Extract text from scanned documents and images
- **Metadata preservation**: Track document origin, structure, and processing history
- **Async processing**: Celery-based background tasks
- **Chunking**: Intelligent document segmentation for optimal memory storage

### SWARM Orchestration
- **Mission-driven**: YAML-based mission definitions
- **Stage dependencies**: Sequential and parallel stage execution
- **Success criteria**: Automated stage completion validation
- **Distributed agents**: Multiple agent instances working in parallel
- **Real-time reporting**: Live mission status and results

### MCP Integration
All features are accessible via MCP-compatible endpoints:
- `/api/v1/mcp/documents/upload` - Upload and process documents
- `/api/v1/mcp/documents/status/{task_id}` - Check processing status
- `/api/v1/mcp/memory/store` - Store memories
- `/api/v1/mcp/memory/search` - Search memories
- `/api/v1/mcp/swarm/mission/start` - Launch security missions
- `/api/v1/mcp/swarm/mission/{mission_id}/status` - Check mission status
- `/api/v1/mcp/swarm/agents/status` - Get agent statuses

## Getting Started

### Prerequisites
```bash
# Install system dependencies
apt-get install nmap tesseract-ocr

# Install Python dependencies
pip install -r requirements/base.txt
```

### Running Document Processing

```python
# Upload a document via API
import httpx

async with httpx.AsyncClient() as client:
    with open("document.pdf", "rb") as f:
        response = await client.post(
            "http://localhost:8000/api/v1/memory/ingest/document",
            files={"file": f},
            headers={"Authorization": f"Bearer {token}"}
        )

    task_id = response.json()["task_id"]

    # Check processing status
    status = await client.get(
        f"http://localhost:8000/api/v1/memory/document/status/{task_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
```

### Deploying the SWARM Arena

```bash
# Start the Arena environment
cd docker
docker-compose -f arena-docker-compose.yml up -d

# View coordinator logs
docker-compose -f arena-docker-compose.yml logs -f swarm_coordinator

# View agent logs
docker-compose -f arena-docker-compose.yml logs -f agent_recon

# Access reports
docker-compose -f arena-docker-compose.yml exec swarm_coordinator ls /reports

# Shutdown
docker-compose -f arena-docker-compose.yml down
```

### Creating Custom Missions

Create a YAML file in `missions/`:

```yaml
mission_id: "custom_mission_001"
mission_name: "Custom Security Assessment"
description: "Your custom mission description"

objective_stages:
  - stage_id: "initial_scan"
    description: "Initial network scan"
    tasks:
      - agent_type: "recon"
        params:
          target_cidr: "192.168.1.0/24"
          scan_type: "comprehensive"
    success_criteria:
      - "blackboard_key_exists:known_hosts"
      - "num_known_hosts >= 1"

  - stage_id: "api_discovery"
    description: "Discover API endpoints"
    depends_on: "initial_scan"
    tasks:
      - agent_type: "apimapper"
        params_from_blackboard:
          target_host: "primary_web_service_host"
          target_port: "primary_web_service_port"
    success_criteria:
      - "api_endpoints_mapped"

overall_success_condition: "All stages completed successfully"
```

## Agent Development

### Creating a New Agent

1. Create `swarm_agents/agent_yourname.py`:

```python
from base_agent import BaseAgent
import uuid

class AgentYourName(BaseAgent):
    def __init__(self, agent_id: str = None):
        agent_id = agent_id or f"yourname-{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id=agent_id, agent_type="yourname")

    async def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Your agent logic here
        return {
            "result": "success",
            "data": {}
        }

if __name__ == "__main__":
    agent = AgentYourName()
    agent.run()
```

2. Add to `docker/arena-docker-compose.yml`:

```yaml
agent_yourname:
  build:
    context: ..
    dockerfile: docker/Dockerfile.swarm_agent
  environment:
    REDIS_URL: redis://swarm_redis:6379/0
    AGENT_TYPE: yourname
  command: python agent_yourname.py
  depends_on:
    - swarm_redis
```

## Security Considerations

- **Arena Isolation**: The Arena network is internal-only with no external access
- **Staging Environment**: Always test against staging, never production
- **Mission Validation**: Review missions before execution
- **Agent Sandboxing**: Agents run in isolated containers
- **Log Monitoring**: All activity is logged and aggregated

## Monitoring and Debugging

### Blackboard Inspection

```python
from app.services.swarm.blackboard import Blackboard

blackboard = Blackboard()

# View all discovered data
keys = blackboard.get_all_keys()
for key in keys:
    print(f"{key}: {blackboard.get(key)}")

# Check pending tasks
print(f"Recon tasks: {blackboard.get_task_count('recon')}")
print(f"Pending reports: {blackboard.get_pending_reports_count()}")
```

### Mission Reports

Mission reports are saved in JSON format in `/reports/` volume:
- `{mission_id}_report.json` - Complete mission execution report
- Contains discovered hosts, endpoints, vulnerabilities
- Includes timing, success rates, and detailed findings

## API Documentation

Full API documentation available at:
- Swagger UI: `http://localhost:8000/api/v1/openapi.json`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

When adding new agents or features:
1. Follow the base agent pattern
2. Add comprehensive error handling
3. Log all significant actions
4. Update mission templates
5. Document in this README

## License

Part of PROJECT MEMSHADOW
