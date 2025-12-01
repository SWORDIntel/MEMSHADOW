# HYDRA SWARM Arena

**Phase 7: Isolated Testing Environment for Autonomous Security Testing**

The Arena is a Docker-based isolated environment for safely running HYDRA SWARM missions against vulnerable test applications.

## Overview

The Arena provides:
- **Isolated Network**: Dedicated bridge network (`172.28.0.0/16`)
- **Test Targets**: Vulnerable applications for testing
- **SWARM Infrastructure**: Coordinator, agents, and blackboard
- **Monitoring**: Prometheus + Grafana for observability
- **Resource Limits**: Prevents impact on host system

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HYDRA SWARM Arena                       │
│                    (Isolated Network)                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Coordinator │◄────►│Redis Blackboard│                    │
│  └──────┬───────┘      └──────────────┘                     │
│         │                                                     │
│    ┌────┴──────┬──────────┬───────────┐                     │
│    ▼           ▼          ▼           ▼                      │
│ ┌─────┐   ┌────────┐ ┌─────────┐ ┌──────────┐              │
│ │Recon│   │API     │ │Auth     │ │Mission   │              │
│ │Agent│   │Mapper  │ │Test     │ │Orchestr. │              │
│ └──┬──┘   └───┬────┘ └────┬────┘ └──────────┘              │
│    │          │           │                                  │
│    └──────────┼───────────┘                                  │
│               │                                               │
│    ┌──────────▼──────────┬──────────────┐                   │
│    │                     │              │                    │
│ ┌──▼────┐         ┌──────▼───┐   ┌─────▼─────┐             │
│ │ DVWA  │         │Mock REST │   │(Future    │             │
│ │(Vuln) │         │   API    │   │ Targets)  │             │
│ └───────┘         └──────────┘   └───────────┘             │
│                                                               │
│  ┌────────────┐        ┌─────────┐                          │
│  │ Prometheus │◄──────►│ Grafana │                          │
│  └────────────┘        └─────────┘                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Infrastructure

1. **arena-redis** (Port 6380)
   - Redis 7 for blackboard communication
   - Persistent storage for agent coordination
   - Health checks enabled

2. **arena-coordinator**
   - C2 server managing agent swarm
   - Mission assignment and monitoring
   - Finding aggregation

3. **arena-orchestrator** (Port 8083)
   - Mission orchestration API
   - Report generation
   - Swarm health monitoring

### Agents

1. **arena-agent-recon**
   - Reconnaissance and enumeration
   - Technology detection
   - Security header analysis

2. **arena-agent-apimapper**
   - API endpoint discovery
   - OpenAPI/Swagger parsing
   - GraphQL introspection

3. **arena-agent-authtest**
   - Authentication testing
   - JWT analysis
   - Session management testing

### Test Targets

1. **arena-target-api** (Port 8081)
   - DVWA (Damn Vulnerable Web Application)
   - MySQL-backed web app
   - Multiple vulnerability categories

2. **arena-target-rest** (Port 8082)
   - Mock REST API (Mockoon)
   - Configurable endpoints
   - Customizable responses

### Monitoring

1. **arena-prometheus** (Port 9091)
   - Metrics collection
   - Alert management
   - Time-series database

2. **arena-grafana** (Port 3001)
   - Visualization dashboards
   - Default credentials: `admin / arena_admin`
   - Pre-configured Prometheus datasource

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB available RAM
- 10GB available disk space

### Launch Arena

```bash
cd docker/arena

# Start the entire arena
docker-compose -f docker-compose.arena.yml up -d

# Check status
docker-compose -f docker-compose.arena.yml ps

# View logs
docker-compose -f docker-compose.arena.yml logs -f

# View specific service logs
docker-compose -f docker-compose.arena.yml logs -f arena-coordinator
```

### Verify Deployment

```bash
# Check Redis
redis-cli -p 6380 ping
# Expected: PONG

# Check coordinator health
curl http://localhost:8083/health

# Check test targets
curl http://localhost:8081  # DVWA
curl http://localhost:8082  # Mock API
```

### Access Services

- **Mission Orchestrator**: http://localhost:8083
- **DVWA (Vulnerable App)**: http://localhost:8081
- **Mock REST API**: http://localhost:8082
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3001 (admin / arena_admin)

## Running Missions

### Via API

```bash
# Start a quick scan mission
curl -X POST http://localhost:8083/missions/run \
  -H "Content-Type: application/json" \
  -d '{
    "template": "quick_scan",
    "target": "http://arena-target-api"
  }'

# Full assessment mission
curl -X POST http://localhost:8083/missions/run \
  -H "Content-Type: application/json" \
  -d '{
    "template": "full_assessment",
    "target": "http://arena-target-rest:3000"
  }'

# Check mission status
curl http://localhost:8083/missions/<mission_id>/status

# Get mission report
curl http://localhost:8083/missions/<mission_id>/report
```

### Via Python

```python
import asyncio
from app.services.hydra.swarm import orchestrator, MissionTemplate

async def run_mission():
    # Start orchestrator
    await orchestrator.start()

    # Run mission
    report = await orchestrator.run_mission(
        template=MissionTemplate.FULL_ASSESSMENT,
        target="http://arena-target-api"
    )

    # Print results
    print(f"Mission completed!")
    print(f"Risk Score: {report.calculate_risk_score()}")
    print(f"Total Findings: {report.total_findings}")
    print(f"Critical: {report.critical_count}")
    print(f"High: {report.high_count}")

    await orchestrator.stop()

asyncio.run(run_mission())
```

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9091

Key metrics:
- `swarm_agents_total`: Total registered agents
- `swarm_agents_active`: Currently active agents
- `swarm_missions_total`: Total missions executed
- `swarm_findings_total`: Total findings discovered
- `swarm_findings_by_severity`: Findings grouped by severity

### Grafana Dashboards

Access Grafana at http://localhost:3001 (admin / arena_admin)

Pre-configured dashboards:
- **SWARM Overview**: Agent health, mission stats
- **Mission Performance**: Duration, success rate
- **Findings Analysis**: Severity distribution, trends

## Configuration

### Custom Test Targets

Add your own test targets to `docker-compose.arena.yml`:

```yaml
arena-target-custom:
  image: your/custom-app:latest
  container_name: arena-target-custom
  networks:
    - arena-network
  ports:
    - "8084:8080"
  restart: unless-stopped
```

### Agent Configuration

Modify agent environment variables:

```yaml
environment:
  - REDIS_URL=redis://arena-redis:6379
  - AGENT_TYPE=ReconAgent
  - SCAN_DEPTH=3              # Custom: scan depth
  - TIMEOUT_SECONDS=30        # Custom: operation timeout
  - MAX_CONCURRENT_REQUESTS=10  # Custom: concurrency
```

### Mock API Configuration

Edit `mock-api.json` to customize the mock REST API:

```json
{
  "name": "Custom Mock API",
  "routes": [
    {
      "uuid": "route-1",
      "method": "get",
      "endpoint": "api/users",
      "responses": [
        {
          "uuid": "response-1",
          "statusCode": 200,
          "body": "{ \"users\": [...] }"
        }
      ]
    }
  ]
}
```

## Resource Limits

Each service has CPU and memory limits to prevent resource exhaustion:

| Service | CPU Limit | Memory Limit |
|---------|-----------|--------------|
| Coordinator | 1.0 | 512MB |
| Orchestrator | 1.0 | 512MB |
| Agents (each) | 0.5 | 256MB |
| Redis | - | - |
| Test Targets | 0.5 | 256MB |
| Monitoring | - | - |

**Total Arena Resources**: ~2-3 CPU cores, ~3-4GB RAM

## Security Considerations

### Isolation

- Arena runs in dedicated Docker network
- No direct internet access by default
- Test targets are intentionally vulnerable

### Production Warning

⚠️ **NEVER deploy Arena in production environments!**

- Arena contains intentionally vulnerable applications
- Designed for security testing only
- Should be run in isolated lab environments

### Network Access

To allow agents to test external targets:

```yaml
# In docker-compose.arena.yml
networks:
  arena-network:
    driver: bridge
    internal: false  # Allow external access
```

**Use with caution!** Only enable when necessary.

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose -f docker-compose.arena.yml logs

# Rebuild images
docker-compose -f docker-compose.arena.yml build --no-cache

# Remove and recreate
docker-compose -f docker-compose.arena.yml down -v
docker-compose -f docker-compose.arena.yml up -d
```

### Agents Not Registering

```bash
# Check Redis connectivity
docker exec -it arena-agent-recon redis-cli -h arena-redis ping

# Check coordinator logs
docker-compose -f docker-compose.arena.yml logs arena-coordinator

# Restart agents
docker-compose -f docker-compose.arena.yml restart arena-agent-recon
```

### No Findings Generated

1. Verify test targets are accessible:
   ```bash
   docker exec -it arena-agent-recon curl http://arena-target-api
   ```

2. Check agent logs for errors:
   ```bash
   docker-compose -f docker-compose.arena.yml logs arena-agent-recon
   ```

3. Verify mission parameters are correct

### High Resource Usage

```bash
# Check resource usage
docker stats

# Scale down agents
docker-compose -f docker-compose.arena.yml scale arena-agent-recon=1

# Adjust resource limits in docker-compose.arena.yml
```

## Cleanup

```bash
# Stop all services
docker-compose -f docker-compose.arena.yml down

# Remove all data volumes
docker-compose -f docker-compose.arena.yml down -v

# Remove all arena containers and networks
docker-compose -f docker-compose.arena.yml down --remove-orphans

# Remove images
docker-compose -f docker-compose.arena.yml down --rmi all
```

## Advanced Usage

### Multi-Arena Deployment

Run multiple isolated arenas:

```bash
# Arena 1
docker-compose -f docker-compose.arena.yml -p arena1 up -d

# Arena 2 (different ports)
COMPOSE_PROJECT_NAME=arena2 \
  docker-compose -f docker-compose.arena.yml up -d
```

### Custom Mission Templates

Create custom mission templates in `mission.py`:

```python
class CustomMissionTemplate(MissionTemplate):
    DEEP_PENTEST = "deep_pentest"

# Configure template
if template == CustomMissionTemplate.DEEP_PENTEST:
    return Mission(
        target=target,
        mode=SwarmMode.PENETRATION,
        objectives=["Full exploitation", "Privilege escalation"],
        required_capabilities=[...],
        timeout_minutes=60
    )
```

## Contributing

To add new test targets or agents:

1. Add service to `docker-compose.arena.yml`
2. Configure networking and resource limits
3. Update this README with usage instructions
4. Test in isolated environment

## License

Part of MEMSHADOW project - See main repository for license

## Version History

### 1.0.0 (Current)
- Initial Arena deployment
- 3 agents (Recon, APIMapper, AuthTest)
- 2 test targets (DVWA, Mock API)
- Prometheus + Grafana monitoring
- Mission orchestration API
