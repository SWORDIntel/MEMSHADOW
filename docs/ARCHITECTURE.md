# MEMSHADOW Architecture

**Comprehensive Technical Architecture Documentation**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Component Architecture](#component-architecture)
- [Security Architecture](#security-architecture)
- [Data Architecture](#data-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Network Architecture](#network-architecture)
- [Integration Architecture](#integration-architecture)
- [Scalability & Performance](#scalability--performance)

---

## Overview

MEMSHADOW is a microservices-based offensive security platform built on modern cloud-native principles. The architecture emphasizes:

- **Security-first design** - Defense-in-depth with 8 security layers
- **Scalability** - Horizontal scaling via Kubernetes
- **Observability** - Comprehensive metrics and logging
- **Modularity** - Loosely coupled services
- **Performance** - AI/ML hardware acceleration

### Key Technologies

- **Application Framework:** FastAPI (Python 3.11)
- **Database:** PostgreSQL 15 (primary), Redis 7 (cache), ChromaDB (vectors)
- **Containerization:** Docker 20.10+, Kubernetes 1.24+
- **Security:** ModSecurity, Suricata, AppArmor, Seccomp
- **Monitoring:** Prometheus, Grafana, AlertManager
- **AI/ML:** PyTorch, CUDA, OpenVINO

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ Operators│  │ Implants │  │  Targets │                     │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘                     │
└────────┼─────────────┼─────────────┼───────────────────────────┘
         │             │             │
         ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Security Perimeter                        │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  WAF (ModSecurity) → IDS (Suricata) → TLS Termination      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                           │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              FastAPI Application                     │       │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │       │
│  │  │  C2    │  │LureCraft│  │Mission │  │ Admin  │    │       │
│  │  │  API   │  │  API    │  │  API   │  │  API   │    │       │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │       │
│  │                                                       │       │
│  │  ┌───────────────────────────────────────────┐      │       │
│  │  │         Middleware Stack                   │      │       │
│  │  │  • Authentication                          │      │       │
│  │  │  • Authorization                           │      │       │
│  │  │  • Rate Limiting                           │      │       │
│  │  │  • APT Security Hardening                  │      │       │
│  │  │  • Request Validation                      │      │       │
│  │  │  • Metrics Collection                      │      │       │
│  │  └───────────────────────────────────────────┘      │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │    C2    │  │LureCraft │  │ Mission  │  │  Threat  │       │
│  │  Service │  │ Service  │  │ Service  │  │  Intel   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │  AI/ML   │  │  Payload │  │  Exfil   │                     │
│  │  Engine  │  │ Generator│  │ Manager  │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Data Layer                                │
│  ┌──────────┐  ┌───────┐  ┌──────────┐  ┌─────────┐          │
│  │PostgreSQL│  │ Redis │  │ChromaDB  │  │  S3/    │          │
│  │(Primary) │  │(Cache)│  │(Vectors) │  │ MinIO   │          │
│  └──────────┘  └───────┘  └──────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Layer                          │
│  ┌────────────┐  ┌─────────┐  ┌──────────────┐                │
│  │ Prometheus │  │ Grafana │  │ AlertManager │                │
│  └────────────┘  └─────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. C2 Framework

**Purpose:** Command and control infrastructure for managing implants.

**Components:**
- **Session Manager:** Track active implant connections
- **Task Queue:** Redis-backed task distribution
- **Encryption Engine:** AES-256-GCM for C2 communications
- **Protocol Handlers:** HTTP/HTTPS, WebSocket, DNS
- **Payload Generator:** Dynamic payload creation
- **Exfiltration Pipeline:** Secure data extraction

**Data Flow:**
```
Implant → TLS → WAF → IDS → C2 API → Session Manager → Task Queue
                                           ↓
                                      PostgreSQL
                                           ↓
                                    Metrics (Prometheus)
```

**Key Features:**
- Multi-protocol support (HTTP, WebSocket, DNS)
- Encrypted C2 channels
- Session resilience and reconnection
- Task scheduling and queuing
- File upload/download
- Real-time communication via WebSocket

### 2. LureCraft (Social Engineering)

**Purpose:** Phishing and social engineering campaign management.

**Components:**
- **Template Engine:** Jinja2-based page generation
- **Credential Harvester:** Form capture and storage
- **Email Sender:** SMTP integration for phishing emails
- **QR Generator:** QR code phishing campaigns
- **Landing Page Server:** Dynamic phishing pages
- **Analytics:** Track campaign success rates

**Data Flow:**
```
Target → Phishing URL → WAF → LureCraft API → Template Engine
                                                    ↓
                                            Credential Store
                                                    ↓
                                               PostgreSQL
```

**Templates Supported:**
- Microsoft/Office 365 login
- Google Workspace
- Generic corporate login
- MFA bypass pages
- Custom templates

### 3. Mission Management

**Purpose:** Campaign planning, tracking, and reporting.

**Components:**
- **Campaign Manager:** Mission lifecycle management
- **Target Profiler:** Intelligence gathering on targets
- **Attack Chain:** MITRE ATT&CK mapping
- **Results Aggregator:** Consolidate operation results
- **Report Generator:** Automated report creation
- **Timeline Tracker:** Operation timeline management

**Data Flow:**
```
Operator → Mission API → Campaign Manager → Attack Chain
                              ↓
                        PostgreSQL
                              ↓
                     Report Generator → PDF/HTML
```

**MITRE ATT&CK Integration:**
- Technique mapping
- Tactic tracking
- Kill chain phases
- Automated detection gap analysis

### 4. Threat Intelligence

**Purpose:** Aggregate and correlate threat intelligence from multiple sources.

**Components:**
- **Intel Aggregator:** Multi-source data collection
- **IOC Parser:** STIX 2.x indicator parsing
- **Correlation Engine:** Deduplicate and correlate IOCs
- **Reputation Tracker:** IP/domain reputation scoring
- **Auto-Blocker:** Automated threat blocking
- **Feed Manager:** Manage threat intel subscriptions

**Supported Sources:**
- MISP (Malware Information Sharing Platform)
- OpenCTI (Open Cyber Threat Intelligence)
- AbuseIPDB
- AlienVault OTX
- VirusTotal
- Custom TAXII feeds

**Data Flow:**
```
MISP API → Intel Aggregator → IOC Parser → Correlation Engine
OpenCTI  ↗                                        ↓
AbuseIPDB ↗                                 PostgreSQL
                                                  ↓
                                          Auto-Blocker
                                                  ↓
                                      API Security Middleware
```

### 5. AI/ML Engine

**Purpose:** AI-powered analysis and decision support.

**Components:**
- **Inference Engine:** PyTorch-based model execution
- **Vulnerability Classifier:** CVSS prediction and prioritization
- **Anomaly Detector:** Network traffic analysis
- **Exploit Predictor:** Exploit likelihood scoring
- **Hardware Abstraction:** GPU/NPU/CPU support
- **Model Registry:** Model version management

**Hardware Support:**
- **NVIDIA GPU:** CUDA acceleration (82 TOPS)
- **Intel NPU:** OpenVINO optimization (48 TOPS)
- **CPU Fallback:** Pure Python execution

**Models:**
```python
models/
├── vulnerability_classifier.pt  # CVSS scoring
├── anomaly_detector.pt         # Network traffic
├── exploit_predictor.pt        # Exploit likelihood
└── lateral_movement.pt         # Lateral movement detection
```

**Acceleration Techniques:**
- Mixed precision inference (FP16/TF32)
- Tensor Core utilization
- Batch processing
- TorchScript compilation
- Model quantization

---

## Security Architecture

### Defense-in-Depth (8 Layers)

```
Layer 1: Network Perimeter
├── WAF (ModSecurity + OWASP CRS)
├── DDoS protection
└── TLS 1.3 termination

Layer 2: Intrusion Detection
├── Suricata IDS
├── APT detection rules
└── Automated alerting

Layer 3: Network Segmentation
├── DMZ network (172.30.0.0/24) - Controlled internet access
└── Internal network (172.29.0.0/24) - Air-gapped

Layer 4: API Security
├── HMAC request signatures
├── Rate limiting (token bucket)
├── SQL injection detection
├── XSS prevention
├── Path traversal blocking
└── Attack tool detection

Layer 5: Container Hardening
├── Read-only root filesystem
├── All capabilities dropped
├── AppArmor LSM profiles
├── Seccomp syscall filtering
├── Non-root users (UID 1000+)
└── Resource limits

Layer 6: Application Security
├── JWT-based authentication
├── Role-based access control (RBAC)
├── Input validation
├── Output sanitization
└── Secure session management

Layer 7: Data Encryption
├── AES-256-GCM at rest
├── TLS 1.3 in transit
├── Database encryption
└── Secret management (Docker secrets)

Layer 8: Monitoring & Alerting
├── Real-time threat detection
├── Anomaly detection (AI/ML)
├── Security event logging
└── Automated incident response
```

### Security Controls

**Authentication:**
- JWT tokens with RS256 signing
- Token expiration (1 hour access, 7 day refresh)
- Multi-factor authentication support
- API key authentication for implants

**Authorization:**
- Role-based access control (RBAC)
- Endpoint-level permissions
- Resource-level permissions
- Operator isolation

**Secrets Management:**
- Docker secrets (file-based)
- 600 permissions on all secret files
- Automatic rotation support
- No secrets in environment variables
- Hardware security module (HSM) ready

**Audit Logging:**
- All API requests logged
- Security events tracked
- Compliance reporting
- Tamper-proof log storage

---

## Data Architecture

### Database Schema

**PostgreSQL (Primary Data Store):**

```
┌─────────────────────────────────────────────────────────────────┐
│                         PostgreSQL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  users                          roles                           │
│  ├── id (PK)                    ├── id (PK)                     │
│  ├── username                   ├── name                        │
│  ├── email                      ├── permissions[]               │
│  ├── password_hash              └── created_at                  │
│  ├── role_id (FK)                                               │
│  └── created_at                 user_roles (many-to-many)       │
│                                                                 │
│  c2_sessions                    c2_tasks                        │
│  ├── id (PK)                    ├── id (PK)                     │
│  ├── implant_id                 ├── session_id (FK)             │
│  ├── os                         ├── command                     │
│  ├── hostname                   ├── parameters                  │
│  ├── ip_address                 ├── status                      │
│  ├── status                     ├── result                      │
│  ├── last_checkin               ├── created_at                  │
│  └── metadata                   └── completed_at                │
│                                                                 │
│  phishing_pages                 phishing_credentials            │
│  ├── id (PK)                    ├── id (PK)                     │
│  ├── template                   ├── page_id (FK)                │
│  ├── url                        ├── username                    │
│  ├── redirect_url               ├── password                    │
│  ├── captures                   ├── additional_data             │
│  └── created_at                 ├── ip_address                  │
│                                 └── captured_at                 │
│                                                                 │
│  missions                       mission_objectives              │
│  ├── id (PK)                    ├── id (PK)                     │
│  ├── name                       ├── mission_id (FK)             │
│  ├── description                ├── description                 │
│  ├── status                     ├── status                      │
│  ├── operator_id (FK)           ├── mitre_technique             │
│  ├── start_date                 └── completed_at                │
│  └── end_date                                                   │
│                                                                 │
│  vulnerabilities                threat_indicators               │
│  ├── id (PK)                    ├── id (PK)                     │
│  ├── mission_id (FK)            ├── type (ip/domain/hash/url)   │
│  ├── cvss_score                 ├── value                       │
│  ├── severity                   ├── confidence                  │
│  ├── description                ├── source                      │
│  └── discovered_at              ├── kill_chain_phase            │
│                                 └── first_seen                  │
└─────────────────────────────────────────────────────────────────┘
```

**Redis (Cache & Queue):**

```
┌─────────────────────────────────────────────────────────────────┐
│                           Redis                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Caching:                                                       │
│  ├── user_sessions:{user_id}     → Session data                │
│  ├── api_rate_limit:{ip}         → Rate limit counters         │
│  ├── threat_intel_cache:{ioc}    → Cached IOC lookups          │
│  └── ml_predictions:{hash}       → Cached ML results           │
│                                                                 │
│  Queues (Redis Lists):                                         │
│  ├── c2_tasks                     → Task queue for implants    │
│  ├── phishing_queue               → Email sending queue        │
│  └── ml_inference_queue           → AI/ML batch processing     │
│                                                                 │
│  Pub/Sub:                                                       │
│  ├── c2_notifications             → Real-time implant updates  │
│  ├── security_alerts              → Security events            │
│  └── metrics_updates              → Live metrics               │
└─────────────────────────────────────────────────────────────────┘
```

**ChromaDB (Vector Store):**

```
┌─────────────────────────────────────────────────────────────────┐
│                         ChromaDB                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Collections:                                                   │
│                                                                 │
│  vulnerability_embeddings                                       │
│  ├── CVE descriptions vectorized                               │
│  ├── Exploit database embeddings                               │
│  └── Similarity search for vuln classification                 │
│                                                                 │
│  threat_intel_embeddings                                        │
│  ├── IOC context vectors                                       │
│  ├── MITRE ATT&CK technique embeddings                         │
│  └── Threat actor TTPs                                         │
│                                                                 │
│  mission_knowledge                                              │
│  ├── Historical operation data                                 │
│  ├── Lessons learned embeddings                                │
│  └── Target profile vectors                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flows

**C2 Session Lifecycle:**
```
1. Implant → Check-in → API
2. API → Create session → PostgreSQL
3. API → Publish event → Redis Pub/Sub
4. Operator → Create task → API
5. API → Queue task → Redis List
6. Implant → Fetch task → API → Redis List
7. Implant → Submit result → API
8. API → Update task → PostgreSQL
9. API → Metrics → Prometheus
```

**Threat Intel Pipeline:**
```
1. MISP API → Intel Aggregator
2. Parse STIX → IOC Extraction
3. Deduplicate → Correlation Engine
4. Store → PostgreSQL
5. Vectorize → ChromaDB
6. Cache → Redis
7. Notify → Security Middleware
8. Auto-block → WAF/IDS rules
```

---

## Deployment Architecture

### Docker Compose (Development/Single-Host)

```
docker-compose.hardened.yml
├── memshadow              (FastAPI app, 2 replicas)
├── postgres               (Primary database)
├── redis                  (Cache & queue)
├── chromadb               (Vector store)
├── waf                    (ModSecurity)
├── ids                    (Suricata)
├── prometheus             (Metrics)
├── grafana                (Dashboards)
└── alertmanager           (Alerting)

Networks:
├── memshadow-internal     (172.29.0.0/24, no internet)
└── memshadow-dmz          (172.30.0.0/24, controlled access)

Volumes:
├── postgres_data          (Database persistence)
├── redis_data             (Cache persistence)
├── chromadb_data          (Vector persistence)
└── memshadow_data         (Application data)
```

### Kubernetes (Production/Multi-Host)

```
Namespace: memshadow

Deployments:
├── memshadow-api          (3+ replicas, HPA enabled)
├── postgres               (1 replica, StatefulSet)
├── redis                  (3 replicas, Redis Cluster)
├── chromadb               (1 replica, StatefulSet)
├── waf                    (2+ replicas)
└── ids                    (1 replica per node, DaemonSet)

Services:
├── memshadow-api          (ClusterIP)
├── postgres               (ClusterIP, headless)
├── redis                  (ClusterIP)
├── chromadb               (ClusterIP)
└── memshadow-ingress      (LoadBalancer/Ingress)

Storage:
├── postgres-pvc           (100Gi, SSD)
├── redis-pvc              (50Gi, SSD)
└── chromadb-pvc           (200Gi, SSD)

ConfigMaps:
├── memshadow-config       (Application config)
├── prometheus-config      (Metrics config)
└── grafana-dashboards     (Dashboard definitions)

Secrets:
├── memshadow-secrets      (DB passwords, API keys)
├── tls-certs              (TLS certificates)
└── jwt-keys               (JWT signing keys)

Horizontal Pod Autoscaler:
├── Target: CPU 70%, Memory 80%
├── Min replicas: 3
└── Max replicas: 20
```

---

## Network Architecture

### Network Topology

```
                        Internet
                           │
                           ▼
                    ┌─────────────┐
                    │ LoadBalancer│ (Cloud LB or HAProxy)
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Ingress   │ (Kubernetes Ingress)
                    │   :443      │ (TLS termination)
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐     ┌─────────┐
    │  WAF    │      │  WAF    │     │  WAF    │ (DMZ Network)
    │ Pod 1   │      │ Pod 2   │     │ Pod 3   │ 172.30.0.0/24
    └────┬────┘      └────┬────┘     └────┬────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │     IDS     │ (Suricata)
                    │  (DaemonSet)│
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                                 │
          ▼                                 ▼
    DMZ Network                    Internal Network
    172.30.0.0/24                  172.29.0.0/24
    (Internet access)              (No internet, air-gapped)
          │                                 │
          ▼                                 ▼
    ┌─────────────┐              ┌──────────────────┐
    │  MEMSHADOW  │              │    PostgreSQL    │
    │   API Pods  │──────────────│    Redis         │
    │  (3+ pods)  │              │    ChromaDB      │
    └─────────────┘              └──────────────────┘
```

### Network Policies

**Ingress Rules:**
```yaml
# memshadow-api pods
ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: memshadow
    - podSelector:
        matchLabels:
          app: waf
    ports:
    - protocol: TCP
      port: 8000
```

**Egress Rules:**
```yaml
# Internal network - NO internet access
egress:
  - to:
    - podSelector:
        matchLabels:
          tier: data
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 8001  # ChromaDB

# DMZ network - Controlled internet access
egress:
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          app: threat-intel-proxy
    ports:
    - protocol: TCP
      port: 443
```

---

## Integration Architecture

### External Integrations

**Threat Intelligence:**
```
MEMSHADOW ←→ MISP Server
         ├─→ REST API (HTTPS)
         ├─→ API Key authentication
         └─→ Pull IOCs every 1 hour

MEMSHADOW ←→ OpenCTI
         ├─→ GraphQL API (HTTPS)
         ├─→ Bearer token auth
         └─→ STIX 2.x format

MEMSHADOW ←→ AbuseIPDB
         ├─→ REST API (HTTPS)
         ├─→ API Key in header
         └─→ IP reputation lookups
```

**Monitoring & Alerting:**
```
Prometheus ←→ MEMSHADOW Metrics Endpoint
          ├─→ Scrape /api/v1/metrics
          └─→ Every 15 seconds

AlertManager ←→ Notification Channels
            ├─→ Slack webhooks
            ├─→ PagerDuty API
            ├─→ Email SMTP
            └─→ Custom webhooks
```

**CI/CD:**
```
GitHub Actions ──→ Build Docker Images
              ├──→ Run Tests (pytest)
              ├──→ Security Scan (Trivy)
              ├──→ Push to Registry
              └──→ Deploy to K8s
```

---

## Scalability & Performance

### Horizontal Scaling

**Auto-scaling Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: memshadow-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: memshadow-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Database Scaling:**
- PostgreSQL: Read replicas for reporting
- Redis: Redis Cluster (3 master, 3 replica minimum)
- ChromaDB: Vertical scaling (SSD-backed storage)

### Performance Optimizations

**Application Layer:**
- Async I/O (FastAPI + asyncio)
- Connection pooling (asyncpg, aioredis)
- Response caching (Redis)
- Database query optimization (indexed columns)
- Lazy loading of relationships

**AI/ML Layer:**
- Batch inference (process multiple requests together)
- Model quantization (reduce model size)
- Mixed precision (FP16/TF32)
- GPU/NPU acceleration
- Result caching

**Database Layer:**
- Indexed columns (all foreign keys, search columns)
- Materialized views for reports
- Partitioning for large tables (by date)
- Connection pooling (PgBouncer)
- Read replicas for reporting

### Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| API Response Time (p95) | < 500ms | Prometheus |
| API Response Time (p99) | < 1000ms | Prometheus |
| C2 Check-in Latency | < 100ms | Prometheus |
| Database Query (p95) | < 50ms | PostgreSQL logs |
| AI/ML Inference | < 100ms | Application metrics |
| Concurrent Sessions | 10,000+ | Load testing (k6) |
| Requests/Second | 1,000+ | Load testing (k6) |

---

## Summary

MEMSHADOW architecture is built on these principles:

1. **Security First:** 8 layers of defense-in-depth
2. **Cloud Native:** Containerized, orchestrated, scalable
3. **Observable:** Comprehensive metrics and logging
4. **Performant:** AI/ML acceleration, caching, async I/O
5. **Modular:** Microservices, loosely coupled
6. **Resilient:** Auto-scaling, self-healing, redundancy

The architecture supports both single-host Docker Compose deployments (development)
and multi-host Kubernetes clusters (production), providing flexibility for
different use cases and scales.

---

**Classification:** UNCLASSIFIED
**Version:** 2.1
**Last Updated:** 2025-11-16
