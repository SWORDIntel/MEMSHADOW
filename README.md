# MEMSHADOW

> **Advanced Offensive Security Platform with AI/ML Acceleration**

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux-blue.svg)](https://www.linux.org/)
[![Docker](https://img.shields.io/badge/docker-20.10+-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Classification](https://img.shields.io/badge/classification-UNCLASSIFIED-green.svg)]()

**MEMSHADOW** is a comprehensive offensive security platform designed for advanced penetration testing, red team operations, and security research. It combines modern C2 capabilities, social engineering tools, AI/ML-powered analysis, and APT-grade defensive hardening.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Security](#security)
- [Documentation](#documentation)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### 🎯 Core Capabilities

**Command & Control (C2) Framework**
- Multi-protocol implant communication (HTTP/HTTPS, WebSocket, DNS)
- Session management with encryption
- Task queuing and execution
- Data exfiltration pipelines
- Payload generation and delivery

**Social Engineering - LureCraft**
- Phishing page generation
- Credential harvesting
- Email template creation
- QR code phishing
- Multi-factor authentication bypass techniques

**Mission Management**
- Campaign planning and tracking
- Target profiling and reconnaissance
- Attack chain orchestration
- Results aggregation and reporting
- MITRE ATT&CK mapping

### 🔒 Security & Defense

**APT-Grade Hardening**
- 8-layer defense-in-depth architecture
- Container hardening (read-only filesystems, capability dropping)
- AppArmor and Seccomp profiles
- Network segmentation (internal + DMZ)
- Web Application Firewall (ModSecurity + OWASP CRS)
- Intrusion Detection System (Suricata)
- Zero-trust architecture

**Threat Intelligence Integration**
- MISP (Malware Information Sharing Platform)
- OpenCTI (Open Cyber Threat Intelligence)
- AbuseIPDB reputation tracking
- STIX 2.x indicator parsing
- IOC correlation and deduplication
- Automated blocking of high-confidence threats

### 🤖 AI/ML Capabilities (130 TOPS)

**Hardware Acceleration**
- NVIDIA CUDA GPU support (82 TOPS)
- Intel NPU optimization (48 TOPS)
- Mixed precision inference (FP16/TF32)
- Tensor Core acceleration
- Batch processing for efficiency

**AI-Powered Analysis**
- Vulnerability classification and prioritization
- Network anomaly detection
- Automated CVSS scoring
- Exploit prediction modeling
- Traffic pattern analysis
- Lateral movement detection

### 📊 Monitoring & Observability

**Metrics Collection**
- 50+ Prometheus metrics
- C2 session tracking
- Mission success rates
- Vulnerability statistics
- System resource monitoring
- API performance metrics

**Visualization**
- Grafana dashboards (13 panels)
- Real-time alerting (AlertManager)
- Classification banner compliance
- Custom metric queries
- Anomaly detection graphs

### 🚀 Deployment & Operations

**Production-Ready**
- Docker and Kubernetes support
- Automated deployment scripts
- systemd service management
- Rolling updates with zero downtime
- Health checks and self-healing
- Comprehensive backup/restore

**CI/CD Pipeline**
- 10-job automated workflow
- Security scanning (Trivy)
- Unit and integration testing
- Performance testing (k6)
- Code coverage tracking
- Automated deployments

---

## Quick Start

### Two Simple Entry Points

MEMSHADOW provides **only two commands** you need to know:

1. **`./install.sh`** - Install MEMSHADOW (one-time setup)
2. **`memshadow`** - Manage MEMSHADOW (all operations)

### Installation (Single Command)

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Run installer (interactive wizard)
sudo ./install.sh
```

The installer will:
- ✓ Check prerequisites (Docker, Python, etc.)
- ✓ Collect configuration through interactive prompts
- ✓ Generate secure secrets automatically
- ✓ Detect hardware (GPU/NPU)
- ✓ Configure threat intelligence feeds
- ✓ Install systemd service
- ✓ Start MEMSHADOW platform

### Post-Installation (All Operations Use `memshadow`)

```bash
# Check status
memshadow status

# View logs
memshadow logs memshadow -f

# Run health check
memshadow health

# Access web interfaces
# Main API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/<your-password>)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMSHADOW Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │    C2       │  │  LureCraft  │  │  Mission    │           │
│  │  Framework  │  │   Social    │  │ Management  │           │
│  │             │  │ Engineering │  │             │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              FastAPI REST + WebSocket                  │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      Security Layers                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌─────────────────┐    │
│  │  WAF   │→ │  IDS   │→ │ API    │→ │ Container       │    │
│  │ModSec  │  │Suricata│  │Hardening  │ Hardening       │    │
│  └────────┘  └────────┘  └────────┘  └─────────────────┘    │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────────────────┐       │
│  │ Threat Intel    │  │     AI/ML Engine             │       │
│  │ MISP | OpenCTI  │  │  GPU (82 TOPS) + NPU (48)    │       │
│  └─────────────────┘  └──────────────────────────────┘       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                       Data Layer                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌───────┐  ┌──────────┐                       │
│  │PostgreSQL│  │ Redis │  │ChromaDB  │                       │
│  │(Primary) │  │(Cache)│  │(Vectors) │                       │
│  └──────────┘  └───────┘  └──────────┘                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Monitoring Stack                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐  ┌─────────┐  ┌──────────────┐              │
│  │ Prometheus │→ │ Grafana │  │ AlertManager │              │
│  └────────────┘  └─────────┘  └──────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Network Architecture

```
                      Internet
                         │
                         ▼
                   ┌──────────┐
                   │   WAF    │  ModSecurity + OWASP CRS
                   │  :443    │
                   └──────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │    DMZ Network        │  172.30.0.0/24
             │  (Controlled Access)  │
             └───────────────────────┘
                         │
                         ▼
                   ┌──────────┐
                   │   IDS    │  Suricata
                   └──────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Internal Network     │  172.29.0.0/24
             │   (No Internet)       │  (Air-gapped)
             └───────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │MEMSHADOW│    │PostgreSQL│    │  Redis   │
   │  :8000  │    │  :5432   │    │  :6379   │
   └─────────┘    └──────────┘    └──────────┘
```

---

## Installation

### Prerequisites

**Required:**
- Linux operating system (Ubuntu 20.04+, Debian 11+, or RHEL 8+)
- Docker 20.10 or higher
- Docker Compose 2.0 or higher
- Python 3.8 or higher
- systemd (for service management)
- OpenSSL (for secret generation)
- Root/sudo access

**Optional:**
- NVIDIA GPU with CUDA support (for AI/ML acceleration)
- Intel CPU with NPU (for additional AI/ML acceleration)
- Kubernetes cluster (for production deployment)
- 16GB+ RAM (recommended for AI/ML features)
- 100GB+ disk space (for logs and data)

### Installation Methods

#### Method 1: Interactive Installer (Recommended)

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Run interactive installer
sudo ./install.sh
```

The installer provides a guided setup with:
- Automatic prerequisite validation
- Interactive configuration prompts
- Secure secret generation
- Hardware detection and optimization
- Service installation and startup

#### Method 2: Docker Compose (Development)

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Start services
./scripts/deploy-docker.sh up

# Initialize database
./scripts/deploy-docker.sh init-db
```

#### Method 3: Kubernetes (Production)

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Deploy to Kubernetes
./scripts/deploy-k8s.sh deploy

# Check status
./scripts/deploy-k8s.sh status
```

---

## Usage

### Management CLI (`memshadow`)

After installation, use the `memshadow` command for **all operations**:

```bash
# Service Control
memshadow start              # Start all services
memshadow stop               # Stop all services
memshadow restart            # Restart all services
memshadow status             # Show service status

# Monitoring
memshadow logs <service>     # View logs
memshadow logs memshadow -f  # Follow application logs
memshadow health             # Run comprehensive health checks

# Configuration
memshadow config show        # Display configuration (secrets masked)
memshadow config edit        # Edit configuration
memshadow config validate    # Validate configuration

# Maintenance
memshadow backup [path]      # Create backup
memshadow restore <file>     # Restore from backup
memshadow update             # Update to latest version

# Monitoring Stack
memshadow enable-monitoring  # Enable Prometheus + Grafana
memshadow disable-monitoring # Disable monitoring

# Removal
memshadow uninstall          # Completely remove MEMSHADOW
```

### API Usage

**Interactive API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Example API Calls:**

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Authentication
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# Register C2 session
curl -X POST http://localhost:8000/api/v1/c2/register \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"implant_id": "test-001", "os": "Windows 10"}'

# Create phishing page
curl -X POST http://localhost:8000/api/v1/lurecraft/pages \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"template": "microsoft-login", "redirect_url": "https://example.com"}'

# View metrics
curl http://localhost:8000/api/v1/metrics
```

### Web Interfaces

**Main Application:**
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Metrics: http://localhost:8000/api/v1/metrics

**Monitoring (if enabled):**
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093

---

## Security

### Hardening Features

**Container Security:**
- Read-only root filesystems
- All capabilities dropped
- AppArmor LSM profiles
- Seccomp syscall filtering
- Non-root users (UID 1000+)
- No SUID/SGID binaries
- Resource limits enforced

**Network Security:**
- Network segmentation (internal + DMZ)
- TLS 1.3 only
- ModSecurity WAF with OWASP CRS
- Suricata IDS with APT detection
- Rate limiting per endpoint
- IP reputation tracking

**Application Security:**
- HMAC request signatures
- SQL injection detection
- XSS prevention
- Path traversal blocking
- Attack tool detection (sqlmap, nmap, etc.)
- Automated threat blocking

**Secrets Management:**
- File-based Docker secrets
- 600 permissions on all secret files
- Automatic secret generation (32-byte tokens)
- No default passwords accepted

### Security Compliance

- CIS Docker Benchmark Level 1
- OWASP Top 10 mitigation
- MITRE ATT&CK mapping
- Zero-trust architecture
- Defense-in-depth (8 layers)

---

## Documentation

### Core Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute getting started guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide
- **[security/HARDENING_GUIDE.md](security/HARDENING_GUIDE.md)** - Security hardening
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - API documentation
- **[docs/OPERATOR_MANUAL.md](docs/OPERATOR_MANUAL.md)** - Operator guide
- **[scripts/README.md](scripts/README.md)** - All operational scripts

### API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## Requirements

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB
- OS: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+)

**Recommended:**
- CPU: 8+ cores (16+ for AI/ML)
- RAM: 16GB+ (32GB+ for AI/ML)
- Disk: 100GB+ SSD
- GPU: NVIDIA with CUDA support (for AI/ML)
- NPU: Intel NPU (for additional AI acceleration)

### Software Requirements

**Required:**
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+ (3.11+ recommended)
- systemd
- OpenSSL
- curl, git

**Optional:**
- Kubernetes 1.24+
- kubectl, Helm 3.0+
- NVIDIA drivers + CUDA 11.8+

---

## Project Structure

```
MEMSHADOW/
├── install.sh                    # ⭐ ENTRY POINT 1: Installer
├── README.md                     # This file
├── QUICKSTART.md                 # Quick start guide
├── ARCHITECTURE.md               # Architecture docs
│
├── app/                          # Application code
│   ├── main.py                   # FastAPI application
│   ├── api/v1/                   # API endpoints
│   ├── core/                     # Core functionality
│   ├── middleware/               # Middleware
│   ├── models/                   # Database models
│   ├── services/                 # Business logic
│   │   ├── c2/                  # C2 framework
│   │   ├── lurecraft/           # Social engineering
│   │   ├── threat_intel/        # Threat intelligence
│   │   └── ai_ml/               # AI/ML engine
│   └── utils/                    # Utilities
│
├── scripts/                      # Operational scripts
│   ├── memshadow-ctl.sh         # ⭐ ENTRY POINT 2: Management CLI
│   ├── deploy-docker.sh         # Docker deployment
│   ├── deploy-k8s.sh            # Kubernetes deployment
│   ├── validate-config.sh       # Config validation
│   ├── uninstall.sh             # Uninstaller
│   └── README.md                # Scripts documentation
│
├── security/                     # Security configuration
│   ├── apparmor/                # AppArmor profiles
│   ├── seccomp/                 # Seccomp profiles
│   ├── waf/                     # WAF configuration
│   └── ids/                     # IDS rules
│
├── k8s/                         # Kubernetes manifests
├── monitoring/                   # Monitoring stack
├── docs/                        # Documentation
├── tests/                       # Test suites
├── migrations/                  # Database migrations
│
├── docker-compose.yml           # Standard deployment
├── docker-compose.hardened.yml  # Production deployment
├── docker-compose.monitoring.yml # Monitoring stack
├── Dockerfile                   # Standard image
└── Dockerfile.hardened          # Hardened image
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Install dependencies
pip install -r requirements/development.txt

# Start development environment
./scripts/deploy-docker.sh up

# Run tests
pytest tests/
```

### Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=app --cov-report=html
```

---

## Troubleshooting

### Common Issues

**Installation fails:**
```bash
# Check prerequisites
docker --version
docker-compose --version

# Review logs
sudo ./install.sh 2>&1 | tee install.log
```

**Services won't start:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Check logs
memshadow logs memshadow

# Validate configuration
./scripts/validate-config.sh
```

**Health checks failing:**
```bash
# Run comprehensive check
memshadow health

# Check API endpoint
curl http://localhost:8000/api/v1/health
```

---

## License

**Proprietary License**

Copyright (c) 2025 SWORDIntel

This software is proprietary and confidential. Unauthorized copying, distribution,
or use of this software, via any medium, is strictly prohibited.

---

## Classification

**UNCLASSIFIED**

This platform handles sensitive security testing data. Ensure proper operational
security when deploying and using MEMSHADOW.

---

## Contact

- **Repository:** https://github.com/SWORDIntel/MEMSHADOW
- **Issues:** https://github.com/SWORDIntel/MEMSHADOW/issues
- **Security:** security@swordintel.com

---

**Built with ⚔️ by SWORDIntel**
