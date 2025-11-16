# MEMSHADOW Scripts Directory

**Classification:** UNCLASSIFIED

This directory contains operational scripts for MEMSHADOW installation, deployment, and management.

## Installation Scripts

### `install.sh` - Interactive Installation Wizard
**Purpose:** Primary entry point for installing MEMSHADOW with guided configuration.

**Features:**
- Interactive configuration wizard
- Automatic secret generation
- Hardware detection (GPU/NPU)
- systemd service installation
- Comprehensive validation

**Usage:**
```bash
sudo ./install.sh
```

**What it does:**
1. Checks prerequisites (Docker, Docker Compose, systemd, etc.)
2. Collects configuration through interactive prompts
3. Generates secure secrets (32-byte random tokens)
4. Creates installation directories with proper permissions
5. Installs systemd service
6. Deploys management CLI
7. Optionally starts services

**Requirements:**
- Root/sudo access
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- OpenSSL
- systemd

---

### `uninstall.sh` - Complete Removal
**Purpose:** Safely remove MEMSHADOW from the system with backup.

**Usage:**
```bash
sudo ./uninstall.sh
```

**What it removes:**
- Docker containers and volumes
- Installation directory (`/opt/memshadow`)
- Data directory (`/var/lib/memshadow`)
- Log directory (`/var/log/memshadow`)
- systemd service
- Service user
- Management CLI

**Safety features:**
- Creates final backup before removal
- Confirmation prompts
- Optional configuration preservation
- Optional Docker image retention

---

### `memshadow-ctl.sh` - Management CLI
**Purpose:** Primary management interface for MEMSHADOW operations.

**Installation:**
Automatically installed to `/usr/local/bin/memshadow` during setup.

**Usage:**
```bash
memshadow <command> [options]
```

**Commands:**

#### Service Control
```bash
memshadow start              # Start all services
memshadow stop               # Stop all services
memshadow restart            # Restart all services
memshadow status             # Show service status
```

#### Monitoring & Logs
```bash
memshadow logs <service>         # View logs (last 100 lines)
memshadow logs <service> -f      # Follow logs in real-time
memshadow health                 # Comprehensive health check
```

#### Configuration
```bash
memshadow config show            # Display current config (secrets masked)
memshadow config edit            # Edit configuration
memshadow config validate        # Validate configuration
```

#### Maintenance
```bash
memshadow update                 # Update to latest version
memshadow backup [path]          # Create backup
memshadow restore <file>         # Restore from backup
```

#### Monitoring Stack
```bash
memshadow enable-monitoring      # Enable Prometheus + Grafana
memshadow disable-monitoring     # Disable monitoring stack
```

**Examples:**
```bash
# Check status
memshadow status

# Follow application logs
memshadow logs memshadow -f

# Run health check
memshadow health

# Create backup
memshadow backup

# Validate configuration
memshadow config validate
```

---

## Deployment Scripts

### `deploy-docker.sh` - Docker Compose Deployment
**Purpose:** Manage Docker Compose deployments (development and production).

**Usage:**
```bash
./scripts/deploy-docker.sh <command> [options]

# Commands:
up [production]        # Start services
down                   # Stop and remove services
restart                # Restart all services
logs [service]         # View logs
build                  # Build images
clean                  # Clean up volumes and networks
status                 # Show service status
init-db                # Initialize database
shell <service>        # Open shell in container
```

**Examples:**
```bash
# Start development environment
./scripts/deploy-docker.sh up

# Start production environment
./scripts/deploy-docker.sh up production

# View all logs
./scripts/deploy-docker.sh logs

# View specific service logs
./scripts/deploy-docker.sh logs memshadow

# Initialize database
./scripts/deploy-docker.sh init-db
```

---

### `deploy-k8s.sh` - Kubernetes Deployment
**Purpose:** Deploy MEMSHADOW to Kubernetes clusters.

**Usage:**
```bash
./scripts/deploy-k8s.sh <command> [options]

# Commands:
deploy                 # Deploy to Kubernetes
delete                 # Remove from Kubernetes
status                 # Show deployment status
logs <pod>             # View pod logs
scale <replicas>       # Scale deployment
update                 # Update deployment
port-forward           # Setup port forwarding
```

**Examples:**
```bash
# Deploy to Kubernetes
./scripts/deploy-k8s.sh deploy

# Check status
./scripts/deploy-k8s.sh status

# Scale to 5 replicas
./scripts/deploy-k8s.sh scale 5

# Port forward for local access
./scripts/deploy-k8s.sh port-forward
```

---

### `validate-deployment.sh` - Deployment Validation
**Purpose:** Comprehensive validation of MEMSHADOW deployments.

**Usage:**
```bash
./scripts/validate-deployment.sh [docker|kubernetes]
```

**Checks performed:**
- Service availability
- Health endpoints
- Database connectivity
- API functionality
- Resource allocation
- Network connectivity
- Security configuration

**Exit codes:**
- `0` - All checks passed
- `1` - Validation failed

---

## Configuration Scripts

### `validate-config.sh` - Configuration Validation
**Purpose:** Validate MEMSHADOW configuration for security and completeness.

**Usage:**
```bash
./scripts/validate-config.sh
```

**Validation checks:**
1. Configuration directory structure
2. File permissions (600 for secrets)
3. Required environment variables
4. Secret strength (minimum 16 characters)
5. Database configuration
6. Application settings
7. Network configuration
8. AI/ML settings
9. Threat intelligence configuration
10. Security hardening settings
11. Monitoring configuration
12. Individual secret files
13. Production readiness
14. Docker requirements

**Exit codes:**
- `0` - Configuration valid
- `1` - Configuration invalid

---

### `generate-secrets.sh` - Secret Generation
**Purpose:** Generate cryptographically secure secrets for MEMSHADOW.

**Usage:**
```bash
./scripts/generate-secrets.sh
```

**Generated secrets:**
- PostgreSQL password (32 bytes)
- Redis password (32 bytes)
- ChromaDB token (32 bytes)
- Application SECRET_KEY (32 bytes)
- JWT secret key (32 bytes)
- Database URL (with generated password)

**Output:**
- Creates `secrets/` directory
- Generates individual secret files
- Sets file permissions to 600
- Creates Docker secret files

**Security:**
- Uses `/dev/urandom` via OpenSSL
- Python `secrets` module for tokens
- No default/predictable values
- Automatic permission hardening

---

## Security Scripts

### `security-audit.sh` - Security Audit
**Purpose:** Perform comprehensive security audit of MEMSHADOW installation.

**Usage:**
```bash
./scripts/security-audit.sh
```

**Audit areas:**
- Container security (capabilities, privileges)
- File permissions
- Network exposure
- Secret management
- TLS/SSL configuration
- AppArmor/Seccomp profiles
- User/group configuration
- Docker security best practices

---

## Utility Scripts

### Helper Functions
All scripts include:
- Color-coded output (info/success/warning/error)
- Banner displays
- Prerequisite checking
- Error handling
- Logging capabilities

### Common Patterns

**Root check:**
```bash
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root"
    exit 1
fi
```

**Installation check:**
```bash
if [[ ! -d "/opt/memshadow" ]]; then
    echo "MEMSHADOW not installed"
    exit 1
fi
```

---

## File Locations

### Installation Paths
- **Installation Root:** `/opt/memshadow`
- **Configuration:** `/etc/memshadow`
- **Secrets:** `/etc/memshadow/secrets`
- **Data:** `/var/lib/memshadow`
- **Logs:** `/var/log/memshadow`
- **Service:** `/etc/systemd/system/memshadow.service`
- **CLI:** `/usr/local/bin/memshadow`

### Configuration Files
- **Main Config:** `/etc/memshadow/secrets.env`
- **Secret Files:** `/etc/memshadow/secrets/*.txt`
- **Docker Secrets:** `/etc/memshadow/secrets/docker/*.txt`

---

## Troubleshooting

### Common Issues

**Script not executable:**
```bash
chmod +x scripts/<script-name>.sh
```

**Permission denied:**
```bash
sudo ./scripts/<script-name>.sh
```

**Docker daemon not running:**
```bash
sudo systemctl start docker
```

**Service fails to start:**
```bash
# Check logs
memshadow logs memshadow

# Validate configuration
./scripts/validate-config.sh

# Check service status
systemctl status memshadow.service
```

**Configuration errors:**
```bash
# Validate
./scripts/validate-config.sh

# Edit configuration
memshadow config edit

# Regenerate secrets
./scripts/generate-secrets.sh
```

---

## Development

### Adding New Scripts

1. **Create script in `scripts/` directory**
2. **Add shebang:** `#!/bin/bash`
3. **Set strict mode:** `set -euo pipefail`
4. **Include logging functions**
5. **Add prerequisite checks**
6. **Make executable:** `chmod +x scripts/<name>.sh`
7. **Update this README**
8. **Add to install.sh if needed**

### Script Template
```bash
#!/bin/bash
# MEMSHADOW <Purpose>
# Classification: UNCLASSIFIED

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Main logic here
```

---

## Security Considerations

### Secret Handling
- All secrets stored with 600 permissions
- Secrets never logged or displayed
- Configuration masked in output
- Backup encryption recommended

### Root Operations
Scripts requiring root:
- `install.sh` - Creates system directories, installs service
- `uninstall.sh` - Removes system files
- `memshadow-ctl.sh` - Service control (most commands)

### Audit Trail
All operations logged to:
- systemd journal: `journalctl -u memshadow.service`
- Docker logs: `docker-compose logs`
- Application logs: `/var/log/memshadow/`

---

## Support

**Documentation:**
- Installation: `/opt/memshadow/security/HARDENING_GUIDE.md`
- Deployment: `/opt/memshadow/docs/DEPLOYMENT.md`
- API: `http://localhost:8000/docs`

**Issues:**
https://github.com/SWORDIntel/MEMSHADOW/issues

**Help:**
```bash
memshadow help
./install.sh --help
```

---

**Classification:** UNCLASSIFIED
**Version:** 2.1
**Last Updated:** 2025-11-16
