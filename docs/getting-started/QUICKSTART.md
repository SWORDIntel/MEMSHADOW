# MEMSHADOW Quick Start Guide

**Get MEMSHADOW running in 5 minutes**

---

## Prerequisites Check

Before installing, ensure you have:

```bash
# Check Docker
docker --version  # Should be 20.10+

# Check Docker Compose
docker-compose --version  # Should be 2.0+

# Check Python
python3 --version  # Should be 3.8+

# Check systemd
systemctl --version

# Check if you have sudo/root access
sudo -v
```

If any command fails, install the missing requirement first.

---

## Installation (3 Steps)

### Step 1: Clone Repository

```bash
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW
```

### Step 2: Run Installer

```bash
sudo ./install.sh
```

The installer will ask you questions. Here's what to expect:

1. **Deployment Mode:**
   - Choose `production` for real use
   - Choose `development` for testing

2. **Database Password:**
   - Press Enter to auto-generate (recommended)
   - Or type your own (16+ characters)

3. **Redis/ChromaDB:**
   - Press Enter to auto-generate (recommended)

4. **Application Secrets:**
   - Press Enter to auto-generate (recommended)

5. **AI/ML Features:**
   - Type `y` if you have NVIDIA GPU
   - Type `n` if no GPU (will use CPU)

6. **Threat Intelligence:**
   - Type `y` if you have MISP/OpenCTI/AbuseIPDB keys
   - Type `n` to skip (can add later)

7. **Monitoring:**
   - Type `y` to enable Grafana dashboards
   - Type `n` to skip

8. **Start Service:**
   - Type `y` to start immediately
   - Type `n` to start manually later

### Step 3: Verify Installation

```bash
# Check service status
memshadow status

# Check health
memshadow health

# View logs
memshadow logs memshadow
```

---

## Access Your Platform

Once installed, access MEMSHADOW at:

**Main Application:**
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

**Monitoring (if enabled):**
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

---

## Essential Commands

### Service Control

```bash
# Start services
memshadow start

# Stop services
memshadow stop

# Restart services
memshadow restart

# Check status
memshadow status
```

### Monitoring

```bash
# View application logs (last 100 lines)
memshadow logs memshadow

# Follow logs in real-time
memshadow logs memshadow -f

# Run comprehensive health check
memshadow health
```

### Configuration

```bash
# View configuration (secrets masked)
memshadow config show

# Validate configuration
memshadow config validate

# Edit configuration
memshadow config edit
```

### Maintenance

```bash
# Create backup
memshadow backup

# Update to latest version
memshadow update

# Enable monitoring stack
memshadow enable-monitoring
```

---

## First API Call

### 1. Check Health

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "timestamp": "2025-11-16T..."
}
```

### 2. Get API Documentation

Open in browser: http://localhost:8000/docs

This provides interactive Swagger UI with all endpoints.

### 3. Authenticate

```bash
# Create user first (if needed)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "YourSecurePassword123!",
    "email": "admin@example.com"
  }'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "YourSecurePassword123!"
  }'
```

Save the token from the response.

### 4. Use Authenticated Endpoint

```bash
# Replace <TOKEN> with your actual token
curl -X GET http://localhost:8000/api/v1/c2/sessions \
  -H "Authorization: Bearer <TOKEN>"
```

---

## Common Tasks

### Create a C2 Session

```bash
curl -X POST http://localhost:8000/api/v1/c2/register \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "implant_id": "test-001",
    "os": "Windows 10",
    "hostname": "target-pc",
    "ip_address": "192.168.1.100"
  }'
```

### Create a Phishing Page

```bash
curl -X POST http://localhost:8000/api/v1/lurecraft/pages \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "microsoft-login",
    "redirect_url": "https://office.com",
    "collect_credentials": true
  }'
```

### Create a Mission

```bash
curl -X POST http://localhost:8000/api/v1/missions \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Red Team Exercise 2025",
    "description": "Simulated APT campaign",
    "objectives": [
      "Gain initial access",
      "Establish persistence",
      "Lateral movement"
    ],
    "target_organization": "Example Corp"
  }'
```

### View Metrics

```bash
# Prometheus format metrics
curl http://localhost:8000/api/v1/metrics
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check Docker is running
sudo systemctl status docker

# If not running, start it
sudo systemctl start docker

# Then try again
memshadow start
```

### Can't Access Web Interface

```bash
# Check if services are running
memshadow status

# Check if port 8000 is in use
sudo lsof -i :8000

# Check firewall
sudo ufw status
sudo ufw allow 8000/tcp
```

### Configuration Errors

```bash
# Validate configuration
./scripts/validate-config.sh

# View logs for errors
memshadow logs memshadow | grep ERROR

# Check database connectivity
docker exec memshadow-postgres pg_isready -U memshadow
```

### Health Checks Failing

```bash
# Run comprehensive check
memshadow health

# Check individual services
docker ps

# View service logs
memshadow logs postgres
memshadow logs redis
memshadow logs chromadb
```

###Installation Fails

```bash
# Check prerequisites
docker --version
docker-compose --version
python3 --version

# Check disk space
df -h

# Check permissions
sudo -v

# Re-run installer with verbose output
sudo ./install.sh 2>&1 | tee install.log
```

---

## Next Steps

### 1. Read Full Documentation

- **[README.md](README.md)** - Complete project documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide
- **[docs/OPERATOR_MANUAL.md](docs/OPERATOR_MANUAL.md)** - Operator manual
- **[security/HARDENING_GUIDE.md](security/HARDENING_GUIDE.md)** - Security guide

### 2. Configure Threat Intelligence

Edit configuration and add your API keys:

```bash
memshadow config edit
```

Add:
```bash
MEMSHADOW_THREAT_INTEL_ENABLED=true
MISP_URL=https://your-misp-server.com
MISP_KEY=your-api-key
OPENCTI_URL=https://your-opencti-server.com
OPENCTI_KEY=your-api-key
ABUSEIPDB_KEY=your-api-key
```

Restart services:
```bash
memshadow restart
```

### 3. Enable Monitoring

```bash
# Enable Prometheus + Grafana
memshadow enable-monitoring

# Access Grafana
# URL: http://localhost:3000
# Default: admin / (password from install)
```

### 4. Configure AI/ML (if you have GPU)

The installer auto-detects GPUs. Verify with:

```bash
# Check GPU detection
nvidia-smi

# Verify configuration
memshadow config show | grep AI
```

### 5. Set Up Backup Schedule

```bash
# Create manual backup
memshadow backup

# Set up cron for daily backups
sudo crontab -e

# Add this line (daily at 2 AM):
0 2 * * * /usr/local/bin/memshadow backup /var/backups/memshadow
```

### 6. Production Hardening

If running in production:

```bash
# Validate security configuration
./scripts/validate-config.sh

# Ensure production mode
memshadow config show | grep ENVIRONMENT
# Should show: ENVIRONMENT=production

# Enable all security features
memshadow config edit
# Set:
# MEMSHADOW_APT_DEFENSE=enabled
# MEMSHADOW_WAF_ENABLED=true
# MEMSHADOW_INTRUSION_DETECTION=enabled

# Restart
memshadow restart
```

---

## Getting Help

**Documentation:**
- In-app API docs: http://localhost:8000/docs
- Project README: [README.md](README.md)
- Scripts help: `memshadow help`

**Support:**
- Issues: https://github.com/SWORDIntel/MEMSHADOW/issues
- Security: security@swordintel.com

**Logs:**
```bash
# Application logs
memshadow logs memshadow

# All services
docker-compose -f /opt/memshadow/docker-compose.hardened.yml logs

# System logs
journalctl -u memshadow.service
```

---

## Uninstallation

If you need to remove MEMSHADOW:

```bash
# Complete removal (creates backup first)
memshadow uninstall

# Or run uninstaller directly
sudo ./scripts/uninstall.sh
```

---

## Summary

**Two commands to remember:**

1. **Installation:**
   ```bash
   sudo ./install.sh
   ```

2. **All Operations:**
   ```bash
   memshadow <command>
   ```

That's it! MEMSHADOW is designed to be simple to install and operate.

---

**Classification:** UNCLASSIFIED
