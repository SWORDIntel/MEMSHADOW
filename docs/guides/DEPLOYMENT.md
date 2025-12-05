# MEMSHADOW Deployment Guide

**Complete deployment guide for production environments**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Production Hardening](#production-hardening)
5. [Validation](#validation)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

For experienced operators:

```bash
# 1. Clone and setup
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# 2. Configure environment
cp config/.env.example .env
# Edit .env with secure credentials

# 3. Start with Docker
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

**For detailed instructions, see:**
- [Getting Started Guide](../getting-started/GETTING_STARTED.md) - New user guide
- [Production Deployment Guide](PRODUCTION_DEPLOYMENT.md) - Production setup
- [DSMILSYSTEM Deployment](../DSMILSYSTEM_DEPLOYMENT.md) - DSMILSYSTEM subsystem

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### Basic Deployment

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f memshadow
```

### Production Deployment

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# With hardened configuration
docker-compose -f docker-compose.hardened.yml up -d
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (optional)

### Quick Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n memshadow

# Access service
kubectl port-forward -n memshadow svc/memshadow-api 8000:8000
```

### Production Deployment

See [Production Deployment Guide](PRODUCTION_DEPLOYMENT.md) for complete Kubernetes setup.

---

## Production Hardening

### Security Checklist

- [ ] Changed default admin credentials
- [ ] Generated strong `WEB_SECRET_KEY`
- [ ] Configured `WEB_CORS_ORIGINS` with explicit domains
- [ ] Set unique passwords for PostgreSQL and Redis
- [ ] Running behind HTTPS/TLS
- [ ] Configured firewall rules
- [ ] Set up monitoring and alerting
- [ ] Configured automated backups

See [Production Security Guide](../PRODUCTION_SECURITY.md) for detailed security configuration.

---

## Validation

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/api/status

# Service health
docker-compose ps
```

### Functional Tests

```bash
# Run test suite
pytest tests/

# Integration tests
pytest tests/integration/

# Security tests
pytest tests/security/
```

---

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker
sudo systemctl status docker

# Check logs
docker-compose logs memshadow

# Check port conflicts
sudo lsof -i :8000
```

**Database connection errors:**
```bash
# Check PostgreSQL
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U memshadow -d memshadow -c "SELECT 1"
```

**Configuration errors:**
```bash
# Validate config
./scripts/validate-config.sh

# Check environment
docker-compose config
```

---

## Additional Resources

- **[Getting Started](../getting-started/GETTING_STARTED.md)** - New user guide
- **[Production Deployment](PRODUCTION_DEPLOYMENT.md)** - Production setup
- **[Architecture](../ARCHITECTURE.md)** - System architecture
- **[API Reference](../API_REFERENCE.md)** - API documentation
- **[Security Guide](../PRODUCTION_SECURITY.md)** - Security best practices

---

**For new users:** Start with [Getting Started Guide](../getting-started/GETTING_STARTED.md)
