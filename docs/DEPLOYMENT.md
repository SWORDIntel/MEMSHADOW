# MEMSHADOW Deployment Guide

**Classification:** UNCLASSIFIED
**Version:** 2.1
**Date:** 2025-11-16

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Hardening](#production-hardening)
6. [Validation](#validation)
7. [Troubleshooting](#troubleshooting)

---

## Overview

MEMSHADOW supports two deployment methods:

1. **Docker Compose** - Recommended for development, testing, and small-scale deployments
2. **Kubernetes** - Recommended for production, high-availability deployments

Both methods provide:
- Automated deployment scripts
- Health monitoring
- Persistent storage
- Network isolation
- Secret management

---

## Prerequisites

### Common Requirements

- **Authorization**: Written authorization for security testing required
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Memory**: Minimum 16GB RAM (32GB+ recommended for production)
- **Storage**: 100GB+ free disk space
- **Network**: Internet connectivity for pulling images

### Docker Deployment

- Docker Engine 20.10+
- Docker Compose 2.0+
- Ports available: 8000, 8001, 8080, 8443, 5432, 6379

### Kubernetes Deployment

- Kubernetes 1.24+
- kubectl configured
- Persistent Volume provisioner
- LoadBalancer or Ingress controller (for external access)
- 4+ CPU cores and 16GB+ RAM per node

---

## Docker Deployment

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# 2. Configure environment
cp .env.example .env
nano .env  # Edit with your configuration

# 3. Deploy
./scripts/deploy-docker.sh up

# 4. Validate
./scripts/validate-deployment.sh
```

### Detailed Steps

#### 1. Environment Configuration

Edit `.env` file and update these critical values:

```bash
# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
CHROMA_TOKEN=$(openssl rand -base64 32)
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Add these to your `.env` file.

#### 2. Build and Deploy

```bash
# Build images (first time or after code changes)
./scripts/deploy-docker.sh build

# Start services
./scripts/deploy-docker.sh up

# Start with production profile (includes nginx reverse proxy)
./scripts/deploy-docker.sh up production
```

#### 3. Initialize Database

```bash
# Database is automatically initialized on first run
# To manually reinitialize:
./scripts/deploy-docker.sh init-db
```

#### 4. Verify Deployment

```bash
# Check service status
./scripts/deploy-docker.sh status

# View logs
./scripts/deploy-docker.sh logs

# Run validation
./scripts/validate-deployment.sh
```

### Access Points

- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **C2 Service**: https://localhost:8443
- **TEMPEST Dashboard**: http://localhost:8080
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **ChromaDB**: http://localhost:8001

### Management Commands

```bash
# View logs
./scripts/deploy-docker.sh logs

# Restart services
./scripts/deploy-docker.sh restart

# Stop services
./scripts/deploy-docker.sh down

# Clean up (removes all data)
./scripts/deploy-docker.sh clean

# Open shell in container
./scripts/deploy-docker.sh shell memshadow
```

---

## Kubernetes Deployment

### Quick Start

```bash
# 1. Configure kubectl
kubectl config use-context your-cluster

# 2. Deploy
./scripts/deploy-k8s.sh deploy

# 3. Validate
./scripts/validate-deployment.sh
```

### Detailed Steps

#### 1. Prepare Cluster

Ensure your Kubernetes cluster has:

```bash
# Check cluster connectivity
kubectl cluster-info

# Verify storage class
kubectl get storageclass

# Verify you have sufficient resources
kubectl top nodes
```

#### 2. Configure Secrets

Secrets are automatically generated during deployment, or create manually:

```bash
# Generate secrets manually
./scripts/deploy-k8s.sh secrets

# Or create from file
kubectl create secret generic memshadow-secrets \
  --from-env-file=.env \
  --namespace memshadow
```

#### 3. Deploy Services

```bash
# Deploy all components
./scripts/deploy-k8s.sh deploy

# The deployment script will:
# - Create namespace
# - Apply ConfigMaps
# - Generate secrets (if needed)
# - Create PVCs
# - Deploy PostgreSQL, Redis, ChromaDB
# - Deploy MEMSHADOW application
# - Apply network policies
```

#### 4. Verify Deployment

```bash
# Check deployment status
./scripts/deploy-k8s.sh status

# View logs
./scripts/deploy-k8s.sh logs

# Run validation
./scripts/validate-deployment.sh
```

### Access Points

#### Using Port Forwarding (Development)

```bash
# Forward all services to localhost
./scripts/deploy-k8s.sh port-forward

# Access at:
# - http://localhost:8000 (Main API)
# - https://localhost:8443 (C2 Service)
# - http://localhost:8080 (TEMPEST)
```

#### Using LoadBalancer (Production)

```bash
# Get external IPs
kubectl get svc -n memshadow

# Access services via external IPs
```

#### Using Ingress (Production)

Update `k8s/base/ingress.yaml` with your domain:

```yaml
spec:
  tls:
  - hosts:
    - memshadow.yourdomain.com
    - c2.yourdomain.com
    - tempest.yourdomain.com
```

Apply ingress:

```bash
kubectl apply -f k8s/base/ingress.yaml
```

### Management Commands

```bash
# View logs
./scripts/deploy-k8s.sh logs memshadow

# Restart deployment
./scripts/deploy-k8s.sh restart

# Scale deployment
./scripts/deploy-k8s.sh scale 5

# Open shell
./scripts/deploy-k8s.sh shell memshadow

# Delete deployment
./scripts/deploy-k8s.sh delete
```

---

## Production Hardening

### Security Checklist

#### 1. Secrets Management

- [ ] Use strong random passwords (32+ characters)
- [ ] Rotate secrets regularly (90 days)
- [ ] Never commit secrets to git
- [ ] Use Kubernetes Secrets or HashiCorp Vault
- [ ] Enable secret encryption at rest

```bash
# Generate secure secrets
openssl rand -base64 32
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 2. Network Security

- [ ] Enable network policies (already configured)
- [ ] Use TLS for all external traffic
- [ ] Restrict C2 service to authorized IPs
- [ ] Configure firewall rules
- [ ] Enable DDoS protection

```bash
# Docker: Use nginx reverse proxy with SSL
./scripts/deploy-docker.sh up production

# Kubernetes: Configure Ingress with TLS
kubectl apply -f k8s/base/ingress.yaml
```

#### 3. Access Control

- [ ] Enable authentication on all services
- [ ] Use RBAC for Kubernetes
- [ ] Implement MFA for operators
- [ ] Audit all access attempts
- [ ] Restrict shell access

#### 4. Monitoring & Logging

- [ ] Enable audit logging
- [ ] Configure log aggregation (ELK, Splunk)
- [ ] Set up alerts for anomalies
- [ ] Monitor resource usage
- [ ] Implement SIEM integration

```bash
# View audit logs
docker-compose exec memshadow tail -f /app/logs/audit.log

# Kubernetes
kubectl logs -n memshadow -l app.kubernetes.io/name=memshadow --tail=100
```

#### 5. Resource Limits

Docker Compose resource limits are configured in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

Kubernetes limits are in deployment manifests.

#### 6. Backup & Recovery

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U memshadow memshadow > backup.sql

# Kubernetes backup
kubectl exec -n memshadow postgres-xxx -- pg_dump -U memshadow memshadow > backup.sql

# Backup ChromaDB
docker-compose exec memshadow tar -czf /data/chromadb-backup.tar.gz /data/chromadb
```

#### 7. Updates & Patching

```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Update Kubernetes deployment
kubectl set image deployment/memshadow memshadow=memshadow:2.2 -n memshadow
```

### TLS Configuration

#### Generate Self-Signed Certificates

```bash
# Create certs directory
mkdir -p certs

# Generate self-signed cert (development only)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/server.key \
  -out certs/server.crt \
  -days 365 \
  -subj "/CN=memshadow.local"
```

#### Production Certificates

Use Let's Encrypt with cert-manager (Kubernetes):

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

---

## Validation

### Automated Validation

```bash
# Run validation script
./scripts/validate-deployment.sh

# Expected output:
# ✓ PostgreSQL container running
# ✓ Redis container running
# ✓ ChromaDB container running
# ✓ MEMSHADOW container running
# ✓ Main API health endpoint
# ...
# All critical checks passed!
```

### Manual Health Checks

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"healthy","version":"2.1"}

# Check PostgreSQL
docker-compose exec postgres pg_isready -U memshadow

# Check Redis
docker-compose exec redis redis-cli ping

# Check ChromaDB
curl http://localhost:8001/api/v1/heartbeat
```

### Performance Testing

```bash
# API load test (requires apache-bench)
ab -n 1000 -c 10 http://localhost:8000/api/v1/health

# Database connection test
docker-compose exec memshadow python -c "
from app.db.session import engine
with engine.connect() as conn:
    result = conn.execute('SELECT 1')
    print('Database OK:', result.fetchone())
"
```

---

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check logs
./scripts/deploy-docker.sh logs

# Common causes:
# - Port conflicts: Check if ports 8000, 5432, 6379, 8001 are in use
# - Resource limits: Ensure sufficient RAM/CPU
# - Permission issues: Check file ownership
```

#### Database Connection Failed

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U memshadow -d memshadow -c "SELECT 1"

# Reset database (WARNING: destroys data)
docker-compose down -v
docker-compose up -d
```

#### C2 Service Not Accessible

```bash
# Check if port 8443 is open
netstat -an | grep 8443

# Check certificate configuration
ls -la certs/

# Generate new certificates if missing
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/server.key \
  -out certs/server.crt \
  -days 365
```

#### Kubernetes Pod CrashLoopBackOff

```bash
# Check pod logs
kubectl logs -n memshadow -l app.kubernetes.io/name=memshadow --tail=100

# Describe pod for events
kubectl describe pod -n memshadow <pod-name>

# Common causes:
# - Missing secrets: ./scripts/deploy-k8s.sh secrets
# - Resource limits: Check node resources
# - Image pull errors: Verify image name/registry
```

#### PVC Pending

```bash
# Check PVC status
kubectl get pvc -n memshadow

# Describe PVC for errors
kubectl describe pvc postgres-pvc -n memshadow

# Common causes:
# - No storage class: kubectl get storageclass
# - Insufficient storage: Check node capacity
# - Permission issues: Check RBAC
```

### Log Collection

```bash
# Docker Compose
docker-compose logs > memshadow-logs.txt

# Kubernetes
kubectl logs -n memshadow --all-containers=true --prefix=true > k8s-logs.txt

# Collect diagnostic info
kubectl describe all -n memshadow > k8s-describe.txt
kubectl get events -n memshadow --sort-by='.lastTimestamp' > k8s-events.txt
```

### Performance Tuning

```bash
# Increase PostgreSQL connections
# Edit docker-compose.yml or k8s deployment:
command: ["-c", "max_connections=200"]

# Increase worker processes
# Edit docker-compose.yml:
command: ["uvicorn", "app.main:app", "--workers", "8"]

# Kubernetes horizontal scaling
kubectl scale deployment/memshadow --replicas=5 -n memshadow

# Enable connection pooling in application config
```

---

## Support

For issues, questions, or feature requests:

1. Check logs and run validation script
2. Review [OPERATOR_MANUAL.md](OPERATOR_MANUAL.md)
3. Consult [API_REFERENCE.md](API_REFERENCE.md)
4. Submit issue to repository

---

**Classification:** UNCLASSIFIED
**For authorized security testing and defensive research only**
