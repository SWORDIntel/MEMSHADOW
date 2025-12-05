# DSMILSYSTEM Memory Subsystem - Pre-Deployment Guide

**Date:** 2025-01-XX  
**Status:** Pre-Deployment Setup  
**Version:** 1.0

---

## Overview

This guide covers the **pre-deployment** setup for MEMSHADOW with DSMILSYSTEM memory subsystem. There is **no existing database to migrate** - this is a fresh deployment.

---

## Architecture

### Dual Table System

MEMSHADOW uses two separate memory tables:

1. **`memories`** - Legacy table (used by existing APIs)
2. **`memories_dsmil`** - DSMILSYSTEM table (used by new APIs)

Both tables coexist independently:
- Legacy APIs → `memories` table
- DSMILSYSTEM APIs → `memories_dsmil` table
- No data migration needed
- No breaking changes

---

## Pre-Deployment Checklist

### 1. Prerequisites

- [ ] PostgreSQL 15+ installed
- [ ] Redis 7+ installed
- [ ] Python 3.11+ installed
- [ ] Docker & Docker Compose (optional, for containerized deployment)

### 2. Database Setup

```bash
# Create PostgreSQL database
createdb memshadow

# Or via Docker
docker run -d \
  --name memshadow-postgres \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=memshadow \
  -p 5432:5432 \
  postgres:15
```

### 3. Redis Setup

```bash
# Start Redis
redis-server

# Or via Docker
docker run -d \
  --name memshadow-redis \
  -p 6379:6379 \
  redis:7
```

### 4. SQLite Warm Tier Setup

```bash
# Create tmpfs mount point
sudo mkdir -p /tmp/memshadow_warm

# Mount tmpfs (adjust size as needed)
sudo mount -t tmpfs -o size=2G tmpfs /tmp/memshadow_warm

# Make it permanent (add to /etc/fstab)
echo "tmpfs /tmp/memshadow_warm tmpfs defaults,size=2G 0 0" | sudo tee -a /etc/fstab
```

### 5. Environment Configuration

Create `.env` file:

```bash
# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=memshadow
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost/memshadow

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION=memshadow_memories

# DSMILSYSTEM Configuration
DSMILSYSTEM_ENABLED=true
DSMILSYSTEM_WARM_TIER_PATH=/tmp/memshadow_warm
DSMILSYSTEM_DEFAULT_LAYER=6
DSMILSYSTEM_DEFAULT_DEVICE=0
DSMILSYSTEM_DEFAULT_CLEARANCE=UNCLASSIFIED

# Security
SECRET_KEY=your_secret_key_here
FIELD_ENCRYPTION_KEY=your_encryption_key_here
```

### 6. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements/base.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 7. Database Schema Creation

```bash
# Run Alembic migrations (creates fresh schema)
alembic upgrade head
```

This creates:
- Legacy `memories` table (if not exists)
- DSMILSYSTEM `memories_dsmil` table
- All indexes and constraints
- Required extensions (pgvector, pgcrypto, pg_trgm)

### 8. Initialize Services

```bash
# Start application
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or via Docker Compose
docker-compose up -d
```

### 9. Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check DSMILSYSTEM API
curl http://localhost:8000/dsmil/memory/store \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "layer_id": 6,
    "device_id": 0,
    "payload": {
      "content": "Test memory",
      "user_id": "00000000-0000-0000-0000-000000000000"
    },
    "tags": ["test"],
    "clearance": "UNCLASSIFIED"
  }'
```

---

## API Endpoints

### Legacy API (Backward Compatible)

```http
POST /api/v1/memory/ingest
GET /api/v1/memory/retrieve
GET /api/v1/memory/{memory_id}
PATCH /api/v1/memory/{memory_id}
DELETE /api/v1/memory/{memory_id}
```

**Uses:** `memories` table

### DSMILSYSTEM API (New)

```http
POST /dsmil/memory/store
POST /dsmil/memory/search
DELETE /dsmil/memory/{memory_id}
POST /dsmil/memory/compact
```

**Uses:** `memories_dsmil` table

---

## Configuration Reference

### Layer Mapping

| Layer ID | Name | Description |
|----------|------|-------------|
| 2 | Network/Infrastructure | Low-level network operations |
| 3 | Transport | Transport layer protocols |
| 4 | Session | Session management |
| 5 | Presentation | Data presentation |
| 6 | Application | Application layer (default) |
| 7 | User | User interactions |
| 8 | Data | Data management |
| 9 | Business Logic | Business logic layer |

### Device Mapping

- **0-103**: 104 devices total
- Device capabilities: CPU, NPU, GPU
- Device-to-layer mapping configurable

### Clearance Levels

- `UNCLASSIFIED` (default)
- `CONFIDENTIAL`
- `SECRET`
- `TOP_SECRET`

---

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database connectivity
psql -h localhost -U postgres -d memshadow -c "SELECT 1"

# Redis connectivity
redis-cli ping
```

### Logs

```bash
# Application logs
tail -f logs/memshadow.log

# Docker logs
docker-compose logs -f memshadow
```

### Metrics

Prometheus metrics available at:
```
http://localhost:8000/metrics
```

---

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -h localhost -U postgres -d memshadow
```

### Redis Connection Issues

```bash
# Check Redis is running
redis-cli ping

# Check Redis configuration
redis-cli CONFIG GET "*"
```

### SQLite Warm Tier Issues

```bash
# Check tmpfs mount
df -h | grep memshadow_warm

# Check directory permissions
ls -la /tmp/memshadow_warm
```

### Migration Issues

```bash
# Check migration status
alembic current

# View migration history
alembic history

# Rollback if needed
alembic downgrade -1
```

---

## Next Steps

1. **Test APIs**
   - Test legacy API endpoints
   - Test DSMILSYSTEM API endpoints
   - Verify both work independently

2. **Configure Monitoring**
   - Set up Prometheus scraping
   - Configure Grafana dashboards
   - Set up alerting

3. **Production Hardening**
   - Review security settings
   - Configure backup strategy
   - Set up log aggregation

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** ✅ Pre-Deployment Guide Complete
