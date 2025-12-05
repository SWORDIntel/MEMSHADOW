# DSMILSYSTEM Memory Subsystem - Quick Start

**Pre-Deployment Setup** - No existing database to migrate

---

## Quick Setup (5 minutes)

### 1. Prerequisites

```bash
# Check PostgreSQL
psql --version  # Should be 15+

# Check Redis
redis-cli --version  # Should be 7+

# Check Python
python3 --version  # Should be 3.11+
```

### 2. Database & Redis Setup

```bash
# Start PostgreSQL (if not running)
sudo systemctl start postgresql
# Or: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15

# Start Redis (if not running)
redis-server
# Or: docker run -d -p 6379:6379 redis:7
```

### 3. Environment Configuration

```bash
# Copy example config
cp config/.env.example .env

# Edit .env and set:
# - POSTGRES_PASSWORD
# - REDIS_URL
# - SECRET_KEY
# - DSMILSYSTEM_ENABLED=true
```

### 4. Install & Run

```bash
# Install dependencies
pip install -r requirements/base.txt

# Create database schema (fresh deployment)
alembic upgrade head

# Start application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Verify

```bash
# Health check
curl http://localhost:8000/health

# Test DSMILSYSTEM API
curl -X POST http://localhost:8000/dsmil/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "layer_id": 6,
    "device_id": 0,
    "payload": {
      "content": "Hello DSMILSYSTEM",
      "user_id": "00000000-0000-0000-0000-000000000000"
    },
    "tags": ["test"],
    "clearance": "UNCLASSIFIED"
  }'
```

---

## Architecture

### Dual Table System

- **`memories`** - Legacy table (existing APIs)
- **`memories_dsmil`** - DSMILSYSTEM table (new APIs)

Both coexist independently - no migration needed!

---

## API Endpoints

### Legacy API
```
POST /api/v1/memory/ingest
GET  /api/v1/memory/retrieve
```

### DSMILSYSTEM API
```
POST   /dsmil/memory/store
POST   /dsmil/memory/search
DELETE /dsmil/memory/{memory_id}
POST   /dsmil/memory/compact
```

---

## Configuration

### Default Values

- **Layer:** 6 (Application layer)
- **Device:** 0
- **Clearance:** UNCLASSIFIED

### Environment Variables

```bash
DSMILSYSTEM_ENABLED=true
DSMILSYSTEM_WARM_TIER_PATH=/tmp/memshadow_warm
DSMILSYSTEM_DEFAULT_LAYER=6
DSMILSYSTEM_DEFAULT_DEVICE=0
DSMILSYSTEM_DEFAULT_CLEARANCE=UNCLASSIFIED
```

---

## Next Steps

1. **Read Full Documentation:**
   - `docs/DSMILSYSTEM_DEPLOYMENT.md` - Complete deployment guide
   - `docs/DSMILSYSTEM_REFACTOR_DESIGN.md` - Architecture design
   - `docs/DSMILSYSTEM_MIGRATION.md` - Deployment details

2. **Test APIs:**
   - Test legacy endpoints
   - Test DSMILSYSTEM endpoints
   - Verify both work independently

3. **Configure Monitoring:**
   - Set up Prometheus
   - Configure Grafana
   - Set up alerts

---

**Status:** ✅ Ready for Pre-Deployment
