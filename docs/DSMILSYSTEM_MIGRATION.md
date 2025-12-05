# MEMSHADOW → DSMILSYSTEM Deployment Guide

**Date:** 2025-01-XX  
**Status:** Pre-Deployment Setup  
**Version:** 1.0

---

## Overview

This guide explains how to deploy MEMSHADOW with DSMILSYSTEM-aligned memory subsystem. This is a **pre-deployment** setup - **no existing database to migrate**. The system supports both legacy and DSMILSYSTEM APIs simultaneously using separate tables.

**For detailed deployment steps, see:** `docs/DSMILSYSTEM_DEPLOYMENT.md`

---

## What Changed

### New Features

1. **Layer Semantics** - All memories now tagged with layer_id (2-9)
2. **Device Semantics** - All memories tagged with device_id (0-103)
3. **Clearance Tokens** - Access control via clearance tokens
4. **Multi-Tier Storage** - Hot (Redis) → Warm (SQLite) → Cold (PostgreSQL)
5. **Event Bus** - Redis Streams for event routing
6. **Upward-Only Flows** - Lower layers cannot read higher-layer memory
7. **ROE Enforcement** - Rules of Engagement metadata and validation

### Backward Compatibility

- Legacy APIs continue to work via adapters
- Legacy APIs use `memories` table (separate from DSMILSYSTEM)
- DSMILSYSTEM APIs use `memories_dsmil` table
- Both systems can coexist independently

---

## Deployment Steps

### 1. Database Schema Creation

This is a **pre-deployment** setup - no existing database to migrate. Run Alembic to create the schema:

```bash
# Apply all migrations (creates fresh schema)
alembic upgrade head
```

**Migration File:** `migrations/versions/f1g2h3i4j5k6_add_dsmilsystem_memory_schema.py`

This migration creates:
- `memories_dsmil` table with DSMILSYSTEM semantics
- All required indexes and constraints
- Enum types for memory tiers

**Note:** The DSMILSYSTEM memory table (`memories_dsmil`) is separate from the legacy `memories` table. Both can coexist:
- Legacy APIs use `memories` table (via adapter)
- DSMILSYSTEM APIs use `memories_dsmil` table

### 2. Embedding Dimension

The DSMILSYSTEM schema uses 2048-dimensional embeddings by default:

```python
# In app/models/memory_dsmil.py
embedding = Column(Vector(2048))  # 2048d INT8 quantized
```

**Note:** The embedding service supports projection to 2048d from smaller models.

### 3. Configure SQLite Warm Tier

Set up tmpfs mount for SQLite warm tier:

```bash
# Create tmpfs mount point
sudo mkdir -p /tmp/memshadow_warm

# Mount tmpfs (adjust size as needed)
sudo mount -t tmpfs -o size=2G tmpfs /tmp/memshadow_warm

# Make it permanent (add to /etc/fstab)
echo "tmpfs /tmp/memshadow_warm tmpfs defaults,size=2G 0 0" | sudo tee -a /etc/fstab
```

### 4. Update Configuration

Add DSMILSYSTEM configuration to `.env`:

```bash
# DSMILSYSTEM Configuration
DSMILSYSTEM_ENABLED=true
DSMILSYSTEM_WARM_TIER_PATH=/tmp/memshadow_warm
DSMILSYSTEM_DEFAULT_LAYER=6
DSMILSYSTEM_DEFAULT_DEVICE=0
DSMILSYSTEM_DEFAULT_CLEARANCE=UNCLASSIFIED
```

### 5. Update Code

#### Option A: Use Legacy Adapter (No Code Changes)

Existing code continues to work via `LegacyMemoryAdapter`:

```python
from app.services.memory_service_legacy_adapter import LegacyMemoryAdapter

# Existing code works unchanged
adapter = LegacyMemoryAdapter(db)
memory = await adapter.create_memory(user_id, content, extra_data)
```

#### Option B: Migrate to DSMILSYSTEM API

Update code to use new API:

```python
from app.services.memory_service_dsmil import MemoryServiceDSMIL

# New DSMILSYSTEM API
service = MemoryServiceDSMIL(db)
memory_id = await service.store_memory(
    layer_id=6,
    device_id=0,
    payload={"content": content, "user_id": str(user_id)},
    tags=["tag1", "tag2"],
    ttl=3600,
    clearance="UNCLASSIFIED",
    context={"correlation_id": "xxx"}
)
```

---

## API Changes

### Legacy API (Still Works)

```http
POST /api/v1/memory/ingest
{
  "content": "Memory content",
  "extra_data": {}
}
```

### New DSMILSYSTEM API

```http
POST /dsmil/memory/store
{
  "layer_id": 6,
  "device_id": 0,
  "payload": {
    "content": "Memory content",
    "user_id": "user-uuid"
  },
  "tags": ["tag1"],
  "ttl": 3600,
  "clearance": "UNCLASSIFIED",
  "context": {
    "correlation_id": "xxx"
  }
}
```

---

## Testing

### Unit Tests

```bash
# Run DSMILSYSTEM tests
pytest tests/test_memory_service_dsmil.py -v

# Run legacy adapter tests
pytest tests/test_memory_service_legacy_adapter.py -v
```

### Integration Tests

```bash
# Test multi-tier storage
pytest tests/integration/test_dsmil_multi_tier.py -v

# Test clearance enforcement
pytest tests/integration/test_dsmil_clearance.py -v

# Test upward-only flows
pytest tests/integration/test_dsmil_upward_only.py -v
```

---

## Rollback Plan

If issues occur during deployment:

1. **Disable DSMILSYSTEM API:**
   ```bash
   export DSMILSYSTEM_ENABLED=false
   ```

2. **Revert database migration:**
   ```bash
   alembic downgrade -1
   ```

3. **Use legacy APIs only:**
   - Legacy APIs use separate `memories` table
   - No impact on DSMILSYSTEM `memories_dsmil` table
   - Can re-enable DSMILSYSTEM when ready

---

## Performance Considerations

### Multi-Tier Storage

- **Hot Tier (Redis):** < 1ms access, 1 hour TTL
- **Warm Tier (SQLite):** < 10ms access, tmpfs-mounted
- **Cold Tier (PostgreSQL):** < 100ms access, persistent

### Search Strategy

Searches check tiers in order:
1. Hot tier (fastest, limited results)
2. Warm tier (fast, recent access)
3. Cold tier (slower, comprehensive)

---

## Monitoring

### Metrics

New Prometheus metrics:
- `memshadow_memory_operations_total{layer,device,operation,tier}`
- `memshadow_memory_access_denied_total{reason}`
- `memshadow_memory_tier_promotions_total{from,to}`
- `memshadow_memory_search_latency_seconds{tier}`

### Logs

Structured logs include:
- `layer_id`, `device_id`, `clearance_token`
- `correlation_id` for event tracking
- `tier` for storage tier
- `access_decision` for clearance checks

---

## Support

For issues or questions:
- **Documentation:** See `docs/DSMILSYSTEM_REFACTOR_DESIGN.md`
- **API Reference:** See `docs/MEMSHADOW_API.md`
- **GitHub Issues:** https://github.com/SWORDIntel/MEMSHADOW/issues

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** ✅ Migration Guide Complete
