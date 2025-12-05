# MEMSHADOW → DSMILSYSTEM Migration Guide

**Date:** 2025-01-XX  
**Status:** Migration Guide  
**Version:** 1.0

---

## Overview

This guide explains how to migrate from the legacy MEMSHADOW API to the new DSMILSYSTEM-aligned memory subsystem. The migration maintains backward compatibility through adapters while enabling new DSMILSYSTEM features.

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
- Default mappings: layer=6, device=0, clearance="UNCLASSIFIED"
- Existing data automatically migrated with defaults
- Gradual migration path available

---

## Migration Steps

### 1. Database Migration

Run the Alembic migration to add new columns:

```bash
# Create migration
alembic revision --autogenerate -m "Add DSMILSYSTEM columns"

# Review migration file
# Edit migrations/versions/XXX_add_dsmilsystem_columns.py

# Apply migration
alembic upgrade head
```

**Migration Script** (`migrations/versions/XXX_add_dsmilsystem_columns.py`):

```python
"""Add DSMILSYSTEM columns

Revision ID: xxxxx
Revises: yyyyy
Create Date: 2025-01-XX
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'xxxxx'
down_revision = 'yyyyy'
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to memories table
    op.add_column('memories', sa.Column('layer_id', sa.Integer(), nullable=True))
    op.add_column('memories', sa.Column('device_id', sa.Integer(), nullable=True))
    op.add_column('memories', sa.Column('clearance_token', sa.String(128), nullable=True))
    op.add_column('memories', sa.Column('correlation_id', sa.String(128), nullable=True))
    op.add_column('memories', sa.Column('roe_metadata', postgresql.JSONB(), nullable=True))
    op.add_column('memories', sa.Column('tier', sa.String(20), nullable=True))
    
    # Set defaults for existing records
    op.execute("UPDATE memories SET layer_id = 6 WHERE layer_id IS NULL")
    op.execute("UPDATE memories SET device_id = 0 WHERE device_id IS NULL")
    op.execute("UPDATE memories SET clearance_token = 'UNCLASSIFIED' WHERE clearance_token IS NULL")
    op.execute("UPDATE memories SET roe_metadata = '{}' WHERE roe_metadata IS NULL")
    op.execute("UPDATE memories SET tier = 'cold' WHERE tier IS NULL")
    
    # Make columns non-nullable
    op.alter_column('memories', 'layer_id', nullable=False)
    op.alter_column('memories', 'device_id', nullable=False)
    op.alter_column('memories', 'clearance_token', nullable=False)
    op.alter_column('memories', 'roe_metadata', nullable=False)
    op.alter_column('memories', 'tier', nullable=False)
    
    # Add indexes
    op.create_index('idx_layer_device', 'memories', ['layer_id', 'device_id'])
    op.create_index('idx_clearance_token', 'memories', ['clearance_token'])
    op.create_index('idx_correlation_id', 'memories', ['correlation_id'])
    op.create_index('idx_tier', 'memories', ['tier'])
    
    # Add check constraints
    op.create_check_constraint('check_layer_range', 'memories', 'layer_id >= 2 AND layer_id <= 9')
    op.create_check_constraint('check_device_range', 'memories', 'device_id >= 0 AND device_id <= 103')

def downgrade():
    # Remove indexes
    op.drop_index('idx_tier', 'memories')
    op.drop_index('idx_correlation_id', 'memories')
    op.drop_index('idx_clearance_token', 'memories')
    op.drop_index('idx_layer_device', 'memories')
    
    # Remove check constraints
    op.drop_constraint('check_device_range', 'memories')
    op.drop_constraint('check_layer_range', 'memories')
    
    # Remove columns
    op.drop_column('memories', 'tier')
    op.drop_column('memories', 'roe_metadata')
    op.drop_column('memories', 'correlation_id')
    op.drop_column('memories', 'clearance_token')
    op.drop_column('memories', 'device_id')
    op.drop_column('memories', 'layer_id')
```

### 2. Update Embedding Dimension

Update embedding dimension from 768d to 2048d:

```python
# In app/models/memory_dsmil.py
embedding = Column(Vector(2048))  # Changed from 768
```

**Note:** Existing embeddings will need to be regenerated or projected to 2048d.

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

If issues occur, rollback steps:

1. **Disable DSMILSYSTEM API:**
   ```bash
   export DSMILSYSTEM_ENABLED=false
   ```

2. **Revert database migration:**
   ```bash
   alembic downgrade -1
   ```

3. **Use legacy APIs only:**
   - All legacy APIs continue to work
   - No data loss
   - Gradual re-migration possible

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
