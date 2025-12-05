# MEMSHADOW → DSMILSYSTEM Refactor Design Note

**Date:** 2025-01-XX  
**Status:** Design Phase  
**Author:** Lead Architect

---

## Executive Summary

This document outlines the refactoring plan to bring MEMSHADOW into full alignment with DSMILSYSTEM architecture, conventions, and operational patterns. MEMSHADOW will become the canonical memory subsystem for all DSMILSYSTEM layers (2-9) and devices (0-103), with proper clearance token enforcement and upward-only data flows.

---

## 1. Current MEMSHADOW Architecture

### 1.1 Core Components

**Memory Service** (`app/services/memory_service.py`)
- User-scoped memory storage (user_id-based)
- PostgreSQL for metadata storage
- ChromaDB for vector embeddings
- Redis for caching
- No layer/device semantics
- No clearance token system
- No upward-only flow enforcement

**Data Fabric**
- **PostgreSQL**: Primary relational store (`app/db/postgres.py`)
  - `memories` table with user_id, content, embedding (768d), extra_data (JSONB)
  - No layer_id, device_id, clearance_token columns
- **Redis**: Caching and rate limiting (`app/db/redis.py`)
  - Basic key-value cache
  - No Redis Streams usage
  - No per-layer/per-device streams
- **ChromaDB**: Vector store (`app/db/chromadb.py`)
  - Collection-based storage
  - No layer/device partitioning
- **SQLite**: Not used in current architecture

**Embedding Service** (`app/services/embedding_service.py`)
- Sentence Transformers or OpenAI backends
- Optional projection layer to 2048d
- No INT8 quantization support
- No DSMILSYSTEM MLOps pipeline integration
- Device detection (CPU/GPU) but no device mapping rules

**API Surface** (`app/api/v1/memory.py`)
- `/ingest` - Create memory
- `/retrieve` - Search memories
- `/get/{id}` - Get memory by ID
- `/update/{id}` - Update memory
- `/delete/{id}` - Delete memory
- No layer/device parameters
- No clearance token parameters

**Event System**
- Celery for background tasks
- No event bus integration
- No Redis Streams for events
- No per-layer IN/OUT streams
- No correlation IDs

**Observability**
- Structlog for logging
- Basic Prometheus metrics (`app/core/metrics.py`)
- No per-layer/per-device metrics
- No clearance denial metrics
- No audit trail for clearance violations

---

## 2. Desired DSMILSYSTEM-Aligned Architecture

### 2.1 Data Fabric Alignment

**Redis Streams** (Hot Tier)
- Per-layer streams: `layer:{layer_id}:in`, `layer:{layer_id}:out`
- Per-device streams: `device:{device_id}:events`
- Event schemas with correlation IDs
- Hot memory cache with TTL

**SQLite** (Warm Tier)
- tmpfs-mounted SQLite for hot memory
- Per-layer databases or partitioned tables
- Fast local access for frequently accessed memories
- Automatic promotion/demotion logic

**PostgreSQL** (Cold Tier)
- Long-term storage with layer/device partitioning
- Vector storage with pgvector extension
- Full audit trail
- Clearance token metadata

**Event Bus Integration**
- All memory operations emit events
- Event routing by layer/device
- Correlation ID tracking
- Schema validation

### 2.2 Layer & Device Semantics

**Layer System** (Layers 2-9)
- Layer 2: Network/Infrastructure
- Layer 3: Transport
- Layer 4: Session
- Layer 5: Presentation
- Layer 6: Application
- Layer 7: User
- Layer 8: Data
- Layer 9: Business Logic

**Device System** (0-103)
- 104 devices total
- Device mapping to layers
- Device capabilities (CPU/NPU/GPU)
- Device-specific memory quotas

**Clearance Tokens**
- Token-based access control
- ROE (Rules of Engagement) metadata
- Upward-only flow enforcement
- Audit logging for violations

### 2.3 MLOps Pipeline Integration

**INT8 Quantization**
- Quantized embedding models
- Model optimization pipeline
- Device-specific model variants
- Performance monitoring

**Model Artifacts**
- Versioned model storage
- A/B testing support
- Rollback capabilities
- Model registry integration

### 2.4 API Surface

**Canonical API**
```python
store_memory(
    layer: int,           # 2-9
    device: int,          # 0-103
    payload: Dict,
    tags: List[str],
    ttl: Optional[int],
    clearance: str,       # Clearance token
    context: Dict         # Correlation ID, etc.
) -> MemoryID

search_memory(
    layer: int,
    device: int,
    query: str,
    k: int,
    filters: Dict,
    clearance: str       # Required for cross-layer reads
) -> List[Memory]

delete_memory(
    memory_id: UUID,
    layer: int,
    device: int,
    clearance: str
) -> bool

compact_memory(
    layer: int,
    device: int,
    clearance: str
) -> CompactResult
```

**Legacy Adapters**
- Thin wrappers around canonical API
- Maintain backward compatibility
- Gradual migration path

---

## 3. Gap Analysis & Incompatibilities

### 3.1 Critical Gaps

| Component | Current State | Required State | Gap |
|-----------|--------------|----------------|-----|
| **Layer Semantics** | None | Layers 2-9 enforced | ❌ Missing entirely |
| **Device Semantics** | Basic device detection | 104 devices, device mapping | ⚠️ Partial |
| **Clearance Tokens** | None | Token-based access control | ❌ Missing entirely |
| **Upward-Only Flows** | None | Enforced at API level | ❌ Missing entirely |
| **Redis Streams** | Not used | Per-layer/per-device streams | ❌ Missing entirely |
| **SQLite Warm Tier** | Not used | tmpfs-mounted SQLite | ❌ Missing entirely |
| **Event Bus** | Celery only | Redis Streams + schemas | ⚠️ Partial |
| **INT8 Quantization** | Not supported | Required for MLOps | ❌ Missing entirely |
| **Per-Layer Metrics** | None | Layer/device breakdown | ❌ Missing entirely |
| **Clearance Audit** | None | Full audit trail | ❌ Missing entirely |

### 3.2 Data Model Incompatibilities

**Current Memory Model:**
```python
class Memory:
    id: UUID
    user_id: UUID          # User-scoped
    content: str
    embedding: Vector(768) # Fixed 768d
    extra_data: JSONB
    created_at: datetime
    updated_at: datetime
```

**Required Memory Model:**
```python
class Memory:
    id: UUID
    layer_id: int          # 2-9
    device_id: int         # 0-103
    clearance_token: str   # Required
    content: str
    embedding: Vector(2048) # 2048d INT8 quantized
    tags: List[str]
    extra_data: JSONB
    correlation_id: str    # Event correlation
    roe_metadata: JSONB    # Rules of Engagement
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    tier: str              # hot/warm/cold
```

### 3.3 API Incompatibilities

**Current API:**
- User-scoped operations
- No layer/device parameters
- No clearance tokens
- Simple search interface

**Required API:**
- Layer/device-scoped operations
- Clearance token required
- Upward-only enforcement
- Multi-tier search (hot/warm/cold)

### 3.4 Configuration Incompatibilities

**Current Config:**
- Environment variables
- Simple key-value pairs
- No layer/device mapping
- No clearance token config

**Required Config:**
- DSMILSYSTEM standard config mechanism
- Layer/device mapping tables
- Clearance token validation rules
- ROE rule definitions

---

## 4. Refactoring Strategy

### 4.1 Phase 1: Foundation (Data Model & Storage)

1. **Database Schema Migration**
   - Add `layer_id`, `device_id`, `clearance_token` columns
   - Add `correlation_id`, `roe_metadata`, `tier` columns
   - Update embedding dimension to 2048d
   - Create indexes on layer_id, device_id, clearance_token
   - Partition tables by layer_id

2. **SQLite Warm Tier**
   - Create tmpfs-mounted SQLite databases
   - Per-layer SQLite instances
   - Promotion/demotion logic
   - Sync mechanism with PostgreSQL

3. **Redis Streams Integration**
   - Create per-layer streams
   - Create per-device streams
   - Event schema definitions
   - Stream consumers

### 4.2 Phase 2: Core Logic (Layer/Device/Clearance)

1. **Layer & Device Semantics**
   - Layer validation (2-9)
   - Device validation (0-103)
   - Device-to-layer mapping
   - Device capability detection

2. **Clearance Token System**
   - Token validation
   - ROE rule engine
   - Upward-only flow enforcement
   - Access denial logging

3. **Memory Service Refactor**
   - Rewrite `MemoryService` with layer/device/clearance
   - Implement tier promotion/demotion
   - Add event emission
   - Add correlation ID tracking

### 4.3 Phase 3: MLOps Integration

1. **Embedding Service Refactor**
   - INT8 quantization support
   - DSMILSYSTEM MLOps pipeline integration
   - Device-specific model selection
   - Model versioning

2. **Model Registry**
   - Versioned model storage
   - A/B testing support
   - Performance monitoring
   - Rollback capabilities

### 4.4 Phase 4: API & Adapters

1. **Canonical API**
   - Implement new API surface
   - Layer/device/clearance parameters
   - Multi-tier search
   - Event correlation

2. **Legacy Adapters**
   - Thin wrappers for existing APIs
   - Default layer/device mapping
   - Backward compatibility
   - Migration helpers

### 4.5 Phase 5: Observability & Testing

1. **Metrics & Logging**
   - Per-layer metrics
   - Per-device metrics
   - Clearance denial metrics
   - Query latency by tier

2. **Audit Trail**
   - All operations logged
   - Clearance violations tracked
   - Correlation ID tracking
   - Compliance reporting

3. **Test Suite**
   - Unit tests for layer/device/clearance
   - Integration tests for multi-tier
   - Failure mode tests
   - Performance tests

---

## 5. Migration Path

### 5.1 Backward Compatibility

**Legacy API Support:**
- Default layer/device mapping for existing calls
- User ID → default layer/device mapping
- Clearance token optional (defaults to "UNCLASSIFIED")
- Gradual migration with feature flags

**Data Migration:**
- Existing memories get default layer_id=6 (Application layer)
- Existing memories get default device_id=0
- Existing memories get default clearance_token="UNCLASSIFIED"
- One-time migration script

### 5.2 Feature Flags

```python
ENABLE_DSMILSYSTEM_LAYERS = True
ENABLE_CLEARANCE_TOKENS = True
ENABLE_UPWARD_ONLY_FLOWS = True
ENABLE_REDIS_STREAMS = True
ENABLE_SQLITE_WARM_TIER = True
ENABLE_INT8_QUANTIZATION = True
```

### 5.3 Rollout Plan

1. **Week 1-2**: Foundation (Data Model & Storage)
2. **Week 3-4**: Core Logic (Layer/Device/Clearance)
3. **Week 5-6**: MLOps Integration
4. **Week 7-8**: API & Adapters
5. **Week 9-10**: Observability & Testing
6. **Week 11-12**: Migration & Documentation

---

## 6. Risk Assessment

### 6.1 High Risk

- **Breaking Changes**: Existing API consumers may break
  - **Mitigation**: Legacy adapters, feature flags, gradual rollout

- **Data Migration**: Large datasets may take time to migrate
  - **Mitigation**: Incremental migration, rollback plan

- **Performance Impact**: Multi-tier storage may affect latency
  - **Mitigation**: Performance testing, caching strategies

### 6.2 Medium Risk

- **Clearance Token Complexity**: ROE rules may be complex
  - **Mitigation**: Comprehensive testing, clear documentation

- **MLOps Integration**: Model pipeline integration may have issues
  - **Mitigation**: Staged rollout, fallback to current system

### 6.3 Low Risk

- **Observability**: New metrics may overwhelm monitoring
  - **Mitigation**: Metric aggregation, sampling

---

## 7. Success Criteria

✅ All memory operations tagged with layer/device/clearance  
✅ Upward-only flows enforced and audited  
✅ Redis Streams used for all events  
✅ SQLite warm tier operational  
✅ INT8 quantized embeddings deployed  
✅ Per-layer/per-device metrics available  
✅ Legacy APIs continue to work via adapters  
✅ Comprehensive test coverage (>80%)  
✅ Migration script successfully migrates existing data  
✅ Documentation complete and accurate  

---

## 8. Next Steps

1. **Review & Approval**: Get stakeholder sign-off on design
2. **Implementation Plan**: Create detailed task breakdown
3. **Prototype**: Build minimal viable implementation
4. **Testing**: Comprehensive test suite development
5. **Documentation**: API docs, migration guides, runbooks

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** ✅ Design Complete - Ready for Implementation
