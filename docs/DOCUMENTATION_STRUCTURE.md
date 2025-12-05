# Documentation Structure

**Last Updated:** 2025-01-XX  
**Status:** ✅ Organized

---

## Root Directory

**Critical Files Only:**
- `README.md` - Main project README (stays in root)

**All other documentation:** Moved to `docs/` directory

---

## Documentation Directory Structure

```
docs/
├── README.md                          # Documentation index
├── DOCUMENTATION_STRUCTURE.md         # This file
│
├── getting-started/                   # Quick start guides
│   ├── README.md
│   ├── GETTING_STARTED.md            # Main quick start (5 min)
│   ├── QUICK_START.md                # Alternative quick start
│   └── QUICKSTART.md                 # Comprehensive quick start
│
├── guides/                            # Deployment guides
│   ├── README.md
│   ├── DEPLOYMENT.md                 # Complete deployment guide
│   └── PRODUCTION_DEPLOYMENT.md      # Production deployment
│
├── specs/                             # Architecture specifications
│   ├── MEMSHADOW.md
│   ├── MEMSHADOW_UNIFIED_ARCHITECTURE.md
│   └── WINDOWS_IMPLEMENTATION_ANALYSIS.md
│
├── components/                        # Component documentation
│   ├── README.md
│   ├── swarm.md
│   ├── chimera.md
│   ├── hydra.md
│   ├── janus.md
│   ├── mfaa.md
│   ├── sdap.md
│   └── memshadow_core.md
│
├── archive/                           # Historical documentation
│   ├── CLAUDEMASTER.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── INTEGRATION_SUMMARY.md
│   ├── PHASE3_SUMMARY.md
│   └── SESSION_SUMMARY.md
│
├── DSMILSYSTEM_*.md                  # DSMILSYSTEM integration docs
│   ├── DSMILSYSTEM_DEPLOYMENT.md
│   ├── DSMILSYSTEM_MIGRATION.md
│   ├── DSMILSYSTEM_QUICKSTART.md
│   ├── DSMILSYSTEM_REFACTOR_DESIGN.md
│   └── DSMILSYSTEM_REFACTOR_SUMMARY.md
│
├── Core Documentation                 # Main documentation files
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── OPERATOR_MANUAL.md
│   ├── PRODUCTION_SECURITY.md
│   ├── SECURITY_IMPROVEMENTS_V1.0.md
│   ├── security.md
│   ├── WEB_INTERFACE.md
│   ├── EMBEDDING_UPGRADE_GUIDE.md
│   ├── INTELLIGENCE_ANALYSIS_GUIDE.md
│   ├── MEMLAYER_EVALUATION.md
│   ├── SWARM_README.md
│   └── roadmap.md
│
└── Phase Documentation                # Development phases
    ├── PHASE1.md
    ├── PHASE2.MD
    ├── PHASE3.md
    ├── PHASE4.md
    ├── PHASE7.md
    ├── PHASE_8_ADVANCED_MEMORY.md
    └── SESSION_SUMMARY_PHASE_8.md
```

---

## Organization Principles

### 1. Root Directory
- **Only critical files:** `README.md` stays in root
- **All other docs:** Moved to `docs/` directory

### 2. Getting Started
- **Location:** `docs/getting-started/`
- **Purpose:** Quick start guides for new users
- **Main file:** `GETTING_STARTED.md` (consolidated guide)

### 3. Deployment Guides
- **Location:** `docs/guides/`
- **Purpose:** Complete deployment documentation
- **Files:** General deployment + production deployment

### 4. Component Docs
- **Location:** `docs/components/`
- **Purpose:** Individual component documentation
- **Files:** One file per component

### 5. Archive
- **Location:** `docs/archive/`
- **Purpose:** Historical documentation
- **Files:** Summaries, integration notes, phase summaries

### 6. Specifications
- **Location:** `docs/specs/`
- **Purpose:** Architecture and design specifications
- **Files:** Core architecture documents

---

## Quick Navigation

### For New Users
1. Start: [Getting Started Guide](getting-started/GETTING_STARTED.md)
2. Deploy: [Deployment Guide](guides/DEPLOYMENT.md)
3. Secure: [Production Security](PRODUCTION_SECURITY.md)

### For Developers
1. Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
2. API: [API_REFERENCE.md](API_REFERENCE.md)
3. Components: [components/](components/)

### For Operators
1. Operations: [OPERATOR_MANUAL.md](OPERATOR_MANUAL.md)
2. Deployment: [guides/PRODUCTION_DEPLOYMENT.md](guides/PRODUCTION_DEPLOYMENT.md)
3. Security: [PRODUCTION_SECURITY.md](PRODUCTION_SECURITY.md)

### For DSMILSYSTEM Integration
1. Quick Start: [DSMILSYSTEM_QUICKSTART.md](DSMILSYSTEM_QUICKSTART.md)
2. Deployment: [DSMILSYSTEM_DEPLOYMENT.md](DSMILSYSTEM_DEPLOYMENT.md)
3. Design: [DSMILSYSTEM_REFACTOR_DESIGN.md](DSMILSYSTEM_REFACTOR_DESIGN.md)

---

## File Counts

- **Root:** 1 markdown file (`README.md`)
- **docs/:** 46 markdown files (organized into subdirectories)
- **Total:** 47 markdown files

---

## Maintenance

When adding new documentation:

1. **Quick Start Guides** → `docs/getting-started/`
2. **Deployment Guides** → `docs/guides/`
3. **Component Docs** → `docs/components/`
4. **Architecture Specs** → `docs/specs/`
5. **Historical Docs** → `docs/archive/`
6. **Core Documentation** → `docs/` (root of docs)

**Never add markdown files to project root** (except README.md)

---

**Status:** ✅ Documentation Organized  
**Last Updated:** 2025-01-XX
