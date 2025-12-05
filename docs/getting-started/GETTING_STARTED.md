# Getting Started with MEMSHADOW

**Quick start guide for new users**

---

## Quick Start (5 minutes)

### Prerequisites

```bash
# Check Docker
docker --version  # Should be 20.10+

# Check Docker Compose
docker-compose --version  # Should be 2.0+

# Check Python
python3 --version  # Should be 3.8+
```

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Copy configuration
cp config/.env.example .env

# Edit .env with your settings
nano .env

# Start services
docker-compose up -d

# Check status
docker-compose ps

# Access application
# Web Interface: http://localhost:8000
# API Docs: http://localhost:8000/api/docs
```

### Option 2: Manual Installation

```bash
# Clone repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Install dependencies
pip install -r requirements/base.txt

# Setup database
createdb memshadow

# Configure environment
cp config/.env.example .env
# Edit .env with your database credentials

# Run migrations
alembic upgrade head

# Start application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Default Credentials

```
Username: admin
Password: admin
```

**⚠️ SECURITY WARNING: Change these credentials immediately before any production deployment!**

---

## First Steps

### 1. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Expected response: {"status": "healthy", "version": "2.0.0"}
```

### 2. Access API Documentation

Open in browser: http://localhost:8000/api/docs

Interactive Swagger UI with all endpoints.

### 3. Create Your First Memory

```bash
# Login and get token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}' \
  | jq -r '.access_token')

# Store a memory
curl -X POST http://localhost:8000/api/v1/memory/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I love Python programming",
    "extra_data": {"tags": ["programming", "python"]}
  }'

# Search memories
curl -X POST http://localhost:8000/api/v1/memory/retrieve \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming languages",
    "limit": 5
  }'
```

---

## Next Steps

- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions
- **[Architecture Overview](ARCHITECTURE.md)** - System architecture
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Security Guide](PRODUCTION_SECURITY.md)** - Security best practices
- **[DSMILSYSTEM Quick Start](DSMILSYSTEM_QUICKSTART.md)** - DSMILSYSTEM memory subsystem

---

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
sudo systemctl status docker

# Check logs
docker-compose logs memshadow

# Check port availability
sudo lsof -i :8000
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U postgres -d memshadow -c "SELECT 1"
```

### Configuration Errors

```bash
# Validate configuration
./scripts/validate-config.sh

# Check environment variables
env | grep MEMSHADOW
```

---

## Support

- **Documentation:** [docs/](README.md)
- **Issues:** https://github.com/SWORDIntel/MEMSHADOW/issues
- **Security:** security@memshadow.internal

---

**Ready to deploy?** See [Deployment Guide](DEPLOYMENT.md) for production setup.
