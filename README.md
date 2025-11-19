# MEMSHADOW 🧠

**Advanced Cross-LLM Memory Persistence Platform**

[![Production Ready](https://img.shields.io/badge/production-ready-brightgreen)](https://github.com/SWORDIntel/MEMSHADOW)
[![Security Grade](https://img.shields.io/badge/security-A-brightgreen)](docs/PRODUCTION_SECURITY.md)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

MEMSHADOW enables seamless context preservation and knowledge accumulation across different AI providers and custom deployments, addressing the critical limitation of session-based memory in current Large Language Model implementations.

---

## 🚀 Quick Start with Docker

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### 1. Clone and Start
```bash
# Clone the repository
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f memshadow
```

### 2. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

### 3. Default Credentials
```
Username: admin
Password: admin
```
**⚠️ CHANGE IMMEDIATELY IN PRODUCTION!**

### 4. Stop Services
```bash
docker-compose down
```

---

## 📦 What's Included

### Core Services
- **MEMSHADOW API** - FastAPI-based REST API
- **PostgreSQL 15** - Primary data store
- **Redis 7** - Caching and session management
- **ChromaDB** - Vector database for embeddings

### Phase 8 Advanced Features
- ✅ **Federated Learning** - Privacy-preserving distributed memory
- ✅ **Meta-Learning (MAML)** - Few-shot adaptation
- ✅ **Consciousness-Inspired** - Global workspace architecture
- ✅ **Self-Modifying** - Safe code improvement (disabled by default)

### Security Features
- ✅ Bcrypt password hashing
- ✅ JWT authentication
- ✅ Rate limiting (brute force protection)
- ✅ Request validation (SQL injection, XSS prevention)
- ✅ Security headers (HSTS, CSP, X-Frame-Options)
- ✅ Audit logging
- ✅ CORS whitelisting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                         │
│                   (FastAPI + Jinja2)                     │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────────────┐
│              MEMSHADOW Core Services                     │
├──────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Federated  │  │   Meta-    │  │Consciousness│        │
│  │  Learning  │  │  Learning  │  │  System     │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │   Self-    │  │   Memory   │  │   Query    │        │
│  │ Modifying  │  │  Storage   │  │  Engine    │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────────────┐
│              Data Layer                                  │
├──────────────────────────────────────────────────────────┤
│  PostgreSQL 15  │  Redis 7  │  ChromaDB                 │
│  (Primary DB)   │  (Cache)  │  (Vectors)                │
└──────────────────────────────────────────────────────────┘
```

---

## 📚 Documentation

### Getting Started
- [Quick Start Guide](docs/DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [Production Security](docs/PRODUCTION_SECURITY.md) - Security best practices
- [API Documentation](http://localhost:8000/api/docs) - Interactive API docs (when running)

### Architecture & Specs
- [Core Concepts](docs/specs/MEMSHADOW.md) - Fundamental architecture
- [Unified Architecture](docs/specs/MEMSHADOW_UNIFIED_ARCHITECURE.md) - System design
- [Security Improvements](docs/SECURITY_IMPROVEMENTS_V1.0.md) - v1.0 security fixes

### Development
- [Contributing Guide](CONTRIBUTING.md) - How to contribute (coming soon)
- [Development Setup](docs/DEVELOPMENT.md) - Local development guide (coming soon)

---

## 🔧 Configuration

### Environment Variables

Copy the example configuration:
```bash
cp config/.env.example .env
```

**Critical Settings:**
```bash
# JWT Secret (REQUIRED)
WEB_SECRET_KEY="generate_with_openssl_rand_hex_32"

# Admin Credentials (REQUIRED)
WEB_ADMIN_USERNAME="your_admin_username"
WEB_ADMIN_PASSWORD="your_secure_password"

# CORS Origins (REQUIRED for production)
WEB_CORS_ORIGINS="https://yourdomain.com"

# Database Passwords (REQUIRED for production)
POSTGRES_PASSWORD="secure_postgres_password"
REDIS_PASSWORD="secure_redis_password"
```

**Generate Secure Secrets:**
```bash
# Generate JWT secret
openssl rand -hex 32

# Generate database password
openssl rand -base64 32

# Generate bcrypt password hash
python -c "from passlib.hash import bcrypt; import getpass; print(bcrypt.hash(getpass.getpass('Password: ')))"
```

---

## 🐳 Docker Commands

### Development
```bash
# Build and start
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f memshadow

# Execute commands in container
docker-compose exec memshadow bash

# Run tests
docker-compose exec memshadow pytest

# Stop services
docker-compose down

# Remove volumes (CAUTION: deletes data)
docker-compose down -v
```

### Production
```bash
# Use production compose file
docker-compose -f docker-compose.production.yml up -d

# View production logs
docker-compose -f docker-compose.production.yml logs -f

# Check health
docker-compose -f docker-compose.production.yml ps
```

---

## 🧪 Testing

### Run Tests in Docker
```bash
# All tests
docker-compose exec memshadow pytest

# Security tests only
docker-compose exec memshadow pytest tests/security/

# With coverage
docker-compose exec memshadow pytest --cov=app tests/
```

### Run Tests Locally
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run specific test file
pytest tests/security/test_auth_security.py
```

---

## 🔒 Security

### Production Checklist

Before deploying to production, ensure:

- [ ] Changed default admin credentials
- [ ] Generated strong `WEB_SECRET_KEY` (32+ chars)
- [ ] Configured `WEB_CORS_ORIGINS` for your domain
- [ ] Set strong database passwords
- [ ] Enabled HTTPS/TLS
- [ ] Reviewed security settings in `config/.env.production.template`
- [ ] Set up monitoring and alerting
- [ ] Configured automated backups

### Security Features

- **Authentication**: JWT-based with bcrypt password hashing
- **Rate Limiting**: Prevents brute force and DoS attacks
- **Input Validation**: SQL injection and XSS prevention
- **Security Headers**: HSTS, CSP, X-Frame-Options, etc.
- **Audit Logging**: All sensitive operations logged
- **CORS Protection**: Whitelist-based origin validation

See [PRODUCTION_SECURITY.md](docs/PRODUCTION_SECURITY.md) for complete security guide.

---

## 📊 Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Service status
curl http://localhost:8000/api/status

# Rate limiter stats
curl http://localhost:8000/api/stats
```

### Docker Health

```bash
# Check container health
docker-compose ps

# View resource usage
docker stats memshadow_app
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting PRs.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📋 Project Components

### Core Systems
- **MEMSHADOW Core** - Memory persistence platform
- **CHIMERA Protocol** - Isolated memory protocol for deception
- **SDAP** - Secure Databurst Archival Protocol
- **HYDRA Protocol** - Automated red team capabilities
- **MFA/A Framework** - Multi-Factor Authentication & Authorization
- **JANUS Protocol** - Portable sealing mechanisms
- **SWARM Project** - Autonomous agent swarm

### Integrations
- Claude AI integration
- OpenAI integration
- Custom LLM support
- Browser extension

---

## 🛠️ Technology Stack

**Backend:**
- Python 3.11
- FastAPI
- SQLAlchemy
- Alembic (migrations)

**Databases:**
- PostgreSQL 15 (primary data)
- Redis 7 (caching)
- ChromaDB (vector embeddings)

**Infrastructure:**
- Docker & Docker Compose
- Nginx (reverse proxy)
- Prometheus (monitoring)
- Grafana (dashboards)

**Security:**
- Passlib (password hashing)
- PyJWT (authentication)
- Rate limiting middleware
- Input validation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/SWORDIntel/MEMSHADOW/issues)
- **Security**: security@memshadow.internal

---

## 📈 Version History

### v1.0.0 (2025-11-18) - Production Ready
- ✅ All critical security vulnerabilities fixed
- ✅ Complete Docker deployment
- ✅ Production-grade security features
- ✅ Comprehensive documentation
- ✅ 100% production readiness

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

---

**MEMSHADOW** - *Persistent Memory Across the AI Landscape*

Made with 🧠 by the MEMSHADOW Team
