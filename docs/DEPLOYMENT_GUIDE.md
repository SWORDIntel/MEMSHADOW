# MEMSHADOW Production Deployment Guide

**Version:** 1.0
**Target Environment:** Production
**Deployment Time:** ~2 hours

---

## Quick Start (For Experienced Operators)

```bash
# 1. Clone and setup
git clone https://github.com/SWORDIntel/MEMSHADOW.git
cd MEMSHADOW

# 2. Configure environment
cp config/.env.example .env
# Edit .env with secure credentials (see below)

# 3. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt

# 4. Initialize database
python scripts/init_database.py

# 5. Run (behind reverse proxy with TLS)
uvicorn app.web.api:app --host 127.0.0.1 --port 8000
```

---

## Detailed Deployment Steps

### Step 1: Prerequisites

**System Requirements:**
- Python 3.10 or higher
- PostgreSQL 14+ or 15+
- Redis 6.2+
- 4GB RAM minimum (8GB recommended)
- 20GB disk space minimum
- Ubuntu 22.04 LTS or similar

**Install System Dependencies:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install -y python3.10 python3.10-venv python3-pip build-essential

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Install Redis
sudo apt install -y redis-server

# Install Nginx (reverse proxy)
sudo apt install -y nginx

# Install certbot (SSL certificates)
sudo apt install -y certbot python3-certbot-nginx
```

### Step 2: Database Setup

**PostgreSQL:**
```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE memshadow;
CREATE USER memshadow WITH ENCRYPTED PASSWORD 'CHANGE_THIS_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE memshadow TO memshadow;
\q

# Test connection
psql -h localhost -U memshadow -d memshadow
```

**Redis:**
```bash
# Edit Redis config
sudo nano /etc/redis/redis.conf

# Set password (find and uncomment/edit):
requirepass YOUR_SECURE_REDIS_PASSWORD

# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

### Step 3: Application Setup

**Create Service User:**
```bash
sudo useradd -r -m -s /bin/bash memshadow
sudo mkdir -p /opt/memshadow
sudo chown memshadow:memshadow /opt/memshadow
```

**Clone Repository:**
```bash
sudo -u memshadow git clone https://github.com/SWORDIntel/MEMSHADOW.git /opt/memshadow
cd /opt/memshadow
```

**Create Virtual Environment:**
```bash
sudo -u memshadow python3 -m venv /opt/memshadow/venv
sudo -u memshadow /opt/memshadow/venv/bin/pip install --upgrade pip
sudo -u memshadow /opt/memshadow/venv/bin/pip install -r requirements.txt -r requirements-web.txt
```

### Step 4: Environment Configuration

**Generate Secrets:**
```bash
# Generate JWT secret
WEB_SECRET=$(openssl rand -hex 32)

# Generate database password
DB_PASSWORD=$(openssl rand -base64 32)

# Generate Redis password (use one from Step 2)
REDIS_PASSWORD="YOUR_REDIS_PASSWORD_FROM_STEP_2"

# Generate admin password hash
ADMIN_HASH=$(sudo -u memshadow /opt/memshadow/venv/bin/python3 -c "from passlib.hash import bcrypt; print(bcrypt.hash('YOUR_ADMIN_PASSWORD'))")
```

**Create `.env` File:**
```bash
sudo -u memshadow tee /opt/memshadow/.env > /dev/null <<EOF
# Project
PROJECT_NAME="MEMSHADOW"
VERSION="1.0.0"
API_V1_STR="/api/v1"

# Web Interface Security
WEB_SECRET_KEY="$WEB_SECRET"
WEB_TOKEN_EXPIRY_HOURS=24
WEB_ADMIN_USERNAME="admin_$(openssl rand -hex 4)"
WEB_ADMIN_PASSWORD="$ADMIN_HASH"
WEB_CORS_ORIGINS="https://yourdomain.com"

# Security Middleware
ENABLE_SECURITY_MIDDLEWARE=true

# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=memshadow
POSTGRES_PASSWORD="$DB_PASSWORD"
POSTGRES_DB=memshadow

# Redis
REDIS_PASSWORD="$REDIS_PASSWORD"
REDIS_URL="redis://:$REDIS_PASSWORD@localhost:6379/0"

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION="memshadow_memories"

# Celery
CELERY_BROKER_URL="redis://:$REDIS_PASSWORD@localhost:6379/0"
CELERY_RESULT_BACKEND="redis://:$REDIS_PASSWORD@localhost:6379/0"

# Security
ALGORITHM="HS256"
BCRYPT_ROUNDS=12

# Embedding
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION=768
EOF

# Secure permissions
sudo chmod 600 /opt/memshadow/.env
sudo chown memshadow:memshadow /opt/memshadow/.env
```

**Save Your Credentials:**
```bash
echo "Admin Username: admin_$(openssl rand -hex 4)" > ~/memshadow_credentials.txt
echo "Admin Password: YOUR_ADMIN_PASSWORD" >> ~/memshadow_credentials.txt
echo "Database Password: $DB_PASSWORD" >> ~/memshadow_credentials.txt
chmod 600 ~/memshadow_credentials.txt
```

### Step 5: Initialize Database

```bash
sudo -u memshadow /opt/memshadow/venv/bin/python /opt/memshadow/scripts/init_database.py
```

### Step 6: Systemd Service

**Create Service File:**
```bash
sudo tee /etc/systemd/system/memshadow.service > /dev/null <<EOF
[Unit]
Description=MEMSHADOW Web API
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=memshadow
Group=memshadow
WorkingDirectory=/opt/memshadow
EnvironmentFile=/opt/memshadow/.env

ExecStart=/opt/memshadow/venv/bin/uvicorn app.web.api:app \\
    --host 127.0.0.1 \\
    --port 8000 \\
    --workers 4 \\
    --log-level info

Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/memshadow/data /opt/memshadow/logs

# Resource limits
LimitNOFILE=65536
LimitNPROC=512

[Install]
WantedBy=multi-user.target
EOF
```

**Enable and Start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable memshadow
sudo systemctl start memshadow
sudo systemctl status memshadow
```

### Step 7: Nginx Reverse Proxy

**Obtain SSL Certificate:**
```bash
sudo certbot --nginx -d yourdomain.com
```

**Configure Nginx:**
```bash
sudo tee /etc/nginx/sites-available/memshadow > /dev/null <<'EOF'
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;

# Upstream
upstream memshadow_backend {
    server 127.0.0.1:8000 fail_timeout=10s max_fails=3;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration (managed by certbot)
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Logging
    access_log /var/log/nginx/memshadow_access.log;
    error_log /var/log/nginx/memshadow_error.log;

    # Max body size
    client_max_body_size 10M;

    # Authentication endpoint (stricter rate limit)
    location /api/auth/ {
        limit_req zone=auth burst=3 nodelay;

        proxy_pass http://memshadow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://memshadow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files
    location /static/ {
        alias /opt/memshadow/app/web/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Root and UI
    location / {
        proxy_pass http://memshadow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
```

**Enable Site:**
```bash
sudo ln -s /etc/nginx/sites-available/memshadow /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 8: Firewall Configuration

```bash
# Allow SSH (from specific IP only - CHANGE THIS)
sudo ufw allow from YOUR_ADMIN_IP to any port 22

# Allow HTTPS
sudo ufw allow 443/tcp

# Allow HTTP (for certbot renewal)
sudo ufw allow 80/tcp

# Enable firewall
sudo ufw --force enable
```

### Step 9: Monitoring Setup

**Install Prometheus Node Exporter:**
```bash
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz
sudo mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
sudo useradd -rs /bin/false node_exporter

sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl start node_exporter
sudo systemctl enable node_exporter
```

### Step 10: Backup Configuration

**Create Backup Script:**
```bash
sudo tee /opt/memshadow/scripts/backup.sh > /dev/null <<'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/var/backups/memshadow"
DATE=$(date +%Y%m%d_%H%M%S)
GPG_RECIPIENT="admin@yourdomain.com"

mkdir -p "$BACKUP_DIR"

# Database backup
pg_dump -h localhost -U memshadow memshadow | \
    gzip | \
    gpg --encrypt --recipient "$GPG_RECIPIENT" \
    > "$BACKUP_DIR/db_$DATE.sql.gz.gpg"

# Redis backup
redis-cli --no-auth-warning -a "$REDIS_PASSWORD" SAVE
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"
gpg --encrypt --recipient "$GPG_RECIPIENT" "$BACKUP_DIR/redis_$DATE.rdb"
rm "$BACKUP_DIR/redis_$DATE.rdb"

# Configuration backup
tar czf "$BACKUP_DIR/config_$DATE.tar.gz" /opt/memshadow/.env /opt/memshadow/config/
gpg --encrypt --recipient "$GPG_RECIPIENT" "$BACKUP_DIR/config_$DATE.tar.gz"
rm "$BACKUP_DIR/config_$DATE.tar.gz"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -type f -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

sudo chmod +x /opt/memshadow/scripts/backup.sh
```

**Schedule Daily Backups:**
```bash
sudo crontab -e
# Add line:
0 2 * * * /opt/memshadow/scripts/backup.sh >> /var/log/memshadow_backup.log 2>&1
```

### Step 11: Health Checks

**Create Health Check Script:**
```bash
sudo tee /opt/memshadow/scripts/health_check.sh > /dev/null <<'EOF'
#!/bin/bash

# Check API
if ! curl -f -s https://yourdomain.com/api/status > /dev/null; then
    echo "API health check failed"
    systemctl restart memshadow
    exit 1
fi

# Check database
if ! sudo -u postgres psql -c "SELECT 1" memshadow > /dev/null 2>&1; then
    echo "Database health check failed"
    exit 1
fi

# Check Redis
if ! redis-cli --no-auth-warning -a "$REDIS_PASSWORD" ping > /dev/null 2>&1; then
    echo "Redis health check failed"
    exit 1
fi

echo "All health checks passed"
EOF

sudo chmod +x /opt/memshadow/scripts/health_check.sh
```

**Schedule Health Checks:**
```bash
sudo crontab -e
# Add:
*/5 * * * * /opt/memshadow/scripts/health_check.sh >> /var/log/memshadow_health.log 2>&1
```

---

## Post-Deployment Verification

### 1. Test API
```bash
# Get status
curl https://yourdomain.com/api/status

# Test login
curl -X POST https://yourdomain.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"YOUR_ADMIN_USERNAME","password":"YOUR_ADMIN_PASSWORD"}'
```

### 2. Check Logs
```bash
# Application logs
sudo journalctl -u memshadow -f

# Nginx logs
sudo tail -f /var/log/nginx/memshadow_access.log
sudo tail -f /var/log/nginx/memshadow_error.log
```

### 3. Verify Security
```bash
# Check SSL
openssl s_client -connect yourdomain.com:443 -tls1_2

# Check headers
curl -I https://yourdomain.com

# Check rate limiting
for i in {1..10}; do curl https://yourdomain.com/api/status; done
```

---

## Maintenance

### Update Application
```bash
cd /opt/memshadow
sudo -u memshadow git pull
sudo -u memshadow /opt/memshadow/venv/bin/pip install -r requirements.txt -r requirements-web.txt
sudo systemctl restart memshadow
```

### Rotate Secrets
```bash
# Generate new JWT secret
NEW_SECRET=$(openssl rand -hex 32)

# Update .env
sudo -u memshadow sed -i "s/WEB_SECRET_KEY=.*/WEB_SECRET_KEY=\"$NEW_SECRET\"/" /opt/memshadow/.env

# Restart
sudo systemctl restart memshadow
```

### View Metrics
```bash
# System metrics
curl http://localhost:9100/metrics

# Application logs
sudo journalctl -u memshadow --since "1 hour ago"
```

---

## Troubleshooting

### Application Won't Start
```bash
# Check logs
sudo journalctl -u memshadow -n 100

# Check port availability
sudo netstat -tlnp | grep 8000

# Test manually
sudo -u memshadow /opt/memshadow/venv/bin/uvicorn app.web.api:app --host 127.0.0.1 --port 8000
```

### Database Connection Issues
```bash
# Test connection
psql -h localhost -U memshadow -d memshadow

# Check PostgreSQL status
sudo systemctl status postgresql

# Check credentials in .env
sudo -u memshadow cat /opt/memshadow/.env | grep POSTGRES
```

### High Memory Usage
```bash
# Check processes
ps aux | grep memshadow

# Restart application
sudo systemctl restart memshadow

# Check for memory leaks (monitor over time)
sudo journalctl -u memshadow | grep -i "memory\|oom"
```

---

## Rollback Procedure

If deployment fails:

```bash
# Stop service
sudo systemctl stop memshadow

# Restore from backup
gpg --decrypt /var/backups/memshadow/db_YYYYMMDD.sql.gz.gpg | gunzip | psql -U memshadow memshadow

# Revert code
cd /opt/memshadow
sudo -u memshadow git checkout <PREVIOUS_COMMIT>

# Restart
sudo systemctl start memshadow
```

---

## Support

**Documentation:**
- Security Guide: `/docs/PRODUCTION_SECURITY.md`
- API Docs: `https://yourdomain.com/api/docs`

**Contacts:**
- Technical Support: support@memshadow.internal
- Security Issues: security@memshadow.internal
- Emergency: [ON-CALL NUMBER]

---

*Last Updated: 2025-11-18*
*Version: 1.0*
