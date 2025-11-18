# MEMSHADOW APT-Grade Security Hardening Guide

**Classification:** UNCLASSIFIED
**Version:** 2.1-hardened
**Defense Level:** Advanced Persistent Threat (APT) Grade

---

## Overview

This guide provides comprehensive security hardening for MEMSHADOW deployments to defend against Advanced Persistent Threats (APTs) and sophisticated adversaries.

## Quick Start (Hardened Deployment)

```bash
# 1. Generate secrets
./scripts/generate-secrets.sh

# 2. Deploy with hardened configuration
docker-compose -f docker-compose.hardened.yml up -d

# 3. Enable monitoring
docker-compose -f docker-compose.hardened.yml \
              -f docker-compose.monitoring.yml up -d

# 4. Verify security
./scripts/security-audit.sh
```

---

## Security Features

### 1. Container Hardening

✅ **Read-Only Root Filesystems**
- All containers run with read-only root FS
- Writable data in dedicated volumes only

✅ **Capability Dropping**
- ALL capabilities dropped by default
- Only essential capabilities added back

✅ **Non-Root Users**
- All services run as non-root (UID 1000+)
- No privileged containers

✅ **Security Profiles**
- AppArmor profiles (Linux)
- Seccomp filters
- No new privileges flag

✅ **Resource Limits**
- CPU limits (prevent resource exhaustion)
- Memory limits
- PID limits (prevent fork bombs)

### 2. Network Segmentation

**Internal Network** (172.29.0.0/24)
- No external access
- Database, cache, vector DB only
- Internal service communication

**DMZ Network** (172.30.0.0/24)
- Controlled external access
- WAF and application gateway
- Rate limiting enforced

### 3. Web Application Firewall (WAF)

**ModSecurity + OWASP CRS**
- SQL injection prevention
- XSS prevention
- Path traversal blocking
- Rate limiting (5-100 req/min per endpoint)
- TLS 1.3 only
- Strong cipher suites

### 4. Intrusion Detection (IDS)

**Suricata**
- Real-time packet inspection
- 8-core optimized
- APT-specific rules
- Anomaly detection
- Alert forwarding to SIEM

### 5. API Security

**APT-Grade Middleware**
- IP reputation tracking
- Request signature verification (HMAC)
- Advanced rate limiting (per-IP + per-endpoint)
- SQL injection detection
- XSS detection
- Path traversal detection
- Attack tool detection

### 6. Threat Intelligence

**Integrated Feeds:**
- MISP (Malware Information Sharing Platform)
- OpenCTI (Open Cyber Threat Intelligence)
- AbuseIPDB (IP reputation)
- AlienVault OTX
- VirusTotal
- URLhaus
- ThreatFox

**Auto-Blocking:**
- High-confidence threats blocked automatically
- IP blacklisting (reputation-based)
- Domain/URL filtering

### 7. AI/ML Security (130 TOPS)

**Hardware Acceleration:**
- NVIDIA CUDA (GPU)
- Intel NPU
- Tensor Cores (FP16/TF32)

**ML Capabilities:**
- Vulnerability classification & prioritization
- Anomaly detection (network traffic)
- Threat prediction
- Attack pattern recognition
- Automated security response

---

## Deployment Configurations

### Secrets Management

**Never commit secrets to git!**

```bash
# Generate secure secrets
mkdir -p secrets
openssl rand -base64 32 > secrets/postgres_password.txt
openssl rand -base64 32 > secrets/redis_password.txt
openssl rand -base64 32 > secrets/chroma_token.txt
python -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/secret_key.txt
python -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/jwt_secret_key.txt

# Create database URL
echo "postgresql://memshadow:$(cat secrets/postgres_password.txt)@postgres:5432/memshadow" > secrets/database_url.txt

# Set strict permissions
chmod 600 secrets/*
```

### TLS Certificates

**Production:** Use Let's Encrypt or corporate PKI

**Development/Testing:**
```bash
mkdir -p security/waf/ssl
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout security/waf/ssl/server.key \
  -out security/waf/ssl/server.crt \
  -days 365 \
  -subj "/CN=memshadow.local"
```

### Environment Variables

```bash
# AI/ML Configuration
MEMSHADOW_AI_ENABLED=true
MEMSHADOW_AI_DEVICE=cuda  # or npu, cpu
MEMSHADOW_NPU_ENABLED=true
MEMSHADOW_TENSOR_CORES=true

# Threat Intelligence
MEMSHADOW_THREAT_INTEL_ENABLED=true
MISP_URL=https://misp.yourdomain.com
MISP_KEY=your-misp-api-key
OPENCTI_URL=https://opencti.yourdomain.com
OPENCTI_KEY=your-opencti-api-key
ABUSEIPDB_KEY=your-abuseipdb-key

# Security
MEMSHADOW_APT_DEFENSE=enabled
MEMSHADOW_WAF_ENABLED=true
MEMSHADOW_INTRUSION_DETECTION=enabled
```

---

## Security Validation

### 1. Container Security Scan

```bash
# Scan images for vulnerabilities
trivy image memshadow:2.1-hardened

# Check for misconfigurations
docker-bench-security
```

### 2. Network Security Test

```bash
# Verify network segmentation
docker network inspect memshadow-internal
docker network inspect memshadow-dmz

# Test WAF rules
curl -X POST https://localhost/api/v1/test \
  -d "test' OR '1'='1"  # Should be blocked
```

### 3. API Security Test

```bash
# Test rate limiting
ab -n 1000 -c 10 https://localhost/api/v1/health

# Test authentication
curl -X POST https://localhost/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"test"}'
```

### 4. IDS Verification

```bash
# Check Suricata is running
docker exec memshadow-ids suricata --build-info

# View alerts
docker exec memshadow-ids tail -f /var/log/suricata/fast.log
```

---

## Incident Response

### Emergency Procedures

**If System Compromised:**

1. **Immediate Isolation**
   ```bash
   # Disconnect from network
   docker network disconnect memshadow-dmz memshadow

   # Stop affected services
   docker stop memshadow-app-hardened
   ```

2. **Preserve Evidence**
   ```bash
   # Export logs
   docker logs memshadow > incident-$(date +%Y%m%d-%H%M%S).log

   # Export IDS alerts
   docker cp memshadow-ids:/var/log/suricata ./evidence/
   ```

3. **Activate Incident Response**
   - Notify security team
   - Follow organizational IR procedures
   - Preserve all evidence
   - Document timeline

---

## Monitoring & Alerting

### Critical Alerts

- **Critical vulnerabilities detected** → Immediate notification
- **APT indicators detected** → Emergency response
- **Anomaly score > 0.9** → Block + Alert
- **Failed authentication > 5** → Block IP
- **Rate limit exceeded** → Temporary block
- **IDS high-severity alert** → Investigate

### Log Aggregation

```bash
# Forward logs to SIEM
# Configure in docker-compose.hardened.yml logging section
```

---

## Compliance

### CIS Docker Benchmark

This configuration aligns with CIS Docker Benchmark recommendations:
- ✅ 4.1: Run containers with non-root user
- ✅ 5.12: Drop all capabilities
- ✅ 5.15: Do not share host process namespace
- ✅ 5.16: Do not share host network namespace
- ✅ 5.25: Restrict container from acquiring additional privileges
- ✅ 5.26: Check container health at runtime

### Security Standards

- NIST Cybersecurity Framework
- ISO 27001 controls
- SOC 2 Type II alignment
- GDPR data protection

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs memshadow-app-hardened

# Common issues:
# 1. Secrets not configured → Run generate-secrets.sh
# 2. Permissions issue → Check file ownership (UID 1000)
# 3. Resource limits → Increase Docker resources
```

### WAF Blocking Legitimate Requests

```bash
# Review WAF logs
docker logs memshadow-waf

# Adjust rules in security/waf/modsecurity.conf
# Whitelist specific IPs if needed
```

### High CPU Usage

```bash
# Check resource allocation
docker stats

# AI/ML workloads are CPU/GPU intensive
# Ensure 130 TOPS hardware is available
# Adjust batch sizes if needed
```

---

## Production Checklist

- [ ] Secrets generated and secured (600 permissions)
- [ ] TLS certificates installed (valid, not self-signed)
- [ ] Threat intelligence feeds configured
- [ ] AI/ML models loaded
- [ ] WAF rules tested
- [ ] IDS rules updated
- [ ] Network segmentation verified
- [ ] Resource limits configured
- [ ] Monitoring stack deployed
- [ ] Alert notifications configured
- [ ] Backup procedures tested
- [ ] Incident response plan documented
- [ ] Team trained on emergency procedures

---

**Classification:** UNCLASSIFIED
**Defense Level:** APT-Grade
**Status:** Production Ready with Maximum Security Hardening

**AUTHORIZATION REQUIRED:** All deployments require written authorization for security testing, defensive research, or educational contexts.
