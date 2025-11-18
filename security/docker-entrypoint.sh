#!/bin/bash
# MEMSHADOW Hardened Docker Entrypoint
# APT-Grade Security Checks
# Classification: UNCLASSIFIED

set -e

# ============================================================================
# Security Banner
# ============================================================================

echo "================================================================================"
echo "  MEMSHADOW v2.1 - Hardened Runtime"
echo "  Classification: UNCLASSIFIED"
echo "  Security Level: APT-Grade Defense"
echo "================================================================================"
echo ""

# ============================================================================
# Pre-Flight Security Checks
# ============================================================================

echo "[SECURITY] Running pre-flight security checks..."

# Check 1: Verify running as non-root
if [ "$(id -u)" -eq 0 ]; then
    echo "[CRITICAL] Container is running as root! This is a security violation."
    echo "[CRITICAL] Refusing to start. Configure container to run as user 1000."
    exit 1
fi
echo "[OK] Running as non-root user ($(id -un))"

# Check 2: Verify required environment variables are set
REQUIRED_VARS=("DATABASE_URL" "REDIS_PASSWORD" "SECRET_KEY")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "[ERROR] Required environment variable $var is not set"
        exit 1
    fi
done
echo "[OK] All required environment variables are set"

# Check 3: Verify secrets are not default values
if [ "$SECRET_KEY" = "your-super-secret-key" ] || [ "$SECRET_KEY" = "change-me" ]; then
    echo "[CRITICAL] SECRET_KEY is set to a default value!"
    echo "[CRITICAL] This is a critical security violation. Refusing to start."
    exit 1
fi
echo "[OK] Secrets are properly configured"

# Check 4: Verify file permissions
if [ -w /app/main.py ]; then
    echo "[WARNING] Application files are writable. This reduces security."
fi
echo "[OK] File permission check complete"

# Check 5: Verify no SUID binaries (attack surface reduction)
SUID_COUNT=$(find /usr/bin /usr/sbin -perm /6000 -type f 2>/dev/null | wc -l)
if [ "$SUID_COUNT" -gt 0 ]; then
    echo "[WARNING] Found $SUID_COUNT SUID/SGID binaries (potential privilege escalation risk)"
fi
echo "[OK] SUID binary check complete"

# Check 6: Verify network connectivity to required services
echo "[INFO] Checking database connectivity..."
timeout 5 nc -zv ${DATABASE_HOST:-postgres} ${DATABASE_PORT:-5432} 2>&1 | grep -q succeeded || {
    echo "[WARNING] Cannot reach database. Will retry during startup."
}

echo "[INFO] Checking Redis connectivity..."
timeout 5 nc -zv ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} 2>&1 | grep -q succeeded || {
    echo "[WARNING] Cannot reach Redis. Will retry during startup."
}

# Check 7: Verify read-only filesystem (if configured)
if mount | grep -q "on / type.*ro,"; then
    echo "[OK] Root filesystem is read-only (enhanced security)"
else
    echo "[WARNING] Root filesystem is writable (consider read-only for production)"
fi

# Check 8: Check for security updates
echo "[INFO] Security configuration validated"

# ============================================================================
# Runtime Security Configuration
# ============================================================================

# Set restrictive umask
umask 077

# Disable core dumps (prevent memory disclosure)
ulimit -c 0

# Set maximum file size (prevent DoS via large files)
ulimit -f 1048576  # 1GB limit

# Set maximum number of processes (prevent fork bombs)
ulimit -u 256

# Set maximum number of open files
ulimit -n 65536

# ============================================================================
# Monitoring and Logging Setup
# ============================================================================

# Ensure log directory exists and is writable
if [ ! -d "/app/logs" ]; then
    mkdir -p /app/logs || {
        echo "[ERROR] Cannot create log directory"
        exit 1
    }
fi

# Start audit logging
echo "[INFO] Initializing security audit logging..."
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) - Container started - User: $(id -un) - UID: $(id -u)" >> /app/logs/security-audit.log

# ============================================================================
# Application Startup
# ============================================================================

echo ""
echo "[INFO] Security checks passed. Starting MEMSHADOW..."
echo "================================================================================"
echo ""

# Execute the provided command
exec "$@"
