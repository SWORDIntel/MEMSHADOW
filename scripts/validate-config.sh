#!/bin/bash
# MEMSHADOW Configuration Validation Script
# Classification: UNCLASSIFIED
# Purpose: Validate installation configuration for security and completeness

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration paths
readonly MEMSHADOW_CONFIG="/etc/memshadow"
readonly SECRETS_ENV="$MEMSHADOW_CONFIG/secrets.env"
readonly SECRETS_DIR="$MEMSHADOW_CONFIG/secrets"

# Exit code
EXIT_CODE=0

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    EXIT_CODE=1
}

# Banner
cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║              MEMSHADOW CONFIGURATION VALIDATOR                   ║
╚══════════════════════════════════════════════════════════════════╝
EOF
echo ""

# Check 1: Configuration directory exists
log_info "Check 1: Configuration directory..."
if [[ -d "$MEMSHADOW_CONFIG" ]]; then
    log_success "Configuration directory exists: $MEMSHADOW_CONFIG"
else
    log_error "Configuration directory not found: $MEMSHADOW_CONFIG"
    exit 1
fi

# Check 2: Secrets environment file exists
log_info "Check 2: Secrets environment file..."
if [[ -f "$SECRETS_ENV" ]]; then
    log_success "Secrets file exists: $SECRETS_ENV"
else
    log_error "Secrets file not found: $SECRETS_ENV"
    exit 1
fi

# Check 3: File permissions
log_info "Check 3: File permissions..."

# Check secrets.env permissions (should be 600 or 400)
SECRETS_ENV_PERMS=$(stat -c '%a' "$SECRETS_ENV")
if [[ "$SECRETS_ENV_PERMS" == "600" ]] || [[ "$SECRETS_ENV_PERMS" == "400" ]]; then
    log_success "Secrets file permissions: $SECRETS_ENV_PERMS (secure)"
else
    log_error "Secrets file permissions: $SECRETS_ENV_PERMS (should be 600 or 400)"
fi

# Check secrets directory permissions
if [[ -d "$SECRETS_DIR" ]]; then
    SECRETS_DIR_PERMS=$(stat -c '%a' "$SECRETS_DIR")
    if [[ "$SECRETS_DIR_PERMS" == "700" ]]; then
        log_success "Secrets directory permissions: $SECRETS_DIR_PERMS (secure)"
    else
        log_warn "Secrets directory permissions: $SECRETS_DIR_PERMS (recommended: 700)"
    fi
fi

# Check 4: Required environment variables
log_info "Check 4: Required environment variables..."

source "$SECRETS_ENV"

# Critical security secrets
REQUIRED_SECRETS=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "CHROMA_TOKEN"
    "SECRET_KEY"
    "JWT_SECRET_KEY"
)

for var in "${REQUIRED_SECRETS[@]}"; do
    if [[ -n "${!var:-}" ]]; then
        # Check if it's not a default/weak value
        if [[ "${!var}" == "changeme" ]] || \
           [[ "${!var}" == "password" ]] || \
           [[ "${!var}" == "secret" ]] || \
           [[ "${!var}" == "your-super-secret-key" ]] || \
           [[ "${!var}" == "dev-secret-key" ]]; then
            log_error "$var: Using default/insecure value"
        elif [[ ${#!var} -lt 16 ]]; then
            log_error "$var: Too short (minimum 16 characters)"
        else
            log_success "$var: Set (${#!var} characters)"
        fi
    else
        log_error "$var: Not set"
    fi
done

# Check 5: Database configuration
log_info "Check 5: Database configuration..."

if [[ -n "${DATABASE_URL:-}" ]]; then
    log_success "DATABASE_URL: Configured"

    # Validate URL format
    if [[ "$DATABASE_URL" =~ ^postgresql://.*@.*:[0-9]+/.* ]]; then
        log_success "DATABASE_URL: Valid format"
    else
        log_error "DATABASE_URL: Invalid format"
    fi
else
    log_error "DATABASE_URL: Not set"
fi

if [[ -n "${POSTGRES_USER:-}" ]]; then
    log_success "POSTGRES_USER: ${POSTGRES_USER}"
else
    log_warn "POSTGRES_USER: Not set (will use default)"
fi

if [[ -n "${POSTGRES_DB:-}" ]]; then
    log_success "POSTGRES_DB: ${POSTGRES_DB}"
else
    log_warn "POSTGRES_DB: Not set (will use default)"
fi

# Check 6: Application configuration
log_info "Check 6: Application configuration..."

if [[ -n "${APP_NAME:-}" ]]; then
    log_success "APP_NAME: ${APP_NAME}"
else
    log_warn "APP_NAME: Not set"
fi

if [[ -n "${ENVIRONMENT:-}" ]]; then
    if [[ "${ENVIRONMENT}" == "production" ]] || [[ "${ENVIRONMENT}" == "development" ]]; then
        log_success "ENVIRONMENT: ${ENVIRONMENT}"
    else
        log_warn "ENVIRONMENT: ${ENVIRONMENT} (should be 'production' or 'development')"
    fi
else
    log_warn "ENVIRONMENT: Not set"
fi

if [[ -n "${DEBUG:-}" ]]; then
    if [[ "${ENVIRONMENT:-}" == "production" ]] && [[ "${DEBUG}" == "true" ]]; then
        log_error "DEBUG: Enabled in production (security risk)"
    else
        log_success "DEBUG: ${DEBUG}"
    fi
fi

# Check 7: Network configuration
log_info "Check 7: Network configuration..."

if [[ -n "${API_PORT:-}" ]]; then
    if [[ "${API_PORT}" =~ ^[0-9]+$ ]] && [[ "${API_PORT}" -ge 1024 ]] && [[ "${API_PORT}" -le 65535 ]]; then
        log_success "API_PORT: ${API_PORT}"
    else
        log_error "API_PORT: ${API_PORT} (should be 1024-65535)"
    fi
else
    log_warn "API_PORT: Not set (will use default)"
fi

# Check 8: AI/ML configuration
log_info "Check 8: AI/ML configuration..."

if [[ "${MEMSHADOW_AI_ENABLED:-false}" == "true" ]]; then
    log_success "MEMSHADOW_AI_ENABLED: true"

    if [[ -n "${MEMSHADOW_AI_DEVICE:-}" ]]; then
        log_success "MEMSHADOW_AI_DEVICE: ${MEMSHADOW_AI_DEVICE}"

        # Check if CUDA is available when configured
        if [[ "${MEMSHADOW_AI_DEVICE}" == "cuda" ]]; then
            if command -v nvidia-smi &>/dev/null; then
                log_success "NVIDIA GPU: Available"
            else
                log_warn "NVIDIA GPU: Not detected (nvidia-smi not found)"
            fi
        fi
    else
        log_warn "MEMSHADOW_AI_DEVICE: Not set (will use CPU)"
    fi

    if [[ "${MEMSHADOW_NPU_ENABLED:-false}" == "true" ]]; then
        log_success "Intel NPU: Enabled"
    fi
else
    log_info "AI/ML features: Disabled"
fi

# Check 9: Threat Intelligence configuration
log_info "Check 9: Threat Intelligence configuration..."

if [[ "${MEMSHADOW_THREAT_INTEL_ENABLED:-false}" == "true" ]]; then
    log_success "Threat Intelligence: Enabled"

    # Check MISP
    if [[ -n "${MISP_URL:-}" ]]; then
        log_success "MISP_URL: ${MISP_URL}"
        if [[ -n "${MISP_KEY:-}" ]]; then
            log_success "MISP_KEY: Set (${#MISP_KEY} characters)"
        else
            log_warn "MISP_KEY: Not set"
        fi
    fi

    # Check OpenCTI
    if [[ -n "${OPENCTI_URL:-}" ]]; then
        log_success "OPENCTI_URL: ${OPENCTI_URL}"
        if [[ -n "${OPENCTI_KEY:-}" ]]; then
            log_success "OPENCTI_KEY: Set (${#OPENCTI_KEY} characters)"
        else
            log_warn "OPENCTI_KEY: Not set"
        fi
    fi

    # Check AbuseIPDB
    if [[ -n "${ABUSEIPDB_KEY:-}" ]]; then
        log_success "ABUSEIPDB_KEY: Set (${#ABUSEIPDB_KEY} characters)"
    else
        log_warn "ABUSEIPDB_KEY: Not set"
    fi
else
    log_info "Threat Intelligence: Disabled"
fi

# Check 10: Security hardening configuration
log_info "Check 10: Security hardening..."

if [[ "${MEMSHADOW_APT_DEFENSE:-disabled}" == "enabled" ]]; then
    log_success "APT Defense: Enabled"
else
    log_warn "APT Defense: Disabled"
fi

if [[ "${MEMSHADOW_WAF_ENABLED:-false}" == "true" ]]; then
    log_success "WAF (ModSecurity): Enabled"
else
    log_warn "WAF: Disabled"
fi

if [[ "${MEMSHADOW_INTRUSION_DETECTION:-disabled}" == "enabled" ]]; then
    log_success "IDS (Suricata): Enabled"
else
    log_warn "IDS: Disabled"
fi

# Check 11: Monitoring configuration
log_info "Check 11: Monitoring configuration..."

if [[ "${MEMSHADOW_MONITORING_ENABLED:-false}" == "true" ]]; then
    log_success "Monitoring: Enabled"

    if [[ -n "${GRAFANA_ADMIN_PASSWORD:-}" ]]; then
        if [[ "${GRAFANA_ADMIN_PASSWORD}" == "admin" ]]; then
            log_error "Grafana admin password: Using default 'admin' (security risk)"
        elif [[ ${#GRAFANA_ADMIN_PASSWORD} -lt 8 ]]; then
            log_error "Grafana admin password: Too short (minimum 8 characters)"
        else
            log_success "Grafana admin password: Set (${#GRAFANA_ADMIN_PASSWORD} characters)"
        fi
    else
        log_warn "Grafana admin password: Not set"
    fi
else
    log_info "Monitoring: Disabled"
fi

# Check 12: Secret files
log_info "Check 12: Individual secret files..."

if [[ -d "$SECRETS_DIR" ]]; then
    SECRET_FILES=(
        "postgres_password.txt"
        "redis_password.txt"
        "chroma_token.txt"
        "secret_key.txt"
        "jwt_secret_key.txt"
        "database_url.txt"
    )

    for file in "${SECRET_FILES[@]}"; do
        if [[ -f "$SECRETS_DIR/$file" ]]; then
            # Check permissions
            PERMS=$(stat -c '%a' "$SECRETS_DIR/$file")
            if [[ "$PERMS" == "600" ]] || [[ "$PERMS" == "400" ]]; then
                log_success "$file: Exists with secure permissions ($PERMS)"
            else
                log_error "$file: Insecure permissions ($PERMS, should be 600)"
            fi

            # Check content is not empty
            if [[ -s "$SECRETS_DIR/$file" ]]; then
                : # File has content, good
            else
                log_error "$file: File is empty"
            fi
        else
            log_warn "$file: Not found"
        fi
    done
else
    log_warn "Secrets directory not found: $SECRETS_DIR"
fi

# Check 13: Production readiness
log_info "Check 13: Production readiness..."

if [[ "${ENVIRONMENT:-}" == "production" ]]; then
    log_info "Performing production-specific checks..."

    # Check debug mode
    if [[ "${DEBUG:-false}" == "true" ]]; then
        log_error "DEBUG mode enabled in production"
    else
        log_success "DEBUG mode: Disabled"
    fi

    # Check security features
    if [[ "${MEMSHADOW_APT_DEFENSE:-disabled}" != "enabled" ]]; then
        log_error "APT Defense should be enabled in production"
    fi

    if [[ "${MEMSHADOW_WAF_ENABLED:-false}" != "true" ]]; then
        log_error "WAF should be enabled in production"
    fi

    if [[ "${MEMSHADOW_INTRUSION_DETECTION:-disabled}" != "enabled" ]]; then
        log_error "IDS should be enabled in production"
    fi

    # Check monitoring
    if [[ "${MEMSHADOW_MONITORING_ENABLED:-false}" != "true" ]]; then
        log_warn "Monitoring should be enabled in production"
    fi
fi

# Check 14: Docker configuration (if applicable)
log_info "Check 14: Docker requirements..."

if command -v docker &>/dev/null; then
    log_success "Docker: Installed"

    # Check if Docker daemon is running
    if docker info &>/dev/null; then
        log_success "Docker daemon: Running"
    else
        log_error "Docker daemon: Not running"
    fi
else
    log_error "Docker: Not installed"
fi

if command -v docker-compose &>/dev/null; then
    log_success "Docker Compose: Installed"
else
    log_error "Docker Compose: Not installed"
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"

if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "CONFIGURATION VALIDATION PASSED"
    echo ""
    log_info "Your MEMSHADOW configuration is valid and secure."
    log_info "You can proceed with deployment."
else
    log_error "CONFIGURATION VALIDATION FAILED"
    echo ""
    log_error "Please review and fix the errors above before deploying."
    log_info "Critical issues must be resolved for secure operation."
fi

echo "════════════════════════════════════════════════════════════════"
echo ""

exit $EXIT_CODE
