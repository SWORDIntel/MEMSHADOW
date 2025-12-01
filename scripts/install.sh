#!/usr/bin/env bash
#
# MEMSHADOW Interactive Installer v2.0
# ====================================
# Configures API keys, generates secrets, sets up Docker, and runs migrations.
#
# Usage:
#   ./install.sh                    # Interactive mode
#   ./install.sh --quick            # Quick install with defaults
#   ./install.sh --headless         # Non-interactive with env vars
#   ./install.sh --help             # Show help
#
# Environment Variables (for --headless mode):
#   MEMSHADOW_ADMIN_USER, MEMSHADOW_ADMIN_PASS, MEMSHADOW_DB_PASS,
#   MEMSHADOW_REDIS_PASS, MEMSHADOW_EMBEDDING_BACKEND, OPENAI_API_KEY
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Defaults
DEFAULT_ADMIN_USER="admin"
DEFAULT_DB_USER="memshadow"
DEFAULT_DB_NAME="memshadow"
DEFAULT_EMBEDDING_BACKEND="sentence-transformers"
DEFAULT_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
DEFAULT_EMBEDDING_DIM="2048"

# Modes
INTERACTIVE=true
QUICK_MODE=false
SKIP_DOCKER=false
SKIP_MIGRATION=false
RUN_AFTER=false

# ============================================================================
# Helpers
# ============================================================================

print_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    __  __ _____ __  __ ____  _   _    _    ____   _____        __
   |  \/  | ____|  \/  / ___|| | | |  / \  |  _ \ / _ \ \      / /
   | |\/| |  _| | |\/| \___ \| |_| | / _ \ | | | | | | \ \ /\ / /
   | |  | | |___| |  | |___) |  _  |/ ___ \| |_| | |_| |\ V  V /
   |_|  |_|_____|_|  |_|____/|_| |_/_/   \_\____/ \___/  \_/\_/

   Advanced Cross-LLM Memory Persistence Platform v2.0
   2048d Embeddings | Production Ready
EOF
    echo -e "${NC}"
}

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}${BOLD}==> $1${NC}"; }

generate_secret() {
    openssl rand -base64 "${1:-32}" 2>/dev/null | tr -dc 'a-zA-Z0-9' | head -c "${1:-32}"
}

generate_password() {
    openssl rand -base64 32 2>/dev/null | tr -dc 'a-zA-Z0-9!@#%' | head -c "${1:-20}"
}

prompt_input() {
    local prompt="$1" default="${2:-}" var="$3" secret="${4:-false}"
    local value

    if [[ "$secret" == "true" ]]; then
        read -s -p "$prompt: " value; echo
    elif [[ -n "$default" ]]; then
        read -p "$prompt [$default]: " value
    else
        read -p "$prompt: " value
    fi

    [[ -z "$value" && -n "$default" ]] && value="$default"
    eval "$var=\"$value\""
}

prompt_yes_no() {
    local prompt="$1" default="${2:-y}"
    local response

    [[ "$default" == "y" ]] && read -p "$prompt [Y/n]: " response || read -p "$prompt [y/N]: " response
    [[ -z "$response" ]] && response="$default"
    [[ "$response" =~ ^[Yy] ]]
}

check_deps() {
    log_step "Checking Dependencies"

    local missing=()
    command -v docker &>/dev/null || missing+=("docker")
    (command -v docker-compose &>/dev/null || docker compose version &>/dev/null) || missing+=("docker-compose")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing: ${missing[*]}"
        echo "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    docker info &>/dev/null || { log_error "Docker not running"; exit 1; }
    log_success "All dependencies OK"
}

# ============================================================================
# Configuration
# ============================================================================

configure_admin() {
    log_step "Admin Account"

    if [[ "$INTERACTIVE" == "true" ]]; then
        prompt_input "Admin username" "$DEFAULT_ADMIN_USER" ADMIN_USER

        if prompt_yes_no "Generate secure password?" "y"; then
            ADMIN_PASS=$(generate_password 16)
            echo -e "  Generated: ${BOLD}$ADMIN_PASS${NC} (save this!)"
        else
            prompt_input "Admin password (min 12 chars)" "" ADMIN_PASS "true"
            while [[ ${#ADMIN_PASS} -lt 12 ]]; do
                log_warn "Password too short (min 12)"
                prompt_input "Admin password" "" ADMIN_PASS "true"
            done
        fi
    else
        ADMIN_USER="${MEMSHADOW_ADMIN_USER:-$DEFAULT_ADMIN_USER}"
        ADMIN_PASS="${MEMSHADOW_ADMIN_PASS:-$(generate_password 16)}"
        log_info "Admin: $ADMIN_USER"
    fi
}

configure_database() {
    log_step "Database Configuration"

    if [[ "$INTERACTIVE" == "true" ]]; then
        prompt_input "PostgreSQL user" "$DEFAULT_DB_USER" DB_USER
        prompt_input "PostgreSQL database" "$DEFAULT_DB_NAME" DB_NAME

        if prompt_yes_no "Generate secure passwords?" "y"; then
            DB_PASS=$(generate_password 24)
            REDIS_PASS=$(generate_password 24)
            log_info "Generated database passwords"
        else
            prompt_input "PostgreSQL password" "" DB_PASS "true"
            prompt_input "Redis password" "" REDIS_PASS "true"
        fi
    else
        DB_USER="${MEMSHADOW_DB_USER:-$DEFAULT_DB_USER}"
        DB_NAME="${MEMSHADOW_DB_NAME:-$DEFAULT_DB_NAME}"
        DB_PASS="${MEMSHADOW_DB_PASS:-$(generate_password 24)}"
        REDIS_PASS="${MEMSHADOW_REDIS_PASS:-$(generate_password 24)}"
    fi
}

configure_embeddings() {
    log_step "Embedding Configuration (2048d)"

    if [[ "$INTERACTIVE" == "true" ]]; then
        echo ""
        echo "  ${BOLD}1) sentence-transformers${NC} (Free, Local) - Recommended"
        echo "  ${BOLD}2) openai${NC} (API, \$0.13/1M tokens)"
        echo ""
        read -p "Select backend [1]: " choice

        case "${choice:-1}" in
            2)
                EMBEDDING_BACKEND="openai"
                EMBEDDING_MODEL="text-embedding-3-large"
                EMBEDDING_USE_PROJECTION="false"

                prompt_input "OpenAI API Key" "" OPENAI_API_KEY "true"
                [[ -z "$OPENAI_API_KEY" ]] && log_warn "No API key - add to .env later"

                echo "  Dimensions: 1) 1536  2) 2048  3) 3072"
                read -p "Select [2]: " dim
                case "${dim:-2}" in
                    1) EMBEDDING_DIM="1536" ;;
                    3) EMBEDDING_DIM="3072" ;;
                    *) EMBEDDING_DIM="2048" ;;
                esac
                ;;
            *)
                EMBEDDING_BACKEND="sentence-transformers"
                OPENAI_API_KEY=""

                echo ""
                echo "  Models:"
                echo "    1) BAAI/bge-large-en-v1.5 (1024d->2048d) - Recommended"
                echo "    2) thenlper/gte-large (1024d->2048d)"
                echo "    3) all-mpnet-base-v2 (768d) - Faster"
                echo "    4) paraphrase-multilingual (768d->2048d) - Multilingual"
                read -p "Select [1]: " model

                case "${model:-1}" in
                    2) EMBEDDING_MODEL="thenlper/gte-large"; EMBEDDING_DIM="2048"; EMBEDDING_USE_PROJECTION="true" ;;
                    3) EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"; EMBEDDING_DIM="768"; EMBEDDING_USE_PROJECTION="false" ;;
                    4) EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"; EMBEDDING_DIM="2048"; EMBEDDING_USE_PROJECTION="true" ;;
                    *) EMBEDDING_MODEL="$DEFAULT_EMBEDDING_MODEL"; EMBEDDING_DIM="$DEFAULT_EMBEDDING_DIM"; EMBEDDING_USE_PROJECTION="true" ;;
                esac
                ;;
        esac
    else
        EMBEDDING_BACKEND="${MEMSHADOW_EMBEDDING_BACKEND:-$DEFAULT_EMBEDDING_BACKEND}"
        EMBEDDING_MODEL="${MEMSHADOW_EMBEDDING_MODEL:-$DEFAULT_EMBEDDING_MODEL}"
        EMBEDDING_DIM="${MEMSHADOW_EMBEDDING_DIM:-$DEFAULT_EMBEDDING_DIM}"
        EMBEDDING_USE_PROJECTION="${MEMSHADOW_EMBEDDING_USE_PROJECTION:-true}"
        OPENAI_API_KEY="${OPENAI_API_KEY:-}"
    fi

    log_info "Backend: $EMBEDDING_BACKEND | Model: $EMBEDDING_MODEL | Dim: ${EMBEDDING_DIM}d"
}

configure_security() {
    log_step "Security Configuration"

    SECRET_KEY=$(generate_secret 64)
    WEB_SECRET_KEY=$(generate_secret 48)
    FIELD_ENCRYPTION_KEY=$(generate_secret 32)

    if [[ "$INTERACTIVE" == "true" ]]; then
        prompt_yes_no "Enable security middleware?" "y" && ENABLE_SECURITY="true" || ENABLE_SECURITY="false"
        prompt_input "CORS origins (comma-separated)" "http://localhost:3000,http://localhost:8080,http://localhost:8000" CORS_ORIGINS
    else
        ENABLE_SECURITY="true"
        CORS_ORIGINS="${MEMSHADOW_CORS_ORIGINS:-http://localhost:3000,http://localhost:8080,http://localhost:8000}"
    fi

    log_success "Secrets generated"
}

configure_features() {
    log_step "Feature Configuration"

    if [[ "$INTERACTIVE" == "true" ]]; then
        prompt_yes_no "Enable Federated Learning?" "y" && ENABLE_FEDERATED="true" || ENABLE_FEDERATED="false"
        prompt_yes_no "Enable Meta-Learning (MAML)?" "y" && ENABLE_META="true" || ENABLE_META="false"
        prompt_yes_no "Enable Consciousness Architecture?" "y" && ENABLE_CONSCIOUSNESS="true" || ENABLE_CONSCIOUSNESS="false"
        prompt_yes_no "Enable Self-Modifying? (ADVANCED, risky)" "n" && ENABLE_SELF_MODIFYING="true" || ENABLE_SELF_MODIFYING="false"
        prompt_yes_no "Enable Advanced NLP?" "y" && { USE_ADVANCED_NLP="true"; NLP_QUERY_EXPANSION="true"; } || { USE_ADVANCED_NLP="false"; NLP_QUERY_EXPANSION="false"; }
        echo ""
        echo "  ${BOLD}Memory Operation Mode:${NC}"
        echo "  1) local (Full enrichment, all features)"
        echo "  2) online (Balanced speed/features)"
        echo "  3) lightweight (Minimal processing)"
        read -p "Select [1]: " mem_mode_choice

        case "${mem_mode_choice:-1}" in
            2) MEMORY_OPERATION_MODE="online" ;;
            3) MEMORY_OPERATION_MODE="lightweight" ;;
            *) MEMORY_OPERATION_MODE="local" ;;
        esac
    else
        ENABLE_FEDERATED="${MEMSHADOW_FEDERATED:-true}"
        ENABLE_META="${MEMSHADOW_META:-true}"
        ENABLE_CONSCIOUSNESS="${MEMSHADOW_CONSCIOUSNESS:-true}"
        ENABLE_SELF_MODIFYING="${MEMSHADOW_SELF_MODIFYING:-false}"
        USE_ADVANCED_NLP="${MEMSHADOW_ADVANCED_NLP:-true}"
        NLP_QUERY_EXPANSION="${MEMSHADOW_NLP_EXPANSION:-true}"
        MEMORY_OPERATION_MODE="${MEMSHADOW_MEMORY_OPERATION_MODE:-local}"
    fi

    log_info "Memory Operation Mode: $MEMORY_OPERATION_MODE"
}

# ============================================================================
# File Generation
# ============================================================================

generate_env() {
    log_step "Generating .env"

    cat > "$ENV_FILE" << EOF
# MEMSHADOW Configuration - Generated $(date)
# ============================================

# API
PROJECT_NAME="MEMSHADOW"
VERSION="2.0.0"
API_V1_STR="/api/v1"
SECRET_KEY="$SECRET_KEY"
ACCESS_TOKEN_EXPIRE_MINUTES=11520
BACKEND_CORS_ORIGINS="$CORS_ORIGINS"

# PostgreSQL
POSTGRES_SERVER=postgres
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASS
POSTGRES_DB=$DB_NAME

# Redis
REDIS_PASSWORD=$REDIS_PASS
REDIS_URL="redis://:$REDIS_PASS@redis:6379/0"

# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION="memshadow_memories"

# Celery
CELERY_BROKER_URL="redis://:$REDIS_PASS@redis:6379/1"
CELERY_RESULT_BACKEND="redis://:$REDIS_PASS@redis:6379/1"

# Security
ALGORITHM="HS256"
BCRYPT_ROUNDS=12
FIELD_ENCRYPTION_KEY="$FIELD_ENCRYPTION_KEY"
WEB_SECRET_KEY="$WEB_SECRET_KEY"
WEB_TOKEN_EXPIRY_HOURS=24
WEB_ADMIN_USERNAME="$ADMIN_USER"
WEB_ADMIN_PASSWORD="$ADMIN_PASS"
WEB_CORS_ORIGINS="$CORS_ORIGINS"
ENABLE_SECURITY_MIDDLEWARE="$ENABLE_SECURITY"

# MFA
MFA_ISSUER="MEMSHADOW"
FIDO2_RP_ID="localhost"
FIDO2_RP_NAME="MEMSHADOW"

# SDAP
SDAP_BACKUP_PATH="/var/backups/memshadow"
SDAP_ARCHIVE_SERVER="backup.memshadow.internal"
SDAP_GPG_KEY_ID=""

# Embeddings (2048d)
EMBEDDING_BACKEND="$EMBEDDING_BACKEND"
EMBEDDING_MODEL="$EMBEDDING_MODEL"
EMBEDDING_DIMENSION=$EMBEDDING_DIM
EMBEDDING_USE_PROJECTION=$EMBEDDING_USE_PROJECTION
EMBEDDING_CACHE_TTL=3600
OPENAI_API_KEY="$OPENAI_API_KEY"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"

# Advanced NLP
USE_ADVANCED_NLP=$USE_ADVANCED_NLP
NLP_QUERY_EXPANSION=$NLP_QUERY_EXPANSION
SEMANTIC_SIMILARITY_THRESHOLD=0.7

# Features
ENABLE_FEDERATED_LEARNING=$ENABLE_FEDERATED
ENABLE_META_LEARNING=$ENABLE_META
ENABLE_CONSCIOUSNESS=$ENABLE_CONSCIOUSNESS
ENABLE_SELF_MODIFYING=$ENABLE_SELF_MODIFYING
MEMORY_OPERATION_MODE=$MEMORY_OPERATION_MODE

# Logging
LOG_LEVEL=INFO
EOF

    chmod 600 "$ENV_FILE"
    log_success "Created $ENV_FILE"
}

update_docker_compose() {
    log_step "Updating docker-compose.yml"

    local file="${SCRIPT_DIR}/docker-compose.yml"
    [[ ! -f "$file" ]] && return

    cp "$file" "${file}.backup"

    sed -i "s/memshadow_dev_password/$DB_PASS/g" "$file"
    sed -i "s|EMBEDDING_MODEL: \".*\"|EMBEDDING_MODEL: \"$EMBEDDING_MODEL\"|g" "$file"
    sed -i "s|EMBEDDING_DIMENSION: \".*\"|EMBEDDING_DIMENSION: \"$EMBEDDING_DIM\"|g" "$file"
    sed -i "s|EMBEDDING_BACKEND: \".*\"|EMBEDDING_BACKEND: \"$EMBEDDING_BACKEND\"|g" "$file"
    sed -i "s|EMBEDDING_USE_PROJECTION: \".*\"|EMBEDDING_USE_PROJECTION: \"$EMBEDDING_USE_PROJECTION\"|g" "$file"

    log_success "Updated docker-compose.yml"
}

# ============================================================================
# Docker
# ============================================================================

build_and_run() {
    [[ "$SKIP_DOCKER" == "true" ]] && return

    log_step "Building Containers"
    cd "$SCRIPT_DIR"
    docker compose build 2>/dev/null || docker-compose build
    log_success "Build complete"

    if [[ "$RUN_AFTER" == "true" ]] || { [[ "$INTERACTIVE" == "true" ]] && prompt_yes_no "Start services now?" "y"; }; then
        log_step "Starting Services"
        docker compose up -d 2>/dev/null || docker-compose up -d

        echo -n "Waiting for health..."
        for i in {1..30}; do
            curl -sf http://localhost:8000/health &>/dev/null && { echo ""; log_success "MEMSHADOW is running!"; break; }
            echo -n "."; sleep 2
        done
    fi
}

run_alembic_migration() {
    [[ "$SKIP_MIGRATION" == "true" ]] && return

    log_step "Running Alembic Database Migrations"
    cd "$SCRIPT_DIR"
    # Ensure Docker containers are running before attempting Alembic migration
    docker compose up -d postgres redis chromadb || docker-compose up -d postgres redis chromadb

    log_info "Executing: alembic upgrade head"
    docker compose run --rm memshadow alembic upgrade head 2>/dev/null || \
    docker-compose run --rm memshadow alembic upgrade head

    if [ $? -eq 0 ]; then
        log_success "Alembic migrations completed successfully"
    else
        log_error "Alembic migrations failed. Check database connection and logs."
        exit 1
    fi
}

run_migration() {
    [[ "$SKIP_MIGRATION" == "true" ]] && return

    if [[ "$INTERACTIVE" == "true" ]]; then
        echo ""
        prompt_yes_no "Run embedding migration for existing data?" "n" || return
    fi

    log_step "Running Migration"
    cd "$SCRIPT_DIR"
    docker compose run --rm memshadow python scripts/migrate_embeddings_to_2048d.py --batch-size 100 2>/dev/null || \
    docker-compose run --rm memshadow python scripts/migrate_embeddings_to_2048d.py --batch-size 100
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    echo ""
    echo -e "${GREEN}${BOLD}======================================================${NC}"
    echo -e "${GREEN}${BOLD}  MEMSHADOW Installation Complete!${NC}"
    echo -e "${GREEN}${BOLD}======================================================${NC}"
    echo ""
    echo -e "  ${BOLD}Web Interface:${NC}     http://localhost:8000"
    echo -e "  ${BOLD}API Docs:${NC}          http://localhost:8000/api/docs"
    echo ""
    echo -e "  ${BOLD}Admin:${NC}    ${CYAN}$ADMIN_USER${NC}"
    echo -e "  ${BOLD}Password:${NC} ${CYAN}$ADMIN_PASS${NC}"
    echo ""
    echo -e "  ${BOLD}Embeddings:${NC} ${CYAN}$EMBEDDING_BACKEND${NC} / ${CYAN}${EMBEDDING_DIM}d${NC}"
    echo -e "  ${BOLD}Model:${NC}      ${CYAN}$EMBEDDING_MODEL${NC}"
    echo ""
    echo -e "  ${BOLD}Config:${NC} ${CYAN}$ENV_FILE${NC}"
    echo ""
    echo -e "${YELLOW}  Save your admin password securely!${NC}"
    echo ""
    echo "Commands:"
    echo "  docker compose ps      # Status"
    echo "  docker compose logs    # Logs"
    echo "  docker compose down    # Stop"
    echo ""
}

print_help() {
    cat << 'EOF'
MEMSHADOW Installer v2.0

Usage: ./install.sh [OPTIONS]

Options:
  --quick           Quick install with defaults
  --headless        Non-interactive (uses env vars)
  --skip-docker     Skip Docker build/start
  --skip-migration  Skip embedding migration
  --run             Start services after install
  --help            Show this help

Environment Variables (--headless):
  MEMSHADOW_ADMIN_USER, MEMSHADOW_ADMIN_PASS
  MEMSHADOW_DB_USER, MEMSHADOW_DB_PASS, MEMSHADOW_REDIS_PASS
  MEMSHADOW_EMBEDDING_BACKEND, MEMSHADOW_EMBEDDING_MODEL
  MEMSHADOW_EMBEDDING_DIM, OPENAI_API_KEY

Examples:
  ./install.sh                          # Interactive
  ./install.sh --quick --run            # Quick with auto-start
  MEMSHADOW_ADMIN_PASS=xxx ./install.sh --headless  # CI/CD
EOF
}

# ============================================================================
# Main
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick) QUICK_MODE=true; INTERACTIVE=false ;;
            --headless) INTERACTIVE=false ;;
            --skip-docker) SKIP_DOCKER=true ;;
            --skip-migration) SKIP_MIGRATION=true ;;
            --run) RUN_AFTER=true ;;
            --help|-h) print_help; exit 0 ;;
            *) log_error "Unknown: $1"; print_help; exit 1 ;;
        esac
        shift
    done
}

main() {
    parse_args "$@"
    print_banner
    check_deps
    configure_admin
    configure_database
    configure_embeddings
    configure_security
    configure_features
    generate_env
    update_docker_compose
    build_and_run
    run_alembic_migration
    run_migration
    print_summary
}

main "$@"
