#!/bin/bash
# MEMSHADOW Interactive Installation Wizard
# Single entry point for complete system setup
# Classification: UNCLASSIFIED

set -e

# ============================================================================
# Configuration
# ============================================================================

INSTALL_DIR="/opt/memshadow"
SERVICE_NAME="memshadow"
SERVICE_USER="memshadow"
DATA_DIR="/var/lib/memshadow"
LOG_DIR="/var/log/memshadow"
CONFIG_DIR="/etc/memshadow"
SECRETS_FILE="$CONFIG_DIR/secrets.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Banner
# ============================================================================

show_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ███╗   ███╗███████╗███╗   ███╗███████╗██╗  ██╗ █████╗ ██████╗  ██████╗║
║   ████╗ ████║██╔════╝████╗ ████║██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗
║   ██╔████╔██║█████╗  ██╔████╔██║███████╗███████║███████║██║  ██║██║   ██║
║   ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║╚════██║██╔══██║██╔══██║██║  ██║██║   ██║
║   ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████║██║  ██║██║  ██║██████╔╝╚██████╔╝
║   ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ║
║                                                                          ║
║              Offensive Security Platform - Installation Wizard          ║
║                        Version 2.1 - APT-Grade                          ║
║                      Classification: UNCLASSIFIED                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo ""
}

# ============================================================================
# Utility Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}${BOLD}═══ $1 ═══${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

prompt_input() {
    local prompt="$1"
    local default="$2"
    local result

    if [ -n "$default" ]; then
        read -p "$(echo -e ${CYAN}${prompt}${NC} [${default}]: )" result
        echo "${result:-$default}"
    else
        read -p "$(echo -e ${CYAN}${prompt}${NC}: )" result
        echo "$result"
    fi
}

prompt_secret() {
    local prompt="$1"
    local result

    read -s -p "$(echo -e ${CYAN}${prompt}${NC}: )" result
    echo ""  # New line after password input
    echo "$result"
}

prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    local result

    if [ "$default" = "y" ]; then
        read -p "$(echo -e ${CYAN}${prompt}${NC} [Y/n]: )" result
        result="${result:-y}"
    else
        read -p "$(echo -e ${CYAN}${prompt}${NC} [y/N]: )" result
        result="${result:-n}"
    fi

    [[ "$result" =~ ^[Yy]$ ]]
}

generate_secret() {
    python3 -c "import secrets; print(secrets.token_urlsafe(32))"
}

# ============================================================================
# Prerequisite Checks
# ============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"

    local missing_deps=()

    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root"
        echo "Please run: sudo ./install.sh"
        exit 1
    fi
    print_success "Running as root"

    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_success "OS: $PRETTY_NAME"
    else
        print_error "Cannot detect OS"
        exit 1
    fi

    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_success "Docker: $DOCKER_VERSION"
    else
        print_error "Docker not found"
        missing_deps+=("docker")
    fi

    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $4}')
        print_success "Docker Compose: $COMPOSE_VERSION"
    elif docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version --short)
        print_success "Docker Compose: $COMPOSE_VERSION"
    else
        print_error "Docker Compose not found"
        missing_deps+=("docker-compose")
    fi

    # Check Python 3
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_success "Python 3: $PYTHON_VERSION"
    else
        print_error "Python 3 not found"
        missing_deps+=("python3")
    fi

    # Check OpenSSL
    if command -v openssl &> /dev/null; then
        OPENSSL_VERSION=$(openssl version | awk '{print $2}')
        print_success "OpenSSL: $OPENSSL_VERSION"
    else
        print_error "OpenSSL not found"
        missing_deps+=("openssl")
    fi

    # Check systemd
    if command -v systemctl &> /dev/null; then
        print_success "systemd detected"
    else
        print_error "systemd not found (required for service installation)"
        missing_deps+=("systemd")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo ""
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo ""
        print_info "Install missing dependencies and try again"
        exit 1
    fi

    echo ""
    print_success "All prerequisites met!"
}

# ============================================================================
# Configuration Wizard
# ============================================================================

collect_configuration() {
    print_header "Configuration Wizard"

    print_info "This wizard will collect necessary configuration and secrets"
    print_info "All sensitive data will be stored securely"
    echo ""

    # Deployment mode
    print_info "Select deployment mode:"
    echo "  1) Development (standard security)"
    echo "  2) Production (APT-grade hardening)"
    read -p "$(echo -e ${CYAN}Choice${NC} [1-2]: )" DEPLOY_MODE
    DEPLOY_MODE=${DEPLOY_MODE:-2}

    if [ "$DEPLOY_MODE" = "2" ]; then
        CONFIG[DEPLOY_MODE]="production"
        CONFIG[COMPOSE_FILE]="docker-compose.hardened.yml"
        print_success "Production mode selected (APT-grade hardening)"
    else
        CONFIG[DEPLOY_MODE]="development"
        CONFIG[COMPOSE_FILE]="docker-compose.yml"
        print_success "Development mode selected"
    fi

    echo ""

    # Database Configuration
    print_header "Database Configuration"

    CONFIG[POSTGRES_USER]=$(prompt_input "PostgreSQL username" "memshadow")

    if prompt_yes_no "Generate secure PostgreSQL password?" "y"; then
        CONFIG[POSTGRES_PASSWORD]=$(generate_secret)
        print_success "Secure password generated"
    else
        CONFIG[POSTGRES_PASSWORD]=$(prompt_secret "PostgreSQL password")
    fi

    CONFIG[POSTGRES_DB]=$(prompt_input "PostgreSQL database name" "memshadow")

    echo ""

    # Redis Configuration
    print_header "Redis Configuration"

    if prompt_yes_no "Generate secure Redis password?" "y"; then
        CONFIG[REDIS_PASSWORD]=$(generate_secret)
        print_success "Secure password generated"
    else
        CONFIG[REDIS_PASSWORD]=$(prompt_secret "Redis password")
    fi

    echo ""

    # ChromaDB Configuration
    print_header "ChromaDB Configuration"

    if prompt_yes_no "Generate secure ChromaDB token?" "y"; then
        CONFIG[CHROMA_TOKEN]=$(generate_secret)
        print_success "Secure token generated"
    else
        CONFIG[CHROMA_TOKEN]=$(prompt_secret "ChromaDB token")
    fi

    echo ""

    # Application Secrets
    print_header "Application Secrets"

    if prompt_yes_no "Generate application secret keys?" "y"; then
        CONFIG[SECRET_KEY]=$(generate_secret)
        CONFIG[JWT_SECRET_KEY]=$(generate_secret)
        print_success "Secure keys generated"
    else
        CONFIG[SECRET_KEY]=$(prompt_secret "Application secret key")
        CONFIG[JWT_SECRET_KEY]=$(prompt_secret "JWT secret key")
    fi

    echo ""

    # AI/ML Configuration
    print_header "AI/ML Configuration (130 TOPS)"

    if prompt_yes_no "Enable AI/ML acceleration?" "y"; then
        CONFIG[MEMSHADOW_AI_ENABLED]="true"

        # Detect GPU
        if command -v nvidia-smi &> /dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            print_success "NVIDIA GPU detected: $GPU_NAME"
            CONFIG[MEMSHADOW_AI_DEVICE]="cuda"
        else
            print_warning "No NVIDIA GPU detected"
            CONFIG[MEMSHADOW_AI_DEVICE]="cpu"
        fi

        if prompt_yes_no "Enable Intel NPU support?" "n"; then
            CONFIG[MEMSHADOW_NPU_ENABLED]="true"
        else
            CONFIG[MEMSHADOW_NPU_ENABLED]="false"
        fi
    else
        CONFIG[MEMSHADOW_AI_ENABLED]="false"
    fi

    echo ""

    # Threat Intelligence
    print_header "Threat Intelligence Integration"

    if prompt_yes_no "Enable threat intelligence feeds?" "y"; then
        CONFIG[MEMSHADOW_THREAT_INTEL_ENABLED]="true"

        # MISP
        if prompt_yes_no "Configure MISP integration?" "n"; then
            CONFIG[MISP_URL]=$(prompt_input "MISP URL" "")
            CONFIG[MISP_KEY]=$(prompt_secret "MISP API key")
            CONFIG[MEMSHADOW_MISP_ENABLED]="true"
        else
            CONFIG[MEMSHADOW_MISP_ENABLED]="false"
        fi

        # OpenCTI
        if prompt_yes_no "Configure OpenCTI integration?" "n"; then
            CONFIG[OPENCTI_URL]=$(prompt_input "OpenCTI URL" "")
            CONFIG[OPENCTI_KEY]=$(prompt_secret "OpenCTI API key")
            CONFIG[MEMSHADOW_OPENCTI_ENABLED]="true"
        else
            CONFIG[MEMSHADOW_OPENCTI_ENABLED]="false"
        fi

        # AbuseIPDB
        if prompt_yes_no "Configure AbuseIPDB?" "n"; then
            CONFIG[ABUSEIPDB_KEY]=$(prompt_secret "AbuseIPDB API key")
        fi
    else
        CONFIG[MEMSHADOW_THREAT_INTEL_ENABLED]="false"
    fi

    echo ""

    # Monitoring
    print_header "Monitoring & Observability"

    if prompt_yes_no "Enable monitoring stack (Prometheus + Grafana)?" "y"; then
        CONFIG[ENABLE_MONITORING]="true"
        CONFIG[GRAFANA_USER]=$(prompt_input "Grafana admin username" "admin")
        CONFIG[GRAFANA_PASSWORD]=$(prompt_secret "Grafana admin password")
    else
        CONFIG[ENABLE_MONITORING]="false"
    fi

    echo ""

    # Network Configuration
    print_header "Network Configuration"

    CONFIG[HTTP_PORT]=$(prompt_input "HTTP port" "80")
    CONFIG[HTTPS_PORT]=$(prompt_input "HTTPS port" "443")
    CONFIG[API_PORT]=$(prompt_input "API port (internal)" "8000")

    echo ""
    print_success "Configuration collected!"
}

# ============================================================================
# Installation
# ============================================================================

declare -A CONFIG

install_memshadow() {
    print_header "Installing MEMSHADOW"

    # Create directories
    print_info "Creating directories..."
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$DATA_DIR"/{postgres,redis,chromadb,data}
    mkdir -p "$LOG_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CONFIG_DIR/secrets"
    print_success "Directories created"

    # Copy files
    print_info "Installing application files..."
    cp -r . "$INSTALL_DIR/"
    chown -R root:root "$INSTALL_DIR"
    print_success "Application files installed"

    # Create service user
    print_info "Creating service user..."
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/false -d "$DATA_DIR" "$SERVICE_USER"
        print_success "User '$SERVICE_USER' created"
    else
        print_warning "User '$SERVICE_USER' already exists"
    fi

    # Set permissions
    print_info "Setting permissions..."
    chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
    chmod 750 "$DATA_DIR"
    chmod 750 "$LOG_DIR"
    chmod 750 "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR/secrets"
    print_success "Permissions set"

    # Generate secrets files
    print_info "Generating secrets files..."
    echo "${CONFIG[POSTGRES_PASSWORD]}" > "$CONFIG_DIR/secrets/postgres_password.txt"
    echo "${CONFIG[REDIS_PASSWORD]}" > "$CONFIG_DIR/secrets/redis_password.txt"
    echo "${CONFIG[CHROMA_TOKEN]}" > "$CONFIG_DIR/secrets/chroma_token.txt"
    echo "${CONFIG[SECRET_KEY]}" > "$CONFIG_DIR/secrets/secret_key.txt"
    echo "${CONFIG[JWT_SECRET_KEY]}" > "$CONFIG_DIR/secrets/jwt_secret_key.txt"
    echo "postgresql://${CONFIG[POSTGRES_USER]}:${CONFIG[POSTGRES_PASSWORD]}@postgres:5432/${CONFIG[POSTGRES_DB]}" > "$CONFIG_DIR/secrets/database_url.txt"

    chmod 600 "$CONFIG_DIR/secrets"/*
    chown root:root "$CONFIG_DIR/secrets"/*
    print_success "Secrets files created"

    # Create environment file
    print_info "Creating environment configuration..."
    cat > "$SECRETS_FILE" << EOF
# MEMSHADOW Configuration
# Classification: UNCLASSIFIED
# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

# Deployment
DEPLOY_MODE=${CONFIG[DEPLOY_MODE]}
ENVIRONMENT=${CONFIG[DEPLOY_MODE]}

# Database
POSTGRES_USER=${CONFIG[POSTGRES_USER]}
POSTGRES_DB=${CONFIG[POSTGRES_DB]}
POSTGRES_PASSWORD=${CONFIG[POSTGRES_PASSWORD]}
DATABASE_URL=postgresql://${CONFIG[POSTGRES_USER]}:${CONFIG[POSTGRES_PASSWORD]}@postgres:5432/${CONFIG[POSTGRES_DB]}

# Redis
REDIS_PASSWORD=${CONFIG[REDIS_PASSWORD]}
REDIS_HOST=redis
REDIS_PORT=6379

# ChromaDB
CHROMA_TOKEN=${CONFIG[CHROMA_TOKEN]}
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# Application
SECRET_KEY=${CONFIG[SECRET_KEY]}
JWT_SECRET_KEY=${CONFIG[JWT_SECRET_KEY]}
PROJECT_NAME=MEMSHADOW
VERSION=2.1

# AI/ML
MEMSHADOW_AI_ENABLED=${CONFIG[MEMSHADOW_AI_ENABLED]:-false}
MEMSHADOW_AI_DEVICE=${CONFIG[MEMSHADOW_AI_DEVICE]:-cpu}
MEMSHADOW_NPU_ENABLED=${CONFIG[MEMSHADOW_NPU_ENABLED]:-false}

# Threat Intelligence
MEMSHADOW_THREAT_INTEL_ENABLED=${CONFIG[MEMSHADOW_THREAT_INTEL_ENABLED]:-false}
MEMSHADOW_MISP_ENABLED=${CONFIG[MEMSHADOW_MISP_ENABLED]:-false}
MEMSHADOW_OPENCTI_ENABLED=${CONFIG[MEMSHADOW_OPENCTI_ENABLED]:-false}
MISP_URL=${CONFIG[MISP_URL]:-}
MISP_KEY=${CONFIG[MISP_KEY]:-}
OPENCTI_URL=${CONFIG[OPENCTI_URL]:-}
OPENCTI_KEY=${CONFIG[OPENCTI_KEY]:-}
ABUSEIPDB_KEY=${CONFIG[ABUSEIPDB_KEY]:-}

# Security
MEMSHADOW_APT_DEFENSE=${CONFIG[DEPLOY_MODE] == "production" && echo "enabled" || echo "disabled"}
MEMSHADOW_WAF_ENABLED=${CONFIG[DEPLOY_MODE] == "production" && echo "true" || echo "false"}

# Monitoring
GRAFANA_USER=${CONFIG[GRAFANA_USER]:-admin}
GRAFANA_PASSWORD=${CONFIG[GRAFANA_PASSWORD]:-}

# Network
HTTP_PORT=${CONFIG[HTTP_PORT]}
HTTPS_PORT=${CONFIG[HTTPS_PORT]}
API_PORT=${CONFIG[API_PORT]}
EOF

    chmod 600 "$SECRETS_FILE"
    print_success "Environment configuration created"

    # Create systemd service
    print_info "Installing systemd service..."
    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=MEMSHADOW Offensive Security Platform
Documentation=https://github.com/SWORDIntel/MEMSHADOW
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$SECRETS_FILE

# Start command
ExecStart=/usr/bin/docker-compose -f ${CONFIG[COMPOSE_FILE]} up -d
ExecStop=/usr/bin/docker-compose -f ${CONFIG[COMPOSE_FILE]} down

# Security
User=root
Group=root
NoNewPrivileges=true
PrivateTmp=true

# Restart policy
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    print_success "Systemd service installed"

    # Install management CLI
    print_info "Installing management CLI..."
    cp "$INSTALL_DIR/scripts/memshadow-ctl.sh" "/usr/local/bin/memshadow"
    chmod +x "/usr/local/bin/memshadow"
    print_success "Management CLI installed: memshadow"

    print_success "Installation complete!"
}

# ============================================================================
# Post-Installation
# ============================================================================

post_install() {
    print_header "Post-Installation Setup"

    # Enable service
    if prompt_yes_no "Enable MEMSHADOW service on boot?" "y"; then
        systemctl enable "$SERVICE_NAME.service"
        print_success "Service enabled"
    fi

    # Start service
    if prompt_yes_no "Start MEMSHADOW now?" "y"; then
        print_info "Starting MEMSHADOW..."
        systemctl start "$SERVICE_NAME.service"

        # Wait for startup
        print_info "Waiting for services to start (30s)..."
        sleep 30

        # Check status
        if systemctl is-active --quiet "$SERVICE_NAME.service"; then
            print_success "MEMSHADOW is running!"

            echo ""
            print_info "Access points:"
            print_info "  Main API:       http://localhost:${CONFIG[API_PORT]}"
            print_info "  API Docs:       http://localhost:${CONFIG[API_PORT]}/docs"
            print_info "  C2 Framework:   https://localhost:8443"
            print_info "  TEMPEST:        http://localhost:8080"

            if [ "${CONFIG[ENABLE_MONITORING]}" = "true" ]; then
                print_info "  Grafana:        http://localhost:3000"
                print_info "  Prometheus:     http://localhost:9090"
            fi
        else
            print_error "Service failed to start"
            print_info "Check logs with: journalctl -u $SERVICE_NAME.service"
        fi
    fi

    echo ""
    print_header "Installation Summary"

    echo -e "${GREEN}MEMSHADOW Successfully Installed!${NC}"
    echo ""
    echo "Installation Directory: $INSTALL_DIR"
    echo "Data Directory:         $DATA_DIR"
    echo "Configuration:          $CONFIG_DIR"
    echo "Logs:                   $LOG_DIR"
    echo ""
    echo "Management Commands:"
    echo "  memshadow start       - Start services"
    echo "  memshadow stop        - Stop services"
    echo "  memshadow restart     - Restart services"
    echo "  memshadow status      - Show status"
    echo "  memshadow logs        - View logs"
    echo "  memshadow update      - Update platform"
    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC} Configuration stored in: $SECRETS_FILE"
    echo -e "${YELLOW}Keep this file secure! It contains sensitive credentials.${NC}"
    echo ""
    echo -e "${CYAN}Classification: UNCLASSIFIED${NC}"
    echo -e "${CYAN}For authorized security testing and defensive research only${NC}"
}

# ============================================================================
# Main
# ============================================================================

main() {
    show_banner

    print_warning "This will install MEMSHADOW as a system service"
    print_warning "Requires root privileges and Docker"
    echo ""

    if ! prompt_yes_no "Continue with installation?" "y"; then
        print_info "Installation cancelled"
        exit 0
    fi

    check_prerequisites
    collect_configuration
    install_memshadow
    post_install

    echo ""
    print_success "Installation wizard complete!"
    echo ""
}

main "$@"
