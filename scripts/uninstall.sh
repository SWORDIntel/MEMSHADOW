#!/bin/bash
# MEMSHADOW Uninstallation Script
# Classification: UNCLASSIFIED
# Purpose: Complete removal of MEMSHADOW platform

set -euo pipefail

# Configuration
readonly MEMSHADOW_ROOT="/opt/memshadow"
readonly MEMSHADOW_CONFIG="/etc/memshadow"
readonly MEMSHADOW_DATA="/var/lib/memshadow"
readonly MEMSHADOW_LOGS="/var/log/memshadow"
readonly SERVICE_NAME="memshadow.service"
readonly COMPOSE_FILE="docker-compose.hardened.yml"
readonly MONITORING_COMPOSE="docker-compose.monitoring.yml"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
show_banner() {
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                    MEMSHADOW UNINSTALLER                         ║
║              Advanced Offensive Security Platform                ║
╚══════════════════════════════════════════════════════════════════╝
EOF
}

# Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (sudo)"
    exit 1
fi

show_banner
echo ""

# Confirmation
log_warn "═══════════════════════════════════════════════════════════════"
log_warn "WARNING: This will completely remove MEMSHADOW from your system"
log_warn "═══════════════════════════════════════════════════════════════"
echo ""
log_warn "The following will be removed:"
echo "  - All Docker containers and volumes"
echo "  - MEMSHADOW installation ($MEMSHADOW_ROOT)"
echo "  - Application data ($MEMSHADOW_DATA)"
echo "  - Log files ($MEMSHADOW_LOGS)"
echo "  - systemd service ($SERVICE_NAME)"
echo "  - Service user (memshadow)"
echo "  - Management CLI (/usr/local/bin/memshadow)"
echo ""
log_info "Configuration and secrets can be optionally preserved."
echo ""

read -p "Are you absolutely sure you want to continue? (type 'yes' to proceed): " confirm

if [[ "$confirm" != "yes" ]]; then
    log_info "Uninstallation cancelled"
    exit 0
fi

echo ""

# Step 1: Create final backup
log_info "Step 1/10: Creating final backup..."

BACKUP_DIR="/var/backups/memshadow/final-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [[ -d "$MEMSHADOW_ROOT" ]] && command -v docker &>/dev/null; then
    # Backup database if running
    if docker ps --filter "name=memshadow-postgres" --filter "status=running" | grep -q postgres; then
        log_info "Backing up PostgreSQL database..."
        docker exec memshadow-postgres pg_dump -U memshadow memshadow > "$BACKUP_DIR/memshadow-db.sql" 2>/dev/null || {
            log_warn "Database backup failed (service may not be running)"
        }
    fi

    # Backup configuration
    if [[ -d "$MEMSHADOW_CONFIG" ]]; then
        log_info "Backing up configuration..."
        cp -r "$MEMSHADOW_CONFIG" "$BACKUP_DIR/" || log_warn "Config backup failed"
    fi

    # Backup data directory (excluding large files)
    if [[ -d "$MEMSHADOW_DATA" ]]; then
        log_info "Backing up application data..."
        tar -czf "$BACKUP_DIR/memshadow-data.tar.gz" -C "$MEMSHADOW_DATA" . 2>/dev/null || log_warn "Data backup failed"
    fi

    log_success "Backup created at: $BACKUP_DIR"
else
    log_warn "Skipping backup (Docker or MEMSHADOW not found)"
fi

# Step 2: Stop and disable systemd service
log_info "Step 2/10: Stopping and disabling systemd service..."

if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    log_info "Stopping $SERVICE_NAME..."
    systemctl stop "$SERVICE_NAME" || log_warn "Failed to stop service"
fi

if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    log_info "Disabling $SERVICE_NAME..."
    systemctl disable "$SERVICE_NAME" || log_warn "Failed to disable service"
fi

log_success "Service stopped and disabled"

# Step 3: Remove Docker containers
log_info "Step 3/10: Removing Docker containers..."

if [[ -d "$MEMSHADOW_ROOT" ]] && command -v docker-compose &>/dev/null; then
    cd "$MEMSHADOW_ROOT"

    # Stop and remove main stack
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_info "Stopping main services..."
        docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || log_warn "Main stack already down"
    fi

    # Stop and remove monitoring stack
    if [[ -f "$MONITORING_COMPOSE" ]]; then
        log_info "Stopping monitoring services..."
        docker-compose -f "$MONITORING_COMPOSE" down 2>/dev/null || log_warn "Monitoring stack already down"
    fi

    log_success "Docker containers removed"
else
    log_warn "Docker Compose not found, skipping container removal"
fi

# Step 4: Remove Docker volumes
log_info "Step 4/10: Removing Docker volumes..."

if command -v docker &>/dev/null; then
    MEMSHADOW_VOLUMES=$(docker volume ls -q | grep memshadow || true)

    if [[ -n "$MEMSHADOW_VOLUMES" ]]; then
        echo "$MEMSHADOW_VOLUMES" | xargs -r docker volume rm || log_warn "Some volumes could not be removed"
        log_success "Docker volumes removed"
    else
        log_info "No MEMSHADOW volumes found"
    fi
fi

# Step 5: Remove Docker images
log_info "Step 5/10: Checking Docker images..."

read -p "Remove MEMSHADOW Docker images? This will save disk space. (yes/no): " remove_images

if [[ "$remove_images" == "yes" ]]; then
    if command -v docker &>/dev/null; then
        MEMSHADOW_IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep memshadow || true)

        if [[ -n "$MEMSHADOW_IMAGES" ]]; then
            log_info "Removing MEMSHADOW images..."
            echo "$MEMSHADOW_IMAGES" | xargs -r docker rmi -f || log_warn "Some images could not be removed"
            log_success "Docker images removed"
        else
            log_info "No MEMSHADOW images found"
        fi
    fi
else
    log_info "Docker images preserved"
fi

# Step 6: Remove systemd service file
log_info "Step 6/10: Removing systemd service file..."

if [[ -f "/etc/systemd/system/$SERVICE_NAME" ]]; then
    rm -f "/etc/systemd/system/$SERVICE_NAME"
    systemctl daemon-reload
    log_success "Systemd service file removed"
else
    log_info "Service file not found"
fi

# Step 7: Remove installation directory
log_info "Step 7/10: Removing installation directory..."

if [[ -d "$MEMSHADOW_ROOT" ]]; then
    rm -rf "$MEMSHADOW_ROOT"
    log_success "Installation directory removed: $MEMSHADOW_ROOT"
else
    log_info "Installation directory not found"
fi

# Step 8: Remove data directory
log_info "Step 8/10: Removing data directory..."

if [[ -d "$MEMSHADOW_DATA" ]]; then
    rm -rf "$MEMSHADOW_DATA"
    log_success "Data directory removed: $MEMSHADOW_DATA"
else
    log_info "Data directory not found"
fi

# Step 9: Remove log directory
log_info "Step 9/10: Removing log directory..."

if [[ -d "$MEMSHADOW_LOGS" ]]; then
    rm -rf "$MEMSHADOW_LOGS"
    log_success "Log directory removed: $MEMSHADOW_LOGS"
else
    log_info "Log directory not found"
fi

# Step 10: Remove configuration and secrets
log_info "Step 10/10: Handling configuration and secrets..."

if [[ -d "$MEMSHADOW_CONFIG" ]]; then
    echo ""
    log_warn "Configuration directory contains sensitive secrets and API keys:"
    log_warn "  $MEMSHADOW_CONFIG"
    echo ""
    read -p "Remove configuration and ALL secrets? (type 'yes' to remove): " remove_config

    if [[ "$remove_config" == "yes" ]]; then
        rm -rf "$MEMSHADOW_CONFIG"
        log_success "Configuration and secrets removed"
    else
        log_info "Configuration preserved at: $MEMSHADOW_CONFIG"
        log_warn "Remember to manually delete this directory to completely remove all secrets"
    fi
else
    log_info "Configuration directory not found"
fi

# Remove service user
log_info "Removing service user..."

if id memshadow &>/dev/null; then
    userdel memshadow 2>/dev/null || log_warn "Could not remove user (may have active processes)"
    log_success "Service user removed"
else
    log_info "Service user not found"
fi

# Remove management CLI
log_info "Removing management CLI..."

if [[ -f "/usr/local/bin/memshadow" ]]; then
    rm -f /usr/local/bin/memshadow
    log_success "Management CLI removed"
else
    log_info "Management CLI not found"
fi

# Final summary
echo ""
log_success "═══════════════════════════════════════════════════════════════"
log_success "         MEMSHADOW UNINSTALLATION COMPLETED                    "
log_success "═══════════════════════════════════════════════════════════════"
echo ""

log_info "What was removed:"
echo "  ✓ Docker containers and volumes"
echo "  ✓ Installation directory ($MEMSHADOW_ROOT)"
echo "  ✓ Data directory ($MEMSHADOW_DATA)"
echo "  ✓ Log directory ($MEMSHADOW_LOGS)"
echo "  ✓ systemd service"
echo "  ✓ Service user"
echo "  ✓ Management CLI"

if [[ -d "$MEMSHADOW_CONFIG" ]]; then
    echo "  ⚠  Configuration preserved: $MEMSHADOW_CONFIG"
fi

if [[ "$remove_images" != "yes" ]]; then
    echo "  ⚠  Docker images preserved"
fi

echo ""
log_info "Backup location: $BACKUP_DIR"
echo ""

# Cleanup recommendations
if command -v docker &>/dev/null; then
    log_info "Optional cleanup commands:"
    echo "  # Remove unused Docker resources:"
    echo "  docker system prune -a --volumes"
    echo ""
    echo "  # Remove all Docker images (if desired):"
    echo "  docker images | grep memshadow | awk '{print \$3}' | xargs docker rmi -f"
    echo ""
fi

if [[ -d "$MEMSHADOW_CONFIG" ]]; then
    log_warn "To completely remove all secrets:"
    echo "  sudo rm -rf $MEMSHADOW_CONFIG"
    echo ""
fi

log_info "Thank you for using MEMSHADOW."
log_info "To reinstall, run: ./install.sh"
echo ""
log_info "Classification: UNCLASSIFIED"
