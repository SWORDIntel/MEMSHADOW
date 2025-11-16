#!/bin/bash
# MEMSHADOW Management CLI
# Classification: UNCLASSIFIED
# Purpose: Service management and control interface for MEMSHADOW platform

set -euo pipefail

# Configuration
readonly MEMSHADOW_ROOT="/opt/memshadow"
readonly MEMSHADOW_CONFIG="/etc/memshadow"
readonly MEMSHADOW_DATA="/var/lib/memshadow"
readonly MEMSHADOW_LOGS="/var/log/memshadow"
readonly SERVICE_NAME="memshadow.service"
readonly COMPOSE_FILE="docker-compose.hardened.yml"
readonly MONITORING_COMPOSE="docker-compose.monitoring.yml"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Banner
show_banner() {
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                         MEMSHADOW                                ║
║              Advanced Offensive Security Platform                ║
║                     Management Console                           ║
╚══════════════════════════════════════════════════════════════════╝
EOF
}

# Logging functions
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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This command requires root privileges. Please run with sudo."
        exit 1
    fi
}

# Check if MEMSHADOW is installed
check_installed() {
    if [[ ! -d "$MEMSHADOW_ROOT" ]] || [[ ! -f "/etc/systemd/system/$SERVICE_NAME" ]]; then
        log_error "MEMSHADOW is not installed. Please run the installation script first."
        exit 1
    fi
}

# Start services
cmd_start() {
    check_root
    check_installed

    log_info "Starting MEMSHADOW services..."

    cd "$MEMSHADOW_ROOT"

    # Load environment
    if [[ -f "$MEMSHADOW_CONFIG/secrets.env" ]]; then
        set -a
        source "$MEMSHADOW_CONFIG/secrets.env"
        set +a
    fi

    # Start with systemd
    systemctl start "$SERVICE_NAME"

    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 5

    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up (healthy)"; then
            log_success "MEMSHADOW services started successfully"
            cmd_status
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 2
    done

    log_warn "Services started but may not be fully healthy yet. Check status with: memshadow status"
}

# Stop services
cmd_stop() {
    check_root
    check_installed

    log_info "Stopping MEMSHADOW services..."

    systemctl stop "$SERVICE_NAME"

    log_success "MEMSHADOW services stopped"
}

# Restart services
cmd_restart() {
    check_root
    check_installed

    log_info "Restarting MEMSHADOW services..."

    cmd_stop
    sleep 2
    cmd_start
}

# Show service status
cmd_status() {
    check_installed

    echo ""
    log_info "=== Systemd Service Status ==="
    systemctl status "$SERVICE_NAME" --no-pager || true

    echo ""
    log_info "=== Docker Container Status ==="
    cd "$MEMSHADOW_ROOT"
    docker-compose -f "$COMPOSE_FILE" ps

    echo ""
    log_info "=== Service Health ==="

    # Check main API
    if curl -sf http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        log_success "API: Healthy"
    else
        log_error "API: Unhealthy or not responding"
    fi

    # Check Grafana (if monitoring enabled)
    if docker ps --filter "name=memshadow-grafana" --filter "status=running" | grep -q grafana; then
        if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
            log_success "Grafana: Healthy"
        else
            log_warn "Grafana: Running but not responding"
        fi
    fi

    # Check Prometheus (if monitoring enabled)
    if docker ps --filter "name=memshadow-prometheus" --filter "status=running" | grep -q prometheus; then
        if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
            log_success "Prometheus: Healthy"
        else
            log_warn "Prometheus: Running but not responding"
        fi
    fi

    echo ""
}

# View logs
cmd_logs() {
    check_installed

    local service="${1:-}"
    local follow="${2:-}"

    cd "$MEMSHADOW_ROOT"

    if [[ -z "$service" ]]; then
        log_info "Available services:"
        docker-compose -f "$COMPOSE_FILE" ps --services
        echo ""
        log_info "Usage: memshadow logs <service> [--follow]"
        log_info "Example: memshadow logs memshadow --follow"
        return 0
    fi

    if [[ "$follow" == "--follow" ]] || [[ "$follow" == "-f" ]]; then
        log_info "Following logs for $service (Ctrl+C to exit)..."
        docker-compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker-compose -f "$COMPOSE_FILE" logs --tail=100 "$service"
    fi
}

# Health check
cmd_health() {
    check_installed

    log_info "Running comprehensive health check..."

    cd "$MEMSHADOW_ROOT"

    local exit_code=0

    # Check 1: Docker daemon
    if docker info > /dev/null 2>&1; then
        log_success "Docker daemon: Running"
    else
        log_error "Docker daemon: Not running"
        exit_code=1
    fi

    # Check 2: Containers running
    local expected_containers=("postgres" "redis" "chromadb" "memshadow")
    for container in "${expected_containers[@]}"; do
        if docker ps --filter "name=memshadow-$container" --filter "status=running" | grep -q "$container"; then
            log_success "Container $container: Running"
        else
            log_error "Container $container: Not running"
            exit_code=1
        fi
    done

    # Check 3: API endpoints
    log_info "Checking API endpoints..."

    if curl -sf http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        log_success "Health endpoint: OK"
    else
        log_error "Health endpoint: Failed"
        exit_code=1
    fi

    # Check 4: Database connectivity
    if docker exec memshadow-postgres pg_isready -U memshadow > /dev/null 2>&1; then
        log_success "PostgreSQL: Ready"
    else
        log_error "PostgreSQL: Not ready"
        exit_code=1
    fi

    # Check 5: Redis connectivity
    if docker exec memshadow-redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis: Responding"
    else
        log_error "Redis: Not responding"
        exit_code=1
    fi

    # Check 6: Disk space
    local disk_usage=$(df -h "$MEMSHADOW_DATA" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 80 ]]; then
        log_success "Disk space: ${disk_usage}% used"
    elif [[ $disk_usage -lt 90 ]]; then
        log_warn "Disk space: ${disk_usage}% used (consider cleanup)"
    else
        log_error "Disk space: ${disk_usage}% used (critical)"
        exit_code=1
    fi

    # Check 7: Memory usage
    local memory_total=$(free -m | awk 'NR==2 {print $2}')
    local memory_used=$(free -m | awk 'NR==2 {print $3}')
    local memory_percent=$((memory_used * 100 / memory_total))

    if [[ $memory_percent -lt 80 ]]; then
        log_success "Memory usage: ${memory_percent}%"
    else
        log_warn "Memory usage: ${memory_percent}% (high)"
    fi

    echo ""
    if [[ $exit_code -eq 0 ]]; then
        log_success "All health checks passed"
    else
        log_error "Some health checks failed. Review the output above."
    fi

    return $exit_code
}

# Update platform
cmd_update() {
    check_root
    check_installed

    log_info "Updating MEMSHADOW platform..."

    cd "$MEMSHADOW_ROOT"

    # Backup current configuration
    log_info "Creating backup..."
    local backup_dir="/tmp/memshadow-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    cp -r "$MEMSHADOW_CONFIG" "$backup_dir/"
    log_success "Configuration backed up to: $backup_dir"

    # Pull latest changes
    log_info "Pulling latest updates from git..."
    if [[ -d ".git" ]]; then
        git pull origin main || {
            log_error "Failed to pull updates. Please check your git configuration."
            return 1
        }
    else
        log_warn "Not a git repository. Skipping git pull."
    fi

    # Rebuild images
    log_info "Rebuilding Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    # Stop services
    log_info "Stopping services for update..."
    cmd_stop

    # Start with new images
    log_info "Starting updated services..."
    cmd_start

    log_success "MEMSHADOW updated successfully"
    log_info "Backup location: $backup_dir"
}

# Backup data
cmd_backup() {
    check_root
    check_installed

    local backup_path="${1:-/var/backups/memshadow}"

    log_info "Creating backup..."

    mkdir -p "$backup_path"
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_file="$backup_path/memshadow-backup-$timestamp.tar.gz"

    # Backup database
    log_info "Backing up PostgreSQL database..."
    docker exec memshadow-postgres pg_dump -U memshadow memshadow > "/tmp/memshadow-db-$timestamp.sql"

    # Create comprehensive backup
    log_info "Creating archive..."
    tar -czf "$backup_file" \
        -C / \
        "etc/memshadow" \
        "var/lib/memshadow" \
        "tmp/memshadow-db-$timestamp.sql"

    # Cleanup temp database dump
    rm -f "/tmp/memshadow-db-$timestamp.sql"

    # Set permissions
    chmod 600 "$backup_file"

    log_success "Backup created: $backup_file"
    log_info "Backup size: $(du -h "$backup_file" | cut -f1)"
}

# Restore from backup
cmd_restore() {
    check_root
    check_installed

    local backup_file="${1:-}"

    if [[ -z "$backup_file" ]]; then
        log_error "Usage: memshadow restore <backup-file>"
        return 1
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log_warn "WARNING: This will restore MEMSHADOW to a previous state."
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log_info "Restore cancelled"
        return 0
    fi

    # Stop services
    log_info "Stopping services..."
    cmd_stop

    # Extract backup
    log_info "Restoring from backup..."
    tar -xzf "$backup_file" -C /

    # Restore database
    local db_dump=$(find /tmp -name "memshadow-db-*.sql" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' ')
    if [[ -n "$db_dump" ]]; then
        log_info "Restoring database..."
        docker exec -i memshadow-postgres psql -U memshadow memshadow < "$db_dump"
        rm -f "$db_dump"
    fi

    # Start services
    log_info "Starting services..."
    cmd_start

    log_success "Restore completed successfully"
}

# View configuration
cmd_config() {
    check_installed

    local action="${1:-show}"

    case "$action" in
        show)
            log_info "Current configuration:"
            echo ""

            if [[ -f "$MEMSHADOW_CONFIG/secrets.env" ]]; then
                # Show config with secrets masked
                cat "$MEMSHADOW_CONFIG/secrets.env" | sed -E 's/(PASSWORD|SECRET|KEY|TOKEN)=.*/\1=********/g'
            else
                log_error "Configuration file not found"
                return 1
            fi
            ;;
        edit)
            check_root
            log_warn "Editing configuration. Be careful not to break the format."
            ${EDITOR:-vi} "$MEMSHADOW_CONFIG/secrets.env"

            log_info "Configuration updated. Restart services for changes to take effect:"
            log_info "  memshadow restart"
            ;;
        validate)
            log_info "Validating configuration..."

            # Check required variables
            local required_vars=(
                "POSTGRES_PASSWORD"
                "REDIS_PASSWORD"
                "CHROMA_TOKEN"
                "SECRET_KEY"
                "JWT_SECRET_KEY"
            )

            local exit_code=0
            for var in "${required_vars[@]}"; do
                if grep -q "^${var}=" "$MEMSHADOW_CONFIG/secrets.env" 2>/dev/null; then
                    local value=$(grep "^${var}=" "$MEMSHADOW_CONFIG/secrets.env" | cut -d'=' -f2-)
                    if [[ -n "$value" ]]; then
                        log_success "$var: Set"
                    else
                        log_error "$var: Empty value"
                        exit_code=1
                    fi
                else
                    log_error "$var: Not set"
                    exit_code=1
                fi
            done

            if [[ $exit_code -eq 0 ]]; then
                log_success "Configuration is valid"
            else
                log_error "Configuration validation failed"
            fi

            return $exit_code
            ;;
        *)
            log_error "Unknown action: $action"
            log_info "Usage: memshadow config [show|edit|validate]"
            return 1
            ;;
    esac
}

# Enable monitoring
cmd_enable_monitoring() {
    check_root
    check_installed

    log_info "Enabling monitoring stack (Prometheus + Grafana)..."

    cd "$MEMSHADOW_ROOT"

    if [[ ! -f "$MONITORING_COMPOSE" ]]; then
        log_error "Monitoring compose file not found: $MONITORING_COMPOSE"
        return 1
    fi

    # Start monitoring services
    docker-compose -f "$COMPOSE_FILE" -f "$MONITORING_COMPOSE" up -d

    log_success "Monitoring enabled"
    log_info "Grafana: http://localhost:3000 (admin/admin)"
    log_info "Prometheus: http://localhost:9090"
    log_info "AlertManager: http://localhost:9093"
}

# Disable monitoring
cmd_disable_monitoring() {
    check_root
    check_installed

    log_info "Disabling monitoring stack..."

    cd "$MEMSHADOW_ROOT"

    # Stop monitoring services
    docker-compose -f "$MONITORING_COMPOSE" down

    log_success "Monitoring disabled"
}

# Uninstall MEMSHADOW
cmd_uninstall() {
    check_root

    log_warn "WARNING: This will completely remove MEMSHADOW from your system."
    log_warn "All data, configuration, and logs will be deleted."
    echo ""
    read -p "Are you absolutely sure you want to uninstall? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log_info "Uninstall cancelled"
        return 0
    fi

    log_info "Creating final backup before uninstall..."
    cmd_backup "/var/backups/memshadow/final"

    # Stop and disable service
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Stopping service..."
        systemctl stop "$SERVICE_NAME"
    fi

    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_info "Disabling service..."
        systemctl disable "$SERVICE_NAME"
    fi

    # Remove Docker containers and volumes
    log_info "Removing Docker containers and volumes..."
    cd "$MEMSHADOW_ROOT"
    docker-compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
    docker-compose -f "$MONITORING_COMPOSE" down -v 2>/dev/null || true

    # Remove Docker images
    read -p "Remove Docker images? (yes/no): " remove_images
    if [[ "$remove_images" == "yes" ]]; then
        log_info "Removing Docker images..."
        docker images | grep memshadow | awk '{print $3}' | xargs -r docker rmi -f
    fi

    # Remove systemd service
    log_info "Removing systemd service..."
    rm -f "/etc/systemd/system/$SERVICE_NAME"
    systemctl daemon-reload

    # Remove files
    log_info "Removing MEMSHADOW files..."
    rm -rf "$MEMSHADOW_ROOT"
    rm -rf "$MEMSHADOW_DATA"
    rm -rf "$MEMSHADOW_LOGS"

    # Remove configuration (ask first)
    read -p "Remove configuration and secrets? (yes/no): " remove_config
    if [[ "$remove_config" == "yes" ]]; then
        rm -rf "$MEMSHADOW_CONFIG"
    else
        log_info "Configuration preserved at: $MEMSHADOW_CONFIG"
    fi

    # Remove user
    if id memshadow &>/dev/null; then
        log_info "Removing service user..."
        userdel memshadow 2>/dev/null || true
    fi

    # Remove CLI
    rm -f /usr/local/bin/memshadow

    log_success "MEMSHADOW uninstalled successfully"
    log_info "Final backup available at: /var/backups/memshadow/final"
}

# Show help
cmd_help() {
    show_banner
    echo ""
    cat << EOF
USAGE:
    memshadow                   Launch interactive TUI dashboard (default)
    memshadow <command> [options]

INTERACTIVE DASHBOARD:
    tui | dashboard     Launch interactive TUI dashboard
    (default when no command specified)

COMMANDS:
    start               Start MEMSHADOW services
    stop                Stop MEMSHADOW services
    restart             Restart MEMSHADOW services
    status              Show service status and health
    logs <service>      View logs for a specific service
                        Use --follow or -f to follow logs

    health              Run comprehensive health checks
    config [action]     Manage configuration (show|edit|validate)

    update              Update MEMSHADOW to latest version
    backup [path]       Create backup (default: /var/backups/memshadow)
    restore <file>      Restore from backup

    enable-monitoring   Enable Prometheus + Grafana monitoring
    disable-monitoring  Disable monitoring stack

    uninstall           Completely remove MEMSHADOW
    help                Show this help message

EXAMPLES:
    memshadow                        # Launch interactive dashboard
    memshadow tui                    # Launch interactive dashboard
    memshadow status
    memshadow logs memshadow --follow
    memshadow backup
    memshadow config validate
    memshadow enable-monitoring

ACCESS POINTS:
    Main API:       http://localhost:8000
    API Docs:       http://localhost:8000/docs
    Metrics:        http://localhost:8000/api/v1/metrics
    Grafana:        http://localhost:3000 (admin/admin)
    Prometheus:     http://localhost:9090

DOCUMENTATION:
    Installation:   $MEMSHADOW_ROOT/security/HARDENING_GUIDE.md
    Deployment:     $MEMSHADOW_ROOT/docs/DEPLOYMENT.md
    API Reference:  http://localhost:8000/docs

SUPPORT:
    Issues: https://github.com/SWORDIntel/MEMSHADOW/issues
    Docs:   https://github.com/SWORDIntel/MEMSHADOW/wiki

Classification: UNCLASSIFIED
EOF
}

# Launch interactive TUI dashboard
cmd_tui() {
    check_installed

    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required for the TUI dashboard"
        exit 1
    fi

    # Check if rich is installed
    if ! python3 -c "import rich" &> /dev/null; then
        log_warn "Installing required Python package 'rich'..."
        pip3 install rich || {
            log_error "Failed to install 'rich'. Install manually: pip3 install rich"
            exit 1
        }
    fi

    # Launch TUI
    cd "$MEMSHADOW_ROOT"
    python3 -m app.tui.dashboard
}

# Main command dispatcher
main() {
    local command="${1:-tui}"
    shift || true

    case "$command" in
        tui|dashboard|"")
            cmd_tui "$@"
            ;;
        start)
            cmd_start "$@"
            ;;
        stop)
            cmd_stop "$@"
            ;;
        restart)
            cmd_restart "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        health)
            cmd_health "$@"
            ;;
        config)
            cmd_config "$@"
            ;;
        update)
            cmd_update "$@"
            ;;
        backup)
            cmd_backup "$@"
            ;;
        restore)
            cmd_restore "$@"
            ;;
        enable-monitoring)
            cmd_enable_monitoring "$@"
            ;;
        disable-monitoring)
            cmd_disable_monitoring "$@"
            ;;
        uninstall)
            cmd_uninstall "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
