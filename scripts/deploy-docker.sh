#!/bin/bash
# MEMSHADOW Docker Deployment Script
# Classification: UNCLASSIFIED
# Automates Docker Compose deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Classification banner
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                              MEMSHADOW v2.1${NC}"
echo -e "${BLUE}                        Docker Deployment Script${NC}"
echo -e "${BLUE}                      Classification: UNCLASSIFIED${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
print_info "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker is installed"

# Check if Docker Compose is installed
print_info "Checking Docker Compose installation..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_success "Docker Compose is installed"

# Check if .env file exists
print_info "Checking environment configuration..."
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    print_warning ".env file not found. Creating from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    print_warning "IMPORTANT: Please edit .env file with your configuration before proceeding!"
    print_warning "At minimum, update these values:"
    print_warning "  - POSTGRES_PASSWORD"
    print_warning "  - REDIS_PASSWORD"
    print_warning "  - CHROMA_TOKEN"
    print_warning "  - SECRET_KEY"
    print_warning "  - JWT_SECRET_KEY"
    echo ""
    read -p "Press Enter after updating .env file, or Ctrl+C to cancel..."
fi
print_success "Environment configuration found"

# Change to project root
cd "$PROJECT_ROOT"

# Parse command line arguments
COMMAND=${1:-"up"}
PROFILE=${2:-""}

case $COMMAND in
    up)
        print_info "Starting MEMSHADOW services..."
        if [ -n "$PROFILE" ]; then
            docker-compose --profile "$PROFILE" up -d
        else
            docker-compose up -d
        fi
        print_success "MEMSHADOW services started"

        # Wait for services to be healthy
        print_info "Waiting for services to be healthy (this may take a minute)..."
        sleep 10

        # Check service health
        print_info "Checking service health..."
        docker-compose ps

        echo ""
        print_success "Deployment complete!"
        echo ""
        print_info "Access points:"
        print_info "  - Main API:        http://localhost:8000"
        print_info "  - API Docs:        http://localhost:8000/docs"
        print_info "  - C2 Service:      https://localhost:8443"
        print_info "  - TEMPEST:         http://localhost:8080"
        print_info "  - PostgreSQL:      localhost:5432"
        print_info "  - Redis:           localhost:6379"
        print_info "  - ChromaDB:        http://localhost:8001"
        echo ""
        print_info "View logs with: docker-compose logs -f"
        ;;

    down)
        print_info "Stopping MEMSHADOW services..."
        docker-compose down
        print_success "MEMSHADOW services stopped"
        ;;

    restart)
        print_info "Restarting MEMSHADOW services..."
        docker-compose restart
        print_success "MEMSHADOW services restarted"
        ;;

    logs)
        print_info "Showing logs (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;

    build)
        print_info "Building MEMSHADOW Docker image..."
        docker-compose build --no-cache
        print_success "Build complete"
        ;;

    clean)
        print_warning "This will remove all containers, volumes, and data. Are you sure? [y/N]"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Cleaning up MEMSHADOW deployment..."
            docker-compose down -v
            print_success "Cleanup complete"
        else
            print_info "Cleanup cancelled"
        fi
        ;;

    status)
        print_info "MEMSHADOW service status:"
        docker-compose ps
        ;;

    init-db)
        print_info "Initializing database..."
        docker-compose exec postgres psql -U memshadow -d memshadow -f /docker-entrypoint-initdb.d/init.sql
        print_success "Database initialized"
        ;;

    shell)
        SERVICE=${2:-memshadow}
        print_info "Opening shell in $SERVICE container..."
        docker-compose exec "$SERVICE" /bin/bash
        ;;

    *)
        echo "Usage: $0 {up|down|restart|logs|build|clean|status|init-db|shell} [options]"
        echo ""
        echo "Commands:"
        echo "  up [profile]    - Start all services (optional profile: production)"
        echo "  down            - Stop all services"
        echo "  restart         - Restart all services"
        echo "  logs            - Show and follow logs"
        echo "  build           - Build Docker images"
        echo "  clean           - Remove all containers and volumes"
        echo "  status          - Show service status"
        echo "  init-db         - Initialize database schema"
        echo "  shell [service] - Open shell in container (default: memshadow)"
        echo ""
        echo "Examples:"
        echo "  $0 up                 # Start in development mode"
        echo "  $0 up production      # Start in production mode with nginx"
        echo "  $0 logs               # Follow logs"
        echo "  $0 shell postgres     # Open PostgreSQL shell"
        exit 1
        ;;
esac

echo ""
print_info "Classification: UNCLASSIFIED"
