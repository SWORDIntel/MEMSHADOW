#!/bin/bash
# MEMSHADOW Deployment Validation Script
# Classification: UNCLASSIFIED
# Validates deployment health and configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Classification banner
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                              MEMSHADOW v2.1${NC}"
echo -e "${BLUE}                      Deployment Validation Script${NC}"
echo -e "${BLUE}                      Classification: UNCLASSIFIED${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_check() {
    echo -ne "${BLUE}[CHECK]${NC} $1 ... "
}

print_pass() {
    echo -e "${GREEN}PASS${NC}"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_fail() {
    echo -e "${RED}FAIL${NC}"
    if [ -n "$1" ]; then
        echo -e "  ${RED}└─ $1${NC}"
    fi
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_warning() {
    echo -e "${YELLOW}WARNING${NC}"
    if [ -n "$1" ]; then
        echo -e "  ${YELLOW}└─ $1${NC}"
    fi
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Determine deployment type
DEPLOYMENT_TYPE=""
if command -v docker-compose &> /dev/null && docker-compose ps &> /dev/null 2>&1; then
    DEPLOYMENT_TYPE="docker"
    print_info "Detected Docker Compose deployment"
elif command -v kubectl &> /dev/null && kubectl get namespace memshadow &> /dev/null 2>&1; then
    DEPLOYMENT_TYPE="kubernetes"
    print_info "Detected Kubernetes deployment"
else
    print_error "No deployment detected. Please deploy MEMSHADOW first."
    exit 1
fi

echo ""
print_info "Starting validation checks..."
echo ""

# Docker Compose Validation
if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
    print_info "=== Docker Compose Checks ==="
    echo ""

    # Check PostgreSQL
    print_check "PostgreSQL container running"
    if docker-compose ps postgres | grep -q "Up"; then
        print_pass
    else
        print_fail "PostgreSQL container is not running"
    fi

    # Check Redis
    print_check "Redis container running"
    if docker-compose ps redis | grep -q "Up"; then
        print_pass
    else
        print_fail "Redis container is not running"
    fi

    # Check ChromaDB
    print_check "ChromaDB container running"
    if docker-compose ps chromadb | grep -q "Up"; then
        print_pass
    else
        print_fail "ChromaDB container is not running"
    fi

    # Check MEMSHADOW
    print_check "MEMSHADOW container running"
    if docker-compose ps memshadow | grep -q "Up"; then
        print_pass
    else
        print_fail "MEMSHADOW container is not running"
    fi

    # Check health endpoints
    echo ""
    print_info "=== Health Endpoint Checks ==="
    echo ""

    print_check "Main API health endpoint"
    if curl -sf http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        print_pass
    else
        print_fail "Health endpoint not responding"
    fi

    print_check "PostgreSQL connectivity"
    if docker-compose exec -T postgres pg_isready -U memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "PostgreSQL not ready"
    fi

    print_check "Redis connectivity"
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_pass
    else
        print_fail "Redis not responding"
    fi

    print_check "ChromaDB API"
    if curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
        print_pass
    else
        print_fail "ChromaDB API not responding"
    fi

# Kubernetes Validation
elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
    print_info "=== Kubernetes Checks ==="
    echo ""

    # Check namespace
    print_check "MEMSHADOW namespace exists"
    if kubectl get namespace memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "Namespace not found"
    fi

    # Check pods
    print_check "PostgreSQL pod running"
    if kubectl get pods -n memshadow -l app.kubernetes.io/name=postgres --field-selector=status.phase=Running 2>/dev/null | grep -q postgres; then
        print_pass
    else
        print_fail "PostgreSQL pod not running"
    fi

    print_check "Redis pod running"
    if kubectl get pods -n memshadow -l app.kubernetes.io/name=redis --field-selector=status.phase=Running 2>/dev/null | grep -q redis; then
        print_pass
    else
        print_fail "Redis pod not running"
    fi

    print_check "ChromaDB pod running"
    if kubectl get pods -n memshadow -l app.kubernetes.io/name=chromadb --field-selector=status.phase=Running 2>/dev/null | grep -q chromadb; then
        print_pass
    else
        print_fail "ChromaDB pod not running"
    fi

    print_check "MEMSHADOW pod running"
    if kubectl get pods -n memshadow -l app.kubernetes.io/name=memshadow --field-selector=status.phase=Running 2>/dev/null | grep -q memshadow; then
        print_pass
    else
        print_fail "MEMSHADOW pod not running"
    fi

    # Check services
    echo ""
    print_info "=== Service Checks ==="
    echo ""

    print_check "PostgreSQL service exists"
    if kubectl get svc postgres-service -n memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "PostgreSQL service not found"
    fi

    print_check "Redis service exists"
    if kubectl get svc redis-service -n memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "Redis service not found"
    fi

    print_check "ChromaDB service exists"
    if kubectl get svc chromadb-service -n memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "ChromaDB service not found"
    fi

    print_check "MEMSHADOW service exists"
    if kubectl get svc memshadow-service -n memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "MEMSHADOW service not found"
    fi

    # Check PVCs
    echo ""
    print_info "=== Storage Checks ==="
    echo ""

    print_check "PostgreSQL PVC bound"
    if kubectl get pvc postgres-pvc -n memshadow 2>/dev/null | grep -q Bound; then
        print_pass
    else
        print_warning "PostgreSQL PVC not bound (may be pending)"
    fi

    print_check "Redis PVC bound"
    if kubectl get pvc redis-pvc -n memshadow 2>/dev/null | grep -q Bound; then
        print_pass
    else
        print_warning "Redis PVC not bound (may be pending)"
    fi

    print_check "ChromaDB PVC bound"
    if kubectl get pvc chromadb-pvc -n memshadow 2>/dev/null | grep -q Bound; then
        print_pass
    else
        print_warning "ChromaDB PVC not bound (may be pending)"
    fi
fi

# Common validation checks
echo ""
print_info "=== Configuration Checks ==="
echo ""

if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
    print_check "Environment file exists"
    if [ -f "$(pwd)/.env" ]; then
        print_pass
    else
        print_warning "No .env file found, using defaults"
    fi

    print_check "Docker network created"
    if docker network ls | grep -q memshadow; then
        print_pass
    else
        print_warning "MEMSHADOW network not found"
    fi
elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
    print_check "ConfigMap exists"
    if kubectl get configmap memshadow-config -n memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "ConfigMap not found"
    fi

    print_check "Secrets exist"
    if kubectl get secret memshadow-secrets -n memshadow > /dev/null 2>&1; then
        print_pass
    else
        print_fail "Secrets not found"
    fi
fi

# Summary
echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                           VALIDATION SUMMARY${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "Total Checks:   $TOTAL_CHECKS"
echo -e "${GREEN}Passed:         $PASSED_CHECKS${NC}"
echo -e "${RED}Failed:         $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings:       $WARNING_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    print_success "All critical checks passed! MEMSHADOW deployment is healthy."
    echo ""
    print_info "Access points:"
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        print_info "  - Main API:        http://localhost:8000"
        print_info "  - API Docs:        http://localhost:8000/docs"
        print_info "  - C2 Service:      https://localhost:8443"
        print_info "  - TEMPEST:         http://localhost:8080"
    else
        print_info "  Use 'kubectl port-forward' or Ingress to access services"
        print_info "  Example: kubectl port-forward -n memshadow svc/memshadow-service 8000:8000"
    fi
    echo ""
    exit 0
else
    print_error "Validation failed with $FAILED_CHECKS critical errors."
    echo ""
    print_info "Troubleshooting steps:"
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        print_info "  1. Check logs: docker-compose logs"
        print_info "  2. Restart services: docker-compose restart"
        print_info "  3. Rebuild: docker-compose build --no-cache"
    else
        print_info "  1. Check pod logs: kubectl logs -n memshadow -l app.kubernetes.io/name=memshadow"
        print_info "  2. Check pod status: kubectl describe pods -n memshadow"
        print_info "  3. Check events: kubectl get events -n memshadow --sort-by='.lastTimestamp'"
    fi
    echo ""
    exit 1
fi
