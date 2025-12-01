#!/bin/bash
# MEMSHADOW Kubernetes Deployment Script
# Classification: UNCLASSIFIED
# Automates Kubernetes deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
K8S_DIR="$PROJECT_ROOT/k8s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Classification banner
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                              MEMSHADOW v2.1${NC}"
echo -e "${BLUE}                     Kubernetes Deployment Script${NC}"
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

# Check if kubectl is installed
print_info "Checking kubectl installation..."
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi
print_success "kubectl is installed"

# Check cluster connectivity
print_info "Checking Kubernetes cluster connectivity..."
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster. Please configure kubectl."
    exit 1
fi
print_success "Connected to Kubernetes cluster"

CLUSTER_NAME=$(kubectl config current-context)
print_info "Current cluster: $CLUSTER_NAME"

# Parse command line arguments
COMMAND=${1:-"deploy"}
OVERLAY=${2:-"development"}

# Generate secrets
generate_secrets() {
    print_info "Generating secrets..."

    # Generate random passwords
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    CHROMA_TOKEN=$(openssl rand -base64 32)
    SECRET_KEY=$(openssl rand -base64 32)
    JWT_SECRET_KEY=$(openssl rand -base64 32)

    # Create secret
    kubectl create secret generic memshadow-secrets \
        --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --from-literal=DATABASE_URL="postgresql://memshadow:${POSTGRES_PASSWORD}@postgres-service:5432/memshadow" \
        --from-literal=REDIS_PASSWORD="$REDIS_PASSWORD" \
        --from-literal=CHROMA_TOKEN="$CHROMA_TOKEN" \
        --from-literal=SECRET_KEY="$SECRET_KEY" \
        --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
        --from-literal=LURECRAFT_SMTP_USER="" \
        --from-literal=LURECRAFT_SMTP_PASSWORD="" \
        --namespace memshadow \
        --dry-run=client -o yaml | kubectl apply -f -

    print_success "Secrets generated and applied"
}

# Deploy function
deploy() {
    print_info "Deploying MEMSHADOW to Kubernetes..."

    # Create namespace
    print_info "Creating namespace..."
    kubectl apply -f "$K8S_DIR/base/namespace.yaml"

    # Apply ConfigMap
    print_info "Applying ConfigMap..."
    kubectl apply -f "$K8S_DIR/base/configmap.yaml"

    # Check if secrets exist, if not generate them
    if ! kubectl get secret memshadow-secrets -n memshadow &> /dev/null; then
        print_warning "Secrets not found, generating..."
        generate_secrets
    else
        print_info "Using existing secrets"
    fi

    # Create PVCs
    print_info "Creating PersistentVolumeClaims..."
    kubectl apply -f "$K8S_DIR/base/pvc.yaml"

    # Deploy PostgreSQL
    print_info "Deploying PostgreSQL..."
    kubectl apply -f "$K8S_DIR/base/postgres-deployment.yaml"

    # Deploy Redis
    print_info "Deploying Redis..."
    kubectl apply -f "$K8S_DIR/base/redis-deployment.yaml"

    # Deploy ChromaDB
    print_info "Deploying ChromaDB..."
    kubectl apply -f "$K8S_DIR/base/chromadb-deployment.yaml"

    # Wait for databases to be ready
    print_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n memshadow --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n memshadow --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=chromadb -n memshadow --timeout=300s

    # Deploy MEMSHADOW application
    print_info "Deploying MEMSHADOW application..."
    kubectl apply -f "$K8S_DIR/base/memshadow-deployment.yaml"

    # Apply Network Policies
    print_info "Applying Network Policies..."
    kubectl apply -f "$K8S_DIR/base/network-policy.yaml"

    # Apply Ingress (optional)
    if [ -f "$K8S_DIR/base/ingress.yaml" ]; then
        print_warning "Ingress configuration found. Apply it? [y/N]"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Applying Ingress..."
            kubectl apply -f "$K8S_DIR/base/ingress.yaml"
        fi
    fi

    # Wait for MEMSHADOW to be ready
    print_info "Waiting for MEMSHADOW to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=memshadow -n memshadow --timeout=300s

    print_success "Deployment complete!"
    echo ""
    print_info "Getting service information..."
    kubectl get pods,svc -n memshadow
}

# Status function
status() {
    print_info "MEMSHADOW Deployment Status:"
    echo ""

    print_info "Pods:"
    kubectl get pods -n memshadow
    echo ""

    print_info "Services:"
    kubectl get svc -n memshadow
    echo ""

    print_info "PersistentVolumeClaims:"
    kubectl get pvc -n memshadow
    echo ""

    print_info "Ingress:"
    kubectl get ingress -n memshadow 2>/dev/null || echo "No ingress configured"
}

# Logs function
logs() {
    SERVICE=${2:-memshadow}
    print_info "Showing logs for $SERVICE..."
    kubectl logs -f -l app.kubernetes.io/name=$SERVICE -n memshadow --tail=100
}

# Delete function
delete() {
    print_warning "This will delete all MEMSHADOW resources. Are you sure? [y/N]"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Deleting MEMSHADOW resources..."
        kubectl delete namespace memshadow
        print_success "MEMSHADOW deleted"
    else
        print_info "Deletion cancelled"
    fi
}

# Shell function
shell() {
    SERVICE=${2:-memshadow}
    print_info "Opening shell in $SERVICE pod..."
    POD=$(kubectl get pod -l app.kubernetes.io/name=$SERVICE -n memshadow -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it "$POD" -n memshadow -- /bin/bash
}

# Port forward function
port_forward() {
    print_info "Setting up port forwarding..."
    print_info "  Main API:     http://localhost:8000"
    print_info "  C2 Service:   https://localhost:8443"
    print_info "  TEMPEST:      http://localhost:8080"
    print_info "Press Ctrl+C to stop port forwarding"
    echo ""

    kubectl port-forward -n memshadow svc/memshadow-service 8000:8000 &
    PF_PID1=$!
    kubectl port-forward -n memshadow svc/memshadow-c2-service 8443:8443 &
    PF_PID2=$!
    kubectl port-forward -n memshadow svc/memshadow-tempest-service 8080:8080 &
    PF_PID3=$!

    trap "kill $PF_PID1 $PF_PID2 $PF_PID3 2>/dev/null" EXIT
    wait
}

# Main command handler
case $COMMAND in
    deploy)
        deploy
        ;;

    status)
        status
        ;;

    logs)
        logs
        ;;

    delete)
        delete
        ;;

    shell)
        shell
        ;;

    secrets)
        generate_secrets
        ;;

    port-forward|pf)
        port_forward
        ;;

    restart)
        print_info "Restarting MEMSHADOW pods..."
        kubectl rollout restart deployment/memshadow -n memshadow
        print_success "Restart initiated"
        ;;

    scale)
        REPLICAS=${2:-2}
        print_info "Scaling MEMSHADOW to $REPLICAS replicas..."
        kubectl scale deployment/memshadow --replicas=$REPLICAS -n memshadow
        print_success "Scaling complete"
        ;;

    *)
        echo "Usage: $0 {deploy|status|logs|delete|shell|secrets|port-forward|restart|scale} [options]"
        echo ""
        echo "Commands:"
        echo "  deploy           - Deploy MEMSHADOW to Kubernetes"
        echo "  status           - Show deployment status"
        echo "  logs [service]   - Show logs (default: memshadow)"
        echo "  delete           - Delete all MEMSHADOW resources"
        echo "  shell [service]  - Open shell in pod (default: memshadow)"
        echo "  secrets          - Generate and apply secrets"
        echo "  port-forward     - Set up port forwarding to services"
        echo "  restart          - Restart MEMSHADOW pods"
        echo "  scale [replicas] - Scale MEMSHADOW deployment"
        echo ""
        echo "Examples:"
        echo "  $0 deploy              # Deploy MEMSHADOW"
        echo "  $0 status              # Check status"
        echo "  $0 logs memshadow      # View logs"
        echo "  $0 port-forward        # Forward ports to localhost"
        echo "  $0 scale 5             # Scale to 5 replicas"
        exit 1
        ;;
esac

echo ""
print_info "Classification: UNCLASSIFIED"
