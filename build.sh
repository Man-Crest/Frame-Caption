#!/bin/bash

# Moondream2 VLM Prerequisites Check Script
# This script checks prerequisites and creates necessary directories

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please ensure Docker Compose V2 is installed."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check available disk space (at least 10GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    available_space_gb=$((available_space / 1024 / 1024))
    if [ $available_space_gb -lt 10 ]; then
        print_warning "Low disk space: ${available_space_gb}GB available. At least 10GB recommended."
    else
        print_success "Disk space: ${available_space_gb}GB available"
    fi
    
    # Check NVIDIA Docker runtime (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_success "NVIDIA Docker runtime is working"
        else
            print_warning "NVIDIA Docker runtime not working. GPU acceleration may not be available."
        fi
    else
        print_warning "No NVIDIA GPU detected. Running on CPU only."
    fi
    
    print_success "Prerequisites check completed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p logs
    
    print_success "Directories created"
}

# Function to show usage information
show_usage() {
    print_status "Prerequisites check completed!"
    echo ""
    echo "🔧 Next Steps:"
    echo "   - Run './docker_build.sh' to build and start the container"
    echo "   - Or run commands manually:"
    echo "     docker compose build --no-cache --progress=plain"
    echo "     docker compose up -d"
    echo ""
    echo "📝 Manual Commands:"
    echo "   - Build only: docker compose build --no-cache --progress=plain"
    echo "   - Start: docker compose up -d"
    echo "   - View logs: docker compose logs -f"
    echo "   - Stop: docker compose down"
    echo "   - Status: docker compose ps"
}

# Main script logic
main() {
    echo "🌙 Moondream2 VLM Prerequisites Check"
    echo "====================================="
    
    case "${1:-check}" in
        "check")
            check_prerequisites
            create_directories
            show_usage
            ;;
        *)
            echo "Usage: $0 {check}"
            echo ""
            echo "Commands:"
            echo "  check - Check prerequisites and create directories (default)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
