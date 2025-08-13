#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to build image
build_image() {
    print_status "Building Docker image..."
    docker compose build --no-cache --progress=plain
    if [ $? -eq 0 ]; then
        print_success "Image built successfully!"
    else
        print_error "Failed to build image"
        exit 1
    fi
}

# Function to start container
start_container() {
    print_status "Starting container..."
    docker compose up -d
    if [ $? -eq 0 ]; then
        print_success "Container started successfully!"
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to check container status
check_status() {
    print_status "Checking container status..."
    docker compose ps
}

# Function to stop container
stop_container() {
    print_status "Stopping container..."
    docker compose down
    print_success "Container stopped"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    docker compose down -v
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    print_status "Showing container logs..."
    docker compose logs -f
}

# Function to show status
show_status() {
    print_status "Container status:"
    docker compose ps
}

# Function to check health
check_health() {
    print_status "Checking API health..."
    curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
}

# Function to test installation
test_installation() {
    print_status "Testing Moondream2 installation..."
    docker compose exec moondream2-vlm python test_installation.py
}

# Main script logic
main() {
    echo "🌙 Moondream2 VLM Docker Build and Management"
    echo "============================================="
    
    case "${1:-build}" in
        "build")
            build_image
            start_container
            check_status
            show_usage
            ;;
        "build-only")
            build_image
            print_success "Build completed. Run './docker_build.sh start' to start the container."
            ;;
        "start")
            start_container
            check_status
            show_usage
            ;;
        "stop")
            stop_container
            ;;
        "restart")
            stop_container
            start_container
            check_status
            show_usage
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "health")
            check_health
            ;;
        "test")
            test_installation
            ;;
        *)
            echo "Usage: $0 {build|build-only|start|stop|restart|cleanup|logs|status|health|test}"
            echo ""
            echo "Commands:"
            echo "  build      - Build and start the container (default)"
            echo "  build-only - Build the container only (no start)"
            echo "  start      - Start the container"
            echo "  stop       - Stop the container"
            echo "  restart    - Restart the container"
            echo "  cleanup    - Stop and remove all data"
            echo "  logs       - Show container logs"
            echo "  status     - Show container status"
            echo "  health     - Check API health"
            echo "  test       - Test Moondream2 installation"
            echo ""
            echo "Manual Commands:"
            echo "  docker compose build --no-cache --progress=plain"
            echo "  docker compose up -d"
            echo "  docker compose logs -f"
            echo "  docker compose down"
            exit 1
            ;;
    esac
}

# Function to show usage
show_usage() {
    echo ""
    echo "🌙 Moondream2 VLM is ready!"
    echo "=========================="
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check: http://localhost:8000/health"
    echo "Model Info: http://localhost:8000/model/info"
    echo ""
    echo "Useful commands:"
    echo "  ./docker_build.sh logs     - View logs"
    echo "  ./docker_build.sh health   - Check health"
    echo "  ./docker_build.sh test     - Test installation"
    echo "  ./docker_build.sh stop     - Stop container"
}

# Run main function
main "$@"
