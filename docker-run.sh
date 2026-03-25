#!/bin/bash
# pixxEngine Docker Helper Script
# Usage: ./docker-run.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

print_help() {
    echo "pixxEngine Docker Helper Script"
    echo ""
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  up          Start containers in detached mode"
    echo "  down        Stop and remove containers"
    echo "  shell       Open a bash shell in the running container"
    echo "  logs        View container logs"
    echo "  restart     Restart containers"
    echo "  status      Show container status"
    echo "  gpu-test    Test GPU access in the container"
    echo "  clean       Remove all project images and containers"
    echo "  help        Show this help message"
    echo ""
}

case "$1" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker compose build
        ;;
    up)
        echo -e "${GREEN}Starting containers...${NC}"
        docker compose up -d
        echo -e "${GREEN}Container started. Use './docker-run.sh shell' to access it.${NC}"
        ;;
    down)
        echo -e "${YELLOW}Stopping containers...${NC}"
        docker compose down
        ;;
    shell)
        echo -e "${GREEN}Opening shell in container...${NC}"
        docker compose exec pixxengine /bin/bash
        ;;
    logs)
        docker compose logs -f
        ;;
    restart)
        echo -e "${YELLOW}Restarting containers...${NC}"
        docker compose restart
        ;;
    status)
        docker compose ps
        ;;
    gpu-test)
        echo -e "${GREEN}Testing GPU access in container...${NC}"
        docker compose exec pixxengine nvidia-smi
        echo ""
        echo -e "${GREEN}Testing PyTorch CUDA...${NC}"
        docker compose exec pixxengine python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
        ;;
    clean)
        echo -e "${RED}Removing all project containers and images...${NC}"
        docker compose down --rmi all -v
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        print_help
        exit 1
        ;;
esac
