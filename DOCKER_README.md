# pixxEngine Docker Setup

This project is now containerized with Docker support for GPU acceleration.

## Prerequisites

1. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
2. **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
3. **NVIDIA Docker Runtime** (for GPU support):
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

## Building the Docker Image

### Option 1: Using Docker Compose (Recommended)
```bash
docker-compose build
```

### Option 2: Using Docker CLI
```bash
docker build -t pixxengine:latest .
```

## Running the Container

### Option 1: Development Mode (Hot-Reload) - RECOMMENDED
```bash
docker-compose -f docker-compose.dev.yml up
```

This enables live code updates without rebuilding. Changes to your files are immediately reflected in the container.

### Option 2: Production Mode
```bash
docker-compose up -d
```

### Option 3: Using Docker CLI with GPU Support
```bash
docker run --gpus all -it \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 8080:8080 \
  pixxengine:latest
```

## Common Commands

### Start Development Mode
```bash
docker-compose -f docker-compose.dev.yml up
```

### Stop the container
```bash
# Production
docker-compose down

# Development
docker-compose -f docker-compose.dev.yml down
```

### View running containers
```bash
docker ps
```

### Execute commands in the container
```bash
# Production
docker-compose exec pixxengine python your_script.py

# Development
docker-compose -f docker-compose.dev.yml exec pixxengine python your_script.py
```

### Access the container shell
```bash
# Production
docker-compose exec pixxengine /bin/bash

# Development
docker-compose -f docker-compose.dev.yml exec pixxengine /bin/bash
```

### View container logs
```bash
# Production
docker-compose logs -f pixxengine

# Development
docker-compose -f docker-compose.dev.yml logs -f pixxengine
```

### Rebuild the image
```bash
# Production
docker-compose build --no-cache

# Development
docker-compose -f docker-compose.dev.yml build --no-cache
```

## Project Structure Inside Container

```
/app/
├── bin/           # Binary files
├── config/        # Configuration files
├── data/          # Data directory
├── logs/          # Logs directory
├── src/           # Source code
└── requirements.txt
```

## GPU Support

The Docker setup includes GPU support via NVIDIA Docker. The container automatically exposes all available GPUs. To check GPU availability inside the container:

```bash
docker-compose exec pixxengine nvidia-smi
```

## Customizing the Dockerfile

Edit [Dockerfile](Dockerfile) to:
- Change the base image (currently: `nvidia/cuda:12.8.0-runtime-ubuntu24.04`)
- Modify system dependencies
- Add additional setup commands

## Troubleshooting

### GPU not detected in container
1. Ensure `nvidia-docker2` is installed
2. Restart Docker daemon: `sudo systemctl restart docker`
3. Verify host GPU: `nvidia-smi`

### Port conflicts
Change port mappings in `docker-compose.yml`:
```yaml
ports:
  - "9000:5000"  # Map host port 9000 to container port 5000
```

### Permission issues
If you have permission issues with mounted volumes:
```bash
docker-compose exec pixxengine chown -R $(id -u):$(id -g) /app
```

## Development Mode Explained

**Development Mode** (`docker-compose.dev.yml`) is designed for active development:
- **Hot-reload**: Code changes are instantly reflected without rebuilding
- **Volume mounts**: Your local files are synchronized with the container
- **Interactive shell**: Keeps the container in the foreground for easy interaction
- **Excluded volumes**: The virtual environment is not synced to maintain performance

**When to use Dev Mode:**
- Active development and debugging
- Testing code changes
- Running scripts and utilities

**When to use Production Mode:**
- Final testing before deployment
- Running scheduled jobs
- Production deployments

## Creating an Alias for Convenience

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
alias pixx-dev='docker-compose -f docker-compose.dev.yml'
alias pixx-prod='docker-compose'
```

Then use:
```bash
pixx-dev up          # Start dev container
pixx-dev exec pixxengine /bin/bash  # Access dev container
pixx-prod up -d      # Start prod container
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
