# pixxEngine Dockerfile
# GPU-accelerated Python application with PyTorch, ONNX Runtime, and Computer Vision libraries

# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: Installing torch and related CUDA packages first for better compatibility
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining requirements (excluding torch packages already installed)
RUN pip install -r requirements.txt || pip install --ignore-installed -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/
COPY config/ ./config/

# Create directories for data and logs (will be mounted as volumes)
RUN mkdir -p /app/data /app/logs

# Expose port for Flask application (adjust if needed)
EXPOSE 5000

# Set default command (adjust based on your main application entry point)
# CMD ["python", "src/main.py"]
CMD ["/bin/bash"]
