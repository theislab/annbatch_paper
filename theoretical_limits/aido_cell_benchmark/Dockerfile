# Dockerfile for AIDO.Cell-10M Benchmark
# Focused environment for running the AIDO Cell benchmark with multi-GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages from requirements file
COPY requirements_aido_cell.txt /tmp/requirements_aido_cell.txt
RUN pip install --no-cache-dir -r /tmp/requirements_aido_cell.txt

# Try to install optional performance packages (don't fail if they don't build)
RUN pip install --no-cache-dir ninja || true
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn installation skipped (optional)"

# Set working directory
WORKDIR /workspace/aido_cell_benchmark

# Copy benchmark scripts
COPY . /workspace/aido_cell_benchmark/

# Set default command
CMD ["/bin/bash"]
