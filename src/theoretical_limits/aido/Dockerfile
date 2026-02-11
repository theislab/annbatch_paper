# Dockerfile for AIDO.Cell-10M Benchmark
# Focused environment for running the AIDO Cell benchmark with multi-GPU support
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to avoid PEP 668 externally-managed-environment error
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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

# Initialize and install aido.cell submodule
RUN if [ -d "aido.cell" ]; then \
        cd aido.cell && \
        pip install --no-cache-dir -e . && \
        cd ..; \
    else \
        echo "Warning: aido.cell submodule not found, skipping installation"; \
    fi

# Set default command
CMD ["/bin/bash"]
