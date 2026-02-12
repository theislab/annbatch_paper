# Dockerfile for ResNet101 Throughput Benchmark
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
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to avoid PEP 668 externally-managed-environment error
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace/cnn_benchmark

# Copy benchmark script
COPY benchmark_resnet50.py /workspace/cnn_benchmark/

# Set default command
CMD ["python3", "benchmark_resnet50.py"]
