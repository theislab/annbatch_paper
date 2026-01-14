# Docker Setup for AIDO.Cell-10M Benchmark

This directory contains Docker configuration for running the AIDO.Cell-10M benchmark in a containerized environment with multi-GPU support.

## Quick Start

### Prerequisites
- Docker Engine 20.10+ with GPU support
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU with CUDA 11.8+ compatible drivers

### Build and Run

```bash
# Build the Docker image
./build_docker.sh

# Or manually
docker build -t aido-cell-benchmark:latest .

# Run with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest
```

## Using the Container

### Quick Test

```bash
# Inside the container
python test_installation.py
```

### Run Benchmark

```bash
# Quick test with 1 GPU
python benchmark_aido_cell.py --batch-size 4096 --n-batches 3 --devices 1

# Full benchmark with 4 GPUs
python benchmark_aido_cell.py --devices 4 --strategy ddp

# Visualize results
python visualize_aido_cell_results.py aido_cell_fit_time_vs_loading_speed.csv
```

## Container Details

### Base Image
- **Base**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.10
- **CUDA**: 11.8
- **cuDNN**: 8

### Installed Packages

All packages from `requirements_aido_cell.txt`:
- PyTorch 2.0+ with CUDA support
- PyTorch Lightning 2.0+
- Transformers 4.30+
- Accelerate 0.20+
- Hugging Face Hub
- pandas, numpy, matplotlib, seaborn

### MockDataset Dependency

The container mounts `../mock_dataset.py` from the parent directory as a read-only volume, so the benchmark can access the shared MockDataset class.

## Building for HPC (Enroot/Singularity)

### Convert to Enroot Image

```bash
# Build Docker image
docker build -t aido-cell-benchmark:latest .

# Save as tar
docker save aido-cell-benchmark:latest -o aido-cell-benchmark.tar

# On HPC cluster, convert to squashfs
enroot import dockerd://aido-cell-benchmark.tar
# This creates aido-cell-benchmark.sqsh
```

### Update SLURM Script

Edit `run_aido_cell_benchmark.sbatch` to use your Enroot image:

```bash
CONTAINER_IMAGE="/path/to/your/aido-cell-benchmark.sqsh"
CONTAINER_MOUNTS="/workspace:/workspace"

srun --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     python /workspace/benchmark_aido_cell.py --devices 4
```

### Convert to Singularity

```bash
# Convert Docker to Singularity
singularity build aido-cell-benchmark.sif docker-daemon://aido-cell-benchmark:latest

# Run with Singularity
singularity exec --nv aido-cell-benchmark.sif python benchmark_aido_cell.py
```

## Volume Mounts

By default, mount the following:
- `.:/workspace/aido_cell_benchmark` - Benchmark code
- `./data:/data` - Results directory (optional)

You can also manually copy the parent's mock_dataset.py:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  -v $(pwd)/../mock_dataset.py:/workspace/aido_cell_benchmark/mock_dataset_parent.py:ro \
  aido-cell-benchmark:latest
```

### Custom Data Directory

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  -v /path/to/results:/data \
  aido-cell-benchmark:latest
```

## Workflow Examples

### Complete Benchmark Workflow

```bash
# 1. Build container
./build_docker.sh

# 2. Run container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest bash

# 3. Inside container, test installation
python test_installation.py

# 4. Run benchmark
python benchmark_aido_cell.py --devices 4 --strategy ddp

# 5. Visualize results
python visualize_aido_cell_results.py *.csv
```

### One-Off Benchmark Run

```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest \
  python benchmark_aido_cell.py --devices 4
```

### Background Container

```bash
# Run in detached mode with name
docker run --gpus all -d --name aido-benchmark \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest tail -f /dev/null

# Execute commands in running container
docker exec -it aido-benchmark python test_installation.py
docker exec -it aido-benchmark python benchmark_aido_cell.py --devices 4

# Stop and remove
docker stop aido-benchmark
docker rm aido-benchmark
```

## Customization

### Add Additional Packages

Edit `requirements_aido_cell.txt` and rebuild:

```bash
echo "your-package-name" >> requirements_aido_cell.txt
./build_docker.sh
```

### Change CUDA Version

Edit `Dockerfile` base image:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If it fails, install nvidia-docker2:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory

Increase shared memory size when running:

```bash
docker run --gpus all -it --rm --shm-size=32g \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest
```

### Model Download Issues

If model download fails:

```bash
# Authenticate with Hugging Face (run on host first)
huggingface-cli login

# Mount cached models
docker run --gpus all -it --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest
```

### Permission Issues

Run as your user:

```bash
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest
```

## Performance Notes

- **Shared Memory**: Set `--shm-size` appropriately for DataLoader workers
- **GPU Memory**: Monitor with `nvidia-smi` inside container
- **Network**: `host` network mode provides best performance
- **Storage**: Use bind mounts for better I/O performance

## Multi-GPU Configuration

### 4 GPUs (Default)

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest \
  python benchmark_aido_cell.py --devices 4 --strategy ddp
```

### 8 GPUs

Specify GPU devices:

```bash
docker run --gpus all -it --rm \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest \
  python benchmark_aido_cell.py --devices 8 --strategy ddp
```

### Specific GPUs

Use only specific GPUs (e.g., GPU 0 and 1):

```bash
docker run --gpus '"device=0,1"' -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest \
  python benchmark_aido_cell.py --devices 2 --strategy ddp
```

### FSDP for Large Models

For very large models that don't fit in GPU memory:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest \
  python benchmark_aido_cell.py --devices 4 --strategy fsdp
```

## Links

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Enroot Documentation](https://github.com/NVIDIA/enroot)
- [AIDO.Cell Model](https://huggingface.co/genbio-ai/AIDO.Cell-10M)
- [PyTorch Lightning DDP](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html)

---

**Last Updated**: January 14, 2026
**Purpose**: AIDO.Cell-10M benchmark containerization

