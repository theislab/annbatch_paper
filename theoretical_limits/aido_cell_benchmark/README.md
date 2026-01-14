# AIDO.Cell-10M Benchmark

Complete benchmarking suite for testing the Hugging Face GenBio-AI AIDO.Cell-10M model with multi-GPU training support.

## Quick Start

### Option 1: Native Installation

```bash
# Navigate to this directory
cd aido_cell_benchmark

# 1. Install dependencies
pip install -r requirements_aido_cell.txt

# 2. Test installation
python test_installation.py

# 3. Run quick test
python benchmark_aido_cell.py --batch-size 4096 --n-batches 3

# 4. Run full benchmark with 4 GPUs
python benchmark_aido_cell.py --devices 4 --strategy ddp

# 5. Visualize results
python visualize_aido_cell_results.py aido_cell_fit_time_vs_loading_speed.csv
```

### Option 2: Docker (Recommended for Reproducibility)

```bash
# Build container
./build_docker.sh

# Run container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/aido_cell_benchmark \
  aido-cell-benchmark:latest

# Inside container
python test_installation.py
python benchmark_aido_cell.py --devices 4
```

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for complete Docker documentation.

## Files in This Directory

### 📜 Scripts
- **`benchmark_aido_cell.py`** - Main benchmark script with multi-GPU support
- **`visualize_aido_cell_results.py`** - Create publication-quality plots
- **`test_installation.py`** - Verify setup before running
- **`run_aido_cell_benchmark.sh`** - Automated shell runner
- **`run_aido_cell_benchmark.sbatch`** - SLURM script for HPC clusters

### 🐳 Docker Files
- **`Dockerfile`** - Container definition for reproducible environment
- **`build_docker.sh`** - Automated Docker build script
- **`.dockerignore`** - Docker build optimization
- **`DOCKER_SETUP.md`** - Complete Docker documentation

### 📚 Documentation
- **`INDEX_AIDO_CELL.md`** - File navigation guide (start here!)
- **`QUICKSTART_AIDO_CELL.md`** - Quick start guide
- **`README_AIDO_CELL_BENCHMARK.md`** - Comprehensive documentation
- **`AIDO_CELL_SUMMARY.md`** - Technical implementation details

### 📦 Configuration
- **`requirements_aido_cell.txt`** - Python dependencies

## Documentation

For detailed information, see:
- **First-time users**: [QUICKSTART_AIDO_CELL.md](QUICKSTART_AIDO_CELL.md)
- **Complete guide**: [README_AIDO_CELL_BENCHMARK.md](README_AIDO_CELL_BENCHMARK.md)
- **All files**: [INDEX_AIDO_CELL.md](INDEX_AIDO_CELL.md)

## Dependencies

The benchmark uses `MockDataset` from the parent directory (`../mock_dataset.py`).

## HPC Usage

For SLURM-based clusters:
```bash
sbatch run_aido_cell_benchmark.sbatch
```

The SLURM script is configured for:
- Partition: `mcml-hgx-h100-94x4`
- QoS: `mcml`
- 4 H100 GPUs
- 256GB RAM
- 4-hour time limit

## Output

Results are saved as CSV files with columns:
- `sleep` - Artificial data loading delay
- `fit_time` - Training time for 20 steps
- `samples_per_sec` - Data loading throughput
- `model` - Model name
- `batch_size` - Batch size used
- `n_gpus` - Number of GPUs
- `strategy` - Training strategy (ddp/fsdp)

---

**Parent Directory**: `../` (theoretical_limits)  
**Shared Dependencies**: `../mock_dataset.py`, `../models.py`

