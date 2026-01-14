# AIDO.Cell-10M Benchmark

This directory contains a benchmark script for testing the AIDO.Cell-10M model from Hugging Face with varying data loading speeds and multi-GPU training.

## Overview

The benchmark measures how fit time varies with data loading speed using the GenBio-AI AIDO.Cell-10M model. This helps understand the relationship between data pipeline performance and training efficiency for large-scale single-cell models.

## Installation

### Prerequisites

```bash
pip install torch pytorch-lightning transformers accelerate pandas numpy
```

For multi-GPU training, ensure you have:
- CUDA-capable GPUs
- Proper CUDA drivers installed
- PyTorch with CUDA support

### Model Access

The AIDO.Cell-10M model is available on Hugging Face:
- Model: `genbio-ai/AIDO.Cell-10M`
- You may need to accept terms of use on Hugging Face and authenticate with `huggingface-cli login`

## Usage

### Basic Usage (Single GPU)

```bash
python benchmark_aido_cell.py
```

### Multi-GPU Training

```bash
# Use all available GPUs with DDP strategy
python benchmark_aido_cell.py --devices -1 --strategy ddp

# Use specific number of GPUs
python benchmark_aido_cell.py --devices 4 --strategy ddp

# Use FSDP (Fully Sharded Data Parallel) for very large models
python benchmark_aido_cell.py --devices 8 --strategy fsdp
```

### Advanced Options

```bash
python benchmark_aido_cell.py \
    --batch-size 16384 \
    --input-dim 20000 \
    --n-batches 10 \
    --devices 4 \
    --strategy ddp \
    --learning-rate 1e-4 \
    --output my_benchmark_results.csv \
    --model-name genbio-ai/AIDO.Cell-10M
```

### Command-Line Arguments

- `--batch-size`: Batch size for training (default: 32768)
- `--input-dim`: Input dimension/number of genes (default: 20000)
- `--n-batches`: Number of batches per epoch (default: 7)
- `--devices`: Number of GPUs to use, -1 for all available (default: -1)
- `--strategy`: Multi-GPU strategy: ddp, ddp_spawn, fsdp (default: ddp)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--output`: Output CSV filename (default: aido_cell_fit_time_vs_loading_speed.csv)
- `--model-name`: Hugging Face model name (default: genbio-ai/AIDO.Cell-10M)

## Multi-GPU Strategies

### DDP (Distributed Data Parallel)
- **Recommended for most use cases**
- Efficient for models that fit in GPU memory
- Good scaling efficiency
```bash
python benchmark_aido_cell.py --strategy ddp --devices 4
```

### FSDP (Fully Sharded Data Parallel)
- **For very large models**
- Shards model parameters across GPUs
- Better memory efficiency
```bash
python benchmark_aido_cell.py --strategy fsdp --devices 8
```

## Output

The script generates:
1. **CSV file** with benchmark results containing:
   - `sleep`: Artificial delay in data loading (seconds)
   - `fit_time`: Total time for one epoch (seconds)
   - `samples_per_sec`: Data loading throughput (samples/second)
   - `model`: Model name
   - `batch_size`: Batch size used
   - `n_gpus`: Number of GPUs used
   - `strategy`: Training strategy used

2. **Console output** with:
   - Progress information
   - Summary statistics
   - Min/max fit times and speedup

## Example Output

```
================================================================================
AIDO.Cell-10M Benchmark Configuration
================================================================================
Model: genbio-ai/AIDO.Cell-10M
Batch size: 32768
Input dimension: 20000
Batches per epoch: 7
Learning rate: 0.0001
Strategy: ddp
Devices: 4
================================================================================

Training with sleep time: 0.0040s
Fit time: 25.43s

Training with sleep time: 0.0127s
Fit time: 26.18s

...

Results saved to: aido_cell_fit_time_vs_loading_speed.csv

Min fit time: 25.43s (fastest loading)
Max fit time: 65.89s (slowest loading)
Speedup: 2.59x
```

## Notes

- The benchmark uses a `MockDataset` that generates synthetic data with controlled loading delays
- The script automatically detects available GPUs and adjusts if CUDA is not available
- For production use with real data, you would replace `MockDataset` with actual single-cell data
- The model wrapper may need adjustments based on the specific API of AIDO.Cell-10M

## Visualization

After running the benchmark, visualize the results:

```bash
# Visualize single benchmark
python visualize_aido_cell_results.py aido_cell_fit_time_vs_loading_speed.csv

# Compare multiple benchmarks (e.g., AIDO.Cell vs other models)
python visualize_aido_cell_results.py \
    aido_cell_fit_time_vs_loading_speed.csv \
    ../theoretical_limits/fit_time_vs_loading_speed.csv \
    --output-prefix comparison
```

The visualization script generates:
- Line plots showing fit time vs loading speed for each model
- Bar charts comparing speedup across models
- Combined CSV with all results
- Summary statistics

## Running on HPC Clusters

For SLURM-based clusters:

```bash
# Submit job to SLURM scheduler
sbatch run_aido_cell_benchmark.sbatch

# Check job status
squeue -u $USER

# View output
tail -f aido_cell_benchmark_<JOB_ID>.out
```

Adjust the SBATCH parameters in `run_aido_cell_benchmark.sbatch` according to your cluster's configuration.

## Complete Workflow Example

```bash
# 1. Install dependencies
pip install -r requirements_aido_cell.txt

# 2. Authenticate with Hugging Face (if needed)
huggingface-cli login

# 3. Run benchmark with 4 GPUs
python benchmark_aido_cell.py --devices 4 --strategy ddp

# 4. Visualize results
python visualize_aido_cell_results.py aido_cell_fit_time_vs_loading_speed.csv

# 5. (Optional) Compare with existing results
python visualize_aido_cell_results.py \
    aido_cell_fit_time_vs_loading_speed.csv \
    fit_time_vs_loading_speed.csv
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Use fewer GPUs or switch to FSDP strategy
- Reduce `--input-dim`

### Model Loading Issues
- Ensure you have sufficient disk space
- Check internet connection for model download
- Authenticate with Hugging Face: `huggingface-cli login`
- Install git-lfs if required: `apt-get install git-lfs` or `brew install git-lfs`

### Multi-GPU Issues
- Ensure all GPUs are visible: `nvidia-smi`
- Check NCCL/CUDA compatibility
- Try `--strategy ddp_spawn` if DDP fails
- Set environment variables: `export NCCL_DEBUG=INFO`

### Import Errors
- Verify all dependencies are installed: `pip install -r requirements_aido_cell.txt`
- Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{annbatch_paper,
  title={Efficient Data Loading for Single-Cell Analysis},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

