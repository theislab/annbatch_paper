# Quick Start Guide: AIDO.Cell-10M Benchmark

## Overview

This guide will help you quickly get started with benchmarking the AIDO.Cell-10M model for data loading speed analysis.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU(s) (optional but recommended)
- ~50GB disk space for model download
- Internet connection for first-time model download

## Installation (2 minutes)

```bash
# Navigate to the theoretical_limits directory
cd /path/to/annbatch_paper/theoretical_limits

# Install dependencies
pip install -r requirements_aido_cell.txt

# (Optional) Authenticate with Hugging Face if the model requires it
huggingface-cli login
```

## Quick Test Run (5 minutes)

Test with a smaller configuration to verify everything works:

```bash
python benchmark_aido_cell.py \
    --batch-size 4096 \
    --n-batches 3 \
    --devices 1 \
    --output test_run.csv
```

## Full Benchmark Run

### Single GPU (~30 minutes)

```bash
python benchmark_aido_cell.py \
    --batch-size 32768 \
    --devices 1 \
    --strategy auto \
    --output aido_cell_1gpu.csv
```

### Multi-GPU (~45 minutes)

```bash
# Using 4 GPUs with DDP
python benchmark_aido_cell.py \
    --batch-size 32768 \
    --devices 4 \
    --strategy ddp \
    --output aido_cell_4gpu.csv
```

### Using the Shell Script

```bash
# Make it executable (first time only)
chmod +x run_aido_cell_benchmark.sh

# Run with default settings
./run_aido_cell_benchmark.sh

# Run with custom settings
./run_aido_cell_benchmark.sh 16384 20000 10 1e-4 custom_output.csv
#                            batch  genes batches lr    output
```

## Visualization

After running benchmarks:

```bash
# Visualize single run
python visualize_aido_cell_results.py aido_cell_4gpu.csv

# Compare multiple runs
python visualize_aido_cell_results.py \
    aido_cell_1gpu.csv \
    aido_cell_4gpu.csv \
    --output-prefix multi_gpu_comparison
```

## HPC/SLURM Clusters

```bash
# Submit job
sbatch run_aido_cell_benchmark.sbatch

# Monitor job
squeue -u $USER
watch -n 1 squeue -u $USER

# View output (replace JOBID)
tail -f aido_cell_benchmark_JOBID.out

# Cancel job if needed
scancel JOBID
```

## Expected Output Files

After running a benchmark, you'll get:

1. **CSV file** with results:
   - `aido_cell_fit_time_vs_loading_speed.csv` (or custom name)
   
2. **Visualization files** (after running visualize script):
   - `*_fit_time.png` and `*_fit_time.svg` - Line plots
   - `*_speedup.png` and `*_speedup.svg` - Bar charts
   - `*_combined.csv` - Combined results
   - `*_speedup_summary.csv` - Speedup statistics

## Understanding the Results

The benchmark measures:
- **Fit Time**: Total training time for one epoch
- **Samples/sec**: Data loading throughput
- **Speedup**: Ratio of slowest to fastest fit time

Key insights:
- Higher samples/sec → Lower fit time (up to a point)
- Plateau indicates model computation is the bottleneck
- Multi-GPU should show better handling of slow data loading

## Example Results Interpretation

```
sleep  | samples_per_sec | fit_time | speedup
-------|----------------|----------|--------
0.004  | 5,000,000      | 25.4s    | 1.00x (baseline)
0.040  | 500,000        | 28.1s    | 1.11x
0.400  | 50,000         | 45.2s    | 1.78x
4.000  | 5,000          | 85.6s    | 3.37x
```

This shows that when data loading drops below ~500,000 samples/sec, training time starts to increase significantly.

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` to 16384 or 8192 |
| Model won't download | Check internet, authenticate with HF |
| Multi-GPU not working | Try `--strategy ddp_spawn` |
| Import errors | `pip install -r requirements_aido_cell.txt` |
| CUDA not available | Script will auto-fallback to CPU |

## Advanced Usage

### Test Different Batch Sizes

```bash
for bs in 4096 8192 16384 32768; do
    python benchmark_aido_cell.py \
        --batch-size $bs \
        --output aido_cell_bs${bs}.csv
done

# Compare results
python visualize_aido_cell_results.py aido_cell_bs*.csv
```

### Test Different GPU Counts

```bash
for gpus in 1 2 4 8; do
    python benchmark_aido_cell.py \
        --devices $gpus \
        --output aido_cell_${gpus}gpu.csv
done
```

### Custom Model

```bash
# Use a different model from Hugging Face
python benchmark_aido_cell.py \
    --model-name "your-org/your-model" \
    --output custom_model.csv
```

## Getting Help

1. Check the detailed README: `README_AIDO_CELL_BENCHMARK.md`
2. Review error messages in the output
3. Enable debug mode: Add `export NCCL_DEBUG=INFO` before running
4. Test with smaller configuration first

## Next Steps

After successful benchmarking:
1. Compare results with other models in the paper
2. Analyze where data loading becomes a bottleneck
3. Use insights to optimize your data pipeline
4. Include results in your publication figures

## Estimated Run Times

| Configuration | GPUs | Time |
|--------------|------|------|
| Quick test | 1 | ~5 min |
| Full benchmark | 1 | ~30 min |
| Full benchmark | 4 | ~45 min |
| Full benchmark | 8 | ~60 min |

*Times vary based on hardware and model loading time*

