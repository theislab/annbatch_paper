# AIDO.Cell-10M Benchmark - File Index

Quick reference guide to all files in the AIDO.Cell benchmark suite.

## 📖 Documentation (Start Here!)

### For First-Time Users
- **[QUICKSTART_AIDO_CELL.md](QUICKSTART_AIDO_CELL.md)** ⭐ START HERE
  - Step-by-step getting started guide
  - Quick test runs
  - Common use cases
  - Troubleshooting quick fixes

### For Detailed Information
- **[README_AIDO_CELL_BENCHMARK.md](README_AIDO_CELL_BENCHMARK.md)**
  - Complete documentation
  - All command-line options
  - Multi-GPU strategies explained
  - Comprehensive troubleshooting

### For Developers
- **[AIDO_CELL_SUMMARY.md](AIDO_CELL_SUMMARY.md)**
  - Implementation details
  - Architecture overview
  - Technical highlights
  - Integration guide

## 🚀 Executable Scripts

### Python Scripts
- **[benchmark_aido_cell.py](benchmark_aido_cell.py)** - Main benchmark script
  - Run the full benchmark
  - Multi-GPU training support
  - Configurable parameters
  - Usage: `python benchmark_aido_cell.py --devices 4`

- **[visualize_aido_cell_results.py](visualize_aido_cell_results.py)** - Visualization script
  - Create plots from results
  - Compare multiple runs
  - Usage: `python visualize_aido_cell_results.py results.csv`

- **[test_installation.py](test_installation.py)** - Installation test
  - Verify setup before running
  - Check dependencies
  - Test GPU availability
  - Usage: `python test_installation.py`

### Shell Scripts
- **[run_aido_cell_benchmark.sh](run_aido_cell_benchmark.sh)** - Automated runner
  - Easy command-line execution
  - Auto-detects GPUs
  - Usage: `./run_aido_cell_benchmark.sh [batch_size] [input_dim] [n_batches]`

- **[run_aido_cell_benchmark.sbatch](run_aido_cell_benchmark.sbatch)** - SLURM script
  - For HPC clusters
  - Multi-node support
  - Usage: `sbatch run_aido_cell_benchmark.sbatch`

## 📦 Configuration Files

- **[requirements_aido_cell.txt](requirements_aido_cell.txt)** - Python dependencies
  - All required packages
  - Version specifications
  - Usage: `pip install -r requirements_aido_cell.txt`

## 📊 Example Workflow

### Complete Workflow (Recommended)
```bash
# 1. Check installation
python test_installation.py

# 2. Install dependencies if needed
pip install -r requirements_aido_cell.txt

# 3. Quick test
python benchmark_aido_cell.py --batch-size 4096 --n-batches 3

# 4. Full benchmark
python benchmark_aido_cell.py --devices 4 --strategy ddp

# 5. Visualize results
python visualize_aido_cell_results.py aido_cell_fit_time_vs_loading_speed.csv
```

### Using Shell Script
```bash
# Make executable (first time only)
chmod +x run_aido_cell_benchmark.sh

# Run with defaults
./run_aido_cell_benchmark.sh

# Run with custom settings
./run_aido_cell_benchmark.sh 16384 20000 10 1e-4 output.csv
```

### On HPC Cluster
```bash
# Submit job
sbatch run_aido_cell_benchmark.sbatch

# Monitor progress
squeue -u $USER
tail -f aido_cell_benchmark_*.out
```

## 🎯 Quick Reference

### What to Run When

| Goal | Command | Time |
|------|---------|------|
| Test setup | `python test_installation.py` | 1 min |
| Quick test | `python benchmark_aido_cell.py --batch-size 4096 --n-batches 3` | 5 min |
| Single GPU | `python benchmark_aido_cell.py --devices 1` | 30 min |
| Multi-GPU | `python benchmark_aido_cell.py --devices 4 --strategy ddp` | 45 min |
| Visualize | `python visualize_aido_cell_results.py results.csv` | 1 min |

### Output Files You'll Get

| File | Description |
|------|-------------|
| `*.csv` | Benchmark results data |
| `*_fit_time.png/svg` | Line plots |
| `*_speedup.png/svg` | Bar charts |
| `*_combined.csv` | Merged results |
| `*_speedup_summary.csv` | Statistics |

## 🔧 Troubleshooting

### Quick Fixes

| Problem | Solution | File |
|---------|----------|------|
| Don't know where to start | Read quickstart | [QUICKSTART_AIDO_CELL.md](QUICKSTART_AIDO_CELL.md) |
| Installation issues | Run test script | `python test_installation.py` |
| Missing packages | Install requirements | `pip install -r requirements_aido_cell.txt` |
| Out of memory | Reduce batch size | `--batch-size 8192` |
| Multi-GPU not working | Check nvidia-smi | `nvidia-smi` |
| Need detailed help | Read full docs | [README_AIDO_CELL_BENCHMARK.md](README_AIDO_CELL_BENCHMARK.md) |

## 📁 Related Files (Not Part of This Suite)

These files are used by the benchmark but existed before:
- **[mock_dataset.py](mock_dataset.py)** - Dataset with controlled loading delays
- **[models.py](models.py)** - SimpleLinearModel and SCVI implementations
- **[loading_speed_vs_fit_time.ipynb](loading_speed_vs_fit_time.ipynb)** - Original notebook
- **[fit_time_vs_loading_speed.csv](fit_time_vs_loading_speed.csv)** - Existing results

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART_AIDO_CELL.md](QUICKSTART_AIDO_CELL.md)
2. Run `test_installation.py`
3. Run quick test
4. Run full benchmark
5. Visualize results

### Intermediate
1. Read [README_AIDO_CELL_BENCHMARK.md](README_AIDO_CELL_BENCHMARK.md)
2. Try different GPU counts
3. Test different batch sizes
4. Compare with existing models
5. Customize parameters

### Advanced
1. Read [AIDO_CELL_SUMMARY.md](AIDO_CELL_SUMMARY.md)
2. Modify training strategies
3. Adapt for different models
4. Run on HPC clusters
5. Integrate with paper workflow

## 📞 Getting Help

1. **Quick issues**: Check [QUICKSTART_AIDO_CELL.md](QUICKSTART_AIDO_CELL.md) troubleshooting
2. **Installation**: Run `python test_installation.py`
3. **Configuration**: See [README_AIDO_CELL_BENCHMARK.md](README_AIDO_CELL_BENCHMARK.md)
4. **Technical details**: Read [AIDO_CELL_SUMMARY.md](AIDO_CELL_SUMMARY.md)

## ✅ Pre-flight Checklist

Before running the benchmark:
- [ ] Read quickstart guide
- [ ] Install dependencies (`pip install -r requirements_aido_cell.txt`)
- [ ] Run test script (`python test_installation.py`)
- [ ] Check GPU availability (`nvidia-smi`)
- [ ] Authenticate with HF if needed (`huggingface-cli login`)
- [ ] Test with small config first
- [ ] Have 50+ GB disk space free

## 🎯 Common Tasks

### First Time Setup
```bash
pip install -r requirements_aido_cell.txt
python test_installation.py
```

### Run Benchmark
```bash
python benchmark_aido_cell.py --devices 4
```

### Visualize Results
```bash
python visualize_aido_cell_results.py *.csv
```

### Compare GPU Counts
```bash
for gpus in 1 2 4; do
    python benchmark_aido_cell.py --devices $gpus --output ${gpus}gpu.csv
done
python visualize_aido_cell_results.py *gpu.csv
```

---

**Last Updated**: January 14, 2026
**Version**: 1.0
**Status**: Production Ready

