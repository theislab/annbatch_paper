#!/bin/bash
# Example script to run AIDO.Cell benchmark with multi-GPU training
#
# Usage:
#   ./run_aido_cell_benchmark.sh [BATCH_SIZE] [INPUT_DIM] [N_BATCHES] [LEARNING_RATE] [ACCUMULATE_GRAD_BATCHES] [OUTPUT] [MODEL_NAME]
#
# Examples:
#   # Default settings (batch_size=32, input_dim=2000)
#   ./run_aido_cell_benchmark.sh
#
#   # Custom batch size with gradient accumulation to simulate large effective batch
#   ./run_aido_cell_benchmark.sh 32 2000 7 1e-4 1024
#   # This gives effective batch size of 32,768 (32 * 1024)
#
#   # Smaller input for memory-constrained GPUs
#   ./run_aido_cell_benchmark.sh 16 1000
#
#   # Larger sequence length (requires more memory)
#   ./run_aido_cell_benchmark.sh 16 4000

# Exit on error
set -e

echo "=========================================="
echo "AIDO.Cell Benchmark Runner"
echo "=========================================="

# Check for CUDA availability
if ! command -v nvidia-smi &> /dev/null
then
    echo "WARNING: nvidia-smi not found. Running on CPU."
    DEVICES=1
    STRATEGY="ddp_spawn"
else
    # Get number of GPUs
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $N_GPUS GPU(s)"
    DEVICES=$N_GPUS
    if [ $N_GPUS -eq 1 ]; then
        STRATEGY="auto"
    else
        STRATEGY="ddp_spawn"
    fi
fi

# Default parameters (can be overridden by command line arguments)
BATCH_SIZE=${1:-256}
INPUT_DIM=${2:-2000}
N_BATCHES=${3:-7}
LEARNING_RATE=${4:-1e-4}
ACCUMULATE_GRAD_BATCHES=${5:-1}
OUTPUT=${6:-"aido_cell_fit_time_vs_loading_speed.csv"}
MODEL_NAME=${7:-"genbio-ai/AIDO.Cell-3M"}

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $ACCUMULATE_GRAD_BATCHES"
echo "  Effective Batch Size: $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"
echo "  Input Dimension: $INPUT_DIM"
echo "  Batches per Epoch: $N_BATCHES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output File: $OUTPUT"
echo "  Devices: $DEVICES"
echo "  Strategy: $STRATEGY"
echo ""
echo "Memory Notes:"
echo "  - Attention mask size: ~$((BATCH_SIZE * INPUT_DIM * INPUT_DIM * 4 / 1024 / 1024)) MB per batch"
echo "  - For OOM errors: reduce --batch-size or --input-dim"
echo "  - To simulate larger batches: increase --accumulate-grad-batches"
echo ""

# Run the benchmark
echo "Starting benchmark..."
python benchmark_aido_cell.py \
    --batch-size $BATCH_SIZE \
    --input-dim $INPUT_DIM \
    --n-batches $N_BATCHES \
    --devices $DEVICES \
    --strategy $STRATEGY \
    --learning-rate $LEARNING_RATE \
    --accumulate-grad-batches $ACCUMULATE_GRAD_BATCHES \
    --model-name "$MODEL_NAME" \
    --output $OUTPUT

echo ""
echo "Benchmark completed! Results saved to: $OUTPUT"
echo ""

# Display results if file exists
if [ -f "$OUTPUT" ]; then
    echo "Results preview:"
    head -n 20 "$OUTPUT"
fi

