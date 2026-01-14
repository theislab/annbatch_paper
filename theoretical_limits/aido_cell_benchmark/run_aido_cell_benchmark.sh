#!/bin/bash
# Example script to run AIDO.Cell-10M benchmark with multi-GPU training

# Exit on error
set -e

echo "=========================================="
echo "AIDO.Cell-10M Benchmark Runner"
echo "=========================================="

# Check for CUDA availability
if ! command -v nvidia-smi &> /dev/null
then
    echo "WARNING: nvidia-smi not found. Running on CPU."
    DEVICES=1
    STRATEGY="auto"
else
    # Get number of GPUs
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $N_GPUS GPU(s)"
    DEVICES=$N_GPUS
    STRATEGY="ddp"
fi

# Default parameters (can be overridden by command line arguments)
BATCH_SIZE=${1:-32768}
INPUT_DIM=${2:-20000}
N_BATCHES=${3:-7}
LEARNING_RATE=${4:-1e-4}
OUTPUT=${5:-"aido_cell_fit_time_vs_loading_speed.csv"}

echo ""
echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Input Dimension: $INPUT_DIM"
echo "  Batches per Epoch: $N_BATCHES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output File: $OUTPUT"
echo "  Devices: $DEVICES"
echo "  Strategy: $STRATEGY"
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
    --output $OUTPUT

echo ""
echo "Benchmark completed! Results saved to: $OUTPUT"
echo ""

# Display results if file exists
if [ -f "$OUTPUT" ]; then
    echo "Results preview:"
    head -n 20 "$OUTPUT"
fi

