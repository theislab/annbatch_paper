#!/bin/bash
# Script to run AIDO.Cell benchmark with multi-GPU training
#
# Usage:
#   ./run_aido_cell_benchmark.sh [OPTIONS]
#
# Examples:
#   # Default settings
#   ./run_aido_cell_benchmark.sh
#
#   # Custom batch size with gradient accumulation to simulate large effective batch
#   ./run_aido_cell_benchmark.sh --batch-size 32 --accumulate-grad-batches 1024
#   # This gives effective batch size of 32,768 (32 * 1024)
#
#   # Smaller input for memory-constrained GPUs
#   ./run_aido_cell_benchmark.sh --batch-size 16 --input-dim 1000
#
#   # Larger sequence length (requires more memory)
#   ./run_aido_cell_benchmark.sh --batch-size 16 --input-dim 4000

# Exit on error
set -e

# Default parameters
BATCH_SIZE=512
INPUT_DIM=2000
N_BATCHES=7
LEARNING_RATE=1e-4
ACCUMULATE_GRAD_BATCHES=1
OUTPUT="aido_cell_fit_time_vs_loading_speed.csv"
MODEL_NAME="genbio-ai/AIDO.Cell-3M"

# Parse named arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --batch-size SIZE              Batch size (default: $BATCH_SIZE)"
    echo "  --input-dim DIM                Input dimension (default: $INPUT_DIM)"
    echo "  --n-batches N                  Number of batches per epoch (default: $N_BATCHES)"
    echo "  --learning-rate LR             Learning rate (default: $LEARNING_RATE)"
    echo "  --accumulate-grad-batches N    Gradient accumulation steps (default: $ACCUMULATE_GRAD_BATCHES)"
    echo "  --output FILE                  Output CSV file (default: $OUTPUT)"
    echo "  --model-name NAME              Model name (default: $MODEL_NAME)"
    echo "  -h, --help                     Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --input-dim)
            INPUT_DIM="$2"
            shift 2
            ;;
        --n-batches)
            N_BATCHES="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --accumulate-grad-batches)
            ACCUMULATE_GRAD_BATCHES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

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
