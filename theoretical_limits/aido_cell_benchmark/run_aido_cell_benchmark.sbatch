#!/bin/bash
#SBATCH --job-name=aido_cell_benchmark
#SBATCH --output=aido_cell_benchmark_%j.out
#SBATCH --error=aido_cell_benchmark_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# SLURM batch script for running AIDO.Cell-10M benchmark on HPC cluster
# Adjust the SBATCH parameters above according to your cluster's configuration

echo "=========================================="
echo "AIDO.Cell-10M Benchmark - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load necessary modules (adjust for your cluster)
# Example for common HPC setups:
# module load cuda/11.8
# module load python/3.10
# module load gcc/11.2.0

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Set environment variables for better multi-GPU performance
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
BATCH_SIZE=${BATCH_SIZE:-32768}
INPUT_DIM=${INPUT_DIM:-20000}
N_BATCHES=${N_BATCHES:-7}
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
STRATEGY=${STRATEGY:-ddp}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
OUTPUT="aido_cell_benchmark_${SLURM_JOB_ID}.csv"

echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Input Dimension: $INPUT_DIM"
echo "  Batches per Epoch: $N_BATCHES"
echo "  Number of GPUs: $N_GPUS"
echo "  Strategy: $STRATEGY"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output File: $OUTPUT"
echo ""

# Run the benchmark
echo "Starting benchmark..."
srun python benchmark_aido_cell.py \
    --batch-size $BATCH_SIZE \
    --input-dim $INPUT_DIM \
    --n-batches $N_BATCHES \
    --devices $N_GPUS \
    --strategy $STRATEGY \
    --learning-rate $LEARNING_RATE \
    --output $OUTPUT

echo ""
echo "Benchmark completed!"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT"
echo ""

# Display results summary
if [ -f "$OUTPUT" ]; then
    echo "Results summary:"
    cat "$OUTPUT"
fi

