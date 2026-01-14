#!/bin/bash
# Build script for AIDO.Cell-10M Benchmark Docker container

set -e

echo "==========================================="
echo "Building AIDO.Cell Benchmark Container"
echo "==========================================="

# Build the Docker image
echo "Building Docker image..."
docker build -t aido-cell-benchmark:latest .

echo ""
echo "✓ Build complete!"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -it --rm \\"
echo "    -v \$(pwd):/workspace/aido_cell_benchmark \\"
echo "    aido-cell-benchmark:latest"
echo ""
echo "Inside the container, run:"
echo "  python benchmark_aido_cell.py --devices 4"
echo ""
echo "To convert to Enroot (on HPC cluster):"
echo "  docker save aido-cell-benchmark:latest -o aido-cell-benchmark.tar"
echo "  enroot import dockerd://aido-cell-benchmark.tar"
