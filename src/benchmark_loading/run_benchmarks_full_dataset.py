from __future__ import annotations

import json
import subprocess
import sys
import pandas as pd

from pathlib import Path


STORE_PATH_ZARR = Path("/vol/data/annbatch_benchmark/tahoe100M/zarr_shuffled")
STORE_PATH_H5AD = Path("/vol/data/annbatch_benchmark/tahoe100M/h5ad_shuffled")


def run_benchmark(cmd, name):
    """Run a benchmark and return results or None on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = json.loads(result.stdout.split("\n")[-2].rstrip())
        print(f"{name}: {output}")
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}:")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return None


N_SAMPLES = 100_000_000  # Run on all 100Mio cells

print("Running benchmarks on full dataset...")
# Benchmark annbatch without GPU
arrayloaders = run_benchmark([
    "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/annbatch/benchmark_annbatch.py",
    f"--store_path={STORE_PATH_ZARR}",
    "--chunk_size=512",
    "--preload_nchunks=32",
    "--preload_to_gpu=False",
    f"--n_samples={N_SAMPLES}"
], "annbatch")

# Benchmark annbatch with GPU
arrayloaders_gpu = run_benchmark([
    "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/annbatch/benchmark_annbatch.py",
    f"--store_path={STORE_PATH_ZARR}",
    "--chunk_size=512",
    "--preload_nchunks=32",
    "--preload_to_gpu=True",
    f"--n_samples={N_SAMPLES}"
], "annbatch (GPU)")

# Benchmark scDataset
scdataset = run_benchmark([
    "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/scDataset/benchmark_scDataset.py",
    f"--store_path={STORE_PATH_H5AD}",
    "--num_workers=6",
    "--batch_size=4096",
    "--block_size=4",
    "--fetch_factor=2",
    f"--n_samples={N_SAMPLES}"
], "scDataset")

# Benchmark MappedCollection
mapped_collection = run_benchmark([
    "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/mapped_collection/benchmark_mapped_collection.py",
    f"--store_path={STORE_PATH_H5AD}",
    "--num_workers=8",
    f"--n_samples={N_SAMPLES}"
], "MappedCollection")


# Save results (only non-None results)
results = [r for r in [arrayloaders, arrayloaders_gpu, scdataset, mapped_collection] if r is not None]
if results:
    res = pd.DataFrame(results)
    res.to_csv(f"{REPO_PATH}/src/benchmark_loading/loading_times_full_epoch_2mio_samples.csv", index=False)
    print(f"Saved {len(results)} benchmark results")
else:
    print("No successful benchmarks to save")
