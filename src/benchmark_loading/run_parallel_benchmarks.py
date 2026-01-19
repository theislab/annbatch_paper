"""
Benchmark data loading performance with multiple parallel processes.
This script runs data loaders in parallel processes and measures the cumulative
throughput (samples/sec) across all processes.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


REPO_PATH = Path(__file__).resolve().parent.parent.parent
STORE_PATH_ZARR = Path("/dss/mcmlscratch/04/di93zer/tahoe100M")
STORE_PATH_H5AD = Path("/dss/mcmlscratch/04/di93zer/tahoe100M_h5ad")

# Number of samples per process
N_SAMPLES_PER_PROCESS = 1_000_000


def run_single_benchmark_process(args):
    """
    Run a single benchmark in a separate process.
    Returns a dict with timing information.
    """
    process_id, loader_type, n_samples = args

    print(f"[Process {process_id}] Starting {loader_type} benchmark...")

    if loader_type == "annbatch":
        cmd = [
            "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/annbatch/benchmark_annbatch.py",
            f"--store_path={STORE_PATH_ZARR}",
            "--chunk_size=256",
            "--preload_nchunks=32",
            "--use_torch_loader=False",
            "--preload_to_gpu=False",
            f"--n_samples={n_samples}"
        ]
    elif loader_type == "annbatch_gpu":
        cmd = [
            "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/annbatch/benchmark_annbatch.py",
            f"--store_path={STORE_PATH_ZARR}",
            "--chunk_size=256",
            "--preload_nchunks=32",
            "--use_torch_loader=False",
            "--preload_to_gpu=True",
            f"--n_samples={n_samples}"
        ]
    elif loader_type == "scdataset":
        cmd = [
            "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/scDataset/benchmark_scDataset.py",
            f"--store_path={STORE_PATH_H5AD}",
            "--num_workers=6",
            "--batch_size=4096",
            "--block_size=4",
            "--fetch_factor=2",
            f"--n_samples={n_samples}"
        ]
    elif loader_type == "mapped_collection":
        cmd = [
            "python", f"{REPO_PATH}/src/benchmark_loading/arrayloader_benchmarks/mapped_collection/benchmark_mapped_collection.py",
            f"--store_path={STORE_PATH_H5AD}",
            "--num_workers=8",
            f"--n_samples={n_samples}"
        ]
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed_time = time.time() - start_time

        # Parse JSON output from the benchmark script
        output = json.loads(result.stdout.split("\n")[-2].rstrip())

        return {
            "process_id": process_id,
            "loader": output["loader"],
            "samples_per_sec": output["samples/sec"],
            "total_time": output["total_time"],
            "elapsed_time": elapsed_time,
            "success": True
        }
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"[Process {process_id}] Error running benchmark:")
        print(f"Exit code: {e.returncode}")
        print(f"STDERR:\n{e.stderr}")
        return {
            "process_id": process_id,
            "loader": loader_type,
            "samples_per_sec": 0,
            "total_time": 0,
            "elapsed_time": elapsed_time,
            "success": False
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[Process {process_id}] Unexpected error: {e}")
        return {
            "process_id": process_id,
            "loader": loader_type,
            "samples_per_sec": 0,
            "total_time": 0,
            "elapsed_time": elapsed_time,
            "success": False
        }


def benchmark_parallel_loading(loader_type, num_processes, n_samples_per_process):
    """
    Run benchmarks in parallel across multiple processes.

    Args:
        loader_type: Type of data loader to benchmark
        num_processes: Number of parallel processes to run
        n_samples_per_process: Number of samples to load per process

    Returns:
        Dictionary with aggregated results
    """
    print(f"\n{'='*80}")
    print(f"Running {loader_type} with {num_processes} parallel process(es)...")
    print(f"{'='*80}")

    # Prepare arguments for each process
    args_list = [(i, loader_type, n_samples_per_process) for i in range(num_processes)]

    # Run benchmarks in parallel
    overall_start = time.time()

    if num_processes == 1:
        # Run single process without multiprocessing overhead
        results = [run_single_benchmark_process(args_list[0])]
    else:
        # Use multiprocessing for parallel execution
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_single_benchmark_process, args_list)

    overall_elapsed = time.time() - overall_start

    # Aggregate results
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print(f"All processes failed for {loader_type} with {num_processes} process(es)")
        return {
            "loader": loader_type,
            "num_processes": num_processes,
            "cumulative_samples_per_sec": 0,
            "avg_samples_per_sec_per_process": 0,
            "total_samples": 0,
            "overall_time": overall_elapsed,
            "num_successful": 0,
            "num_failed": len(results)
        }

    # Calculate cumulative throughput
    cumulative_samples_per_sec = sum(r["samples_per_sec"] for r in successful_results)
    avg_samples_per_sec = cumulative_samples_per_sec / len(successful_results)
    total_samples = n_samples_per_process * len(successful_results)

    print(f"\nResults for {num_processes} process(es):")
    print(f"  Successful processes: {len(successful_results)}/{len(results)}")
    print(f"  Cumulative samples/sec: {cumulative_samples_per_sec:.2f}")
    print(f"  Average samples/sec per process: {avg_samples_per_sec:.2f}")
    print(f"  Total samples loaded: {total_samples:,}")
    print(f"  Overall elapsed time: {overall_elapsed:.2f}s")

    return {
        "loader": loader_type,
        "num_processes": num_processes,
        "cumulative_samples_per_sec": cumulative_samples_per_sec,
        "avg_samples_per_sec_per_process": avg_samples_per_sec,
        "total_samples": total_samples,
        "overall_time": overall_elapsed,
        "num_successful": len(successful_results),
        "num_failed": len(results) - len(successful_results),
        "individual_results": successful_results
    }


def main():
    """Main function to run parallel benchmarks."""
    print("Parallel Data Loading Benchmark")
    print("=" * 80)
    print(f"Repository path: {REPO_PATH}")
    print(f"Zarr store path: {STORE_PATH_ZARR}")
    print(f"H5AD store path: {STORE_PATH_H5AD}")
    print(f"Samples per process: {N_SAMPLES_PER_PROCESS:,}")
    print("=" * 80)

    # Install dependencies
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO_PATH], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lamindb"], check=True)

    # Define loader types to benchmark
    # You can uncomment other loaders as needed
    loader_types = [
        "annbatch",
        # "annbatch_gpu",
        # "scdataset",
        "mapped_collection"
    ]

    all_results = []

    # Number of repetitions for each benchmark configuration
    num_repetitions = 5

    # Run benchmarks with repetitions as the outermost loop
    for repetition in range(1, num_repetitions + 1):
        print(f"\n\n{'='*80}")
        print(f"REPETITION {repetition}/{num_repetitions}")
        print(f"{'='*80}")

        for loader_type in loader_types:
            print(f"\n\n{'#'*80}")
            print(f"# Benchmarking {loader_type.upper()}")
            print(f"{'#'*80}")

            for num_processes in range(1, 9):
                result = benchmark_parallel_loading(
                    loader_type=loader_type,
                    num_processes=num_processes,
                    n_samples_per_process=N_SAMPLES_PER_PROCESS
                )
                result["repetition"] = repetition
                all_results.append(result)

                # Small delay between runs to avoid resource conflicts
                time.sleep(2)

    # Save results to CSV
    if all_results:
        # Create a DataFrame with the main results
        df = pd.DataFrame([
            {
                "loader": r["loader"],
                "num_processes": r["num_processes"],
                "repetition": r["repetition"],
                "cumulative_samples_per_sec": r["cumulative_samples_per_sec"],
                "avg_samples_per_sec_per_process": r["avg_samples_per_sec_per_process"],
                "total_samples": r["total_samples"],
                "overall_time": r["overall_time"],
                "num_successful": r["num_successful"],
                "num_failed": r["num_failed"]
            }
            for r in all_results
        ])

        output_file = f"{REPO_PATH}/src/benchmark_loading/parallel_loading_benchmark_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}")

        # Print summary table
        print("\nSummary:")
        print(df.to_string(index=False))
    else:
        print("\nNo successful benchmarks to save")


if __name__ == "__main__":
    main()

