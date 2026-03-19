from __future__ import annotations

import csv
import time
from pathlib import Path

INPUT_PATH_H5AD = Path("/vol/data/annbatch_benchmark/tahoe100M/raw_h5ad")
INPUT_PATH_ZARR = Path("/vol/data/annbatch_benchmark/tahoe100M/raw_zarr")
OUTPUT_PATH_ZARR = Path("/vol/data/annbatch_benchmark/tahoe100M/zarr_shuffled")
OUTPUT_PATH_H5AD = Path("/vol/data/annbatch_benchmark/tahoe100M/h5ad_shuffled")
OUTPUT_PATH_ZARR_TO_ZARR = Path(
    "/vol/data/annbatch_benchmark/tahoe100M/zarr_to_zarr_shuffled"
)
CSV_PATH = Path("store_creation_time.csv")
BENCHMARKS_TO_RUN = ("h5ad-to-h5ad", "h5ad-to-zarr", "zarr-to-zarr")

BENCHMARK_FORMATS = {
    "h5ad-to-h5ad": "h5ad -> h5ad",
    "h5ad-to-zarr": "h5ad -> zarr",
    "zarr-to-zarr": "zarr -> zarr",
}


def require_existing_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def require_missing_output(path: Path, description: str) -> None:
    if path.exists():
        raise FileExistsError(
            f"{description} already exists: {path}. Remove it or choose a different output path."
        )


def collect_input_paths(input_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {input_dir}")
    return paths


def run_store_creation(
    output_path: Path,
    input_paths: list[Path],
    *,
    is_collection_h5ad: bool = False,
) -> float:
    import zarr
    import zarrs  # noqa: F401
    from annbatch import DatasetCollection

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

    start_time = time.time()
    DatasetCollection(
        output_path,
        mode="w",
        is_collection_h5ad=is_collection_h5ad,
    ).add_adatas(
        input_paths,
        dataset_size=2_097_152,
        n_obs_per_chunk=32,
    )
    return time.time() - start_time


def main() -> None:
    benchmark_specs = {
        "h5ad-to-h5ad": {
            "input_dir": INPUT_PATH_H5AD,
            "pattern": "*.h5ad",
            "output_path": OUTPUT_PATH_H5AD,
            "is_collection_h5ad": True,
        },
        "h5ad-to-zarr": {
            "input_dir": INPUT_PATH_H5AD,
            "pattern": "*.h5ad",
            "output_path": OUTPUT_PATH_ZARR,
            "is_collection_h5ad": False,
        },
        "zarr-to-zarr": {
            "input_dir": INPUT_PATH_ZARR,
            "pattern": "*.zarr",
            "output_path": OUTPUT_PATH_ZARR_TO_ZARR,
            "is_collection_h5ad": False,
        },
    }

    results = []

    for benchmark_name in BENCHMARKS_TO_RUN:
        spec = benchmark_specs[benchmark_name]
        require_existing_directory(spec["input_dir"], f"Input directory for {benchmark_name}")
        require_missing_output(spec["output_path"], f"Output path for {benchmark_name}")
        input_paths = collect_input_paths(spec["input_dir"], spec["pattern"])

        print(
            f"Running {BENCHMARK_FORMATS[benchmark_name]} "
            f"with {len(input_paths)} inputs -> {spec['output_path']}"
        )
        elapsed_seconds = run_store_creation(
            spec["output_path"],
            input_paths,
            is_collection_h5ad=spec["is_collection_h5ad"],
        )
        elapsed_hours = elapsed_seconds / (60.0 * 60.0)
        print(
            f"Completed {BENCHMARK_FORMATS[benchmark_name]} "
            f"in {elapsed_seconds:.2f}s ({elapsed_hours:.4f}h)"
        )

        results.append(
            {
                "benchmark": benchmark_name,
                "format": BENCHMARK_FORMATS[benchmark_name],
                "store_creation_time_seconds": elapsed_seconds,
                "store_creation_time_hours": elapsed_hours,
                "n_input_files": len(input_paths),
                "output_path": str(spec["output_path"]),
            }
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "benchmark",
                "format",
                "store_creation_time_seconds",
                "store_creation_time_hours",
                "n_input_files",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nResults:")
    for result in results:
        print(
            f"{result['format']}: "
            f"{result['store_creation_time_seconds']:.2f}s "
            f"({result['store_creation_time_hours']:.4f}h), "
            f"{result['n_input_files']} inputs, "
            f"output={result['output_path']}"
        )
    print(f"\nSaved results to {CSV_PATH}")


if __name__ == "__main__":
    main()
