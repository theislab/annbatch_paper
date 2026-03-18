from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad
import lamindb as ln


ARTIFACT_UIDS = [
    "aJIqo7bNyJAs9z0r0000",
    "ZFeVfd0ugAHeWCxm0000",
    "XVSrkq9pyF1OBLgG0000",
    "tKTeff0ugWqAm4P70000",
    "EZATJLC4jE7pmwo40000",
    "aAHQ3zbD7n1asyYr0000",
    "DC5cacdJr1VoEXnl0000",
    "czC19UpUEszVH2bU0000",
    "BDttiuV3Te8VB0dU0000",
    "56uA9lPPmJ4zLUcr0000",
    "omn7JStfJMzy8m6O0000",
    "S2h2rPLCaUhZAM9u0000",
    "9L9HZ55HqUL0aqaR0000",
    "vn5cUJCHbjpPPsZx0000",
]

# Set the output path and behavior below.
OUT_PATH_H5AD = Path("/vol/data/annbatch_benchmark/tahoe100M/raw_h5ad")
OUT_PATH_ZARR = Path("/vol/data/annbatch_benchmark/tahoe100M/raw_zarr")
OVERWRITE_EXISTING = False
REMOVE_FROM_CACHE = True
LAMIN_INSTANCE = "laminlabs/arrayloader-benchmarks"


def remove_cached_path(cached_path: Path) -> None:
    if not cached_path.exists():
        return
    if cached_path.is_dir():
        shutil.rmtree(cached_path)
    else:
        cached_path.unlink()


def main() -> None:
    output_dir = OUT_PATH_H5AD.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_output_dir = (
        OUT_PATH_ZARR.expanduser().resolve() if OUT_PATH_ZARR is not None else None
    )
    if zarr_output_dir is not None:
        zarr_output_dir.mkdir(parents=True, exist_ok=True)

    benchmarking_artifacts = ln.Artifact.using(LAMIN_INSTANCE)

    for uid in ARTIFACT_UIDS:
        artifact = benchmarking_artifacts.get(uid)
        destination = output_dir / artifact.key.rsplit("/", 1)[-1]
        cached_path = None

        try:
            if destination.exists() and not OVERWRITE_EXISTING:
                print(f"Skipping existing file: {destination}")
            else:
                cached_path = Path(artifact.cache())
                destination = output_dir / cached_path.name
                shutil.copy2(cached_path, destination)
                print(f"Downloaded {uid} -> {destination}")

            if zarr_output_dir is not None:
                zarr_destination = zarr_output_dir / f"{destination.stem}.zarr"
                if zarr_destination.exists():
                    if OVERWRITE_EXISTING:
                        shutil.rmtree(zarr_destination)
                    else:
                        print(f"Skipping existing zarr: {zarr_destination}")
                        continue

                ad.read_h5ad(destination).write_zarr(zarr_destination)
                print(f"Converted {destination} -> {zarr_destination}")
        finally:
            if REMOVE_FROM_CACHE and cached_path is not None:
                remove_cached_path(cached_path)
                print(f"Removed cached file: {cached_path}")


if __name__ == "__main__":
    main()
