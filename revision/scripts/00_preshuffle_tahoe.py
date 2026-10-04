from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import configure_zarr, load_config, plate_paths, provenance, write_result

# --- on-disk layout (see module docstring) ---------------------------------- #
MEAN_NNZ_PER_ROW = 1440          # measured over all 14 Tahoe plates
N_OBS_PER_CHUNK = 24             # ~34,560 nnz per Zarr chunk  (manuscript: 32,768)
SHARD_SIZE_OBS = 96_000          # ~1.38e8 nnz per Zarr shard  (manuscript: 1.34e8)
DATASET_SIZE_OBS = 2 ** 21       # 2,097,152 obs per dataset   (manuscript: 2**21)
SHUFFLE_CHUNK_SIZE = 1_000       # contiguous rows read per shuffle read (annbatch default)

# obs kept in the store; `global_row` keeps the permutation recoverable.
OBS_COLUMNS = ["cell_line", "drug", "sample"]
GLOBAL_ROW_COLUMN = "global_row"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output collection path (default: config tahoe.preshuffled)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--which", default="train", choices=["train", "test", "all"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    configure_zarr()

    import anndata as ad
    import zarr
    from annbatch import DatasetCollection

    out = Path(args.out or cfg["tahoe"]["preshuffled"])
    paths = plate_paths(cfg, args.which)

    # Row offset of each input store in the unshuffled concatenation.
    offsets: dict[str, int] = {}
    running = 0
    for q in paths:
        offsets[str(q)] = running
        running += int(zarr.open(str(q), mode="r")["X"].attrs["shape"][0])
    n_obs_total = running
    print(f"[preshuffle] total input observations: {n_obs_total:,d}", flush=True)

    def load_adata(path) -> ad.AnnData:
        """Keep only X, the obs columns the tasks need, and the row provenance."""
        adata = ad.experimental.read_lazy(path)
        obs = adata.obs[OBS_COLUMNS].to_memory()
        start = offsets[str(path)]
        obs[GLOBAL_ROW_COLUMN] = np.arange(start, start + obs.shape[0], dtype=np.int64)
        return ad.AnnData(X=adata.X, obs=obs, var=adata.var.to_memory())

    print(f"[preshuffle] inputs ({len(paths)}):", flush=True)
    for q in paths:
        print(f"  {q}", flush=True)
    print(f"[preshuffle] output: {out}", flush=True)
    print(
        f"[preshuffle] layout: n_obs_per_chunk={N_OBS_PER_CHUNK} "
        f"shard_size={SHARD_SIZE_OBS} dataset_size={DATASET_SIZE_OBS} "
        f"shuffle_chunk_size={SHUFFLE_CHUNK_SIZE} seed={args.seed}",
        flush=True,
    )
    if args.dry_run:
        return

    if out.exists():
        raise FileExistsError(f"{out} already exists -- refusing to overwrite a 300 GB store")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    collection = DatasetCollection(zarr.open(str(out), mode="w"))
    collection.add_adatas(
        adata_paths=[str(q) for q in paths],
        load_adata=load_adata,
        shuffle=True,
        n_obs_per_chunk=N_OBS_PER_CHUNK,
        shard_size=SHARD_SIZE_OBS,
        dataset_size=DATASET_SIZE_OBS,
        shuffle_chunk_size=SHUFFLE_CHUNK_SIZE,
        rng=np.random.default_rng(args.seed),
    )
    elapsed = time.perf_counter() - t0

    print(f"[preshuffle] done in {elapsed / 3600:.3f} h ({elapsed:.1f} s)", flush=True)

    write_result(
        Path(cfg["results"]) / "00_preshuffle_tahoe.json",
        {
            "experiment": "preshuffle_tahoe",
            "which_plates": args.which,
            "n_input_stores": len(paths),
            "n_obs_total": n_obs_total,
            "input_paths": [str(q) for q in paths],
            "output_path": str(out),
            "seed": args.seed,
            "layout": {
                "n_obs_per_chunk": N_OBS_PER_CHUNK,
                "shard_size_obs": SHARD_SIZE_OBS,
                "dataset_size_obs": DATASET_SIZE_OBS,
                "shuffle_chunk_size": SHUFFLE_CHUNK_SIZE,
                "mean_nnz_per_row": MEAN_NNZ_PER_ROW,
                "obs_columns": [*OBS_COLUMNS, GLOBAL_ROW_COLUMN],
            },
            "elapsed_s": elapsed,
            "elapsed_h": elapsed / 3600,
            "provenance": provenance(cfg),
        },
    )


if __name__ == "__main__":
    main()
