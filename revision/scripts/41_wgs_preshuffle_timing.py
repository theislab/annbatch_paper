from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from _common import configure_zarr, load_config, provenance, write_result

# Layout of the data-prep pipeline in the annbatch >= 0.2 API; nnz measured from the store.
REGIMES = {
    "common": dict(
        sparse=False,
        n_obs_per_chunk=4,          # zarr_dense_chunk_size=4
        shard_size=512,             # zarr_dense_shard_size=512
        dataset_size=16_384,        # n_obs_per_dataset=16384
        shuffle_chunk_size=64,
        mean_nnz_per_row=None,
        n_var=1_200_000,
    ),
    "rare_maf_0.01": dict(
        sparse=True,
        chunk_elems=2 ** 23,        # zarr_sparse_chunk_size=2**23
        shard_elems=2 ** 19 * 512,  # zarr_sparse_shard_size=2**19 * 512
        dataset_size=2 ** 17,       # n_obs_per_dataset=2**17
        shuffle_chunk_size=128,
        mean_nnz_per_row=223_814,
        n_var=92_087_087,
    ),
    "rare_maf_0.001": dict(
        sparse=True,
        chunk_elems=2 ** 21,        # zarr_sparse_chunk_size=2**21
        shard_elems=2 ** 21 * 512,  # zarr_sparse_shard_size=2**21 * 512
        dataset_size=2 ** 17,
        shuffle_chunk_size=128,
        mean_nnz_per_row=53_738,
        n_var=74_589_081,
    ),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", required=True, choices=sorted(REGIMES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-output", action="store_true",
                   help="keep the re-written store (default: delete it, it is a duplicate)")
    args = p.parse_args()

    cfg = load_config()
    configure_zarr()

    import zarr
    from annbatch import DatasetCollection

    spec = REGIMES[args.regime]
    src_h5ad = Path(cfg["wgs"][args.regime]["h5ad"])
    if not src_h5ad.exists():
        raise FileNotFoundError(src_h5ad)

    scratch = Path(cfg["wgs"]["timing_scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"{args.regime}_timed.zarr"
    if out.exists():
        shutil.rmtree(out)

    if spec["sparse"]:
        nnz = spec["mean_nnz_per_row"]
        n_obs_per_chunk = max(1, round(spec["chunk_elems"] / nnz))
        shard_size = max(n_obs_per_chunk, round(spec["shard_elems"] / nnz))
    else:
        n_obs_per_chunk = spec["n_obs_per_chunk"]
        shard_size = spec["shard_size"]

    input_bytes = src_h5ad.stat().st_size
    print(f"[wgs-preshuffle] regime={args.regime} src={src_h5ad} ({input_bytes / 1024**3:.1f} GiB)", flush=True)
    print(f"[wgs-preshuffle] n_obs_per_chunk={n_obs_per_chunk} shard_size={shard_size} "
          f"dataset_size={spec['dataset_size']} shuffle_chunk_size={spec['shuffle_chunk_size']}", flush=True)

    t0 = time.perf_counter()
    DatasetCollection(zarr.open(str(out), mode="w")).add_adatas(
        adata_paths=[str(src_h5ad)],
        shuffle=True,
        n_obs_per_chunk=n_obs_per_chunk,
        shard_size=shard_size,
        dataset_size=spec["dataset_size"],
        shuffle_chunk_size=spec["shuffle_chunk_size"],
        rng=np.random.default_rng(args.seed),
    )
    elapsed = time.perf_counter() - t0
    print(f"[wgs-preshuffle] {args.regime}: {elapsed / 3600:.3f} h ({elapsed:.1f} s)", flush=True)

    out_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

    write_result(
        Path(cfg["results"]) / f"41_wgs_preshuffle_{args.regime}.json",
        {
            "experiment": "wgs_preshuffle_timing",
            "regime": args.regime,
            "n_var": spec["n_var"],
            "sparse": spec["sparse"],
            "input_h5ad": str(src_h5ad),
            "input_bytes": input_bytes,
            "output_bytes": out_bytes,
            "layout": {
                "n_obs_per_chunk": n_obs_per_chunk,
                "shard_size": shard_size,
                "dataset_size": spec["dataset_size"],
                "shuffle_chunk_size": spec["shuffle_chunk_size"],
                "mean_nnz_per_row": spec["mean_nnz_per_row"],
            },
            "elapsed_s": elapsed,
            "elapsed_h": elapsed / 3600,
            "provenance": provenance(cfg),
        },
    )

    if not args.keep_output:
        shutil.rmtree(out)
        print(f"[wgs-preshuffle] removed {out}", flush=True)


if __name__ == "__main__":
    main()
