from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from _common import configure_zarr, load_config, provenance, write_result

N_OBS_PER_CHUNK = 64
SHARD_SIZE = 32_768
DATASET_SIZE = 262_144
SHUFFLE_CHUNK_SIZE = 128


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="directory holding the *.h5sc files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-output", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()

    import anndata as ad
    import h5py
    import numpy as np
    import scipy.sparse as sp
    import zarr
    from annbatch import DatasetCollection

    in_dir = Path(args.input_dir)
    paths = sorted(in_dir.glob("*.h5sc"))
    if not paths:
        raise SystemExit(f"no *.h5sc files in {in_dir}")
    out = Path(args.out or (in_dir.parent / "scportrait_preshuffled.zarr"))
    if out.exists():
        shutil.rmtree(out)

    def load_adata(path):
        """The notebook's loader: keep obs/var, lazy obsm, placeholder X."""
        f = h5py.File(path, "r")
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
        obsm = {
            k: ad.experimental.read_elem_lazy(
                f["obsm"][k], chunks=(128,) + (-1,) * (f["obsm"][k].ndim - 1))
            for k in f["obsm"]
        }
        uns = ad.io.read_elem(f["uns"]) if "uns" in f else {}
        return ad.AnnData(X=sp.csr_matrix((len(obs), len(var))), obs=obs, var=var,
                          obsm=obsm, uns=uns)

    total_in = sum(p.stat().st_size for p in paths)
    print(f"[microscopy] {len(paths)} inputs, {total_in / 1024**3:.1f} GiB: "
          + ", ".join(p.name for p in paths), flush=True)
    print(f"[microscopy] layout n_obs_per_chunk={N_OBS_PER_CHUNK} shard_size={SHARD_SIZE} "
          f"dataset_size={DATASET_SIZE} shuffle_chunk_size={SHUFFLE_CHUNK_SIZE}", flush=True)

    t0 = time.perf_counter()
    DatasetCollection(zarr.open(str(out), mode="w")).add_adatas(
        adata_paths=[str(p) for p in paths],
        load_adata=load_adata,
        shuffle=True,
        n_obs_per_chunk=N_OBS_PER_CHUNK,
        shard_size=SHARD_SIZE,
        dataset_size=DATASET_SIZE,
        shuffle_chunk_size=SHUFFLE_CHUNK_SIZE,
        rng=np.random.default_rng(args.seed),
    )
    elapsed = time.perf_counter() - t0
    out_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[microscopy] pre-shuffled in {elapsed / 3600:.3f} h ({elapsed:.1f} s); "
          f"output {out_bytes / 1024**3:.1f} GiB", flush=True)

    write_result(
        Path(cfg["results"]) / "08_microscopy_preshuffle.json",
        {
            "experiment": "microscopy_preshuffle_timing",
            "inputs": [str(p) for p in paths],
            "input_bytes": total_in,
            "output_path": str(out),
            "output_bytes": out_bytes,
            "layout": {"n_obs_per_chunk": N_OBS_PER_CHUNK, "shard_size": SHARD_SIZE,
                       "dataset_size": DATASET_SIZE, "shuffle_chunk_size": SHUFFLE_CHUNK_SIZE},
            "elapsed_s": elapsed,
            "elapsed_h": elapsed / 3600,
            "provenance": provenance(cfg),
        },
    )
    if not args.keep_output:
        shutil.rmtree(out)
        print(f"[microscopy] removed {out}", flush=True)


if __name__ == "__main__":
    main()
