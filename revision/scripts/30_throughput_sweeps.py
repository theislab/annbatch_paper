from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    configure_zarr,
    load_config,
    measure_throughput,
    open_preshuffled_datasets,
    provenance,
    write_result,
)

CHUNK_SIZES = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
FIXED_BUFFER_ROWS = 16_384
DEFAULT_PRELOAD_NCHUNKS = 32
SCDATASET_BLOCK_SIZES = (1, 16, 64, 256, 512, 1024, 2048, 4096)
# fetch_factor set so batch_size x fetch_factor == FIXED_BUFFER_ROWS; their 256 needs ~9 GB/fetch.
DATASET_COUNTS = (1, 2, 4, 8, 16, 32, -1)   # -1 = all


def warm_page_cache(datasets, byte_budget: int) -> int:
    """Stream a subset's chunk files into the page cache, newest-first, up to a budget."""
    read = 0
    paths: list[Path] = []
    for d in datasets:
        group = getattr(d, "group", None)
        if group is None:  # pragma: no cover - depends on the backing type
            continue
        root = getattr(getattr(group, "store", None), "root", None)
        if root is None:  # pragma: no cover
            continue
        # Walk this dataset's X, not the collection root: that would warm all 298 GB.
        paths.append(Path(root) / str(group.path))
    for root in paths:
        if not root.exists():  # pragma: no cover
            continue
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            try:
                with f.open("rb", buffering=0) as fh:
                    while chunk := fh.read(8 << 20):
                        read += len(chunk)
                        if read >= byte_budget:
                            return read
            except OSError:  # pragma: no cover
                continue
    return read


def _annbatch_iter(datasets, *, chunk_size, preload_nchunks, batch_size, seed, to_gpu):
    from annbatch import Loader
    from annbatch.samplers import RandomSampler

    sampler = RandomSampler(
        chunk_size=chunk_size, preload_nchunks=preload_nchunks,
        batch_size=batch_size, drop_last=True, rng=np.random.default_rng(seed),
    )
    loader = Loader(batch_sampler=sampler, preload_to_gpu=to_gpu, to="torch" if to_gpu else None)
    loader.add_datasets(datasets)
    return iter(loader)


def _scdataset_iter(datasets, n_obs, *, block_size, fetch_factor, batch_size, num_workers, seed):
    import torch
    from scdataset import BlockShuffling, scDataset
    from torch.utils.data import DataLoader

    from _strategies import _ConcatCSR

    view = _ConcatCSR(datasets, n_obs)
    ds = scDataset(
        view,
        BlockShuffling(block_size=block_size),
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_callback=lambda c, i: c[np.asarray(i)],
    )
    loader = DataLoader(
        ds, batch_size=None, num_workers=num_workers,
        prefetch_factor=2 if num_workers else None,
        generator=torch.Generator().manual_seed(seed),
    )
    return iter(loader)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, choices=["chunk_size", "dataset_size"])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=2_000_000)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--to-gpu", action="store_true", help="annbatch arms: preload_to_gpu (needs cupy)")
    ap.add_argument("--skip-scdataset", action="store_true")
    ap.add_argument("--skip-annbatch", action="store_true")
    ap.add_argument("--warm-cache", action="store_true",
                    help="stream the subset into the page cache before measuring")
    ap.add_argument("--cache-bytes", type=int, default=48 * 1024 ** 3,
                    help="byte budget for the warm-up; keep below the cgroup limit")
    ap.add_argument("--scdataset-block-sizes", nargs="+", type=int, default=None,
                    help="override the scDataset block-size grid (block_size=1 is "
                         "uniform random access and takes hours; annbatch chunk_size=1 "
                         "already provides that reference point)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()
    all_ds, all_n = open_preshuffled_datasets(cfg)
    print(f"[sweep] pre-shuffled collection: {len(all_ds)} datasets, {sum(all_n):,d} obs", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    if args.warm_cache and not args.tag:
        tag = "_warm"
    out_path = Path(cfg["results"]) / f"30_throughput_{args.sweep}{tag}.json"
    rows: list[dict] = []

    def checkpoint() -> None:
        """Write after every measurement: a sweep that dies late must not lose the rest."""
        write_result(out_path, {
            "experiment": f"throughput_{args.sweep}",
            "store": cfg["tahoe"]["preshuffled"],
            "n_datasets_total": len(all_ds),
            "n_obs_total": int(sum(all_n)),
            "batch_size": args.batch_size,
            "n_samples_per_measurement": args.n_samples,
            "warm_cache": args.warm_cache,
            "cache_bytes": args.cache_bytes if args.warm_cache else None,
            "complete": False,
            "results": rows,
            "provenance": provenance(cfg),
        })

    if args.sweep == "chunk_size":
        for regime in () if args.skip_annbatch else ("fixed_buffer", "fixed_nchunks"):
            for k in CHUNK_SIZES:
                nch = max(1, FIXED_BUFFER_ROWS // k) if regime == "fixed_buffer" else DEFAULT_PRELOAD_NCHUNKS
                if k * nch < args.batch_size:
                    print(f"[sweep] skip chunk_size={k} ({regime}): buffer {k * nch} < batch {args.batch_size}",
                          flush=True)
                    continue
                for rep in range(args.repeats):
                    it = _annbatch_iter(all_ds, chunk_size=k, preload_nchunks=nch,
                                        batch_size=args.batch_size, seed=args.seed + rep,
                                        to_gpu=args.to_gpu)
                    res = measure_throughput(
                        it, n_samples=args.n_samples, batch_size=args.batch_size,
                        loader="annbatch", extract=lambda b: b["X"],
                        params={"regime": regime, "chunk_size": k, "preload_nchunks": nch,
                                "buffer_rows": k * nch, "preload_to_gpu": args.to_gpu, "repeat": rep},
                    )
                    print(f"[sweep] annbatch {regime:14s} chunk={k:>5d} nchunks={nch:>5d} "
                          f"-> {res.samples_per_sec:10.1f} samples/s", flush=True)
                    rows.append(res.as_dict())
                    checkpoint()

        if not args.skip_scdataset:
            fetch_factor = max(1, FIXED_BUFFER_ROWS // args.batch_size)
            for b in (args.scdataset_block_sizes or SCDATASET_BLOCK_SIZES):
                for rep in range(args.repeats):
                    it = _scdataset_iter(all_ds, all_n, block_size=b, fetch_factor=fetch_factor,
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                         seed=args.seed + rep)
                    res = measure_throughput(
                        it, n_samples=args.n_samples, batch_size=args.batch_size, loader="scDataset",
                        params={"regime": "fixed_buffer", "block_size": b, "fetch_factor": fetch_factor,
                                "buffer_rows": args.batch_size * fetch_factor,
                                "num_workers": args.num_workers, "repeat": rep},
                    )
                    print(f"[sweep] scDataset  fixed_buffer   block={b:>5d} fetch={fetch_factor} "
                          f"-> {res.samples_per_sec:10.1f} samples/s", flush=True)
                    rows.append(res.as_dict())
                    checkpoint()

    else:  # dataset_size
        for count in DATASET_COUNTS:
            k = len(all_ds) if count == -1 else count
            if k > len(all_ds):
                continue
            ds, n = all_ds[:k], all_n[:k]
            n_obs = sum(n)
            n_samples = min(args.n_samples, (n_obs // args.batch_size) * args.batch_size)
            if args.warm_cache:
                warmed = warm_page_cache(ds, args.cache_bytes)
                print(f"[sweep] warmed page cache with {warmed / 1024**3:.1f} GiB "
                      f"for n_obs={n_obs:,d}", flush=True)
            for rep in range(args.repeats):
                it = _annbatch_iter(ds, chunk_size=512, preload_nchunks=DEFAULT_PRELOAD_NCHUNKS,
                                    batch_size=args.batch_size, seed=args.seed + rep, to_gpu=args.to_gpu)
                res = measure_throughput(it, n_samples=n_samples, batch_size=args.batch_size,
                                         loader="annbatch (chunk_size=512)", extract=lambda b: b["X"],
                                         params={"n_datasets": k, "n_obs": n_obs, "repeat": rep})
                print(f"[sweep] annbatch      n_obs={n_obs:>12,d} -> {res.samples_per_sec:10.1f} samples/s", flush=True)
                rows.append(res.as_dict())
                checkpoint()

                it = _annbatch_iter(ds, chunk_size=1, preload_nchunks=FIXED_BUFFER_ROWS,
                                    batch_size=args.batch_size, seed=args.seed + rep, to_gpu=args.to_gpu)
                res = measure_throughput(it, n_samples=n_samples, batch_size=args.batch_size,
                                         loader="uniform random access (chunk_size=1)",
                                         extract=lambda b: b["X"],
                                         params={"n_datasets": k, "n_obs": n_obs, "repeat": rep,
                                                 "note": "the access pattern of MappedCollection / BioNeMo-SCDL"})
                print(f"[sweep] random access n_obs={n_obs:>12,d} -> {res.samples_per_sec:10.1f} samples/s", flush=True)
                rows.append(res.as_dict())
                checkpoint()

                if not args.skip_scdataset:
                    fetch_factor = max(1, FIXED_BUFFER_ROWS // args.batch_size)
                    it = _scdataset_iter(ds, n, block_size=16, fetch_factor=fetch_factor,
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                         seed=args.seed + rep)
                    res = measure_throughput(it, n_samples=n_samples, batch_size=args.batch_size,
                                             loader=f"scDataset (block=16, fetch={fetch_factor})",
                                             params={"n_datasets": k, "n_obs": n_obs, "repeat": rep,
                                                     "buffer_rows": args.batch_size * fetch_factor})
                    print(f"[sweep] scDataset     n_obs={n_obs:>12,d} -> {res.samples_per_sec:10.1f} samples/s",
                          flush=True)
                    rows.append(res.as_dict())
                    checkpoint()

    write_result(out_path, {
        "experiment": f"throughput_{args.sweep}",
        "store": cfg["tahoe"]["preshuffled"],
        "n_datasets_total": len(all_ds),
        "n_obs_total": int(sum(all_n)),
        "batch_size": args.batch_size,
        "n_samples_per_measurement": args.n_samples,
        "warm_cache": args.warm_cache,
        "cache_bytes": args.cache_bytes if args.warm_cache else None,
        "complete": True,
        "results": rows,
        "provenance": provenance(cfg),
    })


if __name__ == "__main__":
    main()
