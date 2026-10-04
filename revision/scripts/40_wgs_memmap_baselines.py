from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import configure_zarr, load_config, measure_throughput, provenance, write_result

MODES = ("random_row", "random_row_manuscript", "block_shuffled", "sequential", "annbatch")

# Random single-row access is ~4 orders slower than sequential, so it gets a smaller budget.
SLOW_MODES = frozenset({"random_row", "random_row_manuscript"})


# memmap loaders
def _open_memmap(path: Path):
    with path.with_suffix(".meta.json").open() as f:
        meta = json.load(f)
    shape = tuple(meta["shape"])
    mm = np.memmap(path, dtype=meta["dtype"], mode="r", shape=shape)
    return mm, shape, meta


class _RowDataset:
    """One row per __getitem__ -- the manuscript's memmap baseline."""

    def __init__(self, path: Path):
        self.path = path
        self.mm, self.shape, _ = _open_memmap(path)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        import torch

        return torch.from_numpy(self.mm[idx].astype(np.float32).copy())


def iter_random_row_manuscript(path: Path, *, batch_size: int, num_workers: int, pin_memory: bool):
    """The manuscript's baseline, unmodified."""
    from torch.utils.data import DataLoader

    ds = _RowDataset(path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    ), len(ds)


def iter_random_row(path: Path, *, batch_size: int, seed: int):
    """The same access pattern, in-process, without the float32 IPC round trip."""
    import torch

    mm, shape, _ = _open_memmap(path)
    n_obs = shape[0]
    rng = np.random.default_rng(seed)

    def gen():
        order = rng.permutation(n_obs)
        for s in range(0, n_obs - batch_size + 1, batch_size):
            rows = order[s : s + batch_size]
            buf = np.stack([mm[int(r)] for r in rows])
            yield torch.from_numpy(buf.astype(np.float32))

    return gen(), n_obs


def iter_block_shuffled(path: Path, *, batch_size: int, chunk_size: int, preload_nchunks: int, seed: int):
    """Random contiguous blocks -> in-memory shuffle -> mini-batches."""
    import torch

    mm, shape, _ = _open_memmap(path)
    n_obs = shape[0]
    rng = np.random.default_rng(seed)

    def gen():
        starts = np.arange(0, n_obs, chunk_size)
        rng.shuffle(starts)
        for i in range(0, len(starts), preload_nchunks):
            block_starts = starts[i : i + preload_nchunks]
            buf = np.concatenate([mm[s : min(s + chunk_size, n_obs)] for s in block_starts])
            perm = rng.permutation(len(buf))
            for j in range(0, len(buf) - batch_size + 1, batch_size):
                yield torch.from_numpy(buf[perm[j : j + batch_size]].astype(np.float32))

    return gen(), n_obs


def iter_sequential(path: Path, *, batch_size: int):
    import torch

    mm, shape, _ = _open_memmap(path)
    n_obs = shape[0]

    def gen():
        for s in range(0, n_obs - batch_size + 1, batch_size):
            yield torch.from_numpy(mm[s : s + batch_size].astype(np.float32))

    return gen(), n_obs


def iter_annbatch(store_path: str, *, batch_size: int, chunk_size: int, preload_nchunks: int):
    import zarr
    from annbatch import DatasetCollection, Loader

    collection = DatasetCollection(zarr.open(store_path, mode="r"))
    loader = Loader(
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        shuffle=True,
        preload_to_gpu=False,
        to=None,
    )
    loader.use_collection(collection)
    return loader, loader.n_obs


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="all", choices=[*MODES, "all"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--preload-nchunks", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--n-samples-slow", type=int, default=12_800,
                   help="sample budget for the uniform-random-access arms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    cfg = load_config()
    # memmap and Zarr are compared directly, so both must be on the SSD pool.
    print(f"[memmap] storage pools: "
          f"{assert_ssd(cfg, args.memmap_path if getattr(args, 'memmap_path', None) else cfg['wgs']['common']['memmap'], cfg['wgs']['common']['preshuffled_zarr'])}",
          flush=True)
    configure_zarr()

    memmap_path = Path(cfg["wgs"]["common"]["memmap"])
    zarr_path = cfg["wgs"]["common"]["preshuffled_zarr"]
    modes = MODES if args.mode == "all" else (args.mode,)

    rows = []
    for mode in modes:
        for rep in range(args.repeats):
            n_samples = args.n_samples_slow if mode in SLOW_MODES else args.n_samples
            if mode == "random_row":
                it, n_obs = iter_random_row(memmap_path, batch_size=args.batch_size, seed=args.seed + rep)
                params = {"num_workers": 0,
                          "access": "uniform random single-row reads, in-process, uint8 until batched"}
            elif mode == "random_row_manuscript":
                it, n_obs = iter_random_row_manuscript(
                    memmap_path, batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=True,
                )
                params = {"num_workers": args.num_workers, "pin_memory": True, "shuffle": True,
                          "access": "single row per __getitem__ cast to float32, uniform random order",
                          "note": "the manuscript's Fig. 2e baseline, verbatim"}
            elif mode == "block_shuffled":
                it, n_obs = iter_block_shuffled(
                    memmap_path, batch_size=args.batch_size, chunk_size=args.chunk_size,
                    preload_nchunks=args.preload_nchunks, seed=args.seed + rep,
                )
                params = {"chunk_size": args.chunk_size, "preload_nchunks": args.preload_nchunks,
                          "access": "random contiguous blocks, shuffled in memory"}
            elif mode == "sequential":
                it, n_obs = iter_sequential(memmap_path, batch_size=args.batch_size)
                params = {"access": "on-disk order, no randomisation (upper bound only)"}
            elif mode == "annbatch":
                it, n_obs = iter_annbatch(
                    zarr_path, batch_size=args.batch_size, chunk_size=args.chunk_size,
                    preload_nchunks=args.preload_nchunks,
                )
                params = {"chunk_size": args.chunk_size, "preload_nchunks": args.preload_nchunks,
                          "store": zarr_path,
                          "access": "random contiguous chunks from pre-shuffled Zarr, shuffled in memory",
                          "includes_preshuffling": False}
            else:  # pragma: no cover
                raise ValueError(mode)

            extract = (lambda b: b["X"]) if mode == "annbatch" else None
            res = measure_throughput(
                iter(it), n_samples=n_samples, batch_size=args.batch_size,
                loader=mode, params={**params, "n_obs": n_obs, "repeat": rep}, extract=extract,
            )
            print(f"[wgs-memmap] {mode:15s} rep {rep}: {res.samples_per_sec:10.1f} samples/s "
                  f"({res.elapsed_s:.1f} s for {res.n_samples:,d})", flush=True)
            rows.append(res.as_dict())

    tag = f"_{args.tag}" if args.tag else ""
    write_result(
        Path(cfg["results"]) / f"40_wgs_memmap_baselines{tag}.json",
        {
            "experiment": "wgs_memmap_baselines",
            "dataset": "1000 Genomes GRCh38, common variants MAF >= 1%, 1.2M SNPs, replicated to 500,000 individuals",
            "memmap_path": str(memmap_path),
            "zarr_path": zarr_path,
            "batch_size": args.batch_size,
            "n_samples": args.n_samples,
            "n_samples_slow": args.n_samples_slow,
            "results": rows,
            "provenance": provenance(cfg),
        },
    )


if __name__ == "__main__":
    main()
