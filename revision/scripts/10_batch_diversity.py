from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import LabelTable, configure_zarr, load_config, plate_paths, provenance, write_result

DIVERSITY_LABELS = ("plate", "drug", "cell_line")


# index streams
def _annbatch_index_stream(sampler, n_obs: int, max_batches: int):
    """Global row indices produced by an annbatch sampler, without touching X."""
    emitted = 0
    for req in sampler.sample(n_obs):
        buf = np.concatenate([np.arange(s.start, s.stop) for s in req["requests"]])
        for split in req["splits"]:
            yield buf[split]
            emitted += 1
            if emitted >= max_batches:
                return


class _IndexOnly:
    """A data collection that returns the requested indices -- no data at all."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return np.asarray(idx)


def _scdataset_index_stream(n_obs: int, *, block_size: int, fetch_factor: int,
                            batch_size: int, seed: int, max_batches: int):
    from scdataset import BlockShuffling, scDataset

    ds = scDataset(
        _IndexOnly(n_obs),
        BlockShuffling(block_size=block_size),
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_callback=lambda c, i: np.asarray(i),
    )
    for n, batch in enumerate(ds):
        yield np.asarray(batch)
        if n + 1 >= max_batches:
            return


def index_stream(name: str, n_obs: int, *, batch_size: int, seed: int, max_batches: int,
                 preload_nchunks: int = 32, fixed_buffer: int | None = None):
    import re

    from annbatch.samplers import RandomSampler, SequentialSampler

    from _strategies import _SequentialShuffleBufferSampler

    rng = np.random.default_rng(seed)

    def _nchunks(k: int) -> int:
        # annbatch requires the in-memory buffer to hold at least one batch.
        nch = max(1, fixed_buffer // k) if fixed_buffer else preload_nchunks
        return max(nch, -(-batch_size // k))

    if name == "random":
        s = RandomSampler(chunk_size=1, preload_nchunks=16_384, batch_size=batch_size,
                          drop_last=True, rng=rng)
        return _annbatch_index_stream(s, n_obs, max_batches), {"chunk_size": 1}
    if name == "streaming":
        s = SequentialSampler(chunk_size=4096, preload_nchunks=4, batch_size=batch_size, drop_last=True)
        return _annbatch_index_stream(s, n_obs, max_batches), {"chunk_size": 4096, "shuffle": False}
    if name == "streaming_buffer":
        s = _SequentialShuffleBufferSampler(chunk_size=4096, preload_nchunks=4,
                                            batch_size=batch_size, rng=rng)
        return _annbatch_index_stream(s, n_obs, max_batches), {"buffer_rows": 16_384}

    m = re.match(r"^annbatch_(raw|pre)_c(\d+)$", name)
    if m:
        k = int(m.group(2))
        nch = _nchunks(k)
        s = RandomSampler(chunk_size=k, preload_nchunks=nch, batch_size=batch_size,
                          drop_last=True, rng=rng)
        return _annbatch_index_stream(s, n_obs, max_batches), {
            "chunk_size": k, "preload_nchunks": nch, "buffer_rows": k * nch}

    m = re.match(r"^scdataset_b(\d+)$", name)
    if m:
        if not fixed_buffer:
            raise ValueError("scdataset_b<N> without an explicit fetch factor requires --fixed-buffer")
        b = int(m.group(1))
        f = max(1, fixed_buffer // batch_size)
        return _scdataset_index_stream(n_obs, block_size=b, fetch_factor=f, batch_size=batch_size,
                                       seed=seed, max_batches=max_batches), {
            "block_size": b, "fetch_factor": f, "buffer_rows": batch_size * f}

    m = re.match(r"^scdataset_b(\d+)_f(\d+)$", name)
    if m:
        b, f = int(m.group(1)), int(m.group(2))
        return _scdataset_index_stream(n_obs, block_size=b, fetch_factor=f, batch_size=batch_size,
                                       seed=seed, max_batches=max_batches), {
            "block_size": b, "fetch_factor": f, "buffer_rows": batch_size * f}

    raise ValueError(name)


# --------------------------------------------------------------------------- #
def summarise(batches, label_arrays, global_dists):
    """Entropy, distinct-label count and TV distance, averaged over batches."""
    acc = {k: {"entropy": [], "n_distinct": [], "tv_distance": []} for k in label_arrays}
    n_batches = 0
    for idx in batches:
        n_batches += 1
        for key, labels in label_arrays.items():
            lab = labels[idx]
            counts = np.bincount(lab, minlength=len(global_dists[key])).astype(np.float64)
            p = counts / counts.sum()
            nz = p > 0
            acc[key]["entropy"].append(float(-(p[nz] * np.log2(p[nz])).sum()))
            acc[key]["n_distinct"].append(int(nz.sum()))
            acc[key]["tv_distance"].append(float(0.5 * np.abs(p - global_dists[key]).sum()))
    return {
        key: {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_batches": n_batches}
            for m, v in metrics.items()
        }
        for key, metrics in acc.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--store", default="unshuffled", choices=["unshuffled", "preshuffled"],
                    help="which on-disk row order the strategies read")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--n-batches", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-buffer", type=int, default=None,
                    help="hold chunk_size x preload_nchunks at this many rows")
    ap.add_argument("--preload-nchunks", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()
    import zarr

    labels_dir = Path(cfg["tahoe"]["labels_dir"])
    lt = LabelTable.load(labels_dir / "labels_train.npz")
    n_obs = lt.n_obs

    # plate label: the outermost on-disk block structure
    plate_sizes = [int(zarr.open(str(p), mode="r")["X"].attrs["shape"][0]) for p in plate_paths(cfg, "train")]
    plate = np.repeat(np.arange(len(plate_sizes), dtype=np.int16), plate_sizes)
    assert len(plate) == n_obs

    label_arrays = {"plate": plate, "drug": lt.codes["drug"], "cell_line": lt.codes["cell_line"]}
    if args.store == "preshuffled":
        # Permute labels into the store's row order so the stream scores the rows the loader reads.
        from _strategies import _load_global_row

        global_row = _load_global_row(cfg)
        if len(global_row) != n_obs:
            raise ValueError(f"pre-shuffled store has {len(global_row):,d} rows, labels have {n_obs:,d}")
        label_arrays = {k: v[global_row] for k, v in label_arrays.items()}
        print(f"[diversity] scoring against the pre-shuffled row order ({n_obs:,d} rows)", flush=True)
    global_dists = {
        k: np.bincount(v, minlength=int(v.max()) + 1).astype(np.float64) / len(v)
        for k, v in label_arrays.items()
    }
    print("[diversity] global entropy (bits): "
          + ", ".join(f"{k}={-(d[d > 0] * np.log2(d[d > 0])).sum():.3f}" for k, d in global_dists.items()),
          flush=True)

    ks = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if args.strategies:
        strategies = args.strategies
    elif args.store == "preshuffled":
        strategies = ["random"] + [f"annbatch_pre_c{k}" for k in ks]
    elif args.fixed_buffer:
        # Buffer-matched: same rows in memory, so only the number of on-disk blocks differs.
        strategies = (
            ["random", "streaming", "streaming_buffer"]
            + [f"annbatch_raw_c{k}" for k in ks]
            + [f"scdataset_b{k}" for k in ks]
        )
    else:
        strategies = (
            ["random", "streaming", "streaming_buffer"]
            + [f"annbatch_raw_c{k}" for k in ks]
            + [f"scdataset_b{b}_f256" for b in (1, 16, 64, 256, 512, 1024)]
        )

    rows = []
    for name in strategies:
        stream, params = index_stream(
            name, n_obs, batch_size=args.batch_size, seed=args.seed,
            max_batches=args.n_batches, preload_nchunks=args.preload_nchunks,
            fixed_buffer=args.fixed_buffer,
        )
        stats = summarise(stream, label_arrays, global_dists)
        rows.append({"strategy": name, "params": params, "stats": stats})
        print(f"[diversity] {name:24s} plate H={stats['plate']['entropy']['mean']:5.3f}  "
              f"drug H={stats['drug']['entropy']['mean']:5.3f} "
              f"(n={stats['drug']['n_distinct']['mean']:6.1f})  "
              f"cell_line H={stats['cell_line']['entropy']['mean']:5.3f}", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    if args.fixed_buffer and not args.tag:
        tag = f"_buf{args.fixed_buffer}"
    write_result(
        Path(cfg["results"]) / f"10_batch_diversity_{args.store}{tag}.json",
        {
            "experiment": "batch_diversity",
            "store": args.store,
            "note": "index streams only; strategies named annbatch_raw_* operate on the "
                    "unshuffled plate order, which is also the index space of the label tables. "
                    "annbatch on the pre-shuffled store is measured by 11_batch_diversity_preshuffled.py.",
            "n_obs": int(n_obs),
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "fixed_buffer": args.fixed_buffer,
            "preload_nchunks": args.preload_nchunks,
            "global_entropy_bits": {
                k: float(-(d[d > 0] * np.log2(d[d > 0])).sum()) for k, d in global_dists.items()
            },
            "results": rows,
            "provenance": provenance(cfg),
        },
    )


if __name__ == "__main__":
    main()
