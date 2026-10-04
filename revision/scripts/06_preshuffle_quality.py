from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from _common import LabelTable, configure_zarr, load_config, plate_paths, provenance, write_result

SHUFFLE_CHUNK_SIZES = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
DATASET_SIZE = 2 ** 21   # the value 00_preshuffle_tahoe.py was run with


def simulate_preshuffle(n_obs: int, dataset_size: int, shuffle_chunk_size: int,
                        rng: np.random.Generator) -> np.ndarray:
    """The permutation `add_adatas(shuffle=True)` applies, without touching data."""
    n_chunks = math.ceil(n_obs / shuffle_chunk_size)
    order = rng.permutation(n_chunks)
    chunks_per_dataset = max(1, dataset_size // shuffle_chunk_size)
    out = []
    for i in range(0, n_chunks, chunks_per_dataset):
        sel = order[i : i + chunks_per_dataset]
        buf = np.concatenate([
            np.arange(c * shuffle_chunk_size, min((c + 1) * shuffle_chunk_size, n_obs))
            for c in sel
        ])
        rng.shuffle(buf)
        out.append(buf)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--shuffle-chunk-sizes", nargs="+", type=int, default=list(SHUFFLE_CHUNK_SIZES))
    ap.add_argument("--dataset-size", type=int, default=DATASET_SIZE)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--buffer-rows", type=int, default=16_384)
    ap.add_argument("--n-batches", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()
    import zarr

    from importlib.machinery import SourceFileLoader
    div = SourceFileLoader("div", str(Path(__file__).with_name("10_batch_diversity.py"))).load_module()

    lt = LabelTable.load(Path(cfg["tahoe"]["labels_dir"]) / "labels_train.npz")
    n_obs = lt.n_obs
    sizes = [int(zarr.open(str(p), mode="r")["X"].attrs["shape"][0]) for p in plate_paths(cfg, "train")]
    plate = np.repeat(np.arange(len(sizes), dtype=np.int16), sizes)
    base = {"plate": plate, "drug": lt.codes["drug"], "cell_line": lt.codes["cell_line"]}
    global_dists = {
        k: np.bincount(v, minlength=int(v.max()) + 1).astype(np.float64) / len(v)
        for k, v in base.items()
    }
    ideal = {k: float(-(d[d > 0] * np.log2(d[d > 0])).sum()) for k, d in global_dists.items()}
    print(f"[quality] global entropy: {({k: round(v, 3) for k, v in ideal.items()})}", flush=True)

    rows = []
    for B in args.shuffle_chunk_sizes:
        rng = np.random.default_rng(args.seed)
        perm = simulate_preshuffle(n_obs, args.dataset_size, B, rng)
        labels = {k: v[perm] for k, v in base.items()}
        for k in args.chunk_sizes:
            stream, params = div.index_stream(
                f"annbatch_pre_c{k}", n_obs, batch_size=args.batch_size, seed=args.seed,
                max_batches=args.n_batches, fixed_buffer=args.buffer_rows,
            )
            stats = div.summarise(stream, labels, global_dists)
            rows.append({
                "shuffle_chunk_size": B,
                "dataset_size": args.dataset_size,
                "ratio": args.dataset_size / B,
                "loader_chunk_size": k,
                "params": params,
                "stats": stats,
            })
            print(f"[quality] shuffle_chunk_size={B:>9,d} (ratio {args.dataset_size / B:>11,.0f})  "
                  f"loader chunk={k:>5d}  drug H={stats['drug']['entropy']['mean']:6.3f} "
                  f"({100 * stats['drug']['entropy']['mean'] / ideal['drug']:5.1f}% of ideal)  "
                  f"plate H={stats['plate']['entropy']['mean']:5.3f}", flush=True)

    write_result(
        Path(cfg["results"]) / "06_preshuffle_quality.json",
        {
            "experiment": "preshuffle_quality",
            "note": "the pre-shuffler is simulated on the label array; no data is read or written",
            "n_obs": int(n_obs),
            "dataset_size": args.dataset_size,
            "batch_size": args.batch_size,
            "buffer_rows": args.buffer_rows,
            "n_batches": args.n_batches,
            "global_entropy_bits": ideal,
            "shipped_shuffle_chunk_size": 1000,
            "results": rows,
            "provenance": provenance(cfg),
        },
    )


if __name__ == "__main__":
    main()
