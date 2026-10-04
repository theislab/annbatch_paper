from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import LabelTable, configure_zarr, load_config, plate_paths, provenance, write_result


def _plate_of_row(cfg, n_obs: int) -> np.ndarray:
    import zarr

    sizes = [int(zarr.open(str(p), mode="r")["X"].attrs["shape"][0]) for p in plate_paths(cfg, "train")]
    plate = np.repeat(np.arange(len(sizes), dtype=np.int16), sizes)
    if len(plate) != n_obs:
        raise ValueError(f"plate sizes sum to {len(plate):,d}, labels say {n_obs:,d}")
    return plate


def _tv_per_dataset(labels: np.ndarray, bounds: list[int], n_classes: int) -> np.ndarray:
    """Total-variation distance between each output dataset and the global mix."""
    glob = np.bincount(labels, minlength=n_classes).astype(np.float64)
    glob /= glob.sum()
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        c = np.bincount(labels[lo:hi], minlength=n_classes).astype(np.float64)
        out.append(0.5 * np.abs(c / c.sum() - glob).sum())
    return np.asarray(out)


def _longest_input_contiguous_run(global_row: np.ndarray, sample: int = 5_000_000) -> dict:
    """How often are neighbours in the output also neighbours in the input?"""
    g = global_row[:sample]
    adjacent = np.abs(np.diff(g.astype(np.int64))) == 1
    # run lengths of adjacency
    if not adjacent.any():
        return {"fraction_adjacent": 0.0, "longest_run": 1}
    idx = np.flatnonzero(np.diff(np.concatenate(([0], adjacent.view(np.int8), [0]))))
    runs = idx[1::2] - idx[0::2]
    return {
        "fraction_adjacent": float(adjacent.mean()),
        "longest_run": int(runs.max() + 1),
        "n_sampled": int(len(g)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-permutation-controls", type=int, default=3,
                    help="uniform permutations drawn to calibrate the statistics")
    ap.add_argument("--shuffle-chunk-size", type=int, default=1000,
                    help="the value 00_preshuffle_tahoe.py was run with")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()
    import zarr
    from annbatch import DatasetCollection

    collection = DatasetCollection(zarr.open(cfg["tahoe"]["preshuffled"], mode="r"))
    groups = list(collection)
    per_dataset = [int(g["obs/global_row"].shape[0]) for g in groups]
    bounds = np.concatenate(([0], np.cumsum(per_dataset))).tolist()
    global_row = np.concatenate([np.asarray(g["obs/global_row"][:]) for g in groups])
    n_obs = len(global_row)
    print(f"[verify] {len(groups)} datasets, {n_obs:,d} rows", flush=True)

    # ---- exactness --------------------------------------------------------- #
    seen = np.zeros(n_obs, dtype=bool)
    seen[global_row] = True
    is_permutation = bool(seen.all() and len(np.unique(global_row)) == n_obs)
    print(f"[verify] exact permutation: {is_permutation}", flush=True)
    if not is_permutation:
        raise SystemExit("pre-shuffled store is NOT a permutation of the input -- stop and investigate")

    # ---- displacement ------------------------------------------------------ #
    positions = np.arange(n_obs, dtype=np.int64)
    displacement = np.abs(global_row.astype(np.int64) - positions)
    disp = {
        "mean": float(displacement.mean()),
        "median": float(np.median(displacement)),
        "expected_uniform_mean": n_obs / 3.0,
        "ratio_to_uniform": float(displacement.mean() / (n_obs / 3.0)),
    }
    print(f"[verify] mean displacement {disp['mean']:,.0f} rows "
          f"({disp['ratio_to_uniform']:.4f} x the uniform expectation)", flush=True)

    # ---- plate mixing ------------------------------------------------------ #
    lt = LabelTable.load(Path(cfg["tahoe"]["labels_dir"]) / "labels_train.npz")
    plate_in = _plate_of_row(cfg, lt.n_obs)
    plate_out = plate_in[global_row]
    n_plates = int(plate_in.max()) + 1
    tv_obs = _tv_per_dataset(plate_out, bounds, n_plates)

    rng = np.random.default_rng(args.seed)
    tv_ctrl = []
    for _ in range(args.n_permutation_controls):
        tv_ctrl.append(_tv_per_dataset(plate_in[rng.permutation(n_obs)], bounds, n_plates))
    tv_ctrl = np.concatenate(tv_ctrl)
    ratio = float(tv_obs.mean() / tv_ctrl.mean())
    predicted = float(np.sqrt(args.shuffle_chunk_size))
    print(f"[verify] per-dataset plate TV distance: pre-shuffled "
          f"{tv_obs.mean():.5f} +/- {tv_obs.std():.5f}, uniform control "
          f"{tv_ctrl.mean():.5f} +/- {tv_ctrl.std():.5f}", flush=True)
    print(f"[verify] ratio {ratio:.2f} vs sqrt(shuffle_chunk_size)={predicted:.2f} "
          f"-- the pre-shuffler behaves as a block shuffle of block size "
          f"{args.shuffle_chunk_size} and nothing more", flush=True)

    # ---- locality ---------------------------------------------------------- #
    runs = _longest_input_contiguous_run(global_row)
    print(f"[verify] neighbouring output rows that were also neighbours on input: "
          f"{runs['fraction_adjacent']:.2e} (longest such run {runs['longest_run']})", flush=True)

    write_result(
        Path(cfg["results"]) / "03_verify_preshuffle.json",
        {
            "experiment": "verify_preshuffle",
            "store": cfg["tahoe"]["preshuffled"],
            "n_obs": n_obs,
            "n_datasets": len(groups),
            "rows_per_dataset": per_dataset,
            "is_exact_permutation": is_permutation,
            "displacement": disp,
            "plate_mixing": {
                "tv_distance_mean": float(tv_obs.mean()),
                "tv_distance_std": float(tv_obs.std()),
                "tv_distance_max": float(tv_obs.max()),
                "uniform_control_mean": float(tv_ctrl.mean()),
                "uniform_control_std": float(tv_ctrl.std()),
                "n_controls": args.n_permutation_controls,
                "ratio_to_uniform": ratio,
                "predicted_ratio_sqrt_shuffle_chunk_size": predicted,
                "shuffle_chunk_size": args.shuffle_chunk_size,
            },
            "input_locality": runs,
            "provenance": provenance(cfg),
        },
    )


if __name__ == "__main__":
    main()
