from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import configure_zarr, load_config, plate_paths, provenance, write_result

LABELS = ("drug", "cell_line", "sample")
WINDOWS = (16, 64, 128, 256, 512, 1024, 2048, 4096, 16384)


def run_lengths(codes: np.ndarray) -> np.ndarray:
    if len(codes) == 0:
        return np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    edges = np.concatenate(([0], boundaries, [len(codes)]))
    return np.diff(edges)


def distinct_per_window(codes: np.ndarray, window: int, n_windows: int = 200,
                        rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    if len(codes) <= window:
        return float(len(np.unique(codes)))
    starts = rng.integers(0, len(codes) - window, size=n_windows)
    return float(np.mean([len(np.unique(codes[s : s + window])) for s in starts]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rows", type=int, default=4_000_000,
                    help="rows read per plate (from the start of the plate)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()
    import zarr

    rows = []
    for path in plate_paths(cfg, "all"):
        plate = path.name.split("_")[0]
        g = zarr.open(str(path), mode="r")
        n_obs = int(g["X"].attrs["shape"][0])
        for label in LABELS:
            codes = np.asarray(g[f"obs/{label}/codes"][: args.max_rows])
            n_cats = int(g[f"obs/{label}/categories"].shape[0])
            rl = run_lengths(codes)
            rng = np.random.default_rng(args.seed)
            rows.append({
                "plate": plate,
                "n_obs": n_obs,
                "rows_read": int(len(codes)),
                "label": label,
                "n_categories_in_plate": int(n_cats),
                "n_runs": int(len(rl)),
                "run_length_median": float(np.median(rl)),
                "run_length_mean": float(rl.mean()),
                "run_length_p90": float(np.percentile(rl, 90)),
                "run_length_max": int(rl.max()),
                "distinct_per_window": {
                    str(w): distinct_per_window(codes, w, rng=rng) for w in WINDOWS
                },
            })
            print(f"[structure] {plate:>8s} {label:10s} runs={len(rl):>9,d} "
                  f"median={np.median(rl):8.1f} mean={rl.mean():8.1f} max={rl.max():>7,d}  "
                  f"distinct@512={rows[-1]['distinct_per_window']['512']:6.1f}", flush=True)

    write_result(
        Path(cfg["results"]) / "05_ondisk_structure.json",
        {
            "experiment": "ondisk_structure",
            "note": "run lengths and windowed label diversity in the original (unshuffled) "
                    "on-disk order; only obs codes are read",
            "labels": list(LABELS),
            "windows": list(WINDOWS),
            "max_rows_per_plate": args.max_rows,
            "results": rows,
            "provenance": provenance(cfg),
        },
    )


if __name__ == "__main__":
    main()
