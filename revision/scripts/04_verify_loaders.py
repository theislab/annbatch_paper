from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _strategies
from _common import configure_zarr, load_config, open_plate_datasets, plate_paths, provenance, write_result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--n-rows-checked", type=int, default=24,
                    help="rows sampled per batch for the read-back comparison")
    args = ap.parse_args()

    cfg = load_config()
    configure_zarr()

    truth_datasets, truth_n = open_plate_datasets(plate_paths(cfg, "train"))
    truth = _strategies._ConcatCSR(truth_datasets, truth_n)
    rng = np.random.default_rng(0)

    strategies = args.strategies or _strategies.available_strategies()
    rows, failures = [], []
    for name in strategies:
        try:
            run = _strategies.build(name, cfg=cfg, seed=0, batch_size=args.batch_size, num_workers=0)
        except FileNotFoundError as exc:
            print(f"[verify] {name:24s} SKIPPED ({exc})", flush=True)
            continue

        checked, mismatches = 0, 0
        seen_idx = []
        for i, (xb, idx) in enumerate(run.iterator):
            if i >= args.n_batches:
                break
            gidx = run.global_row[idx] if run.global_row is not None else idx
            seen_idx.append(np.asarray(gidx))
            xd = xb.to_dense().numpy() if hasattr(xb, "to_dense") else np.asarray(xb)

            pick = rng.choice(len(gidx), size=min(args.n_rows_checked, len(gidx)), replace=False)
            expected = truth[np.asarray(gidx)[pick]].toarray()
            got = xd[pick]
            mismatches += int((~np.isclose(expected, got)).any(axis=1).sum())
            checked += len(pick)

        all_idx = np.concatenate(seen_idx)
        unique = len(np.unique(all_idx))
        ok = mismatches == 0 and unique == len(all_idx)
        status = "OK  " if ok else "FAIL"
        print(f"[verify] {status} {name:24s} rows checked {checked:5d}  mismatches {mismatches:5d}  "
              f"indices {len(all_idx)} unique {unique}", flush=True)
        rows.append({
            "strategy": name, "store": run.store, "params": run.params,
            "rows_checked": checked, "mismatches": mismatches,
            "indices_yielded": int(len(all_idx)), "indices_unique": int(unique),
            "passed": bool(ok),
        })
        if not ok:
            failures.append(name)

    write_result(
        Path(cfg["results"]) / "04_verify_loaders.json",
        {
            "experiment": "verify_loaders",
            "property": "batch row i equals the store row at global index idx[i], "
                        "and no index is yielded twice within a run",
            "batch_size": args.batch_size,
            "n_batches_per_strategy": args.n_batches,
            "results": rows,
            "failures": failures,
            "provenance": provenance(cfg),
        },
    )
    if failures:
        raise SystemExit(f"loader/index misalignment in: {failures}")
    print("[verify] all strategies aligned", flush=True)


if __name__ == "__main__":
    main()
