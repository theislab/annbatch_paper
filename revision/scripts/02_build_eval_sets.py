from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import TASKS, LabelTable, configure_zarr, load_config, open_plate_datasets, plate_paths, provenance, write_result

HELDOUT_SEED = 20260912  # fixed forever: changing it invalidates every trained arm


def _gather_rows(datasets, n_obs, global_indices):
    """Read the given rows (global index across the concatenation) as one CSR matrix."""
    import scipy.sparse as sp

    offsets = np.concatenate(([0], np.cumsum(n_obs)))
    order = np.argsort(global_indices)
    sorted_idx = np.asarray(global_indices)[order]
    blocks = []
    for d, (lo, hi) in zip(datasets, zip(offsets[:-1], offsets[1:], strict=True), strict=True):
        sel = sorted_idx[(sorted_idx >= lo) & (sorted_idx < hi)] - lo
        if len(sel):
            blocks.append(d[sel])
    X = sp.vstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    # undo the sort so rows line up with `global_indices`
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sp.csr_matrix(X)[inverse]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-val-iid", type=int, default=65_536)
    p.add_argument("--n-val-plate", type=int, default=65_536)
    args = p.parse_args()

    cfg = load_config()
    configure_zarr()
    import scipy.sparse as sp

    labels_dir = Path(cfg["tahoe"]["labels_dir"])
    eval_dir = labels_dir.parent / "eval_sets"
    eval_dir.mkdir(parents=True, exist_ok=True)

    lt_train = LabelTable.load(labels_dir / "labels_train.npz")
    lt_test = LabelTable.load(labels_dir / "labels_test.npz")
    rng = np.random.default_rng(HELDOUT_SEED)

    # ---- val_iid: uniform sample of plates 1-13, excluded from training ----- #
    n_train = lt_train.n_obs
    iid_idx = np.sort(rng.choice(n_train, size=args.n_val_iid, replace=False))
    heldout = np.zeros(n_train, dtype=bool)
    heldout[iid_idx] = True
    np.save(eval_dir / "heldout_mask_train.npy", heldout)
    print(f"[eval] val_iid: {len(iid_idx):,d} of {n_train:,d} training cells "
          f"({100 * len(iid_idx) / n_train:.4f}%)", flush=True)

    train_ds, train_n = open_plate_datasets(plate_paths(cfg, "train"))
    X_iid = _gather_rows(train_ds, train_n, iid_idx)
    sp.save_npz(eval_dir / "val_iid_X.npz", X_iid)
    np.savez(
        eval_dir / "val_iid_y.npz",
        index=iid_idx,
        **{t: lt_train.codes[t][iid_idx] for t in TASKS},
    )
    print(f"[eval] val_iid X: {X_iid.shape}, nnz={X_iid.nnz:,d}", flush=True)

    # ---- val_plate: uniform sample of plate 14 ------------------------------ #
    n_test = lt_test.n_obs
    plate_idx = np.sort(rng.choice(n_test, size=args.n_val_plate, replace=False))
    test_ds, test_n = open_plate_datasets(plate_paths(cfg, "test"))
    X_plate = _gather_rows(test_ds, test_n, plate_idx)
    sp.save_npz(eval_dir / "val_plate_X.npz", X_plate)
    np.savez(
        eval_dir / "val_plate_y.npz",
        index=plate_idx,
        **{t: lt_test.codes[t][plate_idx] for t in TASKS},
    )
    print(f"[eval] val_plate X: {X_plate.shape}, nnz={X_plate.nnz:,d}", flush=True)

    summary = {
        "experiment": "build_eval_sets",
        "heldout_seed": HELDOUT_SEED,
        "eval_dir": str(eval_dir),
        "val_iid": {
            "n": int(args.n_val_iid),
            "drawn_from": "plates 1-13 (training), excluded from training in every arm",
            "classes_present": {t: int(len(np.unique(lt_train.codes[t][iid_idx]))) for t in TASKS},
        },
        "val_plate": {
            "n": int(args.n_val_plate),
            "drawn_from": "plate 14 (held out entirely)",
            "classes_present": {t: int(len(np.unique(lt_test.codes[t][plate_idx]))) for t in TASKS},
        },
        "provenance": provenance(cfg),
    }
    print("[eval] classes present:", summary["val_iid"]["classes_present"],
          summary["val_plate"]["classes_present"], flush=True)
    write_result(Path(cfg["results"]) / "02_eval_sets.json", summary)


if __name__ == "__main__":
    main()
