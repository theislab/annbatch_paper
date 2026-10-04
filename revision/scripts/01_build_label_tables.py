from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _common import TASKS, configure_zarr, load_config, plate_paths, provenance, write_result

CONTROL_DRUG = "DMSO_TF"
CONTROL_LABEL = "control"


def _read_categorical(group, column: str) -> tuple[np.ndarray, list[str]]:
    """Return (codes, categories) for an on-disk AnnData categorical obs column."""
    cats = [str(c) for c in group[f"obs/{column}/categories"][:]]
    codes = np.asarray(group[f"obs/{column}/codes"][:])
    return codes, cats


def build_vocabularies(cfg) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Global, store-independent class vocabularies for the four tasks."""
    import zarr

    cell_lines, drugs = set(), set()
    for p in plate_paths(cfg, "all"):
        g = zarr.open(str(p), mode="r")
        cell_lines |= {str(c) for c in g["obs/cell_line/categories"][:]}
        drugs |= {str(c) for c in g["obs/drug/categories"][:]}

    md = pd.read_parquet(cfg["tahoe"]["drug_metadata"])
    # Two drug names have a trailing space in the stores but not the metadata; match stripped.
    md_broad = dict(zip(md["drug"].astype(str).str.strip(), md["moa-broad"].astype(str), strict=True))
    md_fine = dict(zip(md["drug"].astype(str).str.strip(), md["moa-fine"].astype(str), strict=True))

    drug_to_broad, drug_to_fine, unmapped = {}, {}, []
    for d in drugs:
        key = d.strip()
        if d == CONTROL_DRUG:
            drug_to_broad[d] = CONTROL_LABEL
            drug_to_fine[d] = CONTROL_LABEL
        elif key in md_broad:
            drug_to_broad[d] = md_broad[key]
            drug_to_fine[d] = md_fine[key]
        else:
            unmapped.append(d)
            drug_to_broad[d] = None
            drug_to_fine[d] = None
    if unmapped:
        raise ValueError(f"drugs without mechanism annotation: {sorted(unmapped)}")

    classes = {
        "cell_line": sorted(cell_lines),
        "drug": sorted(drugs),
        "moa_broad": sorted(set(drug_to_broad.values())),
        "moa_fine": sorted(set(drug_to_fine.values())),
    }
    maps = {"moa_broad": drug_to_broad, "moa_fine": drug_to_fine}
    return classes, maps


def codes_for_store(groups, classes, maps) -> dict[str, np.ndarray]:
    """Global label codes for a list of AnnData Zarr groups, concatenated in order."""
    cl_index = {c: i for i, c in enumerate(classes["cell_line"])}
    dr_index = {c: i for i, c in enumerate(classes["drug"])}
    broad_index = {c: i for i, c in enumerate(classes["moa_broad"])}
    fine_index = {c: i for i, c in enumerate(classes["moa_fine"])}

    out = {t: [] for t in TASKS}
    for g in groups:
        cl_codes, cl_cats = _read_categorical(g, "cell_line")
        dr_codes, dr_cats = _read_categorical(g, "drug")

        # Per-store category order differs; remap through the global vocabulary.
        cl_lut = np.array([cl_index[c] for c in cl_cats], dtype=np.int16)
        dr_lut = np.array([dr_index[c] for c in dr_cats], dtype=np.int16)
        broad_lut = np.array([broad_index[maps["moa_broad"][c]] for c in dr_cats], dtype=np.int8)
        fine_lut = np.array([fine_index[maps["moa_fine"][c]] for c in dr_cats], dtype=np.int8)

        if (cl_codes < 0).any() or (dr_codes < 0).any():
            raise ValueError("missing (NaN) categorical entries are not handled")

        out["cell_line"].append(cl_lut[cl_codes])
        out["drug"].append(dr_lut[dr_codes])
        out["moa_broad"].append(broad_lut[dr_codes])
        out["moa_fine"].append(fine_lut[dr_codes])

    return {t: np.concatenate(v) for t, v in out.items()}


def _groups_for(cfg, store: str):
    """Return the ordered AnnData Zarr groups backing a named store."""
    import zarr
    from annbatch import DatasetCollection

    if store in ("train", "test", "all"):
        return [zarr.open(str(p), mode="r") for p in plate_paths(cfg, store)]
    if store == "preshuffled":
        return list(DatasetCollection(zarr.open(cfg["tahoe"]["preshuffled"], mode="r")))
    raise ValueError(store)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", nargs="+", default=["train", "test", "preshuffled"],
                   choices=["train", "test", "all", "preshuffled"])
    args = p.parse_args()

    cfg = load_config()
    configure_zarr()

    classes, maps = build_vocabularies(cfg)
    print("[labels] class counts:", {t: len(c) for t, c in classes.items()}, flush=True)

    labels_dir = Path(cfg["tahoe"]["labels_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)

    with (labels_dir / "vocabularies.json").open("w") as f:
        json.dump({"classes": classes, "drug_to_moa_broad": maps["moa_broad"],
                   "drug_to_moa_fine": maps["moa_fine"]}, f, indent=2)

    summary = {"classes": {t: len(c) for t, c in classes.items()}, "stores": {}}
    for store in args.store:
        groups = _groups_for(cfg, store)
        codes = codes_for_store(groups, classes, maps)
        n_obs = len(codes["cell_line"])
        out = labels_dir / f"labels_{store}.npz"
        np.savez(out, **{f"codes__{k}": v for k, v in codes.items()})
        with out.with_suffix(".classes.json").open("w") as f:
            json.dump({"classes": classes, "n_obs": int(n_obs)}, f, indent=2)
        present = {t: int(len(np.unique(codes[t]))) for t in TASKS}
        print(f"[labels] {store}: n_obs={n_obs:,d} classes_present={present} -> {out}", flush=True)
        summary["stores"][store] = {"n_obs": int(n_obs), "classes_present": present, "path": str(out)}

    summary["provenance"] = provenance(cfg)
    write_result(Path(cfg["results"]) / "01_label_tables.json", summary)


if __name__ == "__main__":
    main()
