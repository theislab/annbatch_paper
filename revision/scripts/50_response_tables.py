from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import figures  # noqa: E402

from _common import load_config  # noqa: E402

TASKS = ("cell_line", "drug", "moa_broad", "moa_fine")
HEADLINE = [
    ("random", "random sampling (reference)", "n/a"),
    ("annbatch_pre_c1024", "**annbatch, pre-shuffled, `chunk_size=1024`**", "16,384"),
    ("scdataset_b16_f32", "scDataset, `block_size=16`, `fetch_factor=32`", "131,072"),
    ("scdataset_b16_f4", "scDataset, `block_size=16`, `fetch_factor=4`", "16,384"),
    # scDataset's recommended setting at their minibatch 64; empty elsewhere.
    ("scdataset_b16_f256", "scDataset, `block_size=16`, `fetch_factor=256`", "16,384"),
    ("annbatch_raw_c1024", "annbatch, **not** pre-shuffled, `chunk_size=1024`", "16,384"),
    ("streaming_buffer", "streaming + 16,384-row shuffle buffer", "16,384"),
    ("streaming", "streaming", "n/a"),
]


def _cell(m: float, sd: float, n: int) -> str:
    return f"{m:.4f}" if n < 2 or not np.isfinite(sd) else f"{m:.4f} ± {sd:.4f}"


def _agg(runs, which, arch="mlp"):
    fm = figures.final_metrics(runs, which, "macro_f1")
    fm = fm[fm.architecture == arch]
    return (fm.groupby(["strategy", "task"])
              .agg(m=("value", "mean"), sd=("value", "std"), n=("value", "size"))
              .reset_index())


def headline_table(runs, which: str, arch: str = "mlp") -> str:
    a = _agg(runs, which, arch)
    lines = ["| strategy | buffer | seeds | " + " | ".join(
        t.replace("_", " ") for t in TASKS) + " |",
        "|---|---:|---:|" + "---:|" * len(TASKS)]
    for strat, label, buf in HEADLINE:
        sub = a[a.strategy == strat]
        if sub.empty:
            continue
        n = int(sub["n"].max())
        cells = []
        for t in TASKS:
            r = sub[sub.task == t]
            cells.append(_cell(r.m.iloc[0], r.sd.iloc[0], n) if len(r) else "n/a")
        lines.append(f"| {label} | {buf} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def chunk_sweep_table(runs, task: str = "drug", arch: str = "mlp") -> str:
    a = _agg(runs, "val_iid", arch)
    a = a[a.task == task]
    a = a[a.strategy.str.match(r"annbatch_(pre|raw)_c\d+$")].copy()
    a["fam"] = a.strategy.str.extract(r"_(pre|raw)_")
    a["k"] = a.strategy.str.extract(r"_c(\d+)$").astype(int)
    lines = ["| `chunk_size` | pre-shuffled | not pre-shuffled |", "|---:|---:|---:|"]
    for k in sorted(a.k.unique()):
        row = {}
        for fam in ("pre", "raw"):
            r = a[(a.k == k) & (a.fam == fam)]
            row[fam] = _cell(r.m.iloc[0], r.sd.iloc[0], int(r.n.iloc[0])) if len(r) else "n/a"
        lines.append(f"| {k} | {row['pre']} | {row['raw']} |")
    return "\n".join(lines)


def scdataset_table(runs, task: str = "drug", arch: str = "mlp") -> str:
    a = _agg(runs, "val_iid", arch)
    a = a[(a.task == task) & a.strategy.str.startswith("scdataset")]
    lines = ["| scDataset arm | seeds | macro-F1 |", "|---|---:|---:|"]
    for _, r in a.sort_values("strategy").iterrows():
        lines.append(f"| `{r.strategy}` | {int(r.n)} | {_cell(r.m, r.sd, int(r.n))} |")
    return "\n".join(lines)


def equivalence_note(runs) -> str:
    from scipy import stats
    out = []
    for which in ("val_iid", "val_plate"):
        fm = figures.final_metrics(runs, which, "macro_f1")
        fm = fm[(fm.architecture == "mlp") & (fm.task == "drug")]
        a = fm[fm.strategy == "annbatch_pre_c1024"]["value"].to_numpy()
        b = fm[fm.strategy == "random"]["value"].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        out.append(f"- **{which}**: annbatch {a.mean():.4f} ± {a.std(ddof=1):.4f} (n={len(a)}) "
                   f"vs random {b.mean():.4f} ± {b.std(ddof=1):.4f} (n={len(b)}); "
                   f"Welch *t* = {t:.2f}, *P* = {p:.2f}; "
                   f"relative difference {100 * (a.mean() - b.mean()) / b.mean():+.2f}%")
    pre = _agg(runs, "val_iid")
    pre = pre[(pre.task == "drug") & pre.strategy.str.startswith("annbatch_pre_c")]
    out.append(f"- spread across the nominally equivalent pre-shuffled arms: "
               f"{pre.m.min():.4f}–{pre.m.max():.4f} (range {pre.m.max() - pre.m.min():.4f})")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    figures.setup()
    runs = [r for r in figures.convergence_runs() if r["batch_size"] == 4096]
    runs64 = [r for r in figures.convergence_runs() if r["batch_size"] == 64]

    parts = [
        "# Response tables, generated from `revision/results/`",
        "",
        "Regenerate with `python revision/scripts/50_response_tables.py`.",
        f"Derived from {len(runs)} full-epoch runs at batch size 4,096 and "
        f"{len(runs64)} at batch size 64.",
        "",
        "## Headline: macro-F1 after one epoch, MLP, i.i.d. hold-out",
        "", headline_table(runs, "val_iid"), "",
        "## Same, held-out plate 14",
        "", headline_table(runs, "val_plate"), "",
        "## Same, linear classifier, i.i.d. hold-out",
        "", headline_table(runs, "val_iid", arch="linear"), "",
        "## Chunk-size sweep (MLP, drug)",
        "", chunk_sweep_table(runs), "",
        "## scDataset arms (MLP, drug)",
        "", scdataset_table(runs), "",
        "## Equivalence of annbatch and random sampling",
        "", equivalence_note(runs), "",
    ]
    if runs64:
        parts += ["## Replication at scDataset's protocol (batch 64, lr 1e-5)",
                  "", headline_table(runs64, "val_iid"), ""]
    try:
        wgs = figures.wgs_preshuffle_table()
        if len(wgs):
            parts += ["## WGS pre-shuffling wall-clock", "",
                      wgs.round(3).to_markdown(index=False), ""]
        be = figures.wgs_epoch_table()
        if len(be):
            parts += ["## WGS break-even (epochs after which pre-shuffling is repaid)", "",
                      be.round(3).to_markdown(index=False), ""]
    except Exception as exc:  # noqa: BLE001 - a missing regime must not stop the rest
        parts += [f"(WGS tables unavailable: {exc})", ""]

    # Guarded independently so an unfinished table cannot suppress the finished ones.
    def _section(title: str, build) -> None:
        try:
            obj = build()
        except Exception as exc:  # noqa: BLE001 - report, never abort
            parts.extend([f"## {title}", "", f"(unavailable: {exc})", ""])
            return
        if obj is None or (hasattr(obj, "__len__") and len(obj) == 0):
            parts.extend([f"## {title}", "", "(no results on disk yet)", ""])
            return
        body = (obj.round(3).to_markdown(index=False)
                if hasattr(obj, "to_markdown") else str(obj))
        parts.extend([f"## {title}", "", body, ""])

    _section("Throughput vs training-set size, real loaders (R2 major 5)",
             figures.dataset_size_real_table)
    _section("Throughput degradation across the dataset-size range, by format",
             figures.dataset_size_degradation)
    _section("WGS on-disk structure: achievable batch diversity",
             figures.wgs_structure_table)
    _section("WGS: is the shuffle needed, or only the conversion to Zarr?",
             figures.wgs_unshuffled_table)
    _section("Codec control: annbatch with and without compression",
             figures.codec_control)
    _section("Microscopy pre-shuffling wall-clock",
             lambda: pd.DataFrame([figures.microscopy_preshuffle()]))

    try:
        alt = figures.convergence_runs("20_convergence_lr3e-4")
        parts += ["## Learning-rate robustness (lr = 3e-4)", "",
                  headline_table(alt, "val_iid"), ""]
    except FileNotFoundError:
        pass

    text = "\n".join(parts)
    if args.stdout:
        print(text)
    else:
        out = Path(cfg["repo"]) / "revision" / "docs" / "RESULTS_TABLES.md"
        out.write_text(text)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
