"""Figure code for the annbatch revision.

Loading and tidying live here so that `revision_figures.ipynb` stays readable;
every function takes the results directory and returns a tidy `DataFrame` or a
`plotnine` object.  Colours and theme match `figure_plots/paper_figures.ipynb`
so the new panels drop straight into the existing figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    geom_vline,
    guide_legend,
    guides,
    ggplot,
    labs,
    scale_alpha_manual,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_size_manual,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_bw,
    theme_set,
    xlab,
    ylab,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Palette from figure_plots/paper_figures.ipynb, extended for the new arms.
PALETTE = {
    "annbatch": "#00BA38",
    "annbatch (pre-shuffled)": "#00BA38",
    "annbatch (not pre-shuffled)": "#B2DF8A",
    "MappedCollection": "#F8766D",
    "uniform random access": "#F8766D",
    "scDataset": "#619CFF",
    "random sampling": "#000000",
    "streaming": "#999999",
    "streaming + shuffle buffer": "#CCCCCC",
    "numpy memmap": "#F8766D",
}


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })
    theme_set(theme_bw())
    FIGURES.mkdir(exist_ok=True)


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; has the corresponding job finished?")
    with path.open() as f:
        return json.load(f)


def save(plot, stem: str, width: float = 6.4, height: float = 4.8) -> None:
    """Write both an SVG (for the manuscript) and a PNG (for quick inspection)."""
    for ext in ("svg", "png"):
        plot.save(FIGURES / f"{stem}.{ext}", width=width, height=height, dpi=300, verbose=False)
    print(f"wrote {FIGURES / stem}.svg / .png")


# --------------------------------------------------------------------------- #
# Minibatch diversity                                                          #
# --------------------------------------------------------------------------- #
#: Which family a strategy belongs to, derived from its own name and its store.
#: Deriving this here rather than trusting the caller is deliberate: the caller
#: used to label a whole file with one family name, which silently drew
#: `streaming` (plate entropy 0.0004) and every `scdataset_*` arm as "annbatch
#: (not pre-shuffled)". The drug figure happened to filter them out; the plate
#: figure did not, so it showed two points at block 4096 and no scDataset line.
def diversity_family(strategy: str, store: str | None) -> str | None:
    """Family label for a strategy, or None if it does not belong on this plot."""
    pre = (store or "").endswith("preshuffled") or "preshuffled" in (store or "")
    if strategy.startswith("scdataset"):
        return None if pre else "scDataset"
    if strategy.startswith("annbatch") or strategy == "random":
        if strategy == "random":
            return None          # the reference is drawn as a rule, not a line
        return "annbatch (pre-shuffled)" if pre else "annbatch (not pre-shuffled)"
    if strategy.startswith("streaming"):
        return "streaming (no randomisation)"
    return None


def diversity_table(files: dict[str, str], label: str = "drug") -> pd.DataFrame:
    """Tidy the batch-diversity JSONs into one frame.

    `files` maps a *hint* to a results filename; the family of each row is
    derived from the strategy name and the store it was measured on, so a
    caller cannot mislabel one arm as another. Rows that do not belong on the
    diversity plot (the random reference, which is drawn as a rule) come back
    with `family` NaN and are dropped by `plot_diversity`.
    """
    rows = []
    for _hint, fname in files.items():
        blob = _load(fname)
        for r in blob["results"]:
            p = r["params"]
            size = p.get("chunk_size", p.get("block_size"))
            stats = r["stats"][label]
            rows.append({
                "family": diversity_family(r["strategy"], blob.get("store")),
                "strategy": r["strategy"],
                "block_size": size,
                "buffer_rows": p.get("buffer_rows"),
                "entropy": stats["entropy"]["mean"],
                "entropy_sd": stats["entropy"]["std"],
                "n_distinct": stats["n_distinct"]["mean"],
                "tv_distance": stats["tv_distance"]["mean"],
                "label": label,
                "global_entropy": blob["global_entropy_bits"][label],
                "store": blob.get("store"),
            })
    return pd.DataFrame(rows)


DIVERSITY_COLORS = {
    "annbatch (pre-shuffled)": PALETTE["annbatch"],
    "annbatch (not pre-shuffled)": PALETTE["annbatch (not pre-shuffled)"],
    "scDataset": PALETTE["scDataset"],
    "streaming (no randomisation)": "#999999",
}
#: scDataset and un-pre-shuffled annbatch are the *same algorithm* on the same
#: data, so their curves coincide to within 0.03 bits and one hides the other.
#: Dashing annbatch keeps both visible, because the coincidence is a result and
#: not a plotting accident, and it is the evidence that the two loaders are
#: statistically equivalent at matched buffer.
DIVERSITY_LINETYPES = {
    "annbatch (pre-shuffled)": "solid",
    "annbatch (not pre-shuffled)": (0, (5, 2)),
    "scDataset": "solid",
    "streaming (no randomisation)": "solid",
}


#: Marker shape and size carry the distinction where colour cannot. scDataset and
#: un-pre-shuffled annbatch agree to within 0.01 bits at every chunk size (0.10 at
#: 4096), so they cannot be separated by position: a large hollow square with a
#: smaller filled circle drawn inside it shows both at once, and shows that they
#: coincide, which is the actual result.
DIVERSITY_SHAPES = {
    "annbatch (pre-shuffled)": "o",
    "annbatch (not pre-shuffled)": "o",
    "scDataset": "s",
    "streaming (no randomisation)": "X",
}
DIVERSITY_SIZES = {
    "annbatch (pre-shuffled)": 3.0,
    "annbatch (not pre-shuffled)": 2.0,
    "scDataset": 5.5,
    "streaming (no randomisation)": 3.0,
}


def plot_diversity(df: pd.DataFrame, label: str = "drug", *,
                   annotate_ceiling: bool = True, streaming_as_rule: bool = True):
    """Minibatch label entropy against chunk size, against the achievable ceiling.

    The dashed rule is the *global* label entropy: what a minibatch would have if
    drawn uniformly at random from the whole collection. It is the ceiling, not a
    fit, and is labelled because an unexplained horizontal line is not
    interpretable.

    `streaming_as_rule` draws sequential streaming as a second horizontal rule
    rather than a lone marker. Streaming has no chunk size (it is the same single
    number wherever it is plotted), so a single dot floating at one x
    position reads as an artefact rather than as the floor it actually is.
    """
    sized = df.dropna(subset=["block_size", "family"]).copy()
    sized["block_size"] = sized["block_size"].astype(int)
    reference = float(df["global_entropy"].iloc[0])

    stream = sized[sized.family == "streaming (no randomisation)"]
    floor = float(stream["entropy"].min()) if len(stream) else None
    if streaming_as_rule and floor is not None:
        sized = sized[sized.family != "streaming (no randomisation)"]

    # scDataset first, un-pre-shuffled annbatch last: they coincide, so the one
    # drawn second has to be the smaller marker or it is hidden.
    order = ["annbatch (pre-shuffled)", "scDataset", "annbatch (not pre-shuffled)",
             "streaming (no randomisation)"]
    present = [f for f in order if f in set(sized.family)]
    sized["family"] = pd.Categorical(sized["family"], categories=present, ordered=True)
    sized = sized.sort_values(["family", "block_size"])

    p = (
        ggplot(sized, aes(x="block_size", y="entropy", color="family",
                          linetype="family", shape="family", size="family"))
        + geom_hline(yintercept=reference, linetype="dashed", color="black", size=0.4)
        + geom_line(size=1.0)
        + geom_point(fill="none", stroke=1.1)
        + scale_x_log10(breaks=[1, 16, 64, 256, 1024, 4096])
        + scale_color_manual(values=DIVERSITY_COLORS)
        + scale_linetype_manual(values=DIVERSITY_LINETYPES)
        + scale_shape_manual(values=[DIVERSITY_SHAPES[f] for f in present])
        + scale_size_manual(values=[DIVERSITY_SIZES[f] for f in present])
        + xlab("Chunk / block size [rows]")
        + ylab(f"Minibatch {label} entropy [bits]")
        + labs(color="", linetype="", shape="", size="")
        + guides(color=guide_legend(ncol=2), linetype=guide_legend(ncol=2),
                 shape=guide_legend(ncol=2), size=guide_legend(ncol=2))
        + theme(figure_size=(7.4, 4.8), legend_position="bottom",
                legend_text=element_text(size=8))
    )
    if annotate_ceiling:
        p = p + geom_text(x=64, y=reference,
                          label=f"ceiling: uniform random sampling = {reference:.2f} bits",
                          ha="center", va="bottom", size=8, color="black")
    if streaming_as_rule and floor is not None:
        p = p + geom_hline(yintercept=floor, linetype="dotted",
                           color="#999999", size=0.6)
        p = p + geom_text(x=64, y=floor,
                          label=f"floor: sequential streaming = {floor:.2f} bits",
                          ha="center", va="bottom", size=8, color="#777777")
    return p


def diversity_agreement(label: str = "drug") -> pd.DataFrame:
    """How closely scDataset and un-pre-shuffled annbatch agree, chunk by chunk.

    The two curves are indistinguishable by eye in the figure. That is the point:
    block sampling and chunked sampling are the same algorithm. But "you cannot
    tell them apart" is worth stating as a number rather than left as a visual
    impression.
    """
    d = diversity_table({"u": "10_batch_diversity_unshuffled_buf16384.json"}, label=label)
    piv = (d[d.family.isin(["scDataset", "annbatch (not pre-shuffled)"])]
           .pivot_table(index="block_size", columns="family", values="entropy"))
    piv["difference_bits"] = (piv["annbatch (not pre-shuffled)"] - piv["scDataset"])
    return piv.reset_index().round(4)


# --------------------------------------------------------------------------- #
# Throughput                                                                   #
# --------------------------------------------------------------------------- #
def throughput_chunk_table(
    fnames: tuple[str, ...] = (
        "30_throughput_chunk_size_annbatch.json",
        "30_throughput_chunk_size_scdataset.json",
    ),
) -> pd.DataFrame:
    """annbatch and scDataset are swept in separate jobs; merge them here.

    The two loaders hold the 16,384-row buffer constant by different means, and
    the table carries a `settings` column saying which, because "chunk size" on
    its own does not identify a configuration. annbatch's buffer is
    `chunk_size x preload_nchunks`, so `preload_nchunks` falls from 16,384 to 4
    as the chunk grows. scDataset's is `fetch_factor x batch_size`, which does
    not involve `block_size` at all, so `fetch_factor` stays at 4 throughout.
    Both therefore hold 16,384 rows at every point, which is the matched
    condition the comparison needs.
    """
    rows = []
    for fname in fnames:
        try:
            blob = _load(fname)
        except FileNotFoundError as exc:
            print(f"skipping: {exc}")
            continue
        for r in blob["results"]:
            p = r["params"]
            is_annbatch = r["loader"] == "annbatch"
            size = p.get("chunk_size", p.get("block_size"))
            if is_annbatch:
                settings = (f"chunk_size={size}, "
                            f"preload_nchunks={p.get('preload_nchunks')}")
            else:
                settings = (f"block_size={size}, "
                            f"fetch_factor={p.get('fetch_factor')}, "
                            f"num_workers={p.get('num_workers')}")
            rows.append({
                "loader": "annbatch" if is_annbatch else "scDataset",
                "regime": p.get("regime", "fixed_buffer"),
                "block_size": size,
                "buffer_rows": p.get("buffer_rows"),
                "settings": settings,
                "batch_size": blob.get("batch_size"),
                "samples_per_sec": r["samples_per_sec"],
                "repeat": p.get("repeat", 0),
            })
    df = pd.DataFrame(rows)
    return (
        df.groupby(["loader", "regime", "block_size", "buffer_rows",
                    "settings", "batch_size"], as_index=False)
        .agg(samples_per_sec=("samples_per_sec", "mean"),
             sd=("samples_per_sec", "std"),
             n=("samples_per_sec", "size"))
        .assign(epoch_hours=lambda d: 94_129_984 / d.samples_per_sec / 3600)
    )


def speed_diversity_tradeoff(label: str = "drug") -> pd.DataFrame:
    """Speed and diversity at each chunk size, for both loaders, in one table.

    This is the paper's central claim in one place, and it replaces the
    separate operating-curve figure, which plotted the same two quantities
    against each other without adding anything a reader could act on.

    scDataset's throughput and its minibatch diversity are coupled: it reaches
    useful speed only at large blocks, and large blocks are exactly what costs
    diversity. Pre-shuffling decouples them, so annbatch sits at the diversity
    ceiling at *every* chunk size while its throughput rises monotonically.
    """
    t = throughput_chunk_table()
    t["regime"] = t["regime"].str.replace("_", " ", regex=False)
    t = t[t.regime == "fixed buffer"]
    d = diversity_table(
        {"unshuffled": "10_batch_diversity_unshuffled_buf16384.json",
         "preshuffled": "10_batch_diversity_preshuffled_buf16384.json"}, label=label)

    speed = t.pivot_table(index="block_size", columns="loader", values="samples_per_sec")
    ent = d.dropna(subset=["family"]).pivot_table(
        index="block_size", columns="family", values="entropy")
    out = pd.DataFrame({
        "scDataset samples/s": speed.get("scDataset"),
        f"scDataset {label} entropy": ent.get("scDataset"),
        "annbatch samples/s": speed.get("annbatch"),
        f"annbatch (pre-shuffled) {label} entropy": ent.get("annbatch (pre-shuffled)"),
    })
    out = out.dropna(how="all").reset_index()
    ceiling = float(d["global_entropy"].iloc[0])
    out.attrs["ceiling_bits"] = ceiling
    return out


def plot_throughput_chunk(df: pd.DataFrame, *, regime: str = "fixed buffer"):
    """Throughput against chunk size at a constant in-memory buffer.

    Only the constant-buffer regime is plotted. The alternative sweep held
    `preload_nchunks` at 32, which makes the buffer *grow* with chunk size (4,096
    rows at chunk 128 up to 131,072 at 4,096), so it confounds chunk size with
    memory budget and measures two things at once. Its apparent divergence above
    chunk 256 also sits inside annbatch's 10-30% run-to-run spread on this
    filesystem, so there was nothing there to interpret. The sweep is still in
    the result JSON if anyone wants it.
    """
    d = df.copy()
    d["regime"] = d["regime"].str.replace("_", " ", regex=False)
    d = d[d.regime == regime]
    d["lo"] = (d.samples_per_sec - d["sd"].fillna(0)).clip(lower=1)
    d["hi"] = d.samples_per_sec + d["sd"].fillna(0)
    return (
        ggplot(d, aes(x="block_size", y="samples_per_sec", color="loader"))
        + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.05, alpha=0.7)
        + geom_line(size=0.9)
        + geom_point(size=2.5)
        + scale_x_log10(breaks=[1, 16, 64, 256, 1024, 4096])
        + scale_y_log10()
        + scale_color_manual(values={"annbatch": PALETTE["annbatch"],
                                     "scDataset": PALETTE["scDataset"]})
        + xlab("Chunk / block size [rows]")
        + ylab("Samples per second [1/s]")
        # The subtitle carries the settings because "chunk size" alone does not
        # identify a configuration, and the two loaders reach the same 16,384-row
        # buffer through different parameters.
        + labs(color="",
               title="Throughput vs chunk size, buffer held at 16,384 rows",
               subtitle="minibatch 4,096; annbatch chunk_size x preload_nchunks "
                        "= 16,384 (nchunks 16,384 down to 4);\n"
                        "scDataset block_size varies at fetch_factor 4 "
                        "(4 x 4,096 = 16,384), 6 workers;\n"
                        "1,000,000 samples per point, 3 repeats, dedicated node; "
                        "bars are 1 s.d.")
        + theme(figure_size=(6.8, 5.0), legend_position="bottom",
                plot_subtitle=element_text(size=7.5, ha="left"))
    )


def throughput_dataset_table(
    fnames: tuple[str, ...] = ("30_throughput_dataset_size.json",
                               "30_throughput_dataset_size_warm.json"),
) -> pd.DataFrame:
    """Throughput vs dataset size, cold and warm page cache.

    The cold curve is flat by construction (a single pass of `n_samples` over a
    subset re-reads nothing, so caching cannot help at any size), and it isolates
    raw access cost.  The warm curve is the one that answers the amortisation
    question, because there the small subsets are resident and the large ones
    are not.
    """
    rows = []
    for fname in fnames:
        try:
            blob = _load(fname)
        except FileNotFoundError as exc:
            print(f"skipping: {exc}")
            continue
        cache = "warm" if blob.get("warm_cache") else "cold"
        for r in blob["results"]:
            p = r["params"]
            rows.append({
                "loader": r["loader"],
                "cache": cache,
                "n_obs": p["n_obs"],
                "samples_per_sec": r["samples_per_sec"],
                "repeat": p.get("repeat", 0),
            })
    df = pd.DataFrame(rows)
    return (
        df.groupby(["loader", "cache", "n_obs"], as_index=False)
        .agg(samples_per_sec=("samples_per_sec", "mean"), sd=("samples_per_sec", "std"))
    )


def plot_throughput_dataset(df: pd.DataFrame):
    return (
        ggplot(df, aes(x="n_obs", y="samples_per_sec", color="loader", linetype="cache"))
        + geom_line()
        + geom_point(size=2)
        + scale_x_log10()
        + scale_y_log10()
        + xlab("Training set size [cells]")
        + ylab("Samples per second [1/s]")
        + labs(color="", linetype="page cache")
        + theme(figure_size=(6.8, 4.2), legend_position="bottom",
                legend_text=element_text(size=7))
    )


def break_even(preshuffle_hours: float, df: pd.DataFrame, annbatch_label: str,
               baseline_labels: list[str], cache: str = "warm",
               n_obs_preshuffled: int | None = None) -> pd.DataFrame:
    """Epochs after which pre-shuffling has paid for itself, per baseline and size.

    `preshuffle_hours` is the wall-clock measured for a store of
    `n_obs_preshuffled` rows. Pre-shuffling reads and writes every row once, so
    its cost is linear in the number of rows; when a *subset* is evaluated the
    cost is scaled accordingly. Charging the full store's pre-shuffling time to
    a 2M-cell subset would overstate the break-even point by ~45x.
    """
    df = df[df.cache == cache] if "cache" in df.columns else df
    wide = df.pivot_table(index="n_obs", columns="loader", values="samples_per_sec")
    ref = n_obs_preshuffled or int(wide.index.max())
    out = []
    for baseline in baseline_labels:
        for n_obs, row in wide.iterrows():
            t_ann = n_obs / row[annbatch_label] / 3600
            t_alt = n_obs / row[baseline] / 3600
            t_pre = preshuffle_hours * n_obs / ref
            out.append({
                "n_obs": n_obs,
                "baseline": baseline,
                "epoch_h_annbatch": t_ann,
                "epoch_h_baseline": t_alt,
                "preshuffle_h": t_pre,
                "break_even_epochs": t_pre / (t_alt - t_ann) if t_alt > t_ann else float("inf"),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Convergence                                                                  #
# --------------------------------------------------------------------------- #
DISPLAY = {
    "random": "random sampling",
    "streaming": "streaming",
    "streaming_buffer": "streaming + shuffle buffer",
}


def _display_name(strategy: str) -> str:
    if strategy in DISPLAY:
        return DISPLAY[strategy]
    if strategy.startswith("annbatch_pre_c"):
        return f"annbatch, pre-shuffled (chunk {strategy.split('_c')[1]})"
    if strategy.startswith("annbatch_raw_c"):
        return f"annbatch, not pre-shuffled (chunk {strategy.split('_c')[1]})"
    if strategy.startswith("scdataset_b"):
        # the fetch factor is load-bearing: it sets the in-memory buffer, which is
        # what governs minibatch diversity, so two arms at the same block size but
        # different fetch factors must not be merged
        b, f = strategy.removeprefix("scdataset_b").split("_f")
        return f"scDataset (block {b}, fetch {f})"
    return strategy


def convergence_runs(subdir: str = "20_convergence") -> list[dict]:
    d = RESULTS / subdir
    if not d.exists():
        raise FileNotFoundError(f"{d} is missing; have the convergence jobs finished?")
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def training_curves(runs: list[dict], head: str = "mlp/drug") -> pd.DataFrame:
    rows = []
    for r in runs:
        for h in r["train_history"]:
            rows.append({
                "strategy": r["strategy"],
                "display": _display_name(r["strategy"]),
                "run": f"{r['strategy']}#{r['seed']}",   # one line per (strategy, seed)
                "seed": r["seed"],
                "batch_size": r["batch_size"],
                "step": h["step"],
                "samples": h["samples"],
                "loss": h["train_loss"][head],
                "head": head,
            })
    return pd.DataFrame(rows)


def validation_curves(runs: list[dict], head: str = "mlp/drug",
                      which: str = "val_iid", metric: str = "macro_f1") -> pd.DataFrame:
    rows = []
    for r in runs:
        for h in r["val_history"]:
            rows.append({
                "strategy": r["strategy"],
                "display": _display_name(r["strategy"]),
                "run": f"{r['strategy']}#{r['seed']}",
                "seed": r["seed"],
                "batch_size": r["batch_size"],
                "step": h["step"],
                "samples": h["samples"],
                "value": h[which][head][metric],
                "head": head,
                "eval_set": which,
                "metric": metric,
            })
    return pd.DataFrame(rows)


def final_metrics(runs: list[dict], which: str = "val_iid", metric: str = "macro_f1") -> pd.DataFrame:
    rows = []
    for r in runs:
        if not r.get("final"):
            continue
        for head, vals in r["final"][which].items():
            arch, task = head.split("/")
            rows.append({
                "strategy": r["strategy"],
                "display": _display_name(r["strategy"]),
                "seed": r["seed"],
                "batch_size": r["batch_size"],
                "architecture": arch,
                "task": task,
                "value": vals[metric],
                "eval_set": which,
                "metric": metric,
            })
    return pd.DataFrame(rows)


#: scDataset's four strategies (their sec. 4.4) mapped onto our arm names. Their
#: BlockShuffling(b=1) is uniform random sampling; their "streaming with a shuffle
#: buffer of 16,384" is our streaming_buffer; their BlockShuffling(b=16, f=256)
#: holds 16,384 rows at minibatch 64, which at our minibatch of 4,096 is the
#: matched-buffer setting b=16, f=4. annbatch is the arm they did not have.
SCDATASET_FIG6_ARMS = {
    "streaming": "Streaming",
    "streaming_buffer": "Streaming + 16,384 shuffle buffer",
    "scdataset_b1024_f4": "scDataset (block 1024)",
    "scdataset_b16_f4": "scDataset (block 16)",
    "random": "Random Sampling",
    "annbatch_pre_c1024": "annbatch, pre-shuffled (chunk 1024)",
}


def plate_boundaries(batch_size: int = 4096) -> list[int]:
    """Step indices at which the sequential reader crosses a plate boundary.

    Derived from the per-plate row counts of the 13 training plates, in the order
    the unshuffled collection concatenates them. These are what the loss spikes in
    scDataset's Figure 6 are spikes *at*, so marking them turns an observation
    into a mechanism.
    """
    rows = [5481420, 8064658, 4705402, 7004356, 6419498, 7545393, 5692117,
            8880979, 5866669, 8044908, 7435869, 10487057, 8501658]
    cum, out = 0, []
    for n in rows[:-1]:
        cum += n
        out.append(cum // batch_size)
    return out


def figure6_loss_curves(runs: list[dict] | None = None, *, batch_size: int = 4096,
                        seed: int | None = 0,
                        arms: dict[str, str] | None = None) -> pd.DataFrame:
    """Training loss per task and architecture: scDataset's Figure 6.

    Their Figure 6 is *training loss curves*, not performance metrics (that is
    their Figure 5). It is the panel that shows the mechanism: streaming exhibits
    periodic loss spikes at plate boundaries, consistent with catastrophic
    forgetting, while shuffled strategies produce smooth curves.

    One seed by default, since the point is the shape of a single trajectory
    rather than a mean; pass `seed=None` to overlay all seeds.
    """
    arms = arms or SCDATASET_FIG6_ARMS
    runs = runs if runs is not None else convergence_runs()
    rows = []
    for r in runs:
        if r["batch_size"] != batch_size or r["strategy"] not in arms:
            continue
        if seed is not None and r["seed"] != seed:
            continue
        for h in r["train_history"]:
            for head, loss in h["train_loss"].items():
                arch, task = head.split("/")
                rows.append({
                    "strategy": r["strategy"],
                    "display": arms[r["strategy"]],
                    "seed": r["seed"],
                    "step": h["step"],
                    "architecture": arch,
                    "task": task,
                    "loss": loss,
                })
    df = pd.DataFrame(rows)
    if len(df):
        df["display"] = pd.Categorical(df["display"], categories=list(arms.values()),
                                       ordered=True)
    return df


#: Legend key for the plate-boundary rules.
PLATE_KEY = "plate boundaries"

#: Colours for the Figure 6 arms, in the manuscript's own palette. annbatch is
#: drawn dashed so that Random Sampling, which lies underneath it, stays visible
#: the two curves being indistinguishable is the result, and a solid line on top
#: of a solid line hides exactly that.
FIG6_COLORS = {
    "Streaming": "#b05c00",
    "Streaming + 16,384 shuffle buffer": "#8a8a8a",
    "scDataset (block 1024)": "#1f5fa8",
    "scDataset (block 16)": PALETTE["scDataset"],
    "Random Sampling": "#000000",
    "annbatch, pre-shuffled (chunk 1024)": PALETTE["annbatch"],
}
#: Line widths, because the arms are not equally important. The two that carry
#: the claim, annbatch and the random reference it must match, are drawn
#: thickest; the failing arms are thinner so they do not swamp the panel while
#: still being clearly visible.
FIG6_SIZES = {
    "Streaming": 0.45,
    "Streaming + 16,384 shuffle buffer": 0.45,
    "scDataset (block 1024)": 0.9,
    "scDataset (block 16)": 0.9,
    "Random Sampling": 1.5,
    "annbatch, pre-shuffled (chunk 1024)": 1.3,
}
#: Line styles carry the distinction as well as colour: five of these arms are
#: near-black-to-blue and a reader cannot separate them by hue alone, especially
#: in print or for a colour-blind reader. annbatch is dashed because it lies
#: exactly on Random Sampling, which is the result.
FIG6_LINETYPES = {
    "Streaming": "solid",
    "Streaming + 16,384 shuffle buffer": (0, (2, 1.5)),
    "scDataset (block 1024)": (0, (6, 2)),
    "scDataset (block 16)": "solid",
    "Random Sampling": "solid",
    "annbatch, pre-shuffled (chunk 1024)": (0, (4, 2)),
}


def partial_epoch_table(head: str = "mlp/drug", *, which: str = "val_iid",
                        metric: str = "macro_f1", batch_size: int = 4096,
                        seed: int = 0,
                        fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00),
                        arms: tuple[str, ...] = ("random", "annbatch_pre_c1024",
                                                 "annbatch_raw_c1024",
                                                 "scdataset_b16_f4", "streaming"),
                        ) -> pd.DataFrame:
    """Held-out performance as a function of how much of one epoch has been seen.

    This answers the strongest version of Reviewer 2's fifth comment. The reviewer
    argues that at scale models are trained for only one or two epochs, so
    pre-shuffling is a large fraction of total cost rather than an amortised
    one-off. The premise is right: `scvi-tools` sets its default epoch budget from
    a constant *sample* budget, which at 94.1M cells asks for 0.085 epochs and
    floors to 1. But the conclusion does not follow.

    In the few-epoch regime, what the model has seen *at all* is determined by
    on-disk order. A chunked reader that has not completed an epoch has seen a
    biased slice of the plates, and cannot average that out over later passes.
    Pre-shuffling is therefore more important at low epoch counts, not less.
    """
    runs = {r["strategy"]: r for r in convergence_runs()
            if r["batch_size"] == batch_size and r["seed"] == seed}
    rows = []
    for frac in fractions:
        row = {"fraction_of_epoch": frac}
        for arm in arms:
            r = runs.get(arm)
            if r is None:
                continue
            vh = r["val_history"]
            total = r.get("max_steps") or vh[-1]["step"]
            nearest = min(vh, key=lambda h: abs(h["step"] / total - frac))
            # The fraction is nominal; record the checkpoint actually scored.
            row.setdefault("optimiser_step", nearest["step"])
            row[_display_name(arm)] = nearest[which][head][metric]
        rows.append(row)
    df = pd.DataFrame(rows)
    ref = _display_name("random")
    if ref in df.columns:
        for c in df.columns:
            if c not in ("fraction_of_epoch", "optimiser_step", ref):
                df[c + " / random"] = (df[c] / df[ref]).round(3)
    df.attrs["what"] = (f"{head.split('/')[-1]} {metric} on {which} "
                        f"({head.split('/')[0].upper()} head, seed {seed}), at the "
                        f"checkpoint nearest each fraction of one epoch")
    return df


def plot_figure6(df: pd.DataFrame, *, architecture: str = "mlp",
                 mark_plates: bool = True, batch_size: int = 4096,
                 tasks: tuple[str, ...] | None = None, ncol: int = 2):
    """scDataset Figure 6, reproduced with annbatch added and plate boundaries marked.

    Their Figure 6 attributes streaming's loss spikes to plate boundaries. Marking
    the boundaries lets the reader check that, and in our data it is only partly
    true: the largest excursions are enriched at boundaries but not confined to
    them, because Tahoe is ordered by well and drug *within* each plate as well.
    See `figure6_spike_alignment()`.
    """
    d = df[df.architecture == architecture].copy()
    if tasks:
        d = d[d.task.isin(tasks)]
    p = ggplot(d, aes(x="step", y="loss", color="display", linetype="display",
                      size="display"))
    if mark_plates:
        # On their own scale, so they get a separate legend block drawn by the
        # vline itself: a vertical key for the boundaries, horizontal keys for
        # the series. Sharing `display` made every key a vertical-plus-horizontal
        # cross, because each key draws the glyph of every layer that maps it.
        bounds = pd.DataFrame({"xintercept": plate_boundaries(batch_size),
                               "key": PLATE_KEY})
        p = p + geom_vline(aes(xintercept="xintercept", alpha="key"), data=bounds,
                           color="#D62728", linetype="dashed", size=0.7,
                           inherit_aes=False)
    return (
        p
        + geom_line()
        + facet_wrap("~ task", scales="free_y", ncol=ncol)
        + scale_color_manual(values=FIG6_COLORS)
        + scale_linetype_manual(values=FIG6_LINETYPES)
        + scale_size_manual(values=FIG6_SIZES)
        + scale_alpha_manual(values={PLATE_KEY: 0.85})
        + xlab(f"Optimiser step (minibatch {batch_size:,})")
        + ylab("Training loss")
        + labs(color="", linetype="", size="", alpha="",
               title="Training loss, " + architecture.upper())
        + guides(color=guide_legend(ncol=2), linetype=guide_legend(ncol=2),
                 size=guide_legend(ncol=2))
        + theme(figure_size=(9.5, 7.0), legend_position="bottom",
                legend_text=element_text(size=8))
    )


def figure6_spike_alignment(head: str = "mlp/drug", *, seed: int = 0,
                            batch_size: int = 4096, n_top: int = 12,
                            window: int = 21) -> pd.DataFrame:
    """Are streaming's largest loss excursions actually at plate boundaries?

    scDataset attributes them to plate transitions. This ranks each logged point
    by how far above its local median the loss sits, takes the largest
    excursions, and reports their distance to the nearest plate boundary against
    the distance expected if they fell uniformly at random.
    """
    runs = [r for r in convergence_runs()
            if r["strategy"] == "streaming" and r["batch_size"] == batch_size
            and r["seed"] == seed]
    if not runs:
        return pd.DataFrame()
    th = runs[0]["train_history"]
    step = np.array([h["step"] for h in th])
    loss = np.array([h["train_loss"][head] for h in th])
    from numpy.lib.stride_tricks import sliding_window_view

    med = np.median(sliding_window_view(loss, window), axis=1)
    half = window // 2
    s2, excess = step[half:-half], loss[half:-half] - med
    top = s2[np.argsort(excess)[::-1][:n_top]]

    bounds = np.array(plate_boundaries(batch_size))
    dist = np.array([np.min(np.abs(bounds - t)) for t in top])
    rng = np.random.default_rng(0)
    null = np.array([np.min(np.abs(bounds - t))
                     for t in rng.integers(0, step.max(), 5000)])
    return pd.DataFrame([{
        "head": head,
        "n_largest_excursions": n_top,
        "within_100_steps_of_boundary": int((dist <= 100).sum()),
        "median_distance_steps": float(np.median(dist)),
        "median_distance_if_random": float(np.median(null)),
        "enriched_at_boundaries": bool(np.median(dist) < np.median(null) * 0.75),
    }])


#: Arms for the chunk-size version of Figure 6, grouped by family. Each entry is
#: (strategy, family, chunk/block size). scDataset's block_size and annbatch's
#: chunk_size are the same quantity, the number of contiguous rows drawn per
#: read, so they are directly comparable at matched buffer.
FIG6_CHUNK_ARMS = [
    ("random", "uniform random (reference)", None),
    ("annbatch_pre_c16", "annbatch, pre-shuffled", 16),
    ("annbatch_pre_c128", "annbatch, pre-shuffled", 128),
    ("annbatch_pre_c512", "annbatch, pre-shuffled", 512),
    ("annbatch_pre_c1024", "annbatch, pre-shuffled", 1024),
    ("annbatch_raw_c16", "annbatch, not pre-shuffled", 16),
    ("annbatch_raw_c128", "annbatch, not pre-shuffled", 128),
    ("annbatch_raw_c512", "annbatch, not pre-shuffled", 512),
    ("annbatch_raw_c1024", "annbatch, not pre-shuffled", 1024),
    ("scdataset_b16_f4", "scDataset", 16),
    ("scdataset_b128_f4", "scDataset", 128),
    ("scdataset_b512_f4", "scDataset", 512),
    ("scdataset_b1024_f4", "scDataset", 1024),
]

#: The reference is thick and solid underneath; annbatch is dashed on top with a
#: gap wide enough to let it show through, since the two coinciding is the result.
FIG6_FAMILY_LINETYPES = {
    "uniform random (reference)": "solid",
    "annbatch, pre-shuffled": (0, (2, 6)),
    "annbatch, not pre-shuffled": "solid",
    "scDataset": (0, (6, 2)),
}
FIG6_FAMILY_SIZES = {
    "uniform random (reference)": 1.9,
    "annbatch, pre-shuffled": 1.0,
    "annbatch, not pre-shuffled": 0.7,
    "scDataset": 0.7,
}

FIG6_FAMILY_COLORS = {
    "uniform random (reference)": "#000000",
    "annbatch, pre-shuffled": PALETTE["annbatch"],
    "annbatch, not pre-shuffled": PALETTE["annbatch (not pre-shuffled)"],
    "scDataset": PALETTE["scDataset"],
}


def figure6_chunk_sweep(head: str = "mlp/drug", *, seed: int = 0,
                        batch_size: int = 4096) -> pd.DataFrame:
    """Training loss against chunk size: Figure 6 as the answer to R3 comment 1.

    Reviewer 3 asks "how much larger can chunk sizes be made while preserving
    comparable mini-batch diversity?". Entropy answers that in expectation; this
    answers it in training dynamics, which is what actually matters. Faceting
    scDataset's Figure 6 by chunk size turns it from a demonstration that
    streaming is bad into a measurement of where the trade-off begins.

    The reference arm is repeated in every facet so each panel is self-contained.
    """
    runs = {(r["strategy"], r["seed"]): r for r in convergence_runs()
            if r["batch_size"] == batch_size}
    sizes = sorted({c for _, _, c in FIG6_CHUNK_ARMS if c})
    rows = []
    for strategy, family, chunk in FIG6_CHUNK_ARMS:
        r = runs.get((strategy, seed))
        if r is None:
            continue
        for facet in (sizes if chunk is None else [chunk]):
            for h in r["train_history"]:
                rows.append({"strategy": strategy, "family": family,
                             "chunk_size": facet, "step": h["step"],
                             "loss": h["train_loss"][head], "head": head})
    df = pd.DataFrame(rows)
    if len(df):
        df["family"] = pd.Categorical(df["family"],
                                      categories=list(FIG6_FAMILY_COLORS), ordered=True)
        df["facet"] = "chunk / block size = " + df.chunk_size.astype(str)
    return df


def plot_figure6_chunk_sweep(df: pd.DataFrame, *, batch_size: int = 4096):
    """Loss curves per chunk size, for each loader family."""
    order = sorted(df.chunk_size.unique())
    d = df.copy()
    d["facet"] = pd.Categorical(d["facet"],
                                categories=[f"chunk / block size = {c}" for c in order],
                                ordered=True)
    # Draw the reference first so annbatch's dashes sit on top of it.
    d["family"] = pd.Categorical(d["family"], ordered=True,
                                 categories=list(FIG6_FAMILY_COLORS))
    d = d.sort_values("family")
    return (
        ggplot(d, aes(x="step", y="loss", color="family", linetype="family",
                      size="family"))
        + geom_line()
        + facet_wrap("~ facet", ncol=2)
        + scale_color_manual(values=FIG6_FAMILY_COLORS)
        + scale_linetype_manual(values=FIG6_FAMILY_LINETYPES)
        + scale_size_manual(values=FIG6_FAMILY_SIZES)
        + xlab(f"Optimiser step (minibatch {batch_size:,})")
        + ylab("Training loss, drug (MLP)")
        + labs(color="", linetype="", size="",
               title="Training loss by chunk / block size, drug (MLP)")
        + theme(figure_size=(9.0, 6.5), legend_position="bottom",
                legend_text=element_text(size=7))
    )


def figure6_chunk_summary(head: str = "mlp/drug", *, seed: int = 0,
                          batch_size: int = 4096) -> pd.DataFrame:
    """Quantify the curves in `figure6_chunk_sweep`: how erratic, and how far converged.

    `mean_abs_deviation` is the mean absolute distance of the logged loss from its
    own rolling median, a direct measure of how erratic training is, independent
    of where it converges. `final_loss` is where it got to.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    runs = {(r["strategy"], r["seed"]): r for r in convergence_runs()
            if r["batch_size"] == batch_size}
    rows = []
    for strategy, family, chunk in FIG6_CHUNK_ARMS:
        r = runs.get((strategy, seed))
        if r is None:
            continue
        loss = np.array([h["train_loss"][head] for h in r["train_history"]])
        if len(loss) < 25:
            continue
        med = np.median(sliding_window_view(loss, 21), axis=1)
        core = loss[10:-10]
        rows.append({"family": family, "chunk_size": chunk,
                     "mean_abs_deviation": float(np.mean(np.abs(core - med))),
                     "max_excursion": float(np.max(core - med)),
                     "final_loss": float(loss[-1])})
    return pd.DataFrame(rows).sort_values(["family", "chunk_size"])


def plot_training_curves(df: pd.DataFrame):
    return (
        ggplot(df, aes(x="samples", y="loss", color="display", group="run"))
        + geom_line(alpha=0.9)
        + scale_x_log10()
        + xlab("Training samples seen")
        + ylab("Training cross-entropy")
        + labs(color="")
        + theme(figure_size=(7.0, 4.2), legend_position="right",
                legend_text=element_text(size=7))
    )


#: The arms worth putting on a single-panel curve plot. Drawing all 19 produced a
#: figure in which the legend took more space than the data and the streaming
#: lines obscured everything else; chunk-size detail belongs in the faceted
#: `plot_figure6_chunk_sweep`, not in one panel.
HEADLINE_ARMS = {
    "random": ("uniform random sampling", "#000000", "solid"),
    "annbatch_pre_c1024": ("annbatch, pre-shuffled (chunk 1024)", PALETTE["annbatch"], (0, (5, 2))),
    "annbatch_raw_c1024": ("annbatch, not pre-shuffled (chunk 1024)",
                           PALETTE["annbatch (not pre-shuffled)"], "solid"),
    "scdataset_b16_f4": ("scDataset (block 16, fetch 4)", PALETTE["scDataset"], "solid"),
    "scdataset_b16_f32": ("scDataset (block 16, fetch 32)", "#9ecae1", "solid"),
    "streaming": ("streaming", "#999999", "solid"),
    "streaming_buffer": ("streaming + 16,384 buffer", "#cccccc", "solid"),
}


def headline_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a curve frame to the headline arms and label them for plotting."""
    d = df[df.strategy.isin(HEADLINE_ARMS)].copy()
    d["arm"] = d.strategy.map(lambda s: HEADLINE_ARMS[s][0])
    d["arm"] = pd.Categorical(d["arm"], ordered=True,
                              categories=[v[0] for v in HEADLINE_ARMS.values()])
    return d


def plot_headline_curves(df: pd.DataFrame, *, y: str = "value",
                         ylabel: str = "Validation macro-F1",
                         log_y: bool = False, x: str = "samples"):
    """One panel, seven arms, readable.

    `log_y` is worth setting for cross-entropy: the streaming arms sit an order of
    magnitude above the rest, and on a linear axis they compress everything else
    into the bottom fifth of the panel.
    """
    d = headline_curves(df)
    p = (
        ggplot(d, aes(x=x, y=y, color="arm", linetype="arm", group="run"
                      if "run" in d.columns else "arm"))
        + geom_line(size=0.55)
        + scale_color_manual(values={v[0]: v[1] for v in HEADLINE_ARMS.values()})
        + scale_linetype_manual(values={v[0]: v[2] for v in HEADLINE_ARMS.values()})
        + xlab("Training samples seen")
        + ylab(ylabel + (" [log scale]" if log_y else ""))
        + labs(color="", linetype="")
        + guides(color=guide_legend(ncol=2), linetype=guide_legend(ncol=2))
        + theme(figure_size=(8.0, 5.0), legend_position="bottom",
                legend_text=element_text(size=7.5))
    )
    p = p + scale_x_log10()
    if log_y:
        p = p + scale_y_log10()
    return p


def plot_validation_curves(df: pd.DataFrame, ylabel: str = "Validation macro-F1"):
    return (
        ggplot(df, aes(x="samples", y="value", color="display", group="run"))
        + geom_line()
        + geom_point(size=0.8)
        + xlab("Training samples seen")
        + ylab(ylabel)
        + labs(color="")
        + theme(figure_size=(7.0, 4.2), legend_position="right",
                legend_text=element_text(size=7))
    )


def plot_final_metrics(df: pd.DataFrame):
    summary = (
        df.groupby(["display", "architecture", "task"], as_index=False)
        .agg(value=("value", "mean"), sd=("value", "std"))
    )
    return (
        ggplot(summary, aes(x="display", y="value", fill="display"))
        + geom_col(show_legend=False)
        + geom_text(aes(label="value"), va="bottom", size=6, format_string="{:.3f}")
        + facet_wrap("~ architecture + task", scales="free_y", ncol=4)
        + xlab("")
        + ylab("Macro-F1")
        + theme(figure_size=(12.0, 6.0),
                axis_text_x=element_text(rotation=60, ha="right", size=6))
    )


# --------------------------------------------------------------------------- #
# WGS                                                                          #
# --------------------------------------------------------------------------- #
def memmap_table(fname: str = "40_wgs_memmap_baselines_bs128.json") -> pd.DataFrame:
    blob = _load(fname)
    rows = [{
        "mode": r["loader"],
        "samples_per_sec": r["samples_per_sec"],
        "repeat": r["params"].get("repeat", 0),
        "access": r["params"].get("access", ""),
    } for r in blob["results"]]
    df = pd.DataFrame(rows)
    return (
        df.groupby(["mode", "access"], as_index=False)
        .agg(samples_per_sec=("samples_per_sec", "mean"), sd=("samples_per_sec", "std"))
        .sort_values("samples_per_sec")
    )


MEMMAP_LABELS = {
    "random_row_manuscript": "numpy memmap\nrandom rows\n(manuscript baseline)",
    "random_row": "numpy memmap\nrandom rows\n(in-process)",
    "block_shuffled": "numpy memmap\nblock-shuffled",
    "sequential": "numpy memmap\nsequential\n(no shuffling)",
    "annbatch": "annbatch\npre-shuffled Zarr",
}


def plot_memmap(df: pd.DataFrame):
    d = df.assign(
        label=lambda x: x["mode"].map(MEMMAP_LABELS).fillna(x["mode"]),
        family=lambda x: x["mode"].map(lambda m: "annbatch" if m == "annbatch" else "numpy memmap"),
    )
    order = d.sort_values("samples_per_sec")["label"].tolist()
    d["label"] = pd.Categorical(d["label"], categories=order, ordered=True)
    return (
        ggplot(d, aes(x="label", y="samples_per_sec", fill="family"))
        + geom_col(show_legend=False)
        + geom_text(aes(label="samples_per_sec"), va="bottom", size=7, format_string="{:.0f}")
        + scale_y_log10()
        + scale_fill_manual(values={"annbatch": PALETTE["annbatch"], "numpy memmap": PALETTE["numpy memmap"]})
        + xlab("")
        + ylab("Samples per second [1/s]")
        + theme(figure_size=(7.0, 4.2), axis_text_x=element_text(size=7))
    )


def wgs_preshuffle_table() -> pd.DataFrame:
    rows = []
    for regime in ("common", "rare_maf_0.01", "rare_maf_0.001"):
        path = RESULTS / f"41_wgs_preshuffle_{regime}.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        rows.append({
            "regime": regime,
            "n_variants": blob["n_var"],
            "sparse": blob["sparse"],
            "input_GiB": blob["input_bytes"] / 1024 ** 3,
            "output_GiB": blob["output_bytes"] / 1024 ** 3,
            "preshuffle_hours": blob["elapsed_h"],
            "host": blob["provenance"]["hostname"],
            "cpu": blob["provenance"]["cpu_model"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# On-disk structure                                                            #
# --------------------------------------------------------------------------- #
def ondisk_structure_table(fname: str = "05_ondisk_structure.json") -> pd.DataFrame:
    """Run lengths and windowed label diversity in the original on-disk order."""
    blob = _load(fname)
    rows = []
    for r in blob["results"]:
        base = {k: r[k] for k in ("plate", "label", "n_categories_in_plate",
                                  "n_runs", "run_length_median", "run_length_mean",
                                  "run_length_p90", "run_length_max")}
        for w, n in r["distinct_per_window"].items():
            rows.append({**base, "window": int(w), "n_distinct": n})
    return pd.DataFrame(rows)


def plot_ondisk_structure(df: pd.DataFrame, label: str = "drug"):
    d = df[df.label == label]
    return (
        ggplot(d, aes(x="window", y="n_distinct", group="plate", color="plate"))
        + geom_line(alpha=0.7)
        + scale_x_log10(breaks=[16, 64, 256, 1024, 4096, 16384])
        + scale_y_log10()
        + xlab("Window size [contiguous rows on disk]")
        + ylab(f"Distinct {label} values in the window")
        + labs(color="")
        + theme(figure_size=(6.4, 4.2), legend_position="right",
                legend_text=element_text(size=6))
    )


# --------------------------------------------------------------------------- #
# WGS epoch times and the pre-shuffling break-even point                       #
# --------------------------------------------------------------------------- #
WGS_N_OBS = 500_000
FIG2E_CSV = Path(__file__).resolve().parents[1] / "resources" / "wgs_fig2e_throughput.csv"


#: Result files ending in one of these are deliberately retained *wrong*
#: measurements: a budget too small to time, a cache regime that did not match
#: the benchmark, a warm run that warmed the wrong format. They document the size
#: of a mistake and must never be read into a figure or table.
SUPERSEDED_SUFFIXES = ("_shortbudget", "_polluted", "_cacheable", "_forktest")


def _remeasured_wgs_throughput() -> pd.DataFrame:
    """Loader throughputs measured during the revision, where they exist.

    Preferred over the Fig. 2e numbers because the pre-shuffling wall-clock was
    also measured here: the break-even point is a ratio of the two, so both
    sides should come from nodes of the same specification.
    """
    rows = []
    common = RESULTS / "40_wgs_memmap_baselines_bs128.json"
    if common.exists():
        blob = json.loads(common.read_text())
        name = {"annbatch": "annbatch",
                "random_row_manuscript": "numpy memmap",
                "random_row": "numpy memmap (in-process)",
                "block_shuffled": "numpy memmap (block-shuffled)",
                "sequential": "numpy memmap (sequential)"}
        df = pd.DataFrame([{"loader": name.get(r["loader"], r["loader"]),
                            "samples_per_sec": r["samples_per_sec"]} for r in blob["results"]])
        agg = df.groupby("loader", as_index=False)["samples_per_sec"].mean()
        agg["regime"] = "common"
        rows.append(agg)
    for path in sorted(RESULTS.glob("42_wgs_throughput_*.json")):
        # Superseded measurements are kept on disk on purpose (the gap between
        # them and the corrected run is the size of the error), but they must
        # never reach a figure: globbing one in produced duplicate regime rows
        # and made `ann[regime]` a Series instead of a scalar.
        if any(path.stem.endswith(sfx) for sfx in SUPERSEDED_SUFFIXES):
            continue
        blob = json.loads(path.read_text())
        df = pd.DataFrame([{"regime": r["params"]["regime"],
                            "loader": {"annbatch": "annbatch", "scdataset": "scDataset",
                                       "mapped_collection": "MappedCollection"}[r["loader"]],
                            "samples_per_sec": r["samples_per_sec"]} for r in blob["results"]])
        rows.append(df.groupby(["regime", "loader"], as_index=False)["samples_per_sec"].mean())
    if not rows:
        return pd.DataFrame(columns=["regime", "loader", "samples_per_sec"])
    out = pd.concat(rows, ignore_index=True)
    out["source"] = "revision (re-measured)"
    return out


def wgs_epoch_table(fig2e_csv: str | Path = FIG2E_CSV, *, prefer_remeasured: bool = True) -> pd.DataFrame:
    """Epoch time and pre-shuffling break-even for every WGS regime.

    Loader throughputs come from the revision's own measurements where they
    exist and from the stored Fig. 2e notebook outputs otherwise; pre-shuffling
    wall-clock comes from `41_wgs_preshuffle_timing.py`. The break-even point is
    `T_pre / (T_baseline - T_annbatch)`, the number of epochs after which
    pre-shuffling has paid for itself against that baseline, which is what
    Reviewer 2 (major concerns 5 and 6) asks for.
    """
    tput = pd.read_csv(fig2e_csv, comment="#")
    tput["source"] = "Fig. 2e (as published)"
    if prefer_remeasured:
        fresh = _remeasured_wgs_throughput()
        if len(fresh):
            keep = ~tput.set_index(["regime", "loader"]).index.isin(
                fresh.set_index(["regime", "loader"]).index)
            tput = pd.concat([tput[keep], fresh], ignore_index=True)
    tput["epoch_hours"] = WGS_N_OBS / tput["samples_per_sec"] / 3600

    pre = wgs_preshuffle_table().set_index("regime")["preshuffle_hours"]
    ann = tput[tput.loader == "annbatch"].set_index("regime")["epoch_hours"]

    rows = []
    for _, r in tput[tput.loader != "annbatch"].iterrows():
        regime = r["regime"]
        if regime not in pre.index:
            continue
        t_ann = ann[regime]
        if not np.isscalar(t_ann) and getattr(t_ann, "size", 1) != 1:
            raise ValueError(
                f"{regime!r} has {t_ann.size} annbatch throughputs, expected one. "
                "A superseded result file is probably being globbed in; see "
                "SUPERSEDED_SUFFIXES.")
        t_ann = float(t_ann)
        t_alt, t_pre = r["epoch_hours"], float(pre[regime])
        rows.append({
            "regime": regime,
            "baseline": r["loader"],
            "source": r.get("source", ""),
            "epoch_h_annbatch": t_ann,
            "epoch_h_baseline": t_alt,
            "preshuffle_h": t_pre,
            "break_even_epochs": t_pre / (t_alt - t_ann) if t_alt > t_ann else float("inf"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Combined operating curve: what to set chunk_size to, and why                 #
# --------------------------------------------------------------------------- #
def operating_curve_table(diversity: pd.DataFrame, throughput: pd.DataFrame,
                          regime: str = "fixed_buffer") -> pd.DataFrame:
    """Throughput and minibatch diversity on one chunk-size axis.

    Both quantities are expressed as a fraction of their ideal so that they can
    share an axis: throughput relative to its saturated value, and minibatch
    entropy relative to true random sampling.  Where the two curves cross is the
    whole parameter-selection question, and pre-shuffling is what removes the
    trade-off.
    """
    tp = (
        throughput[(throughput.loader == "annbatch") & (throughput.regime == regime)]
        .groupby("block_size", as_index=False)["samples_per_sec"].mean()
    )
    tp["throughput_frac"] = tp["samples_per_sec"] / tp["samples_per_sec"].max()

    rows = []
    for family in diversity.family.unique():
        d = diversity[diversity.family == family].dropna(subset=["block_size"])
        reference = diversity["global_entropy"].iloc[0]
        m = d.merge(tp, on="block_size", how="inner")
        for _, r in m.iterrows():
            rows.append({
                "block_size": int(r["block_size"]),
                "family": family,
                "samples_per_sec": r["samples_per_sec"],
                "throughput_frac": r["throughput_frac"],
                "entropy": r["entropy"],
                "entropy_frac": r["entropy"] / reference,
            })
    return pd.DataFrame(rows)


def plot_operating_curve(df: pd.DataFrame):
    long = pd.concat([
        df.assign(quantity="loading throughput\n(fraction of saturated)",
                  value=df.throughput_frac, series="throughput"),
        df.assign(quantity="minibatch diversity\n(fraction of random sampling)",
                  value=df.entropy_frac, series=df.family),
    ])
    # throughput is identical across families, so keep one copy
    long = long[(long.series != "throughput") | (long.family == df.family.iloc[0])]
    return (
        ggplot(long, aes(x="block_size", y="value", color="series", linetype="quantity"))
        + geom_line()
        + geom_point(size=1.6)
        + scale_x_log10(breaks=[1, 16, 64, 256, 1024, 4096])
        + xlab("chunk_size [rows]")
        + ylab("Fraction of the ideal")
        + labs(color="", linetype="")
        + theme(figure_size=(7.0, 4.4), legend_position="right",
                legend_text=element_text(size=7))
    )


# --------------------------------------------------------------------------- #
# How good does the pre-shuffle have to be?                                    #
# --------------------------------------------------------------------------- #
def preshuffle_quality_table(fname: str = "06_preshuffle_quality.json",
                             label: str = "drug") -> pd.DataFrame:
    blob = _load(fname)
    ideal = blob["global_entropy_bits"][label]
    rows = [{
        "shuffle_chunk_size": r["shuffle_chunk_size"],
        "dataset_size": r["dataset_size"],
        "ratio": r["ratio"],
        "loader_chunk_size": r["loader_chunk_size"],
        "entropy": r["stats"][label]["entropy"]["mean"],
        "fraction_of_ideal": r["stats"][label]["entropy"]["mean"] / ideal,
        "n_distinct": r["stats"][label]["n_distinct"]["mean"],
    } for r in blob["results"]]
    df = pd.DataFrame(rows)
    df.attrs["ideal"] = ideal
    df.attrs["shipped"] = blob.get("shipped_shuffle_chunk_size")
    return df


def plot_preshuffle_quality(df: pd.DataFrame, label: str = "drug"):
    d = df.copy()
    d["loader_chunk_size"] = d["loader_chunk_size"].astype(str)
    return (
        ggplot(d, aes(x="shuffle_chunk_size", y="entropy", color="loader_chunk_size"))
        + geom_hline(yintercept=df.attrs["ideal"], linetype="dashed", color="black", size=0.4)
        + geom_line()
        + geom_point(size=2)
        + scale_x_log10(breaks=[1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
        + xlab("shuffle_chunk_size of the pre-shuffler [rows]")
        + ylab(f"Minibatch {label} entropy [bits]")
        + labs(color="loader\nchunk_size")
        + theme(figure_size=(6.4, 4.0), legend_position="right")
    )


# --------------------------------------------------------------------------- #
# Dataset size with the real loaders on their own formats                      #
# --------------------------------------------------------------------------- #
REAL_LOADER_LABELS = {
    "annbatch": "annbatch (zarr)",
    "mapped_collection": "MappedCollection (h5ad)",
    "scdataset_h5ad": "scDataset (h5ad)",
    "scdataset_zarr": "scDataset (zarr)",
}


def dataset_size_real_table(
    fnames: tuple[str, ...] = ("31_dataset_size_real_loaders_cold.json",
                               "31_dataset_size_real_loaders_warm.json",
                               "31_dataset_size_real_loaders_cold_zarr.json",
                               "31_dataset_size_real_loaders_warm_zarr.json",
                               "31_dataset_size_real_loaders_cold_annbatch_hirep.json"),
) -> pd.DataFrame:
    """Throughput vs dataset size for MappedCollection, scDataset and annbatch.

    scDataset-on-Zarr is measured in a separate process from the other three
    loaders, so its results arrive in their own file. In one process that arm
    deadlocks: by the time it runs, the interpreter has imported lamindb
    (Django, psycopg2) for MappedCollection and holds open h5py handles from the
    h5ad arm, and forking DataLoader workers on top of that never returns.
    Measured in a clean process it reaches 75,064 samples/s.

    This supersedes `throughput_dataset_table`, which used annbatch's own
    `chunk_size=1` path as a stand-in for MappedCollection. That stand-in was
    saturated by per-request dispatch and could not resolve a page-cache effect;
    these are the real loaders on the formats they are actually used with.
    """
    rows = []
    for fname in fnames:
        try:
            blob = _load(fname)
        except FileNotFoundError as exc:
            print(f"skipping: {exc}")
            continue
        cache = "warm" if blob.get("warm_cache") else "cold"
        for r in blob["results"]:
            p = r["params"]
            rows.append({
                "loader": REAL_LOADER_LABELS.get(r["loader"], r["loader"]),
                "format": p.get("format", ""),
                "cache": cache,
                "n_shards": p["n_shards"],
                "n_obs": p["n_obs"],
                "samples_per_sec": r["samples_per_sec"],
                "epoch_hours": r.get("epoch_hours"),
                "repeat": p.get("repeat", 0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["loader", "format", "cache", "n_shards", "n_obs"], as_index=False)
              .agg(samples_per_sec=("samples_per_sec", "mean"),
                   sd=("samples_per_sec", "std"),
                   epoch_hours=("epoch_hours", "mean")))


def dataset_size_degradation(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """How much each loader loses across the dataset-size range, per format.

    This is the headline of Reviewer 2's fifth point, so it is computed here
    rather than by hand: the ratio of throughput at the smallest measured size to
    the largest, per loader and cache condition, over whatever sizes that arm
    actually has. `n_sizes` and `max_cv` are reported alongside because the arms
    do not all reach 94.1M cells at the same time and annbatch's spread is wide
    enough that a ratio from few repeats should not be over-read.
    """
    if df is None:
        df = dataset_size_real_table()
    if not len(df):
        return pd.DataFrame()
    rows = []
    for (loader, fmt, cache), g in df.groupby(["loader", "format", "cache"]):
        g = g.sort_values("n_obs")
        if len(g) < 2:
            continue
        lo, hi = g.iloc[0], g.iloc[-1]
        cv = (g["sd"] / g["samples_per_sec"] * 100).max()
        rows.append({
            "loader": loader, "format": fmt, "cache": cache,
            "smallest_n_obs": int(lo["n_obs"]), "at_smallest": lo["samples_per_sec"],
            "largest_n_obs": int(hi["n_obs"]), "at_largest": hi["samples_per_sec"],
            "degradation_x": lo["samples_per_sec"] / hi["samples_per_sec"],
            "n_sizes": len(g), "max_cv_pct": cv,
        })
    return pd.DataFrame(rows).sort_values(["format", "loader", "cache"])


def plot_dataset_size_real(df: pd.DataFrame):
    """Throughput against training-set size, with the spread over repeats shown.

    The error bars are not decoration. annbatch's throughput on this Lustre
    filesystem varies 10-30% between identical repeats, and neither a longer
    measurement window nor a dedicated node removes it: the chunk-size sweep run
    with `--exclusive` still has a median coefficient of variation of 13.7%. The
    slower arms are far steadier (0.4-7.8%), being less sensitive to bandwidth
    jitter. Drawn as bare lines, annbatch's curve looks non-monotonic and invites
    reading structure into noise; the conclusion rests on the difference between
    a ~4x decline for h5ad-backed loaders and ~1.7x for Zarr-backed ones, which
    is far larger than this spread.
    """
    df = df.copy()
    df["lo"] = (df["samples_per_sec"] - df["sd"].fillna(0)).clip(lower=1)
    df["hi"] = df["samples_per_sec"] + df["sd"].fillna(0)
    return (
        ggplot(df, aes(x="n_obs", y="samples_per_sec", color="loader", linetype="cache"))
        + geom_line()
        + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.04, alpha=0.6)
        + geom_point(size=2)
        + scale_x_log10()
        + scale_y_log10()
        + xlab("Training set size [cells]")
        + ylab("Samples per second [1/s]")
        + labs(color="", linetype="page cache")
        + theme(figure_size=(7.2, 4.4), legend_position="right",
                legend_text=element_text(size=7))
    )


# --------------------------------------------------------------------------- #
# WGS: on-disk structure, and whether the shuffle is needed at all             #
# --------------------------------------------------------------------------- #
def wgs_structure_table(regimes=("common", "rare_maf_0.01", "rare_maf_0.001")) -> pd.DataFrame:
    rows = []
    for regime in regimes:
        try:
            blob = _load(f"09_wgs_ondisk_structure_{regime}.json")
        except FileNotFoundError:
            continue
        ideal = blob["global_individual_entropy_bits"]
        for r in blob["results"]:
            s = r["stats"]["individual"]
            rows.append({
                "regime": regime,
                "chunk_size": r["chunk_size"],
                "buffer_rows": r["params"].get("buffer_rows"),
                "entropy": s["entropy"]["mean"],
                "n_distinct": s["n_distinct"]["mean"],
                "batch_size": blob["batch_size"],
                "ceiling_bits": float(np.log2(blob["batch_size"])),
                "global_entropy": ideal,
                "n_individuals": blob["n_individuals"],
                "replication_period": blob["replication_period"],
            })
    return pd.DataFrame(rows)


def plot_wgs_structure(df: pd.DataFrame):
    ceiling = df["ceiling_bits"].iloc[0]
    return (
        ggplot(df, aes(x="chunk_size", y="entropy", color="regime"))
        + geom_hline(yintercept=ceiling, linetype="dashed", color="black", size=0.4)
        + geom_line()
        + geom_point(size=2)
        + scale_x_log10(breaks=[1, 4, 64, 128, 512, 2048])
        + xlab("chunk_size [rows]")
        + ylab("Minibatch individual-identity entropy [bits]")
        + labs(color="")
        + theme(figure_size=(6.4, 4.0), legend_position="bottom")
    )


def wgs_unshuffled_table(regimes=("rare_maf_0.001", "rare_maf_0.01", "common")) -> pd.DataFrame:
    """Does the shuffle buy anything for WGS, or only the conversion to Zarr?"""
    rows = []
    for regime in regimes:
        try:
            blob = _load(f"43_wgs_unshuffled_zarr_{regime}.json")
        except FileNotFoundError:
            continue
        agg: dict[str, list[float]] = {}
        for r in blob["results"]:
            agg.setdefault(r["params"]["store"], []).append(r["samples_per_sec"])
        for store, vals in agg.items():
            rows.append({
                "regime": regime,
                "store": store,
                "samples_per_sec": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                "conversion_no_shuffle_h": blob["conversion_no_shuffle_h"],
            })
    return pd.DataFrame(rows)


def codec_control() -> pd.DataFrame:
    """How much of annbatch's advantage over the memmap baselines is the codec?

    One WGS dataset rewritten with compression removed, then the identical
    random-block access pattern driven over both copies: same data, chunking,
    sharding and reader, so the codec is the only difference.
    """
    blob = _load("46_codec_control.json")
    df = pd.DataFrame(blob["results"])
    out = (df.groupby("arm", as_index=False)
             .agg(rows_per_s=("rows_per_s", "mean"), sd=("rows_per_s", "std")))
    sizes = blob["sizes_bytes"]
    out["on_disk_GiB"] = out["arm"].map(lambda a: sizes[a] / 1024 ** 3)
    return out


def provenance_table(fnames: tuple[str, ...] = ()) -> pd.DataFrame:
    """Where every result came from: host, CPU, git revision, package versions.

    Every result JSON carries a `provenance` block. Surfacing it here means the
    notebook can show, per experiment, which machine and which code produced the
    numbers, rather than asking the reader to trust that they match.
    """
    names = fnames or tuple(sorted(f.name for f in RESULTS.glob("*.json")))
    rows = []
    for name in names:
        try:
            pr = _load(name).get("provenance", {})
        except FileNotFoundError:
            continue
        if not pr:
            continue
        rows.append({
            "result": name,
            "host": (pr.get("hostname") or "").split(".")[0],
            "cpu": (pr.get("cpu_model") or "")[:38],
            "git_rev": (pr.get("git_rev") or "")[:9],
            # Older results recorded this as a count, newer ones as the list of
            # paths; accept either rather than crashing on the mixed schema.
            "uncommitted": (_u if isinstance(_u := pr.get("git_uncommitted_paths"), int)
                            else len(_u or [])),
            "annbatch": (pr.get("packages") or {}).get("annbatch", ""),
            "zarr": (pr.get("packages") or {}).get("zarr", ""),
        })
    return pd.DataFrame(rows)


def preshuffle_summary() -> pd.DataFrame:
    """The Tahoe pre-shuffle itself: wall-clock, layout, and the permutation check.

    Joins `00_preshuffle_tahoe.json` (what was written) with
    `03_verify_preshuffle.json` (whether it is actually a permutation, and how
    well mixed). Every annbatch arm in the revision reads the store these two
    describe, so this is the foundation the rest of the numbers stand on.
    """
    run = _load("00_preshuffle_tahoe.json")
    ver = _load("03_verify_preshuffle.json")
    lay = run["layout"]
    disp, mix = ver["displacement"], ver["plate_mixing"]
    return pd.DataFrame([
        ("input stores (plates)", run["n_input_stores"]),
        ("rows written", f"{run['n_obs_total']:,}"),
        ("wall-clock (h)", round(run["elapsed_h"], 3)),
        ("layout: rows per chunk", lay["n_obs_per_chunk"]),
        ("layout: rows per shard", f"{lay['shard_size_obs']:,}"),
        ("layout: rows per dataset", f"{lay['dataset_size_obs']:,}"),
        ("layout: shuffle_chunk_size", f"{lay['shuffle_chunk_size']:,}"),
        ("mean non-zeros per row", f"{lay['mean_nnz_per_row']:,}"),
        ("is an exact permutation", ver["is_exact_permutation"]),
        ("mean displacement / uniform", round(disp["ratio_to_uniform"], 4)),
        ("plate TV distance", round(mix["tv_distance_mean"], 5)),
        ("  vs uniform control", round(mix["uniform_control_mean"], 6)),
        ("  ratio (predicted sqrt(1000) = 31.6)", round(mix["ratio_to_uniform"], 2)),
    ], columns=["property", "value"])


def label_tables_summary() -> pd.DataFrame:
    """Class counts per task, and how many appear in each store.

    The test plate holds only 95 of the 380 drugs, which is why the held-out-plate
    metrics are reported separately from the i.i.d. ones.
    """
    b = _load("01_label_tables.json")
    rows = []
    for task, n in b["classes"].items():
        r = {"task": task, "classes (global)": n}
        for store, info in b["stores"].items():
            r[f"present in {store}"] = info["classes_present"][task]
        rows.append(r)
    return pd.DataFrame(rows)


def eval_sets_summary() -> pd.DataFrame:
    """The two fixed held-out evaluation sets."""
    b = _load("02_eval_sets.json")
    rows = []
    for name in ("val_iid", "val_plate"):
        e = b[name]
        rows.append({"set": name, "n": e["n"], "drawn from": e["drawn_from"],
                     **{f"{k} classes": v for k, v in e["classes_present"].items()}})
    return pd.DataFrame(rows)


def verification_summary() -> pd.DataFrame:
    """The correctness gates, and whether each passed.

    These are not results but preconditions: if the pre-shuffle were not a
    permutation, or a loader's batch rows did not match the indices it yielded, or
    the h5ad shards were not a faithful copy of the Zarr store, then every number
    downstream would be measuring something other than what it claims.
    """
    rows = []
    v = _load("03_verify_preshuffle.json")
    rows.append({"gate": "pre-shuffle is an exact permutation",
                 "scope": f"{v['n_obs']:,} rows, {v['n_datasets']} datasets",
                 "passed": bool(v["is_exact_permutation"]), "failures": 0})
    l = _load("04_verify_loaders.json")
    rows.append({"gate": "batch row i == store row at yielded index idx[i]",
                 "scope": f"{len(l['results'])} strategies x "
                          f"{l['n_batches_per_strategy']} batches @ bs={l['batch_size']}",
                 "passed": not l["failures"], "failures": len(l["failures"])})
    h = _load("44_verify_h5ad_shards.json")
    rows.append({"gate": "h5ad shard k == Zarr dataset k, same rows same order",
                 "scope": f"{h['n_shards_checked']} shards, {h['total_rows']:,} rows, "
                          f"{h['total_nnz']:,} non-zeros",
                 "passed": not h["failures"], "failures": len(h["failures"])})
    return pd.DataFrame(rows)


def wgs_unshuffled_detail(regime: str = "rare_maf_0.001") -> pd.DataFrame:
    """Is the *shuffle* needed for WGS, or only the conversion to Zarr?

    Converts the source h5ad to Zarr **without** shuffling and benchmarks the
    same loaders against it. Relevant because this cohort is a 157-fold
    replication of 3,202 individuals, so the unshuffled on-disk order already
    yields distinct individuals within any contiguous block.
    """
    b = _load(f"43_wgs_unshuffled_zarr_{regime}.json")
    df = pd.DataFrame(b["results"])
    out = (df.groupby("loader", as_index=False)
             .agg(samples_per_sec=("samples_per_sec", "mean"),
                  sd=("samples_per_sec", "std")))
    out.attrs["conversion_no_shuffle_h"] = b["conversion_no_shuffle_h"]
    out.attrs["unshuffled_GiB"] = b["unshuffled_bytes"] / 1024 ** 3
    return out


def microscopy_preshuffle() -> dict:
    blob = _load("08_microscopy_preshuffle.json")
    return {
        "n_inputs": len(blob["inputs"]),
        "input_GiB": blob["input_bytes"] / 1024 ** 3,
        "output_GiB": blob["output_bytes"] / 1024 ** 3,
        "preshuffle_hours": blob["elapsed_h"],
        "layout": blob["layout"],
        "host": blob["provenance"]["hostname"],
        "cpu": blob["provenance"].get("cpu_model"),
    }
