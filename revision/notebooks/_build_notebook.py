"""Generate the two revision notebooks from the cell definitions below.

`revision_figures.ipynb` is everything: every experiment, every result, with the
method written above each panel. It is the working record.

`revision_reported.ipynb` is only what goes into the response and the
manuscript, organised one section per reviewer comment. It exists because the
full notebook is ~100 cells and a co-author reviewing what we will actually
submit should not have to sift the exploratory material out of it.

Keeping the notebook under version control as generated output means the cells
stay diffable: edit this file, re-run it, commit both.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
("md", """# annbatch revision: figures

Every panel added for the revision, from the JSON produced by
`revision/scripts/*.py`. Nothing is computed here: the notebook only tidies and
plots, so it runs on a laptop in seconds once the results are in place.

Colours and theme match `figure_plots/paper_figures.ipynb` so these panels drop
straight into the existing figures.

| section | reviewer comment |
|---|---|
| 1. Minibatch diversity | R1-2, R2-3, R3-3, R3-4, R3-5 |
| 2. Throughput vs chunk size | R3-1 |
| 3. Throughput vs dataset size, and break-even | R2-5 |
| 4. Convergence: training and validation curves, final metrics | **R2-3** |
| 5. WGS memmap baselines | **R2 minor 3** |
| 6. WGS pre-shuffling wall-clock | R2-6 |
"""),

("code", """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import figures
from figures import PALETTE, RESULTS, FIGURES

figures.setup()
print("results:", RESULTS)
print("figures:", FIGURES)
sorted(p.name for p in RESULTS.glob("*.json"))"""),

("md", """## 0. What this rests on: the store, the labels, and the correctness gates

Before any measurement, four things have to be true. They are checked by
dedicated scripts rather than assumed, and the gates are shown here because if
any failed, every number below would be measuring something other than what it
claims.

### 0a. The pre-shuffled store

`00_preshuffle_tahoe.py` runs annbatch's own pre-shuffler over the 13 training
plates. `03_verify_preshuffle.py` then reads the result back and checks it is an
exact permutation of the input, and quantifies how well mixed it is.

The store this produces is what **every** annbatch arm in the revision reads."""),

("code", """figures.preshuffle_summary()"""),

("md", """Two numbers in that table are worth dwelling on.

`mean displacement / uniform = 1.002` says rows moved as far as a uniform random
permutation would move them, so the shuffle is not merely local.

The plate total-variation distance is 0.030 against a uniform control of 0.0009,
a ratio of **31.9**. That is not a defect: the pre-shuffler reads in chunks of
`shuffle_chunk_size = 1000`, so residual correlation should scale as
`sqrt(1000) = 31.6`. Measured 31.9 against predicted 31.6 means the shuffler
behaves exactly as its design implies, which is also why section 1c can show how
far that parameter can be relaxed."""),

("md", """### 0b. The classification tasks

`01_build_label_tables.py` builds one global vocabulary per task across all 14
plates, so that a class index means the same thing in training and in both
evaluation sets."""),

("code", """figures.label_tables_summary()"""),

("md", """The test plate contains only **95 of the 380 drugs**. That is why
held-out-plate metrics are reported separately from i.i.d. ones throughout, and
why macro-F1 on the plate hold-out is not comparable in absolute terms to the
i.i.d. figure."""),

("code", """figures.eval_sets_summary()"""),

("md", """Both evaluation sets are drawn once, with a fixed seed, and the
`val_iid` rows are **excluded from training in every arm**, enforced through the
`global_row` column that the pre-shuffler writes, so the exclusion survives the
permutation.

### 0c. The correctness gates"""),

("code", """figures.verification_summary()"""),

("md", """The middle gate is the one that caught a real bug. scDataset's
`batch_callback` receives `(data, positions)` and sorts its fetch indices
internally; an earlier version of our adapter assumed `(batch)` and would have
silently paired rows with the wrong labels, producing plausible-looking
convergence curves that were measuring nothing. `04_verify_loaders.py` reads rows
back off disk and asserts that batch row *i* is the store row at the index the
loader yielded, for every strategy.

### 0d. Provenance

Every result JSON records the host, CPU, git revision and package versions that
produced it. Rather than asserting the runs are consistent, here they are."""),

("code", """prov = figures.provenance_table()
print("distinct annbatch versions:", sorted(set(prov.annbatch) - {""}))
print("distinct zarr versions    :", sorted(set(prov.zarr) - {""}))
print("results with uncommitted code at run time:", int((prov.uncommitted > 0).sum()),
      "of", len(prov))
prov.head(12)"""),

("md", """Some early results were produced while the revision code was still
uncommitted, which the `uncommitted` column records. The code is committed now
and byte-identical to what produced them; the containing revision is recorded in
`docs/FULL_PACKAGE.md` section 8.6. This is the one provenance gap deliberately
not closed by re-running, since regenerating ~100 GPU-hours of convergence runs
to change a string would buy nothing."""),

("md", """## 1. Minibatch diversity

> *R1-2, R2-3, R3-3, R3-4, R3-5: does chunked sampling give minibatches as
> diverse as uniform random sampling?*

**How this is measured, and why it can be done at full scale.** A sampling
strategy is fully determined by the *index stream* it emits. So
`10_batch_diversity.py` runs each sampler over the real 94,129,984-row collection,
collects the indices it yields, and looks up labels from the cached tables,
without reading `X` at all. That makes the measurement exact at full scale rather
than extrapolated from a subset, and it is why these numbers carry no page-cache
or filesystem caveats: no bulk I/O happens.

Three quantities per minibatch, averaged over batches: Shannon entropy of the
label distribution, the number of distinct labels, and total-variation distance
from the global label distribution. Entropy is the one plotted; the ceiling is
`min(global entropy, log2(batch_size))`.

The comparison below holds the in-memory buffer fixed at 16,384 rows for both
loaders: annbatch's shipped default (`chunk_size=512 × preload_nchunks=32`) and
also scDataset's own recommended setting (`block_size=16`, `fetch_factor=256` at
minibatch 64). Matching the buffer is the point: minibatch diversity depends on
how many *independent on-disk blocks* the buffer holds, i.e.
`buffer_rows / block_size`, not on either parameter alone."""),

("code", """div = figures.diversity_table(
    {
        "annbatch (not pre-shuffled)": "10_batch_diversity_unshuffled_buf16384.json",
        "scDataset": "10_batch_diversity_unshuffled_buf16384.json",
        "annbatch (pre-shuffled)": "10_batch_diversity_preshuffled_buf16384.json",
    },
    label="drug",
)
# one file supplies both unshuffled families; split them by strategy name
div = div[
    ((div.family == "annbatch (not pre-shuffled)") & div.strategy.str.startswith("annbatch"))
    | ((div.family == "scDataset") & div.strategy.str.startswith("scdataset"))
    | (div.family == "annbatch (pre-shuffled)")
]
div.sort_values(["family", "block_size"])[
    ["family", "strategy", "block_size", "buffer_rows", "entropy", "n_distinct", "tv_distance"]
]"""),

("code", """p_div = figures.plot_diversity(div, label="drug")
figures.save(p_div, "revision_diversity_drug", width=6.4, height=4.0)
p_div"""),

("md", """The dashed rule is the global drug-label entropy (8.420 bits), i.e. what true
random sampling attains.

Two readings:

* **annbatch and scDataset lie on top of each other** at every block size. They
  are the same algorithm, cluster sampling followed by an in-memory reshuffle,
  so any throughput difference between the tools comes from storage and
  implementation, not from one of them sampling less randomly. This answers
  R3-5 and the parameter-correspondence part of R2-2 directly.
* **On pre-shuffled data the curve is flat** at the random-sampling value for
  every chunk size. That is what the pre-shuffling step buys, and it is what
  licenses the large chunk sizes used in Fig. 2."""),

("code", """# The same picture for the plate label (the outermost on-disk block structure)
# and for the number of distinct compounds per minibatch.
div_plate = figures.diversity_table(
    {
        "annbatch (not pre-shuffled)": "10_batch_diversity_unshuffled_buf16384.json",
        "annbatch (pre-shuffled)": "10_batch_diversity_preshuffled_buf16384.json",
    },
    label="plate",
)
p_plate = figures.plot_diversity(div_plate, label="plate")
figures.save(p_plate, "revision_diversity_plate", width=6.4, height=4.0)
p_plate"""),

("md", """### The cost of *not* pre-shuffling, in memory

scDataset's default `fetch_factor=256` at minibatch 4,096 implies a buffer of
1,048,576 rows, 64× annbatch's default. At that buffer, block shuffling does
hold near-random diversity out to block size 1024, but it does so by keeping
~9 GB of Tahoe resident. Pre-shuffling reaches the same diversity with a
16,384-row buffer."""),

("code", """div_big = figures.diversity_table(
    {"scDataset": "10_batch_diversity_unshuffled_buf1048576.json"}, label="drug"
)
div_big = div_big[div_big.strategy.str.startswith("scdataset")]
comparison = (
    div[div.family != "scDataset"]
    .assign(buffer="16,384 rows")
    .merge(div_big.assign(buffer="1,048,576 rows"), how="outer")
)
comparison.pivot_table(index="block_size", columns=["family", "buffer"], values="entropy")"""),

("md", """## 1b. Why the curve has the shape it has: on-disk autocorrelation

The parameter guidance is only transferable if users can map it onto their own
data, and the controlling property belongs to the *dataset*: how long a
contiguous run of one label is on disk (`revision/scripts/05_ondisk_structure.py`).

In twelve of the fourteen Tahoe plates the data is sorted by compound, so a
single drug occupies a contiguous run of 50,000-230,000 cells; plates 3 and 10
are the exception, with runs of ~380-540. A 512-row chunk therefore contains
exactly one compound on twelve plates."""),

("code", """struct = figures.ondisk_structure_table()
(
    struct[struct.window == 512]
    .pivot_table(index="plate", columns="label", values=["run_length_median", "n_distinct"])
    .round(1)
)"""),

("code", """p_struct = figures.plot_ondisk_structure(struct, label="drug")
figures.save(p_struct, "revision_ondisk_structure_drug", width=6.4, height=4.2)
p_struct"""),

("md", """This is the mechanism behind section 1: with a 16,384-row buffer, the number of
distinct compounds in a minibatch is essentially the number of *chunks* in the
buffer, because each chunk falls inside one compound's run: 32 chunks at
`chunk_size=512` against 33.9 distinct compounds measured, 16 chunks at
`chunk_size=1024` against 19.6 measured.

The rule generalises: **degradation sets in once `chunk_size` approaches the
on-disk run length of the label you care about, and the recoverable diversity is
bounded by `buffer_rows / chunk_size`.** Pre-shuffling sets the run length to 1
and removes the constraint."""),

("md", """### How good does the pre-shuffle itself have to be?

The pre-shuffler has two knobs: `dataset_size`, the rows held in memory before a
shuffled block is written, and `shuffle_chunk_size`, the length of the contiguous
runs it reads. Their ratio is the effective number of independent pieces each
output dataset is assembled from.

`revision/scripts/06_preshuffle_quality.py` simulates the pre-shuffler on the
label array (its effect on an ordering is a permutation, so this needs no data)
and measures the resulting minibatch diversity at the real scale of 94 million
rows."""),

("code", """pq = figures.preshuffle_quality_table()
print(f"ideal (global) entropy: {pq.attrs['ideal']:.3f} bits; "
      f"shipped shuffle_chunk_size: {pq.attrs['shipped']}")
pq.pivot_table(index=["shuffle_chunk_size", "ratio"], columns="loader_chunk_size",
               values=["entropy", "fraction_of_ideal"]).round(4)"""),

("code", """p_pq = figures.plot_preshuffle_quality(pq)
figures.save(p_pq, "revision_preshuffle_quality", width=6.4, height=4.0)
p_pq"""),

("md", """The pre-shuffler's read granularity can be raised by **three orders of
magnitude** over a perfect shuffle before minibatch diversity moves at all. The
usable rule is `dataset_size / shuffle_chunk_size >= ~2,000`, which keeps
minibatch entropy within 1% of a perfect shuffle; annbatch's default sits exactly
there (2**21 / 1,000 = 2,097), so it reads long contiguous runs, which is what
makes the pre-shuffling step itself fast, at no measurable cost in randomness."""),

("md", """## 2. Throughput vs chunk size (R3 comment 1)

> *Guidance on how to choose `chunk_size` and `preload_nchunks`.*

**How this was measured.** `30_throughput_sweeps.py` on the pre-shuffled store,
batch size 4,096, **1,000,000 samples per measurement, 3 repeats**, on a node held
with `--exclusive` so no other job shares it. The buffer is held constant at
16,384 rows while `chunk_size` varies, so `preload_nchunks` moves inversely
(`nchunks = buffer_rows / chunk_size`). The point is that diversity depends on
how many *independent on-disk blocks* the buffer holds, not on either parameter
alone. A first warm-up of 5 batches is discarded from every measurement to
exclude thread-pool spin-up and the first shard-index read.

**A caveat that applies to every absolute number in this section.** annbatch's
throughput on this Lustre filesystem varies 10-30% between identical repeats, and
`--exclusive` does not remove it: the median coefficient of variation in this very
sweep is 13.7%, with a maximum of 30.5%. The shape of the curve is solid; individual
values should be read as means over three repeats with that spread. The cause is
below the node level, in the shared filesystem."""),

("code", """tput = figures.throughput_chunk_table()
tput.sort_values(["loader", "regime", "block_size"])"""),

("code", """p_tput = figures.plot_throughput_chunk(tput)
figures.save(p_tput, "revision_throughput_chunk_size", width=6.4, height=4.0)
p_tput"""),

("code", """# Epoch time over the 94.1M training cells, the form the Discussion needs.
tput[tput.regime == "fixed_buffer"][
    ["loader", "block_size", "buffer_rows", "samples_per_sec", "epoch_hours"]
].sort_values(["loader", "block_size"])"""),

("md", """### The parameter-selection question in one panel

Throughput saturates with chunk size; minibatch diversity decays with it, but
only on data that has not been pre-shuffled. Overlaying the two on one axis is
the guidance Reviewer 3 asks for in comments 1 and 3: without pre-shuffling
there is a trade-off to navigate, and with it there is not."""),

("code", """tr_full = figures.speed_diversity_tradeoff("drug")
display(tr_full.round(2))
oc = figures.operating_curve_table(div, tput, regime="fixed_buffer")
p_oc = figures.plot_operating_curve(oc)   # kept for exploration; not a reported figure
p_oc"""),

("code", """# The same thing as a table: what you give up at each chunk size.
oc.pivot_table(index="block_size", columns="family",
               values=["throughput_frac", "entropy_frac"]).round(3)"""),

("md", """## 3. Throughput vs dataset size, and the break-even point (R2 major 5)

The reviewer's premise is that per-epoch loading time is roughly constant in
dataset size, so a fixed pre-shuffling cost is amortised over fewer and fewer
epochs as data grows. That does not hold for random-access loaders: they degrade
as the dataset outgrows the page cache, while chunked sequential access does
not.

The memory budget is the independent variable of this experiment, so the sweep
is run under a deliberate `--mem=64G` cgroup."""),

("code", """# Both cache states in one frame. Cold: each subset is read once, so nothing is
# ever re-read and caching cannot help at any size, which isolates raw access
# cost. Warm: the subset is streamed into the page cache first (up to 48 GiB of
# the 64 GiB cgroup), the steady state training reaches after an epoch.
dsize = figures.throughput_dataset_table()
dsize.pivot_table(index="n_obs", columns=["loader", "cache"], values="samples_per_sec").round(0)"""),

("code", """# Displayed, not saved: this proxy sweep is superseded by 3b's real loaders, and
# writing it to figures/ would put a figure on disk that no reviewer answer uses.
figures.plot_throughput_dataset(dsize)"""),

("md", """Both curves are flat, and that is the result. The advantage of chunked over
random access is **structural, not a caching effect**: one Zarr chunk of the
sparse `data` array holds 34,531 non-zeros, which at 1,439 non-zeros per cell is
exactly 24 rows, so fetching one row at random decompresses 24 to yield 1, a
24-fold read amplification, while a `chunk_size=512` read wastes 8% and
`chunk_size=1024` wastes 3%. Predicted ratio ~23x, measured 18-21x. Warming the
cache moves random access only from ~4,500 to ~5,500-6,200 samples/s.

Since that cost is paid per request rather than per byte of dataset, the
per-epoch advantage does not shrink as the dataset grows, which is precisely
the premise of Reviewer 2's amortisation argument."""),

("code", """import json

preshuffle_hours = json.loads((RESULTS / "00_preshuffle_tahoe.json").read_text())["elapsed_h"]
print(f"Tahoe pre-shuffling (plates 1-13, 94,129,984 cells): {preshuffle_hours:.2f} h")

# Pre-shuffling reads and writes every row once, so its cost is linear in row
# count: a subset must be charged its own share, not the whole store's.
be = figures.break_even(
    preshuffle_hours,
    dsize,
    annbatch_label="annbatch (chunk_size=512)",
    baseline_labels=[c for c in dsize.loader.unique() if c != "annbatch (chunk_size=512)"],
    cache="warm",
    n_obs_preshuffled=94_129_984,
)
be.round(3)"""),

("md", """`break_even_epochs` is `T_pre / (T_alt − T_ann)`, the number of epochs after
which pre-shuffling has paid for itself against each baseline, at each dataset
size.

It comes out **flat at 0.34–0.43 epochs across a 45-fold range of dataset
size**, which is the direct answer to R2-5. Both sides of the ratio scale
linearly with row count: the per-epoch saving because loader throughput is
independent of dataset size (the table above), and the pre-shuffling cost
because it touches every row once. The reviewer's concern that amortisation
weakens as datasets grow therefore does not hold here.

Note the two curves above are flat in *both* cache states. We had expected
random access to degrade as the data outgrew the page cache; making the whole
dataset resident buys it only ~30%, because per-row access into a sparse Zarr
store is limited by per-request overhead rather than by disk. That refutes a
mechanism we expected to find, and is reported as such."""),

("md", """### 3b. The same sweep with the real loaders on their own formats

The sweep above uses annbatch's own `chunk_size=1` path as a stand-in for
MappedCollection. That stand-in turned out to be saturated by per-request
dispatch: a fully cached 2.1M-cell subset gives the same throughput as an
uncached 94.1M-cell one, so it cannot resolve a page-cache effect and must not
be read as evidence about MappedCollection either way.

This is the real experiment: all 94,129,984 cells written to 90 gzip h5ad shards
(`07_write_h5ad_shards.py`, verified against the Zarr store by
`44_verify_h5ad_shards.py`), then swept with **`lamindb.MappedCollection` on
h5ad**, scDataset on h5ad, scDataset on Zarr, and annbatch on Zarr. Shard *k*
and Zarr dataset *k* cover the same global rows in the same order, so the
formats are compared on identical cells. The scDataset-on-Zarr arm is also the
format-isolation data point Reviewer 2's major concern 1 asks for."""),

("code", """try:
    real = figures.dataset_size_real_table()
    display(real.pivot_table(index="n_obs", columns=["loader", "cache"],
                             values="samples_per_sec").round(0))
    p_real = figures.plot_dataset_size_real(real)
    figures.save(p_real, "revision_dataset_size_real_loaders", width=7.2, height=4.4)
    display(p_real)
except Exception as e:
    print("real-loader sweep not finished yet:", e)"""),

("md", """**How this was run, and why each choice matters.**

* **Subsets.** The first *k* of 90 h5ad shards against the first *k* of the Zarr
  datasets, k in {1,2,4,8,16,32,64,90}. Shard *k* and dataset *k* hold the same
  global rows in the same order (asserted by `44_verify_h5ad_shards.py`), so the
  two formats see identical cells rather than merely equal-sized samples.
* **Storage tier.** Both stores are on the `ddn_ssd` Lustre pool, asserted at
  startup by `_common.assert_ssd()`. This is not cosmetic: the shard directory's
  location used to be *derived* from another config path, and had it resolved
  onto the `ddn_hdd` pool the sweep would have invented a 3-4x gap that looks
  exactly like a loader difference.
* **Memory.** `--mem=64G`, which is the axis of the experiment: the h5ad shards
  are 3.34 GiB each, so the working set crosses the allowance between 8 and 16
  shards.
* **Budget.** 2,000,000 samples per measurement, 3 repeats (10 more for annbatch,
  see below). An earlier sweep used 200,000, which is *two seconds* of work for
  annbatch at ~100,000 samples/s, far too short a window, and it produced 2x
  swings between adjacent sizes that looked like structure. Those runs are kept
  as `*_shortbudget.json`.
* **Cache states.** Cold (each subset read once) and warm (the subset streamed
  into the page cache first, half the budget to each format that any requested
  loader actually reads; warming a format no arm touches *evicts* the pages that
  matter, which is a mistake an earlier run made).
* **Process isolation.** `scdataset_zarr` runs as its own job. In one process it
  deadlocks: by the time that arm builds its DataLoader the interpreter has
  imported lamindb (Django, psycopg2) for MappedCollection and holds open h5py
  handles, and forking workers on top of that never returns. Measured alone it
  reaches ~75,000 samples/s. This is why the results arrive in four files."""),

("code", """# Degradation across the size range, per loader and cache state: the headline
# of R2 major 5. Computed from the JSONs, not transcribed.
deg = figures.dataset_size_degradation()
deg.round(3)"""),

("md", """`max_cv_pct` is shown deliberately. annbatch's throughput on this Lustre
filesystem varies 10-30% between identical repeats, and neither a longer
measurement window nor a dedicated node removes it; the chunk-size sweep run
with `--exclusive` still shows a median coefficient of variation of 13.7%. So the
annbatch row should be read as *flat, with no resolvable slope*, not as a precise
ratio; that is why it carries 13 repeats rather than 3.

The pattern that does survive the noise is the one the reviewer asked about: the
two h5ad-backed loaders lose a factor of 3-4 across the range, the two
Zarr-backed ones far less, and cold and warm agree arm for arm."""),

("code", """# The cache cliff: h5ad shards are 3.34 GiB each against a 64 GB allowance.
_h5 = real[(real.format == "h5ad") & (real.cache == "cold")].copy()
_h5["h5ad_working_set_GiB"] = _h5.n_shards * 3.34
_h5.pivot_table(index=["n_shards", "h5ad_working_set_GiB"],
                columns="loader", values="samples_per_sec").round(0)"""),

("md", """Both h5ad arms fall off a cliff between a 27 GiB and a 53 GiB working
set, exactly where the data stops fitting in the 64 GB allowance, and then
flatten once they are comprehensively out of cache. The Zarr arms show no cliff.

This confirms the reviewer's premise and locates it: residency governs random
single-row access into h5ad. Note that pre-warming does *not* repair it
(warm/cold is 0.96-1.04 at every size) because once the working set exceeds
capacity the warmed pages are evicted during the measurement. Caching sets the
answer through capacity, not through warming. Both facts hold at once, and
reading the warm/cold null as evidence *against* a cache explanation was a
mistake I made and corrected."""),

("code", """# Break-even at full scale against the real loaders (epoch hours from the sweep).
import statistics as _st2, collections as _c2, json as _j2

_e = {}
for _f in ("31_dataset_size_real_loaders_cold.json",
           "31_dataset_size_real_loaders_cold_zarr.json"):
    _b = _j2.loads((RESULTS / _f).read_text())
    _d = _c2.defaultdict(list)
    for _r in _b["results"]:
        if _r["params"]["n_obs"] > 94_000_000:
            _d[_r["loader"]].append(_r["epoch_hours"])
    _e.update({_k: _st2.mean(_v) for _k, _v in _d.items()})

_T_pre, _T_ann = 1.57, _e["annbatch"]
print(f"Tahoe pre-shuffling T_pre = {_T_pre} h;  "
      f"annbatch epoch = {_T_ann:.3f} h")
print()
for _k in ("mapped_collection", "scdataset_h5ad", "scdataset_zarr"):
    print(f"  {_k:<20} epoch {_e[_k]:>6.2f} h  ->  break-even "
          f"{_T_pre / (_e[_k] - _T_ann):.3f} epochs")"""),

("md", """Against `MappedCollection`, the baseline people actually use,
pre-shuffling repays itself in about 6% of a single epoch. An earlier draft of the
response quoted 0.22 epochs here, computed from the `chunk_size=1` proxy; the real
loader is stronger than the proxy suggested."""),

("md", """### 3c. WGS: does pre-shuffling buy any batch diversity there?

The 500,000-individual cohort is `ad.concat([adata_3202] * 157)`: 157
consecutive copies of the same 3,202 people, so row *i* and row *i* + 3,202 are
the same individual and a contiguous chunk of 512 rows holds 512 *distinct*
individuals. That is a completely different on-disk structure from Tahoe, and
it is measured rather than assumed."""),

("code", """wgs = figures.wgs_structure_table()
print(f"batch size {wgs.batch_size.iloc[0]}, ceiling log2(batch) = "
      f"{wgs.ceiling_bits.iloc[0]:.3f} bits; "
      f"{wgs.n_individuals.iloc[0]:,} individuals, replication period "
      f"{wgs.replication_period.iloc[0]:,}")
display(wgs.pivot_table(index="chunk_size", columns="regime",
                        values=["entropy", "n_distinct"]).round(3))
p_wgs = figures.plot_wgs_structure(wgs)
figures.save(p_wgs, "revision_wgs_ondisk_structure", width=6.4, height=4.0)
p_wgs"""),

("md", """Every chunk size sits on the `log2(128) = 7`-bit ceiling: a batch of 128 drawn
from 3,202 individuals cannot contain more than 128 distinct ones, and it
contains essentially all of them. **Chunked reading of the unshuffled WGS store
already yields maximum-diversity minibatches**, so pre-shuffling provides no
batch-diversity benefit for this benchmark and its case there rests entirely on
throughput.

That has a sharper consequence, tested in `43_wgs_unshuffled_zarr.py`: if the
shuffle buys no diversity and chunked throughput depends on the store layout
rather than on row content, then for a cohort with this layout one should convert
to Zarr *without* shuffling and skip the shuffle cost."""),

("code", """try:
    unshuf = figures.wgs_unshuffled_table()
    display(unshuf.round(1))
except Exception as e:
    print("unshuffled-Zarr test not finished yet:", e)"""),

("md", """### 3d. Microscopy pre-shuffling wall-clock

Reviewer 2's major concern 6 asks for the pre-shuffling time of the microscopy
benchmark too. Its original inputs live on a de.NBI instance unreachable from
this cluster, so the five Golgi `.h5sc` files are re-downloaded from the
scPortrait manuscript's own data share and the pre-shuffle re-run under a timer
with the notebook's exact layout."""),

("code", """try:
    display(figures.microscopy_preshuffle())
except Exception as e:
    print("microscopy pre-shuffle not finished yet:", e)"""),

("md", """## 4. Convergence (R2 major concern 3)

> *No evaluation of model performance or training quality.*

**The design.** `20_train_convergence.py` trains **eight independent models in one
pass**, a linear classifier and an MLP (512, 512, GELU) for each of four tasks
(cell line, drug, MoA broad, MoA fine), so that all eight see byte-identical
batches in the same order. Any difference between arms is then attributable to the
sampling strategy and nothing else.

* **Scale.** One full epoch of 94,129,984 cells, batch size 4,096 (22,981 steps),
  three seeds per arm. 61 full-epoch runs in total.
* **Normalisation.** `log1p(counts per 1e4)`, applied on the GPU.
* **Learning rate.** 1e-4, chosen by sweeping the *reference* (uniform random)
  arm; see 4c. This matters: linear scaling from scDataset's 1e-5 at batch 64
  would give 6.4e-3, which diverges, and the maximum usable learning rate itself
  depends on sampling diversity, so picking it per-arm would confound the
  comparison. A robustness check at 3e-4 over three seeds is in 4c.
* **Held-out rows.** The 65,536 `val_iid` cells are masked out of training in
  every arm via the `global_row` column, so the permutation cannot leak them.
* **Efficiency.** Losses accumulate on the GPU (`running += torch.stack(losses)`);
  calling `.item()` per head would force eight device synchronisations per step and
  cost ~5x the step time.
* **Wall-clock is not comparable across arms.** Per-step cost differs ~10x between
  this cluster's GPU models (an H100 step ~10 ms, an older card ~105 ms), so each
  result records `gpu_name`. Every loader-throughput claim comes from sections 2
  and 3, which attach no model at all."""),

("md", """### 4a. scDataset Figure 6, reproduced: training loss per task

**This is the panel the co-authors asked for** (F.F.: *"No need for these
benchmarks, we show that our pre-shuffling yields essentially randomly shuffled
dataset... then we refer to e.g. the CorgiPile paper -> reproduce scDataset
figure"*).

Note which figure is which in their paper: **Figure 6 is training loss curves**;
their macro-F1 comparison is **Figure 5**. Figure 6 is the one that shows the
*mechanism* rather than a performance ranking, so it is the primary panel here;
the macro-F1 material follows in 4c for completeness.

Their four strategies map onto our arms as follows. Their BlockShuffling(b=1) is
uniform random sampling. Their "streaming with a shuffle buffer of 16,384
(64 x 256)" is our `streaming_buffer`. Their BlockShuffling(b=16, f=256) holds
16,384 rows at their minibatch of 64; at our minibatch of 4,096 the matched
buffer is b=16, f=4. annbatch is the arm they did not have.

annbatch is drawn **dashed** so that Random Sampling underneath it stays visible:
the two curves being indistinguishable is the result, and a solid line drawn over
a solid line would hide exactly that."""),

("code", """fig6 = figures.figure6_loss_curves()   # seed 0; pass seed=None to overlay all
p_f6 = figures.plot_figure6(fig6, architecture="mlp")
figures.save(p_f6, "revision_scdataset_fig6_mlp", width=9.0, height=6.0)
p_f6"""),

("code", """p_f6_lin = figures.plot_figure6(fig6, architecture="linear")
figures.save(p_f6_lin, "revision_scdataset_fig6_linear", width=9.0, height=6.0)
p_f6_lin"""),

("md", """**What reproduces, and one thing that does not.**

Reproduced: streaming and streaming-with-a-buffer produce violently erratic loss
throughout, while BlockShuffling, Random Sampling and annbatch produce smooth
curves. Reproduced too: cell-line loss collapses immediately under *every*
strategy, because cell-line identity is preserved across plates and provides a
signal regardless of ordering, which is why reporting only that task would hide
the effect entirely.

Not reproduced as stated: scDataset attributes the spikes specifically to *plate
boundaries*. In our data the boundaries account for only part of them."""),

("code", """import pandas as _p6
_p6.concat([figures.figure6_spike_alignment(h)
            for h in ("mlp/drug", "mlp/moa_fine", "mlp/moa_broad", "linear/drug")],
           ignore_index=True)"""),

("md", """For the drug head the twelve largest loss excursions sit a median of
210 steps from the nearest plate boundary against 486 expected if they fell at
random, and five of the twelve land within 45 steps, i.e. within one logging
interval. So they *are* enriched at boundaries. But for the MoA heads the median
distance is 506-534 against the same null of 486, which is no enrichment at all.

The explanation is in section 1b: Tahoe is ordered by well and drug *within* each
plate, with drug runs of 50,000-230,000 rows in 12 of the 14 plates. At minibatch
4,096 that is a change of drug composition every 12-56 steps, so a sequential
reader meets distribution shift more or less continuously rather than only at the
12 plate transitions. The spikes are real and the catastrophic-forgetting reading
is right; the attribution to plate boundaries alone is too narrow, and we would
state it as *on-disk autocorrelation at every scale, plate boundaries being
merely the largest*."""),

("md", """### 4b. Figure 6 across chunk sizes: the answer to Reviewer 3, comment 1

> *"It takes time to perform the pre-shuffling step, but it allows use of larger
> chunks while retaining randomness. However, the authors have not explored this
> trade-off... How much larger can chunk sizes be made while preserving
> comparable mini-batch diversity?"*

Section 4a reproduces scDataset's figure at one setting per strategy, which shows
that streaming is bad and shuffling is good. That is their point, not the
reviewer's. The reviewer asks *where the trade-off begins*, so the same loss
curves, faceted by chunk size, with the reference arm repeated in every panel.

Note that scDataset appears in 4a only at `block_size=16`, which is smooth, so it
looked clean there by omission. `block_size` and `chunk_size` are the same
quantity, contiguous rows per read, so at matched buffer the two loaders are
directly comparable, and scDataset spikes exactly as annbatch does once the block
is large."""),

("code", """chunk6 = figures.figure6_chunk_sweep(head="mlp/drug")
p_chunk6 = figures.plot_figure6_chunk_sweep(chunk6)
figures.save(p_chunk6, "revision_fig6_chunk_sweep", width=9.0, height=6.5)
p_chunk6"""),

("code", """figures.figure6_chunk_summary(head="mlp/drug").round(3)"""),

("md", """**The answer, in one sentence: with pre-shuffling, chunk size can be
raised to at least 1024 with no measurable cost; without it, degradation is
already visible at 128.**

`mean_abs_deviation` is the mean distance of the logged loss from its own rolling
median: how erratic training is, independent of where it converges.

* **annbatch, pre-shuffled** is flat across the whole range: deviation 0.002-0.003
  and final loss 2.119-2.130 at chunk 16, 128, 512 and 1024, against 0.002 and
  2.126 for uniform random sampling. The curves lie on top of the reference in
  every panel. This is the payoff the reviewer is asking about, and it is why
  `chunk_size=512` is a safe default rather than a compromise.
* **annbatch, not pre-shuffled** degrades monotonically: 0.006 -> 0.018 -> 0.031
  -> 0.042, with final loss rising 2.230 -> 2.915 -> 3.739 -> 4.304. Chunk 16 is
  nearly fine; by 128 the gap is unmistakable.
* **scDataset** behaves the same way on the same data: 0.006 at block 16, 0.038 at
  512, 0.045 at 1024. At block 1024 it is statistically indistinguishable from
  annbatch without pre-shuffling (0.045 against 0.042), as it should be, since
  they are the same algorithm. Its remedy is a larger buffer, which costs memory:
  `b=16, f=32` holds 131,072 rows and returns to 0.006.

So the trade-off is not "pre-shuffling versus speed". It is: **buy diversity once
on disk, or buy it every epoch in RAM.** Pre-shuffling moves the cost from a
per-epoch memory budget to a one-off write, which is what makes large chunks,
and therefore the throughput in section 2, available at all."""),

("md", """### 4c. The rest of the convergence experiment

Everything below goes beyond scDataset's figure. It exists because Reviewer 2
asked for "training curves and final metrics", and because the chunk-size sweep is
what answers Reviewer 3's request for parameter guidance."""),

("code", """runs = figures.convergence_runs()
print(f"{len(runs)} runs")
import pandas as pd

pd.DataFrame([
    {
        "strategy": r["strategy"], "store": r["store"], "seed": r["seed"],
        "batch_size": r["batch_size"], "lr": r["lr"], "steps": r["steps"],
        "samples/s": round(r["mean_samples_per_sec"]), "hours": round(r["elapsed_s"] / 3600, 2),
    }
    for r in runs
]).sort_values(["batch_size", "strategy", "seed"])"""),

("md", """The single-panel training-loss plot that used to sit here has been
removed. With all nineteen arms on one axis its legend was larger than the data
and the streaming lines obscured everything else; 4a and 4b replace it, faceted
by task and by chunk size respectively.

The two curve panels below are restricted to the headline arms for the same
reason. `plot_headline_curves` draws seven arms with annbatch dashed over the
random reference, and takes `log_y`, worth setting for cross-entropy, where the
streaming arms sit an order of magnitude above the rest and would otherwise
compress everything else into the bottom fifth of the panel."""),

("code", """main = [r for r in runs if r["batch_size"] == 4096]
vc = figures.validation_curves(main, head="mlp/drug", which="val_iid", metric="macro_f1")
p_val = figures.plot_headline_curves(vc, ylabel="Validation macro-F1 (drug)")
figures.save(p_val, "revision_validation_f1_drug_iid", width=8.0, height=5.0)
p_val"""),

("code", """# Validation loss, same arms
vl = figures.validation_curves(main, head="mlp/drug", which="val_iid", metric="loss")
p_vl = figures.plot_headline_curves(vl, ylabel="Validation cross-entropy", log_y=True)
figures.save(p_vl, "revision_validation_loss_drug_iid", width=8.0, height=5.0)
p_vl"""),

("code", """# Final macro-F1 across all four tasks and both architectures
# (scDataset Fig. 5 equivalent)
fm = figures.final_metrics(main, which="val_iid", metric="macro_f1")
p_final = figures.plot_final_metrics(fm)
figures.save(p_final, "revision_final_macro_f1_iid", width=12.0, height=6.0)
p_final"""),

("code", """# The headline table: annbatch at chunk 1024 against the reference arms.
headline = fm[fm.display.isin([
    "random sampling", "streaming",
    "annbatch, pre-shuffled (chunk 1024)",
    "annbatch, not pre-shuffled (chunk 1024)",
    "scDataset (block 16)",
])]
(
    headline.groupby(["display", "architecture", "task"], as_index=False)
    .agg(macro_f1=("value", "mean"), sd=("value", "std"))
    .pivot_table(index="display", columns=["architecture", "task"], values="macro_f1")
    .round(4)
)"""),

("code", """# Learning-rate robustness: the same five arms at 3e-4 (the reference arm's own
# optimum) rather than the conservative 1e-4 used for the matrix.
try:
    alt = figures.convergence_runs("20_convergence_lr3e-4")
    display(
        figures.final_metrics(alt, which="val_iid", metric="macro_f1")
        .pivot_table(index="display", columns=["architecture", "task"], values="value")
        .round(4)
    )
except FileNotFoundError as e:
    print("learning-rate robustness arms not finished yet:", e)"""),

("code", """# Replication at scDataset's own protocol (minibatch 64, lr 1e-5)
repl = [r for r in runs if r["batch_size"] == 64]
if repl:
    display(figures.final_metrics(repl, which="val_plate", metric="macro_f1")
            .groupby(["display", "architecture", "task"], as_index=False)
            .agg(macro_f1=("value", "mean"))
            .pivot_table(index="display", columns=["architecture", "task"], values="macro_f1")
            .round(4))
else:
    print("replication arms not finished yet")"""),

("md", """## 5. WGS numpy-memmap baselines (R2 minor concern 3)

> *...does not specify whether the memmap is accessed sequentially or randomly,
> or whether the annbatch timing includes preshuffling.*

### What the reviewer is asking, and what we ran

The manuscript quotes a numpy-memmap baseline without saying how rows are drawn
from it. That matters enormously: the same array can be read four different ways
spanning a factor of 13 in throughput. So instead of picking one and calling it
"the" memmap baseline, `revision/scripts/40_wgs_memmap_baselines.py` implements
**four** and reports all of them.

**The array.** `common_variants_500000.memmap`, 502,714 x 1,200,000 `uint8` =
**603,256,800,000 bytes (562 GiB)**. It is the common-variant (MAF >= 1%) 1000
Genomes cohort, 3,202 real individuals replicated 157-fold. Not compressed, not
compressible in place; see the fairness discussion below.

**The node.** 16 vCPU, `--mem=64G`, `slurm/40_wgs_memmap_baselines.sbatch`. The
memory cap is deliberate and load-bearing: the array is **9.4x** the cgroup, so
at most ~11% of it can be resident. Had it fit in RAM the whole comparison would
measure nothing but page-cache speed. This is the same node specification used
for the pre-shuffling timings (section 6) and the epoch times, because the
break-even point is a ratio of those and mixing hardware would make it
meaningless.

**Sample budget.** 102,400 samples per measurement, 12,800 for the two
random-row arms (they are ~40x slower, so a matched budget would have cost hours
per repeat), 3 repeats each, batch size 128.

### The four variants, exactly as implemented

Reproduced here so there is no need to open the script to know what was
measured.

**(1) `random_row_manuscript`** is the manuscript's baseline, unmodified. One row
per `__getitem__`, `DataLoader(shuffle=True)`:

```python
def __getitem__(self, idx):
    return torch.from_numpy(self.mm[idx].astype(np.float32).copy())
```

The `float32` cast happens *inside the worker*, so 1.2M float32 = 4.8 MB per row
crosses the IPC boundary, 614 MB per batch of 128. This arm therefore measures
the published implementation, not the access pattern alone.

**(2) `random_row`** is the same access pattern, in-process, with no IPC round trip:

```python
order = rng.permutation(n_obs)
for s in range(0, n_obs - batch_size + 1, batch_size):
    rows = order[s : s + batch_size]
    buf = np.stack([mm[int(r)] for r in rows])
    yield torch.from_numpy(buf.astype(np.float32))
```

Comparing (1) and (2) separates *how the baseline was written* from *what it
does*.

**(3) `block_shuffled`** is the memmap analogue of annbatch's own sampler. Draw
`preload_nchunks` contiguous blocks of `chunk_size` rows at random offsets,
concatenate, shuffle the buffer in memory, emit minibatches:

```python
starts = np.arange(0, n_obs, chunk_size)
rng.shuffle(starts)
for i in range(0, len(starts), preload_nchunks):
    block_starts = starts[i : i + preload_nchunks]
    buf = np.concatenate([mm[s : min(s + chunk_size, n_obs)] for s in block_starts])
    perm = rng.permutation(len(buf))
    for j in range(0, len(buf) - batch_size + 1, batch_size):
        yield torch.from_numpy(buf[perm[j : j + batch_size]].astype(np.float32))
```

This mirrors `annbatch.samplers.RandomSampler` step for step, so the **only**
difference against the annbatch row is the storage layer. It is the fair
like-for-like comparison and the one to quote.

**(4) `sequential`** is pure streaming in on-disk order:

```python
for s in range(0, n_obs - batch_size + 1, batch_size):
    yield torch.from_numpy(mm[s : s + batch_size].astype(np.float32))
```

The hardware upper bound. Why it cannot be used for training is subtler than it
looks, and specific to this cohort; see the note after the table."""),

("code", """mm = figures.memmap_table()
mm"""),

("code", """p_mm = figures.plot_memmap(mm)
figures.save(p_mm, "revision_wgs_memmap_baselines", width=7.0, height=4.2)
p_mm"""),

("md", """### Why `sequential` is not a usable loader here

Not for the usual reason. The memmap is in **raw replication order**: we
verified directly on the array that row *i* is byte-identical to row *i*+3,202
while adjacent rows differ, so any 128 contiguous rows are 128 *distinct*
individuals and the batches are perfectly diverse.

What is missing is stochasticity *across* epochs: individual 0 is batched with
individuals 1..127 in every single epoch. Batch composition is frozen, and that
is what rules it out for SGD.

On a dataset with real on-disk autocorrelation, such as Tahoe-100M, sequential fails
the opposite way, by producing homogeneous batches (drug macro-F1 0.0001, see
section 4). Two different failure modes, the same lesson: on-disk order has to
be characterised per dataset, not assumed."""),

("code", """# Row i vs row i+3202: the replication structure, checked on the array itself.
import json as _json
import numpy as _np

_mmdir = Path("/lustre/boost_ai/users/selman.ozleyen/annbatch/common_memmap")
_meta = _json.loads((_mmdir / "common_variants_500000.meta.json").read_text())
_arr = _np.memmap(_mmdir / "common_variants_500000.memmap",
                  dtype=_meta["dtype"], mode="r", shape=tuple(_meta["shape"]))
_pd_rows = []
for _i in (0, 1, 7, 100):
    _pd_rows.append({
        "row": _i,
        f"identical to row+3202": bool(_np.array_equal(_arr[_i], _arr[_i + 3202])),
        "identical to neighbour": bool(_np.array_equal(_arr[_i], _arr[_i + 1])),
    })
import pandas as _pandas
_pandas.DataFrame(_pd_rows)"""),

("md", """### Is comparing an uncompressed memmap with a compressed Zarr store fair?

It is the obvious objection, and it deserves a measured answer rather than an
argument. Two parts.

**A memmap cannot be compressed and remain a memmap.** `np.memmap` gives O(1)
random row access precisely because row *i* sits at byte offset
`i x n_var x itemsize`. Compression makes block lengths variable, so the offset
is no longer computable and you need a chunk index plus a decode step, which is
what Zarr *is*. "Compressed memmap" is not a configuration we declined to test;
it is a different data structure. The 2.35x size difference is a consequence of
the format choice, not a handicap imposed on the baseline.

**And it is not where most of the advantage comes from.**
`revision/scripts/46_codec_control.py` rewrites one dataset of the WGS store with
the compression codec removed (same shape, same chunks, same shards, same
reader) and drives the identical random-block access pattern over both copies.
The codec is then the only difference."""),

("md", """**The cache regime has to match the benchmark, and at first it did not.**
The first run of this control used a single dataset: 8.4 GiB compressed and
18.3 GiB uncompressed against an 8 GB cgroup. That leaves the compressed copy at
1.05x oversubscribed, effectively resident, while the uncompressed one is
2.3x over, so the compressed arm enjoyed a cache advantage that exists only at
that cgroup size. The real WGS benchmark runs 257 GB and 603 GB against 64 GB,
i.e. 4.0x and 9.4x, where *neither* is cacheable.

The control was re-run with four datasets (33.5 / 73.2 GiB against 8 GB = 4.2x
and 9.1x), reproducing the benchmark's regime. Both runs are kept below; the
difference between them is the size of the mistake."""),

("code", """codec = figures.codec_control()
codec"""),

("code", """# The superseded run, for comparison: same script, wrong cache regime.
import json as _j, statistics as _s, collections as _c

_rows = []
for _tag, _f in [("matched (4 datasets)", "46_codec_control.json"),
                 ("cacheable (1 dataset)", "46_codec_control_cacheable.json")]:
    _b = _j.loads((RESULTS / _f).read_text())
    _d = _c.defaultdict(list)
    for _r in _b["results"]:
        _d[_r["arm"]].append(_r["rows_per_s"])
    _comp, _unc = _s.mean(_d["compressed"]), _s.mean(_d["uncompressed"])
    _rows.append({
        "regime": _tag,
        "compressed GiB": _b["sizes_bytes"]["compressed"] / 1024**3,
        "uncompressed GiB": _b["sizes_bytes"]["uncompressed"] / 1024**3,
        "compressed rows/s": _comp,
        "uncompressed rows/s": _unc,
        "codec benefit": _comp / _unc,
    })
import pandas as _pandas
_pandas.DataFrame(_rows).round(2)"""),

("code", """# Decompose the measured speed-ups into codec and reader contributions.
import statistics as _st

_c = codec.set_index("arm")["rows_per_s"]
codec_x = _c["compressed"] / _c["uncompressed"]
size_x = (codec.set_index("arm")["on_disk_GiB"]["uncompressed"]
          / codec.set_index("arm")["on_disk_GiB"]["compressed"])

_m = mm.set_index("loader")["samples_per_sec"] if "loader" in mm.columns else None
print(f"codec speed benefit : {codec_x:.2f}x   (on-disk size ratio {size_x:.2f}x)")
print("decomposition of the memmap comparisons:")
for _name, _total in [("sequential", 3.014), ("block_shuffled", 5.343)]:
    print(f"  vs {_name:<16} {_total:.2f}x total = "
          f"{codec_x:.2f}x codec x {_total / codec_x:.2f}x reader")"""),

("md", """Removing compression costs **1.55x**, well short of the 2.19x size
ratio: decompression recovers only about 70% of the byte saving, the rest going
to CPU.

The consequence is the part that matters. With compression removed altogether,
annbatch remains **1.94x** faster than pure sequential streaming of the memmap
and **3.44x** faster than the block-shuffled memmap. The conclusion does not
depend on compression at all, and the corrected, lower codec figure makes the
reader's contribution *larger* than the first run suggested."""),

("md", """## 6. WGS pre-shuffling wall-clock (R2 major concern 6)

Measured by re-running the pre-shuffling step of
`src/benchmark_loading/genetic_data_prep.py` under a timer, at the identical
on-disk layout (`revision/scripts/41_wgs_preshuffle_timing.py`)."""),

("code", """figures.wgs_preshuffle_table()"""),

("md", """Each regime was re-measured by re-running the pre-shuffling step of
`src/benchmark_loading/genetic_data_prep.py` under a timer at the identical
on-disk layout, on the same 16 vCPU / 64 GB node specification as the epoch times.
The break-even point is a ratio of the two, so mixing hardware between numerator
and denominator would make it meaningless.

The `--max-maf 0.01` regime needed 1500 GB of memory: it holds 2^17 rows of
~224,000 non-zeros at once, about 264 GB per buffer before the concatenation copy.
A first attempt at 750 GB was OOM-killed at 50%."""),

("md", """### 6b. Epoch time and break-even per regime"""),

("code", """figures.wgs_epoch_table()"""),

("md", """### 6c. Is the *shuffle* needed for WGS, or only the conversion to Zarr?

This cohort is a 157-fold replication of 3,202 real individuals concatenated in
order, so any contiguous block already contains distinct individuals, unlike
Tahoe. `43_wgs_unshuffled_zarr.py` therefore converts the source h5ad to Zarr
**without** shuffling and benchmarks the same loaders against it, to separate the
benefit of the format from the benefit of the permutation."""),

("code", """unshuf = figures.wgs_unshuffled_detail()
print(f"conversion without shuffling: {unshuf.attrs['conversion_no_shuffle_h']:.2f} h "
      f"for {unshuf.attrs['unshuffled_GiB']:.0f} GiB")
unshuf.round(1)"""),

("md", """So for this benchmark the case for pre-shuffling rests on **throughput,
not batch diversity**: the on-disk order is already uncorrelated with respect to
sample identity. That is a caveat we think belongs in the Methods, and it is
specific to the synthetic replication used to reach 500,000 individuals; it would
not hold for a real cohort of that size."""),

("md", """### Note on the MAF thresholds

`genetic_data_prep.py` passes `--max-maf 0.01` and `--max-maf 0.001` to PLINK2,
i.e. MAF < 1% and MAF < 0.1%. The manuscript describes the same two datasets as
"MAF < 0.1%" and "MAF < 0.01%", a factor of ten too small in both cases. The
variant counts quoted (~90M and ~72M) do identify the datasets unambiguously and
match what is on disk (92,087,087 and 74,589,081), so only the thresholds need
correcting."""),
]


REPORTED_CELLS: list[tuple[str, str]] = [
("md", """# annbatch revision

| section | comment | what is reported |
|---|---|---|
| 1 | R2 major 3 | training-loss reproduction of scDataset Fig. 6 (drug), and the final macro-F1 metrics |
| 2 | R3 comment 1 | the same curves faceted by chunk size, plus throughput vs chunk size |
| 3 | R3 comment 3 | minibatch diversity, the two parameter rules, and the few-epoch regime |
| 4 | R2 minor 3 | the four memmap baselines |
| 5 | R2 major 6 | pre-shuffling wall-clock for WGS and microscopy |
"""),

("code", """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import figures
from figures import RESULTS, FIGURES

figures.setup()
print("results:", RESULTS)"""),

("md", """## 1. R2 major 3: model convergence

* reported panel: drug task
* cell line is learned equally well by every strategy -> serves
  as negative control
* `MappedCollection`: uniform random sampling -> equal to Random Sampling"""),

("code", """fig6 = figures.figure6_loss_curves()
p_drug = figures.plot_figure6(fig6, architecture="mlp", tasks=("drug",), ncol=1)
figures.save(p_drug, "revision_scdataset_fig6_drug", width=8.5, height=5.2)
p_drug"""),

("code", """import pandas as pd
_drug = fig6[(fig6.architecture == "mlp") & (fig6.task == "drug")]
_lo = _drug.groupby("strategy", observed=True).loss.min().round(3)
_f1 = (figures.final_metrics(
           [r for r in figures.convergence_runs() if r["batch_size"] == 4096],
           "val_iid", "macro_f1")
       .query("architecture == 'mlp' and task == 'drug'")
       .groupby("strategy", observed=True).value.mean().round(4))
_name = _drug[["strategy", "display"]].drop_duplicates().set_index("strategy").display
_inv = pd.DataFrame({"min training loss": _lo, "held-out macro-F1": _f1}).dropna()
_inv.rename(index=_name).rename_axis(None).sort_values("min training loss")"""),

("md", """### Final metrics (the second half of R2 major 3)"""),

("code", """runs = [r for r in figures.convergence_runs() if r["batch_size"] == 4096]
fm = figures.final_metrics(runs, "val_iid", "macro_f1")
_mlp = fm[fm.architecture == "mlp"]
# Mean and s.d. over seeds: the s.d. is what the non-inferiority margin rests on.
_agg = _mlp.groupby(["strategy", "task"]).value.agg(["mean", "std", "size"])
_agg.round(4).unstack("task")["mean"].round(4)"""),

("code", """# Drug, with the dispersion the margin argument uses, and the class counts.
print("classes per task:", runs[0]["tasks"])
_agg.xs("drug", level="task")[["mean", "std", "size"]].round(4).sort_values(
    "mean", ascending=False)"""),

("code", """figures.plot_headline_curves(
    figures.validation_curves(runs, "mlp/drug", "val_iid", "loss"),
    ylabel="Validation cross-entropy", log_y=True)"""),

("md", """## 2. R3 comment 1: the pre-shuffling trade-off"""),

("code", """chunk6 = figures.figure6_chunk_sweep(head="mlp/drug")
p_chunk6 = figures.plot_figure6_chunk_sweep(chunk6)
figures.save(p_chunk6, "revision_fig6_chunk_sweep", width=9.0, height=6.5)
p_chunk6"""),

("code", """figures.figure6_chunk_summary(head="mlp/drug").round(3)"""),

("md", """* With pre-shuffling, chunk size can be raised to at least 1,024 at
  no measurable cost. Without it, degradation is already visible at 128."""),

("code", """tput = figures.throughput_chunk_table()
tput[tput.regime == "fixed_buffer"][
    ["loader", "block_size", "settings", "samples_per_sec", "sd", "n", "epoch_hours"]
].round(1)"""),

("code", """p_tput = figures.plot_throughput_chunk(tput)
figures.save(p_tput, "revision_throughput_chunk_size", width=6.8, height=5.0)
p_tput"""),

("md", """## 3. R3 comment 3: guidance for choosing parameters

Parameter 1, `chunk_size` at load time: Minibatch label entropy measured over all 94,129,984 rows from the index"""),

("code", """div = figures.diversity_table(
    {"unshuffled": "10_batch_diversity_unshuffled_buf16384.json",
     "preshuffled": "10_batch_diversity_preshuffled_buf16384.json"}, label="drug")
p_div = figures.plot_diversity(div, "drug")
figures.save(p_div, "revision_diversity_drug", width=7.4, height=4.8)
p_div"""),

("code", """figures.diversity_agreement("drug")"""),

("code", """struct = figures.ondisk_structure_table()
runs_drug = (struct[(struct.label == "drug") & (struct.window == 16)]
             [["plate", "n_categories_in_plate", "n_runs",
               "run_length_median", "run_length_max"]]
             .drop_duplicates("plate").sort_values("run_length_median"))
sorted_by_compound = runs_drug[runs_drug.run_length_median > 10_000]
print(f"{len(sorted_by_compound)} of {len(runs_drug)} plates sorted by compound: "
      f"median drug run {sorted_by_compound.run_length_median.min():,.0f} to "
      f"{sorted_by_compound.run_length_median.max():,.0f} cells")
print("the other two: " + ", ".join(
    f"{r.plate} {r.run_length_median:,.0f}"
    for r in runs_drug[runs_drug.run_length_median <= 10_000].itertuples()))
runs_drug.round(0)"""),

("code", """mech = figures.diversity_table(
    {"unshuffled": "10_batch_diversity_unshuffled_buf16384.json"}, label="drug")
mech = mech[mech.family == "annbatch (not pre-shuffled)"].copy()
mech["chunks_in_buffer"] = mech.buffer_rows / mech.block_size
mech[["block_size", "buffer_rows", "chunks_in_buffer", "n_distinct"]].sort_values(
    "block_size").round(1)"""),

("md", """* Degradation when `chunk_size` approaches the
  on-disk run length of the label, and the recoverable
  diversity is bounded by `buffer_rows / chunk_size`.
* Pre-shuffling sets the run length to 1 and removes the constraint."""),

("md", """Parameter 2: `shuffle_chunk_size` (pre-shuffling)

* Conclusion: `dataset_size / shuffle_chunk_size` above about 2,000 and minibatch entropy stays within 1% of a perfect shuffle (annbatch's default: 2^21 / 1,000 = 2,097)"""),

("code", """pq = figures.preshuffle_quality_table()
print(f"ideal entropy: {pq.attrs['ideal']:.3f} bits; "
      f"shipped shuffle_chunk_size: {pq.attrs['shipped']:,}")
# ratio as an integer: pandas renders it in scientific notation otherwise, and
# the "above about 2,000" rule then has no row a reader can point at.
pq = pq.assign(ratio=pq.ratio.round(0).astype("int64"))
pq[["shuffle_chunk_size", "ratio", "loader_chunk_size",
    "entropy", "fraction_of_ideal"]].round(4)"""),

("md", """* scDataset: high throughput only at large blocks, but large blocks -> low diversity"""),

("code", """tr = figures.speed_diversity_tradeoff("drug")
print(f"diversity ceiling: {tr.attrs['ceiling_bits']:.2f} bits")
tr.round(2)"""),

("md", """### Few-epoch training (R2 major 5)

* Only one epoch (scvi-tools default for a dataset the size of TAHOE-100M with a warning): only on-disk order, no later passes over which to average out a biased slice"""),

("code", """# scvi-tools' own default: min(round((20000 / n_obs) * 400), 400), floored at 1.
_n_obs = 94_129_984
_budget = 20_000 * 400
_epochs_asked = _budget / _n_obs
print(f"constant sample budget      {_budget:,} observations")
print(f"training cells              {_n_obs:,}")
print(f"epochs that budget implies  {_epochs_asked:.3f}")
print(f"max_epochs after floor      {max(1, min(round(_epochs_asked), 400))}")"""),

("code", """pe = figures.partial_epoch_table()
print(pe.attrs["what"])
pe[[c for c in pe.columns if "/ random" not in c]].round(4)"""),

("code", """# The same, divided by the random-sampling column of the table above.
pe[["fraction_of_epoch", "optimiser_step"]
   + [c for c in pe.columns if "/ random" in c]]"""),

("md", """## 4. R2 minor 3: the memmap baseline

* One `uint8` array, 64 GB memory limit. 
* Four variants:
  1. `random_row_manuscript` is the published baseline: one row per sample
     request with `DataLoader(shuffle=True)`; the `float32` cast happens in the
     worker, so 614 MB crosses the IPC boundary per batch of 128.
  2. `random_row` is the same access pattern in-process, with no IPC.
  3. `block_shuffled` mirrors `annbatch.samplers.RandomSampler`; the fair
     like-for-like comparison.
  4. `sequential` is streaming in on-disk order; the hardware upper bound."""),

("code", """# The array, its cache oversubscription, and what the float32 cast costs per batch.
import json
_mm = json.loads((RESULTS / "40_wgs_memmap_baselines_bs128.json").read_text())
_n_obs = _mm["results"][0]["params"]["n_obs"]
_n_var = json.loads((RESULTS / "41_wgs_preshuffle_common.json").read_text())["n_var"]
_batch = _mm["batch_size"]
_bytes = _n_obs * _n_var
_limit = 64 * 1000 ** 3
print(f"array            {_n_obs:,} x {_n_var:,} uint8 = {_bytes / 1024 ** 3:,.0f} GiB")
print(f"memory limit     {_limit / 1000 ** 3:.0f} GB -> {_bytes / _limit:.1f}x oversubscribed, "
      f"at most {100 * _limit / _bytes:.0f}% cacheable")
print(f"float32 per batch of {_batch}: {_batch * _n_var * 4 / 1e6:,.0f} MB across the IPC boundary")"""),

("code", """mm = figures.memmap_table()
p_mm = figures.plot_memmap(mm)
figures.save(p_mm, "revision_wgs_memmap_baselines", width=7.0, height=4.2)
mm.round(1)"""),

("md", """## 5. R2 major 6: pre-shuffling wall-clock for all benchmarks"""),

("code", """figures.wgs_preshuffle_table().round(3)"""),

("code", """import pandas as pd
pd.DataFrame([figures.microscopy_preshuffle()]).round(2)"""),

]


def write(name: str, cell_defs: list[tuple[str, str]]) -> None:
    """Serialise a cell list to a notebook next to this file."""
    for i, (kind, src) in enumerate(cell_defs):
        if kind == "code":
            try:
                compile(src, f"<{name} cell {i}>", "exec")
            except SyntaxError as exc:
                raise SystemExit(f"{name} cell {i} does not compile: {exc}") from exc
    cells = []
    for i, (kind, src) in enumerate(cell_defs):
        lines = src.splitlines(keepends=True)
        cid = f"cell-{i:03d}"
        cells.append({"cell_type": "markdown", "id": cid, "metadata": {},
                      "source": lines} if kind == "md" else
                     {"cell_type": "code", "id": cid, "execution_count": None,
                      "metadata": {}, "outputs": [], "source": lines})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "annbatch_rev",
                                      "language": "python", "name": "annbatch_rev"},
                       "language_info": {"name": "python", "version": "3.12.13"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).with_name(name)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(cells)} cells)")


def main() -> None:
    # `write` compiles every code cell before serialising it. The cell bodies are
    # ordinary (non-raw) triple-quoted strings, so a "\\n" written inside an
    # f-string becomes a real newline here and breaks the literal, which is
    # exactly how a syntax error once reached the executed notebook.
    write("revision_figures.ipynb", CELLS)
    write("revision_reported.ipynb", REPORTED_CELLS)


if __name__ == "__main__":
    main()
