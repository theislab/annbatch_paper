# annbatch: Benchmarks

Code, data and figure notebooks behind the `annbatch` paper. The benchmarks in this repo were run with
`annbatch=0.1.0`.

All benchmarks measure the same quantity — **mini-batch loading throughput in samples/sec** — for `annbatch`
against other mini-batch loaders (`scDataset`, lamindb's `MappedCollection`, a dense NumPy `memmap` baseline,
`scPortrait`), plus a set of "theoretical limit" benchmarks that establish how fast a loader actually has to be
before model compute, rather than I/O, becomes the bottleneck.

## Repo structure

```
├── Dockerfile                     # CUDA/PyTorch image with all benchmark dependencies
├── pyproject.toml                 # installable `arrayloader_benchmarks` package
├── src/                           # everything that produces numbers
│   ├── benchmark_loading/         # loader throughput benchmarks (the main results)
│   ├── benchmark_store_creation/  # store creation timing (legacy notebook)
│   ├── theoretical_limits/        # how fast does loading need to be? (model fit-time benchmarks)
│   └── scportrait_benchmark.ipynb # single-cell imaging example
├── revision/                      # additional experiments for the reviewer response (see revision/README.md)
└── figure_plots/                  # everything that produces figures
    ├── source_data/               # CSVs collected from the benchmark runs
    ├── paper_figures.ipynb        # builds all paper figures from source_data/
    └── figures/                   # rendered SVGs
```

## `src/benchmark_loading` — loader throughput

The installable package `arrayloader_benchmarks` holds one self-contained, `click`-based CLI per loader. Each
one builds its loader, iterates `--n_samples` samples via the shared timing helper in
[`utils.py`](src/benchmark_loading/arrayloader_benchmarks/utils.py), and prints a JSON line with
`samples/sec` and `total_time`, which the runner scripts/notebooks parse.

| Script | What it benchmarks |
| --- | --- |
| [`annbatch/benchmark_annbatch.py`](src/benchmark_loading/arrayloader_benchmarks/annbatch/benchmark_annbatch.py) | `annbatch` `DatasetCollection` + `Loader` over a sharded zarr store; sweeps `chunk_size`/`preload_nchunks` and optionally preloads batches to GPU (`--preload_to_gpu`) |
| [`scDataset/benchmark_scDataset.py`](src/benchmark_loading/arrayloader_benchmarks/scDataset/benchmark_scDataset.py) | `scDataset` with `BlockShuffling` over an `AnnCollection` of backed h5ad (or zarr) shards, driven by a torch `DataLoader` with `num_workers` |
| [`mapped_collection/benchmark_mapped_collection.py`](src/benchmark_loading/arrayloader_benchmarks/mapped_collection/benchmark_mapped_collection.py) | lamindb `MappedCollection` over the same h5ad shards with a shuffling torch `DataLoader` |
| [`memmap/benchmark_memmap.py`](src/benchmark_loading/arrayloader_benchmarks/memmap/benchmark_memmap.py) | upper-bound baseline: a dense `np.memmap` of the same data behind a plain torch `DataLoader` |

Data preparation:

| Script | Purpose |
| --- | --- |
| [`benchmark_setup/download_tahoe100M_h5ads.py`](src/benchmark_loading/arrayloader_benchmarks/benchmark_setup/download_tahoe100M_h5ads.py) | pulls the raw Tahoe-100M h5ad artifacts from the `laminlabs/arrayloader-benchmarks` instance and optionally converts them to zarr |
| [`benchmark_setup/shuffle_and_shard_tahoe100M_h5ads.py`](src/benchmark_loading/arrayloader_benchmarks/benchmark_setup/shuffle_and_shard_tahoe100M_h5ads.py) | writes globally shuffled, equally sized h5ad shards (full and protein-coding gene space) used by the h5ad-based loaders |
| [`annbatch/prepare_store.py`](src/benchmark_loading/arrayloader_benchmarks/annbatch/prepare_store.py) | writes the sharded, chunked, Blosc-compressed zarr store `annbatch` reads from |
| [`genetic_data_prep.py`](src/benchmark_loading/genetic_data_prep.py) + [`1000genomes_grch38.yaml`](src/benchmark_loading/1000genomes_grch38.yaml) | 1000 Genomes pipeline (needs `plink2` and `cellink`): VCF → PGEN → merge → common (MAF ≥ 0.01, dense) and rare (MAF < 0.01, < 0.001, sparse) SNP subsets, replicated 157× to ~500k synthetic patients, then shuffled into zarr stores (plus a dense memmap for the common set) |

Runners (these are what was actually executed, and what the source data was copied from):

| Runner | Scope |
| --- | --- |
| [`run_benchmarks.ipynb`](src/benchmark_loading/run_benchmarks.ipynb) | main comparison on 2 Mio Tahoe-100M cells (on-prem/deNBI), sweeping `annbatch` chunk sizes and each competing loader's recommended settings |
| [`run_benchmarks_AWS.ipynb`](src/benchmark_loading/run_benchmarks_AWS.ipynb) | the same sweep on AWS SageMaker, to show the results hold across infrastructure |
| [`run_benchmarks_full_dataset.py`](src/benchmark_loading/run_benchmarks_full_dataset.py) (+ `.sbatch`) | full-epoch run over all 100 Mio cells, reporting wall-clock time per epoch per loader |
| [`run_parallel_benchmarks.py`](src/benchmark_loading/run_parallel_benchmarks.py) (+ `.sbatch`) | multi-process scaling: N independent loader processes share one filesystem, measuring cumulative throughput (how well each loader scales to multi-GPU training) |
| [`run_benchmarks_1000genomes_dense.ipynb`](src/benchmark_loading/run_benchmarks_1000genomes_dense.ipynb), [`..._rare_maf_0.01`](src/benchmark_loading/run_benchmarks_1000genomes_rare_maf_0.01.ipynb), [`..._rare_maf_0.001`](src/benchmark_loading/run_benchmarks_1000genomes_rare_maf_0.001.ipynb) | the same loader comparison on genomics data — common SNPs (dense) and two rare-variant (sparse) MAF thresholds — showing behaviour at different sparsity levels |
| [`store_creation_time_benchmark.py`](src/benchmark_loading/store_creation_time_benchmark.py) / [`.ipynb`](src/benchmark_loading/store_creation_time_benchmark.ipynb) | one-off cost of building the shuffled store: `h5ad → h5ad` vs `h5ad → zarr` vs `zarr → zarr` |

## `src/theoretical_limits` — how fast does loading need to be?

These benchmarks never touch disk. They feed models a mock dataset that yields a pre-allocated batch (with an
optional `sleep` that simulates a given loading throughput), so the measured fit time isolates pure compute.

| Item | What it shows |
| --- | --- |
| [`loading_speed_vs_fit_time.ipynb`](src/theoretical_limits/loading_speed_vs_fit_time.ipynb) | epoch fit time as a function of simulated loading throughput for a linear model and two scVI sizes — where the loading-limited regime ends and the compute-limited regime begins |
| [`batch_size_vs_fit_time.ipynb`](src/theoretical_limits/batch_size_vs_fit_time.ipynb) | fit time vs. batch size (256 → 65536) for scVI, motivating the large batch sizes used in the loading benchmarks |
| [`block_shuffle_simulation.ipynb`](src/theoretical_limits/block_shuffle_simulation.ipynb) | simulates block shuffling (contiguous reads + shuffle buffer) against a full random shuffle on a maximally autocorrelated 100 Mio element array, and compares shuffling quality via local entropy |
| [`mock_dataset.py`](src/theoretical_limits/mock_dataset.py), [`models.py`](src/theoretical_limits/models.py) | the mock `IterableDataset` and the Lightning models (`SimpleLinearModel`, `SCVI`) used above |
| [`CNN/`](src/theoretical_limits/CNN), [`ViT/`](src/theoretical_limits/ViT) | pure forward+backward throughput of torchvision ResNet-50 and ViT variants on 224×224 mock images — the loading speed image models require |
| [`FNN_Sparse/`](src/theoretical_limits/FNN_Sparse) | throughput of a dense FNN vs. a GenNet-style sparse architecture on mock SNP data |
| [`DeepSet/`](src/theoretical_limits/DeepSet) | throughput of a DeepRVAT-style Deep Set on mock per-gene rare-variant data (UK Biobank-like structure) |
| [`aido/`](src/theoretical_limits/aido) | throughput of the AIDO.Cell single-cell foundation models (3M/10M, BERT-style); vendored `aido.cell` code plus a Docker image and run script |

Each of these subdirectories ships its own `requirements.txt` (and `Dockerfile` where the environment differs),
since the model environments conflict with each other.

## Other benchmarks

- [`src/scportrait_benchmark.ipynb`](src/scportrait_benchmark.ipynb) — imaging case study: builds an `annbatch`
  collection from scPortrait `.h5sc` single-cell image files (dense `obsm` arrays) and compares loading
  throughput against scPortrait's own loader.
- [`src/benchmark_store_creation/create_store.ipynb`](src/benchmark_store_creation/create_store.ipynb) — earlier
  zarr-vs-h5ad store creation timing against the old `create_anndata_collection` API; superseded by
  `src/benchmark_loading/store_creation_time_benchmark.py`.

## `figure_plots` — paper figures

[`paper_figures.ipynb`](figure_plots/paper_figures.ipynb) reads only from `source_data/` (plotnine/matplotlib)
and writes the SVGs in `figures/`:

- **Figure 1** — fit time vs. loading throughput per model, and the same curve extrapolated to large-scale
  multi-GPU training with the current SoTA and `annbatch` loading limits marked.
- **Figure 2** — the loader comparison panel: AWS vs. on-prem throughput, full-epoch wall-clock time, the
  imaging benchmark, store creation time / throughput / disk usage for h5ad vs. zarr, the 1000 Genomes SNP
  benchmarks, multi-process scaling, and a table of required loading speeds per model architecture.
- **Supp. Fig. 1** — fit time vs. batch size.

Source data files map one-to-one onto the runners above:

| File | Produced by |
| --- | --- |
| `fit_time_vs_loading_speed.csv`, `fit_time_vs_batch_size.csv` | `theoretical_limits/*.ipynb` |
| `loading_times_full_epoch_2mio_samples_denbi.csv`, `..._AWS.csv` | `run_benchmarks.ipynb`, `run_benchmarks_AWS.ipynb` |
| `loading_times_full_epoch_fixed.csv` | `run_benchmarks_full_dataset.py` |
| `parallel_loading_benchmark_results.csv` | `run_parallel_benchmarks.py` |
| `snp_benchmark.csv` | the three `run_benchmarks_1000genomes_*.ipynb` notebooks |
| `store_creation_time.csv` | `store_creation_time_benchmark.py` |
| `image_benchmark.csv` | `scportrait_benchmark.ipynb` |

(`speed_comp_aws.parquet` is an earlier version of the AWS results and is no longer read by the notebook.)

## Running the benchmarks

```bash
pip install -e .                      # installs the `arrayloader_benchmarks` package
pip install -e ".[viz]"               # + plotting/jupyter dependencies for figure_plots
```

Or build the [`Dockerfile`](Dockerfile) (based on the NVIDIA PyTorch image), which additionally installs
`lamindb`, `scDataset`, `annbatch[zarrs,torch]` and JupyterLab — this is the image used by the SLURM
`.sbatch` scripts.

All store/output paths are hard-coded as module-level constants at the top of the preparation and runner
scripts; adjust them for your filesystem before running.
