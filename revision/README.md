# Revision experiments

Code for the additional experiments run in response to the reviewers. Scripts write JSON to `results/`,
the notebooks read only from there and render the panels in `figures/`.

```
revision/
├── config/paths.yaml           # data locations on the cluster (entries marked TODO still missing)
├── scripts/                    # numbered pipeline; run from revision/scripts/ (they import _common)
├── notebooks/
│   ├── figures.py              # loading, tidying and plotting helpers
│   ├── _build_notebook.py      # generates both notebooks below — edit this, not the .ipynb
│   ├── revision_figures.ipynb  # every experiment (working record)
│   └── revision_reported.ipynb # only what goes into the response, one section per reviewer comment
└── figures/                    # rendered panels
```

## Pipeline

| Script | Produces |
| --- | --- |
| `00_preshuffle_tahoe.py` | pre-shuffled Tahoe Zarr collection (298 GB) |
| `01_build_label_tables.py` | global label tables for the four prediction tasks |
| `02_build_eval_sets.py` | the two fixed held-out evaluation sets |
| `03_verify_preshuffle.py` | check: the store is a permutation of the input |
| `04_verify_loaders.py` | check: each loading strategy yields rows matching its indices |
| `05_ondisk_structure.py` | on-disk label run lengths per plate |
| `06_preshuffle_quality.py` | minibatch diversity vs. `shuffle_chunk_size` |
| `08_microscopy_preshuffle_timing.py` | scPortrait pre-shuffling wall-clock |
| `10_batch_diversity.py` | minibatch diversity per loading strategy |
| `20_train_convergence.py` | training/validation histories (scDataset Fig. 6 reproduction with annbatch); submitted via `launch_convergence.sh` / `launch_convergence_seeds.sh` |
| `30_throughput_sweeps.py` | throughput vs. chunk / block size and dataset size |
| `40_wgs_memmap_baselines.py` | 1000 Genomes memmap baselines |
| `41_wgs_preshuffle_timing.py` | WGS pre-shuffling wall-clock, three regimes |
| `50`–`55_*.py` | response tables, manifest and consistency checks over `docs/` |

Loading strategies (annbatch with/without pre-shuffling, scDataset, streaming, random) are defined in
`scripts/_strategies.py`; shared config, provenance and I/O helpers in `scripts/_common.py`.

## Not included

The `slurm/` job scripts used by `launch_convergence*.sh`, the `results/` JSON, `docs/` (response drafts read by
scripts 50–55 and `_fill_*.py`) and `resources/wgs_fig2e_throughput.csv` were not part of the code drop. The
notebooks are committed with their outputs, so they can be read without these.
