from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "paths.yaml"

# The four prediction tasks of scDataset (arXiv:2506.01883) section 4.4.
TASKS: tuple[str, ...] = ("cell_line", "drug", "moa_broad", "moa_fine")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or CONFIG_PATH).open() as f:
        return yaml.safe_load(f)


# Provenance
def _git_rev(repo: str | Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _mem_total_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def _git_dirty(repo: str | Path) -> int | None:
    """Number of modified/untracked paths, so `git_rev` is not read as the whole story."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        )
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:  # noqa: BLE001
        return None


def provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to attribute a result to a code state and a machine."""
    pkgs = {}
    for p in ("annbatch", "anndata", "zarr", "zarrs", "torch", "scdataset", "numpy", "scipy"):
        try:
            pkgs[p] = version(p)
        except Exception:  # noqa: BLE001
            pkgs[p] = None
    return {
        "git_rev": _git_rev(cfg["repo"]),
        "git_uncommitted_paths": _git_dirty(cfg["repo"]),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "node_mem_total_gb": _mem_total_gb(),
        "python": platform.python_version(),
        "packages": pkgs,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_mem_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_result(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[_common] wrote {path}", flush=True)


# Zarr configuration
def configure_zarr() -> None:
    """Install the Rust `zarrs` codec pipeline and silence known-noisy warnings."""
    import warnings

    import zarr
    import zarrs  # noqa: F401  (import registers the pipeline)

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    warnings.filterwarnings(
        "ignore",
        message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
        category=UserWarning,
    )
    # The Tahoe plate stores carry obsm/layers that we deliberately do not load.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"annbatch.*")


# Tahoe-100M stores
def plate_paths(cfg: dict[str, Any], which: Literal["train", "test", "all"] = "train") -> list[Path]:
    t = cfg["tahoe"]
    plates = {
        "train": t["train_plates"],
        "test": t["test_plates"],
        "all": t["train_plates"] + t["test_plates"],
    }[which]
    root = Path(t["plates_dir"])
    paths = [root / t["plate_glob"].format(i=i) for i in plates]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing plate stores: {missing}")
    return paths


def open_plate_datasets(paths: list[Path]):
    """Open the `X` CSR datasets of a list of AnnData Zarr plate stores."""
    import anndata as ad
    import zarr

    datasets, n_obs = [], []
    for p in paths:
        g = zarr.open(str(p), mode="r")
        ds = ad.io.sparse_dataset(g["X"])
        datasets.append(ds)
        n_obs.append(ds.shape[0])
    return datasets, n_obs


def open_preshuffled_datasets(cfg: dict[str, Any], path: str | Path | None = None):
    """Open the `X` CSR datasets of a pre-shuffled annbatch DatasetCollection."""
    import anndata as ad
    import zarr
    from annbatch import DatasetCollection

    collection = DatasetCollection(zarr.open(str(path or cfg["tahoe"]["preshuffled"]), mode="r"))
    datasets, n_obs = [], []
    for g in collection:
        ds = ad.io.sparse_dataset(g["X"])
        datasets.append(ds)
        n_obs.append(ds.shape[0])
    return datasets, n_obs


# Label tables
@dataclass
class LabelTable:
    """Global integer labels in the row order of one store."""

    codes: dict[str, np.ndarray]
    classes: dict[str, list[str]]
    n_obs: int

    @property
    def n_classes(self) -> dict[str, int]:
        return {t: len(c) for t, c in self.classes.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f"codes__{k}": v for k, v in self.codes.items()})
        with path.with_suffix(".classes.json").open("w") as f:
            json.dump({"classes": self.classes, "n_obs": self.n_obs}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LabelTable":
        path = Path(path)
        if not path.name.endswith(".npz"):
            path = path.with_suffix(".npz")
        z = np.load(path)
        codes = {k.removeprefix("codes__"): z[k] for k in z.files}
        with path.with_suffix("").with_suffix(".classes.json").open() as f:
            meta = json.load(f)
        return cls(codes=codes, classes=meta["classes"], n_obs=meta["n_obs"])


# Throughput measurement
@dataclass
class ThroughputResult:
    loader: str
    samples_per_sec: float
    n_samples: int
    elapsed_s: float
    batch_size: int
    params: dict[str, Any] = field(default_factory=dict)
    warmup_batches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def shutdown_iterator(iterator) -> None:
    """Tear down a loader's worker processes before the next measurement."""
    import gc

    for attr in ("_shutdown_workers", "_shutdown"):
        fn = getattr(iterator, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001 - teardown must not fail a measurement
                pass
            break
    gc.collect()


class MeasurementTimeout(RuntimeError):
    """A single throughput measurement exceeded its wall-clock budget."""


@contextlib.contextmanager
def time_budget(seconds: int, what: str):
    """Abort the enclosed block if `seconds` pass with no reported progress."""
    if seconds <= 0:
        yield lambda: None
        return

    def _fire(signum, frame):  # noqa: ARG001
        raise MeasurementTimeout(f"{what} made no progress for {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield lambda: signal.alarm(seconds)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def lustre_pool(path) -> str | None:
    """Which Lustre OST pool a path is stored on, or None if undeterminable."""
    path = Path(path)
    probe = path
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.stat().st_size > 1 << 20:
                probe = f
                break
    try:
        out = subprocess.run(["lfs", "getstripe", str(probe)],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"pool:\s+(\S+)", out)
    return m.group(1) if m else None


def assert_ssd(cfg, *paths, strict: bool = True) -> dict[str, str | None]:
    """Refuse to benchmark data that is not on the expected storage pool."""
    want = cfg.get("storage_pool")
    found: dict[str, str | None] = {}
    for p in paths:
        pool = lustre_pool(p)
        found[str(p)] = pool
        if want and pool and pool != want:
            msg = f"{p} is on Lustre pool {pool!r}, expected {want!r}"
            if strict:
                raise RuntimeError(msg)
            print(f"[warn] {msg}", flush=True)
    return found


def measure_throughput(
    iterator,
    *,
    n_samples: int,
    batch_size: int,
    loader: str,
    params: dict[str, Any] | None = None,
    warmup_batches: int = 5,
    extract=None,
    stall_seconds: int = 900,
) -> ThroughputResult:
    """Time a loader over ``n_samples`` observations, discarding a warm-up."""
    n_target = n_samples // batch_size
    seen = 0
    t0 = None
    try:
        with time_budget(stall_seconds, f"{loader} measurement") as ping:
            for i, batch in enumerate(iterator):
                ping()
                if extract is not None:
                    batch = extract(batch)
                if i == warmup_batches:
                    t0 = time.perf_counter()
                if t0 is not None:
                    seen += batch_size
                if i + 1 >= n_target:
                    break
        if t0 is None:
            raise RuntimeError(f"loader produced fewer than {warmup_batches + 1} batches")
        elapsed = time.perf_counter() - t0
    finally:
        shutdown_iterator(iterator)
    return ThroughputResult(
        loader=loader,
        samples_per_sec=seen / elapsed,
        n_samples=seen,
        elapsed_s=elapsed,
        batch_size=batch_size,
        params=params or {},
        warmup_batches=warmup_batches,
    )
