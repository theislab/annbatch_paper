from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from _common import load_config, open_plate_datasets, open_preshuffled_datasets, plate_paths

# Hold the 16,384-row buffer fixed while sweeping chunk_size, so memory is not a confound.
DEFAULT_BUFFER_ROWS = 16_384
STREAM_CHUNK = 4_096
STREAM_PRELOAD = 4                 # 4,096 x 4 = 16,384-row buffer, as in scDataset arm (2)


@dataclass
class StrategyRun:
    iterator: Any
    n_obs: int
    store: str
    params: dict[str, Any] = field(default_factory=dict)
    global_row: np.ndarray | None = None   # preshuffled-store row -> unshuffled row


# helpers
class _SequentialShuffleBufferSampler:
    """Sequential chunk order, shuffled within the in-memory buffer."""

    def __new__(cls, *, chunk_size: int, preload_nchunks: int, batch_size: int, rng):
        from annbatch.samplers._chunk_sampler import _ChunkSampler

        class _Impl(_ChunkSampler):
            def _compute_slices(self, n_obs, rng):  # noqa: ARG002 - sequential by construction
                start, stop = self._resolve_start_stop(n_obs)
                edges = list(range(start, stop, self._chunk_size)) + [stop]
                return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]

        return _Impl(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            shuffle=True,       # shuffles *inside* the buffer only, given the override above
            drop_last=True,
            rng=rng,
        )


def _annbatch_iter(datasets, sampler):
    from annbatch import Loader

    loader = Loader(batch_sampler=sampler, return_index=True, preload_to_gpu=False, to="torch")
    loader.add_datasets(datasets)

    def gen():
        for b in loader:
            yield b["X"], np.asarray(b["index"])

    return gen(), loader.n_obs


class _ConcatCSR:
    """Row-indexable view over several on-disk CSR datasets, in a single index space."""

    def __init__(self, datasets, n_obs):
        self.datasets = datasets
        self.offsets = np.concatenate(([0], np.cumsum(n_obs)))
        self.n_obs = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.n_obs

    def __getitem__(self, indices):
        import scipy.sparse as sp

        indices = np.asarray(indices)
        order = np.argsort(indices, kind="stable")
        srt = indices[order]
        blocks, taken = [], 0
        for d, lo, hi in zip(self.datasets, self.offsets[:-1], self.offsets[1:], strict=True):
            n = int(np.searchsorted(srt, hi) - np.searchsorted(srt, lo))
            if n:
                sel = srt[taken : taken + n] - lo
                blocks.append(d[sel])
                taken += n
        X = sp.vstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return sp.csr_matrix(X)[inverse]


def _scdataset_iter(datasets, n_obs, *, block_size, fetch_factor, batch_size, num_workers, seed):
    """Drive the real `scdataset` package over our Zarr stores."""
    import torch
    from scdataset import BlockShuffling, scDataset
    from torch.utils.data import DataLoader

    view = _ConcatCSR(datasets, n_obs)

    def fetch(collection, indices):
        idx = np.asarray(indices)
        return collection[idx], idx

    def batch_cb(data, positions):
        X, idx = data
        Xb = X[positions].tocsr()
        return (
            torch.sparse_csr_tensor(
                torch.from_numpy(Xb.indptr.astype(np.int64)),
                torch.from_numpy(Xb.indices.astype(np.int64)),
                torch.from_numpy(Xb.data),
                size=Xb.shape,
            ),
            torch.from_numpy(np.asarray(idx)[positions].astype(np.int64)),
        )

    ds = scDataset(
        view,
        BlockShuffling(block_size=block_size),
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_callback=fetch,
        batch_callback=batch_cb,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers else None,
        generator=torch.Generator().manual_seed(seed),
    )

    def gen():
        for X, idx in loader:
            yield X, idx.numpy()

    return gen(), view.n_obs


# registry
_ANNBATCH_RE = re.compile(r"^annbatch_(raw|pre)_c(\d+)$")
_SCDATASET_RE = re.compile(r"^scdataset_b(\d+)_f(\d+)$")


def available_strategies() -> list[str]:
    ks = [16, 64, 128, 256, 512, 1024]
    return (
        ["random", "streaming", "streaming_buffer"]
        + [f"annbatch_pre_c{k}" for k in ks]
        + [f"annbatch_raw_c{k}" for k in ks]
        + [f"scdataset_b{b}_f4" for b in (16, 128, 512, 1024)]
        + ["scdataset_b16_f256"]
    )


def build(name: str, *, cfg=None, seed: int = 0, batch_size: int = 4096,
          num_workers: int = 4, buffer_rows: int = DEFAULT_BUFFER_ROWS) -> StrategyRun:
    cfg = cfg or load_config()
    from annbatch.samplers import RandomSampler, SequentialSampler

    rng = np.random.default_rng(seed)

    def nchunks_for(chunk_size: int) -> int:
        """preload_nchunks holding the buffer at `buffer_rows`, but never below one batch."""
        return max(max(1, buffer_rows // chunk_size), -(-batch_size // chunk_size))

    def raw():
        return open_plate_datasets(plate_paths(cfg, "train"))

    if name == "random":
        ds, n = raw()
        nch = nchunks_for(1)
        s = RandomSampler(chunk_size=1, preload_nchunks=nch,
                          batch_size=batch_size, drop_last=True, rng=rng)
        it, n_obs = _annbatch_iter(ds, s)
        return StrategyRun(it, n_obs, "unshuffled",
                           {"chunk_size": 1, "preload_nchunks": nch, "buffer_rows": nch,
                            "description": "true uniform random sampling"})

    if name == "streaming":
        ds, n = raw()
        s = SequentialSampler(chunk_size=STREAM_CHUNK, preload_nchunks=STREAM_PRELOAD,
                              batch_size=batch_size, drop_last=True)
        it, n_obs = _annbatch_iter(ds, s)
        return StrategyRun(it, n_obs, "unshuffled",
                           {"chunk_size": STREAM_CHUNK, "preload_nchunks": STREAM_PRELOAD,
                            "description": "sequential read, no randomisation"})

    if name == "streaming_buffer":
        ds, n = raw()
        s = _SequentialShuffleBufferSampler(chunk_size=STREAM_CHUNK, preload_nchunks=STREAM_PRELOAD,
                                            batch_size=batch_size, rng=rng)
        it, n_obs = _annbatch_iter(ds, s)
        return StrategyRun(it, n_obs, "unshuffled",
                           {"buffer_rows": STREAM_CHUNK * STREAM_PRELOAD,
                            "description": "sequential read into a 16,384-row shuffle buffer"})

    m = _ANNBATCH_RE.match(name)
    if m:
        where, k = m.group(1), int(m.group(2))
        if where == "raw":
            ds, n = raw()
            store, global_row = "unshuffled", None
        else:
            ds, n = open_preshuffled_datasets(cfg)
            store = "preshuffled"
            global_row = _load_global_row(cfg)
        nch = nchunks_for(k)
        s = RandomSampler(chunk_size=k, preload_nchunks=nch,
                          batch_size=batch_size, drop_last=True, rng=rng)
        it, n_obs = _annbatch_iter(ds, s)
        return StrategyRun(it, n_obs, store,
                           {"chunk_size": k, "preload_nchunks": nch,
                            "buffer_rows": k * nch},
                           global_row=global_row)

    m = _SCDATASET_RE.match(name)
    if m:
        b, f = int(m.group(1)), int(m.group(2))
        ds, n = raw()
        it, n_obs = _scdataset_iter(ds, n, block_size=b, fetch_factor=f, batch_size=batch_size,
                                    num_workers=num_workers, seed=seed)
        return StrategyRun(it, n_obs, "unshuffled",
                           {"block_size": b, "fetch_factor": f, "num_workers": num_workers,
                            "buffer_rows": b * f,
                            "description": "scdataset.BlockShuffling over the same Zarr stores"})

    raise ValueError(f"unknown strategy {name!r}; available: {available_strategies()}")


def _load_global_row(cfg) -> np.ndarray:
    """Row -> unshuffled-concatenation row for the pre-shuffled store."""
    import zarr
    from annbatch import DatasetCollection

    collection = DatasetCollection(zarr.open(cfg["tahoe"]["preshuffled"], mode="r"))
    return np.concatenate([np.asarray(g["obs/global_row"][:]) for g in collection])
