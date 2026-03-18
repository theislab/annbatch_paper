from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from arrayloader_benchmarks.utils import benchmark_loader


class MemmapDataset(Dataset):
    """Dataset backed by a numpy memmap file.

    Expects a companion ``<stem>.meta.json`` with keys ``shape``, ``dtype``,
    and ``format`` (as written by the data preparation script).
    """

    def __init__(self, memmap_path: str):
        meta_path = Path(memmap_path).with_suffix(".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        shape = tuple(meta["shape"])
        dtype = meta["dtype"]

        self.data = np.memmap(memmap_path, dtype=dtype, mode="r", shape=shape)
        self.n_cells, self.n_genes = shape

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx):
        # Copy slice to avoid holding open mmap pages; cast to float32 for torch.
        return torch.from_numpy(self.data[idx].astype(np.float32).copy())


@click.command()
@click.option("--memmap_path", type=str, required=True)
@click.option("--batch_size", type=int, default=4096)
@click.option("--num_workers", type=int, default=4)
@click.option("--pin_memory", type=bool, default=True)
@click.option("--n_samples", type=int, default=2_000_000)
def benchmark(
    memmap_path: str,
    batch_size: int = 4096,
    num_workers: int = 4,
    pin_memory: bool = True,
    n_samples: int = 2_000_000,
):
    dataset = MemmapDataset(memmap_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    n_samples = n_samples if n_samples != -1 else len(dataset)
    samples_per_sec, _, _, total_time = benchmark_loader(loader, n_samples, batch_size)

    click.echo(
        json.dumps(
            {
                "loader": "NumpyMemmap",
                "samples/sec": samples_per_sec,
                "total_time": total_time,
            }
        )
    )


if __name__ == "__main__":
    benchmark()