from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import click
from scdataset import BlockShuffling, scDataset
from torch.utils.data import DataLoader
import zarr

from arrayloader_benchmarks.utils import benchmark_loader


@click.command()
@click.option("--store_path", type=str, default="")
@click.option("--store_type", type=str, default="h5ad")
@click.option("--block_size", type=int, default=4)
@click.option("--fetch_factor", type=int, default=16)
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
@click.option("--multiprocessing_context", type=str, default="fork")
def benchmark(  # noqa: PLR0917
    store_path: str = "",
    store_type: str = "h5ad",
    block_size: int = 4,
    fetch_factor: int = 16,
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
    multiprocessing_context: str = "fork",
):
    if store_path:
        if store_type == "h5ad":
            shards = list(Path(store_path).glob("*.h5ad"))
        elif store_type == "zarr":
            shards = [p for p in Path(store_path).iterdir() if p.is_dir()]
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    else:
        import lamindb as ln

        benchmarking_collections = ln.Collection.using(
            "laminlabs/arrayloader-benchmarks"
        )
        shards = benchmarking_collections.get("eAgoduHMxuDs5Wem0000").cache()

    if store_type == "zarr":
        adatas = [ad.AnnData(X=ad.io.sparse_dataset(zarr.open(shard, mode="r")["X"])) for shard in shards]
    else:
        adatas = [ad.read_h5ad(shard, backed="r") for shard in shards]
    adata_collection = ad.experimental.AnnCollection(adatas)

    def fetch_adata(collection, indices):
        return collection[indices].X

    strategy = BlockShuffling(block_size=block_size)
    dataset = scDataset(
        adata_collection,
        strategy,
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_callback=fetch_adata,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=fetch_factor + 1,
        multiprocessing_context=multiprocessing_context,
    )

    samples_per_sec, _, _, total_time = benchmark_loader(loader, n_samples, batch_size)

    click.echo(
        json.dumps(
            {
                "loader": "scDataset",
                "samples/sec": samples_per_sec,
                "total_time": total_time,
            }
        )
    )


if __name__ == "__main__":
    benchmark()
