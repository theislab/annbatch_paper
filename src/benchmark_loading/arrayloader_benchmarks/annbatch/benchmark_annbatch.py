from __future__ import annotations

import json
import warnings

import anndata as ad
import click
import zarr
import zarrs  # noqa
from annbatch import DatasetCollection, Loader

from arrayloader_benchmarks.utils import benchmark_loader

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

# Suppress zarr vlen-utf8 codec warnings
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
    module="zarr.codecs.vlen_utf8",
)


@click.command()
@click.option("--store_path", type=str)
@click.option("--chunk_size", type=int, default=256)
@click.option("--preload_nchunks", type=int, default=8)
@click.option("--batch_size", type=int, default=4096)
@click.option("--preload_to_gpu", type=bool, default=False)
@click.option("--n_samples", type=int, default=2_000_000)
def benchmark(  # noqa: PLR0917, PLR0913
    store_path: str,
    chunk_size: int = 256,
    preload_nchunks: int = 32,
    batch_size: int = 4096,
    preload_to_gpu: bool = False,  # noqa: FBT001, FBT002
    n_samples: int = 2_000_000,
):
    def load_func(g: zarr.Group) -> ad.AnnData:
        X_store = g["X"]

        if isinstance(X_store, zarr.Group) and "encoding-type" in X_store.attrs:
            if X_store.attrs["encoding-type"] in {"csr_matrix", "csc_matrix"}:
                X = ad.io.sparse_dataset(X_store)
            else:
                X = X_store
        else:
            X = X_store

        obs = ad.io.read_elem(g["obs"])

        if "cell_name" in obs.columns:
            obs = obs[["cell_name"]]
        else:
            obs = obs.copy()
            obs["cell_name"] = obs.index.astype(str)
            obs = obs[["cell_name"]]

        return ad.AnnData(X=X, obs=obs)

    collection = DatasetCollection(zarr.open(store_path))
    ds = Loader(
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        shuffle=True,
        preload_to_gpu=preload_to_gpu,
        to_torch=True,
    )
    ds.use_collection(collection, load_adata=load_func)

    n_samples = n_samples if n_samples != -1 else len(ds)
    samples_per_sec, _, _, total_time = benchmark_loader(ds, n_samples, batch_size)
    click.echo(
        json.dumps(
            {
                "loader": "AnnBatch",
                "samples/sec": samples_per_sec,
                "total_time": total_time,
            }
        )
    )


if __name__ == "__main__":
    benchmark()
