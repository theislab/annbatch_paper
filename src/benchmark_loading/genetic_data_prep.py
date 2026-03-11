import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import logging
import subprocess
from pathlib import Path
from typing import Optional

import anndata as ad
import zarr
import dask

dask.config.set(num_workers=28)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _run(cmd: str, cwd: Optional[Path] = None) -> None:
    """Execute shell command."""
    log.info(cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}\n{r.stderr}")


def vcf_to_pgen_per_chrom(data_dir: Path, chromosomes: list[int]) -> None:
    """Convert per-chromosome VCFs to PGEN."""
    pgen_dir = data_dir / "pgen"
    pgen_dir.mkdir(parents=True, exist_ok=True)

    for chrom in chromosomes:
        stem = pgen_dir / f"chr{chrom}"
        if (stem.parent / (stem.name + ".pgen")).exists():
            continue

        vcf = data_dir / (
            f"20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}"
            ".recalibrated_variants.vcf.gz"
        )
        if not vcf.exists():
            raise FileNotFoundError(vcf)

        _run(
            f"plink2 --vcf {vcf} "
            f"--make-pgen "
            f"--max-alleles 2 "
            f"--snps-only just-acgt "
            f"--set-all-var-ids @:#:\\$r:\\$a "
            f"--new-id-max-allele-len 1000 missing "
            f"--vcf-half-call m "
            f"--output-chr chrM "
            f"--out {stem}"
        )


def merge_pgen(data_dir: Path, chromosomes: list[int]) -> Path:
    """Merge chromosome PGEN files."""
    pgen_dir = data_dir / "pgen"
    merged = pgen_dir / "all_chroms"

    if merged.with_suffix(".pgen").exists():
        return merged

    list_file = pgen_dir / "pgen_list.txt"
    with open(list_file, "w") as f:
        for c in chromosomes:
            f.write(str(pgen_dir / f"chr{c}") + "\n")

    _run(
        f"plink2 --pmerge-list {list_file} pfile "
        f"--make-pgen "
        f"--out {merged}"
    )
    return merged


def extract_rare(merged: Path, data_dir: Path, maf: float) -> Path:
    """Extract variants with MAF < threshold into per-MAF directory."""
    maf_str = f"{maf:g}"
    out_dir = data_dir / f"rare_maf_{maf_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_dir / f"rare_maf_{maf_str}"

    if (stem.parent / (stem.name + ".pgen")).exists():
        return stem

    _run(
        f"plink2 --pfile {merged} "
        f"--max-maf {maf - 1e-9:.8f} "
        f"--maf 1e-9 "
        f"--snps-only just-acgt "
        f"--make-pgen "
        f"--out {stem}"
    )
    return stem


def extract_common(merged: Path, data_dir: Path, maf_min: float = 0.01) -> Path:
    """Extract common variants (MAF >= maf_min) into a dedicated directory."""
    out_dir = data_dir / "common"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_dir / "common_variants"

    if (stem.parent / (stem.name + ".pgen")).exists():
        return stem

    _run(
        f"plink2 --pfile {merged} "
        f"--maf {maf_min:.8f} "
        f"--snps-only just-acgt "
        f"--make-pgen "
        f"--out {stem}"
    )
    return stem


def pgen_to_zarr(
    pgen_stem: Path,
    zarr_path: Path,
    sparse: bool = True,
    max_variants: Optional[int] = None,
) -> None:
    """Stream PGEN to Zarr (sparse or dense)."""
    if zarr_path.exists():
        return

    from cellink.io import stream_pgen_to_zarr

    stream_pgen_to_zarr(
        pgen_path=str(pgen_stem),
        output_path=str(zarr_path),
        chunk_samples=5000,
        chunk_variants=32000,
        memory_limit_gb=450,
        sparse=sparse,
        sparse_format="csr",
        max_variants=max_variants,
    )


def write_500k_patients(zarr_path: Path, ssd_root: Path) -> Path:
    """Replicate 157x and write _500k.h5ad into per-MAF/type SSD directory."""
    from cellink.io import read_pgen_zarr

    subdir_name = zarr_path.parent.name   
    ssd_subdir = ssd_root / subdir_name
    ssd_subdir.mkdir(parents=True, exist_ok=True)

    out_h5ad = ssd_subdir / f"{zarr_path.stem}_500k.h5ad"
    if out_h5ad.exists():
        return out_h5ad

    adata_sub = read_pgen_zarr(zarr_path)

    adata_sub.var.index = adata_sub.var["2"]
    for col in list(adata_sub.var.columns):
        del adata_sub.var[col]

    adata_sub.obs.index = adata_sub.obs["iid"]
    for col in list(adata_sub.obs.columns):
        del adata_sub.obs[col]

    adata = ad.concat([adata_sub] * 157)

    adata.write_h5ad(out_h5ad, compression="gzip")
    log.info(f"Wrote {out_h5ad}")

    return out_h5ad


def shuffle_to_zarr(h5ad_path: Path, maf: Optional[float], sparse: bool) -> None:
    """Shuffle 500k h5ad into a shuffled zarr next to the h5ad."""
    from annbatch import DatasetCollection

    out_zarr = h5ad_path.with_name(h5ad_path.stem + "_shuffled.zarr")
    if out_zarr.exists():
        return

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    collection = DatasetCollection(zarr.open(str(out_zarr), mode="w"))

    if sparse:
        # Rare datasets
        if maf == 0.01:
            collection.add_adatas(
                adata_paths=[str(h5ad_path)],
                shuffle=True,
                n_obs_per_dataset=2**17,
                zarr_sparse_chunk_size=2**23,
                zarr_sparse_shard_size=2**19 * 512,
                shuffle_chunk_size=128,
            )
        elif maf == 0.001:
            collection.add_adatas(
                adata_paths=[str(h5ad_path)],
                shuffle=True,
                n_obs_per_dataset=2**17,
                zarr_sparse_chunk_size=2**21,
                zarr_sparse_shard_size=2**21 * 512,
                shuffle_chunk_size=128,
            )
        else:
            raise ValueError(f"Unsupported MAF for sparse shuffle: {maf}")
    else:
        collection.add_adatas(
            adata_paths=[str(h5ad_path)],
            shuffle=True,
            n_obs_per_dataset=16384,
            zarr_dense_chunk_size=4,
            zarr_dense_shard_size=512,
            shuffle_chunk_size=64,
        )

    log.info(f"Shuffled → {out_zarr}")


def concatenate_zarr_to_memmap(
    base_zarr: Path,
    *,
    ssd_dir: Path,
    target_n_obs: int,
    block_size: int = 2048,
) -> Path:
    """Concatenate (repeat) a base zarr-backed AnnData to reach target_n_obs and stream directly into a single memmap on SSD. Returns path to the primary memmap file."""
    import json
    import math
    import numpy as np

    from cellink.io import read_pgen_zarr

    base_zarr = Path(base_zarr)
    ssd_dir = Path(ssd_dir)
    ssd_dir.mkdir(parents=True, exist_ok=True)

    adata = read_pgen_zarr(base_zarr)
    X = adata.X

    base_n, n_vars = adata.shape
    repeats = math.ceil(target_n_obs / base_n)
    final_n = repeats * base_n

    log.info(f"Base samples: {base_n:,}")
    log.info(f"Target samples: {target_n_obs:,}")
    log.info(f"Actual written samples: {final_n:,}")
    log.info(f"Variants: {n_vars:,}")

    stem = ssd_dir / (base_zarr.stem + f"_{target_n_obs}")

    memmap_path = stem.with_suffix(".memmap")
    mm = np.memmap(
        memmap_path,
        mode="w+",
        dtype=np.uint8,
        shape=(final_n, n_vars),
    )

    write_row = 0
    for r in range(repeats):
        log.info(f"Repeat {r + 1}/{repeats}")
        for start in range(0, base_n, block_size):
            end = min(start + block_size, base_n)
            rows = end - start
            mm[write_row:write_row + rows, :] = np.asarray(
                X[start:end, :], dtype=np.uint8
            )
            write_row += rows

    mm.flush()

    meta = {"shape": [final_n, n_vars], "dtype": "uint8", "format": "dense"}
    with open(stem.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f)

    log.info("Dense concatenated memmap written.")
    return memmap_path


def build_datasets(data_dir: Path, ssd_dir: Path, chromosomes: list[int]) -> None:
    """Build rare (MAF<0.01, MAF<0.001) and common (MAF>=0.01) datasets."""
    data_dir = Path(data_dir)
    ssd_dir  = Path(ssd_dir)
    ssd_dir.mkdir(parents=True, exist_ok=True)

    vcf_to_pgen_per_chrom(data_dir, chromosomes)
    merged = merge_pgen(data_dir, chromosomes)

    for maf in [0.01, 0.001]: 
        stem = extract_rare(merged, data_dir, maf)

        zarr_path = stem.parent / (stem.name + ".zarr")
        pgen_to_zarr(stem, zarr_path, sparse=True)

        h5ad_path = write_500k_patients(zarr_path, ssd_dir)

        shuffle_to_zarr(h5ad_path, maf=maf, sparse=True)

    common_stem = extract_common(merged, data_dir, maf_min=0.01)

    common_zarr = common_stem.parent / (common_stem.name + ".zarr")
    pgen_to_zarr(common_stem, common_zarr, sparse=False, max_variants=1_200_000)

    common_h5ad = write_500k_patients(common_zarr, ssd_dir)

    shuffle_to_zarr(common_h5ad, maf=None, sparse=False)

    concatenate_zarr_to_memmap(
        common_zarr,
        ssd_dir=ssd_dir / "common_memmap",
        target_n_obs=500_000,
    )

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ssd-dir", required=True)
    p.add_argument("--cellink-data-home", required=True)
    p.add_argument("--chromosomes", nargs="+", type=int, default=list(range(1, 23)))
    args = p.parse_args()

    from cellink.resources import get_1000genomes_grch38
    get_1000genomes_grch38(config_path="./src/benchmark_loading/1000genomes_grch38.yaml", data_home=args.cellink_data_home, verify_checksum=False, only_download=True)

    build_datasets(
        data_dir=Path(args.data_dir),
        ssd_dir=Path(args.ssd_dir),
        chromosomes=args.chromosomes,
    )