import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
"""
Complete pipeline: 1000 Genomes → PLINK2 filtering → vcf2zarr → simulation → h5ad output

Strategy for common variants:
  - Download the publicly available UKB WES capture BED (IDT xGen, GRCh38, Resource 3803)
  - Use plink2 --extract-intersect to keep only SNPs that fall inside exome capture regions
  - This yields ~7M common variants (MAF >= 1%) that overlap WES capture space

Requirements (all must be on PATH):
    plink2, bcftools, tabix/bgzip
    pip: sgkit anndata scipy numpy pandas h5py
"""

import logging
import os
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import dask
dask.config.set(num_workers=28)
from annbatch import DatasetCollection
import zarr
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Shell helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: str, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    log.info(f"RUN: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip():
        log.debug(r.stdout.strip())
    if r.stderr.strip():
        log.debug(r.stderr.strip())
    if check and r.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {r.returncode}):\n  {cmd}\nSTDERR:\n{r.stderr}"
        )
    return r

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Download the public UKB WES capture BED (no UKB access needed)
# ─────────────────────────────────────────────────────────────────────────────

def download_ukb_wes_bed(out_dir: Path) -> Path:
    """
    Download the publicly available UKB WES exome capture BED file.

    This is UK Biobank Resource 3803 – the IDT xGen Exome Research Panel v1.0
    + supplemental probes in GRCh38 coordinates. No UKB data access required.
    ~39 Mbp of exome capture regions.

    Returns path to the downloaded BED file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bed_path = out_dir / "xgen_plus_spikein.GRCh38.bed"

    if bed_path.exists():
        log.info(f"UKB WES BED already present: {bed_path}")
        return bed_path

    url = "https://biobank.ndph.ox.ac.uk/ukb/ukb/auxdata/xgen_plus_spikein.GRCh38.bed"
    log.info(f"Downloading UKB WES BED from {url}")
    _run(f"wget -q -O {bed_path} {url}")

    # Sanity check
    lines = int(_run(f"wc -l < {bed_path}", check=False).stdout.strip() or 0)
    log.info(f"UKB WES BED downloaded: {lines} regions")
    if lines < 100:
        raise RuntimeError(
            f"BED file looks too small ({lines} lines). "
            "Download may have failed. Check: {bed_path}"
        )
    return bed_path


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VCF → pgen (per chromosome, biallelic SNPs only)
# ─────────────────────────────────────────────────────────────────────────────

def vcf_to_pgen_per_chrom(data_dir: Path, chromosomes: list[int]) -> None:
    """Convert per-chromosome 1000G VCFs to PLINK2 .pgen, biallelic SNPs only."""
    pgen_dir = data_dir / "pgen"
    pgen_dir.mkdir(exist_ok=True)

    for chrom in chromosomes:
        out = pgen_dir / f"chr{chrom}"
        if (pgen_dir / f"chr{chrom}.pgen").exists():
            log.info(f"chr{chrom} pgen exists, skipping.")
            continue

        vcf = (
            data_dir
            / f"20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}.recalibrated_variants.vcf.gz"
        )
        if not vcf.exists():
            raise FileNotFoundError(f"VCF missing: {vcf}")

        _run(
            f"plink2 --vcf {vcf} "
            f"--make-pgen "
            f"--max-alleles 2 "
            f"--snps-only just-acgt "
            f"--set-all-var-ids @:#:\\$r:\\$a "
            f"--new-id-max-allele-len 1000 missing "
            f"--vcf-half-call m "
            f"--output-chr chrM "
            f"--out {out}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Merge chromosomes into a single pgen (makes --extract-intersect easier)
# ─────────────────────────────────────────────────────────────────────────────

def merge_pgen(data_dir: Path, chromosomes: list[int]) -> Path:
    """Merge per-chromosome pgen files into one."""
    pgen_dir = data_dir / "pgen"
    merged = pgen_dir / "all_chroms"

    if (pgen_dir / "all_chroms.pgen").exists():
        log.info("Merged pgen exists, skipping.")
        return merged

    # Write pgen-list file
    list_file = pgen_dir / "pgen_list.txt"
    with open(list_file, "w") as f:
        for c in chromosomes:
            f.write(f"{pgen_dir / f'chr{c}'}\n")

    _run(
        f"plink2 --pmerge-list {list_file} pfile "
        f"--make-pgen "
        f"--out {merged}"
    )
    return merged


def extract_rare_variants(
    data_dir: Path,
    merged_pgen: Path,
    maf_threshold: float = 0.001,
) -> Path:
    """
    MAF < maf_threshold, SNPs only, biallelic.
    Outputs PLINK2 PGEN.
    """
    out_dir = data_dir / "rare"
    out_dir.mkdir(exist_ok=True)
    out_stem = out_dir / "rare_variants"
    out_pgen = Path(str(out_stem) + ".pgen")

    if out_pgen.exists():
        log.info(f"Rare PGEN exists: {out_pgen}")
        return out_pgen

    _run(
        f"plink2 --pfile {merged_pgen} "
        f"--max-maf {maf_threshold - 1e-9:.8f} "
        f"--maf 1e-9 "
        f"--snps-only just-acgt "
        f"--make-pgen "
        f"--out {out_stem}"
    )

    # Count variants from .pvar (faster than bcftools)
    pvar = Path(str(out_stem) + ".pvar")
    n = int(_run(f"grep -vc '^#' {pvar}").stdout.strip())
    log.info(f"Rare variants: {n:,}")

    return out_pgen

def extract_common_variants(
    data_dir: Path,
    merged_pgen: Path,
    maf_min: float = 0.01,
) -> Path:
    """
    MAF >= maf_min, SNPs only.
    Outputs PLINK2 PGEN.
    """
    out_dir = data_dir / "common"
    out_dir.mkdir(exist_ok=True)
    out_stem = out_dir / "common_variants"
    out_pgen = Path(str(out_stem) + ".pgen")

    if out_pgen.exists():
        log.info(f"Common PGEN exists: {out_pgen}")
        return out_pgen

    _run(
        f"plink2 --pfile {merged_pgen} "
        f"--maf {maf_min:.8f} "
        f"--snps-only just-acgt "
        f"--make-pgen "
        f"--out {out_stem}"
    )

    pvar = Path(str(out_stem) + ".pvar")
    n = int(_run(f"grep -vc '^#' {pvar}").stdout.strip())
    log.info(f"Common variants: {n:,}")

    return out_pgen


def convert_and_write(
    pgen_stem: Path,
    work_zarr: Path,
    chunk_samples: int = 3202,
    chunk_variants: int = 32000,
    memory_limit_gb: float = 450.0,
    sparse: bool = False,
) -> None:
    """Call stream_pgen_to_zarr(), then write h5ad with gzip compression."""
    if work_zarr.exists():
        log.info(f"zarr exists: {work_zarr}")
        return

    from cellink.io import stream_pgen_to_zarr

    log.info(f"stream_pgen_to_zarr: {pgen_stem} → {work_zarr}")
    stream_pgen_to_zarr(
        pgen_path=str(pgen_stem),
        output_path=str(work_zarr),
        chunk_samples=chunk_samples,
        chunk_variants=chunk_variants,
        memory_limit_gb=memory_limit_gb,
        sparse=sparse,
        sparse_format="csr",
    )
    #adata.write_h5ad(out_h5ad, compression="gzip", compression_opts=4)
    #log.info(f"  Done. {out_h5ad.stat().st_size / 1e9:.2f} GB")

def write_100000_patients(work_zarr: Path):

    from cellink.io import read_pgen_zarr

    adata_sub = read_pgen_zarr(work_zarr)
    adata = ad.concat([adata_sub]*32)#[:5000]
    adata.write_h5ad(work_zarr.with_name(work_zarr.stem + '_100k.h5ad'), compression="gzip")

def prep_annbatch(work_zarr: Path):

    zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
    )

    collection = DatasetCollection(
        zarr.open(work_zarr.with_suffix('_100k_shuffled.zarr'), mode="w")
    )
    collection.add_adatas(
        adata_paths=[
            work_zarr.with_name(work_zarr.stem + '_100k.h5ad')
        ],
        shuffle=True, # shuffling is needed if you want to use chunked access, but is the default
        n_obs_per_dataset=10000,
    )


def shuffle_to_zarr(work_zarr: Path) -> None:
    """
    Use annbatch DatasetCollection to shuffle an h5ad and write a shuffled zarr.
    """
    from annbatch import DatasetCollection

    h5ad_path = work_zarr.with_name(work_zarr.stem + '_100k.h5ad')
    shuffled_zarr_path = work_zarr.with_name(work_zarr.stem + '_100k_shuffled.zarr')

    if shuffled_zarr_path.exists():
        log.info(f"Shuffled zarr already exists, skipping: {shuffled_zarr_path}")
        return

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

    log.info(f"Shuffling {h5ad_path} → {shuffled_zarr_path}")
    collection = DatasetCollection(
        zarr.open(str(shuffled_zarr_path), mode="w")
    )
    collection.add_adatas(
        adata_paths=[str(h5ad_path)],
        shuffle=True,
        n_obs_per_dataset=4096,
        zarr_dense_chunk_size=64,
        zarr_dense_shard_size=2048,
        #zarr_sparse_chunk_size=64,
        #zarr_sparse_shard_size=2048,
        shuffle_chunk_size=64,
    )
    log.info(f"Done: {shuffled_zarr_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_simulation_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    n_simulated: int = 500_000,
    maf_rare_threshold: float = 0.001,
    max_common_variants: int = 10_000_000,
    worker_processes: int = 8,
    max_memory: str = "64G",
    chromosomes: list[int] | None = None,
    chunk_samples_rare: int = 5_000,
    chunk_samples_common: int = 2_000,
    rng_seed: int = 42,
    rerun_zarr: bool = False,
) -> None:
    """
    End-to-end pipeline:

    1.  Download public UKB WES capture BED (no UKB access required)
    2.  Convert 1000G VCFs → pgen (biallelic SNPs)
    3.  Merge chromosomes → single pgen
    4.  Filter → rare VCF  (MAF < 1%)
    5.  Filter → common VCF (MAF ≥ 1%, inside WES capture regions)
    6.  Convert both VCFs → zarr via vcf2zarr
    7.  Load zarr → AnnData, compute empirical MAF from 2504 1000G samples
    8.  Simulate 500k individuals under HWE
    9.  Write rare  → sparse  .h5ad  (gzip-4)
       Write common → dense   .h5ad  (gzip-4)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chromosomes = chromosomes or list(range(1, 23))

    from cellink.resources import get_1000genomes_grch38
    gdata = get_1000genomes_grch38(config_path="../cellink/src/cellink/resources/config/1000genomes_grch38.yaml", data_home="../../data/cellink_data", verify_checksum=False, only_download=True)

    """
    # ── 1. UKB WES BED ────────────────────────────────────────────────────────
    log.info("=== Step 1: Download UKB WES capture BED ===")
    ukb_bed = download_ukb_wes_bed(data_dir / "ukb_wes")

    # ── 2–3. VCF → pgen ───────────────────────────────────────────────────────
    log.info("=== Step 2: VCF → per-chrom pgen ===")
    vcf_to_pgen_per_chrom(data_dir, chromosomes)

    log.info("=== Step 3: Merge pgen chromosomes ===")
    merged_pgen = merge_pgen(data_dir, chromosomes)
    """
    ukb_bed = "/home/icb/lucas.arnoldt/workspace/data/cellink_data/1000genomes_grch38/ukb_wes/xgen_plus_spikein.GRCh38.bed"
    merged_pgen = "/home/icb/lucas.arnoldt/workspace/data/cellink_data/1000genomes_grch38/pgen/all_chroms"
    rare_vcf = "/home/icb/lucas.arnoldt/workspace/data/cellink_data/1000genomes_grch38/rare/rare_variants.vcf.gz"

    # ── 4. Rare VCF ───────────────────────────────────────────────────────────
    log.info("=== Step 4: Extract rare variants (MAF < 1%) ===")
    rare_vcf = extract_rare_variants(data_dir, merged_pgen, 0.001)
    
    # ── 5. Common VCF ─────────────────────────────────────────────────────────
    log.info("=== Step 5: Extract common variants (MAF ≥ 1%, WES-intersect) ===")
    common_vcf = extract_common_variants(
        data_dir, merged_pgen, #ukb_bed,
        maf_min=0.01,
        #max_variants=max_common_variants,
    )

    #convert_and_write(common_vcf, common_vcf.with_suffix('.zarr'))

    #convert_and_write(rare_vcf, rare_vcf.with_suffix('.zarr'), sparse=True)

    #write_100000_patients(common_vcf.with_suffix('.zarr'))

    #write_100000_patients(rare_vcf.with_suffix('.zarr'))

    #shuffle_to_zarr(common_vcf.with_suffix('.zarr'))

    shuffle_to_zarr(rare_vcf.with_suffix('.zarr'))


# ─────────────────────────────────────────────────────────────────────────────
# 13.  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="1000G → simulated 500k genotype pipeline")
    #p.add_argument("--data-dir",    required=True, help="Dir containing 1000G VCF files")
    #p.add_argument("--output-dir",  required=True, help="Output dir for .h5ad files")
    p.add_argument("--n-simulated", type=int, default=500_000)
    p.add_argument("--chromosomes", nargs="+", type=int, default=None)
    p.add_argument("--worker-processes", type=int, default=8)
    p.add_argument("--max-memory",  default="64G")
    p.add_argument("--max-common-variants", type=int, default=10_000_000)
    p.add_argument("--chunk-samples-rare",   type=int, default=5_000)
    p.add_argument("--chunk-samples-common", type=int, default=2_000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--rerun-zarr",  action="store_true")
    args = p.parse_args()

    data_dir = "/home/icb/lucas.arnoldt/workspace/data/cellink_data/1000genomes_grch38"
    output_dir = "/home/icb/lucas.arnoldt/workspace/data/annbatch_1000genomes_grch38"

    build_simulation_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        n_simulated=args.n_simulated,
        chromosomes=args.chromosomes,
        worker_processes=args.worker_processes,
        max_memory=args.max_memory,
        max_common_variants=args.max_common_variants,
        chunk_samples_rare=args.chunk_samples_rare,
        chunk_samples_common=args.chunk_samples_common,
        rng_seed=args.seed,
        rerun_zarr=args.rerun_zarr,
    )