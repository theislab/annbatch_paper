#!/bin/bash
# Usage: ./run_chr_batch_parallel_download.sh <start_chr> <end_chr>
# Example: ./run_chr_batch_parallel_download.sh 1 8

set -euo pipefail

START_CHR=$1
END_CHR=$2

BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot"
DATA_DIR="/vol/data/annbatch/cellink_data/1000genomes_grch38"
PGEN_DIR="${DATA_DIR}/pgen"

mkdir -p "${DATA_DIR}" "${PGEN_DIR}" logs

########################################
# Step 1: Launch background downloads
########################################
echo "Starting background downloads for chromosomes ${START_CHR}-${END_CHR}..."

for CHROM in $(seq "${START_CHR}" "${END_CHR}"); do
    VCF="20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr${CHROM}.recalibrated_variants.vcf.gz"
    VCF_PATH="${DATA_DIR}/${VCF}"
    TBI_PATH="${VCF_PATH}.tbi"

    # Skip if already downloaded
    if [[ -f "${VCF_PATH}" && -f "${TBI_PATH}" ]]; then
        echo "Chr${CHROM} already downloaded."
        continue
    fi

    # Download VCF in background
    (
        echo "Downloading chr${CHROM}..."
        wget -c "${BASE_URL}/${VCF}" -O "${VCF_PATH}"
        wget -c "${BASE_URL}/${VCF}.tbi" -O "${TBI_PATH}"
        echo "Download complete for chr${CHROM}"
    ) &
done

# Wait for all downloads to start
# ensures all background wget processes are launched (but not necessarily finished)
echo "All downloads initiated (some may still be running in background)."

########################################
# Step 2: Sequential PGEN conversion
########################################
for CHROM in $(seq "${START_CHR}" "${END_CHR}"); do
    VCF="20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr${CHROM}.recalibrated_variants.vcf.gz"
    VCF_PATH="${DATA_DIR}/${VCF}"
    TBI_PATH="${VCF_PATH}.tbi"
    OUT="${PGEN_DIR}/chr${CHROM}"

    # Skip if PGEN exists
    if [[ -f "${OUT}.pgen" ]]; then
        echo "PGEN already exists for chr${CHROM}, skipping."
        continue
    fi

    # Wait for VCF & TBI to be ready
    while [[ ! -f "${VCF_PATH}" || ! -f "${TBI_PATH}" ]]; do
        echo "Waiting for downloads of chr${CHROM} to finish..."
        sleep 30
    done

    echo "Running plink2 for chr${CHROM}..."
    plink2 \
      --vcf "${VCF_PATH}" \
      --make-pgen \
      --max-alleles 2 \
      --snps-only just-acgt \
      --set-all-var-ids @:#:\$r:\$a \
      --new-id-max-allele-len 1000 missing \
      --vcf-half-call m \
      --threads 6 \
      --memory 30000 \
      --out "${OUT}"

    echo "Finished chr${CHROM}, deleting VCF and index..."
    rm -f "${VCF_PATH}" "${TBI_PATH}"
done

echo "Batch ${START_CHR}-${END_CHR} complete."



#tmux new -s chr_batch
#./run_chr_batch_parallel_download.sh 1 8

#tmux new -s chr_batch2
#./run_chr_batch_parallel_download.sh 9 22
