#!/bin/bash

set -e  

# Configuration
TOTAL_FILES=966
NUM_BATCHES=9
OUTPUT_DIR="./vcf_files"
BASE_URL="gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/exome/vcf"

# Processing parameters
MAX_PARALLEL_BATCHES=9  # Process all 9 batches in parallel
EXPLODE_WORKER_PROCESSES=5  # Worker processes for each explode job
ENCODE_WORKER_PROCESSES=8  # Worker processes for encode (use more since sequential)
ENCODE_MAX_MEMORY="100G"  # Max memory for encode (leave ~20GB for system)
VARIANTS_CHUNK_SIZE=16000
SAMPLES_CHUNK_SIZE=16000

# Fields to exclude from final VCZ (keep call_genotype_phased and call_genotype_mask as they are required)
EXCLUDE_FIELDS='"call_AD", "call_RGQ", "call_PS", "call_PID", "call_PGT", "call_GQ", "call_FT"'

# Google Cloud Project (must be set in environment)
if [ -z "$GOOGLE_PROJECT" ]; then
    echo "Error: GOOGLE_PROJECT environment variable must be set"
    echo "Example: export GOOGLE_PROJECT=your-project-id"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "VCF to Zarr Conversion Pipeline"
echo "=========================================="
echo "Total VCF files: $TOTAL_FILES"
echo "Number of batches: $NUM_BATCHES"
echo "Files per batch: ~$(($TOTAL_FILES / $NUM_BATCHES))"
echo "Output directory: $OUTPUT_DIR"
echo "Max parallel batches: $MAX_PARALLEL_BATCHES"
echo "Explode worker processes: $EXPLODE_WORKER_PROCESSES per batch"
echo "Encode worker processes: $ENCODE_WORKER_PROCESSES"
echo "Encode max memory: $ENCODE_MAX_MEMORY"
echo "Variants chunk size: $VARIANTS_CHUNK_SIZE"
echo "Samples chunk size: $SAMPLES_CHUNK_SIZE"
echo "Google Project: $GOOGLE_PROJECT"
echo "=========================================="

FILES_PER_BATCH=$(( ($TOTAL_FILES + $NUM_BATCHES - 1) / $NUM_BATCHES ))

process_batch() {
    local batch_num=$1
    local start_index=$(($batch_num * $FILES_PER_BATCH))
    local end_index=$(($start_index + $FILES_PER_BATCH - 1))
    
    if [ $end_index -ge $TOTAL_FILES ]; then
        end_index=$(($TOTAL_FILES - 1))
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing Batch $batch_num"
    echo "Files: $start_index to $end_index ($(($end_index - $start_index + 1)) files)"
    echo "=========================================="
    
    local icf_path="$OUTPUT_DIR/batch_${batch_num}.icf"
    
    if [ -d "$icf_path" ]; then
        echo "[Batch $batch_num] ✓ ICF already exists, skipping download and explode"
        return 0
    fi
    
    echo ""
    echo "[Batch $batch_num] Step 1: Downloading VCF files..."
    echo "----------------------------------------"
    
    local vcf_files=()
    for file_index in $(seq $start_index $end_index); do
        local file_id=$(printf "%010d" $file_index)
        local vcf_file="$OUTPUT_DIR/${file_id}.vcf.bgz"
        local tbi_file="${vcf_file}.tbi"
        
        if [ ! -f "$vcf_file" ]; then
            echo "  Downloading ${file_id}.vcf.bgz..."
            gsutil -u "$GOOGLE_PROJECT" cp "${BASE_URL}/${file_id}.vcf.bgz" "$vcf_file"
        else
            echo "  ✓ ${file_id}.vcf.bgz already exists, skipping"
        fi
        
        if [ ! -f "$tbi_file" ]; then
            echo "  Downloading ${file_id}.vcf.bgz.tbi..."
            gsutil -u "$GOOGLE_PROJECT" cp "${BASE_URL}/${file_id}.vcf.bgz.tbi" "$tbi_file"
        else
            echo "  ✓ ${file_id}.vcf.bgz.tbi already exists, skipping"
        fi
        
        vcf_files+=("$vcf_file")
    done
    
    echo "[Batch $batch_num] ✓ Download complete"
    
    echo ""
    echo "[Batch $batch_num] Step 2: Exploding VCF files to ICF..."
    echo "----------------------------------------"
    
    echo "[Batch $batch_num] Running vcf2zarr explode with $(($end_index - $start_index + 1)) VCF files and $EXPLODE_WORKER_PROCESSES worker processes..."
    
    if vcf2zarr explode -p "$EXPLODE_WORKER_PROCESSES" "${vcf_files[@]}" "$icf_path"; then
        echo "[Batch $batch_num] ✓ Successfully exploded to $icf_path"
        
        echo "[Batch $batch_num] Cleaning up VCF files..."
        for vcf_file in "${vcf_files[@]}"; do
            rm -f "$vcf_file"
            rm -f "${vcf_file}.tbi"
        done
        echo "[Batch $batch_num] ✓ VCF files deleted"
    else
        echo "[Batch $batch_num] ✗ Error during explode"
        return 1
    fi
    
    echo "[Batch $batch_num] ✓ Batch processing complete"
    return 0
}

export -f process_batch
export OUTPUT_DIR
export GOOGLE_PROJECT
export BASE_URL
export FILES_PER_BATCH
export TOTAL_FILES
export EXPLODE_WORKER_PROCESSES

echo ""
echo "=========================================="
echo "Phase 1: Download + Explode (all batches in parallel)"
echo "=========================================="

seq 0 $(($NUM_BATCHES - 1)) | \
    xargs -P "$MAX_PARALLEL_BATCHES" -I {} bash -c 'process_batch "$@"' _ {}

echo ""
echo "=========================================="
echo "Phase 1 Complete: All batches exploded"
echo "=========================================="

echo ""
echo "=========================================="
echo "Phase 2: Encode ICF to VCZ (sequential)"
echo "=========================================="

for batch_num in $(seq 0 $(($NUM_BATCHES - 1))); do
    icf_path="$OUTPUT_DIR/batch_${batch_num}.icf"
    vcz_path="$OUTPUT_DIR/batch_${batch_num}.vcz"
    schema_path="$OUTPUT_DIR/batch_${batch_num}_minimal.schema.json"
    
    echo ""
    echo "----------------------------------------"
    echo "Encoding Batch $batch_num"
    echo "----------------------------------------"
    
    if [ ! -d "$icf_path" ]; then
        echo "⚠ ICF not found for batch $batch_num, skipping"
        continue
    fi
    
    if [ -d "$vcz_path" ]; then
        echo "✓ VCZ already exists for batch $batch_num, skipping"
        continue
    fi
    
    echo "Generating minimal schema..."
    vcf2zarr mkschema "$icf_path" | \
        jq ".dimensions.variants.chunk_size = $VARIANTS_CHUNK_SIZE | 
            .dimensions.samples.chunk_size = $SAMPLES_CHUNK_SIZE | 
            del(.fields[] | select(.name | IN($EXCLUDE_FIELDS)))" \
        > "$schema_path"
    
    if [ $? -ne 0 ]; then
        echo "✗ Error generating schema for batch $batch_num"
        continue
    fi
    echo "✓ Schema generated"
    
    echo "Encoding to VCZ with $ENCODE_WORKER_PROCESSES worker processes and max memory $ENCODE_MAX_MEMORY..."
    if vcf2zarr encode \
        -p "$ENCODE_WORKER_PROCESSES" \
        -M "$ENCODE_MAX_MEMORY" \
        -s "$schema_path" \
        "$icf_path" \
        "$vcz_path"; then
        echo "✓ Successfully encoded batch $batch_num"
    else
        echo "✗ Error encoding batch $batch_num"
        continue
    fi
    
    echo "✓ Batch $batch_num encode complete!"
done

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Total VCZ batches created: $(find "$OUTPUT_DIR" -name "batch_*.vcz" -type d | wc -l)"
echo "  Remaining ICF files: $(find "$OUTPUT_DIR" -name "batch_*.icf" -type d | wc -l)"
echo "  Remaining VCF files: $(find "$OUTPUT_DIR" -name "*.vcf.bgz" -type f | wc -l)"
echo ""
echo "Output VCZ files:"
find "$OUTPUT_DIR" -name "batch_*.vcz" -type d