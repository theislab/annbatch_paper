#!/usr/bin/env bash
# Add seeds 1 and 2 to the single-seed arms of the convergence matrix.
#
#   LR=1e-4 bash revision/scripts/launch_convergence_seeds.sh [--dry-run]
#
# The headline arms were run with three seeds from the start; the chunk-size
# sensitivity sweep and the large-block scDataset arms were run with one, which
# is enough to establish the trend but not enough to put error bars on a figure.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SB="$REPO/revision/slurm/20_train_convergence.sbatch"
OUT="$REPO/revision/results/20_convergence"
DRY=${1:-}
LR=${LR:?set LR to the value used for the matrix, e.g. LR=1e-4}

submit () {
    local strat=$1 seed=$2
    local f="$OUT/${strat}__bs4096__seed${seed}.json"
    [[ -f "$f" ]] && { echo "skip (exists): $(basename "$f")" >&2; return; }
    if [[ "$DRY" == "--dry-run" ]]; then
        echo "sbatch -J ab_cv_${strat}_bs4096_s${seed} ... --strategy $strat --seed $seed"
    else
        sbatch --nice=300 -J "ab_cv_${strat}_bs4096_s${seed}" "$SB" \
            --strategy "$strat" --seed "$seed" --batch-size 4096 --lr "$LR" \
            --eval-every 500 --log-every 50 --out "$f" >/dev/null
        echo "submitted ${strat} seed ${seed}"
    fi
}

for seed in 1 2; do
    for k in 16 64 128 256 512; do
        submit "annbatch_pre_c${k}" "$seed"
        submit "annbatch_raw_c${k}" "$seed"
    done
    submit streaming_buffer "$seed"
    submit scdataset_b512_f4 "$seed"
    submit scdataset_b1024_f4 "$seed"
    submit scdataset_b16_f32 "$seed"
done
