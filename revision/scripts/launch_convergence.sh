#!/usr/bin/env bash
# Submit the full convergence run matrix (Reviewer 2, major concern 3).
#
#   bash revision/scripts/launch_convergence.sh [--dry-run]
#
# Three seeds for the five headline arms, one seed for the chunk-size sweep.
#
# The scDataset arms use fetch_factor=4 at batch_size 4096, i.e. a 16,384-row
# buffer -- the same buffer as annbatch's shipped default, and the same buffer
# as scDataset's own recommended setting (block 16, fetch 256) at its minibatch
# of 64.  10_batch_diversity.py shows that minibatch diversity depends on
# buffer_rows / block_size, so matching the buffer is what makes the comparison
# a comparison of implementations rather than of memory budgets.
#
# The replication block reruns four arms at scDataset's own protocol
# (batch_size=64, lr=1e-5) so nobody can object that we changed their setup.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SB="$REPO/revision/slurm/20_train_convergence.sbatch"
DRY=${1:-}
LR=${LR:?set LR to the value chosen by the learning-rate sweep, e.g. LR=1e-4}

submit () {  # submit <strategy> <seed> <batch_size> <lr> [extra...]
    local strat=$1 seed=$2 bs=$3 lr=$4; shift 4
    local name="ab_cv_${strat}_bs${bs}_s${seed}"
    if [[ "$DRY" == "--dry-run" ]]; then
        echo "sbatch -J $name $SB --strategy $strat --seed $seed --batch-size $bs --lr $lr $*"
    else
        sbatch -J "$name" "$SB" --strategy "$strat" --seed "$seed" --batch-size "$bs" --lr "$lr" "$@"
    fi
}

# --- headline arms, 3 seeds ------------------------------------------------- #
for seed in 0 1 2; do
    for strat in random streaming annbatch_pre_c1024 annbatch_raw_c1024 scdataset_b16_f4; do
        submit "$strat" "$seed" 4096 "$LR" --eval-every 500 --log-every 50
    done
done

# --- chunk-size sensitivity, 1 seed ---------------------------------------- #
for k in 16 64 128 256 512; do
    submit "annbatch_pre_c${k}" 0 4096 "$LR" --eval-every 500 --log-every 50
    submit "annbatch_raw_c${k}" 0 4096 "$LR" --eval-every 500 --log-every 50
done
submit streaming_buffer 0 4096 "$LR" --eval-every 500 --log-every 50
submit scdataset_b512_f4 0 4096 "$LR" --eval-every 500 --log-every 50
submit scdataset_b1024_f4 0 4096 "$LR" --eval-every 500 --log-every 50

# --- replication at scDataset's own protocol -------------------------------- #
# ~10 h per arm: at minibatch 64 the Adam step over 158M parameters dominates and
# is batch-size independent, so throughput drops to ~2,600 samples/s.  These are
# a corroboration of the primary sweep rather than the headline, so they are
# submitted at a lower priority and must not delay the arms above.
NICE=200
for strat in random streaming scdataset_b16_f256 annbatch_pre_c1024; do
    if [[ "$DRY" == "--dry-run" ]]; then
        echo "sbatch --nice=$NICE -J ab_cv_${strat}_bs64_s0 $SB --strategy $strat --seed 0 --batch-size 64 --lr 1e-5 --eval-every 20000 --log-every 2000"
    else
        sbatch --nice="$NICE" -J "ab_cv_${strat}_bs64_s0" "$SB" --strategy "$strat" --seed 0 \
            --batch-size 64 --lr 1e-5 --eval-every 20000 --log-every 2000
    fi
done
