#!/bin/sh

#SBATCH -J parallel_bench
#SBATCH --output=parallel_bench.%j
#SBATCH --error=parallel_bench.%j
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --mem=356GB
#SBATCH --cpus-per-task=64


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/annbatch.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/annbatch_paper/src/run_parallel_benchmarks.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/annbatch_paper"


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "pip install -e ${GIT_REPO} && pip install lamindb && pip install cupy-cuda13x && pip install scDataset && python -u ${SCRIPT}"
