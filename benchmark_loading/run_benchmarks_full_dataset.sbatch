#!/bin/sh

#SBATCH -J annbatch_2
#SBATCH --output=annbatch_2.%j
#SBATCH --error=annbatch_2.%j
#SBATCH --partition=lrz-dgx-a100-40x8-mig
#SBATCH --qos=mig
#SBATCH --gres=gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/annbatch.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/arrayloader-benchmarks/src/run_benchmarks_full_dataset.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/arrayloader-benchmarks"


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "pip install -e ${GIT_REPO} && pip install lamindb && pip install cupy-cuda13x && pip install scDataset && python -u ${SCRIPT}"
