#!/bin/bash
# Dataset-level steps of the OpenNeuro release (single job).
#   sbatch release_global.sh write_metadata   # dataset_description, README, participants, sidecars
#   sbatch release_global.sh validate          # bids-validator (deno) on the upload tree
#   sbatch release_global.sh audit_release     # whole-tree identifier audit (exit 1 on problems)
#SBATCH --job-name=np_release_global
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/gdehol/logs/np_release_global_%j.txt

set -eo pipefail
STEP=${1:?step name required}
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
scontrol update jobid="${SLURM_JOB_ID}" name="np_${STEP}" 2>/dev/null || true
echo "step ${STEP} on $(hostname), $(date)"

ENV="$HOME/data/conda/envs/np_deface"
export PYTHONPATH="$HOME/git/neural_priors"
export PYTHONUNBUFFERED=1
TARGET=/shares/zne.uzh/gdehol/ds-neuralpriors-openneuro

if [[ "$STEP" == validate ]]; then
    export DENO_DIR="/scratch//deno"
    mkdir -p /shares/zne.uzh/gdehol/ds-neuralpriors-openneuro-work/audit
    "$HOME/bin/deno" run -ERWN jsr:@bids/validator "$TARGET" -v \
        > /shares/zne.uzh/gdehol/ds-neuralpriors-openneuro-work/audit/bids_validator.txt 2>&1 || true
    tail -60 /shares/zne.uzh/gdehol/ds-neuralpriors-openneuro-work/audit/bids_validator.txt
else
    "$ENV/bin/python" -u -m "neural_priors.data_release.${STEP}"
fi
echo "finished $(date)"
