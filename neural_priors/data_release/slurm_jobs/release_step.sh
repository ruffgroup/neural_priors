#!/bin/bash
# One per-subject step of the OpenNeuro release, as a SLURM array over subject numbers.
#   sbatch --array=1-10,12-22,24-41 release_step.sh stage_raw
#   steps: stage_raw | stage_fmriprep | deface_subject | qc_subject
#SBATCH --job-name=np_release
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-cva0000-303  # /shares not visible there (2026-09-16): empty listings, EACCES
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null

set -eo pipefail
STEP=${1:?step name required}
SUBJECT=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
sleep $(( RANDOM % 20 ))

LOGFILE="$HOME/logs/np_release_${STEP}_sub-${SUBJECT}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.txt"
exec >"$LOGFILE" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" name="np_${STEP}" 2>/dev/null || true
echo "Host: $(hostname)  sub-${SUBJECT}  step ${STEP}  started $(date)"

ENV="$HOME/data/conda/envs/np_deface"
export PATH="$ENV/bin:$PATH"
export FSLDIR="$ENV"
export FSLOUTPUTTYPE=NIFTI_GZ
export PYTHONPATH="$HOME/git/neural_priors"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

"$ENV/bin/python" -u -m "neural_priors.data_release.${STEP}" "$SUBJECT"
echo "Finished $(date)"
rm -rf "$TMPDIR"
