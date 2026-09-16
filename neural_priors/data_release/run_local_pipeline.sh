#!/bin/bash
# Run the whole release pipeline on a single machine without SLURM (e.g. a
# ScienceCloud VM), N subjects in parallel per step. Set NP_RELEASE_SOURCE /
# NP_RELEASE_TARGET / NP_RELEASE_WORK and ENV (conda env with FSL flirt + pydeface).
#   ENV=/data/miniforge3/envs/np_deface N=12 bash run_local_pipeline.sh [steps...]
set -eo pipefail
ENV=${ENV:?set ENV to the np_deface conda env}
N=${N:-12}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STEPS=${*:-stage_raw stage_fmriprep deface_subject qc_subject write_metadata}
LOGDIR="$NP_RELEASE_WORK/logs"
mkdir -p "$LOGDIR"
export PATH="$ENV/bin:$PATH" FSLDIR="$ENV" FSLOUTPUTTYPE=NIFTI_GZ PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
SUBJECTS=$("$ENV/bin/python" -c 'from neural_priors.data_release.release import get_subjects; print(" ".join(get_subjects()))')

for step in $STEPS; do
  echo "== $step  $(date)"
  if [[ "$step" == write_metadata || "$step" == audit_release ]]; then
    "$ENV/bin/python" -u -m "neural_priors.data_release.$step" > "$LOGDIR/$step.txt" 2>&1 || { echo "FAILED $step (see $LOGDIR/$step.txt)"; [[ "$step" == audit_release ]] || exit 1; }
    continue
  fi
  failed=0
  printf '%s\n' $SUBJECTS | xargs -P "$N" -I{} bash -c \
    "\"$ENV/bin/python\" -u -m neural_priors.data_release.$step {} > \"$LOGDIR/${step}_sub-{}.txt\" 2>&1 && echo ok sub-{} || echo FAILED sub-{}" \
    | tee "$LOGDIR/${step}_summary.txt"
  if grep -q FAILED "$LOGDIR/${step}_summary.txt"; then echo "step $step had failures"; exit 1; fi
done
echo "== all done $(date)"
