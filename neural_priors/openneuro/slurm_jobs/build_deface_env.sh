#!/bin/bash
#SBATCH --job-name=build_np_deface
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=45:00
#SBATCH --output=/home/gdehol/logs/build_np_deface_%j.txt

set -eo pipefail
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda env create -y -p "$HOME/data/conda/envs/np_deface" \
    -f "$HOME/git/neural_priors/neural_priors/openneuro/environment_deface.yml"
conda activate "$HOME/data/conda/envs/np_deface"
echo "FSLDIR=$FSLDIR"; which flirt; pydeface --help | head -40
conda env export -p "$HOME/data/conda/envs/np_deface" \
    > "$HOME/git/neural_priors/neural_priors/openneuro/environment_deface.lock.yml"
