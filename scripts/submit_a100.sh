#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=sire-a100
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@120
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the SIRE repository root}"
# Discover module name with module avail. Override if this site's name differs.
module load "${SIRE_CONDA_MODULE:-anaconda/2021.11}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SIRE_CONDA_ENV:-sire}"
if [[ $# -eq 0 ]]; then
    set -- run --config configs/a100_40gb.yaml --run-id "a100-${SLURM_JOB_ID}"
fi
# Without exec, Bash receives Slurm's advance signal and forwards it to Python.
child=''
trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 143' TERM INT
bash scripts/run.sh "$@" &
child=$!
wait "$child"
