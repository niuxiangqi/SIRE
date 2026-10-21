#!/usr/bin/env bash
# Portable launcher: activate conda before calling, or set SIRE_CONDA_SH.
set -euo pipefail
SIRE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIRE_REPO_ROOT"
if [[ -n "${SIRE_CONDA_SH:-}" ]]; then
    source "$SIRE_CONDA_SH"
    conda activate "${SIRE_CONDA_ENV:-sire}"
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo 'Activate the sire conda environment first (or set SIRE_CONDA_SH).' >&2
    exit 2
fi
export PYTHONPATH="$SIRE_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# Device visibility is configured by Python; in Slurm it is never overridden.
exec python -m sire.cli "$@"
