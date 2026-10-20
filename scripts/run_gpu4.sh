#!/usr/bin/env bash
# SIRE v0 launcher. Resolves the UUID of physical GPU 4 and exports it before
# Python (and therefore torch) starts. Never switches cards, never kills others.
set -euo pipefail
cd /home/nxq/SIRE

if [ ! -f /home/nxq/miniconda3/etc/profile.d/conda.sh ]; then
  echo "error: Miniconda not found at /home/nxq/miniconda3" >&2
  exit 2
fi
# shellcheck disable=SC1091
source /home/nxq/miniconda3/etc/profile.d/conda.sh
conda activate sire

UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==4 {print $2}')"
if [ -z "${UUID:-}" ]; then
  echo "error: physical GPU 4 not found" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=/home/nxq/SIRE/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[launcher] physical GPU 4 -> $UUID, python $(command -v python)" >&2
exec python -m sire.cli "$@"
