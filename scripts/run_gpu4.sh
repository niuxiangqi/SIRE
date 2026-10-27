#!/usr/bin/env bash
# SIRE v0 launcher. Resolves the UUID of the configured physical GPU and exports
# it before Python (and therefore torch) starts. Never switches cards silently and
# never kills other users' processes.
#
# Default physical index is 2 (user-authorized change on 2026-10-27 after GPU 4
# failed at the PCIe level). Override with SIRE_GPU_INDEX=<index>.
set -euo pipefail
cd /home/nxq/SIRE

if [ ! -f /home/nxq/miniconda3/etc/profile.d/conda.sh ]; then
  echo "error: Miniconda not found at /home/nxq/miniconda3" >&2
  exit 2
fi
# shellcheck disable=SC1091
source /home/nxq/miniconda3/etc/profile.d/conda.sh
conda activate sire

GPU_INDEX="${SIRE_GPU_INDEX:-2}"
UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
  | awk -F', ' -v idx="$GPU_INDEX" '$1==idx {print $2}')"
if [ -z "${UUID:-}" ]; then
  echo "error: physical GPU ${GPU_INDEX} not found" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=/home/nxq/SIRE/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[launcher] physical GPU ${GPU_INDEX} -> $UUID, python $(command -v python)" >&2
exec python -m sire.cli "$@"
