#!/usr/bin/env bash
# Read-only login-node checks. No GPU allocation, package install or dataset compute.
set -uo pipefail
SIRE_PREFLIGHT_FILE="${1:-cluster-preflight-$(date +%Y%m%d-%H%M%S).txt}"
{
    echo 'SIRE cluster preparation: read-only inspection'
    date -Is
    hostname
    id -un
    uname -m
    cat /etc/os-release
    echo '--- user home usage (not quota) ---'
    timeout -k 2s 20s du -sh "$HOME" || true
    echo '--- quota: refusal/timeout does not mean unlimited ---'
    timeout -k 2s 10s quota -s || true
    echo '--- configured modules ---'
    if type module >/dev/null 2>&1; then module avail; else echo 'module not available in this shell'; fi
    echo '--- site prebuilt packages (names only) ---'
    ls -lh /home/bingxing2/apps/package || true
    echo '--- existing user conda ---'
    command -v conda || true
    echo '--- current Python (no model/GPU execution) ---'
    command -v python3 || true
    python3 --version || true
    echo '--- GPU partition metadata ---'
    timeout -k 2s 10s scontrol show partition gpu || true
} 2>&1 | tee "$SIRE_PREFLIGHT_FILE"
echo "Saved inspection to $SIRE_PREFLIGHT_FILE"
