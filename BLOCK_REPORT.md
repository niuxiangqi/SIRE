# BLOCK REPORT — SIRE v0

Status: **blocked by hardware unavailability of physical GPU 4. Formal run not started.**
Date: 2026-10-20 (UTC).

This is a block report, not an acceptance report. No target numbers are reported as
measured results, because the held-out test was never opened.

## 1. What was being done when the block occurred

The GPU-dependent development flow was being exercised end-to-end on physical GPU 4
(UUID `GPU-d7952610-13af-406a-e102-716a3f90ad66`) before starting the formal 7200 s run:

1. `prepare-data` succeeded: NuminaMath-CoT cn_k12 numeric subset, near-duplicate
   clustering, frozen pilot/calibration/search/selection/test + No Robots monitor splits.
2. `audit` succeeded on the real models: 339/339 parameter names, no shape or dtype
   mismatch; tokenizer vocabulary identical (151665 entries, no same-token/different-id);
   config differences `rope_theta` (1e6 vs 1e4) and `sliding_window` (131072 vs 4096).
3. `collect-features` succeeded: D/G/U feature files (37 GB), 8 calibration samples,
   FP32 R parameters with TF32 matmul, per-sample backward, FP32 accumulation.
   `U = mean|g|` vs `|mean g|` max relative difference 0.473 (non-substitutable).
4. A real 4-item pilot generation on full R succeeded: 4/4 correct, 5301 new tokens,
   58 s wall, no truncation, mean 1325 new tokens/item.
5. Correctness checks were run on real weights: 12/13 passed; the one failure was a
   bitwise-vs-tolerance comparison in `hand_criteria_matches_dsl_real`, fixed to
   `allclose`; the fixed test could not be re-run after the device failed.
6. The first real search stage was launched. During the first candidate evaluation,
   after mask construction and weight swap, the CUDA context failed in the sampling
   kernel with `CUDA error: unknown error`; the device then disappeared from
   `nvidia-smi` as `0000:1A:00.0: Unknown Error`.

## 2. Raw evidence

All evidence is in `evidence/gpu4_failure_20261020/`:

- `evidence.txt` — `nvidia-smi` query, `lspci`, sysfs AER/link state, time.
- `last_error.json` — full traceback from the failed search process.
- `budget.json` — dev run ledger (20.5 s GPU booked; this is development time, not the
  formal 7200 s budget).
- `gpu_monitor.jsonl` — device monitor from the failed process.

Key raw lines:

```
Unable to determine the device handle for GPU4: 0000:1A:00.0: Unknown Error
No devices were found
1a:00.0 3D controller: NVIDIA Corporation Device 20f5 (rev ff)
current_link_speed: Unknown
current_link_width: 63
torch.AcceleratorError: CUDA error: unknown error
```

`/proc/driver/nvidia/gpus/0000:1a:00.0/information` still names the correct UUID
`GPU-d7952610-13af-406a-e102-716a3f90ad66` and device minor 4, so this is the intended
card; the driver can no longer obtain a device handle.

Attempted recovery, all failed without root:

- `nvidia-smi --gpu-reset -i 4`
- `nvidia-smi -r -i 4`
- `nvidia-smi --gpu-reset -i GPU-d7952610-...`
- waited several minutes; no recovery
- `sudo` is not available (password required); sysfs unbind/rebind is root-owned
- `dmesg` is not readable by this user, so the Xid code cannot be captured from here

Other GPUs (3, 5, 6, 7) are occupied by other users. Per the contract, switching cards,
killing other users' processes, or rebooting the host is not an allowed recovery path.

## 3. Which acceptance conditions were actually met

| ID | Status | Evidence |
|---|---|---|
| C1 | **Partial** | `runs/dev-audit/env.json` shows `/envs/sire` and GPU UUID `d795...`; every real torch process reported exactly one visible device with that UUID. The 5 s monitor had a bug at first (it logged the first `nvidia-smi` row, i.e. GPU 0); this was found and fixed to select the expected UUID. No formal-run monitor exists because the formal run never started. |
| C2 | **Met for the frozen data** | `data/dataset/SIRE/prepared/splits.json` and `split_hashes.json`: requested sizes 4/8/8/8/16 + 8 monitor, 48,013 dedup clusters, 0 split ID/cluster collisions, 0 GSM8K overlaps by SimHash/Jaccard. |
| C3 | **Partial** | Automated small-network checks 7/7 pass; real-weight checks 12/13 passed before the device failed, including exact quota, `rho=1` bitwise, non-target R and no-drift. The one tolerance fix was not re-run. |
| C4 | **Not met** | Search started but no candidate completed after the device failure. |
| C5 | **Not met** | Test split was never opened; `freeze.json` does not exist. |
| C6 | **Not met** | Formal 7200 s ledger does not exist. |
| C7 | **Partial** | Single-command launcher, resumable state, stage ledger, dependency snapshot and correctness tests exist; the interrupt-and-resume test was not performed on the formal run. |
| C8 | **Not met** | No complete run, therefore no root `RESULTS.md` / `FORMULA.txt` / `curve.csv` / `curve.png`. None was fabricated. |

## 4. Minimum input needed to continue

One of:

1. physical GPU 4 restored (host-level GPU reset or reboot), with the same UUID
   `GPU-d7952610-13af-406a-e102-716a3f90ad66`; or
2. a documented change to the fixed-hardware constraint permitting another physical
   GPU (the contract currently forbids switching).

When GPU 4 is visible again, resume/start with:

```bash
cd /home/nxq/SIRE
nvidia-smi --query-gpu=index,uuid --format=csv,noheader   # GPU 4 must be listed
bash scripts/run_gpu4.sh run --config configs/v0.yaml --gpu-budget-seconds 7200
# after a clean stage-boundary interrupt:
bash scripts/run_gpu4.sh resume --run-id RUN_ID
bash scripts/run_gpu4.sh verify --run-id RUN_ID
```

The formal path re-runs `prepare-data` with the frozen default sizes, uses the pilot to
apply the one permitted reduction to search/selection/test = 4/4/8 if the projection
exceeds 88% of the remaining budget, and records the decision in `pilot.json` and
`frozen_config.yaml` before any formal evaluation.

## 5. Honest limitations already visible

- Only 4 pilot items and 8 calibration items were exercised; this is a v0 feasibility
  run, not a capability claim.
- Direct Hugging Face access was unavailable; data came from ModelScope mirrors and
  was hash-pinned.
- The reference base is Qwen2.5-7B while R1-Distill-Qwen-7B is documented as distilled
  from Qwen2.5-Math-7B; Δ is against this chosen reference, not purely distillation.
- Sparse Δ is not a sparse model and is not an inference speedup.
