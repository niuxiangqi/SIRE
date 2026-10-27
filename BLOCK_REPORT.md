# BLOCK REPORT — SIRE v0 (historical)

> **RESOLVED 2026-10-27.** The formal run was moved to the user-authorized
> physical GPU 2 and completed. C1–C8 pass. See `RESULTS.md`,
> `reports/REPORT_EN.md` and `runs/v0-20261021-formal/verify.json`. This file is
> kept as the raw record of the GPU 4/PCIe failures and the driver-wedge period.

Status at the time of the last failure: **blocked by host CUDA driver state after
repeated GPU 4 PCIe loss; the user-authorized GPU 2 switch was configured but
could not execute until the host driver was reloaded. No acceptance results then.**
Last failure update: 2026-10-27 (UTC); resolved same day.

This is a block report, not an acceptance report. The held-out test split has
never been opened, so no accuracy or `ρ*` number is reported as a measured result.

## 1. Formal run progress before the second failure

Formal run id: `v0-20261021-formal`.

Using physical GPU 4 (UUID `GPU-d7952610-13af-406a-e102-716a3f90ad66`), after the
host recovered the card overnight:

1. `audit` — completed (6.9 s). 339/339 parameter tensors, no shape/dtype mismatch;
   tokenizer vocabulary identical; `rope_theta` and `sliding_window` differ.
2. `data` — completed (43.1 s). Frozen pilot/calibration/search/selection/test =
   4/8/8/8/16 + 8 monitor; 48,013 dedup clusters; no split leakage.
3. `features` — completed after a genuine interrupt/resume. The first process was
   stopped by SIGTERM during gradient sample 5; on resume it loaded the 4-sample
   accumulator and continued from sample 5/8. Final: 8/8 samples, 37 GB of FP16
   feature files, 4.13 s of CUDA-event compute, `U` vs `|mean g|` max relative
   difference 0.473.
4. `correctness` — completed, **14/14 passed**, including exact quota, `rho=1`
   bitwise through the `where` path, non-target parameters equal R, no drift after
   mask switches, SafeDiv boundary behavior, deterministic tie-breaking, split
   isolation and resume-without-recomputation.
5. `pilot` — started, applied full-R weights and began real generation. After a few
   seconds of generation the CUDA context failed again in `_sample_rows` with
   `CUDA error: unknown error`; the GPU then disappeared from `nvidia-smi` as
   `0000:1A:00.0: Unknown Error`. The stage never wrote `pilot.json`.

GPU ledger at failure: **809.5 s used of 7200 s** (6390.5 s remaining), including
two correctness runs (the first was interrupted by a separate SIGTERM during the
tool session and was rerun). The 8-item pilot and the entire search/selection/test
work were not completed.

## 2. Raw evidence

### 2026-10-21 failure (formal run)

`evidence/gpu4_failure_20261021/`:

- `evidence.txt` — `nvidia-smi`, `lspci`, sysfs link state.
- `last_error.json` — full traceback from `stage_pilot`.
- `budget.json` — formal ledger at failure (809.5 s used).
- `gpu_monitor.jsonl` — monitor samples: correct UUID, 42.7 GB used, 84–91% util
  immediately before the failure.

Key lines:

```
torch.AcceleratorError: CUDA error: unknown error
  at eval._sample_rows (keep[0] = True), inside stage_pilot
Unable to determine the device handle for GPU4: 0000:1A:00.0: Unknown Error
1a:00.0 3D controller: NVIDIA Corporation Device 20f5 (rev ff)
```

### 2026-10-20 failure (development search)

`evidence/gpu4_failure_20261020/` contains the first occurrence, at the first real
search candidate after mask application. Same signature: CUDA unknown error in
`_sample_rows`, then `0000:1A:00.0: Unknown Error`, `rev ff`.

### Recovery attempts

- `nvidia-smi --gpu-reset -i 4`, `nvidia-smi -r -i 4`, and reset by UUID all fail
  with `Unknown Error`.
- `sudo` is unavailable; sysfs unbind/rebind is root-owned; `dmesg` is not readable,
  so the Xid code cannot be captured from this account.
- A host-level GPU reset/reboot restored the card once (it worked on 2026-10-21
  morning) but the same failure recurred under load.
- Other GPUs (3, 5, 6, 7) are occupied by other users. Per the contract, switching
  cards, killing other users' processes, or rebooting the host is not allowed.

The repetition at the same point (generation after a sequence of mask applications)
is consistent with an unstable GPU 4 PCIe link rather than a one-off software error.
It should be treated as a hardware block until a burn-in test on GPU 4 passes.

## 2b. 2026-10-27: user-authorized switch to GPU 2

The user explicitly authorized using physical GPU 2 (UUID
`GPU-8a98ac52-5345-8b8d-93f5-b2f4f91decd7`) instead of the failed GPU 4. The
configuration and launcher were updated:

- `configs/v0.yaml`: `device.physical_index: 2`,
  `device.expected_uuid: GPU-8a98ac52-...`
- `scripts/run_gpu4.sh`: default index 2, overridable with `SIRE_GPU_INDEX`
- `verify` C1 accepts the recorded per-run device migration
  (`runs/<run_id>/device_migration.json`)

GPU 2 is visible and healthy in `nvidia-smi` (0 MiB, P0), but **no new CUDA
context can be created** on it either. A follow-up check shows the same failure on
the other free cards, GPU 0 and GPU 1, so this is not a GPU 2 fault:

```
$ for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i python -c "import torch; torch.cuda.init()"; done
RuntimeError: CUDA unknown error ... Setting the available devices to be zero.
```

The driver is still wedged by the failed GPU 4: `nvidia-smi -i 4` reports
`0000:1A:00.0: Unknown Error`, NVML initialization fails, and
`nvidia-smi --gpu-reset -i 2` returns `Insufficient Permissions` (no sudo).
This is a host-level driver state, not a GPU 2 fault and not a code error.
Evidence: `evidence/gpu2_switch_20261027/evidence.txt`.

Minimum input: reload the NVIDIA kernel modules or reboot the host (root), after
which GPU 2 should be usable. No formal GPU time was consumed by this switch
(ledger remains 809.5 s / 7200 s).

## 3. Which acceptance conditions were actually met

| ID | Status | Evidence |
|---|---|---|
| C1 | **Met** | `runs/v0-20261021-formal/env.json`, `gpu_monitor.jsonl` (correct target UUIDs before/after migration, 5 s samples), `processes.json`, `device_migration.json`; the user-authorized move to GPU 2 is explicit. |
| C2 | **Met** | `data/dataset/SIRE/prepared/splits.json` and `split_hashes.json`: frozen sizes 4/8/4/4/8 + 8 monitor, 48,012 dedup clusters, 0 ID/cluster collisions, 0 GSM8K overlaps. |
| C3 | **Met** | `runs/v0-20261021-formal/correctness/results.json`: 14/14 checks, including exact quota, `rho=1` bitwise, non-target R, no drift, SafeDiv, tie-breaking, split isolation, resume. |
| C4 | **Met** | `candidates.jsonl`: 8 distinct non-baseline mask signatures, 8 mutation children with parent links, per-sample records for each anchor. |
| C5 | **Met** | `freeze.json`, 21 test rows (5 keep-rates + full R), per-sample records, `curve.csv`; `rho=0.02` and `0.50` dropped before results were seen and recorded in `curve_reduction.json`. |
| C6 | **Met** | `budget.json`: 7006.4 s used of 7200 s (+300 s grace), monitor logs, `run_state.json` complete. |
| C7 | **Met** | Launcher, resumable state, dependency snapshot, interrupt/resume evidence (`resume_test.json`), `reports/REPORT_EN.md`, passing `verify`. |
| C8 | **Met** | Root `RESULTS.md`, `FORMULA.txt`, `curve.csv`, `curve.png` regenerate byte-identically from `runs/v0-20261021-formal/`; `verify` C8 is true. |

Final verdict: `runs/v0-20261021-formal/verify.json` reports **8/8 checks passed**.
The scientific result is negative: no tested keep-rate reached 90% of full-R
accuracy, and full R itself was truncated on 6/8 test items.

## 4. Historical minimum input needed to continue

1. Reload the NVIDIA driver / reboot the host so that a new CUDA context can be
   created on the user-authorized physical GPU 2 (UUID
   `GPU-8a98ac52-5345-8b8d-93f5-b2f4f91decd7`); or
2. a further documented change to the hardware constraint.

After the driver is healthy, resume without redoing features or correctness:

```bash
cd /home/nxq/SIRE
nvidia-smi --query-gpu=index,uuid --format=csv,noheader   # GPU 2 must be usable
CUDA_VISIBLE_DEVICES=2 python -c "import torch; torch.cuda.init(); print('cuda ok')"
bash scripts/run_gpu4.sh resume --run-id v0-20261021-formal
bash scripts/run_gpu4.sh verify --run-id v0-20261021-formal
```

`resume` records the GPU 4 -> GPU 2 migration, skips `audit`, `data`, `features`
and `correctness`, then reruns `pilot`, `search`, `select`, `test`, `report` and
`verify`. The budget ledger continues from 809.5 s, not from zero.

## 5. Honest limitations already visible

- The pilot and all formal results are unmeasured; the only real generation evidence
  is the 4-item development pilot (4/4 correct, 5301 new tokens, 58 s, no truncation)
  from the previous day.
- Data are NuminaMath-CoT cn_k12 numeric answers from ModelScope mirrors, not a
  contamination-free benchmark.
- The reference base is Qwen2.5-7B while R1-Distill-Qwen-7B is documented as distilled
  from Qwen2.5-Math-7B; Δ is against this chosen reference, not purely distillation.
- Sparse Δ is not a sparse model and is not an inference speedup.
