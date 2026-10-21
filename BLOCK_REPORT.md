# BLOCK REPORT — SIRE v0

Status: **blocked by repeated hardware loss of physical GPU 4. The formal run
advanced to the pilot and then lost the device again. No acceptance results.**
Last update: 2026-10-21 (UTC).

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

## 3. Which acceptance conditions were actually met

| ID | Status | Evidence |
|---|---|---|
| C1 | **Met for the formal run up to failure** | `runs/v0-20261021-formal/env.json`: `sire` env, one visible device with UUID `d795...`; `gpu_monitor.jsonl` samples the correct UUID and task PID at 5 s intervals; `processes.json` records the resume PIDs. |
| C2 | **Met** | `data/dataset/SIRE/prepared/splits.json` and `split_hashes.json`: requested sizes 4/8/8/8/16 + 8 monitor, 48,013 dedup clusters, 0 ID/cluster collisions, 0 GSM8K overlaps. |
| C3 | **Met** | `runs/v0-20261021-formal/correctness/results.json`: 14/14 checks, including exact quota, `rho=1` bitwise, non-target R, no drift, SafeDiv, tie-breaking, split isolation, resume. |
| C4 | **Not met** | Search never ran in the formal run; the device failed at the pilot. |
| C5 | **Not met** | Test split never opened; `freeze.json` does not exist. |
| C6 | **Partial** | Formal ledger exists (809.5 s / 7200 s), monitor logs exist, but the run did not complete; `run_state.json` is `failed`. No budget exhaustion. |
| C7 | **Partial** | Launcher, resumable state, dependency snapshot and the interrupt/resume test exist (`resume_test.json`); the run could not proceed to completion. `reports/REPORT_EN.md` does not exist because no complete run exists. |
| C8 | **Not met** | No complete run, therefore no root `RESULTS.md` / `FORMULA.txt` / `curve.csv` / `curve.png`. None was fabricated. |

## 4. Minimum input needed to continue

1. A stable physical GPU 4 (same UUID), verified by a burn-in test that includes
   repeated 13 GB weight swaps and long batched generation; or
2. a documented change to the fixed-hardware constraint permitting another physical
   GPU.

When GPU 4 is visible again, the formal run can resume without redoing features or
correctness:

```bash
cd /home/nxq/SIRE
nvidia-smi --query-gpu=index,uuid --format=csv,noheader   # require GPU 4 = d795...
bash scripts/run_gpu4.sh resume --run-id v0-20261021-formal
bash scripts/run_gpu4.sh verify --run-id v0-20261021-formal
```

`resume` skips `audit`, `data`, `features` and `correctness`, then reruns `pilot`,
`search`, `select`, `test`, `report` and `verify`. The budget ledger continues from
809.5 s, not from zero.

## 5. Honest limitations already visible

- The pilot and all formal results are unmeasured; the only real generation evidence
  is the 4-item development pilot (4/4 correct, 5301 new tokens, 58 s, no truncation)
  from the previous day.
- Data are NuminaMath-CoT cn_k12 numeric answers from ModelScope mirrors, not a
  contamination-free benchmark.
- The reference base is Qwen2.5-7B while R1-Distill-Qwen-7B is documented as distilled
  from Qwen2.5-Math-7B; Δ is against this chosen reference, not purely distillation.
- Sparse Δ is not a sparse model and is not an inference speedup.
