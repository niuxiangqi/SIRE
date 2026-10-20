# HANDOFF — retrying the SIRE v0 experiment

This document is for someone who **clones this repository and wants to run the
experiment from scratch**. It assumes the protocol in [`GOAL.md`](GOAL.md) and the
decisions in [`DECISIONS.md`](DECISIONS.md).

> Current status: the formal run has **not** been completed. The reference machine
> lost physical GPU 4 (`0000:1A:00.0: Unknown Error`, `lspci ... rev ff`) during the
> first real search candidate. See [`BLOCK_REPORT.md`](BLOCK_REPORT.md). The
> `runs/dev-*` directories are development evidence, not results.

## 0. What you need

| Item | Requirement |
|---|---|
| GPU | one 80 GB-class NVIDIA GPU (reference: A800 80 GB PCIe) |
| Driver/CUDA | driver able to run the CUDA 13 build of torch used here |
| Disk | ~60 GB for the two 7B checkpoints, ~10 GB for data/artifacts, plus ~60 GB temporary during feature extraction |
| RAM | ≥ 256 GB recommended (reference: 1 TB; features are streamed, not all held) |
| Python | 3.12 (reference env: conda `sire`) |
| Network | Hugging Face or ModelScope access for models/data |

All commands below are run from the repository root.

## 1. Clone and create the environment

```bash
git clone https://github.com/niuxiangqi/SIRE.git
cd SIRE

conda create -n sire python=3.12 -y
conda activate sire
# Install the torch CUDA build that matches your driver first, then:
pip install -r requirements.txt
# optional exact dev environment (215 packages, CUDA 13 wheels):
# pip install -r requirements.lock.txt
```

The launcher hardcodes the Miniconda path and physical GPU index:

- `scripts/run_gpu4.sh`: change `/home/nxq/miniconda3`, the `awk` index `$1==4`,
  and the env name if your setup differs.
- `configs/v0.yaml`: change `device.physical_index` and
  `device.expected_uuid` if you use a different GPU.

**Do not silently change the GPU during a run.** The launcher resolves the UUID
and exports `CUDA_VISIBLE_DEVICES` before Python imports torch; `Ctx` re-verifies
that exactly one device is visible and that its UUID matches the config.

## 2. Put the models in place

The repository does **not** contain weights. Download:

| Role | Model | Config path |
|---|---|---|
| B | `Qwen/Qwen2.5-7B` | `paths.base_model` |
| R | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `paths.reasoning_model` |

Option A — helper (ModelScope, which was reachable from the reference host):

```bash
python scripts/fetch_models.py --config configs/v0.yaml --source modelscope
```

Option B — your existing Hugging Face cache/snapshot, then set the two absolute
paths in `configs/v0.yaml` to the snapshot directories.

Expected layout for the default config:

```text
/home/nxq/data/KEPT/models/Qwen2.5-7B/
/home/nxq/data/KEPT/models/DeepSeek-R1-Distill-Qwen-7B/
```

Sanity check — the audit must find 339 parameter tensors in both models with no
shape/dtype mismatch, and no same-token/different-id tokenizer mappings:

```bash
bash scripts/run_gpu4.sh audit --config configs/v0.yaml --run-id dev-audit
python - <<'PY'
import json
d=json.load(open('runs/dev-audit/audit/model_audit.json'))
print(d['parameter_names'], d['shape_mismatch'], d['dtype_mismatch'])
print(d['config_diff'])
PY
```

## 3. Fetch the datasets

The protocol uses:

- `AI-MO/NuminaMath-CoT`, `source=cn_k12` — numeric-answer items only
- `HuggingFaceH4/no_robots` — 8 single-turn monitor samples

```bash
python scripts/fetch_data.py --config configs/v0.yaml --source modelscope
```

This creates:

```text
/home/nxq/data/dataset/SIRE/raw/NuminaMath-CoT/
/home/nxq/data/dataset/SIRE/raw/no_robots/
```

If you prefer Hugging Face, use `--source hf`; if neither endpoint is reachable,
the protocol cannot be run as specified and synthetic data must **not** substitute
for acceptance evidence.

## 4. Freeze the data splits (CPU only)

```bash
bash scripts/run_gpu4.sh prepare-data --config configs/v0.yaml
```

Expected output: requested sizes `pilot 4 / calibration 8 / search 8 / selection 8 /
test 16 / monitor 8`, about 48k dedup clusters, zero duplicate IDs or clusters
across splits. The command writes:

- `data_root/prepared/splits.json` — items, prompts, ground truths, hashes, seeds,
  dataset file hashes, dedup and GSM8K cross-check audit
- `data_root/prepared/split_hashes.json`

Check:

```bash
python - <<'PY'
import json
m=json.load(open('/home/nxq/data/dataset/SIRE/prepared/splits.json'))
print(m['sizes']); print(m['dedup']); print(m['crosscheck_gsm8k'])
PY
```

If your data root differs, change `paths.data_root` and `paths.raw_data` in
`configs/v0.yaml` before this step.

## 5. Run the formal experiment

Start with the default sizes. The pilot measures real tokens/second and truncation;
if the projection exceeds 88% of the remaining budget, the protocol permits one
pre-formal reduction to `search/selection/test = 4/4/8` (and, if still needed,
dropping `rho=0.02` then `rho=0.50`). The decision is written to
`runs/<id>/pilot.json` and `frozen_config.yaml` **before** formal evaluation.

```bash
bash scripts/run_gpu4.sh run --config configs/v0.yaml --gpu-budget-seconds 7200
```

Stage order:

1. `audit` — model/tokenizer/hash audit, environment snapshot.
2. `data` — frozen splits.
3. `features` — D/A/G/U/B/R; FP32 R model, per-sample backward, FP32 accumulation.
   The first run takes about 15 min wall, mostly a 54 GB gradient checkpoint write;
   the ledger books only CUDA-event compute, not checkpoint I/O.
4. `correctness` — small-network checks, then real-weight quota/`rho=1`/
   non-target/no-drift/SafeDiv/tie-break checks.
5. `pilot` — full R on 4 items; measures throughput and truncation.
6. `search` — 4 seeds + ≥8 non-baseline formulas, two anchor `rho` values.
7. `select` — up to 2 formulas on the independent selection split; freezes one.
8. `test` — opens the held-out split only after `freeze.json` exists; measures full
   R, random, `A`, `Neg(U)` and the frozen formula on the active curve grid.
9. `report` — regenerates root `RESULTS.md`, `FORMULA.txt`, `curve.csv`,
   `curve.png`.
10. `verify` — C1–C8; exits non-zero on any failure.

## 6. Resume and acceptance

Every stage and every generated sample is checkpointed. A clean `SIGINT` exits with
code 130 and leaves `run_state.json`:

```bash
bash scripts/run_gpu4.sh resume --run-id RUN_ID
```

Resume never recomputes finished stages or samples. To test this deliberately
before a long run:

```bash
SIRE_INTERRUPT_AFTER=features bash scripts/run_gpu4.sh run \
  --config configs/v0.yaml --run-id resume-test --gpu-budget-seconds 7200
bash scripts/run_gpu4.sh resume --run-id resume-test
```

Acceptance:

```bash
bash scripts/run_gpu4.sh verify --run-id RUN_ID
bash scripts/run_gpu4.sh report --run-id RUN_ID
```

`verify` recomputes retained counts, item counts, hashes, budget and curve
aggregates from logs, re-derives the artifacts, and passes only if C1–C8 all hold.
It does not read a hand-written "passed" field.

## 7. What to check if something fails

| Symptom | Check |
|---|---|
| `physical GPU N not found` | `nvidia-smi` index/UUID; edit config + launcher; never switch cards mid-run |
| CUDA OOM in generation | lower `generation.batch_size`; features already use checkpointing; do not change model/device |
| `features/meta.json` missing | `collect-features` did not finish; re-run it; partial accumulators are checkpointed every 4 samples |
| calibration count changed | your split/data changed; re-run `prepare-data` and do not reuse old features |
| `formula is not frozen` | `test` refuses to open before `freeze.json`; run `select` first |
| `verify` C8 mismatch | do not edit root artifacts by hand; run `report` and re-check |
| `U` and `|G|` too similar | gradient statistics are wrong; re-run feature collection and inspect `u_vs_abs_g_max_relative_diff` |

## 8. Reference timings observed during development

These are development measurements, not formal results:

- Model load (bf16 R + pristine B/R copies): ~15–20 s.
- Feature extraction: ~15 min wall; 37 GB of FP16 feature files; ~6 s of
  CUDA-event compute for 8 samples (most wall time is gradient copy/accumulation
  and the 54 GB checkpoint).
- Mask construction and weight swap: ~30 s per `(formula/method, rho)` over 196
  tensors on the reference host.
- Generation: ~91 tokens/s for a batch of 4 with mean 1325 new tokens/item;
  ~260 tokens/s for a batch of 8 on a synthetic benchmark.
- Pilot: 4/4 correct, 5301 new tokens, 58 s, no truncation.

Budget the formal run from the pilot, not from these numbers.

## 9. Current blocker and what “done” means

The reference machine's GPU 4 failed at the PCIe level. Raw evidence is in
`evidence/gpu4_failure_20261020/`. On a healthy GPU 4, the next command is:

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader   # GPU 4 must be present
bash scripts/run_gpu4.sh run --config configs/v0.yaml --gpu-budget-seconds 7200
```

The experiment is complete only when C1–C8 pass and the root artifacts exist and
match regenerated logs. A running process, passing unit tests, a full budget
ledger, or one lucky candidate is **not** completion. Never edit ground truth,
loosen quotas, drop failing items, or fabricate timings to reach a passing state.
