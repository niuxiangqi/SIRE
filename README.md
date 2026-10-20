# SIRE v0 — Sparse Reasoning Task Vectors

Search a compact symbolic score over a task vector `Δ = θ_R − θ_B`, keep only a
fraction `ρ` of its entries, and measure how much reasoning accuracy survives.

> **Status: blocked, no acceptance results.**
> The implementation, frozen data pipeline, model audit, feature extraction and a
> 4-item real-7B pilot are complete and were exercised on real weights. The formal
> run stopped when **physical GPU 4 disappeared from the PCIe bus**
> (`0000:1A:00.0: Unknown Error`). The held-out test split was never opened.
> See [`BLOCK_REPORT.md`](BLOCK_REPORT.md) and [`HANDOFF.md`](HANDOFF.md).

This repository is the full, auditable implementation of the v0 protocol in
[`GOAL.md`](GOAL.md). It intentionally contains **no model weights and no raw
datasets**; both are fetched separately. No numbers in this repository are
fabricated results.

## The question

Under one fixed protocol, can an automatically searched, short symbolic score
`s = f(base symbols)` retain more reasoning accuracy at low keep-rates than
magnitude, low-gradient, and random retention?

A negative result is acceptable. An unaudited positive result is not.

Scope: sparsifying `Δ` reduces the number of **retained task-vector parameters**.
It does **not** reduce neurons, layers, FLOPs or latency. No speedup is claimed.

## Protocol in one page

- **Base symbols (exactly six):** `D = θ_R − θ_B`, `A = |D|`,
  `G = mean_z g(z)`, `U = mean_z |g(z)|`, `B = θ_B`, `R = θ_R`.
- **Operators (exactly six):** `Neg`, `Abs`, `Add`, `Sub`, `Mul`, `SafeDiv`
  (`SafeDiv(x,y) = x / (sign(y)·max(|y|,1e-6))`, `sign(0)=+1`).
- **Formulas:** typed trees, depth ≤ 3, ≤ 4 operator nodes, per-tensor positive
  RMS normalization, canonical AST + mask-hash dedup.
- **Target set S:** `q/k/v/o/gate/up/down` projection weights in every layer;
  every other parameter stays at `R`. Per tensor, keep exactly
  `floor(ρN)` entries; reconstruct with `where(mask, θ_R, θ_B)`.
- **Metric:**
  `ρ*(f) = min { ρ : accuracy(f, ρ) ≥ 0.90 × accuracy(full R) }`, lower is better.
  No interpolation between tested points.
- **Baselines:** full R, random retention, magnitude `A`, low-gradient `Neg(U)`.
- **Search objective:** mean accuracy at `ρ ∈ {0.05, 0.20}` on the search split.
- **Held-out evaluation:** one frozen formula, full curve on the test split.

## Repository layout

```text
GOAL.md                  execution contract and acceptance conditions
BLOCK_REPORT.md          why the formal run stopped (GPU 4 hardware loss)
PLAN.md                  milestones and verification commands
DECISIONS.md             consequential choices and evidence
HANDOFF.md               step-by-step retry guide for a fresh clone
RUNBOOK.md               commands, environment, artifacts
README.md                this file

configs/v0.yaml          frozen protocol (edit paths for your machine)
src/sire/
  audit.py               B/R architecture, tokenizer and hash audit
  data.py                NuminaMath-CoT filtering, dedup, frozen splits
  answers.py             safe answer extraction/comparison (no code execution)
  dsl.py                 leaf/operator DSL, canonicalization, complexity
  features.py            D/A/G/U/B/R feature collection (FP32 statistics)
  masking.py             target specs, exact-quota masks, in-place reconstruction
  eval.py                batched generation with per-item seeds, records
  search.py              single-seed evolutionary search, archive, freeze
  correctness.py         small-network and real-weight checks
  report.py              root artifacts regenerated from logs
  verify.py              C1–C8 acceptance verification
  cli.py                 subcommands and resumable stage orchestration
scripts/
  run_gpu4.sh            launcher: GPU-4 UUID resolution before torch import
  run_tests.py           CPU small-network checks
  fetch_data.py          fetch NuminaMath-CoT / no_robots into the data root
  fetch_models.py        fetch B/R checkpoints (optional helper)
tests/                   (add project-specific tests here)
runs/                    per-run evidence (feature arrays are git-ignored)
```

## Quick start

### 1. Hardware

One 80 GB-class NVIDIA GPU. The protocol and launcher are written for **physical
GPU index 4** and verify its UUID before importing torch:

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
# GPU 4 must be listed; the expected UUID is in configs/v0.yaml
```

If your GPU index is different, edit `device.physical_index` in
`configs/v0.yaml` and the `awk` index in `scripts/run_gpu4.sh`. Do not silently
change the card during a run.

### 2. Models (not included)

The audited checkpoints are:

| Role | Model | Expected path in `configs/v0.yaml` |
|---|---|---|
| B (reference base) | `Qwen/Qwen2.5-7B` | `/home/nxq/data/KEPT/models/Qwen2.5-7B` |
| R (reasoning) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `/home/nxq/data/KEPT/models/DeepSeek-R1-Distill-Qwen-7B` |

Either download them with your usual Hugging Face/ModelScope workflow, or use:

```bash
python scripts/fetch_models.py --config configs/v0.yaml --source modelscope
```

R1-Distill-Qwen-7B is officially distilled from Qwen2.5-Math-7B. `Δ` against
Qwen2.5-7B is therefore a difference against **this chosen reference**, not purely
a distillation effect. All runs use R's config, tokenizer and chat template.

### 3. Data (not included)

- `AI-MO/NuminaMath-CoT`, `source=cn_k12` subset (numeric-answer items only)
- `HuggingFaceH4/no_robots` (general-ability monitor, 8 single-turn samples)

```bash
python scripts/fetch_data.py --config configs/v0.yaml --source modelscope
```

This populates `data_root/raw/NuminaMath-CoT` and `data_root/raw/no_robots`.
The preparation stage filters, deduplicates and freezes splits; hashes are stored
in `prepared/splits.json`.

### 4. Environment

The exercised environment is Python 3.12 + torch 2.13 (CUDA 13) +
transformers 5.17. A conda clone is convenient if you already have a compatible
torch stack:

```bash
conda create -n sire python=3.12 -y
conda activate sire
# install the CUDA build appropriate for your driver, then:
pip install -r requirements.txt
# optional: exact versions exercised during development
# pip install -r requirements.lock.txt
```

Miniconda path in `scripts/run_gpu4.sh` is `/home/nxq/miniconda3`; edit it if
needed. The launcher exports `CUDA_VISIBLE_DEVICES` **before** Python imports
torch.

### 5. Run

```bash
cd SIRE

# CPU: filter, dedup, freeze splits
bash scripts/run_gpu4.sh prepare-data --config configs/v0.yaml

# One command: audit -> data -> features -> correctness -> pilot -> search
#              -> select/freeze -> test -> report -> verify
bash scripts/run_gpu4.sh run --config configs/v0.yaml --gpu-budget-seconds 7200

# After a clean interrupt:
bash scripts/run_gpu4.sh resume --run-id RUN_ID

# Acceptance + artifacts (also run automatically at the end of `run`)
bash scripts/run_gpu4.sh verify --run-id RUN_ID
bash scripts/run_gpu4.sh report --run-id RUN_ID
```

Individual stages (`collect-features`, `baselines`, `search`, `select`, `test`,
`report`, `verify`) accept `--run-id` and are resumable from
`runs/<run_id>/run_state.json`.

## Outputs

After a completed run, the root contains artifacts regenerated **only from
on-disk logs**:

- `RESULTS.md` — headline `ρ*` table plus one section per run
- `FORMULA.txt` — frozen formula, canonical AST, mask hash, complexity
- `curve.csv` — `method,formula_id,rho,n_items,n_correct,accuracy,rel_to_full_R,mean_new_tokens,truncation_rate,nll_delta,seconds`
- `curve.png` — accuracy vs `ρ` (log x), full-R line, 90% band
- `runs/<run_id>/` — per-sample predictions, candidates and parent links, mask
  audits, budget ledger, GPU monitor, correctness results, freeze record

Feature arrays (`runs/*/features/*.npy`) are large derived data and are
git-ignored. They are regenerated by `collect-features` (about 15 minutes wall on
the reference machine, dominated by the one-time 54 GB gradient checkpoint).

## What is in this repository now

- `runs/dev-audit/` — real B/R audit (339/339 tensors, no shape/dtype mismatch;
  tokenizer vocab identical; config differences recorded).
- `runs/dev-features/` — feature metadata and small logs for 8 calibration
  samples. The 37 GB of `.npy` features stayed local and are git-ignored.
- `runs/dev-flow/` — partial search attempt and the raw CUDA failure traceback.
- `evidence/gpu4_failure_20261020/` — `nvidia-smi`/`lspci`/sysfs evidence of the
  GPU 4 PCIe loss.

These are **development evidence, not acceptance results**. The test split was
never opened and no `RESULTS.md`/`curve.csv` was fabricated.

## Reproducibility and audit hooks

- Per-item generation seeds derive from `(global_seed, item_id)`; resume order
  cannot change sampling.
- Search sees a manifest view containing only search/selection splits.
- `ρ=1` is checked bit-for-bit against pristine R through the `where` path.
- Tie-breaking is `(parameter name, flat index)`; SafeDiv is explicit.
- `verify` recomputes quotas, item counts, hashes, budget and curve aggregates,
  and regenerates `RESULTS.md`/`FORMULA.txt`/`curve.csv`/`curve.png` for a byte
  comparison. It does not read a human-written "passed" field.
- Dataset files, splits, prompts, generation config and seeds are hash-pinned.

## Limitations

- Small v0 samples (pilot 4, calibration 8, search/selection 8, test 16; the
  protocol permits one reduction to 4/4/8 if the pilot projects over budget).
- Accuracy is accuracy under the frozen output cap, not unbounded capability.
- Numeric-answer filtering excludes proof/open/expression answers.
- Near-duplicate control is SimHash+LSH, not exhaustive pairwise comparison.
- Only GSM8K overlap was checked locally; MATH/AIME overlap is un-audited.
- No zero-contamination claim. No statistical significance claim.
- Search is a single seed; no random-search control was run.
- Sparse `Δ` is not a sparse model and is not an inference speedup.

## License and contact

No license file is included in this repository; add one before redistribution.
Issues and pull requests are welcome at
`https://github.com/niuxiangqi/SIRE`.
