# SIRE v0 — Sparse Reasoning Task Vectors

Search a compact symbolic score over a task vector `Δ = θ_R − θ_B`, keep only a
fraction `ρ` of its entries, and measure how much reasoning accuracy survives.

> **Status: formal v0 run completed on 2026-10-27 (physical GPU 2). Negative result.**
> C1–C8 pass (`runs/v0-20261021-formal/verify.json`). The frozen formula is
> `s = (A * U)`. On 8 held-out test items, full R scored 3/8 (0.375) and the
> searched formula peaked at 2/8 (rho=0.30); random, magnitude `A` and `Neg(U)`
> scored 0/8 at every tested keep-rate. No tested keep-rate reached 90% of full R,
> so `rho* = n/a` for every method. Full R itself hit the 2048-token cap on 6/8
> items, so all accuracies are output-budget-limited. See
> [`RESULTS.md`](RESULTS.md), [`reports/REPORT_EN.md`](reports/REPORT_EN.md) and
> [`BLOCK_REPORT.md`](BLOCK_REPORT.md) (the latter preserves the GPU-failure
> history from the earlier attempts).

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
BLOCK_REPORT.md          historical GPU 4 failure evidence and final resolution
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

One 80 GB-class NVIDIA GPU. The checked-in config currently targets **physical
GPU index 2** (the user-authorized replacement after GPU 4 failed at the PCIe
level) and verifies its UUID before importing torch:

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
# GPU 2 must be listed with the UUID in configs/v0.yaml
```

For a different machine, edit `device.physical_index` and
`device.expected_uuid` in `configs/v0.yaml`, and set `SIRE_GPU_INDEX` (or the
default in `scripts/run_gpu4.sh`) to the same index. Do not silently change the
card during a run.

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

- `runs/v0-20261021-formal/` — the completed formal run: audit, frozen splits,
  feature metadata, 14/14 correctness results, search candidates, freeze record,
  per-sample test predictions, budget ledger, GPU monitor, `verify.json` (C1–C8
  all pass).
- `RESULTS.md`, `FORMULA.txt`, `curve.csv`, `curve.png` — regenerated from the
  per-sample logs, not hand-written.
- `reports/REPORT_EN.md` — English report with the result and its limits.
- `runs/dev-audit/`, `runs/dev-features/`, `runs/dev-flow/` — development evidence.
  The 37 GB of `.npy` features stayed local and are git-ignored.
- `evidence/gpu4_failure_20261020/`, `evidence/gpu4_failure_20261021/`,
  `evidence/gpu2_switch_20261027/` — raw hardware-failure evidence. The run was
  completed on the user-authorized GPU 2 after GPU 4 failed twice at the PCIe
  level; the device migration is recorded in
  `runs/v0-20261021-formal/device_migration.json`.

The result is negative: no tested low keep-rate reached 90% of full-R accuracy.
The searched formula is the only low-rho method with non-zero accuracy, but 1/8–2/8
on eight items is not evidence of a reliable ranking. No positive claim is made.

## Post-hoc external tests (GSM8K / MMLU)

After the pre-registered cn_k12 run, the frozen formula `s = (A * U)` and the
baselines were evaluated on external benchmarks. These are **post-hoc**, not part
of C1–C8, and were not used to drive search/selection:

- **GSM8K** test split, 8 deterministic items, **pass@4** (4 samples/item,
  temperature 0.6, top_p 0.95, 2048-token cap).
- **MMLU** 4 medical subjects × 4 items = 16 items, pass@1.

| dataset | method | rho | pass@k | truncation |
|---|---|---|---|---|
| GSM8K | full R | 1 | **1.000** (pass@1 0.375) | 0.188 |
| GSM8K | searched `A*U` | 0.30 | 0.750 | 0.469 |
| GSM8K | searched `A*U` | 0.10 | 0.000 | 1.000 |
| GSM8K | random / A / Neg(U) | 0.10, 0.30 | 0.000 | 0.875–1.000 |
| MMLU | full R | 1 | **0.500** | 0.312 |
| MMLU | searched `A*U` | 0.30 | 0.188 | 0.562 |
| MMLU | magnitude A | 0.30 | 0.312 | 0.938 |
| MMLU | random / Neg(U) | 0.10, 0.30 | 0.000–0.125 | 1.000 |

At rho=0.30 the searched formula retains 75% of full-R GSM8K pass@4 (below the
90% criterion) but only 37.5% of full-R MMLU accuracy; magnitude `A` is better on
MMLU. At rho≤0.10 every method collapses with ~100% truncation. These are
debug-scale samples (8 and 16 items); one item is 12.5 or 6.25 points. Full
details: `runs/v0-20261021-formal/external/SUMMARY.md` and `summary.csv`.
The external GPU ledger is separate from the formal 7200 s v0 budget.

### Larger external budget (chosen: 4096 tokens, 32 GSM8K + 32 MMLU items)

The 2048-token cap was the v0 bottleneck: full R was often truncated. A probe at
`max_new_tokens=4096` and 32 items per benchmark gives a usable baseline:

| method | GSM8K pass@1 | GSM trunc | MMLU pass@1 | MMLU trunc |
|---|---|---|---|---|
| full R | **0.812** | 0.094 | **0.438** | 0.188 |
| searched `A*U`, rho=0.30 | 0.406 | 0.500 | 0.219 | 0.469 |
| magnitude A, rho=0.30 | 0.000 | 1.000 | 0.188 | 1.000 |
| random, rho=0.30 | 0.000 | 1.000 | 0.031 | 1.000 |
| Neg(U), rho=0.30 | 0.000 | 0.844 | 0.094 | 0.906 |

This is the selected evaluation budget. At rho=0.30 the frozen formula retains
50% of full-R accuracy on both benchmarks, below 90%, but beats random/A/Neg(U)
on GSM8K and random/Neg(U) on MMLU. Magnitude A is essentially tied on MMLU.

### Unbounded v1 search

A second search was run without the 7200 s cap, on GSM8K **train** (12 search + 16
selection items), objective pass@1 at rho=0.30, cap 4096. It evaluated 24 distinct
new mask signatures (30 candidate records). The seed `A*U` stayed best
(search J=0.917, selection J=0.625); no mutation improved it. The frozen formula
is therefore still `s = (A * U)`. Full details:
`runs/v1-gsm8k-search/SUMMARY.md`.

## Reproducibility and audit hooks

- Per-item generation seeds derive from `(global_seed, item_id)`; resume order
  cannot change sampling.
- **No sparse model checkpoints are written.** A candidate is reconstructed from
  pristine B/R, the formula, and the frozen feature cache; only the canonical AST,
  mask hashes and per-sample outputs are persisted. The optional 54 GB feature
  accumulator is disabled by default (`features.save_accumulator: false`) because
  recomputing eight calibration gradients is cheaper than writing/reading it.
- Search sees a manifest view containing only search/selection splits.
- `ρ=1` is checked bit-for-bit against pristine R through the `where` path.
- Tie-breaking is `(parameter name, flat index)`; SafeDiv is explicit.
- `verify` recomputes quotas, item counts, hashes, budget and curve aggregates,
  and regenerates `RESULTS.md`/`FORMULA.txt`/`curve.csv`/`curve.png` for a byte
  comparison. It does not read a human-written "passed" field.
- Dataset files, splits, prompts, generation config and seeds are hash-pinned.

## Limitations

- Small v0 samples: pilot 4, calibration 8, search/selection 4, test 8. The
  protocol permits one pre-formal reduction; this run used 4/4/8.
- The test curve has 5 keep-rates (0.01, 0.05, 0.10, 0.20, 0.30) plus full R.
  `rho=0.02` and `rho=0.50` were dropped in the frozen order before their results
  were seen, because the projected test cost exceeded the remaining GPU budget.
  Recorded in `runs/v0-20261021-formal/curve_reduction.json`.
- Full R hit the 2048-token cap on 6/8 test items; the searched formula also
  truncated heavily at low keep-rates. Accuracy is accuracy under this output
  budget, not unbounded capability.
- Numeric-answer filtering excludes proof/open/expression answers.
- Near-duplicate control is SimHash+LSH, not exhaustive pairwise comparison.
- Only GSM8K overlap was checked locally; MATH/AIME overlap is un-audited.
- No zero-contamination claim. No statistical significance claim; 1/8 vs 2/8 is
  not a reliable ranking.
- Search is a single seed; no random-search control was run.
- Sparse `Δ` is not a sparse model and is not an inference speedup.
- The formal run used physical GPU 2 after GPU 4 failed twice at the PCIe level;
  model, data, precision and decoding were unchanged, and the migration is
  recorded per run.

## License and contact

No license file is included in this repository; add one before redistribution.
Issues and pull requests are welcome at
`https://github.com/niuxiangqi/SIRE`.
