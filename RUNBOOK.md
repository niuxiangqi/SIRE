# RUNBOOK — SIRE v0

## Environment

- Miniconda: `/home/nxq/miniconda3`, env `sire` (Python 3.12, torch 2.13 + cu130,
  transformers 5.17, datasets, accelerate, pandas, matplotlib, safetensors, modelscope).
- Source: `/home/nxq/SIRE/src` (install not required; launcher sets `PYTHONPATH`).
- GPU: physical index 2 by default (user-authorized replacement after GPU 4 failed
  twice at the PCIe level). `scripts/run_gpu4.sh` resolves the configured UUID
  (override index with `SIRE_GPU_INDEX`) and exports `CUDA_VISIBLE_DEVICES` before
  Python starts; Python re-verifies the UUID and that exactly one device is visible.

## Commands

```bash
cd /home/nxq/SIRE

# CPU data preparation (writes /home/nxq/data/dataset/SIRE/prepared/splits.json)
bash scripts/run_gpu4.sh prepare-data --config configs/v0.yaml

# Model audit (CPU + tokenizer; writes runs/<id>/audit and env.json)
bash scripts/run_gpu4.sh audit --config configs/v0.yaml --run-id RUN_ID

# Feature collection (D/A/G/U/B/R; writes runs/<id>/features)
bash scripts/run_gpu4.sh collect-features --config configs/v0.yaml --run-id RUN_ID

# Formal run: audit -> data -> features -> correctness -> pilot -> search -> select
#              -> test -> report -> verify
bash scripts/run_gpu4.sh run --config configs/v0.yaml --gpu-budget-seconds 7200

# Resume from run_state.json (never recomputes finished stages/samples)
bash scripts/run_gpu4.sh resume --run-id RUN_ID
bash scripts/run_gpu4.sh verify --run-id RUN_ID
bash scripts/run_gpu4.sh report --run-id RUN_ID
```

Individual stages: `collect-features`, `baselines`, `search`, `select`, `test`,
`report`, `verify` all accept `--run-id`.

## State and evidence

- `PLAN.md`, `DECISIONS.md`, `HANDOFF.md`, `BLOCK_REPORT.md`.
- `runs/<run_id>/run_state.json` — stage status and resume command.
- `runs/<run_id>/budget.json` — per-stage wall / CUDA-event / GPU-work ledger.
- `runs/<run_id>/gpu_monitor.jsonl` — 5 s samples of the target UUID and PID.
- `runs/<run_id>/candidates.jsonl` — AST, parent link, anchors, mask hashes.
- `runs/<run_id>/evals/<method>__<id>__rho<rho>__<split>/samples.jsonl` — every
  generated sample with seed, text, correctness, tokens, truncation, time.
- `runs/<run_id>/freeze.json` — the single frozen formula before test opens.
- Root artifacts are regenerated only by `report` from `runs/`.

## Tests

```bash
/home/nxq/miniconda3/envs/sire/bin/python scripts/run_tests.py
```

Small-network checks are CPU-only. Real-weight checks run inside `run` after features
and before the pilot; they cover quota/boundaries, `rho=1` bitwise, non-target R,
no drift, SafeDiv, tie-breaking, split isolation, resume, and parse-failure handling.

## Status

The formal run `v0-20261021-formal` completed on 2026-10-27 on the user-authorized
physical GPU 2 after GPU 4 failed twice at the PCIe level. C1–C8 pass; the result is
negative (`rho*=n/a`, full R 3/8, searched formula best 2/8). See `RESULTS.md`,
`reports/REPORT_EN.md`, `runs/v0-20261021-formal/verify.json`, and the historical
`BLOCK_REPORT.md` for the hardware-failure evidence.
