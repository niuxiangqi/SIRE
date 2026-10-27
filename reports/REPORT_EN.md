# SIRE v0 — English report

Run: `v0-20261021-formal`

## What was measured

A reasoning model R (DeepSeek-R1-Distill-Qwen-7B) and a reference base B (Qwen2.5-7B) define D = θ_R − θ_B on seven projection matrices per layer. Each formula keeps the top floor(rho·N) entries of every target tensor and reconstructs θ = where(mask, R, B). Accuracy is measured by real generation on a held-out NuminaMath-CoT cn_k12 numeric split, under one frozen decoding budget.

Frozen formula: `s = (A * U)` (complexity 2).

## Test accuracy

| method | rho=0.01 | rho=0.05 | rho=0.1 | rho=0.2 | rho=0.3 | rho=1 |
|---|---|---|---|---|---|---|
| full R | - | - | - | - | - | 3/8 |
| random | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | - |
| A | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | - |
| Neg(U) | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | - |
| searched | 0/8 | 1/8 | 0/8 | 0/8 | 2/8 | - |

Full R accuracy: 0.375.

Full R truncation: 6/8 outputs hit the 2048-token cap (mean 1777 new tokens). All accuracies are accuracy under this output budget, not unbounded capability.

Searched formula best: 2/8 at rho=0.3. rho* is n/a: no tested keep-rate reached 0.90 x full-R accuracy (0.338).

Curve reduction: dropped rho=0.02, dropped rho=0.50, before their results were seen, because the projected test cost exceeded the remaining GPU budget. Tested points: 0.01, 0.05, 0.1, 0.2, 0.3, 1.

Acceptance: `verify.json` reports **8/8 C1–C8 checks passed** (passed=True). GPU ledger: 7006.399 s used of 7200.0 s (+300 s grace).

## Interpretation limits

- Sparse Δ reduces retained task-vector parameters only; no neuron, layer, FLOP or latency reduction is claimed and no speedup is claimed.
- The officially documented base of R1-Distill-Qwen-7B is Qwen2.5-Math-7B. D against Qwen2.5-7B is a difference against this chosen reference, not purely a distillation effect.
- Test item counts are small (8 items here); every accuracy has a wide binomial uncertainty; no significance test is claimed. A 1/8 vs 2/8 difference is not evidence of a reliable ranking.
- The full-R baseline itself was heavily truncated (see above), so the 90% threshold is computed against a low, output-budget-limited reference.
- Generation is capped; accuracy is accuracy under this output budget.
- Search is a single seed; no random-search control was run, so no claim is made that evolution beats random search.
- Data are NuminaMath-CoT cn_k12 with numeric answers only; contamination is not claimed to be zero. Only GSM8K overlap was checked locally.
- The formal run used physical GPU 2 after GPU 4 failed twice at the PCIe level; the migration is recorded in `device_migration.json` and did not change model, data, precision or decoding.
