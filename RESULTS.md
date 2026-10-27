# SIRE v0 results

Per-run sections below are regenerated from per-sample logs. Sparse Δ reduces retained task-vector parameters only; it does not reduce neurons, layers, FLOPs or latency, and no speedup is claimed.

## Headline: rho*(f) = min { rho : accuracy(f, rho) >= 0.90 * accuracy(full R) }

Lower is better. No interpolation between tested points.

Latest run: `v0-20261021-formal`, formula `s = (A * U)`.

| method | rho* | best accuracy |
|---|---|---|
| full R | 1 | 0.375 |
| random retention | n/a | 0.000 |
| magnitude A | n/a | 0.000 |
| low-gradient Neg(U) | n/a | 0.000 |
| searched formula | n/a | 0.250 |

## Run `v0-20261021-formal`

- Frozen formula: `s = (A * U)`
- Canonical AST: `(Mul A U)`
- Complexity: 2 (unique operator nodes + G/U reductions)
- Test items: 8; accuracy is accuracy under the frozen output budget, not unbounded capability.

| method | rho* | best acc | rho=0.01 | rho=0.05 | rho=0.1 | rho=0.2 | rho=0.3 | rho=1 |
|---|---|---|---|---|---|---|---|---|
| full R | 1 | 0.375 | — | — | — | — | — | 3/8 |
| random retention | n/a | 0.000 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | — |
| magnitude A | n/a | 0.000 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | — |
| low-gradient Neg(U) | n/a | 0.000 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | — |
| searched formula | n/a | 0.250 | 0/8 | 1/8 | 0/8 | 0/8 | 2/8 | — |

Mask hashes (formula): rho=0.0500:`694d6dc4e1931699`, rho=0.2000:`abc3b41c8c9e3e16`
- Full R truncation rate at rho=1: 0.750 (6/8 outputs hit the output cap).
- Tested keep-rates: 0.01, 0.05, 0.1, 0.2, 0.3, 1.
- Curve reduction: dropped rho=0.02, dropped rho=0.50, before their results were seen, because the projected test cost exceeded the remaining GPU budget.
