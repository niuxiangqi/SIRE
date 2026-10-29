# v1: evaluation-budget probe and unbounded symbolic search

Date: 2026-10-29. GPU: physical GPU 2 only, per instruction.
This is post-v0 exploratory work; it does not change the v0 C1–C8 acceptance record.

## 1. What is a useful test budget?

The bottleneck in v0 was `max_new_tokens=2048`: full R was truncated on 6/8
cn_k12 test items, and almost every sparse method produced 100% truncated output.
We probed a larger output budget on GSM8K test and MMLU medical subsets.

| budget | dataset | full R | full-R truncation | searched `A*U` ρ=0.30 | searched truncation |
|---|---|---|---|---|---|
| 2048 tokens, 8/16 items | GSM8K (pass@1) | 0.375 | 0.188 | 0.250 (pass@4 0.750) | 0.469 |
| 2048 tokens, 8/16 items | MMLU (pass@1) | 0.500 | 0.312 | 0.188 | 0.562 |
| **4096 tokens, 32/32 items** | **GSM8K (pass@1)** | **0.812** | **0.094** | **0.406** | **0.500** |
| **4096 tokens, 32/32 items** | **MMLU (pass@1)** | **0.438** | **0.188** | **0.219** | **0.469** |

At 4096 tokens and 32 items per benchmark, full R is high enough and truncation
is low enough for a meaningful comparison. At 2048 the full-R baseline is too
truncated for low-keep-rate work. Recommended test budget: **max_new_tokens=4096,
32 GSM8K test items, 32 MMLU items**, one sample per item.

### Complete comparison at the chosen 4096/32 budget (GSM8K pass@1, MMLU pass@1)

| method | GSM8K | GSM trunc | MMLU | MMLU trunc |
|---|---|---|---|---|
| full R | **0.812** | 0.094 | **0.438** | 0.188 |
| random retention ρ=0.30 | 0.000 | 1.000 | 0.031 | 1.000 |
| magnitude A ρ=0.30 | 0.000 | 1.000 | 0.188 | 1.000 |
| low-gradient Neg(U) ρ=0.30 | 0.000 | 0.844 | 0.094 | 0.906 |
| searched `A*U` ρ=0.30 | 0.406 | 0.500 | 0.219 | 0.469 |

At ρ=0.30 the searched formula retains 50% of full-R GSM8K accuracy and 50% of
full-R MMLU accuracy, below the 90% criterion, but it is clearly above the three
baselines on GSM8K and above random/Neg(U) on MMLU. Magnitude A is close on MMLU
(0.188 vs 0.219); with 32 items this is a 1-item difference.

## 2. Unbounded symbolic search

Search/selection uses **GSM8K train only** (12 search + 16 selection items), so
GSM8K test and MMLU stay held out. Objective: pass@1 at ρ=0.30, `max_new_tokens=4096`.
No 7200 s cap; 24 new mask signatures were evaluated (30 candidate records total:
4 seeds + 26 mutation records, including two mask-equivalent duplicates).

| item | value |
|---|---|
| best search J | 0.917 (seed `A*U`, 11/12) |
| next best | 0.500 (`A+U`, 6/12) |
| all other mutations | 0.000 |
| selection J (16 items) | 0.625 |
| frozen formula | `s = (A * U)` (seed, not a mutation) |
| search wall | 8218 s |
| selection wall | 545 s |

No mutation beat the seed. The search space around `A*U` mostly collapsed to 0,
which is itself a result: under this budget, the seed is a strong local optimum.
The GSM8K train split is easier than the test split (J 0.917 vs test 0.406), so
the search-to-test gap is large; this is a generalization/difficulty shift, not
leakage (train and test items are disjoint).

## 3. Final held-out check of the frozen v1 formula

The v1 search froze the same canonical formula as v0: `(Mul A U)`. The 4096/32
held-out rows above therefore apply unchanged: GSM8K 0.406, MMLU 0.219 at ρ=0.30,
versus full R 0.812/0.438. The formula fails the 90% criterion on both external
benchmarks. This is a negative result.

## 4. Files and caveats

- v1 candidates/freeze: `runs/v1-gsm8k-search/candidates.jsonl`, `freeze.json`,
  `v1_state.json`, `budget.json`, `gpu_monitor.jsonl`.
- external rows: `runs/v0-20261021-formal/external/*n32_cap4096*/summary.json`
  and `samples.jsonl`.
- MMLU uses 4 medical subjects (`anatomy`, `college_biology`, `college_medicine`,
  `professional_medicine`), 8 items each. The local MMLU files are test splits;
  search/selection use GSM8K train so MMLU is fully held out.
- GSM8K test items are disjoint from the GSM8K train search/selection items.
- 32-item samples still carry wide uncertainty (one item = 3.1 points); no
  significance test is claimed.
- The search budget is unbounded in the sense that no 7200 s formal cap was
  applied; it stopped at the pre-declared 24 new-mask candidate cap.
- Sparse Δ is not a sparse model; no speedup is claimed.
