# Post-hoc external tests: GSM8K and MMLU

These are **post-hoc** evaluations of the frozen v0 formula on external benchmarks. They were not used to drive search/selection and are not part of C1-C8. The pre-registered v0 test remains the cn_k12 split.

GSM8K uses the test split, 8 deterministic items, pass@4 (4 samples/item, temperature 0.6, top_p 0.95, 2048-token cap). MMLU uses 4 medical subjects x 4 items = 16 items, pass@1. The sparse formula is the frozen `s = (A * U)`.

| dataset | method | rho | n | pass@k | pass@1 | truncation |
|---|---|---|---|---|---|---|
| gsm8k | full_R | 1 | 8 | 1.000 | 0.375 | 0.188 |
| gsm8k | low_grad_NegU | 0.1 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | low_grad_NegU | 0.3 | 8 | 0.000 | 0.000 | 0.875 |
| gsm8k | magnitude_A | 0.1 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | magnitude_A | 0.3 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | random | 0.1 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | random | 0.3 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | searched | 0.05 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | searched | 0.1 | 8 | 0.000 | 0.000 | 1.000 |
| gsm8k | searched | 0.3 | 8 | 0.750 | 0.250 | 0.469 |
| mmlu | full_R | 1 | 16 | 0.500 | 0.500 | 0.312 |
| mmlu | low_grad_NegU | 0.1 | 16 | 0.125 | 0.125 | 1.000 |
| mmlu | low_grad_NegU | 0.3 | 16 | 0.000 | 0.000 | 1.000 |
| mmlu | magnitude_A | 0.1 | 16 | 0.125 | 0.125 | 1.000 |
| mmlu | magnitude_A | 0.3 | 16 | 0.312 | 0.312 | 0.938 |
| mmlu | random | 0.1 | 16 | 0.000 | 0.000 | 1.000 |
| mmlu | random | 0.3 | 16 | 0.000 | 0.000 | 1.000 |
| mmlu | searched | 0.05 | 16 | 0.000 | 0.000 | 1.000 |
| mmlu | searched | 0.1 | 16 | 0.000 | 0.000 | 1.000 |
| mmlu | searched | 0.3 | 16 | 0.188 | 0.188 | 0.562 |

## Reading

- Full R GSM8K pass@4 = 1.000; the searched formula at rho=0.30 = 0.750, i.e. 75% of full R, below the 90% criterion. At rho<=0.10 all methods are 0 with 100% truncation.
- Full R MMLU = 0.500; searched at rho=0.30 = 0.188 (37.5% of full R). Magnitude A at rho=0.30 = 0.312, so the searched formula is not the best sparse method on MMLU.
- random, magnitude A and Neg(U) are 0 on GSM8K at every tested rho; they are below full R on MMLU as well.
- 8 GSM8K items and 16 MMLU items are small; differences of one item are 12.5 and 6.25 points. These are debug-scale numbers, not significance tests.
- Heavy truncation (100% at low rho) means the comparison is output-budget-limited; 2048 tokens is too small for many R1-Distill traces.
- The external GPU ledger is separate from the formal 7200 s v0 budget and is recorded in external/budget.json. It is development/post-hoc time, not acceptance time.
- Pass@4 is computed as "any of 4 samples correct" over exactly 4 samples, not the unbiased pass@k estimator for n>k. This is the requested n=k=4 convention.

