# External tests at the 4096-token / 32-item budget

GSM8K test 32 items, pass@1; MMLU 4 medical subjects x 8 = 32 items, pass@1; max_new_tokens=4096.

| dataset | method | rho | pass@1 | truncation |
|---|---|---|---|---|
| gsm8k | full_R | 1 | 0.812 | 0.094 |
| gsm8k | low_grad_NegU | 0.3 | 0.000 | 0.844 |
| gsm8k | magnitude_A | 0.3 | 0.000 | 1.000 |
| gsm8k | random | 0.3 | 0.000 | 1.000 |
| gsm8k | searched | 0.3 | 0.406 | 0.500 |
| mmlu | full_R | 1 | 0.438 | 0.188 |
| mmlu | low_grad_NegU | 0.3 | 0.094 | 0.906 |
| mmlu | magnitude_A | 0.3 | 0.188 | 1.000 |
| mmlu | random | 0.3 | 0.031 | 1.000 |
| mmlu | searched | 0.3 | 0.219 | 0.469 |

This is the budget selected by the probe in `runs/v1-gsm8k-search/SUMMARY.md`.
