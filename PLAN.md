# PLAN — SIRE v0

## Milestones

- [x] M0: Goal contract read; environment and resource audit started.
- [x] M1: Miniconda env `sire` (clone of existing torch env) + dependencies; physical GPU 4 UUID verified.
- [x] M2: Code scaffold: data, DSL, features, masking, eval, search, report, verify, CLI.
- [x] M3: Frozen data: NuminaMath-CoT cn_k12 numeric subset, dedup clusters, pilot/calibration/search/selection/test + No Robots monitor.
- [x] M4: Model audit (B vs R architecture/tokenizer/hashes) — dev evidence in `runs/dev-audit/`.
- [x] M5: Feature collection (D/A/G/U/B/R with RMS scales) — dev evidence in `runs/dev-features/`.
- [x] M6: Correctness checks: small-network 7/7; real-weight suite 14/14 in the formal run.
- [~] M7: Real-7B pilot — development pilot passed (4/4 correct, 58 s); formal-run pilot interrupted by the second GPU 4 PCIe failure.
- [x] M8: Formal search and selection/freeze — 8 distinct non-baseline mask signatures; frozen `s = (A * U)`.
- [x] M9: Held-out test curve — 8 test items, 5 keep-rates + full R, all baselines; negative result.
- [x] M10: C1–C8 verify all pass; root artifacts and `reports/REPORT_EN.md` generated.

## Verification commands

```bash
cd /home/nxq/SIRE
bash scripts/run_gpu4.sh audit --config configs/v0.yaml
bash scripts/run_gpu4.sh prepare-data --config configs/v0.yaml
bash scripts/run_gpu4.sh run --config configs/v0.yaml --gpu-budget-seconds 7200
bash scripts/run_gpu4.sh resume --run-id RUN_ID
bash scripts/run_gpu4.sh verify --run-id RUN_ID
bash scripts/run_gpu4.sh report --run-id RUN_ID
/usr/bin/env python scripts/run_tests.py   # small-network correctness
```

## Notes

- GPU budget is work-time, not saturated-kernel time. Installs, downloads, CPU sorting and sleeps are not booked.
- Test split is opened only after `freeze.json` exists; `search.py` receives no test data.
