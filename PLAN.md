# PLAN — SIRE v0

## Milestones

- [x] M0: Goal contract read; environment and resource audit started.
- [x] M1: Miniconda env `sire` (clone of existing torch env) + dependencies; physical GPU 4 UUID verified.
- [x] M2: Code scaffold: data, DSL, features, masking, eval, search, report, verify, CLI.
- [x] M3: Frozen data: NuminaMath-CoT cn_k12 numeric subset, dedup clusters, pilot/calibration/search/selection/test + No Robots monitor.
- [x] M4: Model audit (B vs R architecture/tokenizer/hashes) — dev evidence in `runs/dev-audit/`.
- [x] M5: Feature collection (D/A/G/U/B/R with RMS scales) — dev evidence in `runs/dev-features/`.
- [~] M6: Correctness checks: small-network 7/7; real weights 12/13 before the GPU fault; one tolerance fix not re-run.
- [x] M7: Real-7B pilot: 4/4 correct, 5301 tokens in 58 s, no truncation.
- [ ] M8: Formal search and selection/freeze — **blocked** by GPU 4 loss.
- [ ] M9: Held-out test curve — not started; test never opened.
- [ ] M10: C1–C8 verify — blocked. See `BLOCK_REPORT.md`.

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
