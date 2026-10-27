# GOAL: SIRE v0 — Search a Compact Symbolic Score for Sparse Reasoning Task Vectors

Version: 2026-10-20. Initial state: goal defined only. No environment, code, or
result is confirmed by this document. This is an execution contract that a CLI
coding agent can take and run on its own.

## 1. What we are trying to find out

A reasoning model R and a reference base B differ by a task vector
`Δ = θ_R − θ_B`. If we keep only a fraction `ρ` of Δ's entries and drop the
rest back to B, how much reasoning ability survives, and does the *choice of
which entries to keep* matter more than existing fixed heuristics?

v0 answers one question with auditable evidence:

> Under one fixed protocol, can an automatically searched, short symbolic score
> `s = f(base symbols)` retain more reasoning accuracy at low keep-rates than
> magnitude, low-gradient, and random retention?

A negative result is an acceptable outcome. An unaudited positive result is not.

Scope honesty, stated up front and repeated in the report: sparsifying Δ reduces
the number of *retained task-vector parameters*. It does not reduce neurons,
layers, FLOPs, or latency. No speedup is claimed.

## 2. Hard constraints

| Item | Fixed value |
|---|---|
| Working directory | `/home/nxq/SIRE/` |
| Reference base B | `/home/nxq/data/KEPT/models/Qwen2.5-7B/` |
| Reasoning model R | `/home/nxq/data/KEPT/models/DeepSeek-R1-Distill-Qwen-7B/` |
| Data root | `/home/nxq/data/dataset/` (project data under `.../SIRE/`) |
| Python | Miniconda env, default name `sire` |
| Compute | physical GPU 4 only, UUID verified at launch |
| GPU budget | 7200 s of useful GPU stage time, +300 s grace for the last unit |

Allowed without asking: inspect and reuse existing code, create the env, install
dependencies, prepare data, modify the implementation, run tests and
experiments, recover from failures, record results.

Not allowed: other GPUs, system Python, substituting models, multi-GPU, extending
the compute budget, killing other users' processes, overwriting original weights
or datasets. If a file already exists in the working directory, read it, diff it,
merge it. Do not clobber.

Out of scope for v0: expert merging, compensation training, soft masks, per-layer
quota search, differentiable symbolic networks, cross-model transfer, and
paper-grade significance testing.

## 3. Root-level deliverables (the thing the user looks at first)

After every completed run, these files exist in `/home/nxq/SIRE/` and are
regenerated from on-disk logs, never hand-written:

| File | Content |
|---|---|
| `RESULTS.md` | One table per run: run_id, frozen formula in infix form, complexity, accuracy at every keep-rate, headline metric, baselines on the same rows |
| `FORMULA.txt` | The frozen formula of the latest run, infix, one line, plus its canonical AST and mask hash |
| `curve.csv` | Long-format curve data: `method,formula_id,rho,n_items,n_correct,accuracy,rel_to_full_R,mean_new_tokens,truncation_rate,nll_delta,seconds` |
| `curve.png` | Degradation curves: x = keep-rate ρ (log scale), y = accuracy; one line per method, dashed horizontal line at full-R accuracy, shaded band at 90% of full R |
| `runs/<run_id>/` | Everything else: per-sample predictions, candidates, checkpoints, budget, GPU monitor logs |

`RESULTS.md` accumulates one section per run and is never truncated. Each run
appends; earlier runs stay readable so formula drift across runs is visible.

Headline metric, printed at the top of `RESULTS.md`:

```
ρ*(f) = min { ρ : accuracy(f, ρ) ≥ 0.90 × accuracy(full R) }
```

Lower is better. Report `ρ*` for the searched formula and for each baseline. If
no tested ρ reaches the threshold, report `ρ* = n/a` and the best accuracy
achieved. Do not interpolate between tested points.

## 4. Base symbols: explicit and obvious

Exactly six leaves. No others. No free constants, no layer index, no position
index, no learned scalars.

| Symbol | Definition | Meaning |
|---|---|---|
| `D` | `θ_R − θ_B` | signed task vector, what the reasoning tuning added |
| `A` | `abs(D)` | task-vector magnitude |
| `G` | `mean_z g(z)` | signed mean gradient of reasoning loss w.r.t. the parameter |
| `U` | `mean_z abs(g(z))` | gradient magnitude, per-sample sensitivity |
| `B` | `θ_B` | base weight |
| `R` | `θ_R` | reasoning weight |

`g(z)` is the gradient of the mean cross-entropy over assistant reference tokens
of calibration sample `z`, taken on the full R model, FP32, microbatch 1, samples
equally weighted. `A` and `U` are distinct and non-substitutable:
`U = Mean(Abs(g)) ≠ Abs(Mean(g)) = Abs(G)`. A correctness test must assert this.

Exactly six operators: `Neg`, `Abs`, `Add`, `Sub`, `Mul`, `SafeDiv`.

`SafeDiv(x, y) = x / (sign(y) · max(|y|, 1e-6))`, `sign(0) = +1`.

Formula space: typed expression tree, at most 4 parameter-level operator nodes,
depth at most 3. Every leaf is divided per-tensor by its own fixed positive RMS
scale with sign preserved, so all leaves are dimensionless and comparable. A
candidate producing any non-finite value is void.

Complexity = number of unique operator nodes in the expanded DAG. The `Mean` and
`Mean∘Abs` reductions inside `G` and `U` are not free; `G` and `U` each cost 1.
`D`, `A`, `B`, `R` cost 0 beyond their node.

Canonicalization is mandatory: commutative-argument ordering, double-negation
folding, canonical AST string, mask-hash dedup. Ties in score are broken by
(parameter name, flat index), never by anything candidate-dependent.

Every formula is written to disk in two forms: canonical AST and human infix,
e.g. `s = SafeDiv(Mul(A, U), Add(A, U))` → `s = (A * U) / (A + U)`.

## 5. Masking and reconstruction

Target set S: `weight` of `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj` in every transformer layer. All other parameters stay at R.

Per target tensor, keep exactly `floor(ρN)` positions with the highest score.
Reconstruct as `θ = θ_B + mask ⊙ Δ`, no rescaling. Implementation may use the
equivalent `where(mask, θ_R, θ_B)` to avoid subtract-add drift; Δ features are
computed in FP32.

`ρ = 1` must reproduce R exactly. `ρ = 0` reverts S to B and is not "full B".
Original weights are read-only; every candidate is loaded from pristine B/R, so
mask switching cannot accumulate error. One resident model, weights swapped in
place, generation state cleared between candidates.

Verify before anything else: architecture, parameter names, shapes, config, and
tokenizer mapping of B vs R. No truncating or padding to force a shape match. All
runs use R's config, tokenizer, and chat template; the audit records actual
vocabulary and special-token differences. Note that the officially documented
training base of R1-Distill-Qwen-7B is Qwen2.5-Math-7B, so Δ against Qwen2.5-7B
is a difference against *this chosen reference*, not purely distillation effect.

## 6. The sparsity sweep

Curve grid, frozen before any evaluation:

```
ρ ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50}  plus  full R (ρ = 1)
```

Search objective, evaluated only on the search split and only at the two cheap
anchor points:

```
J(f) = mean over ρ ∈ {0.05, 0.20} of accuracy_search(f, ρ)
```

The full 7-point curve is measured only for: the frozen formula, and each
baseline. This keeps the curve informative without spending the search budget on
it. If the budget cannot cover 7 points for all methods, drop points from the
dense middle (`0.02`, `0.50`) first, in that fixed order, and record the
reduction in the frozen config. Never drop a point after seeing its result.

Mandatory baselines, identical protocol, identical quota, identical calibration:
full R; random retention (1 seed); magnitude `A`; low-gradient `Neg(U)`. The
low-gradient baseline is named a *criterion baseline*. It is not a reproduction
of ReasonAny, and no RCP-Merging reproduction is claimed.

## 7. Data and freezing protocol

Math: `AI-MO/NuminaMath-CoT`, `source=cn_k12` subset. General-ability monitor:
`HuggingFaceH4/no_robots`. Check actual fields, version, and availability first;
store under `/home/nxq/data/dataset/SIRE/` with hashes recorded in the project.

Keep only text-solvable items with a single verifiable ground-truth answer.
Exclude image-dependent, proof-style, corrupted, and unreliable-answer items.
Never select items by whether the model answers them correctly. Split by
near-duplicate cluster, not by row.

Initial sizes: pilot 4, calibration 8, search 8, selection 8, test 16. Plus 8
single-turn No Robots samples for assistant-token NLL monitoring, which never
enter formula inputs and are not a v0 gate.

If the pilot shows the budget is short, sizes may be reduced once, before any
formal evaluation, to search/selection/test = 4/4/8 under a fixed seed, with the
reason written into the frozen config. If still infeasible, report a budget block.
Never change item counts, swap items, or pick keep-rates after seeing candidate
performance.

Calibration uses only samples whose full sequence is ≤ 2048 tokens, with the
final answer never truncated. Gradients are collected on S only, with autograd
on, model in eval, no parameter updates, KV cache off, checkpointing and
CPU/mmap streaming accumulation as needed. Per-sample gradients are never all
held in memory.

Freeze item IDs, prompt hashes, ground-truth answers, generation config, and
seeds for all four splits. The test split opens only after the formula is frozen;
the search module has no read path to test content or test results. Generation
errors, timeouts, and parse failures count as wrong and stay in the denominator.

No GSM8K / MATH / AIME used to drive the search. Internal dedup is mandatory;
cross-check against external benchmark sets where available, and record the
un-audited surface where not. Do not claim zero contamination. If data cannot be
obtained, report a block; synthetic data never substitutes for acceptance
evidence.

## 8. Search and selection

Single-seed discrete evolutionary search over expressions. Initialize from the
simple seeds (`A`, `Neg(U)`, `Mul(A, U)`, `SafeDiv(A, U)`). Propose children by
leaf substitution, operator substitution, subtree insertion/deletion, or a random
fresh tree. Select parents from a non-dominated archive on (J, complexity).

At least 8 valid new formulas, up to 16 if time allows. Existing baselines and
mask-hash-equivalent candidates reuse cached evaluations and do not count as new
candidates. No independent random-search control is run, therefore no claim is
made that evolution beats random search.

At most 2 formulas advance to the independent selection split. Rank by the same
`J`, freeze exactly one shared formula across all keep-rates. Do not pick a
different formula per keep-rate. Then measure the full curve on test.

Tie-break: lower complexity first, then canonical AST lexicographic order.

## 9. Generation and answer checking

One decoding setup for every method: same template, prompt, backend, precision,
and length cap. Boxed final answer required; no duplicated think-prefix.
Defaults: single sample, `temperature=0.6`, `top_p=0.95`,
`max_new_tokens=2048`. If the pilot shows heavy truncation and budget allows,
raise to 4096 uniformly *before* formal evaluation and freeze it.

Per-item seeds derive from (global seed, item ID) so resume order cannot change
sampling. Answer checking handles integers, fractions, and expressions it can
verify reliably; symbolic parsing runs in a time-limited subprocess and never
executes generated code. Save raw output, ground truth, parsed answer,
correctness, token count, truncation flag, and wall time for every item.
Accuracy is accuracy *under this output budget*, not unbounded capability.

## 10. Success conditions and evidence

| ID | Observable result | Evidence |
|---|---|---|
| C1 | Miniconda env used; all GPU work on physical GPU 4 only | env snapshot, `sys.executable`, `CONDA_PREFIX`, target UUID, task PID, device monitor log |
| C2 | No shared or confirmed near-duplicate items across calibration / search / selection / test; sizes and versions frozen | data manifest, split IDs and hashes, dedup audit |
| C3 | Exact per-tensor quota, correct boundaries, non-target params equal R, no drift across candidate switches | automated correctness tests, sampled real-weight audit |
| C4 | ≥ 4 distinct-mask non-baseline formulas evaluated by real generation; ≥ 1 child selected from a parent based on measured results | candidates.jsonl with AST, parent link, mask hash, per-sample outputs, search log |
| C5 | One frozen formula measured on the held-out test split across the full keep-rate grid against full R, random, magnitude, low-gradient | freeze record, per-sample records, `curve.csv`, `curve.png`, `RESULTS.md` |
| C6 | ~7200 s of useful GPU stage time, no unrecovered exception, failures and recoveries logged | `budget.json`, per-stage ledger, GPU monitor, final exit status |
| C7 | Single-command start, resumable from checkpoint, report traceable to config and logs | source, dependency snapshot, run instructions, `reports/REPORT_EN.md` |
| C8 | Root-level `RESULTS.md`, `FORMULA.txt`, `curve.csv`, `curve.png` present, regenerated from logs, consistent with `runs/<run_id>/` | file diff of regenerated vs committed artifacts |

A `verify` command checks each condition against files and content, prints
per-item status with evidence paths, returns 0 only if all pass. It recomputes
retained counts, item counts, hashes, budget, and result aggregates. It must not
read a human-written "passed" field. It must also re-derive `curve.csv` and
`curve.png` from per-sample logs and fail on mismatch.

Code written, tests passing, job launched, process alive for two hours, budget
exhausted, or a single lucky improvement — none of these alone means done.

Never modify acceptance logic, delete failing items, edit ground truth, relax a
quota, or fabricate timing to reach a passing state.

## 11. Budget and timing

Order: environment and model audit → CPU small-network correctness → minimal
real-7B loop and pilot → freeze config → formal run → acceptance and report.

Development-time GPU debugging is accounted separately. The formal run covers
gradient statistics, correctness checks, baselines, search, selection, and test.
Target 7200 s of GPU stage time, ≤ 300 s grace at the end. Installs, downloads,
CPU sorting, disk waits, and sleeps do not count.

Stage timing uses a monotonic clock with synchronization at GPU boundaries. Mixed
CPU/GPU stages are split further; whole-function wall clock is never booked as
GPU time. Report CUDA-event compute time, stage wall clock, CPU/IO, and total
wall clock separately. Sample target UUID, task PID, utilization, and memory
about every 5 s; utilization corroborates isolation and is not used as billing.
This is a single-card *work-time* budget. No claim of two hours of saturated
kernel compute.

Indicative schedule, corrected by pilot throughput: calibration and checks 20
min, baselines 25 min, search 40 min, selection and test 30 min, slack 5 min.
Predict cost before opening a new candidate and always reserve time for C5 and
acceptance. Never exhaust the budget and then skip the held-out comparison.

Spare time goes first to more search before test opens. If the main flow finishes
early, run the pre-declared gradient-prefix stability check or a second search
seed on the search split. Never reopen test for tuning. If no useful work remains,
report the shortfall honestly; idling to pad time is forbidden.

Launcher: `nvidia-smi` resolves the UUID of physical index 4, then
`CUDA_VISIBLE_DEVICES=<that UUID>` is set *before* importing torch. The process
sees one device, uses `cuda:0`, and verifies UUID and visible-device count.
Confirm Miniconda's real install path rather than assuming it. If GPU 4 is busy,
do not switch cards and do not kill other processes.

On OOM: reduce generation batch, enable checkpointing, chunk per tensor, offload
to CPU. Do not change the model or the device. Statistics and scoring stay FP32;
inference prefers BF16, falls back to uniform FP16 with a record. Any protocol
change invalidates inconsistent earlier results, which are then rerun.

## 12. Iteration loop and persistent state

Each cycle: read latest state → pick the next step with the most evidence behind
it → implement one verifiable increment → run checks → record → continue. Fix
failing correctness checks before adding experiments. Log implementation changes
and candidate evaluations separately; a code change is not a search iteration.

Backend tuning, caching, batch sizes, and module layout may change freely.
Anything affecting fair comparison — data, formula space, generation budget,
precision — must be frozen before formal evaluation or explicitly invalidated and
rerun under one protocol. Paths, device, and budget never change silently.

Three state files: `PLAN.md` (milestones, progress, verification commands),
`DECISIONS.md` (consequential choices and reasons), `HANDOFF.md` (current stage,
run_id, PID, latest evidence, remaining budget, last failure, next runnable
command). After any context reset, read these and the current config before
acting; never reconstruct state from memory.

Checkpoint at every stage boundary and before exit: statistic accumulators and
sample cursor, candidates and parent links, archive, evaluation cache, RNG state,
cumulative budget, model/data/config hashes. Atomic writes. Verify hashes before
resuming; never reset the budget or recompute finished samples. Signals exit
cleanly, and one real interrupt-and-resume test is performed and recorded.

## 13. Commands

```bash
cd /home/nxq/SIRE/
bash scripts/run_gpu4.sh audit         --config configs/v0.yaml
bash scripts/run_gpu4.sh prepare-data  --config configs/v0.yaml
bash scripts/run_gpu4.sh run           --config configs/v0.yaml --gpu-budget-seconds 7200
bash scripts/run_gpu4.sh resume        --run-id RUN_ID
bash scripts/run_gpu4.sh verify        --run-id RUN_ID
bash scripts/run_gpu4.sh report        --run-id RUN_ID   # regenerates root artifacts
```

Subcommands to implement: `audit`, `prepare-data`, `collect-features`,
`baselines`, `search`, `select`, `test`, `report`, `run`, `resume`, `verify`.
These are contracts to build and then actually exercise, not existing tools.

Suggested layout: `src/sire/{data,features,dsl,masking,search,eval,report}/`.

Correctness tests cover at minimum: exact quota and boundary cases; non-target
parameters unchanged; no drift across candidate switches; `Mean(Abs(g))` vs
`Abs(Mean(g))`; hand-written criteria matching their DSL equivalents;
deterministic tie-breaking; SafeDiv behavior at and near zero; split isolation;
parse failure counted wrong; resume not recomputing finished samples; `ρ=1`
reproducing R bit-for-bit under the `where` path. Small networks first, then
sampled real weights and real generation.

## 14. Stop and block

Stop and deliver when C1–C8 all pass. If the budget runs out first, mark
"budget exhausted, incomplete", keep all evidence and the resume entry point.
Budget expiry never substitutes for acceptance.

Debug ordinary errors inside the constraints. After 3 consecutive failures from
the same cause with no new evidence, stop guessing and switch to an
evidence-backed alternative. If no legal path remains — incompatible shapes,
Miniconda unavailable, GPU 4 unavailable, data unobtainable, minimal loop
measured over budget — stop the dependent steps, finish the independent code
work, and write a block report.

A block report states: what was tried, raw errors and evidence, which conditions
were actually met, what is unmet, and the minimum input or resource needed to
continue. A success report states only what was executed. Target numbers are
never written as measured values. Small sample sizes, output-length caps, and the
reference-base mismatch must appear in the interpretation. Sparse Δ is not a
sparse model and is not an inference speedup.

## 15. Initial progress

- [x] Goal contract and v0 acceptance conditions written.
- [x] Environment, existing code, and resource audit complete.
- [x] Correctness checks and real-model pilot complete.
- [x] Config and data frozen.
- [x] Formal search and held-out curve measurement complete.
- [x] C1–C8 verified; root artifacts and report generated.

Unchecked items must be updated by the executing agent from real evidence. This
document does not assert anything has run.
