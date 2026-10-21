# Portable cluster runtime implementation plan

> **For agentic workers:** Execute each task with its checks; preserve the original v0 profile and all old run evidence.

**Goal:** Run SIRE from configurable paths on a local GPU or one Slurm-allocated GPU, with a separate BF16 A100-40GB profile and bounded working memory.

**Architecture:** Config overrides are resolved once before launch and frozen per run. The legacy GPU4 launcher remains unchanged. Low-memory execution uses per-tensor durable FP32 accumulators and on-demand pristine weights.

**Tech Stack:** Python, PyTorch, NumPy, YAML, Bash, Slurm.

**Spec:** User request: portable CLI parameters and a new configuration for the N32-H A100-40GB environment, preserving source weights and the old experiment.

## Global constraints
- Existing v0 defaults and evidence remain unchanged.
- Slurm mode never expands scheduler GPU visibility; exactly one visible GPU is checked.
- Precision changes define a new run, never a continuation of FP32 v0.
- Original checkpoints are read-only; G/U accumulation and scoring stay FP32.
- No GPU experiment or claimed memory measurement without real hardware execution.

## Review focus
- Missing or ambiguous GPU allocation must fail before model load.
- Resume must not silently change precision, paths, device policy or data configuration.
- Interrupted accumulator writes must not double-count a sample.
- Streaming references must preserve rho=1 bitwise reconstruction.
- Empty/missing quota output does not imply unlimited disk.

## Tasks
1. [x] Add portable config expansion, CLI overrides and allocation-aware GPU selection. Test environment masks, config precedence and invalid selection.
2. [x] Add bounded per-tensor FP32 accumulation with atomic checkpoint replacement. Test interrupt/retry, configuration identity and equality to in-memory statistics.
3. [x] Add configurable gradient dtype, streamed weight references and bounded feature cache. Test reconstruction and cache bounds; run existing small-network checks.
4. [x] Add A100 profile, portable launcher, Slurm script and migration documentation. Syntax-check shell and CLI; document ARM dependency and unmeasured GPU limits.
5. [x] Run tests, inspect diff, commit and attempt authorized branch push; if unavailable provide a git bundle/patch for application on the user's server.

## Validation

9 portable CPU checks passed, including a tiny-model BF16 disk-statistics pipeline
with mocked CUDA boundaries; original 7 small-network checks passed. Python
compilation, shell syntax and whitespace checks passed. No real GPU/ARM model
execution or 40GB fit measurement was performed.
