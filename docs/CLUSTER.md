# Portable runtime and A100 40GB migration

This is a new BF16 experiment profile, not a continuation or successful result of
the original FP32 v0 run. The user authorized a new hardware/precision profile.
Do not resume `v0-20261021-formal` with it. No GPU memory or ARM compatibility
measurement has been performed by this change.

## Profiles and paths

- `configs/v0.yaml` + `scripts/run_gpu4.sh`: original A800 setup, unchanged.
- `configs/local_bf16.yaml`: local development, original GPU4 UUID, BF16 gradients,
  disk FP32 accumulators and streamed references; separate runs and prepared data.
- `configs/a100_40gb.yaml`: Slurm allocation, same low-memory settings, batch 1;
  defaults to `~/SIRE`, `~/data/models`, `~/data/SIRE`, `~/SIRE/runs-cluster`.

Move original model/data files by copy or point paths at them. Original weights
are never modified. Downloads expand `~` and environment variables in `paths`:

```bash
# Login node; activate a working conda environment before these commands.
cd ~/SIRE
python scripts/fetch_models.py --config configs/a100_40gb.yaml --source modelscope
python scripts/fetch_data.py --config configs/a100_40gb.yaml --source modelscope
```

The N32-H machines are ARM/aarch64. Do not blindly install the original CUDA-13
pinned `requirements.txt` on this cluster. Ask support for a compatible ARM CUDA
PyTorch wheel/environment, then check the remaining dependency versions and API
compatibility. `module avail` and `/home/bingxing2/apps/package` are the starting
points. Test imports, `dtype=` loading and CUDA BF16 support in a short GPU job.
The new code does not install packages or run experiments automatically.

## CPU checks on the original A800 host (GPU4 may remain broken)

```bash
conda activate sire
python scripts/run_tests.py
PYTHONPATH=src python -m unittest discover -s tests -v
bash scripts/run.sh --help
```

No GPU is required for these checks. A CPU test pass does not demonstrate the
A100 job fits in 40GB. Data freezing does not need a GPU, but on N32-H it belongs
in a batch job because the login node is not a CPU compute service.

## Submit from the repository root

First confirm the site's conda module with `module avail`. The submission script
defaults to the manual's `anaconda/2021.11`; override `SIRE_CONDA_MODULE` if needed.
`SIRE_CONDA_ENV` defaults to `sire`. Install/activate the compatible runtime first.

```bash
# Freeze data; no GPU compute, but allocate according to the site's GPU-only rule.
sbatch --time=00:15:00 scripts/submit_a100.sh prepare-data --config configs/a100_40gb.yaml

# Short model/tokenizer/environment audit (fresh run ID).
sbatch --time=00:10:00 scripts/submit_a100.sh audit --config configs/a100_40gb.yaml --run-id a100-audit-1

# Measured resource pilot: run the real stages, stop at the pilot before search.
# Slurm --export supplies SIRE_INTERRUPT_AFTER=pilot to the batch environment.
sbatch --time=00:30:00 --export=ALL,SIRE_INTERRUPT_AFTER=pilot scripts/submit_a100.sh run --config configs/a100_40gb.yaml --run-id a100-pilot-1

# Formal run only after inspecting peak GPU AND host memory and numerical checks.
# Default wall limit: 2h. Internal compute ledger and billable elapsed time differ.
sbatch scripts/submit_a100.sh

parajobs
# Substitute the actual job ID, then inspect output and recorded run state.
tail -n 80 slurm-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

`sacct` availability and GPU memory reporting are site dependent. The runtime
writes `gpu_monitor.jsonl` (MiB); Slurm MaxRSS is host memory, often on the batch
step row. Partial jobs may be incomplete rather than passing acceptance. The
internal budget does not pad idle time and does not equal the site's billable time.

Resume the resource pilot only after reviewing its artifacts, using its recorded
configuration; do not export `SIRE_INTERRUPT_AFTER=pilot` again:

```bash
sbatch scripts/submit_a100.sh resume --config ~/SIRE/runs-cluster/a100-pilot-1/runtime_config.yaml --run-id a100-pilot-1
```

Slurm may allocate another physical GPU on a later job. The code keeps scheduler
visibility intact and verifies one CUDA device. Each process/allocation's actual
UUID and job ID are recorded and checked against its monitor records. An explicit
`device.expected_uuid` still pins a single physical card across resumes.

## CLI overrides (before freezing)

```bash
bash scripts/run.sh run --config configs/local_bf16.yaml --run-id local-bf16-1 \
  --base-model /home/nxq/data/KEPT/models/Qwen2.5-7B \
  --reasoning-model /home/nxq/data/KEPT/models/DeepSeek-R1-Distill-Qwen-7B \
  --gradient-dtype bfloat16 --batch-size 1 \
  --set features.accumulator=disk --set runtime.reference_storage=stream
```

This GPU command is for use only when the configured card is usable. CPU editing
and tests on the faulty machine are sufficient to prepare the migration.

`--device-mode`, `--gpu-index`, `--expected-gpu-uuid`, `--work-dir`, `--runs-dir`,
`--data-root` and repeatable `--set KEY=VALUE` are also supported. `--data-root`
updates both the prepared-data root and its `raw` child; use `--set paths.raw_data=...`
if reusing raw data elsewhere. Path overrides are independent: changing work-dir
does not automatically rewrite model/data/run paths. `--set` takes precedence over
named options. Resume rejects conflicting public configuration before GPU access.
The pilot-frozen configuration is adopted when resuming from the recorded runtime
configuration. Keep the run directory in the same absolute path.

Local `physical` mode uses the configured physical index and optional UUID before
torch import; `visible` mode requires one preselected CUDA_VISIBLE_DEVICES entry;
`slurm` additionally requires SLURM_JOB_ID and never overwrites the device mask.
Do not select physical indices inside a Slurm allocation.

## Memory, disk and statistical implications

- BF16 model/parameter gradients; FP32 `G/U` summation and scoring. This changes
  numerical results relative to FP32 and is recorded in feature metadata.
- Stream pristine B/R target tensors from their safetensors shards, rather than
  keeping both target sets on the GPU. Every mask starts from pristine weights;
  rho=1 and no-drift correctness checks remain enabled.
- Feature cache is cleared at each tensor-score boundary. Final FP16 D/G/U files
  remain about 37GB for the reference models; they were already 16-bit.
- Disk accumulators hold about 54GB total of FP32 sums on disk, but load only one
  target tensor's sums at a time. Atomic replacement needs extra space for one
  tensor. Each tensor records its own sample cursor, so replay after interruption
  cannot double-count the tensors already committed for that sample.
- Statistics are deleted only after durable feature metadata is written. Disk
  mode does more I/O (each sample rewrites the statistics); on a shared NFS this
  may cost substantially more wall time. Do not delete partial states manually
  while a job is running. Limit concurrent experiments and duplicate caches.
- Expect roughly 30GB decimal for the two BF16 checkpoints, about 37GB features,
  about 54GB temporary statistics, plus datasets, dependencies and download caches.
  These are estimates, not measured quota or peak memory for this migration.
- 300GB is plausible for one run, but the 55GB host RAM allocation is separate.
  Real-model GPU/host peaks and runtime must be measured. If a single calibration
  backward still exceeds memory, further offload/layerwise gradients are needed.
- The existing Hugging Face eval/checkpointing behavior has not been revalidated
  on the cluster's transformer version; inspect the real backward peak.

`data.gsm8k_files` is now configurable. The new profiles leave it empty; the
manifest explicitly records `not_audited`. Point it at real GSM8K reference
parquet files before freezing if the overlap check is required. Missing references
are never interpreted as a successful contamination audit.

## Storage quota

```bash
timeout -k 2s 10s quota -s
du -sh "$HOME"
```

If NFS quota RPC returns Connection refused, use the control panel or ask support.
It does not mean unlimited storage. Shared filesystem free space is not a personal
quota. The user's control panel currently states 300GB; home usage was 48KB before
migration. No quota change is made by this code.

## Failure evidence and deployment status

The CLI now writes `last_error.json` and append-only `failures.jsonl` in the run
folder. Early configuration/import/data-preparation failures go under
`diagnostics/startup-PID/`. Records include the stage, traceback, failure category,
an explicitly unconfirmed cause hypothesis, suggested next check, scheduler job
ID, host peak RSS, CUDA allocated/reserved peaks when available, filesystem free
space (explicitly NOT personal quota), and best-effort nvidia-smi output. CUDA
errors alone are never labeled proven hardware failures. Failure logging is best
effort when storage is full or a device is lost.

`resources.jsonl` records resource snapshots after successful stages. CUDA/RSS
peaks are process-lifetime high-water marks, not isolated per-stage measurements.
A SIGKILL/host OOM kill cannot be caught; inspect Slurm State/ExitCode/MaxRSS and
batch output if no Python error record was written.

Before installing or downloading on N32-H, run this read-only preparation check:

```bash
bash scripts/cluster_preflight.sh
```

Send the output for environment selection. The script does not install Python,
change quota, allocate a GPU, or run model/data processing on the login node.

At this revision, forward activation statistics have NOT been added to the DSL.
The inputs remain D/A/G/U/B/R. Combining forward activation features with backward
gradient features requires a separate explicit protocol revision before freezing
any formal run. No claim of safe 40GB execution follows from CPU tests.
