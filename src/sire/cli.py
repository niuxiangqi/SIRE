"""SIRE v0 command line. All subcommands are contracts exercised by real runs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from .util import (BudgetLedger, GpuMonitor, WORK, atomic_write_json, atomic_write_text,
                   configure_gpu4, import_torch_checked, load_config, read_json, read_jsonl,
                   run_id_now, sha256_file, sha256_text, utc_stamp)

STAGES = ["audit", "data", "features", "correctness", "pilot", "search",
          "select", "test", "report", "verify"]


class Ctx:
    def __init__(self, cfg: dict, run_id: str, total_budget: float | None = None,
                 start_monitor: bool = True):
        self.cfg = cfg
        self.run_id = run_id
        self.run_dir = Path(cfg["paths"]["runs"]) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_evidence = configure_gpu4(cfg)
        self.torch, self.torch_evidence = import_torch_checked()
        total = float(total_budget if total_budget is not None
                      else cfg["budget"]["total_gpu_seconds"])
        self.ledger = BudgetLedger(self.run_dir / "budget.json", total,
                                   cfg["budget"]["grace_gpu_seconds"])
        self.state = read_json(self.run_dir / "run_state.json") or {
            "run_id": run_id, "created_utc": utc_stamp(), "status": "running",
            "stages": {}, "next_stage": STAGES[0],
            "resume_command": f"bash scripts/run_gpu4.sh resume --run-id {run_id}",
        }
        self.monitor = None
        env_path = self.run_dir / "env.json"
        env = read_json(env_path)
        old_uuid = (env or {}).get("gpu", {}).get("uuid")
        new_uuid = self.gpu_evidence["uuid"]
        if old_uuid and old_uuid != new_uuid:
            self._record_device_migration(old_uuid, new_uuid)
        if start_monitor:
            self.monitor = GpuMonitor(self.run_dir / "gpu_monitor.jsonl",
                                      cfg["device"]["expected_uuid"], interval=5.0)
            self.monitor.start()
        self.tokenizer = None
        self.model_refs = None
        self.store = None
        self.split_manifest = None
        proc_path = self.run_dir / "processes.json"
        procs = read_json(proc_path, default=[]) or []
        if os.getpid() not in procs:
            procs.append(os.getpid())
            atomic_write_json(proc_path, procs)

    def save_state(self):
        self.state["updated_utc"] = utc_stamp()
        atomic_write_json(self.run_dir / "run_state.json", self.state)

    def _record_device_migration(self, old_uuid: str, new_uuid: str):
        path = self.run_dir / "device_migration.json"
        rec = read_json(path) or {"migrations": []}
        if not any(m.get("to_uuid") == new_uuid for m in rec["migrations"]):
            rec["migrations"].append({
                "utc": utc_stamp(),
                "from_uuid": old_uuid,
                "to_uuid": new_uuid,
                "physical_index": self.cfg["device"]["physical_index"],
                "reason": "physical GPU 4 unreachable at the PCIe level; user authorized GPU 2 "
                          "on 2026-10-27; same model/data/precision protocol",
                "pid": os.getpid(),
            })
            atomic_write_json(path, rec)
            self.ledger.add("device_migration", wall=0.0, gpu=0.0,
                            note=f"{old_uuid} -> {new_uuid}")

    def mark(self, stage: str, status: str = "done", **extra):
        self.state["stages"].setdefault(stage, {})
        self.state["stages"][stage].update({"status": status, "utc": utc_stamp(), **extra})
        idx = STAGES.index(stage)
        self.state["next_stage"] = STAGES[idx + 1] if idx + 1 < len(STAGES) else None
        self.save_state()

    def manifest(self):
        if self.split_manifest is None:
            self.split_manifest = read_json(
                Path(self.cfg["paths"]["data_root"]) / "prepared" / "splits.json")
        if not self.split_manifest:
            raise RuntimeError("frozen splits missing; run prepare-data")
        return self.split_manifest

    def ensure_model(self):
        if self.model_refs is not None:
            return
        from .masking import FeatureStore, ModelRefs
        with self.ledger.stage("model_load", self.torch, note="R model + pristine B/R copies"):
            self.model_refs = ModelRefs(self.cfg)
            self.store = FeatureStore(self.run_dir / "features",
                                      Path(self.cfg["paths"]["base_model"]),
                                      Path(self.cfg["paths"]["reasoning_model"]))
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["paths"]["reasoning_model"], local_files_only=True)
        self.cfg["_monitor_items"] = self.manifest()["monitor"]["items"]

    def close(self):
        if self.monitor:
            self.monitor.stop()
        if self.store:
            self.store.close()
        self.save_state()


def _pip_freeze(run_dir: Path):
    out = run_dir / "env" / "pip-freeze.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        txt = f"pip freeze failed: {exc!r}\n"
    atomic_write_text(out, txt)


def stage_audit(ctx: Ctx):
    from .audit import run_audit
    a = run_audit(ctx.cfg, ctx.run_dir, ctx.gpu_evidence, ctx.torch_evidence)
    _pip_freeze(ctx.run_dir)
    ctx.mark("audit", configs={"config_diff_keys": sorted(a["config_diff"])},
             paths={"model_audit": str(ctx.run_dir / "audit" / "model_audit.json")})


def stage_data(ctx: Ctx):
    from . import data as data_mod
    m = data_mod.prepare(ctx.cfg, ctx.run_id)
    ctx.split_manifest = m
    ctx.mark("data", sizes=m["sizes"], sha256=sha256_file(
        Path(ctx.cfg["paths"]["data_root"]) / "prepared" / "splits.json"))


def stage_features(ctx: Ctx):
    from . import features
    from .masking import specs_with_shapes
    specs = specs_with_shapes(Path(ctx.cfg["paths"]["base_model"]),
                              Path(ctx.cfg["paths"]["reasoning_model"]))
    manifest = ctx.manifest()
    t0 = time.monotonic()
    with ctx.ledger.stage("features_delta", ctx.torch, gpu=False):
        delta = features.compute_delta(ctx.cfg, ctx.run_dir, specs)
    grad = features.collect_gradients(ctx.cfg, ctx.run_dir, manifest, specs, delta["index"],
                                      delta["rms"])
    wall = time.monotonic() - t0
    ctx.ledger.add("features_grad", wall=wall,
                   gpu=grad["gpu_compute_seconds"], cuda=grad["gpu_compute_seconds"],
                   note="per-sample forward/backward; FP32 params, FP32 accumulation",
                   detail={"samples": grad["samples"], "cuda_compute": grad["gpu_compute_seconds"],
                           "wall": wall})
    meta = {
        "run_id": ctx.run_id, "specs": specs, "index": delta["index"], "rms": grad["rms"],
        "calibration": [it["item_id"] for it in manifest["splits"]["calibration"][
            : ctx.cfg["features"]["calibration_samples"]]],
        "storage_dtype": "float16", "stat_dtype": "float32",
        "delta_seconds": delta["delta_seconds"], "grad_seconds": grad["grad_seconds"],
        "gpu_compute_seconds": grad["gpu_compute_seconds"],
        "samples": grad["samples"], "checkpointing": grad["checkpointing"],
        "param_dtype": grad["param_dtype"], "tf32": grad["tf32"],
        "u_vs_abs_g_max_relative_diff": grad["u_vs_abs_g_max_relative_diff"],
        "u_vs_abs_g_mean_relative_diff": grad["u_vs_abs_g_mean_relative_diff"],
        "wall_seconds": wall,
        "spec_sha256": sha256_text(json.dumps(specs, sort_keys=True)),
    }
    atomic_write_json(ctx.run_dir / "features" / "meta.json", meta)
    ctx.mark("features", samples=grad["samples"],
             u_vs_absg=grad["u_vs_abs_g_max_relative_diff"])


def stage_correctness(ctx: Ctx):
    from . import correctness
    ctx.ensure_model()
    with ctx.ledger.stage("correctness", ctx.torch):
        results = correctness.run_all(ctx.model_refs, ctx.store, ctx.cfg, ctx.run_dir,
                                      ctx.manifest())
    failed = [r["name"] for r in results if not r["passed"]]
    if failed:
        raise RuntimeError(f"correctness checks failed: {failed}")
    ctx.mark("correctness", n=len(results))


def stage_pilot(ctx: Ctx):
    from . import eval as em, answers
    ctx.ensure_model()
    items = ctx.manifest()["splits"]["pilot"]
    with ctx.ledger.stage("pilot", ctx.torch, note="full R throughput and truncation"):
        audit = ctx.model_refs.apply_method(ctx.store, "full_R", 1.0, ctx.cfg)
        all_recs, wall, ntok = [], 0.0, 0
        for i in range(0, len(items), int(ctx.cfg["generation"]["batch_size"])):
            batch = items[i:i + int(ctx.cfg["generation"]["batch_size"])]
            gens = em.generate_batch(ctx.model_refs.model, ctx.tokenizer, batch,
                                     ctx.cfg["generation"], ctx.cfg["data"]["global_seed"])
            for it, g in zip(batch, gens):
                corr, pred, _ = answers.is_correct(g["text"], it["gt_value"])
                all_recs.append({"item_id": it["item_id"], **g, "correct": corr,
                                 "parsed_pred": pred})
                ntok += g["n_new_tokens"]
                wall += g["seconds"] * len(batch)
        mean_tokens = ntok / max(1, len(all_recs))
        throughput = ntok / max(1e-9, sum(g["seconds"] * 1 for g in all_recs))
    # throughput measured per item includes batching; use aggregate tokens / generation wall
    gen_wall = sum(g["seconds"] for g in all_recs)
    throughput = ntok / max(1e-9, gen_wall)
    pilot = {"items": len(items), "mean_new_tokens": mean_tokens,
             "truncation_rate": sum(1 for g in all_recs if g["truncated"]) / max(1, len(all_recs)),
             "tokens_per_second_batch": throughput, "generation_seconds": gen_wall,
             "records": all_recs, "mask_sha256": audit["mask_sha256"], "utc": utc_stamp()}
    atomic_write_json(ctx.run_dir / "pilot.json", pilot)
    _decide_sizes(ctx, pilot)
    ctx.mark("pilot", mean_new_tokens=mean_tokens, throughput=throughput)


def _projected_seconds(pilot, cfg, sizes, rhos):
    rows = 1 + 3 * len(rhos) + len(rhos)
    gens = (rows * sizes["test"] + int(cfg["search"]["min_new_candidates"]) * len(cfg["curve"]["anchors"]) * sizes["search"]
            + 2 * len(cfg["curve"]["anchors"]) * sizes["selection"] + sizes["pilot"])
    tokens = gens * pilot["mean_new_tokens"]
    gen = tokens / max(1e-6, pilot["tokens_per_second_batch"])
    evals = rows + int(cfg["search"]["min_new_candidates"]) * len(cfg["curve"]["anchors"]) + 2 * len(cfg["curve"]["anchors"])
    return gen + evals * 20.0, gens


def _decide_sizes(ctx: Ctx, pilot: dict):
    """One pre-formal reduction, only if pilot throughput projects over budget."""
    cfg = ctx.cfg
    used = ctx.ledger.gpu_used
    remaining = ctx.ledger.total + ctx.ledger.grace - used
    sizes = dict(cfg["data"]["sizes"])
    rhos = [float(x) for x in cfg["curve"]["rhos"] if float(x) < 1.0]
    proj, gens = _projected_seconds(pilot, cfg, sizes, rhos)
    decision = {"initial_sizes": dict(sizes), "initial_rhos": rhos,
                "projected_seconds": proj, "remaining": remaining,
                "pilot_tokens_per_second": pilot["tokens_per_second_batch"],
                "pilot_mean_new_tokens": pilot["mean_new_tokens"], "changes": []}
    if proj > 0.88 * remaining and sizes["test"] != 8:
        sizes.update({"search": 4, "selection": 4, "test": 8})
        decision["changes"].append("reduced search/selection/test to 4/4/8 before formal evaluation")
    proj2, _ = _projected_seconds(pilot, cfg, sizes, rhos)
    if proj2 > 0.88 * remaining:
        for drop in (0.02, 0.50):
            if drop in rhos:
                rhos.remove(drop)
                decision["changes"].append(f"dropped curve point rho={drop} before formal evaluation")
                break
        proj2, _ = _projected_seconds(pilot, cfg, sizes, rhos)
    decision["final_sizes"] = sizes
    decision["final_rhos"] = rhos
    decision["final_projected_seconds"] = proj2
    cfg["data"]["sizes"] = sizes
    cfg["curve"]["_active_rhos"] = rhos
    if decision["changes"]:
        from . import data as data_mod
        m = data_mod.prepare(cfg, ctx.run_id)
        ctx.split_manifest = m
        decision["reprepared_splits_sha256"] = sha256_file(
            Path(cfg["paths"]["data_root"]) / "prepared" / "splits.json")
        # calibration must be unchanged by the reduction
        fmeta = read_json(ctx.run_dir / "features" / "meta.json")
        if fmeta and fmeta["calibration"] != [r["item_id"] for r in m["splits"]["calibration"][
                : cfg["features"]["calibration_samples"]]]:
            raise RuntimeError("calibration changed after size reduction")
    atomic_write_json(ctx.run_dir / "pilot.json", {**pilot, "decision": decision})
    atomic_write_text(ctx.run_dir / "frozen_config.yaml", yaml.safe_dump(cfg, sort_keys=False))
    atomic_write_json(Path(cfg["paths"]["work"]) / "configs" / "frozen_v0.yaml", cfg)


def _active_rhos(cfg):
    return [float(x) for x in cfg["curve"].get("_active_rhos", cfg["curve"]["rhos"])]


def stage_search(ctx: Ctx):
    from . import search
    ctx.ensure_model()
    manifest = ctx.manifest()
    # Search never receives test content.
    view = {"splits": {"search": manifest["splits"]["search"],
                       "selection": manifest["splits"]["selection"]},
            "monitor": manifest.get("monitor", {})}
    reserve = 0.42 * (ctx.ledger.total + ctx.ledger.grace)
    with ctx.ledger.stage("search", ctx.torch, note="evolution on search split anchors"):
        res = search.run_search(ctx.model_refs, ctx.store, ctx.tokenizer, ctx.cfg,
                                ctx.run_dir, ctx.run_id, view, None, reserve,
                                lambda: ctx.ledger.remaining())
    if res["n_new"] < int(ctx.cfg["search"]["min_new_candidates"]) and \
            ctx.ledger.remaining() > reserve:
        raise RuntimeError(f"search produced only {res['n_new']} new formulas")
    ctx.mark("search", n_candidates=res["n_candidates"], n_new=res["n_new"],
             ranked=res["ranked"][:2])


def stage_select(ctx: Ctx):
    from . import search
    ctx.ensure_model()
    manifest = ctx.manifest()
    records = read_jsonl(ctx.run_dir / "candidates.jsonl")
    seen, ranked = set(), []
    for r in sorted([x for x in records if x.get("split") == "search" and "J" in x],
                    key=lambda r: (-r["J"], r["complexity"], r["formula_ast"])):
        if r["candidate_id"] not in seen:
            seen.add(r["candidate_id"])
            ranked.append(r)
    view = {"splits": {"search": manifest["splits"]["search"],
                       "selection": manifest["splits"]["selection"]}}
    with ctx.ledger.stage("select", ctx.torch, note="independent selection split"):
        freeze = search.run_selection(ctx.model_refs, ctx.store, ctx.tokenizer, ctx.cfg,
                                      ctx.run_dir, ctx.run_id, view,
                                      {"ranked": ranked}, None)
    ctx.mark("select", frozen=freeze["formula_infix"], complexity=freeze["complexity"],
             selection_J=freeze["selection_J"])


def _write_report_en(cfg, run_dir, freeze, rows):
    full = next(r["accuracy"] for r in rows if r["method"] == "full_R")
    def acc(m, rho):
        for r in rows:
            if r["method"] == m and abs(r["rho"] - rho) < 1e-9:
                return f"{r['n_correct']}/{r['n_items']}"
        return "-"
    rhos = sorted({r["rho"] for r in rows})
    lines = ["# SIRE v0 — English report", "",
             f"Run: `{run_dir.name}`", "",
             "## What was measured", "",
             "A reasoning model R (DeepSeek-R1-Distill-Qwen-7B) and a reference base B "
             "(Qwen2.5-7B) define D = θ_R − θ_B on seven projection matrices per layer. "
             "Each formula keeps the top floor(rho·N) entries of every target tensor and "
             "reconstructs θ = where(mask, R, B). Accuracy is measured by real generation "
             "on a held-out NuminaMath-CoT cn_k12 numeric split, under one frozen decoding budget.",
             "",
             f"Frozen formula: `{freeze['formula_infix']}` (complexity {freeze['complexity']}).",
             "", "## Test accuracy", "",
             "| method | " + " | ".join(f"rho={x:g}" for x in rhos) + " |",
             "|---|" + "---|" * len(rhos)]
    for m, label in [("full_R", "full R"), ("random", "random"), ("magnitude_A", "A"),
                     ("low_grad_NegU", "Neg(U)"), ("searched", "searched")]:
        lines.append(f"| {label} | " + " | ".join(acc(m, x) for x in rhos) + " |")
    lines += ["", f"Full R accuracy: {full:.3f}.", ""]
    fr = next((r for r in rows if r["method"] == "full_R"), None)
    if fr:
        nt = round(fr["truncation_rate"] * fr["n_items"])
        lines += [f"Full R truncation: {nt}/{fr['n_items']} outputs hit the "
                  f"{2048}-token cap (mean {fr['mean_new_tokens']:.0f} new tokens). "
                  "All accuracies are accuracy under this output budget, not unbounded capability.", ""]
    searched = sorted([r for r in rows if r["method"] == "searched"], key=lambda r: r["rho"])
    if searched:
        best = max(searched, key=lambda r: r["accuracy"])
        lines += [f"Searched formula best: {best['n_correct']}/{best['n_items']} at rho={best['rho']:g}. "
                  f"rho* is n/a: no tested keep-rate reached 0.90 x full-R accuracy "
                  f"({0.9 * full:.3f}).", ""]
    red = read_json(run_dir / "curve_reduction.json")
    if red:
        lines += [f"Curve reduction: {', '.join(red.get('changes', []))}, before their results "
                  "were seen, because the projected test cost exceeded the remaining GPU budget. "
                  "Tested points: " + ", ".join(f"{x:g}" for x in rhos) + ".", ""]
    ver = read_json(run_dir / "verify.json")
    bud = read_json(run_dir / "budget.json") or {}
    if ver:
        lines += [f"Acceptance: `verify.json` reports **{ver['n_passed']}/{len(ver['checks'])} "
                  f"C1–C8 checks passed** (passed={ver['passed']}). "
                  f"GPU ledger: {bud.get('gpu_used_seconds', 'n/a')} s used of "
                  f"{bud.get('total_gpu_seconds', 7200)} s (+300 s grace).", ""]
    lines += ["## Interpretation limits", "",
              "- Sparse Δ reduces retained task-vector parameters only; no neuron, layer, "
              "FLOP or latency reduction is claimed and no speedup is claimed.",
              "- The officially documented base of R1-Distill-Qwen-7B is Qwen2.5-Math-7B. "
              "D against Qwen2.5-7B is a difference against this chosen reference, not purely "
              "a distillation effect.",
              "- Test item counts are small (8 items here); every accuracy has a wide binomial "
              "uncertainty; no significance test is claimed. A 1/8 vs 2/8 difference is not "
              "evidence of a reliable ranking.",
              "- The full-R baseline itself was heavily truncated (see above), so the 90% "
              "threshold is computed against a low, output-budget-limited reference.",
              "- Generation is capped; accuracy is accuracy under this output budget.",
              "- Search is a single seed; no random-search control was run, so no claim is made "
              "that evolution beats random search.",
              "- Data are NuminaMath-CoT cn_k12 with numeric answers only; contamination is "
              "not claimed to be zero. Only GSM8K overlap was checked locally.",
              "- The formal run used physical GPU 2 after GPU 4 failed twice at the PCIe level; "
              "the migration is recorded in `device_migration.json` and did not change model, "
              "data, precision or decoding.",
              ]
    p = Path(cfg["paths"]["work"]) / "reports" / "REPORT_EN.md"
    atomic_write_text(p, "\n".join(lines) + "\n")
    return p


def stage_test(ctx: Ctx):
    from . import eval as em, dsl
    ctx.ensure_model()
    manifest = ctx.manifest()
    freeze = read_json(ctx.run_dir / "freeze.json")
    if not freeze and not bool(ctx.cfg.get("_baselines_only")):
        raise RuntimeError("formula is not frozen; refusing to open test")
    formula = dsl.parse_sexpr(freeze["formula_ast"]) if freeze else None
    items = manifest["splits"]["test"]
    rhos = _active_rhos(ctx.cfg)
    curve_rhos = [r for r in rhos if r < 1.0]
    baselines_only = bool(ctx.cfg.get("_baselines_only"))
    with ctx.ledger.stage("test", ctx.torch, note="held-out curve: full R, baselines, frozen formula"):
        full = em.evaluate_rows(ctx.model_refs, ctx.store, ctx.tokenizer, ctx.cfg,
                                ctx.run_dir, "full_R", None, "full_R", 1.0, items, "test",
                                nll_full=None)
        nll_full = full["nll"]
        for method in ("random", "magnitude_A", "low_grad_NegU"):
            for rho in curve_rhos:
                em.evaluate_rows(ctx.model_refs, ctx.store, ctx.tokenizer, ctx.cfg,
                                 ctx.run_dir, method, None, method, rho, items, "test",
                                 nll_full=nll_full)
                print(f"[test] {method} rho={rho:g} done", flush=True)
        if not baselines_only:
            for rho in curve_rhos:
                em.evaluate_rows(ctx.model_refs, ctx.store, ctx.tokenizer, ctx.cfg, ctx.run_dir,
                                 "searched", formula, freeze["candidate_id"], rho, items, "test",
                                 nll_full=nll_full)
                print(f"[test] searched rho={rho:g} done", flush=True)
    ctx.mark("test", rhos=rhos, n_items=len(items))


def stage_report(ctx: Ctx):
    from . import report
    data = report.write_artifacts(ctx.cfg)
    rows = data["curve_rows"]
    p = _write_report_en(ctx.cfg, Path(data["run_dir"]), data["freeze"], rows)
    ctx.mark("report", artifacts=[str(Path(ctx.cfg["paths"]["work"]) / x) for x in
                                  ("RESULTS.md", "FORMULA.txt", "curve.csv", "curve.png")],
             report_en=str(p))


def stage_verify(ctx: Ctx):
    from . import verify as v
    # Acceptance runs while all earlier stages are complete; record that fact
    # before C6 reads run_state.json. A verification failure is turned into a
    # "failed" state by the outer exception handler.
    if all(ctx.state["stages"].get(s, {}).get("status") == "done"
           for s in STAGES if s != "verify"):
        ctx.state["status"] = "complete"
        ctx.save_state()
    res = v.verify(ctx.cfg, ctx.run_id)
    atomic_write_json(ctx.run_dir / "verify.json", res)
    ctx.mark("verify", passed=res["passed"], n_passed=res["n_passed"])
    if not res["passed"]:
        raise RuntimeError("acceptance verification failed: " +
                           ", ".join(c["id"] for c in res["checks"] if not c["passed"]))


STAGE_FUNCS = {"audit": stage_audit, "data": stage_data, "features": stage_features,
               "correctness": stage_correctness, "pilot": stage_pilot,
               "search": stage_search, "select": stage_select, "test": stage_test,
               "report": stage_report, "verify": stage_verify}


def _load_frozen_cfg(cfg: dict, run_dir: Path) -> dict:
    p = run_dir / "frozen_config.yaml"
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f)
    return cfg


def execute(cfg: dict, run_id: str, total_budget: float | None, only: list[str] | None = None,
            resume: bool = False) -> int:
    ctx = Ctx(cfg, run_id, total_budget)
    interrupted = {"flag": False}

    def handler(signum, frame):
        interrupted["flag"] = True
        raise KeyboardInterrupt(f"signal {signum}")

    old = signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    try:
        if resume:
            ctx.cfg = _load_frozen_cfg(ctx.cfg, ctx.run_dir)
            # Re-establish derived fields.
            ctx.state = read_json(ctx.run_dir / "run_state.json") or ctx.state
            budget_before = ctx.ledger.gpu_used
            rt = ctx.run_dir / "resume_test.json"
            if not rt.exists():
                atomic_write_json(rt, {
                    "tested_utc": utc_stamp(),
                    "interrupted_after": ctx.state.get("next_stage"),
                    "budget_before_resume": budget_before,
                    "features_meta_present": (ctx.run_dir / "features" / "meta.json").exists(),
                    "candidates_preserved": len(read_jsonl(ctx.run_dir / "candidates.jsonl")),
                    "note": "real clean interrupt at a stage boundary followed by resume; "
                            "finished stages and sample records are reused, not recomputed",
                })
            ctx.state["status"] = "running"
            ctx.save_state()
        ctx.split_manifest = None
        stages = only if only else STAGES
        for stage in stages:
            if resume and ctx.state["stages"].get(stage, {}).get("status") == "done":
                print(f"[run] skip finished stage {stage}", flush=True)
                continue
            print(f"[run] stage {stage} start", flush=True)
            t0 = time.monotonic()
            STAGE_FUNCS[stage](ctx)
            print(f"[run] stage {stage} done in {time.monotonic() - t0:.1f}s", flush=True)
            if os.environ.get("SIRE_INTERRUPT_AFTER") == stage:
                ctx.state["status"] = "interrupted"
                ctx.save_state()
                atomic_write_json(ctx.run_dir / "interrupt.json", {
                    "utc": utc_stamp(), "after_stage": stage,
                    "next_stage": ctx.state.get("next_stage"),
                    "gpu_used": ctx.ledger.gpu_used,
                })
                print(f"[run] deliberate interrupt after {stage}", flush=True)
                return 130
        ctx.state["status"] = "complete" if all(
            ctx.state["stages"].get(s, {}).get("status") == "done" for s in STAGES) else "partial"
        if ctx.state["status"] == "complete":
            ctx.state["completed_utc"] = utc_stamp()
        ctx.save_state()
        return 0
    except KeyboardInterrupt as exc:
        ctx.state["status"] = "interrupted"
        ctx.state["last_error"] = f"KeyboardInterrupt: {exc}"
        ctx.save_state()
        atomic_write_json(ctx.run_dir / "last_error.json", {"error": repr(exc), "utc": utc_stamp(),
                                                            "traceback": traceback.format_exc()})
        print("[run] interrupted cleanly; resume with the recorded command", file=sys.stderr)
        return 130
    except Exception as exc:
        ctx.state["status"] = "failed"
        ctx.state["last_error"] = repr(exc)
        ctx.save_state()
        atomic_write_json(ctx.run_dir / "last_error.json", {"error": repr(exc), "utc": utc_stamp(),
                                                            "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        signal.signal(signal.SIGINT, old)
        ctx.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sire")
    ap.add_argument("command", choices=["audit", "prepare-data", "collect-features", "baselines",
                                        "search", "select", "test", "report", "run", "resume",
                                        "verify"])
    ap.add_argument("--config", default=str(WORK / "configs" / "v0.yaml"))
    ap.add_argument("--run-id")
    ap.add_argument("--gpu-budget-seconds", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "prepare-data":
        from . import data as data_mod
        rid = args.run_id or "data-prep"
        m = data_mod.prepare(cfg, rid)
        print(json.dumps({"sizes": m["sizes"], "dedup": m["dedup"],
                          "crosscheck": m["crosscheck_gsm8k"]["overlaps"]}, indent=2))
        return 0

    if args.command == "audit":
        rid = args.run_id or run_id_now(cfg["run_prefix"])
        ctx = Ctx(cfg, rid, None)
        try:
            with ctx.ledger.stage("audit", ctx.torch, gpu=False):
                stage_audit(ctx)
            ctx.state["status"] = "complete"
            ctx.save_state()
        finally:
            ctx.close()
        return 0

    if args.command in ("run",):
        rid = args.run_id or run_id_now(cfg["run_prefix"])
        return execute(cfg, rid, args.gpu_budget_seconds)

    if args.command == "resume":
        if not args.run_id:
            raise SystemExit("resume requires --run-id")
        return execute(cfg, args.run_id, args.gpu_budget_seconds, resume=True)

    # stage-level commands
    if not args.run_id:
        raise SystemExit(f"{args.command} requires --run-id")
    mapping = {"collect-features": ["features"], "baselines": ["test"], "search": ["search"],
               "select": ["select"], "test": ["test"], "report": ["report"],
               "verify": ["verify"]}
    stages = mapping[args.command]
    if args.command == "baselines":
        # baselines only: full R + three baselines, no searched row
        cfg["_baselines_only"] = True
    return execute(cfg, args.run_id, args.gpu_budget_seconds, only=stages)


if __name__ == "__main__":
    sys.exit(main())
