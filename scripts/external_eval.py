#!/usr/bin/env python3
"""Post-hoc external evaluation on GSM8K (pass@k) and MMLU subsets.

Usage:
  python scripts/external_eval.py --run-id v0-20261021-formal \
      --methods full_R searched --rhos 0.05 0.3 --gsm-n 8 --mmlu-per 4

This does not touch the frozen v0 budget ledger or the pre-registered test
results. It writes to runs/<id>/external/ and uses its own budget file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from sire import dsl, eval as em, external  # noqa: E402
from sire.masking import FeatureStore, ModelRefs  # noqa: E402
from sire.util import (BudgetLedger, GpuMonitor, atomic_write_json, configure_gpu4,  # noqa: E402
                       import_torch_checked, load_config, nvidia_smi_uuid, read_json,
                       read_jsonl, sha256_text, utc_stamp)

SUBJECTS_DEFAULT = ["anatomy", "college_biology", "college_medicine",
                    "professional_medicine"]


def _summ(records, n_items, k, dataset, method, fid, rho):
    by = {}
    for r in records:
        by.setdefault(r["base_id"], []).append(r)
    passk = 0
    pass1 = 0
    for base, rs in by.items():
        rs = sorted(rs, key=lambda x: x["sample_index"])
        if any(x["correct"] for x in rs):
            passk += 1
        if rs[0]["correct"]:
            pass1 += 1
    n = len(by)
    return {"dataset": dataset, "method": method, "formula_id": fid, "rho": rho,
            "pass_k": k, "n_items": n, "n_samples": len(records),
            "pass_at_k": passk / n if n else None, "pass_at_1": pass1 / n if n else None,
            "mean_new_tokens": sum(r["n_new_tokens"] for r in records) / max(1, len(records)),
            "truncation_rate": sum(x["truncated"] for x in records) / max(1, len(records)),
            "utc": utc_stamp()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--run-id", default="v0-20261021-formal")
    ap.add_argument("--methods", nargs="+", default=["full_R", "searched"],
                    choices=["full_R", "searched", "magnitude_A", "low_grad_NegU", "random"])
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.05, 0.3])
    ap.add_argument("--gsm-n", type=int, default=8)
    ap.add_argument("--mmlu-per", type=int, default=4)
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS_DEFAULT)
    ap.add_argument("--pass-k-gsm", type=int, default=4)
    ap.add_argument("--pass-k-mmlu", type=int, default=1)
    ap.add_argument("--budget-seconds", type=float, default=1800.0)
    ap.add_argument("--gpu-index", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Explicit per-invocation GPU selection; the user authorized 0/1/2 until
    # 2026-10-28 06:00 Beijing time.
    cfg["device"]["physical_index"] = args.gpu_index
    cfg["device"]["expected_uuid"] = nvidia_smi_uuid(args.gpu_index)
    gpu_ev = configure_gpu4(cfg)
    torch, t_ev = import_torch_checked()
    print(f"[external] GPU {args.gpu_index} {gpu_ev['uuid']} {t_ev['device_name']}", flush=True)

    run_dir = Path(cfg["paths"]["runs"]) / args.run_id
    ext_dir = run_dir / "external"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(ext_dir / "budget.json", args.budget_seconds, 0.0)
    monitor = GpuMonitor(ext_dir / "gpu_monitor.jsonl", gpu_ev["uuid"], interval=5.0)
    monitor.start()
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["reasoning_model"],
                                               local_files_only=True)
    store = None
    try:
        with ledger.stage("model_load", torch):
            model_refs = ModelRefs(cfg)
            store = FeatureStore(run_dir / "features",
                                 Path(cfg["paths"]["base_model"]),
                                 Path(cfg["paths"]["reasoning_model"]))
        split_path = ext_dir / "splits.json"
        if split_path.exists():
            split = read_json(split_path)
        else:
            split = external.prepare(Path("/home/nxq/data/KEPT/datasets/gsm8k_test.parquet"),
                                     Path("/home/nxq/data/KEPT/datasets"),
                                     args.gsm_n, args.subjects, args.mmlu_per,
                                     cfg["data"]["global_seed"])
            atomic_write_json(split_path, split)
        print(f"[external] GSM8K {split['gsm8k']['n']} MMLU {split['mmlu']['n']}", flush=True)

        freeze = read_json(run_dir / "freeze.json")
        formula = dsl.parse_sexpr(freeze["formula_ast"])
        fid = freeze["candidate_id"]

        for method in args.methods:
            rhos = [1.0] if method == "full_R" else list(args.rhos)
            for rho in rhos:
                tag = f"{method}__{fid if method == 'searched' else method}__rho{rho:.4f}"
                print(f"[external] {tag}", flush=True)
                with ledger.stage(f"mask_{method}_{rho}", torch):
                    if method == "full_R":
                        audit = model_refs.apply_method(store, "full_R", 1.0, cfg)
                    elif method == "searched":
                        audit = model_refs.apply(store, expr=formula, rho=rho)
                    else:
                        audit = model_refs.apply_method(store, method, rho, cfg)
                for ds_name, ds, k in (("gsm8k", split["gsm8k"]["items"], args.pass_k_gsm),
                                       ("mmlu", split["mmlu"]["items"], args.pass_k_mmlu)):
                    outdir = ext_dir / f"{ds_name}__{tag}"
                    outdir.mkdir(parents=True, exist_ok=True)
                    rec_path = outdir / "samples.jsonl"
                    if rec_path.exists():
                        print(f"[external] skip existing {outdir.name}", flush=True)
                        continue
                    sample_items = external.make_sample_items(ds, k)
                    records = []
                    t0 = time.monotonic()
                    with ledger.stage(f"gen_{ds_name}_{method}_{rho}", torch):
                        for i in range(0, len(sample_items), int(cfg["generation"]["batch_size"])):
                            batch = sample_items[i:i + int(cfg["generation"]["batch_size"])]
                            gens = em.generate_batch(model_refs.model, tokenizer, batch,
                                                     cfg["generation"], cfg["data"]["global_seed"])
                            for it, g in zip(batch, gens):
                                ok, pred = external.check(g["text"], it["gt"], it["source"])
                                records.append({"dataset": ds_name, "item_id": it["base_id"],
                                                "sample_index": it["sample_index"],
                                                "base_id": it["base_id"],
                                                "correct": bool(ok), "parsed": pred,
                                                "gt": it["gt"], "n_new_tokens": g["n_new_tokens"],
                                                "truncated": g["truncated"], "text": g["text"]})
                    with open(rec_path, "w") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    summary = _summ(records, len(ds), k, ds_name, method, fid, rho)
                    summary["mask_sha256"] = audit["mask_sha256"]
                    summary["seconds"] = time.monotonic() - t0
                    atomic_write_json(outdir / "summary.json", summary)
                    print(f"[external]   {ds_name}: pass@{k}={summary['pass_at_k']:.3f} "
                          f"pass@1={summary['pass_at_1']:.3f} trunc={summary['truncation_rate']:.3f}",
                          flush=True)
    finally:
        monitor.stop()
        try:
            store.close()
        except Exception:
            pass
        print(f"[external] ledger used {ledger.gpu_used:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
