#!/usr/bin/env python3
"""Start an unbounded v1 symbolic search on a GSM8K-derived search split.

Search/selection use GSM8K **train**, so the GSM8K test split and MMLU remain
untouched for the final external test. The chosen evaluation budget is
max_new_tokens=4096; at 2048 the full-R baseline was severely truncated.
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

from sire import external, search  # noqa: E402
from sire.masking import FeatureStore, ModelRefs  # noqa: E402
from sire.util import (BudgetLedger, GpuMonitor, atomic_write_json, configure_gpu4,  # noqa: E402
                       import_torch_checked, load_config, nvidia_smi_uuid, read_json)

SEED = 20261020


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--run-id", default="v1-gsm8k-search")
    ap.add_argument("--source-run", default="v0-20261021-formal")
    ap.add_argument("--search-n", type=int, default=12)
    ap.add_argument("--selection-n", type=int, default=16)
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--gpu-index", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"]["physical_index"] = args.gpu_index
    cfg["device"]["expected_uuid"] = nvidia_smi_uuid(args.gpu_index)
    cfg["generation"]["max_new_tokens"] = args.max_new
    cfg["curve"]["anchors"] = [0.3]
    cfg["curve"]["rhos"] = [0.1, 0.2, 0.3, 1.0]
    cfg["search"]["min_new_candidates"] = 8
    cfg["search"]["max_new_candidates"] = args.max_candidates

    gpu_ev = configure_gpu4(cfg)
    torch, t_ev = import_torch_checked()
    run_dir = Path(cfg["paths"]["runs"]) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    src_features = Path(cfg["paths"]["runs"]) / args.source_run / "features"
    if not (run_dir / "features").exists():
        os.symlink(src_features.resolve(), run_dir / "features")
    ledger = BudgetLedger(run_dir / "budget.json", 10**9, 0.0)
    monitor = GpuMonitor(run_dir / "gpu_monitor.jsonl", gpu_ev["uuid"], interval=5.0)
    monitor.start()
    print(f"[v1] GPU {args.gpu_index} {gpu_ev['uuid']}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["reasoning_model"],
                                               local_files_only=True)
    store = None
    t0 = time.monotonic()
    try:
        with ledger.stage("model_load", torch):
            model_refs = ModelRefs(cfg)
            store = FeatureStore(run_dir / "features",
                                 Path(cfg["paths"]["base_model"]),
                                 Path(cfg["paths"]["reasoning_model"]))
        train = external.load_gsm8k(Path("/home/nxq/data/KEPT/datasets/gsm8k_train.parquet"),
                                    args.search_n + args.selection_n, SEED)
        if len(train) < args.search_n + args.selection_n:
            raise RuntimeError("not enough GSM8K train items")
        search_items = train[:args.search_n]
        selection_items = train[args.search_n:args.search_n + args.selection_n]
        view = {"splits": {"search": search_items, "selection": selection_items}}
        atomic_write_json(run_dir / "v1_splits.json", {
            "source": "gsm8k_train", "search": search_items, "selection": selection_items,
            "max_new_tokens": args.max_new, "anchors": cfg["curve"]["anchors"],
            "note": "GSM8K train only; GSM8K test and MMLU are held out for final external test.",
        })
        t_search = time.monotonic()
        res = search.run_search(model_refs, store, tokenizer, cfg, run_dir, args.run_id,
                                view, None, 0.0, lambda: 10**9)
        print(f"[v1] search done: {res['n_new']} new masks in {time.monotonic()-t_search:.0f}s",
              flush=True)
        freeze = search.run_selection(model_refs, store, tokenizer, cfg, run_dir,
                                      args.run_id, view, res, None)
        atomic_write_json(run_dir / "v1_state.json", {
            "status": "search_and_selection_complete",
            "run_id": args.run_id, "source_run": args.source_run,
            "search_n": args.search_n, "selection_n": args.selection_n,
            "max_new_tokens": args.max_new, "max_candidates": args.max_candidates,
            "n_new": res["n_new"], "frozen": freeze["formula_infix"],
            "frozen_at": freeze["frozen_at"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime()),
        })
        print(f"[v1] frozen: {freeze['formula_infix']} ({freeze['complexity']})", flush=True)
    finally:
        monitor.stop()
        if store:
            store.close()
        print(f"[v1] wall {time.monotonic()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
