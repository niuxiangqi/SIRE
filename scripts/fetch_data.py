#!/usr/bin/env python3
"""Fetch the two frozen datasets into the raw data root.

No dataset rows are embedded in this repository. This helper only downloads the
upstream artifacts; `prepare-data` performs filtering, dedup and split freezing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sire.util import load_config

DATASETS = {
    "NuminaMath-CoT": "AI-MO/NuminaMath-CoT",
    "no_robots": "HuggingFaceH4/no_robots",
}


def already_there(dest: Path) -> bool:
    return dest.exists() and any(dest.rglob("*.parquet"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw_data"])
    raw.mkdir(parents=True, exist_ok=True)

    for local, repo in DATASETS.items():
        dest = raw / local
        if already_there(dest) and not args.force:
            print(f"[skip] {local}: parquet already present at {dest}")
            continue
        print(f"[fetch] {repo} -> {dest} via {args.source}", flush=True)
        if args.source == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(repo, repo_type="dataset", local_dir=str(dest))
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, repo_type="dataset", local_dir=str(dest))
        if not already_there(dest):
            print(f"[error] no parquet found under {dest}", file=sys.stderr)
            return 1
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
