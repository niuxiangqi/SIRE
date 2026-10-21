#!/usr/bin/env python3
"""Fetch the reference base (B) and reasoning (R) checkpoints.

Model weights are never committed. The defaults are the checkpoints audited in
GOAL.md; if you already have them locally, set the paths in configs/v0.yaml and
do not run this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sire.util import load_config

MODELS = [
    ("Qwen/Qwen2.5-7B", "base_model"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "reasoning_model"),
]


def complete(dest: Path) -> bool:
    return (dest / "config.json").exists() and any(dest.glob("*.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    for repo, key in MODELS:
        dest = Path(cfg["paths"][key])
        if complete(dest) and not args.force:
            print(f"[skip] {repo}: checkpoint already present at {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {repo} -> {dest} via {args.source}", flush=True)
        if args.source == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(repo, local_dir=str(dest))
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=str(dest))
        if not complete(dest):
            print(f"[error] no safetensors/config found under {dest}", file=sys.stderr)
            return 1
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
