"""Model/tokenizer audit and environment evidence (part of C1)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .util import atomic_write_json, atomic_write_text, sha256_file, utc_stamp


def _config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


def _index(path: Path) -> dict:
    return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]


def run_audit(cfg: dict, run_dir: Path, gpu_evidence: dict, torch_evidence: dict) -> dict:
    from transformers import AutoTokenizer

    base = Path(cfg["paths"]["base_model"])
    reas = Path(cfg["paths"]["reasoning_model"])
    cb, cr = _config(base), _config(reas)
    keys = sorted(set(cb) | set(cr))
    config_diff = {k: {"B": cb.get(k), "R": cr.get(k)} for k in keys if cb.get(k) != cr.get(k)}
    ib, ir = _index(base), _index(reas)
    names_b, names_r = set(ib), set(ir)
    missing = sorted(names_r - names_b)
    extra = sorted(names_b - names_r)

    from safetensors import safe_open

    shape_mismatch = []
    dtype_mismatch = []
    handles = {}
    for n in sorted(names_b & names_r):
        for tag, root, idx in (("B", base, ib), ("R", reas, ir)):
            key = (str(root), idx[n])
            if key not in handles:
                handles[key] = safe_open(str(root / idx[n]), framework="pt", device="cpu")
        tb = handles[(str(base), ib[n])].get_slice(n)
        tr = handles[(str(reas), ir[n])].get_slice(n)
        if tuple(tb.get_shape()) != tuple(tr.get_shape()):
            shape_mismatch.append(n)
        if str(tb.get_dtype()) != str(tr.get_dtype()):
            dtype_mismatch.append(n)

    tokb = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokr = AutoTokenizer.from_pretrained(reas, local_files_only=True)
    vb, vr = tokb.get_vocab(), tokr.get_vocab()
    only_b = sorted(set(vb) - set(vr))[:50]
    only_r = sorted(set(vr) - set(vb))[:50]
    diff_id = sorted([{"token": t, "B": vb[t], "R": vr[t]} for t in set(vb) & set(vr)
                      if vb[t] != vr[t]], key=lambda x: x["B"])[:100]

    files = {}
    for p in sorted(base.iterdir()) + sorted(reas.iterdir()):
        if p.is_file() and p.suffix in (".json", ".txt", ".safetensors") and p.stat().st_size < 3e6:
            files[str(p)] = {"size": p.stat().st_size, "sha256": sha256_file(p)}

    audit = {
        "created_utc": utc_stamp(),
        "base": str(base), "reasoning": str(reas),
        "architecture": {"B": cb.get("architectures"), "R": cr.get("architectures"),
                         "model_type": {"B": cb.get("model_type"), "R": cr.get("model_type")}},
        "config_diff": config_diff,
        "config_used": "R",
        "parameter_names": {"n_B": len(names_b), "n_R": len(names_r),
                            "missing_in_B": missing, "extra_in_B": extra},
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "tokenizer": {
            "B_vocab_size": len(vb), "R_vocab_size": len(vr),
            "B_special": {k: v for k, v in tokb.special_tokens_map.items()},
            "R_special": {k: v for k, v in tokr.special_tokens_map.items()},
            "special_ids": {"B": {k: tokb.convert_tokens_to_ids(v) if isinstance(v, str) else v
                                   for k, v in tokb.special_tokens_map.items()},
                            "R": {k: tokr.convert_tokens_to_ids(v) if isinstance(v, str) else v
                                  for k, v in tokr.special_tokens_map.items()}},
            "only_in_B_sample": only_b, "only_in_R_sample": only_r,
            "same_token_different_id_sample": diff_id,
        },
        "file_hashes": files,
        "gpu": gpu_evidence,
        "torch": torch_evidence,
        "notes": [
            "All runs use R's config, tokenizer and chat template.",
            "R1-Distill-Qwen-7B is officially distilled from Qwen2.5-Math-7B; D against Qwen2.5-7B is a difference against this chosen reference, not purely the distillation effect.",
            "config_diff records rope_theta and sliding_window differences; no shape is truncated or padded.",
        ],
    }
    out = Path(run_dir) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "model_audit.json", audit)
    atomic_write_json(Path(run_dir) / "env.json", {
        "created_utc": utc_stamp(),
        "sys_executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX") or str(Path(sys.executable).parents[1]),
        "python": sys.version,
        "gpu": gpu_evidence,
        "torch": torch_evidence,
    })
    lines = ["# Model audit", "", f"- Base: `{base}`", f"- Reasoning: `{reas}`",
             f"- Config used: R", "",
             "## Config differences", "```json",
             json.dumps(config_diff, indent=2), "```", "",
             "## Parameter compatibility",
             f"- B tensors: {len(names_b)}; R tensors: {len(names_r)}",
             f"- Missing in B: {len(missing)}; extra in B: {len(extra)}",
             f"- Shape mismatches: {len(shape_mismatch)}; dtype mismatches: {len(dtype_mismatch)}",
             "", "## Tokenizer",
             f"- vocab: B={len(vb)}, R={len(vr)}",
             f"- same token different id: {len(diff_id)} (sample)",
             f"- only in B: {len(only_b)} sample; only in R: {len(only_r)} sample",
             "", "See model_audit.json for full detail and hashes."]
    atomic_write_text(out / "MODEL_AUDIT.md", "\n".join(lines) + "\n")
    return audit
