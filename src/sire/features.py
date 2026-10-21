"""Feature collection: per-tensor D, A, G, U, B, R with fixed RMS scales.

G = mean_z g(z) and U = mean_z |g(z)| are computed on the full R model with
autograd, per-sample forward/backward, configurable FP32/BF16 parameters and FP32 accumulation. Statistics can be
kept in RAM (legacy profile) or transactionally accumulated on disk per tensor.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from . import data as data_mod
from .masking import specs_with_shapes
from .util import atomic_write_json, read_json


def _rms(t) -> float:
    import torch

    v = float(torch.sum(t.double().square()).item() / t.numel())
    return math.sqrt(v)


def compute_delta(cfg: dict, run_dir: Path, specs: list[dict]) -> dict:
    """D = R - B in FP32; store FP16 plus exact RMS scales of every raw leaf."""
    import torch
    from safetensors import safe_open

    feat = run_dir / "features"
    feat.mkdir(parents=True, exist_ok=True)
    base = Path(cfg["paths"]["base_model"])
    reas = Path(cfg["paths"]["reasoning_model"])
    bmap = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    rmap = json.loads((reas / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    rms = {}
    index = {s["name"]: i for i, s in enumerate(specs)}
    t0 = time.monotonic()
    for i, spec in enumerate(specs):
        name = spec["name"]
        key = (str(base), bmap[name])
        if key not in handles:
            handles[key] = safe_open(str(base / bmap[name]), framework="pt", device="cpu")
        b = handles[key].get_tensor(name)
        key = (str(reas), rmap[name])
        if key not in handles:
            handles[key] = safe_open(str(reas / rmap[name]), framework="pt", device="cpu")
        r = handles[key].get_tensor(name)
        d = r.to(torch.float32) - b.to(torch.float32)
        if not torch.isfinite(d).all():
            raise RuntimeError(f"non-finite D in {name}")
        np.save(feat / f"{i:04d}.D.npy", d.half().numpy())
        entry = {
            "D": _rms(d),
            "A": _rms(torch.abs(d)),
            "B": _rms(b.to(torch.float32)),
            "R": _rms(r.to(torch.float32)),
        }
        if entry["D"] == 0:
            raise RuntimeError(f"zero delta in {name}")
        rms[name] = entry
        del b, r, d
        if (i + 1) % 28 == 0:
            print(f"[features] delta {i + 1}/{len(specs)} tensors, {time.monotonic() - t0:.1f}s",
                  flush=True)
    return {"index": index, "rms": rms, "delta_seconds": time.monotonic() - t0}


def _calibration_examples(cfg: dict, split_manifest: dict):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["paths"]["reasoning_model"], local_files_only=True)
    items = split_manifest["splits"]["calibration"][: cfg["features"]["calibration_samples"]]
    out = []
    for it in items:
        ex = data_mod.tokenize_chat_example(tok, it["problem"], it["reference_solution"],
                                            cfg["features"]["max_seq_len"])
        if ex is None:
            raise RuntimeError(f"calibration item too long at feature time: {it['item_id']}")
        out.append({"item_id": it["item_id"], **ex})
    return out


def collect_gradients(cfg: dict, run_dir: Path, split_manifest: dict, specs: list[dict],
                      index: dict, base_rms: dict) -> dict:
    """Accumulate G and U over calibration samples, checkpointing every 4."""
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    feat = run_dir / "features"
    examples = _calibration_examples(cfg, split_manifest)
    n = len(examples)
    param_dtype = cfg["features"].get("param_dtype", "float32")
    if param_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"unsupported gradient dtype: {param_dtype}")
    if param_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU/runtime does not support BF16")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["paths"]["reasoning_model"], dtype=getattr(torch, param_dtype), device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    if cfg["features"]["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.eval()
    params = {}
    for p in model.parameters():
        p.requires_grad_(False)
    for spec in specs:
        p = model.get_parameter(spec["name"])
        p.requires_grad_(True)
        params[spec["name"]] = p

    disk_stats = cfg["features"].get("accumulator", "memory") == "disk"
    accum = None
    if disk_stats:
        from .accumulation import TensorAccumulators
        from .util import sha256_file
        identity = {"specs": specs, "samples": [e["item_id"] for e in examples],
                    "param_dtype": param_dtype, "max_seq_len": cfg["features"]["max_seq_len"],
                    "checkpointing": cfg["features"]["grad_checkpointing"],
                    "split_sha256": sha256_file(Path(cfg["paths"]["data_root"]) / "prepared" / "splits.json"),
                    "model": str(Path(cfg["paths"]["reasoning_model"]).resolve())}
        accum = TensorAccumulators(feat / "accumulators", specs, identity)
        g_sum, u_sum = {}, {}
    else:
        g_sum = {s["name"]: torch.zeros(s["numel"], dtype=torch.float32) for s in specs}
        u_sum = {s["name"]: torch.zeros(s["numel"], dtype=torch.float32) for s in specs}
    cursor = 0
    partial = feat / "accum_partial.pt"
    if not disk_stats and partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=False)
        if state.get("n_items") != n:
            raise RuntimeError("partial gradient state has different calibration count")
        for name in g_sum:
            g_sum[name] = state["g_sum"][name]
            u_sum[name] = state["u_sum"][name]
        cursor = int(state["cursor"])
        print(f"[features] resumed gradients at sample {cursor}/{n}", flush=True)

    if disk_stats:
        cursor = accum.cursor()
        if cursor > n:
            raise ValueError("gradient cursor exceeds calibration sample count")
        print(f"[features] disk statistics cursor {cursor}/{n}", flush=True)
    t0 = time.monotonic()
    gpu_compute = 0.0
    for si in range(cursor, n):
        ex = examples[si]
        ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([ex["labels"]], dtype=torch.long, device="cuda")
        attn = torch.ones_like(ids)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()
        ev1.record()
        torch.cuda.synchronize()
        gpu_compute += ev0.elapsed_time(ev1) / 1000.0
        for spec in specs:
            name = spec["name"]
            g = params[name].grad
            if g is None:
                raise RuntimeError(f"missing gradient for {name}")
            gf = g.detach().to(torch.float32).reshape(-1).cpu()
            if disk_stats:
                accum.add(name, si, gf)
            else:
                g_sum[name] += gf
                u_sum[name] += gf.abs()
            params[name].grad = None
            del gf, g
        del out, ids, labels, attn
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[features] grad sample {si + 1}/{n} ({time.monotonic() - t0:.1f}s)", flush=True)
        if not disk_stats and ((si + 1) % 4 == 0 or si + 1 == n):
            torch.save({"cursor": si + 1, "n_items": n, "g_sum": g_sum, "u_sum": u_sum},
                       partial)

    grad_seconds = time.monotonic() - t0
    del model
    gc.collect()
    torch.cuda.empty_cache()

    rms = {}
    for name, vals in base_rms.items():
        rms[name] = dict(vals)
    # Recompute delta RMS if this is a fresh directory (meta written later).
    for spec in specs:
        name, numel = spec["name"], spec["numel"]
        if disk_stats:
            G, U = accum.means(name, n)
        else:
            G = g_sum[name] / n
            U = u_sum[name] / n
        if not torch.isfinite(G).all() or not torch.isfinite(U).all():
            raise RuntimeError(f"non-finite gradient statistic in {name}")
        np.save(feat / f"{index[name]:04d}.G.npy", G.half().numpy())
        np.save(feat / f"{index[name]:04d}.U.npy", U.half().numpy())
        rms.setdefault(name, {})["G"] = _rms(G)
        rms[name]["U"] = _rms(U)
        rms[name]["G_abs_mean"] = float(G.abs().mean())
        rms[name]["U_mean"] = float(U.mean())
        del G, U
    # A and U are distinct: Mean(Abs(g)) != Abs(Mean(g)).
    diffs = [abs(rms[s["name"]]["U_mean"] - rms[s["name"]]["G_abs_mean"]) for s in specs]
    relative = [d / max(1e-30, rms[s["name"]]["U_mean"]) for d, s in zip(diffs, specs)]
    if max(relative) < 1e-4:
        raise RuntimeError("U and |G| appear identical; gradient statistics are wrong")
    partial.unlink(missing_ok=True)
    return {"grad_seconds": grad_seconds, "gpu_compute_seconds": gpu_compute,
            "samples": n, "rms": rms,
            "u_vs_abs_g_max_relative_diff": max(relative),
            "u_vs_abs_g_mean_relative_diff": sum(relative) / len(relative),
            "checkpointing": bool(cfg["features"]["grad_checkpointing"]),
            "param_dtype": param_dtype, "tf32": True}


def collect(cfg: dict, run_id: str, run_dir: Path) -> dict:
    specs = specs_with_shapes(Path(cfg["paths"]["base_model"]),
                              Path(cfg["paths"]["reasoning_model"]))
    split_manifest = read_json(Path(cfg["paths"]["data_root"]) / "prepared" / "splits.json")
    if split_manifest is None:
        raise RuntimeError("splits.json missing; run prepare-data first")
    if split_manifest["generation_config"] != cfg["generation"]:
        raise RuntimeError("generation config drifted from frozen data manifest")
    t0 = time.monotonic()
    delta = compute_delta(cfg, run_dir, specs)
    grad = collect_gradients(cfg, run_dir, split_manifest, specs, delta["index"], delta["rms"])
    meta = {
        "run_id": run_id,
        "specs": specs,
        "index": delta["index"],
        "rms": grad["rms"],
        "calibration": [it["item_id"] for it in
                        split_manifest["splits"]["calibration"][: cfg["features"]["calibration_samples"]]],
        "storage_dtype": "float16",
        "stat_dtype": "float32",
        "delta_seconds": delta["delta_seconds"],
        "grad_seconds": grad["grad_seconds"],
        "samples": grad["samples"],
        "checkpointing": grad["checkpointing"],
        "param_dtype": grad["param_dtype"],
        "tf32": grad["tf32"],
        "u_vs_abs_g_max_relative_diff": grad["u_vs_abs_g_max_relative_diff"],
        "u_vs_abs_g_mean_relative_diff": grad["u_vs_abs_g_mean_relative_diff"],
        "wall_seconds": time.monotonic() - t0,
        "spec_sha256": __import__("hashlib").sha256(
            json.dumps(specs, sort_keys=True).encode()).hexdigest(),
    }
    atomic_write_json(Path(run_dir) / "features" / "meta.json", meta)
    return meta
