"""Batched generation with per-item seeds, answer checking, evaluation records.

One decoding setup for every method. A generation error, timeout, parse failure
or truncation counts as wrong and stays in the denominator.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from . import answers, data as data_mod, dsl
from .util import append_jsonl, atomic_write_json, read_json, read_jsonl, sha256_text


def item_seed(global_seed: int, item_id: str) -> int:
    h = int.from_bytes(hashlib.sha256(f"{global_seed}|{item_id}".encode()).digest()[:8], "big")
    return h & ((1 << 63) - 1)


def _sample_rows(logits, generators, temperature: float, top_p: float):
    """Per-row categorical sampling with the row's own generator."""
    out = []
    for i in range(logits.shape[0]):
        l = logits[i].float() / max(temperature, 1e-6)
        probs = torch.softmax(l, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = (cum - sp) < top_p
        keep[0] = True
        sp = sp[keep]
        si = si[keep]
        sp = sp / sp.sum()
        j = torch.multinomial(sp, 1, generator=generators[i])
        out.append(int(si[j].item()))
    return torch.tensor(out, dtype=torch.long, device=logits.device)


@torch.inference_mode()
def generate_batch(model, tok, batch_items, gen_cfg, global_seed):
    """Left-padded batch generation with independent per-item RNG streams."""
    from transformers import DynamicCache

    texts = [it["prompt"] for it in batch_items]
    enc = [tok(t, add_special_tokens=False).input_ids for t in texts]
    B = len(enc)
    L = max(len(e) for e in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((B, L), pad, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, L - len(e):] = torch.tensor(e)
        mask[i, L - len(e):] = 1
    ids = ids.cuda()
    mask = mask.cuda()
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    gens = [torch.Generator(device="cuda").manual_seed(item_seed(global_seed, it["item_id"]))
            for it in batch_items]
    stop = set(int(x) for x in gen_cfg["stop_ids"])
    max_new = int(gen_cfg["max_new_tokens"])
    generated = [[] for _ in range(B)]
    finished = [False] * B
    stopped = [False] * B
    cache = DynamicCache()
    t0 = time.monotonic()
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1]
    del out
    newpos = mask.sum(-1, keepdim=True)
    tok_now = _sample_rows(logits, gens, gen_cfg["temperature"], gen_cfg["top_p"])
    for step in range(max_new):
        tok_list = tok_now.tolist()
        for i, t in enumerate(tok_list):
            if not finished[i]:
                generated[i].append(t)
                if t in stop:
                    finished[i] = True
                    stopped[i] = True
                elif len(generated[i]) >= max_new:
                    finished[i] = True
        if all(finished):
            break
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device="cuda")], dim=1)
        out = model(input_ids=tok_now[:, None], attention_mask=mask, position_ids=newpos,
                    past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1]
        del out
        tok_now = _sample_rows(logits, gens, gen_cfg["temperature"], gen_cfg["top_p"])
        newpos = newpos + 1
    wall = time.monotonic() - t0
    results = []
    for i in range(B):
        toks = generated[i]
        # trim at first stop token
        cut = len(toks)
        for j, t in enumerate(toks):
            if t in stop:
                cut = j
                break
        text = tok.decode(toks[:cut], skip_special_tokens=True)
        results.append({
            "text": text,
            "n_new_tokens": len(toks[:cut]),
            "truncated": bool(not stopped[i] or len(toks) >= max_new),
            "seconds": wall / B,
        })
    del cache
    torch.cuda.empty_cache()
    return results


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def monitor_nll(model, tok, monitor_items, max_len=1024):
    """Mean assistant-token NLL on frozen No Robots samples (not a v0 gate)."""
    if not monitor_items:
        return None
    vals = []
    for it in monitor_items:
        try:
            ex = data_mod.tokenize_chat_example(tok, it["problem"], it["reference"], max_len)
        except Exception:
            ex = None
        if ex is None:
            continue
        ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([ex["labels"]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                        use_cache=False)
        vals.append(float(out.loss))
        del out, ids, labels
    torch.cuda.empty_cache()
    return sum(vals) / len(vals) if vals else None


def eval_dir(run_dir: Path, method: str, formula_id: str, rho: float, split: str) -> Path:
    tag = f"{method}__{formula_id}__rho{rho:.4f}__{split}"
    return Path(run_dir) / "evals" / tag


def evaluate_rows(model_refs, store, tok, cfg, run_dir, method: str, formula,
                  formula_id: str, rho: float, items: list[dict], split: str,
                  mask_audit: dict | None = None, nll_full: float | None = None,
                  compute_nll: bool = True) -> dict:
    """Apply a mask, generate on all items, persist every sample, return summary."""
    outdir = eval_dir(run_dir, method, formula_id, rho, split)
    outdir.mkdir(parents=True, exist_ok=True)
    records_path = outdir / "samples.jsonl"
    existing = {r["item_id"]: r for r in read_jsonl(records_path)}
    if len(existing) >= len(items) and all(it["item_id"] in existing for it in items):
        cached_summary = read_json(outdir / "summary.json")
        if cached_summary:
            return cached_summary
    if mask_audit is None:
        if method in ("full_R", "magnitude_A", "low_grad_NegU", "random"):
            mask_audit = model_refs.apply_method(store, method, rho, cfg)
        else:
            mask_audit = model_refs.apply(store, expr=formula, rho=rho)
        mask_audit["formula_id"] = formula_id
        atomic_write_json(outdir / "mask_audit.json", mask_audit)
    if model_refs is not None and mask_audit.get("total_kept", 0) != sum(
            int(math.floor(rho * s["numel"])) for s in model_refs.specs):
        raise RuntimeError("mask quota does not match rho")
    todo = [it for it in items if it["item_id"] not in existing]
    t0 = time.monotonic()
    for batch in _batches(todo, int(cfg["generation"]["batch_size"])):
        try:
            gens = generate_batch(model_refs.model, tok, batch, cfg["generation"],
                                  cfg["data"]["global_seed"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gens = []
            for it in batch:  # fall back to one item at a time
                gens.extend(generate_batch(model_refs.model, tok, [it], cfg["generation"],
                                           cfg["data"]["global_seed"]))
        for it, g in zip(batch, gens):
            correct, pred, gt_parsed = answers.is_correct(g["text"], it["gt_value"])
            rec = {
                "method": method,
                "formula_id": formula_id,
                "rho": rho,
                "split": split,
                "item_id": it["item_id"],
                "seed": item_seed(cfg["data"]["global_seed"], it["item_id"]),
                "gt_raw": it["gt_value"],
                "gt_parsed": gt_parsed,
                "parsed_pred": pred,
                "correct": bool(correct),
                "n_new_tokens": g["n_new_tokens"],
                "truncated": bool(g["truncated"]),
                "seconds": g["seconds"],
                "text": g["text"],
            }
            append_jsonl(records_path, rec)
            existing[it["item_id"]] = rec
    nll = monitor_nll(model_refs.model, tok, cfg.get("_monitor_items", [])) if compute_nll else None
    recs = [existing[it["item_id"]] for it in items]
    n = len(recs)
    n_correct = sum(1 for r in recs if r["correct"])
    summary = {
        "method": method,
        "formula_id": formula_id,
        "formula_ast": None if formula is None else dsl.canonical_str(formula),
        "rho": rho,
        "split": split,
        "n_items": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "mean_new_tokens": sum(r["n_new_tokens"] for r in recs) / n,
        "truncation_rate": sum(1 for r in recs if r["truncated"]) / n,
        "nll": nll,
        "nll_delta": None if (nll is None or nll_full is None) else nll - nll_full,
        "seconds": time.monotonic() - t0,
        "mask_sha256": mask_audit["mask_sha256"],
        "records": str(records_path),
    }
    atomic_write_json(outdir / "summary.json", summary)
    return summary
