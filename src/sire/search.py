"""Single-seed discrete evolutionary search over the six-leaf formula DSL.

Search sees only the search split (and later the selection split). It has no
read path to test content or test results.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from . import dsl, eval as eval_mod
from .util import append_jsonl, atomic_write_json, read_json, read_jsonl, sha256_text

BASELINE_IDS = {"A": "magnitude_A", "Neg(U)": "low_grad_NegU"}


def _eval_index_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cache" / "eval_index.json"


def load_eval_index(run_dir: Path) -> dict:
    return read_json(_eval_index_path(run_dir), default={}) or {}


def _save_eval_index(run_dir: Path, idx: dict) -> None:
    atomic_write_json(_eval_index_path(run_dir), idx)


def _cache_key(mask_hash: str, split: str, rho: float) -> str:
    return f"{mask_hash}|{split}|{rho:.4f}"


def evaluate_candidate(model_refs, store, tok, cfg, run_dir, run_id, items, split,
                       formula, cid, parent_id, origin, rhos, nll_full, eval_index,
                       method="candidate") -> dict:
    import shutil

    anchors = {}
    total_seconds = 0.0
    for rho in rhos:
        audit = model_refs.apply(store, expr=formula, rho=rho)
        key = _cache_key(audit["mask_sha256"], split, rho)
        outdir = eval_mod.eval_dir(run_dir, method, cid, rho, split)
        cached = eval_index.get(key)
        if cached and Path(cached["records"]).exists():
            outdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached["records"], outdir / "samples.jsonl")
            summary = dict(cached)
            summary["method"] = method
            summary["formula_id"] = cid
            summary["records"] = str(outdir / "samples.jsonl")
            summary["seconds"] = 0.0
            atomic_write_json(outdir / "mask_audit.json", audit)
            atomic_write_json(outdir / "summary.json", summary)
        else:
            summary = eval_mod.evaluate_rows(
                model_refs, store, tok, cfg, run_dir, method, formula, cid, rho,
                items, split, mask_audit=audit, nll_full=nll_full, compute_nll=False)
            eval_index[key] = summary
        total_seconds += summary["seconds"]
        anchors[f"{rho:.4f}"] = {
            "accuracy": summary["accuracy"],
            "n_correct": summary["n_correct"],
            "n_items": summary["n_items"],
            "mask_sha256": audit["mask_sha256"],
            "summary": str(outdir / "summary.json"),
            "records": summary["records"],
        }
    J = sum(a["accuracy"] for a in anchors.values()) / len(anchors)
    rec = {
        "candidate_id": cid,
        "formula_ast": dsl.canonical_str(formula),
        "formula_infix": f"s = {dsl.infix(formula)}",
        "complexity": dsl.complexity(formula),
        "depth": dsl.depth(formula),
        "n_ops": dsl.n_ops(formula),
        "parent_id": parent_id,
        "origin": origin,
        "split": split,
        "anchors": anchors,
        "J": J,
        "seconds": total_seconds,
        "evaluated_at": time.time(),
    }
    append_jsonl(Path(run_dir) / "candidates.jsonl", rec)
    return rec


def dominated(a: dict, b: dict) -> bool:
    """a dominates b on (J maximise, complexity minimise)."""
    return (a["J"] >= b["J"] and a["complexity"] <= b["complexity"] and
            (a["J"] > b["J"] or a["complexity"] < b["complexity"]))


def archive_of(records: list[dict]) -> list[dict]:
    valid = [r for r in records if r.get("split") == "search" and "J" in r]
    # dedup by candidate id, keep first
    seen, uniq = set(), []
    for r in valid:
        if r["candidate_id"] not in seen:
            seen.add(r["candidate_id"])
            uniq.append(r)
    return [r for r in uniq if not any(dominated(o, r) for o in uniq if o is not r)]


def _nodes(e):
    return [dsl.canonical_str(n) for n in dsl.walk(e)]


def propose(parent, rng: random.Random, max_ops: int, max_depth: int):
    """One mutation proposal: leaf/operator substitution, subtree in/out, fresh."""
    for _ in range(200):
        mode = rng.choice(["leaf", "op", "insert", "delete", "fresh"])
        try:
            if mode == "fresh":
                cand = dsl.random_tree(rng, max_ops=max_ops, max_depth=max_depth)
            else:
                p = dsl.canonical(parent)
                nodes = _nodes(p)
                leaf_targets = [n for n in nodes if n in dsl.LEAVES]
                op_targets = [n for n in nodes if n not in dsl.LEAVES]
                if mode == "leaf":
                    if not leaf_targets:
                        continue
                    target = rng.choice(leaf_targets)
                    repl = dsl.leaf(rng.choice([x for x in dsl.LEAVES if x != target]
                                               or list(dsl.LEAVES)))
                elif mode == "op":
                    if not op_targets:
                        continue
                    target = rng.choice(op_targets)
                    t = _find(p, target)
                    if t is None or dsl.is_leaf(t):
                        continue
                    pool = [o for o in (dsl.UNARY if t[0] in dsl.UNARY else dsl.BINARY)
                            if o != t[0]]
                    repl = (rng.choice(pool),) + tuple(t[1:])
                elif mode == "insert":
                    if not leaf_targets:
                        continue
                    target = rng.choice(leaf_targets)
                    t = _find(p, target)
                    if t is None or not dsl.is_leaf(t):
                        continue
                    leaf2 = dsl.leaf(rng.choice(dsl.LEAVES))
                    if rng.random() < 0.5:
                        repl = (rng.choice(dsl.UNARY), t)
                    else:
                        repl = (rng.choice(dsl.BINARY), t, leaf2)
                else:  # delete
                    if not op_targets:
                        continue
                    target = rng.choice(op_targets)
                    t = _find(p, target)
                    if t is None or dsl.is_leaf(t):
                        continue
                    repl = dsl.leaf(rng.choice(dsl.LEAVES))
                cand = dsl.replace_subtree(p, target, repl)
            cand = dsl.canonical(cand)
            dsl.validate(cand)
            if dsl.n_ops(cand) < 1:
                continue
            return cand
        except Exception:
            continue
    return None


def _find(e, target: str):
    if dsl.canonical_str(e) == target:
        return e
    for c in dsl.children(e):
        found = _find(c, target)
        if found is not None:
            return found
    return None


def run_search(model_refs, store, tok, cfg, run_dir, run_id, split_manifest,
               nll_full: float | None, reserve_seconds: float, budget_left) -> dict:
    rng = random.Random(int(cfg["search"]["seed"]))
    items = split_manifest["splits"]["search"]
    rhos = [float(x) for x in cfg["curve"]["anchors"]]
    eval_index = load_eval_index(run_dir)
    records = read_jsonl(Path(run_dir) / "candidates.jsonl")
    done = {r["candidate_id"] for r in records}

    # Seed archive
    seeds = []
    for s in cfg["search"]["initial"]:
        f = dsl.parse(s)
        cid = dsl.formula_id(f)
        if cid in done:
            rec = next(r for r in records if r["candidate_id"] == cid)
        else:
            rec = evaluate_candidate(
                model_refs, store, tok, cfg, run_dir, run_id, items, "search", f, cid,
                None, "seed", rhos, nll_full, eval_index)
            _save_eval_index(run_dir, eval_index)
            records.append(rec)
            done.add(cid)
        seeds.append(rec)
        print(f"[search] seed {s}: J={rec['J']:.3f} complexity={rec['complexity']}", flush=True)

    new_count = len([r for r in records if r["formula_ast"] not in ("A", "(Neg U)")])
    target_new = int(cfg["search"]["min_new_candidates"])
    max_new = int(cfg["search"]["max_new_candidates"])
    t0 = time.monotonic()
    last_cost = max((r.get("seconds", 90.0) for r in records), default=90.0)

    while new_count < target_new:
        if budget_left() - last_cost * 1.2 < reserve_seconds:
            print("[search] stopping early: reserve would be violated", flush=True)
            break
        archive = archive_of(records)
        parent = rng.choice(archive)["candidate_id"] if archive else seeds[0]["candidate_id"]
        parent_rec = next(r for r in records if r["candidate_id"] == parent)
        parent_f = dsl.parse_sexpr(parent_rec["formula_ast"])
        cand = propose(parent_f, rng, cfg["search"]["max_ops"], cfg["search"]["max_depth"])
        if cand is None:
            continue
        cid = dsl.formula_id(cand)
        if cid in done or cid in {s["candidate_id"] for s in seeds}:
            continue
        rec = evaluate_candidate(
            model_refs, store, tok, cfg, run_dir, run_id, items, "search", cand, cid,
            parent, "mutation", rhos, nll_full, eval_index)
        _save_eval_index(run_dir, eval_index)
        records.append(rec)
        done.add(cid)
        new_count = len([r for r in records if r["formula_ast"] not in ("A", "(Neg U)")])
        last_cost = rec["seconds"] + 20.0
        print(f"[search] {new_count} new | {rec['formula_infix']} | J={rec['J']:.3f} "
              f"complexity={rec['complexity']} ({rec['seconds']:.0f}s)", flush=True)
        if new_count >= max_new:
            break

    archive = archive_of(records)
    ranked = sorted(archive, key=lambda r: (-r["J"], r["complexity"], r["formula_ast"]))
    return {"n_candidates": len(records), "n_new": len([r for r in records if is_new(r)]),
            "archive": [r["candidate_id"] for r in archive],
            "ranked": ranked[:4], "wall_seconds": time.monotonic() - t0,
            "seed_ids": [s["candidate_id"] for s in seeds]}


def run_selection(model_refs, store, tok, cfg, run_dir, run_id, split_manifest,
                  search_result: dict, nll_full: float | None, max_formulas: int = 2) -> dict:
    items = split_manifest["splits"]["selection"]
    rhos = [float(x) for x in cfg["curve"]["anchors"]]
    eval_index = load_eval_index(run_dir)
    records = read_jsonl(Path(run_dir) / "candidates.jsonl")
    by_id = {r["candidate_id"]: r for r in records}
    # best search entries; allow at most max_formulas, both non-baseline.
    chosen = []
    for r in search_result["ranked"]:
        fid = r["candidate_id"]
        f = dsl.parse_sexpr(by_id[fid]["formula_ast"])
        formula_ast = dsl.canonical_str(f)
        if formula_ast in ("A", "(Neg U)"):
            continue
        chosen.append((fid, f))
        if len(chosen) >= max_formulas:
            break
    results = []
    for fid, f in chosen:
        rec = evaluate_candidate(
            model_refs, store, tok, cfg, run_dir, run_id, items, "selection", f, fid,
            None, "selection", rhos, nll_full, eval_index, method="selection")
        _save_eval_index(run_dir, eval_index)
        results.append(rec)
        print(f"[selection] {rec['formula_infix']} J={rec['J']:.3f}", flush=True)
    if not results:
        raise RuntimeError("no non-baseline formula reached selection")
    best = sorted(results, key=lambda r: (-r["J"], r["complexity"], r["formula_ast"]))[0]
    freeze = {
        "run_id": run_id,
        "frozen_at": time.time(),
        "candidate_id": best["candidate_id"],
        "formula_ast": best["formula_ast"],
        "formula_infix": best["formula_infix"],
        "complexity": best["complexity"],
        "depth": best["depth"],
        "n_ops": best["n_ops"],
        "search_J": by_id[best["candidate_id"]]["J"],
        "selection_J": best["J"],
        "selection_record": best,
        "selection_candidates": [r["candidate_id"] for r in results],
        "rule": "rank by J on selection split; ties lower complexity then canonical AST",
    }
    atomic_write_json(Path(run_dir) / "freeze.json", freeze)
    return freeze
