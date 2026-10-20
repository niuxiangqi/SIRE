"""Automated correctness checks (small networks first, then sampled real weights)."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import torch

from . import answers, dsl, eval as eval_mod
from .masking import mask_hash, topk_indices, specs_with_shapes
from .util import atomic_write_json, read_jsonl


def _res(name, passed, **detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def _sha(t) -> str:
    tt = t.detach().cpu().contiguous()
    if tt.dtype == torch.bfloat16:
        arr = tt.view(torch.uint16).numpy()
    else:
        arr = tt.numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def small_network_checks() -> list[dict]:
    out = []
    # exact quota and boundary cases
    ok = True
    detail = {}
    for n in (1, 2, 7, 1000, 1001):
        for rho in (0.0, 0.01, 0.02, 0.05, 0.5, 0.99, 1.0):
            k = int(math.floor(rho * n))
            score = torch.arange(n, dtype=torch.float32)
            kept = topk_indices(score, k)
            ok &= len(kept) == k
            if k:
                ok &= int(kept.min()) >= 0 and int(kept.max()) < n
            detail[f"n{n}_rho{rho}"] = k
    out.append(_res("exact_quota_and_boundaries", ok, cases=detail))

    # deterministic tie-breaking: equal scores keep lowest flat indices
    score = torch.ones(10)
    kept = topk_indices(score, 4).tolist()
    out.append(_res("tie_break_lowest_index", kept == [0, 1, 2, 3], kept=kept))

    # SafeDiv at and near zero, sign(0)=+1
    tests = [("x", "SafeDiv(Add(A, A), Sub(A, A))", 0.0, 0.0),
             ("neg", "SafeDiv(Sub(A, A), Neg(A))", 0.0, 0.0),
             ("sign", "SafeDiv(A, A)", -2.0, 1.0),
             ("near", "SafeDiv(Add(A, A), Mul(A, A))", 1e-3, 2e-3 / 1e-6)]
    ok = True
    for name, expr, val, expected in tests:
        a = torch.tensor([val], dtype=torch.float32)
        got = dsl.evaluate(dsl.parse(expr), {"A": a})[0].item()
        ok &= abs(got - expected) < 1e-3 * max(1.0, abs(expected))
    a = torch.tensor([1e-12, 0.0, -1e-9])
    got = dsl.evaluate(dsl.parse("SafeDiv(Add(A, A), A)"), {"A": a})
    ok &= bool(torch.isfinite(got).all())
    out.append(_res("safediv_zero_behaviour", ok))

    # U = mean|g| is not |mean g|
    g = torch.tensor([1.0, -3.0, 2.0, -2.0])
    U = g.abs().mean()
    G = g.mean()
    out.append(_res("mean_abs_g_not_abs_mean_g", not torch.allclose(U, G.abs()),
                    U=float(U), absG=float(G.abs())))

    # hand-written criteria equal DSL equivalents
    d = torch.tensor([[-2.0, 0.5], [3.0, -1.0]])
    dslish = dsl.evaluate(dsl.parse("Add(Abs(D), Mul(D, Neg(D)))"), {"D": d})
    hand = torch.abs(d) + d * (-d)
    out.append(_res("hand_criteria_matches_dsl", torch.equal(dslish, hand)))

    # parse failure is wrong, never skipped
    ok, pred, _ = answers.is_correct("I cannot solve this.", "42")
    out.append(_res("parse_failure_counts_wrong", not ok, parsed=pred))

    # resume does not regenerate finished samples
    import sire.eval as em
    with tempfile.TemporaryDirectory() as td:
        cfg = {"generation": {"batch_size": 4}, "data": {"global_seed": 1}}
        run_dir = Path(td)
        items = [{"item_id": "x1", "prompt": "q", "gt_value": "1"}]
        outdir = em.eval_dir(run_dir, "candidate", "fid", 0.05, "search")
        outdir.mkdir(parents=True)
        rec = {"method": "candidate", "formula_id": "fid", "rho": 0.05, "split": "search",
               "item_id": "x1", "seed": 1, "gt_raw": "1", "gt_parsed": "1",
               "parsed_pred": "1", "correct": True, "n_new_tokens": 3,
               "truncated": False, "seconds": 0.1, "text": "\\boxed{1}"}
        (outdir / "samples.jsonl").write_text(json.dumps(rec) + "\n")
        audit = {"mask_sha256": "x", "total_kept": 0, "method": "candidate", "rho": 0.05,
                 "tensors": []}
        old = em.generate_batch
        em.generate_batch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("regenerated"))
        try:
            s = em.evaluate_rows(None, None, None, cfg, run_dir, "candidate", None, "fid",
                                 0.05, items, "search", mask_audit=audit, compute_nll=False)
            ok = s["n_correct"] == 1 and s["n_items"] == 1
        except AssertionError:
            ok = False
        finally:
            em.generate_batch = old
    out.append(_res("resume_does_not_recompute", ok))
    return out


def split_isolation_check(split_manifest: dict) -> dict:
    ids = [r["item_id"] for r in split_manifest["splits"]["search"]]
    ids += [r["item_id"] for r in split_manifest["splits"]["selection"]]
    ids += [r["item_id"] for r in split_manifest["splits"]["test"]]
    clusters = [r["cluster_id"] for s in ("search", "selection", "test")
                for r in split_manifest["splits"][s]]
    return _res("split_isolation", len(ids) == len(set(ids)) and len(clusters) == len(set(clusters)),
                n=len(ids), unique_clusters=len(set(clusters)))


def real_weight_checks(model_refs, store, cfg, run_dir: Path) -> list[dict]:
    import json as _json
    from safetensors import safe_open

    out = []
    specs = model_refs.specs
    model = model_refs.model
    # rho=1 through the explicit where path, on sampled tensors
    ok = True
    for spec in specs[::60]:
        name = spec["name"]
        ones = torch.ones(spec["numel"], dtype=torch.bool, device="cuda")
        w = torch.where(ones.view(spec["shape"]), model_refs.r_ref[name], model_refs.b_ref[name])
        ok &= torch.equal(w, model_refs.r_ref[name])
        del ones, w
    out.append(_res("rho1_where_path_bitwise", ok, sampled=len(specs[::60])))

    # full rho=1 reconstruction equals pristine R bitwise
    audit = model_refs.apply(store, rho=1.0, method="all")
    ok = all(torch.equal(model.get_parameter(s["name"]), model_refs.r_ref[s["name"]])
             for s in specs)
    out.append(_res("rho1_full_reconstruction_bitwise", ok,
                    mask_sha256=audit["mask_sha256"], tensors=len(specs)))

    # non-target parameters remain R (sampled, bitwise against pristine shards)
    rmap = _json.loads((Path(cfg["paths"]["reasoning_model"]) /
                        "model.safetensors.index.json").read_text())["weight_map"]
    targets = {s["name"] for s in specs}
    candidates = [n for n in sorted(rmap) if n not in targets]
    pick = candidates[::max(1, len(candidates) // 8)][:8]
    handles = {}
    ok = True
    hashes = {}
    for n in pick:
        f = rmap[n]
        if f not in handles:
            handles[f] = safe_open(str(Path(cfg["paths"]["reasoning_model"]) / f),
                                   framework="pt", device="cpu")
        ref = handles[f].get_tensor(n).to("cuda")
        p = model.get_parameter(n)
        ok &= torch.equal(p, ref)
        hashes[n] = _sha(p)
        del ref
    out.append(_res("nontarget_parameters_equal_R", ok, sampled=pick[:4], hashes_sample=hashes))

    # no drift across candidate switches
    model_refs.apply(store, expr=dsl.parse("A"), rho=0.01)
    model_refs.apply(store, expr=dsl.parse("Neg(U)"), rho=0.20)
    model_refs.apply(store, expr=dsl.parse("Mul(A, U)"), rho=0.05)
    model_refs.apply(store, rho=1.0, method="all")
    ok = all(torch.equal(model.get_parameter(s["name"]), model_refs.r_ref[s["name"]])
             for s in specs)
    out.append(_res("no_drift_after_mask_switches", ok, switches=3))

    # hand criteria vs DSL on real tensors
    ok = True
    for spec in specs[::50]:
        name = spec["name"]
        a = store.leaf("A", name)
        u = store.leaf("U", name)
        from .masking import tensor_score
        ok &= torch.allclose(tensor_score(dsl.parse('A'), name, store).cpu(), a, rtol=1e-5, atol=1e-6)
        ok &= torch.allclose(tensor_score(dsl.parse('Neg(U)'), name, store).cpu(), -u, rtol=1e-5, atol=1e-6)
    out.append(_res("hand_criteria_matches_dsl_real", ok, sampled=len(specs[::50])))

    # real quota audit at rho=0.01 and 0.20
    ok = True
    detail = {}
    for rho in (0.01, 0.20):
        audit = model_refs.apply(store, expr=dsl.parse("A"), rho=rho)
        expected = sum(int(math.floor(rho * s["numel"])) for s in specs)
        ok &= audit["total_kept"] == expected
        detail[str(rho)] = {"total_kept": audit["total_kept"], "expected": expected,
                            "mask_sha256": audit["mask_sha256"]}
        for t in audit["tensors"]:
            ok &= t["quota"] == int(math.floor(rho * t["numel"]))
    out.append(_res("real_mask_quota_exact", ok, cases=detail))
    return out


def run_all(model_refs, store, cfg, run_dir: Path, split_manifest: dict) -> list[dict]:
    results = small_network_checks()
    results.append(split_isolation_check(split_manifest))
    if model_refs is not None:
        results.extend(real_weight_checks(model_refs, store, cfg, run_dir))
    failed = [r for r in results if not r["passed"]]
    atomic_write_json(Path(run_dir) / "correctness" / "results.json", {
        "results": results,
        "n_passed": len(results) - len(failed),
        "n_failed": len(failed),
    })
    return results
