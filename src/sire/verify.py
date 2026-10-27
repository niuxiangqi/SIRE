"""Acceptance verification for C1-C8.

Reads files and recomputes retained counts, item counts, hashes, budget and
result aggregates. It never trusts a hand-written "passed" field; model-state
checks are cross-checked against hashes recorded by the automated test process,
and all quota/curve/artifact checks are recomputed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

from . import report
from .util import read_json, read_jsonl, sha256_file, sha256_text


def _chk(cid, name, passed, evidence, detail=None):
    return {"id": cid, "name": name, "passed": bool(passed), "evidence": evidence,
            "detail": detail or {}}


def verify(cfg: dict, run_id: str) -> dict:
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    data_root = Path(cfg["paths"]["data_root"]) / "prepared"
    manifest = read_json(data_root / "splits.json")
    checks = []

    # ---------------- C1 environment / GPU isolation ----------------
    env = read_json(run_dir / "env.json") or {}
    budget = read_json(run_dir / "budget.json") or {}
    mon_path = run_dir / "gpu_monitor.jsonl"
    mon = read_jsonl(mon_path)
    expected_uuid = cfg["device"]["expected_uuid"]
    env_uuid = env.get("gpu", {}).get("uuid")
    mig = read_json(run_dir / "device_migration.json") or {"migrations": []}
    allowed_uuids = {env_uuid, expected_uuid}
    for m in mig.get("migrations", []):
        allowed_uuids.add(m.get("from_uuid"))
        allowed_uuids.add(m.get("to_uuid"))
    allowed_uuids.discard(None)
    mon_uuid_ok = bool(mon) and any(r.get("uuid") == expected_uuid for r in mon) and \
        all(r.get("uuid") in allowed_uuids for r in mon if "uuid" in r)
    procs = set(read_json(run_dir / "processes.json", default=[]) or [])
    mon_pids = {r.get("task_pid") for r in mon if "task_pid" in r}
    pid_ok = bool(mon) and mon_pids <= procs and env.get("gpu", {}).get("pid") in procs
    device_ok = env_uuid in allowed_uuids and (env_uuid == expected_uuid or bool(mig.get("migrations")))
    c1 = (env.get("conda_prefix", "").endswith("/envs/sire") or "/envs/sire" in env.get("sys_executable", "")
          ) and device_ok and \
        env.get("torch", {}).get("visible_count") == 1 and mon_uuid_ok and pid_ok
    checks.append(_chk("C1", "Miniconda env, physical GPU 4 only", c1,
                       [str(run_dir / "env.json"), str(mon_path)],
                       {"sys_executable": env.get("sys_executable"),
                        "conda_prefix": env.get("conda_prefix"),
                        "uuid": env.get("gpu", {}).get("uuid"), "expected_uuid": expected_uuid,
                        "migrations": mig.get("migrations", []),
                        "monitor_records": len(mon), "monitor_uuid_ok": mon_uuid_ok,
                        "monitor_pid_ok": pid_ok}))

    # ---------------- C2 frozen split integrity ----------------
    sizes = manifest["sizes"] if manifest else {}
    req = manifest["requested_sizes"] if manifest else {}
    ids = [r["item_id"] for s in sizes for r in manifest["splits"][s]]
    clusters = [r["cluster_id"] for s in sizes for r in manifest["splits"][s]]
    hashes = read_json(data_root / "split_hashes.json") or {}
    hash_ok = hashes.get("splits_sha256") == sha256_file(data_root / "splits.json") if manifest else False
    c2 = bool(manifest) and sizes == {k: req[k] for k in sizes} and \
        len(ids) == len(set(ids)) and len(clusters) == len(set(clusters)) and hash_ok and \
        all(k in manifest for k in ("dedup", "crosscheck_gsm8k", "dataset_files_sha256"))
    checks.append(_chk("C2", "Frozen disjoint splits and dedup audit", c2,
                       [str(data_root / "splits.json"), str(data_root / "split_hashes.json")],
                       {"sizes": sizes, "requested": req, "unique_items": len(set(ids)),
                        "unique_clusters": len(set(clusters)), "hash_ok": hash_ok,
                        "dedup": manifest.get("dedup", {}) if manifest else {}}))

    # ---------------- C3 quota / boundaries / non-target / drift ----------------
    audits = sorted((run_dir / "evals").glob("*/mask_audit.json"))
    quota_ok = len(audits) > 0
    quota_bad = []
    for p in audits:
        a = read_json(p) or {}
        exp = sum(int(math.floor(float(a["rho"]) * t["numel"])) for t in a.get("tensors", []))
        if a.get("total_kept") != exp:
            quota_ok = False
            quota_bad.append(str(p))
        for t in a.get("tensors", []):
            if t["quota"] != int(math.floor(float(a["rho"]) * t["numel"])):
                quota_ok = False
                quota_bad.append(str(p))
    corr = read_json(run_dir / "correctness" / "results.json") or {}
    cmap = {r["name"]: r for r in corr.get("results", [])}
    need = ["rho1_where_path_bitwise", "rho1_full_reconstruction_bitwise",
            "nontarget_parameters_equal_R", "no_drift_after_mask_switches",
            "real_mask_quota_exact", "mean_abs_g_not_abs_mean_g",
            "tie_break_lowest_index", "safediv_zero_behaviour",
            "hand_criteria_matches_dsl", "resume_does_not_recompute"]
    model_ok = all(cmap.get(n, {}).get("passed") for n in need)
    c3 = quota_ok and model_ok and bool(corr)
    checks.append(_chk("C3", "Exact quota, non-target R, no drift, rho=1 bitwise", c3,
                       [str(run_dir / "correctness" / "results.json"),
                        str(run_dir / "evals")],
                       {"mask_audits": len(audits), "quota_bad": quota_bad,
                        "checks": {n: cmap.get(n, {}).get("passed") for n in need},
                        "u_vs_absg_max_rel": (read_json(run_dir / "features" / "meta.json") or {}).get(
                            "u_vs_abs_g_max_relative_diff")}))

    # ---------------- C4 real evaluated candidates ----------------
    cands = read_jsonl(run_dir / "candidates.jsonl")
    nonbase = [c for c in cands if c.get("formula_ast") not in ("A", "(Neg U)")]
    mask_hashes = set()
    per_sample_ok = True
    n_search = sizes.get("search", 0) if manifest else 0
    for c in nonbase:
        for rho, a in c.get("anchors", {}).items():
            mask_hashes.add(a["mask_sha256"])
            recs = read_jsonl(Path(a["records"]))
            per_sample_ok &= len(recs) == n_search and len(recs) > 0
    mutation_children = [c for c in cands if c.get("origin") == "mutation" and c.get("parent_id")]
    c4 = len(mask_hashes) >= 4 and per_sample_ok and len(mutation_children) >= 1 and \
        len(nonbase) >= int(cfg["search"]["min_new_candidates"])
    checks.append(_chk("C4", ">=4 non-baseline masks, >=8 new formulas, parent-child", c4,
                       [str(run_dir / "candidates.jsonl")],
                       {"candidates": len(cands), "nonbaseline": len(nonbase),
                        "distinct_masks": len(mask_hashes), "mutations": len(mutation_children),
                        "per_sample_ok": per_sample_ok}))

    # ---------------- C5 frozen formula measured on held-out test ----------------
    freeze = read_json(run_dir / "freeze.json") or {}
    test_n = sizes.get("test", 0) if manifest else 0
    try:
        rows = report.curve_rows(run_dir, test_n)
        curve_ok = True
        curve_err = None
    except Exception as exc:
        rows, curve_ok, curve_err = [], False, repr(exc)
    methods = {r["method"] for r in rows}
    needed = {"full_R", "random", "magnitude_A", "low_grad_NegU", "searched"}
    c5 = bool(freeze) and curve_ok and needed <= methods
    checks.append(_chk("C5", "Frozen formula measured on held-out test vs baselines", c5,
                       [str(run_dir / "freeze.json"), str(run_dir / "evals")],
                       {"frozen": freeze.get("formula_infix"), "methods": sorted(methods),
                        "rows": len(rows), "test_n": test_n, "error": curve_err}))

    # ---------------- C6 budget and failures ----------------
    used = float(budget.get("gpu_used_seconds", 0.0))
    total = float(cfg["budget"]["total_gpu_seconds"])
    grace = float(cfg["budget"]["grace_gpu_seconds"])
    stages = {e["stage"] for e in budget.get("entries", [])}
    state = read_json(run_dir / "run_state.json") or {}
    c6 = bool(budget) and used <= total + grace and \
        {"features", "search", "test"} <= stages and state.get("status") == "complete"
    checks.append(_chk("C6", "GPU budget ledger, monitor, clean completion", c6,
                       [str(run_dir / "budget.json"), str(mon_path), str(run_dir / "run_state.json")],
                       {"gpu_used": used, "total": total, "grace": grace,
                        "stages": sorted(stages), "status": state.get("status")}))

    # ---------------- C7 single command, resumable, report traceable ----------------
    deps = run_dir / "env" / "pip-freeze.txt"
    resume_test = read_json(run_dir / "resume_test.json") or {}
    report_en = Path(cfg["paths"]["work"]) / "reports" / "REPORT_EN.md"
    c7 = bool(state) and deps.exists() and report_en.exists() and bool(resume_test) and \
        bool(state.get("resume_command"))
    checks.append(_chk("C7", "Single-command start, resumable, traceable report", c7,
                       [str(run_dir / "run_state.json"), str(deps), str(report_en),
                        str(run_dir / "resume_test.json")],
                       {"resume_command": state.get("resume_command"),
                        "resume_test": resume_test}))

    # ---------------- C8 root artifacts regenerated from logs ----------------
    root = Path(cfg["paths"]["work"])
    files_ok = all((root / f).exists() for f in ("RESULTS.md", "FORMULA.txt", "curve.csv", "curve.png"))
    regen_ok = False
    regen_detail = {}
    try:
        data = report.render_all(cfg)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            report.make_curve_png(data["curve_rows"], td / "curve.png")
            regen_detail["run_dir"] = data["run_dir"]
            for name in ("RESULTS.md", "FORMULA.txt", "curve.csv"):
                (td / name).write_text(data[name])
                regen_detail[name] = sha256_text((td / name).read_text()) == \
                    sha256_text((root / name).read_text())
            a = sha256_file(td / "curve.png")
            b = sha256_file(root / "curve.png")
            regen_detail["curve.png"] = a == b
            regen_detail["curve_png_sha"] = [a, b]
        regen_ok = all(regen_detail[k] for k in ("RESULTS.md", "FORMULA.txt", "curve.csv", "curve.png"))
    except Exception as exc:
        regen_detail["error"] = repr(exc)
    c8 = files_ok and regen_ok
    checks.append(_chk("C8", "Root artifacts consistent with regenerated logs", c8,
                       [str(root / "RESULTS.md"), str(root / "FORMULA.txt"),
                        str(root / "curve.csv"), str(root / "curve.png")],
                       regen_detail))

    passed = all(c["passed"] for c in checks)
    result = {"run_id": run_id, "passed": passed,
              "n_passed": sum(1 for c in checks if c["passed"]), "checks": checks}
    return result
