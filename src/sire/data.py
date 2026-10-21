"""Data preparation: NuminaMath-CoT cn_k12 filtering, near-duplicate clustering,
split freezing, and No Robots monitor samples.

Selection never looks at model correctness. Answers are numeric-only, so a
single verifiable scalar exists by construction; non-numeric/proof/open items
are excluded and the exclusion counts are recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import answers
from .util import atomic_write_json, sha256_file, sha256_text

INSTRUCTION = (
    "Solve the following problem. Reason step by step, then put the final answer "
    "in \\boxed{{}}.\n\nProblem: {problem}"
)

_IMG_RE = re.compile(r"(!\[|<img|\\includegraphics|https?://\S+\.(?:png|jpg|jpeg|gif))", re.I)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff+\-=/^*().]+")


def normalize_problem(text: str) -> str:
    t = text.lower()
    t = _WS_RE.sub(" ", t)
    return t.strip()


def shingles(text: str, k: int = 4) -> set[str]:
    t = _PUNCT_RE.sub("", text)
    t = _WS_RE.sub("", t)
    if len(t) <= k:
        return {t}
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def simhash(sh: set[str]) -> int:
    if not sh:
        return 0
    hs = np.fromiter((_hash64(s) for s in sh), dtype=np.uint64, count=len(sh))
    bits = ((hs[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.int16)
    acc = 2 * bits.sum(axis=0, dtype=np.int32) - len(hs)
    out = 0
    for b in np.flatnonzero(acc > 0):
        out |= 1 << int(b)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def extract_ground_truth(solution: str):
    """First parseable numeric boxed answer scanning from the end."""
    boxes = []
    for m in re.finditer(r"\\boxed\s*\{", solution):
        inner = answers._balanced_braces(solution, m.end())
        if inner is not None:
            boxes.append(inner.strip())
    for b in reversed(boxes):
        v = answers.parse_value(b)
        if v is not None:
            return b, v
    return None, None


def load_cn_k12(raw_dir: Path, seed: int, pool_cap: int = 50000) -> tuple[list[dict], dict]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "NuminaMath-CoT" / "data").glob("train-*.parquet"))
    files = [f for f in files if not f.name.endswith(".incomplete")]
    stats = {"files": [f.name for f in files], "rows": 0, "source_cn_k12": 0,
             "excluded_image": 0, "excluded_length": 0, "excluded_no_numeric_answer": 0,
             "excluded_duplicate_exact": 0, "pool_cap": pool_cap,
             "items_before_cap": 0, "items_after_cap": 0}
    items = []
    for f in files:
        t = pq.read_table(f, columns=["source", "problem", "solution", "messages"])
        stats["rows"] += t.num_rows
        t = t.filter(pc.equal(t.column("source"), "cn_k12"))
        stats["source_cn_k12"] += t.num_rows
        for r in t.to_pylist():
            problem = (r["problem"] or "").strip()
            solution = (r["solution"] or "").strip()
            if not problem or not solution:
                continue
            if _IMG_RE.search(problem) or _IMG_RE.search(solution):
                stats["excluded_image"] += 1
                continue
            if not (20 <= len(problem) <= 2000):
                stats["excluded_length"] += 1
                continue
            gt, val = extract_ground_truth(solution)
            if gt is None:
                stats["excluded_no_numeric_answer"] += 1
                continue
            items.append({
                "problem": problem,
                "solution": solution,
                "gt_raw": gt,
                "gt_value": str(val),
                "norm": normalize_problem(problem),
            })
    stats["items_before_cap"] = len(items)
    if pool_cap and len(items) > pool_cap:
        items.sort(key=lambda it: sha256_text(f"{seed}|pool|{it['norm']}"))
        items = items[:pool_cap]
    stats["items_after_cap"] = len(items)
    return items, stats


def cluster_items(items: list[dict], seed: int, hamming: int = 3, jac: float = 0.8):
    """Exact-normalised dedup + SimHash/LSH near-duplicate clustering."""
    seen = {}
    unique = []
    n_exact = 0
    for it in items:
        key = it["norm"]
        if key in seen:
            n_exact += 1
            continue
        seen[key] = True
        it["shingles"] = shingles(it["norm"])
        it["simhash"] = simhash(it["shingles"])
        unique.append(it)

    # 4 bands of 16 bits: any pair with SimHash Hamming distance <= 3 must share
    # at least one band, so near-duplicates are not missed by banding.
    bands: dict[tuple[int, int], list[int]] = {}
    parent = list(range(len(unique)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, it in enumerate(unique):
        for band in range(4):
            key = (band, (it["simhash"] >> (band * 16)) & 0xFFFF)
            pool = bands.setdefault(key, [])
            for j in pool:
                a, b = unique[j], it
                if bin(a["simhash"] ^ b["simhash"]).count("1") <= hamming or \
                        jaccard(a["shingles"], b["shingles"]) >= jac:
                    union(i, j)
            pool.append(i)

    groups: dict[int, list[int]] = {}
    for i in range(len(unique)):
        groups.setdefault(find(i), []).append(i)

    reps = []
    for gid, members in groups.items():
        # Deterministic representative.
        best = min(members, key=lambda i: sha256_text(f"{seed}|{unique[i]['norm']}"))
        it = unique[best]
        it["cluster_id"] = sha256_text(f"cluster|{it['norm']}")[:16]
        it["cluster_size"] = len(members)
        reps.append(it)
    return reps, {"exact_duplicates_removed": n_exact, "clusters": len(groups),
                  "items_after_dedup": len(reps),
                  "largest_cluster": max((r["cluster_size"] for r in reps), default=0)}


def crosscheck_gsm8k(reps: list[dict], gsm_files: list[Path]) -> dict:
    """Exact-normalised and Shingle-Jaccard overlap check against GSM8K."""
    import pyarrow.parquet as pq

    refs = []
    for f in gsm_files:
        if not f.exists():
            continue
        t = pq.read_table(f)
        for r in t.to_pylist():
            q = r.get("question") or r.get("problem") or ""
            if q:
                norm = normalize_problem(q)
                sh = shingles(norm)
                refs.append((norm, sh, simhash(sh)))
    exact = {r[0] for r in refs}
    bands: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h) in enumerate(refs):
        for band in range(4):
            bands.setdefault((band, (h >> (band * 16)) & 0xFFFF), []).append(i)
    hits = []
    for it in reps:
        if it["norm"] in exact:
            hits.append({"id": it["cluster_id"], "kind": "exact"})
            continue
        cand = set()
        for band in range(4):
            cand.update(bands.get((band, (it["simhash"] >> (band * 16)) & 0xFFFF), []))
        for i in cand:
            if jaccard(it["shingles"], refs[i][1]) >= 0.8:
                hits.append({"id": it["cluster_id"], "kind": "near",
                             "ref": sha256_text(refs[i][0])[:12]})
                break
    return {"gsm8k_reference_items": len(refs), "overlaps": hits}


def build_prompt(problem: str) -> str:
    return INSTRUCTION.format(problem=problem)


def tokenize_chat_example(tok, problem: str, solution: str, max_tokens: int):
    """Full SFT example: prompt tokens (label -100) + assistant tokens."""
    prompt = build_prompt(problem)
    msgs_p = [{"role": "user", "content": prompt}]
    msgs_full = [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": solution}]
    p_text = tok.apply_chat_template(msgs_p, tokenize=False, add_generation_prompt=True)
    f_text = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
    p_ids = tok(p_text, add_special_tokens=False).input_ids
    f_ids = tok(f_text, add_special_tokens=False).input_ids
    if f_ids[:len(p_ids)] != p_ids:
        # longest common prefix fallback
        n = 0
        while n < min(len(p_ids), len(f_ids)) and p_ids[n] == f_ids[n]:
            n += 1
        p_ids = f_ids[:n]
    if len(f_ids) > max_tokens:
        return None
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    if not any(x != -100 for x in labels):
        return None
    return {"input_ids": f_ids, "labels": labels, "n_tokens": len(f_ids),
            "n_prompt_tokens": len(p_ids), "n_assistant_tokens": len(f_ids) - len(p_ids)}


def load_no_robots(raw_dir: Path, seed: int, n: int) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "no_robots").rglob("*.parquet"))
    files = [f for f in files if not f.name.endswith(".incomplete")]
    rows = []
    for f in files:
        t = pq.read_table(f)
        rows.extend(t.to_pylist())
    singles = []
    for r in rows:
        msgs = r.get("messages") or []
        if len(msgs) != 2:
            continue
        if [m.get("role") for m in msgs] != ["user", "assistant"]:
            continue
        p = (msgs[0].get("content") or "").strip()
        a = (msgs[1].get("content") or "").strip()
        if not p or not a:
            continue
        singles.append({"problem": p, "solution": a})
    singles.sort(key=lambda r: sha256_text(f"{seed}|nr|{r['problem']}"))
    chosen = singles[:n]
    out = []
    for i, r in enumerate(chosen):
        prompt = r["problem"]
        out.append({
            "item_id": "nr-" + sha256_text(r["problem"])[:16],
            "problem": prompt,
            "reference": r["solution"],
            "prompt_sha256": sha256_text(prompt),
        })
    return out, {"files": [f.name for f in files], "rows": len(rows),
                 "single_turn": len(singles), "selected": len(out)}


def prepare(cfg: dict, run_id: str) -> dict:
    """Build and freeze all splits. CPU-only."""
    root = Path(cfg["paths"]["data_root"])
    raw = Path(cfg["paths"]["raw_data"])
    seed = cfg["data"]["global_seed"]
    sizes = cfg["data"]["sizes"]
    out_dir = root / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["paths"]["reasoning_model"], local_files_only=True)

    items, load_stats = load_cn_k12(raw, seed, pool_cap=cfg["data"].get("pool_cap", 50000))
    reps, dedup_stats = cluster_items(items, seed, hamming=cfg["data"]["near_duplicate_hamming"])
    gsm = [Path("/home/nxq/data/KEPT/datasets/gsm8k.parquet"),
           Path("/home/nxq/data/KEPT/datasets/gsm8k_test.parquet"),
           Path("/home/nxq/data/KEPT/datasets/gsm8k_train.parquet")]
    gsm = [Path(os.path.expandvars(os.path.expanduser(p)))
           for p in cfg["data"].get("gsm8k_files", [str(p) for p in gsm])]
    cross = crosscheck_gsm8k(reps, gsm)
    cross["requested_files"] = [str(p) for p in gsm]
    cross["missing_files"] = [str(p) for p in gsm if not p.exists()]
    cross["status"] = "checked" if cross["gsm8k_reference_items"] else "not_audited"

    # Deterministic order; splits are assigned from disjoint clusters.
    reps.sort(key=lambda it: sha256_text(f"{seed}|order|{it['cluster_id']}"))
    cursor = 0
    splits: dict[str, list[dict]] = {k: [] for k in ("pilot", "calibration", "search",
                                                     "selection", "test")}

    def take(n, split):
        nonlocal cursor
        got = []
        while len(got) < n:
            if cursor >= len(reps):
                raise RuntimeError(f"not enough deduplicated items for {split}")
            it = reps[cursor]
            cursor += 1
            rec = {
                "item_id": "nm-" + it["cluster_id"],
                "cluster_id": it["cluster_id"],
                "cluster_size": it["cluster_size"],
                "problem": it["problem"],
                "reference_solution": it["solution"],
                "gt_raw": it["gt_raw"],
                "gt_value": it["gt_value"],
                "prompt": build_prompt(it["problem"]),
                "prompt_sha256": sha256_text(build_prompt(it["problem"])),
                "problem_sha256": sha256_text(it["problem"]),
            }
            got.append(rec)
        splits[split] = got

    take(sizes["pilot"], "pilot")

    # Calibration needs full prompt+reference <= max_tokens with the answer intact.
    max_tok = cfg["data"]["max_calibration_tokens"]
    cal = []
    while len(cal) < sizes["calibration"]:
        if cursor >= len(reps):
            raise RuntimeError("not enough calibration candidates")
        it = reps[cursor]
        cursor += 1
        ex = tokenize_chat_example(tok, it["problem"], it["solution"], max_tok)
        if ex is None:
            continue
        rec = {
            "item_id": "nm-" + it["cluster_id"],
            "cluster_id": it["cluster_id"],
            "cluster_size": it["cluster_size"],
            "problem": it["problem"],
            "reference_solution": it["solution"],
            "gt_raw": it["gt_raw"],
            "gt_value": it["gt_value"],
            "prompt": build_prompt(it["problem"]),
            "prompt_sha256": sha256_text(build_prompt(it["problem"])),
            "problem_sha256": sha256_text(it["problem"]),
            "n_tokens": ex["n_tokens"],
            "n_prompt_tokens": ex["n_prompt_tokens"],
            "n_assistant_tokens": ex["n_assistant_tokens"],
        }
        checksum = hashlib.sha256(np.asarray(ex["input_ids"], dtype=np.int32).tobytes()).hexdigest()
        rec["tokens_sha256"] = checksum
        cal.append(rec)
    splits["calibration"] = cal

    take(sizes["search"], "search")
    take(sizes["selection"], "selection")
    take(sizes["test"], "test")

    monitor, monitor_stats = load_no_robots(raw, seed, sizes["monitor"])

    hashes = {}
    for f in sorted((raw / "NuminaMath-CoT" / "data").glob("train-*.parquet")):
        if f.name.endswith(".incomplete"):
            continue
        hashes[f.name] = sha256_file(f)
    for f in sorted((raw / "no_robots").rglob("*.parquet")):
        if f.name.endswith(".incomplete"):
            continue
        hashes["no_robots/" + f.name] = sha256_file(f)

    all_ids = [r["item_id"] for s in splits.values() for r in s]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("split leakage: duplicate item ids across splits")
    all_clusters = [r["cluster_id"] for s in splits.values() for r in s]
    if len(all_clusters) != len(set(all_clusters)):
        raise RuntimeError("split leakage: duplicate clusters across splits")

    manifest = {
        "version": cfg["version"],
        "run_id": run_id,
        "created_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "seed": seed,
        "sizes": {k: len(v) for k, v in splits.items()},
        "requested_sizes": sizes,
        "dataset": cfg["data"]["dataset"],
        "dataset_source": cfg["data"]["dataset_source"],
        "monitor_dataset": cfg["data"]["monitor_dataset"],
        "dataset_files_sha256": hashes,
        "load_stats": load_stats,
        "dedup": {"method": "exact normalised + 64-bit SimHash(4x16 bands, verified Jaccard>=0.8 or Hamming<=%d)" % cfg["data"]["near_duplicate_hamming"],
                  **dedup_stats},
        "crosscheck_gsm8k": cross,
        "generation_config": cfg["generation"],
        "prompt_template": INSTRUCTION,
        "splits": splits,
        "monitor": {"stats": monitor_stats, "items": monitor},
        "caveats": [
            "Answers are restricted to numeric scalars parseable from a boxed expression; proofs, open answers, expressions and multiple-choice letters are excluded.",
            "Near-duplicate detection is SimHash+LSH, not an exhaustive pairwise comparison; recall is audited only by the recorded method.",
            "Only GSM8K overlap was checked locally; MATH/AIME overlap is un-audited.",
            "NuminaMath-CoT cn_k12 is not a contamination-free source; no zero-contamination claim is made.",
        ],
    }
    atomic_write_json(out_dir / "splits.json", manifest)
    atomic_write_json(out_dir / "split_hashes.json", {
        "splits_sha256": sha256_file(out_dir / "splits.json"),
        "split_ids": {"pilot": [r["item_id"] for r in splits["pilot"]],
                      "calibration": [r["item_id"] for r in splits["calibration"]],
                      "search": [r["item_id"] for r in splits["search"]],
                      "selection": [r["item_id"] for r in splits["selection"]],
                      "test": [r["item_id"] for r in splits["test"]]},
    })
    return manifest
