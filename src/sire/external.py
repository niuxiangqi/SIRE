"""Post-hoc external tests: GSM8K (pass@k) and MMLU subsets.

This module is deliberately separate from the frozen v0 acceptance path. It
reuses the frozen formula, pristine B/R and feature cache, but measures on
external benchmarks that were not used to drive search/selection. Results are
recorded as post-hoc evidence, not as a replacement for the pre-registered v0
test split.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import answers

GSM_INSTRUCTION = (
    "Solve the following problem. Reason step by step, then put the final answer "
    "in \\boxed{{}}.\n\nProblem: {problem}"
)
MMLU_INSTRUCTION = (
    "Answer the following multiple-choice question. Choose exactly one of A, B, C, D "
    "and put only the letter in \\boxed{{}}.\n\nQuestion: {question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}"
)


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def load_gsm8k(path: Path, n: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or "####" not in a:
            continue
        gt = a.rsplit("####", 1)[1].strip().replace(",", "")
        if answers.parse_value(gt) is None:
            continue
        prompt = GSM_INSTRUCTION.format(problem=q)
        out.append({"item_id": "gsm8k-" + _h(seed, q)[:16], "problem": q,
                    "prompt": prompt, "gt": gt, "gt_value": gt, "source": "gsm8k"})
    out.sort(key=lambda x: _h(seed, "order", x["item_id"]))
    return out[:n]


def load_mmlu(dataset_dir: Path, subjects: list[str], per_subject: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for subj in subjects:
        p = dataset_dir / f"mmlu_{subj}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        rows = pq.read_table(p).to_pylist()
        for r in rows:
            q = (r.get("question") or "").strip()
            choices = r.get("choices") or []
            if not q or len(choices) != 4:
                continue
            ans = int(r["answer"])
            if not 0 <= ans < 4:
                continue
            letters = "ABCD"
            prompt = MMLU_INSTRUCTION.format(question=q, a=choices[0], b=choices[1],
                                             c=choices[2], d=choices[3])
            out.append({"item_id": f"mmlu-{subj}-" + _h(seed, subj, q)[:12],
                        "problem": q, "prompt": prompt, "gt": letters[ans],
                        "subject": subj, "choices": list(choices), "source": f"mmlu/{subj}"})
    # deterministic per-subject sample
    picked = []
    for subj in subjects:
        block = [r for r in out if r["subject"] == subj]
        block.sort(key=lambda x: _h(seed, "order", x["item_id"]))
        picked.extend(block[:per_subject])
    picked.sort(key=lambda x: _h(seed, "order-all", x["item_id"]))
    return picked


def prepare(gsm_path: Path, mmlu_dir: Path, gsm_n: int, subjects: list[str],
            per_subject: int, seed: int) -> dict:
    gsm = load_gsm8k(gsm_path, gsm_n, seed)
    mmlu = load_mmlu(mmlu_dir, subjects, per_subject, seed)
    return {
        "seed": seed,
        "gsm8k": {"path": str(gsm_path), "n": len(gsm), "items": gsm},
        "mmlu": {"subjects": subjects, "per_subject": per_subject, "n": len(mmlu),
                 "items": mmlu},
        "note": "Post-hoc external sets; not used to drive v0 search/selection.",
    }


def _parse_mmlu(text: str) -> str | None:
    boxed = answers.extract_boxed(text)
    cand = boxed if boxed is not None else text
    found = re.findall(r"\b([ABCD])\b", cand.upper())
    return found[-1] if found else None


def make_sample_items(items: list[dict], k: int) -> list[dict]:
    out = []
    for it in items:
        for s in range(k):
            out.append({"item_id": f"{it['item_id']}#s{s}", "base_id": it["item_id"],
                        "sample_index": s, "prompt": it["prompt"], "gt": it["gt"],
                        "source": it["source"]})
    return out


def check(text: str, gt: str, source: str):
    if source.startswith("mmlu"):
        pred = _parse_mmlu(text)
        return pred == gt, pred
    ok, pred, _ = answers.is_correct(text, gt)
    return ok, pred
