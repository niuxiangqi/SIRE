"""Regenerate root artifacts from on-disk logs only.

RESULTS.md is rebuilt from every run directory that has a frozen formula and a
measured test curve; earlier sections are preserved by reconstruction, and new
runs append. curve.csv/png describe the latest complete run.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .util import atomic_write_json, atomic_write_text, read_json, read_jsonl, sha256_file

METHOD_ORDER = ["full_R", "random", "magnitude_A", "low_grad_NegU", "searched"]
METHOD_LABEL = {
    "full_R": "full R",
    "random": "random retention",
    "magnitude_A": "magnitude A",
    "low_grad_NegU": "low-gradient Neg(U)",
    "searched": "searched formula",
}
CURVE_METHODS = set(METHOD_ORDER)


def _read_summaries(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted((run_dir / "evals").glob("*/summary.json")):
        try:
            s = read_json(p)
        except Exception:
            continue
        if not s:
            continue
        s["_path"] = str(p)
        out.append(s)
    return out


def _recompute_accuracy(summary: dict) -> tuple[int, int]:
    recs = read_jsonl(Path(summary["records"]))
    return len(recs), sum(1 for r in recs if r.get("correct"))


def curve_rows(run_dir: Path, test_n: int) -> list[dict]:
    summaries = [s for s in _read_summaries(Path(run_dir)) if s.get("split") == "test"]
    by_method: dict[str, dict[float, dict]] = {}
    for s in summaries:
        m = s.get("method")
        if m not in CURVE_METHODS:
            continue
        n, correct = _recompute_accuracy(s)
        if n != test_n:
            raise RuntimeError(f"curve row {m} {s.get('rho')} has {n} items, expected {test_n}")
        by_method.setdefault(m, {})[float(s["rho"])] = {
            "method": m,
            "formula_id": s["formula_id"],
            "rho": float(s["rho"]),
            "n_items": n,
            "n_correct": correct,
            "accuracy": correct / n,
            "mean_new_tokens": s["mean_new_tokens"],
            "truncation_rate": s["truncation_rate"],
            "nll_delta": s.get("nll_delta"),
            "seconds": s.get("seconds", 0.0),
        }
    full = by_method.get("full_R", {}).get(1.0)
    if full is None:
        raise RuntimeError("full R test row missing")
    rows = []
    for m in METHOD_ORDER:
        for rho in sorted(by_method.get(m, {})):
            r = dict(by_method[m][rho])
            r["rel_to_full_R"] = r["accuracy"] / full["accuracy"] if full["accuracy"] else None
            rows.append(r)
    return rows


def rho_star(rows: list[dict], method: str, full_acc: float) -> tuple[float | None, float]:
    pts = [r for r in rows if r["method"] == method]
    best = max((r["accuracy"] for r in pts), default=float("nan"))
    ok = sorted(r["rho"] for r in pts if r["accuracy"] >= 0.90 * full_acc)
    return (ok[0] if ok else None), best


def _fmt_rho(x):
    return "n/a" if x is None else f"{x:g}"


def _fmt_acc(x):
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_metric(x, nd=3):
    if x is None:
        return "n/a"
    try:
        v = float(x)
        if v != v:
            return "n/a"
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def curve_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["method", "formula_id", "rho", "n_items", "n_correct", "accuracy",
                "rel_to_full_R", "mean_new_tokens", "truncation_rate", "nll_delta", "seconds"])
    for r in rows:
        w.writerow([r["method"], r["formula_id"], f"{r['rho']:g}", r["n_items"],
                    r["n_correct"], f"{r['accuracy']:.6f}",
                    "" if r["rel_to_full_R"] is None else f"{r['rel_to_full_R']:.6f}",
                    f"{r['mean_new_tokens']:.1f}", f"{r['truncation_rate']:.6f}",
                    "" if r["nll_delta"] is None else f"{r['nll_delta']:.6f}",
                    f"{r['seconds']:.1f}"])
    return buf.getvalue()


def _run_section(run_dir: Path, rows: list[dict], freeze: dict, test_n: int) -> str:
    full_acc = next(r["accuracy"] for r in rows if r["method"] == "full_R")
    lines = []
    lines.append(f"## Run `{run_dir.name}`")
    lines.append("")
    lines.append(f"- Frozen formula: `{freeze['formula_infix']}`")
    lines.append(f"- Canonical AST: `{freeze['formula_ast']}`")
    lines.append(f"- Complexity: {freeze['complexity']} (unique operator nodes + G/U reductions)")
    lines.append(f"- Test items: {test_n}; accuracy is accuracy under the frozen output budget, "
                 f"not unbounded capability.")
    lines.append("")
    lines.append("| method | rho* | best acc | " +
                 " | ".join(f"rho={_fmt_rho(x)}" for x in sorted({r['rho'] for r in rows})) + " |")
    lines.append("|---|---|---|" + "---|" * len(sorted({r['rho'] for r in rows})))
    rhos = sorted({r["rho"] for r in rows})
    for m in METHOD_ORDER:
        pts = {r["rho"]: r for r in rows if r["method"] == m}
        if not pts:
            continue
        rs, best = rho_star(rows, m, full_acc)
        cells = [f"{pts[x]['n_correct']}/{pts[x]['n_items']}" if x in pts else "—" for x in rhos]
        lines.append(f"| {METHOD_LABEL[m]} | {_fmt_rho(rs)} | {_fmt_acc(best)} | " +
                     " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mask hashes (formula): " + ", ".join(
        f"rho={k}:`{v['mask_sha256'][:16]}`" for k, v in
        sorted(freeze.get("selection_record", {}).get("anchors", {}).items())))
    lines.append("")
    return "\n".join(lines)


def _collect_complete_runs(runs_dir: Path) -> list[tuple[Path, list[dict], dict]]:
    out = []
    for run_dir in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        freeze = read_json(run_dir / "freeze.json")
        if not freeze:
            continue
        manifest = read_json(Path("/home/nxq/data/dataset/SIRE/prepared/splits.json"))
        test_n = len(manifest["splits"]["test"]) if manifest else None
        try:
            rows = curve_rows(run_dir, test_n)
        except Exception:
            continue
        out.append((run_dir, rows, freeze))
    return out


def render_all(cfg: dict) -> dict[str, bytes | str]:
    runs_dir = Path(cfg["paths"]["runs"])
    complete = _collect_complete_runs(runs_dir)
    if not complete:
        raise RuntimeError("no complete run found")
    latest_dir, latest_rows, latest_freeze = complete[-1]
    full_acc = next(r["accuracy"] for r in latest_rows if r["method"] == "full_R")

    # Headline table for the latest run.
    head = [f"# SIRE v0 results", "",
            "Per-run sections below are regenerated from per-sample logs. "
            "Sparse Δ reduces retained task-vector parameters only; it does not reduce "
            "neurons, layers, FLOPs or latency, and no speedup is claimed.", "",
            "## Headline: rho*(f) = min { rho : accuracy(f, rho) >= 0.90 * accuracy(full R) }",
            "", "Lower is better. No interpolation between tested points.", "",
            f"Latest run: `{latest_dir.name}`, formula `{latest_freeze['formula_infix']}`.", "",
            "| method | rho* | best accuracy |", "|---|---|---|"]
    for m in METHOD_ORDER:
        if not any(r["method"] == m for r in latest_rows):
            continue
        rs, best = rho_star(latest_rows, m, full_acc)
        head.append(f"| {METHOD_LABEL[m]} | {_fmt_rho(rs)} | {_fmt_acc(best)} |")
    head.append("")
    sections = []
    for run_dir, rows, freeze in complete:
        manifest = read_json(Path("/home/nxq/data/dataset/SIRE/prepared/splits.json"))
        test_n = len(manifest["splits"]["test"])
        sections.append(_run_section(run_dir, rows, freeze, test_n))
    results_md = "\n".join(head) + "\n" + "\n".join(sections)

    formula_txt = (f"{latest_freeze['formula_infix']}\n"
                   f"canonical_ast: {latest_freeze['formula_ast']}\n"
                   f"complexity: {latest_freeze['complexity']}\n"
                   f"mask_sha256: " + ",".join(
                       f"{k}={v['mask_sha256']}" for k, v in
                       sorted(latest_freeze.get("selection_record", {}).get("anchors", {}).items())) + "\n")

    return {
        "RESULTS.md": results_md,
        "FORMULA.txt": formula_txt,
        "curve.csv": curve_csv(latest_rows),
        "curve_rows": latest_rows,
        "run_dir": str(latest_dir),
        "freeze": latest_freeze,
    }


def make_curve_png(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_acc = next(r["accuracy"] for r in rows if r["method"] == "full_R")
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    styles = {"full_R": ("k", "--", "full R"),
              "random": ("tab:gray", "o-", "random retention"),
              "magnitude_A": ("tab:blue", "s-", "magnitude A"),
              "low_grad_NegU": ("tab:orange", "^-", "low-gradient Neg(U)"),
              "searched": ("tab:red", "D-", "searched formula")}
    for m in METHOD_ORDER:
        pts = sorted([r for r in rows if r["method"] == m], key=lambda r: r["rho"])
        if not pts:
            continue
        c, st, lab = styles[m]
        ax.plot([p["rho"] for p in pts], [p["accuracy"] for p in pts], st, color=c,
                label=lab, markersize=5, linewidth=1.8)
    ax.axhline(full_acc, color="k", linestyle=":", linewidth=1,
               label=f"full R accuracy {full_acc:.3f}")
    ax.axhspan(0.90 * full_acc, full_acc, color="tab:green", alpha=0.08,
               label="90% of full R band")
    ax.set_xscale("log")
    ax.set_xlabel("keep-rate rho (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("SIRE v0: degradation curves (test split)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_artifacts(cfg: dict, out_dir: Path | None = None, with_png: bool = True) -> dict:
    out_dir = Path(out_dir or cfg["paths"]["work"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = render_all(cfg)
    for name in ("RESULTS.md", "FORMULA.txt", "curve.csv"):
        atomic_write_text(out_dir / name, data[name])
    if with_png:
        make_curve_png(data["curve_rows"], out_dir / "curve.png")
    return data
