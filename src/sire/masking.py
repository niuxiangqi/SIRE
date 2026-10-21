"""Target-tensor specification, feature loading, mask construction and in-place
weight reconstruction.

Target set S: `weight` of q/k/v/o/gate/up/down projections in every transformer
layer. All other parameters stay at R. Per tensor, exactly floor(rho*N) positions
are kept. Reconstruction uses where(mask, R, B), never subtract-add.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def discover_specs(model_dir: Path) -> list[dict]:
    """Sorted target tensor specs from the safetensors index."""
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    specs = []
    for name in sorted(weight_map):
        if not TARGET_RE.match(name):
            continue
        specs.append({"name": name, "file": weight_map[name]})
    if len(specs) != 196:
        raise RuntimeError(f"expected 196 target tensors, found {len(specs)}")
    return specs


def specs_with_shapes(base_dir: Path, reasoning_dir: Path) -> list[dict]:
    specs = discover_specs(reasoning_dir)
    from safetensors import safe_open

    bi = json.loads((Path(base_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    ri = json.loads((Path(reasoning_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    handles = {}
    for spec in specs:
        n = spec["name"]
        if n not in bi:
            raise RuntimeError(f"missing in B: {n}")
        bfile, rfile = bi[n], ri[n]
        for kind, path, file in (("B", Path(base_dir), bfile), ("R", Path(reasoning_dir), rfile)):
            key = (str(path), file)
            if key not in handles:
                handles[key] = safe_open(str(path / file), framework="pt", device="cpu")
            t = handles[key].get_slice(n)
            shape, dtype = tuple(t.get_shape()), str(t.get_dtype())
            if kind == "B":
                bshape, bdtype = shape, dtype
            else:
                if shape != bshape or dtype != bdtype:
                    raise RuntimeError(f"B/R mismatch for {n}: {bshape}{bdtype} vs {shape}{dtype}")
        out.append({"name": n, "shape": list(bshape), "dtype": bdtype,
                    "numel": int(np.prod(bshape)), "file_r": rfile, "file_b": bfile})
    return out


def total_numel(specs: list[dict]) -> int:
    return sum(s["numel"] for s in specs)


class FeatureStore:
    """Lazy per-tensor leaf access, normalised by fixed positive RMS scales."""

    def __init__(self, feature_dir: Path, base_dir: Path, reasoning_dir: Path):
        from contextlib import ExitStack

        self.dir = Path(feature_dir)
        self.base_dir = Path(base_dir)
        self.reasoning_dir = Path(reasoning_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.rms = self.meta["rms"]
        self.index = self.meta["index"]
        self.shapes = {s["name"]: tuple(s["shape"]) for s in self.meta["specs"]}
        self._cache: dict[tuple[str, str], object] = {}
        self._stack = ExitStack()
        self._handles = {}
        self._bmap = json.loads((self.base_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._rmap = json.loads((self.reasoning_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def _handle(self, root: Path, filename: str):
        from safetensors import safe_open

        key = str(root / filename)
        if key not in self._handles:
            self._handles[key] = self._stack.enter_context(
                safe_open(key, framework="pt", device="cpu"))
        return self._handles[key]

    def clear_cache(self):
        self._cache.clear()

    def _raw(self, sym: str, name: str):
        import torch

        key = (sym, name)
        if key in self._cache:
            return self._cache[key]
        if sym in ("D", "G", "U"):
            arr = np.load(self.dir / f"{int(self.index[name]):04d}.{sym}.npy")
            t = torch.from_numpy(arr)
        elif sym == "A":
            d = self._raw("D", name)
            t = torch.abs(d)
        elif sym == "B":
            t = self._handle(self.base_dir, self._bmap[name]).get_tensor(name)
        elif sym == "R":
            t = self._handle(self.reasoning_dir, self._rmap[name]).get_tensor(name)
        else:
            raise KeyError(sym)
        self._cache[key] = t
        return t

    def leaf(self, sym: str, name: str):
        import torch

        t = self._raw(sym, name)
        scale = float(self.rms[name][sym])
        if not (scale > 0):
            raise RuntimeError(f"non-positive RMS for {sym} {name}: {scale}")
        return (t.to(torch.float32) / scale).reshape(self.shapes[name])

    def close(self):
        self._cache.clear()
        self._stack.close()


def tensor_score(expr, name: str, store: FeatureStore):
    import torch
    from . import dsl

    store.clear_cache()
    needed = dsl.leaves_of(expr)
    leaves = {s: store.leaf(s, name) for s in needed}
    gpu = {s: t.cuda(non_blocking=True) for s, t in leaves.items()}
    with torch.no_grad():
        score = dsl.evaluate(expr, gpu)
    if score.shape != gpu[next(iter(needed))].shape:
        raise RuntimeError("score shape mismatch")
    if not torch.isfinite(score).all():
        raise FloatingPointError(f"non-finite score in {name}")
    del gpu, leaves
    store.clear_cache()
    return score.reshape(-1)


def topk_indices(score, k: int):
    """Deterministic top-k: descending score, ties by flat index ascending."""
    import torch

    n = score.numel()
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside [0,{n}]")
    if k == 0:
        return torch.empty(0, dtype=torch.int64)
    if k == n:
        return torch.arange(n, dtype=torch.int64)
    t = torch.kthvalue(score, n - k + 1).values
    gt = torch.nonzero(score > t, as_tuple=False).flatten()
    need = k - gt.numel()
    if need > 0:
        eq = torch.nonzero(score == t, as_tuple=False).flatten()
        kept = torch.cat([gt, eq[:need]])
    else:
        kept = gt[:k]
    return torch.sort(kept).values


def mask_hash(name: str, kept) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(np.asarray(kept, dtype="<i4").tobytes())
    return h.hexdigest()


def topk_indices_gpu(score, k: int):
    """Deterministic top-k on GPU: descending score, ties by flat index."""
    import torch

    n = score.numel()
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside [0,{n}]")
    if k == 0:
        return torch.empty(0, dtype=torch.int64, device=score.device)
    if k == n:
        return torch.arange(n, dtype=torch.int64, device=score.device)
    vals, idx = torch.topk(score, k, largest=True, sorted=False)
    t = vals.min()
    count_gt = int((score > t).sum().item())
    need = k - count_gt
    if need > 0:
        eq = torch.nonzero(score == t, as_tuple=False).flatten()
        gt = torch.nonzero(score > t, as_tuple=False).flatten()
        kept = torch.cat([gt, eq[:need]])
        if kept.numel() != k:
            raise RuntimeError(f"top-k tie resolution produced {kept.numel()} != {k}")
    else:
        kept = idx
    return kept.sort().values


def random_indices(numel: int, k: int, seed_key: str):
    """Deterministic random retention, independent of evaluation order."""
    if k == 0:
        return import_torch().empty(0, dtype=torch.int64)
    if k == numel:
        return import_torch().arange(numel, dtype=torch.int64)
    import torch

    h = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    g = torch.Generator(device="cuda").manual_seed(h & ((1 << 63) - 1))
    return torch.randperm(numel, device="cuda", generator=g)[:k].sort().values


def import_torch():
    import torch
    return torch


def build_apply(model, specs, r_ref, b_ref, store, expr=None, rho: float = None,
                random_seed: str | None = None, rho_value: float | None = None,
                method: str = "formula") -> dict:
    """Build the mask for this (method, rho) and swap target weights in place.

    Returns an audit dict with per-tensor quota, kept hash and the overall mask
    hash. No full mask is persisted; verify recomputes it from frozen inputs.
    """
    import torch

    rho = float(rho if rho is not None else rho_value)
    total_h = hashlib.sha256()
    tensors = []
    for spec in specs:
        name, n = spec["name"], spec["numel"]
        k = int(math.floor(rho * n))
        if method == "random":
            if random_seed is None:
                raise ValueError("random retention needs a seed")
            kept = random_indices(n, k, f"{random_seed}|{rho:.6f}|{name}")
        elif method == "all":
            kept = torch.arange(n, dtype=torch.int64, device="cuda")
        else:
            score = tensor_score(expr, name, store)
            kept = topk_indices_gpu(score, k)
            del score
        if method == "all" and rho != 1.0:
            raise ValueError("all-mask only valid at rho=1")
        h = mask_hash(name, kept.detach().cpu())
        total_h.update(name.encode())
        total_h.update(h.encode())
        tensors.append({"name": name, "numel": n, "quota": k, "rho": rho,
                        "kept_sha256": h})
        p = model.get_parameter(name)
        if k == n:
            p.data.copy_(r_ref[name])
        else:
            mask = torch.zeros(n, dtype=torch.bool, device=p.device)
            if k:
                mask[kept] = True
            p.data.copy_(torch.where(mask.view(spec["shape"]), r_ref[name], b_ref[name]))
            del mask
        del kept
    torch.cuda.synchronize()
    return {"method": method, "rho": rho, "formula_ast": None,
            "mask_sha256": total_h.hexdigest(), "tensors": tensors,
            "total_numel": sum(t["numel"] for t in tensors),
            "total_kept": sum(t["quota"] for t in tensors)}


class CheckpointRefs:
    """Read pristine tensors on demand; never cache CUDA copies."""
    def __init__(self, root, device, dtype):
        self.root = Path(root)
        self.device, self.dtype = device, dtype
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, name):
        from safetensors import safe_open
        with safe_open(str(self.root / self.weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(device=self.device, dtype=self.dtype)


class ModelRefs:
    """One resident R model, pristine R/B copies for target tensors."""

    def __init__(self, cfg: dict):
        import torch
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.cfg = cfg
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["paths"]["reasoning_model"], dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True)
        self.model.eval()
        self.specs = specs_with_shapes(Path(cfg["paths"]["base_model"]),
                                       Path(cfg["paths"]["reasoning_model"]))
        storage = cfg.get("runtime", {}).get("reference_storage", "gpu")
        if storage == "stream":
            self.r_ref = CheckpointRefs(cfg["paths"]["reasoning_model"], "cuda:0", torch.bfloat16)
            self.b_ref = CheckpointRefs(cfg["paths"]["base_model"], "cuda:0", torch.bfloat16)
            return
        if storage != "gpu":
            raise ValueError(f"unsupported reference storage: {storage}")
        self.r_ref, self.b_ref = {}, {}
        b_index = json.loads((Path(cfg["paths"]["base_model"]) /
                              "model.safetensors.index.json").read_text())["weight_map"]
        handles = {}
        with torch.no_grad():
            for spec in self.specs:
                name = spec["name"]
                p = self.model.get_parameter(name)
                self.r_ref[name] = p.detach().clone()
                f = b_index[name]
                if f not in handles:
                    handles[f] = safe_open(str(Path(cfg["paths"]["base_model"]) / f),
                                           framework="pt", device="cpu")
                b = handles[f].get_tensor(name)
                if tuple(b.shape) != tuple(p.shape):
                    raise RuntimeError(f"shape mismatch {name}")
                self.b_ref[name] = b.to(device=p.device, dtype=p.dtype, non_blocking=True)
        torch.cuda.synchronize()

    def apply(self, store: FeatureStore, expr=None, rho: float = 1.0,
              method: str = "formula", random_seed: str | None = None) -> dict:
        return build_apply(self.model, self.specs, self.r_ref, self.b_ref, store,
                           expr=expr, rho=rho, method=method, random_seed=random_seed)

    def apply_method(self, store, method, rho, cfg):
        if method == "full_R":
            return self.apply(store, rho=1.0, method="all")
        if method == "random":
            return self.apply(store, rho=rho, method="random",
                              random_seed=str(cfg["baselines"]["random_seed"]))
        if method == "magnitude_A":
            from . import dsl
            return self.apply(store, expr=dsl.parse("A"), rho=rho)
        if method == "low_grad_NegU":
            from . import dsl
            return self.apply(store, expr=dsl.parse("Neg(U)"), rho=rho)
        raise KeyError(method)
