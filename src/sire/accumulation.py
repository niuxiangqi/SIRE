"""Bounded, transactional per-tensor FP32 gradient accumulation.

Each tensor owns its sample cursor. Replaying a partially committed sample skips
already committed tensors. A crash before os.replace leaves the prior checkpoint
intact. Only one tensor's statistics are loaded at once (no model-wide RAM copy).
"""
from __future__ import annotations

import os
from pathlib import Path

from .util import atomic_write_json, read_json


class TensorAccumulators:
    def __init__(self, directory: Path, specs: list[dict], identity: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.specs = specs
        manifest = self.directory / "identity.json"
        old = read_json(manifest)
        if old is not None and old != identity:
            raise ValueError("gradient checkpoint identity changed; use a new run")
        atomic_write_json(manifest, identity)
        self.paths = {s["name"]: self.directory / f"{i:04d}.pt"
                      for i, s in enumerate(specs)}

    def read(self, name):
        import torch
        path = self.paths[name]
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def cursor(self):
        cursors = []
        for spec in self.specs:
            state = self.read(spec["name"])
            cursors.append(state["cursor"] if state else 0)
            del state
        if not cursors:
            raise ValueError("no target tensors")
        if max(cursors) - min(cursors) > 1:
            raise ValueError("inconsistent per-tensor sample cursors")
        return min(cursors)

    def add(self, name, sample_index, gradient):
        import torch
        state = self.read(name)
        cursor = state["cursor"] if state else 0
        if cursor == sample_index + 1:
            return  # interrupted sample replay: this tensor was committed
        if cursor != sample_index:
            raise ValueError(f"unexpected sample cursor {cursor} for {name}: {sample_index}")
        g = gradient.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not torch.isfinite(g).all():
            raise FloatingPointError(f"non-finite gradient in {name}")
        if state is None:
            state = {"g_sum": g.clone(), "u_sum": g.abs()}
        else:
            state["g_sum"].add_(g)
            state["u_sum"].add_(g.abs())
        state["cursor"] = sample_index + 1
        path = self.paths[name]
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def means(self, name, count):
        state = self.read(name)
        if not state or state["cursor"] != count:
            raise ValueError(f"incomplete statistics for {name}")
        return state["g_sum"].div_(count), state["u_sum"].div_(count)

    def cleanup(self):
        for path in self.paths.values():
            path.unlink(missing_ok=True)
            path.with_suffix(".tmp").unlink(missing_ok=True)
        (self.directory / "identity.json").unlink(missing_ok=True)
        self.directory.rmdir()
