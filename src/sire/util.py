"""Shared utilities: hashing, atomic writes, timing ledger, GPU isolation."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

WORK = Path("/home/nxq/SIRE")


# ----------------------------------------------------------------------------
# Hashing / atomic writes
# ----------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False) + "\n")


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_config(path: str | Path = None) -> dict:
    if path is None:
        path = WORK / "configs" / "v0.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# GPU isolation: resolve UUID of physical index, set CUDA_VISIBLE_DEVICES
# ----------------------------------------------------------------------------
def nvidia_smi_uuid(index: int) -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    for line in out.strip().splitlines():
        i, uuid = [x.strip() for x in line.split(",")]
        if int(i) == index:
            return uuid
    raise RuntimeError(f"physical GPU {index} not found")


def configure_gpu4(cfg: dict) -> dict:
    """Must be called before importing torch. Returns an evidence dict."""
    idx = cfg["device"]["physical_index"]
    expected = cfg["device"]["expected_uuid"]
    uuid = nvidia_smi_uuid(idx)
    if uuid != expected:
        raise RuntimeError(f"GPU {idx} UUID {uuid} != expected {expected}")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return {"physical_index": idx, "uuid": uuid, "pid": os.getpid(), "host": socket.gethostname(),
            "CUDA_VISIBLE_DEVICES": uuid}


def import_torch_checked():
    import torch

    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {n}")
    name = torch.cuda.get_device_name(0)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    return torch, {"visible_count": n, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device_name": name, "device_uuid": uuid,
                   "sys_executable": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX")}


# ----------------------------------------------------------------------------
# Budget ledger
# ----------------------------------------------------------------------------
class BudgetLedger:
    """GPU-stage ledger. GPU seconds are wall-clock of stages that use the GPU,
    bracketed by cuda synchronisation. CPU/IO and total wall are recorded too."""

    def __init__(self, path: Path, total: float, grace: float):
        self.path = Path(path)
        self.total = float(total)
        self.grace = float(grace)
        self.entries: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.entries = data.get("entries", [])

    @property
    def gpu_used(self) -> float:
        return sum(e.get("gpu_seconds", 0.0) for e in self.entries)

    def remaining(self) -> float:
        return self.total - self.gpu_used

    def would_exceed(self, predicted: float) -> bool:
        return self.gpu_used + predicted > self.total + self.grace

    def add(self, stage: str, wall: float, gpu: float | None = None, cuda: float | None = None,
            note: str = "", detail: dict | None = None) -> dict:
        e = {
            "stage": stage,
            "wall_seconds": round(float(wall), 3),
            "gpu_seconds": round(float(gpu if gpu is not None else wall), 3),
            "cuda_seconds": None if cuda is None else round(float(cuda), 3),
            "note": note,
            "ts": time.time(),
        }
        if detail:
            e["detail"] = detail
        self.entries.append(e)
        self.save()
        return e

    def save(self) -> None:
        atomic_write_json(self.path, {
            "total_gpu_seconds": self.total,
            "grace_gpu_seconds": self.grace,
            "gpu_used_seconds": round(self.gpu_used, 3),
            "remaining_seconds": round(self.remaining(), 3),
            "entries": self.entries,
        })

    @contextlib.contextmanager
    def stage(self, name: str, torch=None, gpu: bool = True, note: str = "", detail: dict | None = None):
        if torch is not None and gpu:
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        t0 = time.monotonic()
        info = {}
        try:
            yield info
        finally:
            wall = time.monotonic() - t0
            cuda = None
            if torch is not None and gpu:
                ev1.record()
                torch.cuda.synchronize()
                cuda = ev0.elapsed_time(ev1) / 1000.0
            self.add(name, wall, wall if gpu else 0.0, cuda, note, detail)


class GpuMonitor(threading.Thread):
    """Sample the target UUID, task PID, utilization and memory every ~5 s."""

    def __init__(self, path: Path, expected_uuid: str, interval: float = 5.0):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.expected_uuid = expected_uuid
        self.interval = interval
        self._stop_event = threading.Event()
        self.pid = os.getpid()

    def run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=10)
                rec = None
                for line in out.strip().splitlines():
                    uuid, util, used, total = [x.strip() for x in line.split(",")]
                    if uuid == self.expected_uuid:
                        rec = {"ts": time.time(), "uuid": uuid, "utilization_pct": float(util),
                               "memory_used_mib": float(used), "memory_total_mib": float(total),
                               "task_pid": self.pid}
                        break
                if rec is None:
                    rec = {"ts": time.time(), "error": "target UUID not visible",
                           "expected_uuid": self.expected_uuid, "task_pid": self.pid}
                append_jsonl(self.path, rec)
            except Exception as exc:  # monitoring must never kill the run
                append_jsonl(self.path, {"ts": time.time(), "error": repr(exc), "task_pid": self.pid})
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=15)


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id_now(prefix: str = "v0") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
