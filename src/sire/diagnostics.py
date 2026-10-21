"""Best-effort failure evidence; classifications are hypotheses, never root-cause proof."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

from .util import append_jsonl, atomic_write_json, utc_stamp


def classify_failure(exc: BaseException) -> dict:
    message = str(exc).lower()
    code = getattr(exc, 'errno', None)
    category, hypothesis, action = 'unknown', 'Cause has not been determined.', 'Inspect the traceback and stage/resource records.'
    if isinstance(exc, KeyboardInterrupt):
        category, hypothesis, action = 'interrupted', 'KeyboardInterrupt or a handled termination signal was received.', 'Inspect Slurm state/ExitCode and resume with the recorded configuration.'
    elif code == errno.EDQUOT or 'disk quota exceeded' in message:
        category, hypothesis, action = 'storage_quota', 'A write was rejected by a quota limit.', 'Check the control-panel quota and usage; preserve checkpoints before removing files.'
    elif code == errno.ENOSPC or 'no space left on device' in message:
        category, hypothesis, action = 'storage_full', 'The target filesystem may have exhausted blocks or inodes.', 'Check df -h and df -i for the target path; shared free space is not a user quota.'
    elif 'out of memory' in message and ('cuda' in message or type(exc).__name__ == 'OutOfMemoryError'):
        category, hypothesis, action = 'cuda_oom', 'A CUDA allocation failed; tensor demand, workspace or fragmentation may be involved.', 'Inspect allocated/reserved peaks and the active stage; do not blindly repeat the same job.'
    elif isinstance(exc, MemoryError) or 'cannot allocate memory' in message:
        category, hypothesis, action = 'host_memory', 'Host memory allocation failed.', 'Inspect MaxRSS and the Slurm memory limit; check for full-model CPU caches.'
    elif 'cuda' in message or 'device handle' in message:
        category, hypothesis, action = 'cuda_runtime', 'A CUDA/device error occurred; this alone does not establish hardware failure.', 'Compare nvidia-smi evidence with scheduler/node logs; ask support for Xid/PCIe evidence if needed.'
    elif isinstance(exc, (ModuleNotFoundError, ImportError)):
        category, hypothesis, action = 'environment', 'A Python module or native dependency could not be imported.', 'Check the active conda environment, architecture and compatible wheel versions.'
    elif isinstance(exc, FileNotFoundError):
        category, hypothesis, action = 'missing_file', 'A required path or command was missing.', 'Check configured model/data paths and the named file in the traceback.'
    return {'category': category, 'root_cause_confirmed': False,
            'hypothesis': hypothesis, 'next_check': action}


def resource_snapshot(torch=None, paths=None) -> dict:
    snapshot = {'utc': utc_stamp(), 'pid': os.getpid(), 'host': os.uname().nodename,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'disk': {}}
    for label, path in (paths or {}).items():
        try:
            parent = Path(path).expanduser().resolve()
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            d = shutil.disk_usage(parent)
            snapshot['disk'][label] = {'path': str(parent), 'filesystem_free_bytes': d.free,
                                       'filesystem_total_bytes': d.total,
                                       'is_personal_quota': False}
        except Exception as exc:
            snapshot['disk'][label] = {'error': repr(exc)}
    if torch is not None:
        try:
            # Do not initialize CUDA just to diagnose an early import/config failure.
            if torch.cuda.is_initialized():
                snapshot['cuda'] = {
                    'allocated_bytes': torch.cuda.memory_allocated(),
                    'reserved_bytes': torch.cuda.memory_reserved(),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        except Exception as exc:
            snapshot['cuda_snapshot_error'] = repr(exc)
    return snapshot


def record_failure(directory: Path, exc: BaseException, stage: str, traceback_text: str,
                   torch=None, paths=None) -> None:
    """Never mask the original exception, even if the disk is full or GPU is lost."""
    try:
        report = {'error': repr(exc), 'error_type': type(exc).__name__, 'stage': stage,
                  'utc': utc_stamp(), 'traceback': traceback_text,
                  'diagnosis': classify_failure(exc), 'resources': resource_snapshot(torch, paths)}
        try:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total',
                                    '--format=csv'], capture_output=True, text=True, timeout=5)
            report['nvidia_smi'] = {'returncode': probe.returncode, 'stdout': probe.stdout,
                                    'stderr': probe.stderr}
        except Exception as probe_error:
            report['nvidia_smi'] = {'probe_error': repr(probe_error)}
        directory = Path(directory)
        atomic_write_json(directory / 'last_error.json', report)
        append_jsonl(directory / 'failures.jsonl', report)
    except Exception as recording_error:
        print(f'[diagnostics] Could not persist failure evidence: {recording_error!r}; '
              f'original error: {exc!r}', file=sys.stderr)
