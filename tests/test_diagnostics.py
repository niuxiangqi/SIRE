import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sire.diagnostics import classify_failure, record_failure, resource_snapshot


class DiagnosticsTests(unittest.TestCase):
    def test_categories_are_hypotheses(self):
        examples = [(OSError(errno.EDQUOT, 'Disk quota exceeded'), 'storage_quota'),
                    (OSError(errno.ENOSPC, 'No space left on device'), 'storage_full'),
                    (RuntimeError('CUDA out of memory'), 'cuda_oom'),
                    (RuntimeError('CUDA error: unknown error'), 'cuda_runtime'),
                    (MemoryError(), 'host_memory'),
                    (ModuleNotFoundError('No module named transformers'), 'environment')]
        for error, expected in examples:
            report = classify_failure(error)
            self.assertEqual(report['category'], expected)
            self.assertFalse(report['root_cause_confirmed'])

    def test_failure_history_and_probe_failure(self):
        with tempfile.TemporaryDirectory() as td:
            with patch('sire.diagnostics.subprocess.run', side_effect=FileNotFoundError('nvidia-smi')):
                record_failure(Path(td), RuntimeError('CUDA out of memory'), 'features', 'traceback',
                               paths={'run': td})
                record_failure(Path(td), MemoryError(), 'features', 'traceback')
            history = (Path(td) / 'failures.jsonl').read_text().splitlines()
            self.assertEqual(len(history), 2)
            self.assertEqual(json.loads(history[0])['stage'], 'features')
            self.assertEqual(json.loads((Path(td) / 'last_error.json').read_text())['diagnosis']['category'], 'host_memory')

    def test_full_disk_does_not_mask_original_failure(self):
        with patch('sire.diagnostics.atomic_write_json', side_effect=OSError(errno.ENOSPC, 'full')), \
             patch('sire.diagnostics.subprocess.run', side_effect=FileNotFoundError()), \
             patch('sys.stderr'):
            record_failure(Path('/not-written'), RuntimeError('original'), 'test', 'trace')

    def test_filesystem_free_is_never_a_personal_quota(self):
        snapshot = resource_snapshot(paths={'work': '.'})
        self.assertFalse(snapshot['disk']['work']['is_personal_quota'])


if __name__ == '__main__':
    unittest.main()
