import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sire.accumulation import TensorAccumulators
from sire.cli import apply_overrides, resolve_run_config
from sire.masking import CheckpointRefs, FeatureStore
from sire.util import atomic_write_text, configure_gpu4, load_config
import yaml

ROOT = Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def test_profiles_and_overrides(self):
        cfg = load_config(ROOT / 'configs/a100_40gb.yaml')
        self.assertEqual(cfg['features']['param_dtype'], 'bfloat16')
        self.assertTrue(Path(cfg['paths']['work']).is_absolute())
        new = apply_overrides(cfg, ['device.mode=physical', 'device.physical_index=2',
                                    'features.param_dtype=float32', 'generation.batch_size=2'])
        self.assertEqual(new['device']['physical_index'], 2)
        self.assertEqual(cfg['device']['mode'], 'slurm')
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ['generation.batch_size=0'])
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ['features.param_dtype=float16'])

    def test_slurm_visibility_is_not_rewritten(self):
        cfg = {'device': {'mode': 'slurm'}}
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '3'}, clear=True):
            with patch('sire.util.nvidia_smi_uuid', side_effect=AssertionError('physical lookup')):
                evidence = configure_gpu4(cfg)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(evidence['slurm_job_id'], '123')
        for mask in ['', '-1', '0,1']:
            with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': mask}, clear=True):
                with self.assertRaises(RuntimeError):
                    configure_gpu4(cfg)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-abc'}, clear=True):
            with self.assertRaises(RuntimeError):
                configure_gpu4(cfg)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True):
            with self.assertRaises(RuntimeError):
                configure_gpu4({'device': {'mode': 'physical'}})

    def test_physical_pin(self):
        cfg = {'device': {'mode': 'physical', 'physical_index': 4, 'expected_uuid': 'GPU-a'}}
        with patch.dict(os.environ, {}, clear=True), patch('sire.util.nvidia_smi_uuid', return_value='GPU-a'):
            self.assertEqual(configure_gpu4(cfg)['uuid'], 'GPU-a')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-a')
        with patch.dict(os.environ, {}, clear=True), patch('sire.util.nvidia_smi_uuid', return_value='GPU-b'):
            with self.assertRaises(RuntimeError):
                configure_gpu4(cfg)

    def test_accumulators_replay_interrupted_sample(self):
        specs = [{'name': 'a', 'numel': 3}, {'name': 'b', 'numel': 3}]
        gradients = [torch.tensor([1., -2., 3.]), torch.tensor([-3., 4., -1.])]
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td) / 'stats'
            acc = TensorAccumulators(directory, specs, {'dtype': 'bfloat16'})
            acc.add('a', 0, gradients[0])  # interrupt between tensors
            self.assertEqual(acc.cursor(), 0)
            # A failed write for b must not leave an advanced cursor.
            with patch('sire.accumulation.os.replace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    acc.add('b', 0, gradients[0])
            acc = TensorAccumulators(directory, specs, {'dtype': 'bfloat16'})
            for i, g in enumerate(gradients):
                for spec in specs:
                    acc.add(spec['name'], i, g.to(torch.bfloat16))
            self.assertEqual(acc.cursor(), 2)
            for spec in specs:
                G, U = acc.means(spec['name'], 2)
                self.assertTrue(torch.equal(G, torch.stack(gradients).mean(0)))
                self.assertTrue(torch.equal(U, torch.stack(gradients).abs().mean(0)))
                self.assertEqual(G.dtype, torch.float32)
                self.assertFalse(torch.equal(U, G.abs()))
            with self.assertRaises(ValueError):
                TensorAccumulators(directory, specs, {'dtype': 'float32'})
            acc.cleanup()
            self.assertFalse(directory.exists())

    def test_streamed_refs_pristine_reconstruction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = torch.tensor([1., 2., 3.], dtype=torch.bfloat16)
            reas = torch.tensor([4., 5., 6.], dtype=torch.bfloat16)
            for role, tensor in [('B', base), ('R', reas)]:
                folder = root / role
                folder.mkdir()
                save_file({'w': tensor}, folder / 'model.safetensors')
                (folder / 'model.safetensors.index.json').write_text(json.dumps(
                    {'weight_map': {'w': 'model.safetensors'}}))
            B = CheckpointRefs(root / 'B', 'cpu', torch.bfloat16)
            R = CheckpointRefs(root / 'R', 'cpu', torch.bfloat16)
            self.assertTrue(torch.equal(torch.where(torch.ones(3, dtype=torch.bool), R['w'], B['w']), reas))
            mutable = R['w'].clone()
            mutable.fill_(0)
            self.assertTrue(torch.equal(R['w'], reas))
            for mask in [torch.tensor([True, False, True]), torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)]:
                mutable.copy_(torch.where(mask, R['w'], B['w']))
            self.assertTrue(torch.equal(mutable, reas))

    def test_feature_cache_can_be_released(self):
        store = FeatureStore.__new__(FeatureStore)
        store._cache = {('D', 'a'): torch.ones(10)}
        store.clear_cache()
        self.assertEqual(len(store._cache), 0)

    def test_resume_rejects_precision_change_before_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(ROOT / 'configs/a100_40gb.yaml')
            cfg['paths']['runs'] = td
            run = Path(td) / 'test'
            run.mkdir()
            atomic_write_text(run / 'runtime_config.yaml', yaml.safe_dump(cfg))
            self.assertEqual(resolve_run_config(cfg, 'test', True), cfg)
            changed = copy.deepcopy(cfg)
            changed['features']['param_dtype'] = 'float32'
            with self.assertRaisesRegex(ValueError, 'features'):
                resolve_run_config(changed, 'test', True)

    def test_bf16_disk_feature_pipeline_on_tiny_model(self):
        import types
        import numpy as np
        from sire.features import collect_gradients
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
                self.config = types.SimpleNamespace(use_cache=True)
            def gradient_checkpointing_enable(self):
                pass
            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(loss=(self.w * input_ids.reshape(-1)).sum())
        fake = types.ModuleType('transformers')
        model = Tiny()
        class Auto:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                self.assertEqual(kwargs['dtype'], torch.bfloat16)
                return model
        fake.AutoModelForCausalLM = Auto
        tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return tensor(*args, **kwargs)
        event = types.SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 1.0)
        examples = [{'item_id': str(i), 'input_ids': g, 'labels': g}
                    for i, g in enumerate([[1, -2, 3], [-3, 4, -1]])]
        specs = [{'name': 'w', 'numel': 3, 'shape': [3]}]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'features').mkdir()
            (root / 'data/prepared').mkdir(parents=True)
            (root / 'data/prepared/splits.json').write_text('{}')
            cfg = {'paths': {'reasoning_model': str(root / 'model'), 'data_root': str(root / 'data')},
                   'features': {'param_dtype': 'bfloat16', 'accumulator': 'disk',
                                'grad_checkpointing': True, 'max_seq_len': 2048}}
            with patch.dict('sys.modules', {'transformers': fake}), \
                 patch('sire.features._calibration_examples', return_value=examples), \
                 patch('torch.tensor', side_effect=cpu_tensor), \
                 patch('torch.cuda.is_bf16_supported', return_value=True), \
                 patch('torch.cuda.Event', return_value=event), \
                 patch('torch.cuda.synchronize'), patch('torch.cuda.empty_cache'):
                meta = collect_gradients(cfg, root, {}, specs, {'w': 0}, {'w': {'D': 1.0}})
            self.assertEqual(meta['param_dtype'], 'bfloat16')
            np.testing.assert_array_equal(np.load(root / 'features/0000.G.npy'), [-1., 1., 1.])
            np.testing.assert_array_equal(np.load(root / 'features/0000.U.npy'), [2., 3., 2.])

    def test_allocation_verification_rejects_wrong_uuid(self):
        from sire.verify import verify
        from sire.util import atomic_write_json, append_jsonl
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(ROOT / 'configs/a100_40gb.yaml')
            cfg['paths'].update({'runs': td, 'work': td, 'data_root': td})
            run = Path(td) / 'check'
            run.mkdir()
            atomic_write_json(run / 'processes.json', [123, 456])
            atomic_write_json(run / 'env.json', {
                'conda_prefix': '/opt/conda/envs/sire',
                'sys_executable': '/opt/conda/envs/sire/bin/python',
                'gpu': {'pid': 123, 'uuid': 'GPU-first'}, 'torch': {'visible_count': 1}})
            for pid, uuid in [(123, 'GPU-first'), (456, 'GPU-second')]:
                append_jsonl(run / 'allocations.jsonl', {'pid': pid, 'uuid': uuid,
                    'CUDA_VISIBLE_DEVICES': '0', 'slurm_job_id': str(pid)})
                append_jsonl(run / 'gpu_monitor.jsonl', {'task_pid': pid, 'uuid': uuid})
            self.assertTrue(verify(cfg, 'check')['checks'][0]['passed'])
            append_jsonl(run / 'gpu_monitor.jsonl', {'task_pid': 456, 'uuid': 'GPU-other-user'})
            self.assertFalse(verify(cfg, 'check')['checks'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
