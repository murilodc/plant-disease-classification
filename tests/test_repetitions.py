"""Repetition tests use synthetic artifacts only; no training or real images."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import test_compare_models as fixtures
from compare_models import MODEL_SPECS
from compare_repetitions import compare_repetitions
from run_repetition import prepare_repetition, run_repetition, split_sha256


class RepetitionTests(unittest.TestCase):
    def setUp(self):
        fixtures.ComparisonTests.setUp(self)
        self.project = self.root.parent
        (self.project / 'src').mkdir()
        self.images = self.project / 'images'
        self.images.mkdir()
        (self.root / 'plantvillage_metadata_split.csv').write_text('fixed synthetic split', encoding='utf-8')
        for spec in MODEL_SPECS:
            path = self.root / spec.key / 'experiment_metadata.json'
            metadata = json.loads(path.read_text())
            metadata['gpu_name'] = 'NVIDIA L4'
            metadata['learning_rate'] = .001 if spec.key == 'baseline_cnn' else .0001
            metadata['optimizer'] = 'Adam'
            path.write_text(json.dumps(metadata), encoding='utf-8')

    write_json = fixtures.ComparisonTests.write_json

    def add_runs(self):
        for seed, scale in ((123, 2), (456, 3)):
            for spec in MODEL_SPECS:
                original = self.root / spec.key
                output = self.root / 'repetitions' / f'seed_{seed}' / spec.key
                shutil.copytree(original, output)
                path = output / 'experiment_metadata.json'
                metadata = json.loads(path.read_text())
                metadata['seed'] = seed
                value = .8 if seed == 123 else 1.
                metadata['best_model_validation_accuracy'] = value
                metrics = pd.read_csv(output / 'validation_metrics.csv')
                metrics.loc[:, ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                                'precision_weighted', 'recall_weighted', 'f1_weighted']] = value
                metrics.to_csv(output / 'validation_metrics.csv', index=False)
                classes = pd.read_csv(output / 'validation_metrics_by_class.csv')
                classes.loc[:, ['precision', 'recall', 'f1_score']] = value
                classes.to_csv(output / 'validation_metrics_by_class.csv', index=False)
                path.write_text(json.dumps(metadata))
                history = pd.read_csv(output / spec.history_name)
                history.epoch_time_seconds *= scale
                history.to_csv(output / spec.history_name, index=False)
                path = output / 'timing_summary.json'
                timing = json.loads(path.read_text())
                for field in ('total_time_seconds', 'average_time_per_epoch_seconds'):
                    timing['training'][field] *= scale
                for field in ('total_time_seconds', 'average_time_per_image_seconds'):
                    timing['validation_inference'][field] *= scale
                timing['validation_inference']['throughput_images_per_second'] /= scale
                path.write_text(json.dumps(timing))
                manifest = dict(status='complete', model=spec.key, seed=seed,
                                metadata_csv_sha256=split_sha256(self.root / 'plantvillage_metadata_split.csv'),
                                reference_metadata_sha256=split_sha256(original / 'experiment_metadata.json'))
                (output / 'repetition_manifest.json').write_text(json.dumps(manifest))

    def test_three_seeds_mean_and_sample_std_and_preserved_sources(self):
        self.add_runs()
        before = {p: split_sha256(p) for p in self.root.rglob('*') if p.is_file()}
        result = compare_repetitions(self.root)
        self.assertEqual(result.individual.shape[0], 12)
        self.assertEqual(result.aggregate.n_runs.tolist(), [3] * 4)
        self.assertEqual(result.warnings, [])
        for row in result.aggregate.itertuples():
            self.assertAlmostEqual(row.training_time_seconds_mean, 44)
            self.assertAlmostEqual(row.training_time_seconds_std, 22)
            self.assertAlmostEqual(row.inference_seconds_per_image_std, .001)
            self.assertAlmostEqual(row.accuracy_mean, .9)
            self.assertAlmostEqual(row.accuracy_std, .1)
        self.assertEqual(len(result.per_class), 12 * 38)
        self.assertEqual(len(result.histories), 24)
        for path, checksum in before.items():
            self.assertEqual(split_sha256(path), checksum)

    def test_pending_runs_have_nan_std_not_zero(self):
        result = compare_repetitions(self.root, save=False)
        self.assertEqual(result.aggregate.n_runs.tolist(), [1] * 4)
        self.assertTrue(result.aggregate.accuracy_std.isna().all())
        self.assertEqual(result.aggregate.status.tolist(), ['partial'] * 4)
        shutil.rmtree(self.root / 'baseline_cnn')
        result = compare_repetitions(self.root, save=False)
        self.assertEqual(result.aggregate.iloc[0].n_runs, 0)
        self.assertTrue(pd.isna(result.aggregate.iloc[0].accuracy_mean))

    def test_mismatched_seed_and_split_provenance_are_excluded(self):
        self.add_runs()
        directory = self.root / 'repetitions/seed_123/baseline_cnn'
        path = directory / 'experiment_metadata.json'
        metadata = json.loads(path.read_text())
        metadata['seed'] = 456
        path.write_text(json.dumps(metadata))
        path = self.root / 'repetitions/seed_456/resnet50/repetition_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['metadata_csv_sha256'] = 'wrong'
        path.write_text(json.dumps(manifest))
        result = compare_repetitions(self.root, save=False)
        self.assertEqual(result.aggregate.n_runs.tolist(), [2, 2, 3, 3])
        self.assertEqual((result.individual.status == 'invalid').sum(), 2)

    def test_hardware_difference_warns(self):
        self.add_runs()
        path = self.root / 'repetitions/seed_456/resnet50/experiment_metadata.json'
        metadata = json.loads(path.read_text())
        metadata['gpu_name'] = 'different'
        path.write_text(json.dumps(metadata))
        result = compare_repetitions(self.root, save=False)
        self.assertTrue(any('Hardware diferente' in w for w in result.warnings))

    def test_preflight_all_models_and_seeds_has_no_side_effects(self):
        with patch('run_repetition.subprocess.run', side_effect=AssertionError('training forbidden')):
            for spec in MODEL_SPECS:
                for seed in (123, 456):
                    command, manifest = prepare_repetition(self.project, spec.key, seed, self.images)
                    self.assertIn(str(seed), command)
                    self.assertEqual(command[command.index('--batch-size') + 1], '32')
                    self.assertEqual(command[command.index('--num-workers') + 1], '2')
                    self.assertFalse(Path(manifest['output_dir']).exists())
                    self.assertNotIn('--test', command)
        with self.assertRaises(ValueError):
            prepare_repetition(self.project, 'baseline_cnn', 42, self.images)
        output = self.root / 'repetitions/seed_123/baseline_cnn'
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            run_repetition(self.project, 'baseline_cnn', 123, self.images)

    def test_trainers_guard_existing_outputs_before_training(self):
        import importlib
        for module, function in [('train_baseline', 'train_baseline'),
                                 ('train_resnet50', 'train_resnet50'),
                                 ('train_densenet201', 'train_densenet201'),
                                 ('train_efficientnet_b0', 'train_efficientnet_b0')]:
            trainer = importlib.import_module(module)
            with patch.object(trainer, 'set_seed', side_effect=AssertionError('must stop earlier')):
                with self.assertRaises(FileExistsError):
                    getattr(trainer, function)('absent.csv', self.images, self.root / 'baseline_cnn', seed=123)

    def test_loaders_never_construct_test_dataset(self):
        import plantvillage_pytorch as pipeline
        from unittest.mock import MagicMock
        metadata = pd.DataFrame({'classe': [f'c{i}' for i in range(38)]})
        with patch.object(pipeline, 'PlantVillageDataset', return_value=MagicMock()) as dataset, \
             patch.object(pipeline, 'DataLoader'):
            loaders = pipeline.create_dataloaders(metadata, self.images, splits=('train', 'validation'))
        self.assertEqual(set(loaders), {'train', 'validation'})
        self.assertEqual([c.kwargs['split'] for c in dataset.call_args_list], ['train', 'validation'])
        for path in (Path(__file__).resolve().parents[1] / 'src').glob('train_*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == 'create_dataloaders']
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.literal_eval(next(k.value for k in calls[0].keywords if k.arg == 'splits')),
                             ('train', 'validation'))

    def test_notebook_valid_and_training_disabled(self):
        import nbformat
        path = Path(__file__).resolve().parents[1] / 'notebooks/07_repeticoes_individuais.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(path), 'exec')
                self.assertEqual(cell.outputs, [])
                if cell.id == 'selection':
                    exec(cell.source, namespace)
        self.assertFalse(namespace['EXECUTE_TRAINING'])
        self.assertEqual(namespace['SEED'], 123)
        namespace.update(PROJECT_ROOT=self.project, IMAGE_ROOT=self.images,
                         display=lambda value: None)
        with patch('run_repetition.subprocess.run', side_effect=AssertionError('training forbidden')):
            for cell in notebook.cells:
                if cell.id in ('preflight', 'train', 'compare'):
                    exec(cell.source, namespace)

    def test_runner_checks_hardware_and_records_mocked_launch(self):
        import torch
        import torchvision
        for spec in MODEL_SPECS:
            path = self.root / spec.key / 'experiment_metadata.json'
            metadata = json.loads(path.read_text())
            metadata.update(pytorch_version=torch.__version__, pytorch_cuda_version=torch.version.cuda,
                            torchvision_version=torchvision.__version__)
            path.write_text(json.dumps(metadata))
        before = split_sha256(self.root / 'baseline_cnn/experiment_metadata.json')
        with patch('torch.cuda.is_available', return_value=False), \
             patch('run_repetition.subprocess.run') as launch:
            with self.assertRaises(RuntimeError):
                run_repetition(self.project, 'baseline_cnn', 123, self.images)
            launch.assert_not_called()
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA L4'), \
             patch('run_repetition.subprocess.run') as launch:
            output = run_repetition(self.project, 'baseline_cnn', 123, self.images)
            launch.assert_called_once()
        self.assertEqual(json.loads((output / 'repetition_manifest.json').read_text())['status'], 'complete')
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA L4'), \
             patch('run_repetition.subprocess.run', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                run_repetition(self.project, 'baseline_cnn', 456, self.images)
        failed = self.root / 'repetitions/seed_456/baseline_cnn/repetition_manifest.json'
        self.assertEqual(json.loads(failed.read_text())['status'], 'failed')
        self.assertEqual(split_sha256(self.root / 'baseline_cnn/experiment_metadata.json'), before)

    def test_changed_split_is_blocked(self):
        self.add_runs()
        (self.root / 'plantvillage_metadata_split.csv').write_text('changed')
        shutil.rmtree(self.root / 'repetitions/seed_123/baseline_cnn')
        with self.assertRaisesRegex(ValueError, 'splits diverge'):
            prepare_repetition(self.project, 'baseline_cnn', 123, self.images)

    def test_comparison_cannot_write_inside_individual_run(self):
        with self.assertRaises(ValueError):
            compare_repetitions(self.root, output_dir=self.root / 'baseline_cnn')


if __name__ == '__main__':
    unittest.main()
