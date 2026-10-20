"""Validation-only regression tests; all artifacts are synthetic and temporary."""

import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from compare_models import MODEL_SPECS, compare_validation_results


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "results"
        for spec in MODEL_SPECS:
            directory = self.root / spec.directory_name
            directory.mkdir(parents=True)
            # Only existence/size is inspected; no checkpoint is loaded.
            (directory / spec.checkpoint_name).write_bytes(b"synthetic checkpoint")
            (directory / "validation_confusion_matrix.png").write_bytes(b"placeholder")
            metrics = {name: 0.9 for name in (
                "accuracy", "precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted",
            )}
            pd.DataFrame([dict(metrics, num_samples=8172, num_classes=38)]).to_csv(
                directory / "validation_metrics.csv", index=False
            )
            pd.DataFrame({
                "class_index": range(38),
                "class_name": [f"class_{i}" for i in range(38)],
                "precision": [0.9] * 38, "recall": [0.9] * 38,
                "f1_score": [0.9] * 38, "support": [215] * 37 + [217],
            }).to_csv(directory / "validation_metrics_by_class.csv", index=False)
            pd.DataFrame({
                "epoch": [1, 2], "train_loss": [0.4, 0.2],
                "train_accuracy": [0.8, 0.9], "validation_loss": [0.5, 0.3],
                "validation_accuracy": [0.8, 0.9], "epoch_time_seconds": [10, 12],
            }).to_csv(directory / spec.history_name, index=False)
            self.write_json(spec, "architecture_summary.json", {
                "total_parameters": 1000, "trainable_parameters": 1000,
            })
            self.write_json(spec, "timing_summary.json", {
                "training": {
                    "measurement_scope": "sum_of_epoch_train_and_validation_passes",
                    "epochs_executed": 2, "best_epoch": 2,
                    "best_validation_loss": 0.3, "total_time_seconds": 22,
                    "average_time_per_epoch_seconds": 11,
                },
                "validation_inference": {
                    "measurement_scope": "end_to_end_dataloader",
                    "total_time_seconds": 8.172,
                    "average_time_per_image_seconds": 0.001,
                    "throughput_images_per_second": 1000,
                    "num_images": 8172, "warmup_batches": 1,
                },
            })
            self.write_json(spec, "experiment_metadata.json", {
                "batch_size": 32, "max_epochs": 30, "early_stopping_patience": 5,
                "min_delta": 1e-4, "seed": 42, "input_image_size": [3, 224, 224],
                "num_classes": 38, "train_samples": 38008, "validation_samples": 8172,
                "early_stopping_monitor": "validation_loss", "early_stopping_mode": "min",
                "evaluation_split": "validation", "test_split_used": False,
                "best_checkpoint_restored_before_validation": True,
                "best_model_validation_accuracy": 0.9,
                "selected_device": "cuda", "gpu_name": "Tesla T4",
                "gpu_total_memory_bytes": 16_000_000_000, "num_workers": 2,
                "pytorch_version": "synthetic-torch-a", "pytorch_cuda_version": "cuda-a",
                "preprocessing": {
                    "image_size": [224, 224], "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "compatible_with_shared_pipeline": True, "applied_resize": [224, 224],
                    "official_weights_preset_geometry_applied": False,
                    "geometry_decision_rationale": "shared pipeline",
                    "weights_preset_reference": {},
                },
            })
        self.dense = MODEL_SPECS[2]

    def write_json(self, spec, name, value):
        (self.root / spec.directory_name / name).write_text(
            json.dumps(value), encoding="utf-8"
        )

    def update_metadata(self, **updates):
        path = self.root / self.dense.directory_name / "experiment_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(updates)
        self.write_json(self.dense, path.name, metadata)

    def compare(self):
        return compare_validation_results(self.root, save=False)

    def dense_status(self, result):
        return result.summary.set_index("model_key").loc[self.dense.key, "status"]

    def test_same_conditions_are_complete_without_warnings(self):
        result = self.compare()
        self.assertEqual(result.summary.status.tolist(), ["complete"] * 4)
        self.assertEqual(result.warnings, [])

    def test_different_batch_preserves_all_original_values_and_files(self):
        original = self.compare()
        self.update_metadata(batch_size=16)
        before = {p: hashlib.sha256(p.read_bytes()).digest()
                  for p in self.root.rglob("*") if p.is_file()}
        output = Path(self.temporary.name) / "comparison"
        result = compare_validation_results(self.root, output_dir=output)
        self.assertEqual(result.summary.status.tolist(), ["complete"] * 4)
        self.assertEqual(result.summary.batch_size.tolist(), [32, 32, 16, 32])
        pd.testing.assert_frame_equal(
            result.summary.drop(columns="batch_size"),
            original.summary.drop(columns="batch_size"),
        )
        pd.testing.assert_frame_equal(result.per_class, original.per_class)
        for name in original.histories:
            pd.testing.assert_frame_equal(result.histories[name], original.histories[name])
        warning = "\n".join(result.warnings)
        self.assertIn("batch_size diferente", warning)
        self.assertIn("DenseNet201=16", warning)
        self.assertIn("tempos de treinamento", warning)
        self.assertIn(warning, result.warnings_path.read_text(encoding="utf-8"))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).digest()
                                 for p in self.root.rglob("*") if p.is_file()})

    def test_invalid_or_missing_batch_is_not_accepted(self):
        for value in (None, 0, -1, 1.5, True, "16", float("nan"), float("inf")):
            with self.subTest(batch_size=value):
                self.update_metadata(batch_size=value)
                result = self.compare()
                self.assertEqual(self.dense_status(result), "invalid")
                self.assertTrue(any("batch_size inválido" in w for w in result.warnings))
        path = self.root / self.dense.directory_name / "experiment_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        del metadata["batch_size"]
        self.write_json(self.dense, path.name, metadata)
        self.assertEqual(self.dense_status(self.compare()), "invalid")

    def test_positive_batch_sizes_are_not_limited_to_16_and_32(self):
        for value in (1, 8, 64):
            with self.subTest(batch_size=value):
                self.update_metadata(batch_size=value)
                self.assertEqual(self.dense_status(self.compare()), "complete")

    def test_each_environment_version_difference_warns_without_invalidating(self):
        for field in ("pytorch_version", "pytorch_cuda_version"):
            with self.subTest(field=field):
                self.update_metadata(**{field: "different"})
                result = self.compare()
                self.assertEqual(self.dense_status(result), "complete")
                self.assertTrue(any("Ambiente PyTorch/CUDA diferente" in w
                                    and "DenseNet201" in w for w in result.warnings))
                self.update_metadata(pytorch_version="synthetic-torch-a",
                                     pytorch_cuda_version="cuda-a")

    def test_protocol_inconsistencies_remain_invalid_with_batch_16(self):
        for field, value in (("seed", 43), ("validation_samples", 8000),
                             ("evaluation_split", "test"), ("test_split_used", True),
                             ("best_checkpoint_restored_before_validation", False),
                             ("input_image_size", [3, 256, 256])):
            with self.subTest(field=field):
                path = self.root / self.dense.directory_name / "experiment_metadata.json"
                original = path.read_bytes()
                self.update_metadata(batch_size=16, **{field: value})
                self.assertEqual(self.dense_status(self.compare()), "invalid")
                path.write_bytes(original)

    def test_history_and_metrics_inconsistencies_remain_invalid(self):
        self.update_metadata(batch_size=16)
        for name, column, value in (
            (self.dense.history_name, "epoch_time_seconds", 99),
            ("validation_metrics.csv", "f1_macro", 0.1),
        ):
            with self.subTest(artifact=name):
                path = self.root / self.dense.directory_name / name
                original = path.read_bytes()
                frame = pd.read_csv(path)
                frame.loc[0, column] = value
                frame.to_csv(path, index=False)
                self.assertEqual(self.dense_status(self.compare()), "invalid")
                path.write_bytes(original)

    def test_notebook_comparison_cells_with_synthetic_results(self):
        self.update_metadata(batch_size=16, pytorch_version="different")
        path = Path(__file__).resolve().parents[1] / "notebooks/06_compare_validation_results.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {
            "RESULTS_DIR": self.root,
            "COMPARISON_DIR": Path(self.temporary.name) / "notebook_comparison",
            "display": lambda value: None,
            "Image": lambda **kwargs: None,
        }
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = compile("".join(cell["source"]), str(path) + ":" + cell["id"], "exec")
            # Skip only environment discovery/Drive mounting/package installation.
            if cell["id"] != "paths":
                with redirect_stdout(io.StringIO()):
                    exec(code, namespace)
        self.assertEqual(namespace["comparison"].summary.status.tolist(), ["complete"] * 4)
        self.assertEqual(len(namespace["comparison_plot_paths"]), 6)
        self.assertEqual(len(namespace["curve_plot_paths"]), 2)
        self.assertTrue(all(p.is_file() for p in namespace["generated_paths"]))


if __name__ == "__main__":
    unittest.main()
