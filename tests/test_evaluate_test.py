"""Synthetic fixtures only: no repository results, real images or training."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_compare_models as fixtures
import evaluate_test as evaluation
from compare_models import MODEL_SPECS


class InferenceModel(nn.Sequential):
    def forward(self, inputs):
        assert not self.training, "eval() required"
        assert torch.is_inference_mode_enabled(), "inference_mode() required"
        return super().forward(inputs)


def tiny_model(*args, **kwargs):
    return InferenceModel(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 38))


def synthetic_metadata(counts):
    rows = []
    for split, count in counts.items():
        for i in range(count):
            name = f"class_{i % 38:02d}"
            filename = f"{split}_{i}.png"
            rows.append(dict(zip_path=f"raw/color/{name}/{filename}", classe=name,
                             leaf_id=f"{split}_{i}", split=split))
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    write_json = fixtures.ComparisonTests.write_json

    def setUp(self):
        fixtures.ComparisonTests.setUp(self)
        # Entire split CSV is synthetic, including its 54,305 rows.
        self.metadata = synthetic_metadata(evaluation.COUNTS)
        self.split_path = self.root / "plantvillage_metadata_split.csv"
        self.metadata.to_csv(self.split_path, index=False)
        self.mapping = {f"class_{i:02d}": i for i in range(38)}
        self.factory = patch.object(evaluation, "create_model", side_effect=tiny_model)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        for spec in MODEL_SPECS:
            original = self.root / spec.key
            path = original / "experiment_metadata.json"
            info = evaluation.read_json(path)
            info["training"] = {"best_epoch": 2, "best_validation_loss": .3}
            evaluation.write_json(path, info)
            classes = pd.read_csv(original / "validation_metrics_by_class.csv")
            classes["class_name"] = list(self.mapping)
            classes.to_csv(original / "validation_metrics_by_class.csv", index=False)
            for seed in evaluation.SEEDS:
                directory = original if seed == 42 else self.root / "repetitions" / f"seed_{seed}" / spec.key
                if seed != 42:
                    shutil.copytree(original, directory)
                info = evaluation.read_json(directory / "experiment_metadata.json")
                info["seed"] = seed
                evaluation.write_json(directory / "experiment_metadata.json", info)
                torch.manual_seed(seed + len(spec.key))
                checkpoint = {"epoch": 2, "validation_loss": .3, "validation_accuracy": .9,
                              "model_state_dict": tiny_model().state_dict(),
                              "class_to_idx": self.mapping,
                              "idx_to_class": {i: n for n, i in self.mapping.items()}}
                torch.save(checkpoint, directory / spec.checkpoint_name)
                if seed != 42:
                    evaluation.write_json(directory / "repetition_manifest.json", {
                        "model": spec.key, "seed": seed, "status": "complete",
                        "metadata_csv_sha256": evaluation.sha256(self.split_path),
                        "reference_metadata_sha256": evaluation.sha256(original / "experiment_metadata.json")})

    def test_preflight_twelve_checkpoints_without_images_or_writes(self):
        before = {p: evaluation.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(Image, "open", side_effect=AssertionError("No real or synthetic images during preflight")):
            plan = evaluation.prepare_test_evaluation(self.root)
        self.assertEqual(len(plan.runs), 12)
        self.assertEqual(plan.class_to_idx, self.mapping)
        self.assertEqual({p: evaluation.sha256(p) for p in self.root.rglob("*") if p.is_file()}, before)

    def test_metadata_seed_manifest_and_split_tampering_blocked(self):
        for path, field, value in (
            (self.root / "repetitions/seed_123/resnet50/experiment_metadata.json", "seed", 456),
            (self.root / "repetitions/seed_123/resnet50/repetition_manifest.json", "metadata_csv_sha256", "wrong"),
            (self.root / "repetitions/seed_123/resnet50/repetition_manifest.json", "status", "failed"),
        ):
            original = path.read_bytes()
            data = evaluation.read_json(path)
            data[field] = value
            evaluation.write_json(path, data)
            with self.assertRaises(ValueError):
                evaluation.prepare_test_evaluation(self.root)
            path.write_bytes(original)
        self.split_path.write_text(self.split_path.read_text() + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluation.prepare_test_evaluation(self.root)

    def test_missing_corrupt_and_duplicated_checkpoint_blocked(self):
        path = self.root / "baseline_cnn/baseline_cnn_best.pth"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError):
            evaluation.prepare_test_evaluation(self.root)
        path.write_bytes(b"broken checkpoint")
        with self.assertRaises(Exception):
            evaluation.prepare_test_evaluation(self.root)
        path.write_bytes(original)
        other = self.root / "repetitions/seed_123/baseline_cnn/baseline_cnn_best.pth"
        other.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "duplicado"):
            evaluation.prepare_test_evaluation(self.root)

    def test_checkpoint_semantics_strict_loading_and_nonfinite_weights(self):
        path = self.root / "baseline_cnn/baseline_cnn_best.pth"
        checkpoint = torch.load(path, weights_only=True)
        info = evaluation.read_json(path.parent / "experiment_metadata.json")
        mutations = [lambda c: c.update(epoch=1), lambda c: c.update(validation_loss=.9),
                     lambda c: c.update(seed=456), lambda c: c["class_to_idx"].update(class_00=1),
                     lambda c: c["idx_to_class"].update({0: "wrong"}),
                     lambda c: c["model_state_dict"].pop("2.weight"),
                     lambda c: c["model_state_dict"]["2.weight"].fill_(float("nan"))]
        for mutation in mutations:
            changed = copy.deepcopy(checkpoint)
            mutation(changed)
            torch.save(changed, path)
            with self.assertRaises((ValueError, RuntimeError)):
                evaluation.restore_checked(path, "baseline_cnn", info, self.mapping)

    def test_full_synthetic_inference_outputs_and_source_preservation(self):
        plan = evaluation.prepare_test_evaluation(self.root)
        small_counts = dict(train=38, validation=38, test=38)
        small = synthetic_metadata(small_counts)
        images = self.root.parent / "synthetic_images"
        # Only test files exist. Any train/validation access fails immediately.
        for row in small.loc[small.split.eq("test")].itertuples():
            path = images.joinpath(*evaluation._raw_color_relative_parts(row.zip_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 15), color=(20, 50, 90)).save(path)
        small.to_csv(self.split_path, index=False)
        plan.sources[str(self.split_path)] = evaluation.sha256(self.split_path)
        before = {p: evaluation.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(evaluation, "COUNTS", small_counts), \
             patch.object(evaluation, "prepare_test_evaluation", return_value=plan), \
             patch.object(evaluation, "save_confusion_matrix") as plot:
            output = evaluation.run_test_evaluation(self.root, images, execute=True, plan=plan,
                                                    num_workers=0, device="cpu", run_id="synthetic")
            self.assertEqual(plot.call_count, 12)
            with self.assertRaises(FileExistsError):
                evaluation.run_test_evaluation(self.root, images, execute=True, run_id="synthetic")
            plan.sources[str(self.split_path)] = "changed"
            with self.assertRaisesRegex(ValueError, "alterado"):
                evaluation.run_test_evaluation(self.root, images, execute=True, run_id="changed")
            plan.sources[str(self.split_path)] = evaluation.sha256(self.split_path)
            image_path = next(images.rglob('*.png'))
            image_path.write_bytes(b'corrupted synthetic image')
            with self.assertRaises(Exception):
                evaluation.run_test_evaluation(self.root, images, execute=True, run_id="corrupt")
            failed = evaluation.read_json(self.root / 'test_evaluation/corrupt/evaluation_manifest.json')
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['runs'], [])
        for path, checksum in before.items():
            self.assertEqual(evaluation.sha256(path), checksum)
        manifest = evaluation.read_json(output / "evaluation_manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["runs"]), 12)
        individual = pd.read_csv(output / "test_individual.csv")
        self.assertEqual(len(individual), 12)
        self.assertTrue(individual.num_samples.eq(38).all())
        self.assertEqual(pd.read_csv(output / "test_aggregate.csv").n_runs.tolist(), [3] * 4)
        self.assertEqual(len(pd.read_csv(output / "test_by_class_individual.csv")), 12 * 38)
        self.assertEqual(len(pd.read_csv(output / "test_by_class_aggregate.csv")), 4 * 38)
        for spec in MODEL_SPECS:
            for seed in evaluation.SEEDS:
                directory = output / spec.key / f"seed_{seed}"
                predictions = pd.read_csv(directory / "predictions.csv")
                self.assertEqual(predictions.zip_path.tolist(), small.loc[small.split.eq("test"), "zip_path"].tolist())
                matrix = pd.read_csv(directory / "confusion_matrix.csv", index_col=0).to_numpy()
                self.assertEqual(matrix.shape, (38, 38))
                self.assertEqual(matrix.sum(), 38)
                self.assertEqual(evaluation.read_json(directory / "inference.json")["warmup_batches"], 0)

    def test_guard_does_not_prepare_or_open_images(self):
        with patch.object(evaluation, "prepare_test_evaluation") as prepare, \
             patch.object(Image, "open", side_effect=AssertionError("image access forbidden")):
            with self.assertRaisesRegex(RuntimeError, "desativada"):
                evaluation.run_test_evaluation(self.root, "absent")
            prepare.assert_not_called()


class PureEvaluationTests(unittest.TestCase):
    def test_split_integrity(self):
        counts = dict(train=38, validation=38, test=38)
        frame = synthetic_metadata(counts)
        self.assertEqual(len(evaluation.validate_splits(frame, counts)), 38)
        for mutate in (
            lambda d: d.loc.__setitem__((0, "leaf_id"), d.iloc[38].leaf_id),
            lambda d: d.loc.__setitem__((0, "zip_path"), d.iloc[38].zip_path),
            lambda d: d.loc.__setitem__((0, "zip_path"), "raw/color/../bad.png"),
            lambda d: d.loc.__setitem__((0, "classe"), None),
            lambda d: d.loc.__setitem__((0, "split"), "test"),
            lambda d: d.loc.__setitem__((0, "zip_path"), "raw/color/wrong/bad.png"),
        ):
            changed = frame.copy()
            mutate(changed)
            with self.assertRaises(ValueError):
                evaluation.validate_splits(changed, counts)
        with self.assertRaises(ValueError):
            evaluation.validate_splits(frame)  # production count cannot silently shrink

    def test_deterministic_preprocessing_matches_validation(self):
        image = Image.fromarray(np.random.default_rng(3).integers(0, 255, (50, 80, 3), dtype=np.uint8))
        for spec in MODEL_SPECS:
            transform = evaluation.evaluation_transform(spec.key, {})
            result = transform(image)
            self.assertEqual(tuple(result.shape), (3, 224, 224))
            self.assertTrue(torch.equal(result, transform(image)))
            validation = (evaluation.build_resnet50_transforms("validation") if spec.key == "resnet50"
                          else evaluation.build_image_transforms("validation"))
            self.assertTrue(torch.equal(result, validation(image)))
            self.assertNotIn("Random", repr(transform))
        with self.assertRaises(ValueError):
            evaluation.evaluation_transform("baseline_cnn", {"preprocessing": {"image_size": [128, 128]}})

    def test_real_architectures_construct_without_pretrained_download(self):
        for spec in MODEL_SPECS:
            with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download forbidden")):
                model = evaluation.create_model(spec.key, 38)
                # Strict round trip checks the actual architecture, without an expensive forward.
                model.load_state_dict(model.state_dict(), strict=True)
                heads = [m for m in model.modules() if isinstance(m, nn.Linear)]
                self.assertEqual(heads[-1].out_features, 38)

    def test_known_metrics_and_sample_std(self):
        metrics, classes = evaluation.compute_classification_metrics([0, 0, 1, 2], [0, 1, 1, 1], ["a", "b", "c"])
        self.assertAlmostEqual(metrics["accuracy"], .5)
        self.assertAlmostEqual(metrics["precision_macro"], 4 / 9)
        self.assertAlmostEqual(metrics["recall_macro"], .5)
        self.assertAlmostEqual(metrics["f1_macro"], 7 / 18)
        self.assertAlmostEqual(metrics["precision_weighted"], 7 / 12)
        self.assertAlmostEqual(metrics["recall_weighted"], .5)
        self.assertAlmostEqual(metrics["f1_weighted"], 11 / 24)
        self.assertEqual([c["support"] for c in classes], [2, 1, 1])
        individual, per_class = [], []
        for spec in MODEL_SPECS:
            for seed, value in zip(evaluation.SEEDS, [.7, .8, .9]):
                identity = dict(model_key=spec.key, model_name=spec.display_name, seed=seed)
                individual.append({**identity, **dict.fromkeys(evaluation.METRICS, value)})
                for i in range(2):
                    per_class.append({**identity, "class_index": i, "class_name": str(i),
                                      "precision": value, "recall": value, "f1_score": value, "support": 2})
        frame, class_frame = pd.DataFrame(individual), pd.DataFrame(per_class)
        aggregate, class_aggregate = evaluation.aggregate_results(frame, class_frame)
        np.testing.assert_allclose(aggregate.accuracy_mean, .8)
        np.testing.assert_allclose(aggregate.accuracy_std, .1)
        np.testing.assert_allclose(class_aggregate.f1_score_std, .1)
        with self.assertRaises(ValueError):
            evaluation.aggregate_results(frame.iloc[:-1], class_frame)
        with self.assertRaises(ValueError):
            evaluation.aggregate_results(pd.concat([frame.iloc[:-1], frame.iloc[:1]]), class_frame)

    def test_notebook_schema_compilation_and_disabled_execution(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/08_avaliacao_teste.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                if cell.id == "configuration":
                    exec(cell.source, namespace)
        self.assertFalse(namespace["EXECUTE_TEST_EVALUATION"])
        self.assertFalse(namespace["EXTRACT_TEST_IMAGES"])
        with patch.object(evaluation, "run_test_evaluation", side_effect=AssertionError("evaluation forbidden")):
            for cell in notebook.cells:
                if cell.id in ("extract", "evaluate", "summary"):
                    exec(cell.source, namespace)
