"""Analysis math and notebook guards; no model or GPU dependencies."""
import ast
from pathlib import Path
import sys
import unittest

import nbformat
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze_results as analysis


class AnalysisTests(unittest.TestCase):
    def test_aggregation_is_sample_std(self):
        source = pd.DataFrame(dict(model_key=["m"] * 3, model_name=["M"] * 3, accuracy=[.7, .8, .9]))
        summary = analysis.aggregate(source, ["accuracy"]).iloc[0]
        self.assertAlmostEqual(summary.accuracy_mean, .8)
        self.assertAlmostEqual(summary.accuracy_std, .1)
        population = summary.to_frame().T
        population["accuracy_std"] = np.std([.7, .8, .9], ddof=0)
        with self.assertRaises(ValueError):
            analysis.check_aggregate(summary.to_frame().T, population,
                                     ["model_key", "model_name"], ["accuracy"], "synthetic")

    def test_metrics_from_saved_matrix_conventions(self):
        matrix = np.zeros((38, 38), dtype=int)
        matrix[0, 0], matrix[0, 1], matrix[1, 1], matrix[2, 1] = 1, 1, 1, 1
        overall, classes, support = analysis.metrics_from_matrix(matrix)
        self.assertAlmostEqual(overall["accuracy"], .5)
        self.assertAlmostEqual(overall["recall_weighted"], .5)
        self.assertAlmostEqual(overall["f1_weighted"], 11 / 24)
        self.assertAlmostEqual(overall["f1_macro"], (2 / 3 + .5) / 38)
        self.assertEqual(support.sum(), 4)
        self.assertTrue(np.isfinite(classes).all())

    def test_runs_reject_duplicate_seed(self):
        frame = pd.DataFrame([dict(model_key=model, model_name=name, seed=seed,
                                   **dict.fromkeys(analysis.METRICS, .9))
                              for model, name in analysis.MODELS.items() for seed in analysis.SEEDS])
        analysis.validate_runs(frame, "synthetic")
        frame.loc[0, "seed"] = 123
        with self.assertRaises(ValueError):
            analysis.validate_runs(frame, "synthetic")

    def test_cpu_only_module_and_notebook_compile(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "src/analyze_results.py").read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(any(name and any(t in name for t in ("torch", "train_", "evaluate_test", "experiment_utils")) for name in imported))
        notebook = nbformat.read(root / "notebooks/09_analise_resultados.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "notebook09", "exec")
                self.assertFalse(any(o.output_type == 'error' for o in cell.outputs))

    def test_latex_escape_and_percent_units(self):
        text = analysis.latex_table(pd.DataFrame({"Modelo": ["A_B"], "F1 (%)": ["99,1 ± 0,1"]}), "Teste", "tab:test")
        self.assertIn(r"A\_B", text)
        self.assertIn(r"F1 (\%)", text)
        self.assertIn(r"$\pm$", text)


if __name__ == "__main__":
    unittest.main()
