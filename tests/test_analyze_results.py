"""Analysis math and notebook guards; no model or GPU dependencies."""
import ast
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nbformat
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze_results as analysis


class AnalysisTests(unittest.TestCase):
    def test_colab_bootstrap_uses_separate_clone_and_surfaces_git_error(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks/09_analise_resultados.ipynb", as_version=4)
        source = next(c.source for c in notebook.cells if c.id == "paths")
        colab = ModuleType("google.colab")
        colab.drive = SimpleNamespace(mount=Mock())
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "experiment"
            (project / "results").mkdir(parents=True)
            selected = project / 'results/test_evaluation/20261022T233940Z_c146fc70d66e'
            selected.mkdir(parents=True)
            (selected / 'evaluation_manifest.json').write_text('{"status":"complete"}')
            older = project / 'results/test_evaluation/20261022T225848Z_a96e51c9a0fc'
            older.mkdir()
            (older / 'evaluation_manifest.json').write_text('{"status":"complete"}')
            sentinel = project / 'results/unchanged.txt'
            sentinel.write_text('original')
            prepared = source.replace('PROJECT_ROOT_OVERRIDE = None', f'PROJECT_ROOT_OVERRIDE = Path({str(project)!r})')
            code_root = Path(temporary) / "code"

            def clone(command, **kwargs):
                self.assertEqual(command[:4], ['git', 'clone', '--depth', '1'])
                self.assertNotIn(str(project), command)
                target = Path(command[-1]) / 'src/analyze_results.py'
                target.parent.mkdir(parents=True)
                target.write_text('# fixture module')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with patch.dict(sys.modules, {'google': ModuleType('google'), 'google.colab': colab}), \
                 patch.object(sys, 'path', sys.path.copy()), \
                 patch('tempfile.mkdtemp', return_value=str(code_root)), \
                 patch('subprocess.run', side_effect=clone) as git:
                namespace = {}
                exec(prepared, namespace)
                self.assertEqual(namespace['PROJECT_ROOT'], project.resolve())
                self.assertEqual(namespace['CODE_ROOT'], code_root / 'repo')
                self.assertEqual(namespace['TEST_DIR'], selected)
                # Re-running configuration must preserve the requested new run
                # even when more than one completed evaluation exists.
                self.assertNotIn('TEST_DIR = None', source)
                self.assertEqual(sentinel.read_text(), 'original')
                self.assertEqual(git.call_count, 1)
            with patch.dict(sys.modules, {'google': ModuleType('google'), 'google.colab': colab}), \
                 patch('tempfile.mkdtemp', return_value=str(code_root)), \
                 patch('subprocess.run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='network unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'network unavailable'):
                    exec(prepared, {})

    def test_notebook_finds_nested_local_project_and_drive_export(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks/09_analise_resultados.ipynb", as_version=4)
        source = next(c.source for c in notebook.cells if c.id == "paths")
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "find_project_root")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "paths", "exec"), namespace)
        find_root = namespace["find_project_root"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            (project / "src").mkdir(parents=True)
            (project / "src/analyze_results.py").write_text("# fixture")
            nested = project / "notebooks/nested"
            nested.mkdir(parents=True)
            self.assertEqual(find_root([nested, *nested.parents]), project.resolve())
            drive = root / "drive/MyDrive/plant-disease-classification"
            (drive / "src").mkdir(parents=True)
            (drive / "src/analyze_results.py").write_text("# fixture")
            (drive / "results/repetitions_comparison").mkdir(parents=True)
            (drive / "results/repetitions_comparison/validation_individual.csv").write_text("fixture")
            self.assertEqual(find_root([project, drive]), drive.resolve())
            with self.assertRaisesRegex(FileNotFoundError, "não sincroniza"):
                find_root([root / "missing"])
            self.assertIn("PROJECT_ROOT_OVERRIDE", source)
            self.assertIn("drive.mount('/content/drive')", source)

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
