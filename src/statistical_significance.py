"""Paired Wilcoxon tests on saved test metrics; no training or inference.

The unit is a training seed, conditional on the fixed dataset split. The default
family contains all six architecture pairs for accuracy and macro F1 (12 tests).
Three seeds cannot yield a two-sided exact p-value below 0.25.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import PermutationMethod, wilcoxon

MODELS = {
    "baseline_cnn": "CNN própria",
    "resnet50": "ResNet50",
    "densenet201": "DenseNet201",
    "efficientnet_b0": "EfficientNet-B0",
}
SEEDS = (42, 123, 456)
METRICS = ("accuracy", "f1_macro")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "reports/resultados/20261023T005818Z_67a582ae/tables/test_individual.csv"


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, restored to original order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite numbers in [0, 1].")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def validate_metrics(frame):
    required = {"model_key", "seed", "num_samples", "num_classes", *METRICS}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if frame.duplicated(["model_key", "seed"]).any():
        raise ValueError("Duplicate model/seed pair.")
    if set(zip(frame.model_key, frame.seed)) != expected or len(frame) != len(expected):
        raise ValueError("Expected exactly four architectures and seeds 42, 123, 456.")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Metrics must be finite proportions in [0, 1].")
    if not frame.num_samples.eq(8125).all() or not frame.num_classes.eq(38).all():
        raise ValueError("Expected 8125 test samples and 38 classes in every run.")


def paired_test(differences):
    # SciPy recommends explicit rounding to prevent floating-point rank artifacts.
    d = np.round(np.asarray(differences, dtype=float), 12)
    if d.shape != (len(SEEDS),) or not np.isfinite(d).all():
        raise ValueError("Expected three finite paired differences.")
    nonzero = d[d != 0]
    if not len(nonzero):
        return 0.0, 1.0, "all_zero", 0
    ties = len(np.unique(abs(nonzero))) != len(nonzero)
    if len(nonzero) != len(d) or ties:
        # Exhaustive sign permutations also cover zeros and equal absolute ranks.
        method, method_name = PermutationMethod(n_resamples=np.inf), "exhaustive_permutation"
    else:
        method, method_name = "exact", "exact"
    result = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method=method)
    return float(result.statistic), float(result.pvalue), method_name, len(nonzero)


def compare_metrics(frame, alpha=0.05):
    validate_metrics(frame)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    indexed = frame.set_index(["model_key", "seed"])
    rows = []
    for metric in METRICS:
        for a, b in combinations(MODELS, 2):
            x = np.array([indexed.loc[(a, seed), metric] for seed in SEEDS], dtype=float)
            y = np.array([indexed.loc[(b, seed), metric] for seed in SEEDS], dtype=float)
            statistic, pvalue, method, n_nonzero = paired_test(x - y)
            rows.append({
                "metric": metric, "model_a": a, "model_b": b,
                "n_pairs": len(SEEDS), "n_nonzero": n_nonzero,
                "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                "mean_difference_pp_a_minus_b": float((x - y).mean() * 100),
                "median_difference_pp_a_minus_b": float(np.median(x - y) * 100),
                "statistic": statistic, "p_value": pvalue, "method": method,
            })
    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_adjust(result.p_value)
    result["reject_h0_holm"] = result.p_value_holm.le(alpha)
    return result


def run_analysis(source, output_dir, alpha=0.05):
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    raw = source.read_bytes()
    frame = pd.read_csv(source, float_precision="round_trip")
    result = compare_metrics(frame, alpha)
    # Never overwrite a previous analysis or alter its input.
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "wilcoxon_pairs.csv", index=False)
    frame.sort_values(["model_key", "seed"]).to_csv(output / "test_metrics_used.csv", index=False)
    manifest = {
        "status": "complete", "test": "paired Wilcoxon signed-rank",
        "alternative": "two-sided", "alpha": alpha,
        "family": "6 architecture pairs x 2 metrics", "correction": "Holm",
        "seeds": list(SEEDS), "metrics": list(METRICS), "n_tests": len(result),
        "source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "difference_rounding_decimals": 12,
        "inference_executed": False, "training_executed": False,
        "python": platform.python_version(), "scipy": scipy.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Only three pairs; minimum exact two-sided p-value is 0.25.",
            "Inference is conditional on the fixed dataset split, not new datasets.",
            "Signed-rank interpretation assumes independent paired differences and symmetry under H0; three pairs cannot assess symmetry reliably.",
            "Non-rejection does not establish equality or equivalence of architectures.",
            "Only the published metric export was available; original prediction files and evaluation manifest were not re-audited.",
        ],
    }
    (output / "statistical_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(args.input, args.output_dir)
    print(result[["metric", "model_a", "model_b", "p_value", "p_value_holm"]].to_string(index=False))
    print("Three paired seeds: no statistical superiority or equivalence can be established at 5%.")


if __name__ == "__main__":
    main()
