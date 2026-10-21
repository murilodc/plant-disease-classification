"""Aggregate saved validation repetitions without loading images or checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from compare_models import (MODEL_SPECS, compare_validation_results,
                            _validate_class_consistency, _validate_hardware_consistency)
from run_repetition import split_sha256

SEEDS = (42, 123, 456)
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro",
           "precision_weighted", "recall_weighted", "f1_weighted",
           "best_validation_loss", "epochs_executed", "best_epoch",
           "training_time_seconds", "training_time_minutes", "average_time_per_epoch_seconds",
           "inference_time_seconds", "inference_seconds_per_image",
           "inference_milliseconds_per_image", "throughput_images_per_second")


@dataclass
class RepetitionComparison:
    individual: pd.DataFrame
    aggregate: pd.DataFrame
    per_class: pd.DataFrame
    per_class_aggregate: pd.DataFrame
    histories: pd.DataFrame
    warnings: list[str]


def compare_repetitions(results_root, output_dir=None, *, save=True):
    root = Path(results_root)
    frames, classes, histories, warnings, metadata = [], [], [], [], {}
    split_path = root / "plantvillage_metadata_split.csv"
    expected_hash = split_sha256(split_path) if split_path.is_file() else None
    for seed in SEEDS:
        seed_root = root if seed == 42 else root / "repetitions" / f"seed_{seed}"
        result = compare_validation_results(seed_root, save=False, expected_seed=seed)
        warnings.extend(f"seed {seed}: {w}" for w in result.warnings)
        individual = result.summary.assign(seed=seed)
        for spec in MODEL_SPECS:
            directory = seed_root / spec.directory_name
            mask = individual.model_key.eq(spec.key)
            if seed != 42 and individual.loc[mask, "status"].iloc[0] == "complete":
                try:
                    manifest = json.loads((directory / "repetition_manifest.json").read_text(encoding="utf-8"))
                    reference = root / spec.directory_name / "experiment_metadata.json"
                    if (manifest.get("status") != "complete" or manifest.get("seed") != seed
                            or manifest.get("model") != spec.key or expected_hash is None
                            or manifest.get("metadata_csv_sha256") != expected_hash
                            or manifest.get("reference_metadata_sha256") != split_sha256(reference)):
                        raise ValueError("proveniência/splits divergentes ou execução não concluída")
                except (OSError, ValueError) as exc:
                    individual.loc[mask, "status"] = "invalid"
                    warnings.append(f"{spec.key} seed {seed}: {exc}")
            if individual.loc[mask, "status"].iloc[0] == "complete":
                metadata[f"{spec.display_name} seed {seed}"] = json.loads(
                    (directory / "experiment_metadata.json").read_text(encoding="utf-8"))
        valid_keys = individual.loc[individual.status.eq("complete"), "model_key"]
        per_class = result.per_class[result.per_class.model_key.isin(valid_keys)].copy()
        per_class["seed"] = seed
        classes.append(per_class)
        for name, history in result.histories.items():
            histories.append(history.assign(model_name=name, seed=seed))
        frames.append(individual)
    individual = pd.concat(frames, ignore_index=True)
    per_class = pd.concat(classes, ignore_index=True)
    warnings.extend(_validate_hardware_consistency(metadata))
    if not per_class.empty:
        consistency = per_class.copy()
        consistency["model_name"] = consistency.model_name + " seed " + consistency.seed.astype(str)
        warnings.extend(_validate_class_consistency(consistency))
    rows = []
    for spec in MODEL_SPECS:
        complete = individual.loc[individual.model_key.eq(spec.key) & individual.status.eq("complete")]
        row = {"model_key": spec.key, "model_name": spec.display_name,
               "n_runs": len(complete), "expected_runs": len(SEEDS),
               "seeds": ";".join(str(s) for s in complete.seed),
               "status": "complete" if len(complete) == len(SEEDS) else "partial"}
        for metric in METRICS:
            values = pd.to_numeric(complete[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    if per_class.empty:
        class_aggregate = pd.DataFrame()
    else:
        class_aggregate = per_class.groupby(["model_key", "model_name", "class_index", "class_name"])[
            ["precision", "recall", "f1_score"]].agg(["count", "mean", "std"])
        class_aggregate.columns = ["_".join(c) for c in class_aggregate.columns]
        class_aggregate = class_aggregate.reset_index()
    history_frame = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    result = RepetitionComparison(individual, aggregate, per_class, class_aggregate,
                                  history_frame, list(dict.fromkeys(warnings)))
    if save:
        output = Path(output_dir) if output_dir else root / "repetitions_comparison"
        # Comparison outputs must never be placed inside an individual run.
        for directory in [root / s.directory_name for s in MODEL_SPECS] + [
            root / "repetitions" / f"seed_{seed}" / s.directory_name
            for seed in SEEDS[1:] for s in MODEL_SPECS]:
            if output.resolve().is_relative_to(directory.resolve()):
                raise ValueError("A comparação deve ficar fora dos diretórios individuais.")
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in (("validation_individual", individual), ("validation_aggregate", aggregate),
                            ("validation_by_class_individual", per_class),
                            ("validation_by_class_aggregate", class_aggregate),
                            ("histories_individual", history_frame)):
            frame.to_csv(output / f"{name}.csv", index=False)
        (output / "comparison_warnings.txt").write_text(
            "\n".join(result.warnings) or "Nenhum aviso de comparabilidade.", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = compare_repetitions(args.results_root, args.output_dir)
    print(result.aggregate.to_string(index=False))
    for warning in result.warnings:
        print(warning)
