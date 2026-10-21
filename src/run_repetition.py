"""Prepare one protected repetition using the existing training entry points."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from compare_models import MODEL_SPECS, compare_validation_results

SCRIPTS = dict(zip((s.key for s in MODEL_SPECS), (
    "train_baseline.py", "train_resnet50.py", "train_densenet201.py", "train_efficientnet_b0.py",
)))
SEEDS = (123, 456)


def split_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_repetition(project_root, model, seed, image_root):
    """Read-only preflight. Return a command and its provenance; never train here."""
    root = Path(project_root).resolve()
    if model not in SCRIPTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Selecione um dos quatro modelos e seed 123 ou 456.")
    results = root / "results"
    metadata_csv = results / "plantvillage_metadata_split.csv"
    reference = results / model / "experiment_metadata.json"
    metadata = json.loads(reference.read_text(encoding="utf-8"))
    comparison = compare_validation_results(results, save=False)
    row = comparison.summary.set_index("model_key").loc[model]
    if row["status"] != "complete":
        raise ValueError("Seed 42 de referência inválida/incompleta: " + " | ".join(comparison.warnings))
    if metadata.get("gpu_name") != "NVIDIA L4":
        raise ValueError("A referência deve ser a execução original NVIDIA L4 no Drive.")
    if not Path(image_root).is_dir():
        raise FileNotFoundError(image_root)
    output = results / "repetitions" / f"seed_{seed}" / model
    if output.exists():
        raise FileExistsError(f"Execução já reservada: {output}; nada será sobrescrito.")
    # All repetitions must use the same immutable CSV, even across runtime sessions.
    checksum = split_sha256(metadata_csv)
    for previous in (results / "repetitions").glob("seed_*/*/repetition_manifest.json"):
        recorded = json.loads(previous.read_text(encoding="utf-8"))
        if recorded.get("metadata_csv_sha256") != checksum:
            raise ValueError("CSV de splits diverge de uma repetição já reservada.")
    learning_rate = 0.001 if model == "baseline_cnn" else 0.0001
    if (metadata.get("learning_rate") != learning_rate or metadata.get("optimizer") != "Adam"
            or type(metadata.get("num_workers")) is not int or metadata["num_workers"] < 0):
        raise ValueError("Learning rate, otimizador ou num_workers da referência incompatíveis.")
    command = [sys.executable, str(root / "src" / SCRIPTS[model]),
               "--metadata-csv", str(metadata_csv), "--image-root", str(image_root),
               "--output-dir", str(output), "--seed", str(seed),
               "--batch-size", "32", "--epochs", "30", "--patience", "5",
               "--min-delta", "0.0001", "--num-classes", "38",
               "--num-workers", str(metadata["num_workers"]),
               "--lr", str(learning_rate)]
    manifest = {"model": model, "seed": seed, "reference_seed": 42,
                "metadata_csv_sha256": split_sha256(metadata_csv),
                "reference_metadata_sha256": split_sha256(reference),
                "evaluation_split": "validation", "test_split_used": False,
                "command": command, "output_dir": str(output)}
    return command, manifest


def run_repetition(project_root, model, seed, image_root):
    """Explicit execution only; reserve a new directory atomically before launch."""
    command, manifest = prepare_repetition(project_root, model, seed, image_root)
    import torch
    import torchvision
    reference = json.loads((Path(project_root) / "results" / model /
                            "experiment_metadata.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA L4":
        raise RuntimeError("Selecione GPU NVIDIA L4 para manter o protocolo de tempos.")
    for field, actual in (("pytorch_version", torch.__version__),
                          ("pytorch_cuda_version", torch.version.cuda),
                          ("torchvision_version", torchvision.__version__)):
        if reference.get(field) != actual:
            raise RuntimeError(f"Ambiente diferente da seed 42: {field}: {actual} != {reference.get(field)}")
    output = Path(manifest["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "repetition_manifest.json"
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        subprocess.run(command, check=True)
        if split_sha256(Path(project_root) / "results/plantvillage_metadata_split.csv") != manifest["metadata_csv_sha256"]:
            raise RuntimeError("O CSV de splits foi modificado durante a execução.")
    except BaseException:
        manifest["status"] = "failed"
        raise
    else:
        manifest["status"] = "complete"
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=SCRIPTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Inicia treinamento; sem esta opção só exibe o plano.")
    args = parser.parse_args()
    if args.execute:
        print(run_repetition(args.project_root, args.model, args.seed, args.image_root))
    else:
        _, manifest = prepare_repetition(args.project_root, args.model, args.seed, args.image_root)
        print(json.dumps(manifest, indent=2))
