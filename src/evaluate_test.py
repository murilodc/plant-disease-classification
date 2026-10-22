"""Final, explicitly enabled inference on the original held-out split. No training."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from baseline_cnn import create_baseline_cnn
from compare_models import MODEL_SPECS
from compare_repetitions import compare_repetitions, SEEDS
from densenet201_model import create_densenet201, validate_densenet201_preprocessing
from efficientnet_b0_model import create_efficientnet_b0, validate_efficientnet_b0_preprocessing
from experiment_utils import (compute_classification_metrics, measure_inference,
                              save_classification_metrics, save_confusion_matrix)
from plantvillage_pytorch import (PlantVillageDataset, build_class_to_idx,
                                 build_image_transforms, _raw_color_relative_parts)
from plantvillage_split import validate_leaf_id_single_class
from resnet50_model import create_resnet50, build_resnet50_transforms

COUNTS = {"train": 38008, "validation": 8172, "test": 8125}
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro",
           "precision_weighted", "recall_weighted", "f1_weighted")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def validate_splits(metadata, expected_counts=None, expected_classes=38):
    """Inspect metadata only; never regenerate the split or open any image."""
    expected_counts = COUNTS if expected_counts is None else expected_counts
    required = ["zip_path", "classe", "leaf_id", "split"]
    if not set(required).issubset(metadata) or metadata[required].isna().any().any():
        raise ValueError("CSV com colunas ausentes ou valores nulos.")
    if metadata["split"].value_counts().to_dict() != expected_counts:
        raise ValueError(f"Divisão diferente da original: esperada {expected_counts}.")
    validate_leaf_id_single_class(metadata)
    if (metadata.groupby("leaf_id")["split"].nunique() != 1).any():
        raise ValueError("Vazamento de leaf_id entre splits.")
    relative = []
    for row in metadata.itertuples():
        parts = _raw_color_relative_parts(row.zip_path)
        if len(parts) != 2 or parts[0] != row.classe:
            raise ValueError("Caminho/classe incompatíveis no CSV.")
        relative.append("/".join(parts).casefold())
    if len(set(relative)) != len(relative):
        raise ValueError("Caminhos de imagem duplicados, inclusive entre splits.")
    mapping = build_class_to_idx(metadata, expected_classes=expected_classes)
    if set(metadata.loc[metadata.split.eq("test"), "classe"]) != set(mapping):
        raise ValueError("O teste deve representar todas as classes.")
    return mapping


def evaluation_transform(model_key, metadata):
    """Match the existing validation pipeline, including its 224x224 geometry."""
    preprocessing = metadata.get("preprocessing")
    if preprocessing is not None:
        expected = {"image_size": [224, 224], "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]}
        if any(preprocessing.get(k) != v for k, v in expected.items()):
            raise ValueError("Pré-processamento incompatível com o treinamento.")
    if model_key == "resnet50":
        resolved = metadata.get("torchvision_weights_resolved")
        weights = ResNet50_Weights.IMAGENET1K_V2 if resolved is None else ResNet50_Weights[resolved.split(".")[-1]]
        return build_resnet50_transforms("test", weights=weights)
    if model_key == "densenet201":
        validate_densenet201_preprocessing()
    elif model_key == "efficientnet_b0":
        validate_efficientnet_b0_preprocessing()
    elif model_key != "baseline_cnn":
        raise ValueError(f"Modelo desconhecido: {model_key}")
    return build_image_transforms("test")


def create_model(model_key, num_classes):
    # weights=None prevents downloads and replacement of any trained weights.
    factories = {"baseline_cnn": create_baseline_cnn, "resnet50": create_resnet50,
                 "densenet201": create_densenet201, "efficientnet_b0": create_efficientnet_b0}
    if model_key == "baseline_cnn":
        return factories[model_key](num_classes=num_classes)
    return factories[model_key](num_classes=num_classes, weights=None)


def restore_checked(path, model_key, metadata, mapping):
    """Validate checkpoint semantics and strict architecture compatibility on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    inverse = {i: name for name, i in mapping.items()}
    if checkpoint.get("class_to_idx") != mapping or checkpoint.get("idx_to_class") != inverse:
        raise ValueError(f"Índices/classes incompatíveis: {path}")
    # Older checkpoints contain neither seed nor model: use associated metadata
    # and repetition manifests, and record this provenance limitation explicitly.
    if "seed" in checkpoint and checkpoint["seed"] != metadata["seed"]:
        raise ValueError("Semente interna do checkpoint incompatível.")
    training = metadata["training"]
    if checkpoint.get("epoch") != training["best_epoch"]:
        raise ValueError("Checkpoint não corresponde à época escolhida na validação.")
    for actual, expected in ((checkpoint.get("validation_loss"), training["best_validation_loss"]),
                             (checkpoint.get("validation_accuracy"), metadata["best_model_validation_accuracy"])):
        if actual is None or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
            raise ValueError("Métricas do checkpoint divergem dos metadados de validação.")
    model = create_model(model_key, len(mapping))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if any(not torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise ValueError("Checkpoint contém pesos não finitos.")
    model.requires_grad_(False)
    model.eval()
    return model


@dataclass(frozen=True)
class TestPlan:
    results_root: Path
    sources: dict[str, str]
    class_to_idx: dict[str, int]
    runs: tuple[dict, ...]
    warnings: tuple[str, ...]


def prepare_test_evaluation(results_root):
    """Read-only preflight of all twelve runs. Does not access image files."""
    root = Path(results_root).resolve()
    split_path = root / "plantvillage_metadata_split.csv"
    sources = {str(split_path): sha256(split_path)}
    metadata = pd.read_csv(split_path)
    mapping = validate_splits(metadata)
    comparison = compare_repetitions(root, save=False)
    if (len(comparison.individual) != 12 or
            not comparison.individual.status.eq("complete").all() or
            comparison.aggregate.n_runs.tolist() != [3] * 4):
        raise ValueError("Exigidas 12 execuções completas e válidas. " + " | ".join(comparison.warnings))
    runs = []
    checkpoint_hashes = set()
    for spec in MODEL_SPECS:
        for seed in SEEDS:
            directory = root / spec.directory_name if seed == 42 else root / "repetitions" / f"seed_{seed}" / spec.directory_name
            for name in spec.artifact_names() + (() if seed == 42 else ("repetition_manifest.json",)):
                path = directory / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Artefato ausente/vazio: {path}")
                sources[str(path)] = sha256(path)
            info = read_json(directory / "experiment_metadata.json")
            if "model" in info and info["model"] != spec.display_name:
                raise ValueError("Modelo dos metadados incompatível com o diretório.")
            if type(info.get("seed")) is not int or info["seed"] != seed:
                raise ValueError("Semente dos metadados incompatível.")
            if info.get("evaluation_split", "validation") != "validation" or info.get("test_split_used", False) is not False:
                raise ValueError("Execução anterior já utilizou teste.")
            for field, expected in (("num_classes", 38), ("train_samples", COUNTS["train"]),
                                    ("validation_samples", COUNTS["validation"]), ("input_image_size", [3, 224, 224])):
                if info.get(field) != expected:
                    raise ValueError(f"Metadados incompatíveis: {field}")
            if seed != 42:
                manifest = read_json(directory / "repetition_manifest.json")
                if (manifest.get("status") != "complete" or manifest.get("model") != spec.key or
                        type(manifest.get("seed")) is not int or manifest["seed"] != seed or
                        manifest.get("metadata_csv_sha256") != sources[str(split_path)] or
                        manifest.get("reference_metadata_sha256") != sources[str(root / spec.directory_name / "experiment_metadata.json")]):
                    raise ValueError("Manifesto/semente/hash dos splits incompatível.")
            class_frame = pd.read_csv(directory / "validation_metrics_by_class.csv").sort_values("class_index")
            if (class_frame.class_index.tolist() != list(range(38)) or
                    class_frame.class_name.tolist() != list(mapping)):
                raise ValueError("Classes de validação incompatíveis com o CSV original.")
            evaluation_transform(spec.key, info)
            checkpoint = directory / spec.checkpoint_name
            digest = sources[str(checkpoint)]
            if digest in checkpoint_hashes:
                raise ValueError("Checkpoint duplicado em execuções diferentes.")
            checkpoint_hashes.add(digest)
            model = restore_checked(checkpoint, spec.key, info, mapping)
            del model
            runs.append({"model_key": spec.key, "model_name": spec.display_name,
                         "seed": seed, "checkpoint": str(checkpoint), "metadata": info})
    assert_sources(sources)
    warnings = tuple(comparison.warnings) + (
        "Checkpoints legados não registram seed/modelo/hash do CSV internamente; a identidade depende dos metadados e manifestos associados.",
        "O vínculo histórico da seed 42 ao CSV depende da preservação do original; os hashes das repetições são conferidos.",)
    return TestPlan(root, sources, mapping, tuple(runs), warnings)


def assert_sources(sources):
    for path, expected in sources.items():
        if sha256(path) != expected:
            raise ValueError(f"Arquivo alterado desde a preparação: {path}")


def aggregate_results(individual, per_class):
    expected = {(s.key, seed) for s in MODEL_SPECS for seed in SEEDS}
    actual = list(zip(individual.model_key, individual.seed))
    if len(actual) != 12 or set(actual) != expected:
        raise ValueError("Consolidação exige os 12 pares modelo/semente, sem duplicação.")
    if not np.isfinite(individual[list(METRICS)].to_numpy()).all():
        raise ValueError("Métricas não finitas na consolidação.")
    aggregate = individual.groupby(["model_key", "model_name"], sort=False)[list(METRICS)].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index().assign(n_runs=3, seeds="42;123;456", std_ddof=1)
    grouped = per_class.groupby(["model_key", "model_name", "class_index", "class_name"], sort=False)
    if not grouped.seed.apply(lambda s: len(s) == 3 and set(s) == set(SEEDS)).all():
        raise ValueError("Sementes incompletas/duplicadas por classe.")
    if not grouped.support.nunique().eq(1).all():
        raise ValueError("Suporte por classe diverge entre sementes.")
    classes = grouped[["precision", "recall", "f1_score"]].agg(["mean", "std"])
    classes.columns = ["_".join(c) for c in classes.columns]
    classes = classes.join(grouped.support.first()).reset_index().assign(n_runs=3, std_ddof=1)
    return aggregate, classes


def run_test_evaluation(results_root, image_root, *, execute=False, plan=None,
                        batch_size=32, num_workers=2, device=None, run_id=None):
    """Only explicit execute=True may open images. Outputs always use a fresh folder."""
    if execute is not True:
        raise RuntimeError("Avaliação desativada. Defina execute=True explicitamente.")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size/num_workers inválidos.")
    current = prepare_test_evaluation(results_root)
    if plan is not None and (plan.sources != current.sources or plan.class_to_idx != current.class_to_idx or plan.runs != current.runs):
        raise ValueError("Plano alterado; prepare novamente antes de executar.")
    plan = current
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12])
    if not run_id or run_id in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id deve ser um identificador simples.")
    output = plan.results_root / "test_evaluation" / run_id
    if output.parent.is_symlink() or output.resolve().parent != (plan.results_root / "test_evaluation").absolute():
        raise ValueError("Destino de avaliação redirecionado.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "evaluation_split": "test", "test_samples": COUNTS["test"],
                "seeds": list(SEEDS), "checkpoint_selection": "saved_best_validation_loss_only",
                "test_based_selection": False, "std_ddof": 1, "sources_sha256": plan.sources,
                "class_to_idx": plan.class_to_idx, "warnings": list(plan.warnings),
                "batch_size": batch_size, "num_workers": num_workers, "device": str(device),
                "python": platform.python_version(), "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "started_at": datetime.now(timezone.utc).isoformat(), "runs": []}
    write_json(output / "evaluation_manifest.json", manifest)
    try:
        metadata = pd.read_csv(plan.results_root / "plantvillage_metadata_split.csv")
        validate_splits(metadata)
        names = list(plan.class_to_idx)
        reference = PlantVillageDataset(metadata, image_root, "test", plan.class_to_idx)
        # Full decode + hashes before inference: no silent skipping of corrupt files.
        images = []
        for path, row in zip(reference.image_paths, reference.metadata.itertuples()):
            if not path.resolve().is_relative_to(Path(image_root).resolve()):
                raise ValueError("Imagem fora de image_root.")
            with Image.open(path) as image:
                image.convert("RGB").load()
            images.append({"zip_path": row.zip_path, "sha256": sha256(path)})
        pd.DataFrame(images).to_csv(output / "test_images_sha256.csv", index=False)
        reference.metadata.to_csv(output / "test_split.csv", index=False)
        write_json(output / "class_to_idx.json", plan.class_to_idx)
        individuals, classes = [], []
        for run in plan.runs:
            key, seed = run["model_key"], run["seed"]
            print(f"Teste: {key}, seed {seed}", flush=True)
            checked_paths = (run["checkpoint"], str(Path(run["checkpoint"]).parent / "experiment_metadata.json"),
                             str(plan.results_root / "plantvillage_metadata_split.csv"))
            assert_sources({p: plan.sources[p] for p in checked_paths})
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            model = restore_checked(run["checkpoint"], key, run["metadata"], plan.class_to_idx).to(device)
            transform = evaluation_transform(key, run["metadata"])
            dataset = PlantVillageDataset(metadata, image_root, "test", plan.class_to_idx, transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers, pin_memory=device.type == "cuda")
            result = measure_inference(model, loader, device, warmup_batches=0)
            if result.num_images != COUNTS["test"] or not np.array_equal(result.targets, reference.labels):
                raise ValueError("Quantidade/ordem/rótulos do teste divergentes.")
            metrics, per_class = compute_classification_metrics(result.targets, result.predictions, names)
            directory = output / key / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            save_classification_metrics(metrics, per_class, directory / "test_metrics.csv",
                                        directory / "test_metrics_by_class.csv")
            predictions = dataset.metadata[["zip_path", "leaf_id", "classe"]].copy()
            predictions["target_index"] = result.targets
            predictions["prediction_index"] = result.predictions
            predictions["prediction_class"] = [names[i] for i in result.predictions]
            predictions.to_csv(directory / "predictions.csv", index=False)
            matrix = confusion_matrix(result.targets, result.predictions, labels=np.arange(len(names)))
            pd.DataFrame(matrix, index=names, columns=names).to_csv(directory / "confusion_matrix.csv", index_label="true_class")
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            pd.DataFrame(normalized, index=names, columns=names).to_csv(directory / "confusion_matrix_normalized.csv", index_label="true_class")
            save_confusion_matrix(result.targets, result.predictions, names,
                                  directory / "confusion_matrix.png", title=f"Teste — {run['model_name']} — seed {seed}")
            write_json(directory / "inference.json", {**result.timing_metrics(), "preprocessing": repr(transform),
                                                       "checkpoint_sha256": plan.sources[run["checkpoint"]]})
            identity = {"model_key": key, "model_name": run["model_name"], "seed": seed}
            individuals.append({**identity, **metrics})
            classes.extend({**identity, **row} for row in per_class)
            manifest["runs"].append({**identity, "status": "complete"})
            write_json(output / "evaluation_manifest.json", manifest)
            del model, loader, dataset
        individual, per_class = pd.DataFrame(individuals), pd.DataFrame(classes)
        aggregate, class_aggregate = aggregate_results(individual, per_class)
        for name, frame in (("test_individual", individual), ("test_aggregate", aggregate),
                            ("test_by_class_individual", per_class), ("test_by_class_aggregate", class_aggregate)):
            frame.to_csv(output / f"{name}.csv", index=False)
        assert_sources(plan.sources)
        for path, row in zip(reference.image_paths, images):
            if sha256(path) != row["sha256"]:
                raise ValueError("Imagem modificada durante a avaliação.")
        manifest["artifacts_sha256"] = {str(p.relative_to(output)): sha256(p)
                                         for p in output.rglob("*") if p.is_file() and p.name != "evaluation_manifest.json"}
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "evaluation_manifest.json", manifest)
    return output
