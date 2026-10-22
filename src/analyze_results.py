"""CPU-only scientific analysis of saved results. Never imports model/training code."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import textwrap
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

MODELS = {"baseline_cnn": "CNN baseline", "resnet50": "ResNet50",
          "densenet201": "DenseNet201", "efficientnet_b0": "EfficientNet-B0"}
SEEDS = (42, 123, 456)
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro",
           "precision_weighted", "recall_weighted", "f1_weighted")
LABELS = ("Acurácia", "Precisão macro", "Recall macro", "F1 macro",
          "Precisão pond.", "Recall pond.", "F1 pond.")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
CLASS_METRICS = ("precision", "recall", "f1_score")
EXPECTED_PAIRS = {(model, seed) for model in MODELS for seed in SEEDS}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                        rtol=1e-7, atol=1e-9, equal_nan=False), f"Inconsistência: {context}")


def validate_runs(frame, source):
    require(len(frame) == 12 and set(zip(frame.model_key, frame.seed)) == EXPECTED_PAIRS,
            f"{source}: exigidos exatamente os 12 pares modelo/seed.")
    require(frame.seed.map(lambda v: isinstance(v, (int, np.integer))).all(), f"{source}: seeds não inteiras.")
    require(all(row.model_name == MODELS[row.model_key] for row in frame.itertuples()),
            f"{source}: nomes dos modelos divergentes.")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
            f"{source}: métricas inválidas.")
    if "status" in frame:
        require(frame.status.eq("complete").all(), f"{source}: execução incompleta.")


def aggregate(frame, metrics, keys=("model_key", "model_name")):
    result = frame.groupby(list(keys), sort=False)[list(metrics)].agg(["mean", "std"])
    result.columns = ["_".join(c) for c in result.columns]
    return result.reset_index()


def check_aggregate(computed, saved, keys, metrics, source):
    require(not saved.duplicated(list(keys)).any(), f"{source}: linhas duplicadas.")
    a, b = computed.set_index(list(keys)).sort_index(), saved.set_index(list(keys)).sort_index()
    require(a.index.equals(b.index), f"{source}: chaves divergentes.")
    for metric in metrics:
        for stat in ("mean", "std"):
            close(a[f"{metric}_{stat}"], b[f"{metric}_{stat}"], f"{source}: {metric}_{stat}, ddof=1")


def metrics_from_matrix(matrix):
    """Audit already saved predictions; does not perform model inference."""
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros(38), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(38), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(38), where=(precision + recall) > 0)
    general = {"accuracy": float(true_positive.sum() / matrix.sum())}
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        general[f"{name}_macro"] = float(values.mean())
        general[f"{name}_weighted"] = float(np.average(values, weights=support))
    return general, np.column_stack([precision, recall, f1]), support


@dataclass
class SavedResults:
    root: Path
    test_dir: Path
    manifest: dict
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_classes: pd.DataFrame
    test_classes: pd.DataFrame
    histories: pd.DataFrame
    matrices: dict
    predictions: dict
    inference: pd.DataFrame
    sources: dict
    inventory: pd.DataFrame
    historical_sources: pd.DataFrame
    notes: list[str]
    split: pd.DataFrame


def load_saved_results(results_root, test_dir=None):
    root = Path(results_root).resolve()
    if test_dir is None:
        candidates = [p.parent for p in (root / "test_evaluation").glob("*/evaluation_manifest.json")
                      if read_json(p).get("status") == "complete"]
        require(len(candidates) == 1,
                "Informe TEST_DIR explicitamente: exigida uma única avaliação completa, "
                f"encontradas {len(candidates)}: {[str(p) for p in candidates]}")
        test_dir = candidates[0]
    test_dir = Path(test_dir).resolve()
    sources = {}

    def csv(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo necessário: {path}")
        sources[str(path)] = digest(path)
        return pd.read_csv(path)

    def document(path):
        sources[str(path)] = digest(path)
        return read_json(path)

    manifest = document(test_dir / "evaluation_manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("evaluation_split") == "test" and
            manifest.get("test_samples") == 8125 and set(manifest.get("seeds", [])) == set(SEEDS),
            "Manifesto de teste incompleto/incompatível.")
    require(manifest.get("test_based_selection") is False and
            manifest.get("checkpoint_selection") == "saved_best_validation_loss_only",
            "Protocolo de seleção do checkpoint incompatível.")
    runs = manifest["runs"]
    require(len(runs) == 12 and {(r["model_key"], r["seed"]) for r in runs} == EXPECTED_PAIRS and
            all(r.get("status") == "complete" for r in runs), "Manifesto: execuções inválidas.")
    require(bool(manifest.get("artifacts_sha256")), "Manifesto sem hashes de resultados.")
    for relative, expected in manifest["artifacts_sha256"].items():
        path = (test_dir / relative).resolve()
        require(path.is_relative_to(test_dir), "Manifesto com caminho fora da avaliação.")
        if not path.is_file():
            raise FileNotFoundError(f"Artefato de teste necessário: {path}")
        sources[str(path)] = digest(path)
        require(sources[str(path)] == expected, f"Hash de teste divergente: {path}")

    repeat = root / "repetitions_comparison"
    validation = csv(repeat / "validation_individual.csv")
    test = csv(test_dir / "test_individual.csv")
    validate_runs(validation, "validation_individual.csv")
    validate_runs(test, "test_individual.csv")
    require(test.num_samples.eq(8125).all() and test.num_classes.eq(38).all(), "Dimensão do teste inválida.")
    validation_classes = csv(repeat / "validation_by_class_individual.csv")
    test_classes = csv(test_dir / "test_by_class_individual.csv")
    mapping = document(test_dir / "class_to_idx.json")
    require(mapping == manifest["class_to_idx"] and sorted(mapping.values()) == list(range(38)), "Mapa de classes divergente.")
    names = [name for name, i in sorted(mapping.items(), key=lambda item: item[1])]
    for phase, classes, individual, count in (("validação", validation_classes, validation, 8172),
                                               ("teste", test_classes, test, 8125)):
        require(len(classes) == 12 * 38 and not classes.duplicated(["model_key", "seed", "class_index"]).any(),
                f"Classes de {phase}: linhas incompletas/duplicadas.")
        require(set(zip(classes.model_key, classes.seed)) == EXPECTED_PAIRS, f"Classes de {phase}: seeds divergentes.")
        reference_support = None
        for (model, seed), group in classes.groupby(["model_key", "seed"]):
            group = group.sort_values("class_index")
            require(group.class_index.tolist() == list(range(38)) and group.class_name.tolist() == names,
                    f"{phase}: índices/nomes de classes divergentes.")
            support = group.support.to_numpy()
            require((support > 0).all() and np.equal(support, np.floor(support)).all() and support.sum() == count,
                    f"{phase}: suportes inválidos.")
            if reference_support is None:
                reference_support = support
            close(support, reference_support, f"{phase}: suportes entre runs")
            values = group[list(CLASS_METRICS)].to_numpy(float)
            require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f"{phase}: métricas por classe inválidas.")
            overall = individual.set_index(["model_key", "seed"]).loc[(model, seed)]
            for col, metric in zip(CLASS_METRICS, ("precision", "recall", "f1")):
                close(group[col].mean(), overall[f"{metric}_macro"], f"{phase}: macro")
                close(np.average(group[col], weights=support), overall[f"{metric}_weighted"], f"{phase}: ponderado")
            close(overall.accuracy, overall.recall_weighted, f"{phase}: acurácia/recall ponderado")
        check_aggregate(aggregate(individual, METRICS),
                        csv((repeat if phase == "validação" else test_dir) / ("validation_aggregate.csv" if phase == "validação" else "test_aggregate.csv")),
                        ("model_key", "model_name"), METRICS, phase)
        check_aggregate(aggregate(classes, CLASS_METRICS, ("model_key", "class_index", "class_name")),
                        csv((repeat if phase == "validação" else test_dir) / ("validation_by_class_aggregate.csv" if phase == "validação" else "test_by_class_aggregate.csv")),
                        ("model_key", "class_index", "class_name"), CLASS_METRICS, f"{phase} por classe")

    histories = csv(repeat / "histories_individual.csv")
    require(not histories.duplicated(["model_name", "seed", "epoch"]).any(), "Épocas duplicadas no histórico.")
    require(set(zip(histories.model_name, histories.seed)) == {(MODELS[m], s) for m, s in EXPECTED_PAIRS}, "Históricos incompletos.")
    for row in validation.itertuples():
        history = histories.loc[histories.model_name.eq(row.model_name) & histories.seed.eq(row.seed)].sort_values("epoch")
        require(history.epoch.tolist() == list(range(1, int(row.epochs_executed) + 1)), "Histórico com épocas faltantes.")
        close(history.epoch_time_seconds.sum(), row.training_time_seconds, "tempo acumulado/histórico")
        close(history.epoch_time_seconds.mean(), row.average_time_per_epoch_seconds, "tempo por época/histórico")
        close(history.validation_loss.min(), row.best_validation_loss, "menor loss de validação")
        require(history.loc[history.validation_loss.idxmin(), "epoch"] == row.best_epoch, "Melhor época divergente.")
        close(row.training_time_seconds / 60, row.training_time_minutes, "conversão de minutos")
        close(row.total_parameters / 1e6, row.parameters_millions, "parâmetros em milhões")
        close(row.inference_seconds_per_image * 1000, row.inference_milliseconds_per_image, "latência em ms")
        close(row.inference_time_seconds / 8172, row.inference_seconds_per_image, "tempo de validação por imagem")

    split = csv(test_dir / "test_split.csv")
    require(len(split) == 8125 and split.split.eq("test").all() and not split.zip_path.duplicated().any(), "Seleção de teste inválida.")
    matrices, predictions, inference = {}, {}, []
    expected_targets = split.classe.map(mapping).to_numpy()
    require(not pd.isna(expected_targets).any(), "Classe desconhecida no teste.")
    for model in MODELS:
        for seed in SEEDS:
            directory = test_dir / model / f"seed_{seed}"
            matrix = csv(directory / "confusion_matrix.csv").set_index("true_class")
            require(matrix.index.tolist() == names and matrix.columns.tolist() == names, "Eixos da matriz divergentes.")
            values = matrix.to_numpy(float)
            require(np.isfinite(values).all() and (values >= 0).all() and np.equal(values, np.floor(values)).all(), "Matriz não inteira/negativa.")
            values = values.astype(np.int64)
            prediction = csv(directory / "predictions.csv")
            require(prediction.zip_path.tolist() == split.zip_path.tolist() and
                    prediction.leaf_id.tolist() == split.leaf_id.tolist() and prediction.classe.tolist() == split.classe.tolist(),
                    "Predições não correspondem à seleção/ordem original de teste.")
            close(prediction.target_index, expected_targets, "rótulos das predições")
            predicted = prediction.prediction_index.to_numpy()
            require(np.isfinite(predicted).all() and np.equal(predicted, np.floor(predicted)).all() and
                    ((predicted >= 0) & (predicted < 38)).all(), "Índices preditos inválidos.")
            require(prediction.prediction_class.tolist() == [names[int(i)] for i in predicted], "Nomes preditos divergentes.")
            rebuilt = np.zeros((38, 38), dtype=np.int64)
            np.add.at(rebuilt, (expected_targets.astype(int), predicted.astype(int)), 1)
            close(values, rebuilt, "matriz/predições salvas")
            metrics, class_values, support = metrics_from_matrix(values)
            overall = test.set_index(["model_key", "seed"]).loc[(model, seed)]
            close([metrics[k] for k in METRICS], overall[list(METRICS)], "métricas de teste/matriz")
            group = test_classes.loc[test_classes.model_key.eq(model) & test_classes.seed.eq(seed)].sort_values("class_index")
            close(class_values, group[list(CLASS_METRICS)], "métricas por classe/matriz")
            close(support, group.support, "suporte/matriz")
            single = csv(directory / "test_metrics.csv")
            close(single[list(METRICS)].iloc[0], overall[list(METRICS)], "CSV individual de teste")
            individual_classes = csv(directory / "test_metrics_by_class.csv").sort_values("class_index")
            require(individual_classes.class_name.tolist() == names, "CSV por classe individual divergente.")
            close(individual_classes[list(CLASS_METRICS)], group[list(CLASS_METRICS)], "CSV individual por classe")
            normalized = csv(directory / "confusion_matrix_normalized.csv").set_index("true_class")
            require(normalized.index.tolist() == names and normalized.columns.tolist() == names, "Eixos normalizados divergentes.")
            close(normalized, values / support[:, None], "matriz normalizada")
            timing = document(directory / "inference.json")
            require(timing.get("num_images") == 8125 and timing.get("measurement_scope") == "end_to_end_dataloader", "Escopo de inferência incompatível.")
            close(timing["total_time_seconds"] / 8125, timing["average_time_per_image_seconds"], "latência do teste")
            close(8125 / timing["total_time_seconds"], timing["throughput_images_per_second"], "throughput do teste")
            checkpoint_relative = f"{model}/{model}_best.pth" if seed == 42 else f"repetitions/seed_{seed}/{model}/{model}_best.pth"
            expected_hash = [v for p, v in manifest["sources_sha256"].items() if p.endswith("/results/" + checkpoint_relative)]
            require(len(expected_hash) == 1 and timing["checkpoint_sha256"] == expected_hash[0], "Hash do checkpoint na inferência divergente.")
            matrices[(model, seed)], predictions[(model, seed)] = values, prediction
            inference.append(dict(model_key=model, model_name=MODELS[model], seed=seed,
                                  test_inference_seconds=timing["total_time_seconds"],
                                  test_latency_ms=timing["average_time_per_image_seconds"] * 1000,
                                  test_throughput=timing["throughput_images_per_second"],
                                  warmup_batches=timing["warmup_batches"]))

    notes = list(manifest.get("warnings", []))
    warnings_path = repeat / "comparison_warnings.txt"
    if warnings_path.exists():
        sources[str(warnings_path)] = digest(warnings_path)
        notes.append("Aviso salvo na comparação de validação: " + warnings_path.read_text(encoding="utf-8").strip())
    historical = []
    for original, expected in manifest.get("sources_sha256", {}).items():
        relative = original.split("/results/", 1)[-1]
        path = root / relative
        status = "ausente"
        if path.is_file():
            sources[str(path)] = digest(path)
            if sources[str(path)] == expected:
                status = "hash_idêntico"
            elif hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected:
                status = "equivalente_após_CRLF_para_LF"
            else:
                status = "cópia_local_divergente_não_utilizada"
        historical.append(dict(path=str(path), original_path=original, status=status,
                               expected_sha256=expected, local_sha256=sources.get(str(path))))
    # Full metadata is inspected as metadata only; no image or checkpoint is opened.
    metadata_path = root / "plantvillage_metadata_split.csv"
    if metadata_path.is_file():
        metadata = csv(metadata_path)
        require(metadata.split.value_counts().to_dict() == {"train": 38008, "validation": 8172, "test": 8125}, "Contagens locais de splits incompatíveis.")
        require(metadata.groupby("leaf_id").split.nunique().eq(1).all(), "Vazamento de leaf_id nos metadados.")
        require(metadata.groupby("leaf_id").classe.nunique().eq(1).all(), "leaf_id com classes divergentes.")
        local_test = metadata.loc[metadata.split.eq("test")].reset_index(drop=True)
        require(local_test.equals(split), "Metadados locais/test_split diferem semanticamente.")
        notes.append("O teste local é semanticamente idêntico ao test_split salvo. A normalização CRLF→LF é registrada separadamente do hash bruto.")
    else:
        notes.append(f"Metadados originais ausentes: {metadata_path}; usado exclusivamente test_split.csv com hash verificado.")
    notes.append("A análise usa os consolidados de validação do notebook 07; não substitui seus valores por artefatos individuais locais divergentes.")
    inventory = []
    for p in sorted(root.rglob("*.csv")):
        inventory.append(dict(path=str(p), bytes=p.stat().st_size,
                              used_in_analysis=str(p.resolve()) in sources))
    return SavedResults(root, test_dir, manifest, validation, test, validation_classes, test_classes,
                        histories, matrices, predictions, pd.DataFrame(inference), sources,
                        pd.DataFrame(inventory), pd.DataFrame(historical), notes, split)


def scientific_tables(data):
    tables = {}
    for phase, individual, classes in (("validation", data.validation, data.validation_classes),
                                        ("test", data.test, data.test_classes)):
        tables[phase + "_individual"] = individual.copy()
        tables[phase + "_summary"] = aggregate(individual, METRICS).assign(n_runs=3, std_ddof=1)
        per_class = aggregate(classes, CLASS_METRICS, ("model_key", "model_name", "class_index", "class_name"))
        support = classes.groupby(["model_key", "class_index"]).support.first().rename("support")
        per_class = per_class.set_index(["model_key", "class_index"]).join(support).reset_index()
        tables[phase + "_classes"] = per_class
        tables[phase + "_difficult_classes"] = per_class.sort_values(["model_key", "f1_score_mean", "class_index"]).groupby("model_key", sort=False).head(5)
    paired = data.validation[["model_key", "model_name", "seed", *METRICS]].merge(
        data.test[["model_key", "seed", *METRICS]], on=["model_key", "seed"], suffixes=("_validation", "_test"), validate="one_to_one")
    for metric in METRICS:
        paired[metric + "_delta_pp"] = 100 * (paired[metric + "_test"] - paired[metric + "_validation"])
    tables["validation_test_differences"] = paired
    cost_metrics = ("training_time_minutes", "average_time_per_epoch_seconds", "epochs_executed",
                    "inference_milliseconds_per_image", "inference_time_seconds",
                    "throughput_images_per_second", "test_latency_ms", "test_inference_seconds", "test_throughput")
    cost = data.validation.merge(data.inference, on=["model_key", "model_name", "seed"], validate="one_to_one")
    for model, group in cost.groupby("model_key"):
        require(group.total_parameters.nunique() == 1, f"Parâmetros variam entre seeds: {model}")
    tables["cost_individual"] = cost
    costs = aggregate(cost, cost_metrics)
    params = cost.groupby("model_key").agg(total_parameters=("total_parameters", "first"),
                                           parameters_millions=("parameters_millions", "first"),
                                           training_total_three_seeds_minutes=("training_time_minutes", "sum"))
    tables["cost_summary"] = costs.set_index("model_key").join(params).reset_index()
    names = [name for name, index in sorted(data.manifest["class_to_idx"].items(), key=lambda x: x[1])]
    support = data.split.groupby("classe").agg(test_images=("zip_path", "size"), test_leaves=("leaf_id", "nunique")).reindex(names)
    support = support.rename_axis("class_name").reset_index()
    support["class_index"] = range(38)
    support["test_share_percent"] = 100 * support.test_images / 8125
    support["one_error_recall_pp"] = 100 / support.test_images
    tables["class_support"] = support
    confusion_rows, stability = [], []
    for model in MODELS:
        matrices = np.stack([data.matrices[(model, seed)] for seed in SEEDS])
        normalized = matrices / matrices.sum(axis=2, keepdims=True)
        for i in range(38):
            for j in range(38):
                if i != j and matrices[:, i, j].sum() > 0:
                    confusion_rows.append(dict(model_key=model, model_name=MODELS[model],
                                               true_index=i, predicted_index=j, true_class=names[i], predicted_class=names[j],
                                               errors_mean=matrices[:, i, j].mean(), errors_std=matrices[:, i, j].std(ddof=1),
                                               rate_mean=normalized[:, i, j].mean(), rate_std=normalized[:, i, j].std(ddof=1),
                                               seeds_with_error=int((matrices[:, i, j] > 0).sum()),
                                               source_support=int(matrices[0, i].sum())))
        errors = np.stack([(data.predictions[(model, seed)].target_index !=
                            data.predictions[(model, seed)].prediction_index).to_numpy() for seed in SEEDS])
        stability.append(dict(model_key=model, model_name=MODELS[model],
                               errors_seed_42=int(errors[0].sum()), errors_seed_123=int(errors[1].sum()),
                               errors_seed_456=int(errors[2].sum()), errors_mean=errors.sum(axis=1).mean(),
                               errors_std=errors.sum(axis=1).std(ddof=1),
                               images_wrong_all_seeds=int(errors.all(axis=0).sum()),
                               images_wrong_any_seed=int(errors.any(axis=0).sum())))
    confusion = pd.DataFrame(confusion_rows)
    tables["test_confusions_all"] = confusion
    tables["test_confusions_top_count"] = confusion.sort_values(["model_key", "errors_mean", "true_index", "predicted_index"], ascending=[True, False, True, True]).groupby("model_key", sort=False).head(5)
    tables["test_confusions_top_rate"] = confusion.sort_values(["model_key", "rate_mean", "true_index", "predicted_index"], ascending=[True, False, True, True]).groupby("model_key", sort=False).head(5)
    tables["test_error_stability"] = pd.DataFrame(stability)
    return tables


def number(value, digits=3):
    return f"{float(value):.{digits}f}".replace(".", ",")


def mean_std(row, metric, scale=1, digits=3):
    return number(row[metric + "_mean"] * scale, digits) + " ± " + number(row[metric + "_std"] * scale, digits)


def metric_display(summary, metrics=METRICS, labels=LABELS):
    rows = []
    for model, name in MODELS.items():
        row = summary.set_index("model_key").loc[model]
        rows.append({"Modelo": name, **{label + " (%)": mean_std(row, metric, 100)
                                        for metric, label in zip(metrics, labels)}})
    return pd.DataFrame(rows)


def markdown_table(frame):
    # Avoid optional tabulate dependency in local Jupyter.
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, frame.columns)) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def latex_table(frame, caption, label):
    def escape(value):
        mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                   "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        return "".join(mapping.get(c, c) for c in str(value)).replace("±", r"$\pm$")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + escape(caption) + "}",
             r"\label{" + label + "}", r"\small", r"\begin{tabular}{l" + "r" * (len(frame.columns) - 1) + "}",
             r"\toprule", " & ".join(map(escape, frame.columns)) + r" \\", r"\midrule"]
    lines += [" & ".join(map(escape, row)) + r" \\" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def create_figures(data, tables, output):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    captions = {}

    def save(fig, name, caption):
        fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
        fig.savefig(output / (name + ".png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        captions[name] = caption

    def points(ax, summary, individual, metric, factor=100, ylabel="(%)"):
        for i, (model, name) in enumerate(MODELS.items()):
            row = summary.set_index("model_key").loc[model]
            values = individual.loc[individual.model_key.eq(model)].sort_values("seed")[metric].to_numpy()
            ax.errorbar(i, row[metric + "_mean"] * factor, yerr=row[metric + "_std"] * factor,
                        fmt="o", color=COLORS[i], capsize=4, ms=6, zorder=3)
            ax.scatter(i + np.array([-.10, 0, .10]), values * factor, marker="x", color=COLORS[i], s=20, zorder=4)
        ax.set_xticks(range(4), list(MODELS.values()), rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: number(x, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for i, (phase, title) in enumerate((("validation", "Validação"), ("test", "Teste"))):
        for j, (metric, label) in enumerate((("accuracy", "Acurácia"), ("f1_macro", "F1 macro"))):
            points(axes[i, j], tables[phase + "_summary"], tables[phase + "_individual"], metric)
            axes[i, j].set_title(f"{title}: {label}")
    save(fig, "01_metricas", "Acurácia e F1 macro na validação e no teste. Círculos: média; barras: ±1 DP amostral entre três sementes; cruzes: execuções individuais. Eixos verticais ampliados, sem origem em zero. Não são intervalos de confiança.")

    classes = tables["test_classes"]
    names = [name for name, i in sorted(data.manifest["class_to_idx"].items(), key=lambda x: x[1])]
    heat = classes.pivot(index="class_index", columns="model_key", values="f1_score_mean").reindex(columns=list(MODELS))
    fig, ax = plt.subplots(figsize=(9, 11), layout="constrained")
    im = ax.imshow(heat.to_numpy() * 100, vmin=0, vmax=100, cmap="cividis", aspect="auto")
    display_names = [f"C{i:02d} " + textwrap.fill(name.replace("___", " / ").replace("_", " "), 47) for i, name in enumerate(names)]
    ax.set_yticks(range(38), display_names, fontsize=7)
    ax.set_xticks(range(4), list(MODELS.values()), fontsize=8)
    for i in range(38):
        for j in range(4):
            ax.text(j, i, number(heat.iloc[i, j] * 100, 1), ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, label="F1 médio (%)", fraction=.025)
    save(fig, "02_f1_por_classe", "F1 por classe no teste, média aritmética entre as três sementes. Escala comum 0–100%. Consulte test_classes.csv para DP e suporte. IDs C00–C37 seguem o mapa original de classes.")

    for off_diagonal in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
        arrays = []
        for model in MODELS:
            values = np.stack([data.matrices[(model, s)] for s in SEEDS])
            array = (values / values.sum(axis=2, keepdims=True)).mean(axis=0) * 100
            if off_diagonal:
                np.fill_diagonal(array, 0)
            arrays.append(array)
        upper = max(float(a.max()) for a in arrays) if off_diagonal else 100
        for ax, (model, array) in zip(axes.flat, zip(MODELS, arrays)):
            im = ax.imshow(array, vmin=0, vmax=upper, cmap="Blues", aspect="equal")
            ax.set_title(MODELS[model])
            ax.set_xticks(range(38), range(38), fontsize=4, rotation=90)
            ax.set_yticks(range(38), range(38), fontsize=4)
            ax.set_xlabel("Classe predita (ID)")
            ax.set_ylabel("Classe real (ID)")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65, label="Percentual da classe real (%)")
        name = "04_confusao_erros" if off_diagonal else "03_confusao_completa"
        save(fig, name, "Matrizes de confusão do teste normalizadas por classe real e depois promediadas entre sementes. " +
             ("Diagonal omitida (zero) para revelar erros; escala comum entre modelos, ampliada em relação à figura completa. " if off_diagonal else "Diagonal incluída, escala comum 0–100%. ") +
             "Linhas: classe real; colunas: predição. IDs em class_support.csv. As três passagens usam as mesmas imagens, não 24.375 observações independentes.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for ax, (model, name), color in zip(axes.flat, MODELS.items(), COLORS):
        rows = tables["test_confusions_top_count"].loc[lambda f: f.model_key.eq(model)].iloc[::-1]
        ax.barh(range(len(rows)), rows.errors_mean, xerr=rows.errors_std, color=color, capsize=3)
        ax.set_yticks(range(len(rows)), [f"C{r.true_index:02d} → C{r.predicted_index:02d}" for r in rows.itertuples()])
        ax.set_title(name)
        ax.set_xlabel("Erros por execução (média ± DP)")
        ax.grid(axis="x", alpha=.25)
    save(fig, "05_principais_confusoes", "Até cinco pares direcionais com mais erros médios por modelo no teste. Contagem e DP entre sementes; não são somas de imagens independentes. Frequência absoluta depende do suporte; consulte também test_confusions_top_rate.csv para taxas por classe de origem.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    cost = tables["cost_summary"]
    specifications = (("training_time_minutes", "Treinamento por execução", "Minutos", 1),
                      ("average_time_per_epoch_seconds", "Tempo por época", "Segundos", 1),
                      ("parameters_millions", "Parâmetros totais", "Milhões", None),
                      ("test_latency_ms", "Inferência no teste", "ms/imagem", 1))
    for ax, (metric, title, unit, scale) in zip(axes.flat, specifications):
        if scale is None:
            rows = cost.set_index("model_key").reindex(list(MODELS))
            ax.bar(range(4), rows.parameters_millions, color=COLORS)
            ax.set_xticks(range(4), list(MODELS.values()), rotation=15, ha="right")
            ax.set_ylabel(unit)
            ax.grid(axis="y", alpha=.25)
        else:
            points(ax, cost, tables["cost_individual"], metric, 1, unit)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    save(fig, "06_custo_computacional", "Custo observado nos experimentos originais. Tempos: média ± DP, três sementes; parâmetros: contagem fixa. Treinamento inclui treino e validação por época e depende do número real de épocas. Inferência do teste inclui carregamento de dados e transferência; não é latência isolada de uma imagem nem benchmark no PC local.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for ax, (model, name) in zip(axes.flat, MODELS.items()):
        for seed, color, style in zip(SEEDS, COLORS, ("-", "--", "-.")):
            h = data.histories.loc[data.histories.model_name.eq(name) & data.histories.seed.eq(seed)].sort_values("epoch")
            ax.plot(h.epoch, h.validation_loss, color=color, linestyle=style, label=f"Seed {seed}", lw=1.3)
        ax.set_title(name)
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss de validação")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: number(x, 3)))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    save(fig, "07_curvas_validacao", "Curvas de loss de validação salvas nos históricos. Cada linha termina na época realmente executada; sem interpolação ou alinhamento artificial entre treinamentos com early stopping.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    delta = tables["validation_test_differences"]
    for ax, (metric, label) in zip(axes, (("accuracy", "Acurácia"), ("f1_macro", "F1 macro"))):
        col = metric + "_delta_pp"
        for i, model in enumerate(MODELS):
            values = delta.loc[delta.model_key.eq(model)].sort_values("seed")[col].to_numpy()
            ax.scatter(i + np.array([-.1, 0, .1]), values, color=COLORS[i], marker="x")
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="o", color=COLORS[i], capsize=4)
        ax.axhline(0, color="gray", ls="--", lw=.8)
        ax.set_xticks(range(4), list(MODELS.values()), rotation=15, ha="right")
        ax.set_ylabel("Teste − validação (p.p.)")
        ax.set_title(label)
        ax.grid(axis="y", alpha=.25)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: number(x, 2)))
    save(fig, "08_diferenca_validacao_teste", "Diferenças descritivas entre teste e validação por execução, em pontos percentuais. Média ± DP entre sementes. Os splits contêm imagens distintas; a diferença não é um teste de hipótese nem mede isoladamente sobreajuste.")
    return captions


def create_report(data, tables, captions):
    lines = ["# Resultados e discussão: base empírica para o TCC2", "",
             "## Escopo e método de análise", "",
             "Foram analisados exclusivamente CSVs e JSONs de resultados já salvos. Nenhum modelo foi carregado, treinado ou executado. "
             "As predições existentes foram usadas para conferir matrizes e métricas, sem nova inferência. "
             "São quatro arquiteturas, três sementes (42, 123 e 456), 8.172 imagens de validação e 8.125 de teste, com 38 classes.", "",
             "As médias são aritméticas entre sementes. O desvio padrão é amostral, com ddof=1 e n=3. "
             "Nas tabelas percentuais, os desvios são expressos na mesma escala, em pontos percentuais. "
             "Não representam intervalos de confiança nem incerteza sobre uma população de imagens. "
             "As mesmas imagens são usadas nas três sementes, portanto não há 24.375 observações de teste independentes.", "",
             "Os checkpoints foram fixados pela menor loss de validação antes do teste, conforme o manifesto final. "
             "A comparação de teste é descritiva: não seleciona arquitetura, hiperparâmetros ou checkpoint. "
             "Não foi realizado teste inferencial de superioridade estatística.", "",
             "## Resultados de validação", "", markdown_table(metric_display(tables["validation_summary"])), "",
             "## Resultados finais de teste", "", markdown_table(metric_display(tables["test_summary"])), ""]
    test = tables["test_summary"].set_index("model_key")
    val = tables["validation_summary"].set_index("model_key")
    lines += ["### Interpretação dos resultados globais", ""]
    for model, name in MODELS.items():
        row = test.loc[model]
        gap = (row.f1_weighted_mean - row.f1_macro_mean) * 100
        da = (row.accuracy_mean - val.loc[model].accuracy_mean) * 100
        df = (row.f1_macro_mean - val.loc[model].f1_macro_mean) * 100
        lines += [f"- **{name}:** acurácia de {mean_std(row, 'accuracy', 100)}% e F1 macro de {mean_std(row, 'f1_macro', 100)}%. "
                  f"A diferença média teste − validação foi {number(da)} p.p. em acurácia e {number(df)} p.p. em F1 macro. "
                  f"O F1 ponderado excedeu o macro em {number(gap)} p.p., indicando que a agregação pelo suporte favorece classes mais frequentes com maior desempenho."]
    max_accuracy = test.accuracy_mean.idxmax()
    max_f1 = test.f1_macro_mean.idxmax()
    lines += ["", f"A maior média observada de acurácia foi a de {MODELS[max_accuracy]}, e a maior média observada de F1 macro foi a de {MODELS[max_f1]}. "
              "Isso descreve esta divisão fixa e estas três sementes, sem demonstrar superioridade estatística ou justificar nova seleção pelo teste.", "",
              f"DenseNet201 − ResNet50 correspondeu a {number(100 * (test.loc['densenet201'].accuracy_mean - test.loc['resnet50'].accuracy_mean))} p.p. de acurácia e "
              f"{number(100 * (test.loc['densenet201'].f1_macro_mean - test.loc['resnet50'].f1_macro_mean))} p.p. de F1 macro. "
              "A ordenação depende da métrica. A acurácia é influenciada pelo suporte das classes, enquanto o F1 macro dá igual peso a cada classe. "
              "O recall ponderado coincide com a acurácia neste problema de classificação multiclasse com um rótulo por imagem, por identidade matemática, não como evidência independente.", "",
              "As diferenças entre validação e teste não indicam isoladamente sobreajuste: os conjuntos são distintos, com composição e suporte próprios, e a validação foi usada para escolher a época. "
              "Nenhum intervalo de confiança ou valor-p é inferido das barras de desvio padrão.", "",
              "## Dificuldades de classificação e suporte", ""]
    support = tables["class_support"]
    small, large = support.loc[support.test_images.idxmin()], support.loc[support.test_images.idxmax()]
    lines += [f"O suporte de teste varia de {int(small.test_images)} imagens ({small.class_name}) a {int(large.test_images)} ({large.class_name}), "
              f"razão de {number(large.test_images / small.test_images, 2)} entre extremos. "
              f"Há {data.split.leaf_id.nunique()} leaf_id distintos no teste. Imagens da mesma folha podem permanecer correlacionadas dentro do split. "
              "A tabela class_support.csv fornece imagens, folhas distintas e a variação de recall causada por um único erro por classe.", ""]
    for model, name in MODELS.items():
        difficult = tables["test_difficult_classes"].loc[lambda f: f.model_key.eq(model)]
        display = pd.DataFrame([{"Classe": r.class_name, "n": int(r.support),
                                 "Precisão (%)": mean_std(r._asdict(), "precision", 100, 2),
                                 "Recall (%)": mean_std(r._asdict(), "recall", 100, 2),
                                 "F1 (%)": mean_std(r._asdict(), "f1_score", 100, 2)} for r in difficult.itertuples()])
        lines += [f"### {name}: cinco menores F1 médios de teste", "", markdown_table(display), ""]
        top = tables["test_confusions_top_count"].loc[lambda f: f.model_key.eq(model)]
        conf = pd.DataFrame([{"Classe real": r.true_class, "Classe predita": r.predicted_class,
                              "Erros/execução": mean_std(r._asdict(), "errors", 1, 2),
                              "Taxa origem (%)": mean_std(r._asdict(), "rate", 100, 2),
                              "n origem": int(r.source_support), "Seeds com erro": int(r.seeds_with_error)} for r in top.itertuples()])
        lines += ["Principais confusões direcionais por contagem média:", "", markdown_table(conf), ""]
    lines += ["Os pares acima são observações das matrizes, sem inspeção visual das imagens. "
              "Semelhança de sintomas, qualidade da imagem ou erros de anotação seriam hipóteses a investigar, não causas demonstradas por estas métricas. "
              "Taxas por classe de origem e contagens respondem a perguntas diferentes: uma classe frequente pode contribuir mais erros absolutos sem ter a maior taxa. "
              "O arquivo test_confusions_top_rate.csv apresenta a comparação por taxa.", "",
              "### Persistência dos erros entre sementes", "", markdown_table(tables["test_error_stability"][["model_name", "errors_seed_42", "errors_seed_123", "errors_seed_456", "images_wrong_all_seeds", "images_wrong_any_seed"]]), "",
              "Erros em todas as sementes apontam exemplos persistentemente difíceis nas execuções disponíveis. "
              "A união e a interseção não são desempenho de ensemble: nenhuma combinação de modelos foi executada ou selecionada.", "",
              "## Custo computacional observado", ""]
    costs = tables["cost_summary"].set_index("model_key")
    display = []
    for model, name in MODELS.items():
        row = costs.loc[model]
        display.append({"Modelo": name, "Parâmetros (M)": number(row.parameters_millions, 3),
                        "Treino min": mean_std(row, "training_time_minutes", 1, 2),
                        "Época s": mean_std(row, "average_time_per_epoch_seconds", 1, 2),
                        "Épocas": mean_std(row, "epochs_executed", 1, 2),
                        "Inferência val. ms/img": mean_std(row, "inference_milliseconds_per_image", 1, 3),
                        "Inferência teste ms/img": mean_std(row, "test_latency_ms", 1, 3)})
    lines += [markdown_table(pd.DataFrame(display)), ""]
    environment = [c for c in ("gpu", "gpu_total_memory_gb", "selected_device", "pytorch_version", "cuda_version", "batch_size", "num_workers", "input_size") if c in data.validation]
    conditions = data.validation[environment].drop_duplicates()
    lines += ["Condições registradas na validação:", "", markdown_table(conditions), "",
              f"Teste: GPU {data.manifest.get('gpu')}, PyTorch {data.manifest.get('torch')}, torchvision {data.manifest.get('torchvision')}, "
              f"CUDA {data.manifest.get('cuda')}, batch {data.manifest.get('batch_size')}, workers {data.manifest.get('num_workers')}. "
              f"Aquecimento de teste registrado: {sorted(data.inference.warmup_batches.unique().tolist())} batches.", ""]
    if len(conditions) != 1:
        lines += ["**As condições registradas variam entre execuções. Os tempos não sustentam uma comparação controlada entre arquiteturas.**", ""]
    lines += [f"O tempo acumulado de treinamento das 12 execuções, somando os tempos por época registrados, foi "
              f"{number(data.validation.training_time_seconds.sum() / 3600, 2)} horas. "
              "Esse valor inclui treino e validação por época, mas não necessariamente inicialização, leitura de checkpoints, gráficos ou demais custos fora do loop. "
              "A soma das três seeds por arquitetura está em cost_summary.csv.", "",
              "O tempo total depende tanto do custo por época quanto do early stopping. "
              "O número de épocas não é mantido igual entre arquiteturas, portanto o total não mede somente complexidade da rede. "
              "A variabilidade do tempo por época é calculada entre as médias de cada execução, sem tratar cada época como uma repetição independente.", ""]
    if data.validation.loc[data.validation.model_key.eq("baseline_cnn"), "epochs_executed"].eq(30).all():
        lines += ["A baseline alcançou 30 épocas em todas as sementes. Isso descreve a duração observada; não demonstra, por si só, convergência, subtreinamento ou necessidade de alterar o protocolo após observar o teste.", ""]
    lines += [f"EfficientNet-B0 registrou {number(costs.loc['efficientnet_b0'].parameters_millions / costs.loc['baseline_cnn'].parameters_millions, 2)} vezes os parâmetros da baseline, "
              f"com tempo médio por época de {mean_std(costs.loc['efficientnet_b0'], 'average_time_per_epoch_seconds', 1, 2)} s frente a "
              f"{mean_std(costs.loc['baseline_cnn'], 'average_time_per_epoch_seconds', 1, 2)} s. "
              "Contagem de parâmetros não equivale a FLOPs, memória, consumo de energia ou latência: esses fatores não podem ser concluídos sem medições próprias.", "",
              "Os tempos de inferência são medidas end-to-end com batches, carregamento e transferência de dados, divididas pelo número de imagens. "
              "Não são latências de requisições individuais. O teste foi registrado sem aquecimento adicional, e o protocolo de validação usa aquecimento em CUDA no código de origem. "
              "Os tempos dos dois splits são apresentados separadamente e não interpretados como aceleração causada pelo split. "
              "Executar esta análise no PC apenas processa arquivos: não mede o desempenho dos modelos no hardware local.", "",
              "## Integridade e arquivos ausentes", "",
              f"Os {len(data.manifest['artifacts_sha256'])} artefatos do manifesto de teste foram encontrados e seus hashes conferidos. "
              "As matrizes absolutas e normalizadas, predições, suportes e sete métricas foram cruzadas. "
              "As médias e DPs publicados foram conferidos com os CSVs individuais, incluindo validação. "
              "Os tempos e melhores épocas foram conferidos com os históricos de validação.", ""]
    historical = data.historical_sources
    lines += [markdown_table(historical.groupby("status").size().rename("Arquivos").reset_index()), "",
              "A lista exata de caminhos está em tables/historical_sources.csv. Artefatos históricos ausentes/divergentes "
              "não impedem a análise dos CSVs consolidados existentes, mas limitam a auditoria retrospectiva dos treinamentos. "
              "Nenhum resultado individual local divergente foi usado para substituir uma linha consolidada.", "",
              "### Arquivos adicionais para auditoria retrospectiva completa", ""]
    for row in historical.loc[~historical.status.isin(["hash_idêntico", "equivalente_após_CRLF_para_LF"])].itertuples():
        lines.append(f"- `{row.path}` — {row.status}. Fonte esperada no experimento: `{row.original_path}`.")
    lines += ["", "### Matrizes de validação", "",
              "O conjunto consolidado de validação contém métricas por classe, mas não contém matrizes numéricas nem predições por imagem das 12 execuções. "
              "O pipeline de origem salva validation_confusion_matrix.png. Uma figura PNG não permite recuperar contagens exatas com segurança. "
              "Por isso, a análise direcional de confusões usa exclusivamente as matrizes reais de teste. "
              "Para estendê-la à validação, seriam necessários CSVs de matrizes ou predições já salvas para cada um dos 12 pares modelo/seed. "
              "Esta tarefa não executa inferência para criá-los.", "",
              "## Limitações e alcance das conclusões", "",
              "- Três sementes quantificam apenas uma parte da variabilidade do treinamento em uma divisão fixa. Não substituem replicações de splits ou validação externa.",
              "- Nenhum procedimento inferencial de comparação entre modelos foi aplicado. Não há declaração de superioridade estatística, e sobreposição de barras de DP não constitui um teste.",
              "- O isolamento por leaf_id foi conferido nos metadados disponíveis. Isso não prova ausência de todas as fontes de dependência, duplicação visual ou diferenças entre ambientes de captura.",
              "- O experimento é interno ao PlantVillage. Generalização para folhas fotografadas em campo, outros dispositivos e condições de iluminação exige dados externos. Não se afirma desempenho em campo a partir destes resultados.",
              "- A comparação entre baseline e redes de transferência combina diferenças de arquitetura e inicialização pré-treinada. Não isola causalmente o efeito de cada fator.",
              "- Classes com pouco suporte têm métricas sensíveis a poucos erros. F1 perfeito em uma classe pequena não implica ausência de incerteza.",
              "- Não foram fornecidas medidas de FLOPs, pico de memória ou energia. Não são estimadas a partir da quantidade de parâmetros.",
              "- A autenticidade histórica das validações consolidadas depende dos arquivos exportados. A consistência interna foi conferida, mas a ausência dos originais de treinamento impede verificar todos os seus hashes localmente.",
              "- Os resultados de teste não serão usados para refazer escolhas experimentais. Observações por classe são análises posteriores descritivas.", "",
              "## Observações de proveniência", ""]
    lines += ["- " + note for note in data.notes]
    lines += ["", "## Figuras e utilização no LaTeX", "",
              "As figuras estão em PDF vetorial com fontes incorporadas e PNG de 300 dpi. Use os PDFs no documento LaTeX. "
              "As tabelas .tex requerem booktabs, e as figuras requerem graphicx. Tabelas extensas também estão em CSV para adaptação à largura do texto. "
              "As tabelas compactas de desempenho foram divididas em duas partes por split para evitar sete métricas em uma única linha larga.", ""]
    for name, caption in captions.items():
        lines += [f"### {name}", "", f"![{caption}](figures/{name}.png)", "", caption,
                  f" PDF: [figures/{name}.pdf](figures/{name}.pdf).", ""]
    lines += ["Exemplo (ajuste o caminho relativo ao projeto LaTeX):", "", "```latex",
              r"\begin{figure}[htbp]", r"  \centering", r"  \includegraphics[width=\linewidth]{figures/01_metricas.pdf}",
              r"  \caption{Acurácia e F1 macro na validação e no teste. Média e desvio padrão amostral entre três sementes.}",
              r"  \label{fig:metricas}", r"\end{figure}", "```", "",
              "## Fontes locais", "", f"- Validação: `{data.root / 'repetitions_comparison'}`.",
              f"- Teste: `{data.test_dir}`.", "- Hashes de todas as fontes consultadas: `analysis_manifest.json`.",
              "- Inventário de CSVs encontrados: `tables/csv_inventory.csv`.", ""]
    return "\n".join(lines)


def run_analysis(project_root, test_dir=None, output_dir=None):
    project = Path(project_root).resolve()
    data = load_saved_results(project / "results", test_dir)
    for relative in ("src/analyze_results.py", "notebooks/09_analise_resultados.ipynb", "requirements-analysis.txt"):
        path = project / relative
        if path.is_file():
            data.sources[str(path)] = digest(path)
    tables = scientific_tables(data)
    output = Path(output_dir).resolve() if output_dir else project / "reports" / "resultados" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    require(not output.is_relative_to(data.root), "A saída deve ficar fora de results/.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    (output / "figures").mkdir()
    manifest = {"status": "running", "sources_sha256": data.sources, "test_dir": str(data.test_dir),
                "std_ddof": 1, "n_seeds": 3, "model_inference_executed": False,
                "statistical_superiority_test_performed": False,
                "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    try:
        tables["csv_inventory"] = data.inventory
        tables["historical_sources"] = data.historical_sources
        for name, frame in tables.items():
            frame.to_csv(output / "tables" / (name + ".csv"), index=False)
        for phase, label in (("validation", "Validação"), ("test", "Teste")):
            for suffix, metrics, labels in (("macro", METRICS[:4], LABELS[:4]),
                                            ("weighted", METRICS[4:], LABELS[4:])):
                display = metric_display(tables[phase + "_summary"], metrics, labels)
                display.to_csv(output / "tables" / f"{phase}_{suffix}_formatted.csv", index=False)
                (output / "tables" / f"{phase}_{suffix}.tex").write_text(
                    latex_table(display, f"{label}: média e DP amostral entre três sementes. Valores em porcentagem, DP na mesma escala.",
                                f"tab:{phase}-{suffix}"), encoding="utf-8")
        cost_summary = tables["cost_summary"]
        cost_display = pd.DataFrame([{"Modelo": r.model_name, "Parâm. (M)": number(r.parameters_millions, 3),
                                      "Treino (min)": mean_std(r._asdict(), "training_time_minutes", 1, 2),
                                      "Época (s)": mean_std(r._asdict(), "average_time_per_epoch_seconds", 1, 2),
                                      "Teste (ms/img)": mean_std(r._asdict(), "test_latency_ms", 1, 3)} for r in cost_summary.itertuples()])
        (output / "tables/cost.tex").write_text(latex_table(cost_display, "Custo computacional observado. Tempos: média e DP entre três sementes.", "tab:cost"), encoding="utf-8")
        captions = create_figures(data, tables, output / "figures")
        (output / "relatorio_resultados.md").write_text(create_report(data, tables, captions), encoding="utf-8")
        (output / "figures/captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
        for path, before in data.sources.items():
            require(digest(path) == before, f"Fonte modificada durante a análise: {path}")
        manifest["status"] = "complete"
        manifest["outputs_sha256"] = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(run_analysis(args.project_root, args.test_dir, args.output_dir))
