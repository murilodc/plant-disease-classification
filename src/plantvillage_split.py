"""Utilities for splitting PlantVillage metadata by leaf_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_SEED = 42
SPLIT_PROPORTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
REQUIRED_COLUMNS = ["classe", "leaf_id"]


def split_metadata_by_leaf_id(
    metadata: pd.DataFrame,
    proportions: dict[str, float] | None = None,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Return metadata with a split column, keeping each leaf_id indivisible."""
    proportions = SPLIT_PROPORTIONS if proportions is None else proportions
    _validate_proportions(proportions)
    proportions = {split: float(proportions[split]) for split in SPLIT_PROPORTIONS}
    split_names = list(SPLIT_PROPORTIONS)
    validate_leaf_id_single_class(metadata)

    groups = (
        metadata.groupby(["classe", "leaf_id"], dropna=False)
        .size()
        .rename("quantidade_imagens")
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    assignments: dict[object, str] = {}

    for classe in sorted(groups["classe"].unique()):
        class_groups = groups.loc[groups["classe"].eq(classe)].copy()
        class_assignments = _split_class_groups(class_groups, proportions, split_names, rng)
        assignments.update(class_assignments)

    result = metadata.copy()
    result["split"] = result["leaf_id"].map(assignments)

    if result["split"].isna().any():
        missing = int(result["split"].isna().sum())
        raise RuntimeError(f"{missing} imagens ficaram sem split.")

    return result


def validate_leaf_id_single_class(metadata: pd.DataFrame) -> None:
    """Raise an error if any leaf_id appears in more than one class."""
    _validate_required_columns(metadata, REQUIRED_COLUMNS)

    missing = metadata[REQUIRED_COLUMNS].isna().any()
    if missing.any():
        columns = ", ".join(missing[missing].index)
        raise ValueError(f"Colunas obrigatorias com valores ausentes: {columns}")

    class_counts = metadata.groupby("leaf_id", dropna=False)["classe"].nunique(dropna=False)
    invalid = class_counts.loc[class_counts > 1]
    if invalid.empty:
        return

    examples = (
        metadata.loc[metadata["leaf_id"].isin(invalid.index)]
        .groupby("leaf_id", dropna=False)["classe"]
        .apply(lambda values: ", ".join(sorted(values.astype(str).unique())))
        .head(10)
    )
    details = "; ".join(f"{leaf_id}: {classes}" for leaf_id, classes in examples.items())
    raise ValueError(
        "Ha leaf_id associado a mais de uma classe. "
        f"Total de leaf_id invalidos: {len(invalid)}. Exemplos: {details}"
    )


def split_diagnostics(
    metadata_split: pd.DataFrame,
    expected_total: int = 54305,
    expected_classes: int = 38,
    split_names: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, pd.DataFrame]:
    """Build validation tables for the metadata split."""
    _validate_required_columns(metadata_split, [*REQUIRED_COLUMNS, "split"])

    total_images = len(metadata_split)
    split_series = metadata_split["split"].astype("string")

    imagens_por_split = (
        split_series.value_counts()
        .reindex(split_names, fill_value=0)
        .rename_axis("split")
        .reset_index(name="quantidade_imagens")
    )
    imagens_por_split["percentual_imagens"] = (
        imagens_por_split["quantidade_imagens"].div(total_images).mul(100).round(4)
    )

    leaf_split = metadata_split[["leaf_id", "split"]].drop_duplicates()
    leaf_ids_por_split = (
        leaf_split["split"]
        .value_counts()
        .reindex(split_names, fill_value=0)
        .rename_axis("split")
        .reset_index(name="quantidade_leaf_id")
    )

    imagens_por_classe_split = (
        pd.crosstab(metadata_split["classe"], metadata_split["split"])
        .reindex(columns=split_names, fill_value=0)
        .sort_index()
    )
    percentual_classe_por_split = (
        imagens_por_classe_split.div(imagens_por_classe_split.sum(axis=1), axis=0)
        .mul(100)
        .round(4)
    )

    classes_por_split = (
        metadata_split.groupby("split")["classe"]
        .nunique()
        .reindex(split_names, fill_value=0)
        .rename_axis("split")
        .reset_index(name="quantidade_classes")
    )

    leaf_split_counts = metadata_split.groupby("leaf_id", dropna=False)["split"].nunique()
    leaf_id_ok = bool(leaf_split_counts.le(1).all())
    invalid_splits = ~split_series.isin(split_names)
    assigned_ok = bool(not invalid_splits.any() and total_images == expected_total)
    classes_ok = bool(
        metadata_split["classe"].nunique() == expected_classes
        and classes_por_split["quantidade_classes"].eq(expected_classes).all()
        and imagens_por_classe_split.gt(0).all(axis=None)
    )

    validacoes = pd.DataFrame(
        [
            {
                "validacao": "nenhum_leaf_id_em_mais_de_um_split",
                "ok": leaf_id_ok,
                "detalhe": f"{int(leaf_split_counts.gt(1).sum())} leaf_id repetidos",
            },
            {
                "validacao": "todas_as_54305_imagens_atribuidas",
                "ok": assigned_ok,
                "detalhe": (
                    f"{total_images} imagens; {int(split_series.isna().sum())} sem split; "
                    f"{int(invalid_splits.sum())} splits invalidos"
                ),
            },
            {
                "validacao": "as_38_classes_aparecem_nos_tres_splits",
                "ok": classes_ok,
                "detalhe": (
                    f"{metadata_split['classe'].nunique()} classes no total; "
                    f"minimo por split: {int(classes_por_split['quantidade_classes'].min())}"
                ),
            },
        ]
    )

    return {
        "imagens_por_split": imagens_por_split,
        "leaf_ids_por_split": leaf_ids_por_split,
        "imagens_por_classe_split": imagens_por_classe_split,
        "percentual_classe_por_split": percentual_classe_por_split,
        "classes_por_split": classes_por_split,
        "validacoes": validacoes,
    }


def save_split_metadata_csv(metadata_split: pd.DataFrame, output_path: str | Path) -> Path:
    """Save split metadata CSV and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_split.to_csv(path, index=False)
    return path


def _split_class_groups(
    class_groups: pd.DataFrame,
    proportions: dict[str, float],
    split_names: list[str],
    rng: np.random.Generator,
) -> dict[object, str]:
    if len(class_groups) < len(split_names):
        classe = class_groups["classe"].iloc[0]
        raise ValueError(f"A classe {classe} nao possui leaf_id suficientes para tres splits.")

    total = int(class_groups["quantidade_imagens"].sum())
    targets = {split: total * proportions[split] for split in split_names}

    groups = class_groups.assign(_random=rng.random(len(class_groups))).sort_values(
        ["quantidade_imagens", "_random", "leaf_id"],
        ascending=[False, True, True],
    )

    counts = {split: 0 for split in split_names}
    leaf_counts = {split: 0 for split in split_names}
    assignments: dict[object, str] = {}

    for position, row in enumerate(groups.itertuples(index=False)):
        leaf_id = row.leaf_id
        weight = int(row.quantidade_imagens)
        remaining = len(groups) - position - 1
        empty_splits = [split for split in split_names if leaf_counts[split] == 0]
        candidates = empty_splits if len(empty_splits) > remaining else split_names
        split = min(
            candidates,
            key=lambda candidate: _score_after_move(counts, targets, candidate, weight),
        )
        assignments[leaf_id] = split
        counts[split] += weight
        leaf_counts[split] += 1

    return _improve_assignments(groups, assignments, counts, leaf_counts, targets, split_names)


def _improve_assignments(
    groups: pd.DataFrame,
    assignments: dict[object, str],
    counts: dict[str, int],
    leaf_counts: dict[str, int],
    targets: dict[str, float],
    split_names: list[str],
) -> dict[object, str]:
    current_score = _score(counts, targets)

    while True:
        best_move = None
        best_score = current_score

        for row in groups.itertuples(index=False):
            leaf_id = row.leaf_id
            weight = int(row.quantidade_imagens)
            origin = assignments[leaf_id]

            if leaf_counts[origin] <= 1:
                continue

            for destination in split_names:
                if destination == origin:
                    continue

                next_counts = counts.copy()
                next_counts[origin] -= weight
                next_counts[destination] += weight
                next_score = _score(next_counts, targets)

                if next_score < best_score:
                    best_score = next_score
                    best_move = (leaf_id, weight, origin, destination)

        if best_move is None:
            return assignments

        leaf_id, weight, origin, destination = best_move
        assignments[leaf_id] = destination
        counts[origin] -= weight
        counts[destination] += weight
        leaf_counts[origin] -= 1
        leaf_counts[destination] += 1
        current_score = best_score


def _score_after_move(
    counts: dict[str, int],
    targets: dict[str, float],
    split: str,
    weight: int,
) -> float:
    next_counts = counts.copy()
    next_counts[split] += weight
    return _score(next_counts, targets)


def _score(counts: dict[str, int], targets: dict[str, float]) -> float:
    total = sum(targets.values())
    return sum(((counts[split] - target) / total) ** 2 for split, target in targets.items())


def _validate_proportions(proportions: dict[str, float]) -> None:
    if set(proportions) != set(SPLIT_PROPORTIONS):
        expected = ", ".join(SPLIT_PROPORTIONS)
        raise ValueError(f"Splits esperados: {expected}")

    total = sum(proportions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"As proporcoes devem somar 1.0, mas somam {total}.")

    if any(value <= 0 for value in proportions.values()):
        raise ValueError("Todas as proporcoes devem ser positivas.")


def _validate_required_columns(metadata: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in metadata]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(missing)}")


__all__ = [
    "REQUIRED_COLUMNS",
    "SPLIT_PROPORTIONS",
    "SPLIT_SEED",
    "save_split_metadata_csv",
    "split_diagnostics",
    "split_metadata_by_leaf_id",
    "validate_leaf_id_single_class",
]
