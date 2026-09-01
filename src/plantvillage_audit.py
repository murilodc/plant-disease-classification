"""Utilities for auditing PlantVillage metadata without extracting images."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


DEFAULT_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
FALLBACK_PREFIX = "fallback_"

METADATA_COLUMNS = [
    "zip_path",
    "filename",
    "classe",
    "cultura",
    "doenca",
    "leaf_id",
    "leaf_id_found",
    "leaf_id_source",
    "leaf_match_status",
    "leaf_lookup_key",
    "leaf_suggestions_count",
]


def normalize_image_identifier(filename: str) -> str:
    """Normalize an image filename using the PlantVillage loader logic."""
    image_identifier = str(filename).replace("_final_masked", "")

    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]

    image_identifier = image_identifier.split("copy")[0]
    image_identifier = (
        image_identifier.replace(".jpg", "")
        .replace(".JPG", "")
        .replace(".png", "")
        .replace(".PNG", "")
    )
    image_identifier = image_identifier.strip()
    return image_identifier.lower()


def load_leaf_map(leaf_map_path: str | Path) -> dict[str, list[str]]:
    """Load leaf-map.json and normalize keys for lookup."""
    path = Path(leaf_map_path)
    with path.open("r", encoding="utf-8") as file:
        raw_leaf_map = json.load(file)

    if not isinstance(raw_leaf_map, Mapping):
        raise TypeError("leaf-map.json must contain a JSON object.")

    leaf_map: dict[str, list[str]] = {}
    for key, value in raw_leaf_map.items():
        lookup_key = str(key).strip().lower()
        leaf_map.setdefault(lookup_key, []).extend(_coerce_suggestions(value))

    return leaf_map


def resolve_leaf_id(
    filename: str,
    classe: str,
    leaf_map: Mapping[str, Sequence[str]],
    fallback_prefix: str = FALLBACK_PREFIX,
) -> dict[str, Any]:
    """Find the PlantVillage grouping identifier for one image."""
    lookup_key = normalize_image_identifier(filename)
    fallback_leaf_id = f"{fallback_prefix}{lookup_key}"
    suggestions = leaf_map.get(lookup_key)

    if suggestions is None:
        return {
            "leaf_id": fallback_leaf_id,
            "leaf_id_found": False,
            "leaf_id_source": "fallback",
            "leaf_match_status": "not_found",
            "leaf_lookup_key": lookup_key,
            "leaf_suggestions_count": 0,
        }

    suggestions = _coerce_suggestions(suggestions)
    if len(suggestions) == 1:
        return {
            "leaf_id": suggestions[0],
            "leaf_id_found": True,
            "leaf_id_source": "leaf-map",
            "leaf_match_status": "matched_unique",
            "leaf_lookup_key": lookup_key,
            "leaf_suggestions_count": 1,
        }

    if len(suggestions) > 1:
        for suggestion in suggestions:
            if classe in suggestion:
                return {
                    "leaf_id": suggestion,
                    "leaf_id_found": True,
                    "leaf_id_source": "leaf-map",
                    "leaf_match_status": "matched_by_class",
                    "leaf_lookup_key": lookup_key,
                    "leaf_suggestions_count": len(suggestions),
                }

        return {
            "leaf_id": fallback_leaf_id,
            "leaf_id_found": False,
            "leaf_id_source": "fallback",
            "leaf_match_status": "ambiguous_no_class_match",
            "leaf_lookup_key": lookup_key,
            "leaf_suggestions_count": len(suggestions),
        }

    return {
        "leaf_id": fallback_leaf_id,
        "leaf_id_found": False,
        "leaf_id_source": "fallback",
        "leaf_match_status": "empty_suggestions",
        "leaf_lookup_key": lookup_key,
        "leaf_suggestions_count": 0,
    }


def build_metadata_dataframe(
    zip_path: str | Path,
    leaf_map_path: str | Path,
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> pd.DataFrame:
    """Create one metadata row per image in raw/color/ inside data.zip."""
    leaf_map = load_leaf_map(leaf_map_path)
    extensions = {extension.lower() for extension in image_extensions}
    records: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue

            path_info = _extract_raw_color_image_path(member.filename, extensions)
            if path_info is None:
                continue

            classe = path_info["classe"]
            cultura, doenca = split_class_name(classe)
            leaf_info = resolve_leaf_id(path_info["filename"], classe, leaf_map)

            records.append(
                {
                    "zip_path": path_info["zip_path"],
                    "filename": path_info["filename"],
                    "classe": classe,
                    "cultura": cultura,
                    "doenca": doenca,
                    **leaf_info,
                }
            )

    return pd.DataFrame.from_records(records, columns=METADATA_COLUMNS)


def audit_metadata(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize PlantVillage metadata quality and class distribution."""
    _validate_columns(metadata)

    total_images = int(len(metadata))
    source = _source_series(metadata["leaf_id_source"])
    from_leaf_map = source.eq("leaf-map")
    using_fallback = source.eq("fallback")
    leaf_map_count = int(from_leaf_map.sum())
    fallback_count = int(using_fallback.sum())
    fallback_percent = round((fallback_count / total_images) * 100, 4) if total_images else 0.0

    fallback_metadata = metadata.loc[using_fallback].copy()
    group_counts = metadata.groupby("leaf_id").size()
    groups_with_multiple_images = group_counts[group_counts > 1]
    fallback_group_summary = _leaf_id_group_summary(fallback_metadata)
    fallbacks_with_multiple_images = fallback_group_summary.loc[
        fallback_group_summary["quantidade_imagens"] > 1
    ]
    fallbacks_multiclass = _fallbacks_multiclass_table(fallback_metadata)

    diseases = metadata.loc[
        ~metadata["doenca"].astype("string").str.lower().eq("healthy").fillna(False),
        ["cultura", "doenca"],
    ].drop_duplicates()

    summary = pd.DataFrame(
        [
            {
                "total_imagens": total_images,
                "numero_classes": int(metadata["classe"].nunique(dropna=True)),
                "numero_culturas": int(metadata["cultura"].nunique(dropna=True)),
                "numero_doencas": int(len(diseases)),
                "imagens_associadas_leaf_map": leaf_map_count,
                "imagens_usando_fallback": fallback_count,
                "percentual_usando_fallback": fallback_percent,
                "numero_identificadores_agrupamento_unicos": int(
                    metadata["leaf_id"].nunique(dropna=True)
                ),
                "imagens_em_grupos_com_multiplas_imagens": int(
                    groups_with_multiple_images.sum()
                ),
                "grupos_com_multiplas_imagens": int(len(groups_with_multiple_images)),
                "fallbacks_unicos": int(fallback_metadata["leaf_id"].nunique(dropna=True)),
                "fallbacks_com_mais_de_uma_imagem": int(len(fallbacks_with_multiple_images)),
                "fallbacks_em_mais_de_uma_classe": int(len(fallbacks_multiclass)),
            }
        ]
    )

    class_counts = (
        metadata["classe"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("classe")
        .reset_index(name="quantidade")
    )

    leaf_status_counts = (
        metadata["leaf_match_status"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("leaf_match_status")
        .reset_index(name="quantidade")
    )

    leaf_source_counts = (
        metadata["leaf_id_source"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("leaf_id_source")
        .reset_index(name="quantidade")
    )

    return {
        "resumo": summary,
        "imagens_por_classe": class_counts,
        "origem_leaf_id": leaf_source_counts,
        "status_leaf_id": leaf_status_counts,
        "maiores_grupos_leaf_id": _leaf_id_group_summary(metadata).head(20),
        "maiores_grupos_fallback": fallback_group_summary.head(20),
        "fallback_r_imagens": _leaf_id_detail(metadata, "fallback_r"),
        "fallback_r_imagens_por_classe": _leaf_id_class_counts(metadata, "fallback_r"),
        "fallbacks_multiclasse": fallbacks_multiclass,
    }


def save_metadata_csv(metadata: pd.DataFrame, output_path: str | Path) -> Path:
    """Save metadata CSV and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(path, index=False)
    return path


def split_class_name(classe: str) -> tuple[str, str]:
    """Split a PlantVillage class folder into crop and disease/condition."""
    parts = str(classe).split("___", 1)
    cultura = parts[0]
    doenca = parts[1] if len(parts) > 1 else "unknown"
    return cultura, doenca


def _coerce_suggestions(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, Sequence):
        return [str(item) for item in value]

    return [str(value)]


def _extract_raw_color_image_path(
    zip_member_path: str,
    image_extensions: set[str],
) -> dict[str, str] | None:
    normalized_path = str(zip_member_path).replace("\\", "/").lstrip("/")
    parts = [part for part in normalized_path.split("/") if part]
    lower_parts = [part.lower() for part in parts]

    for index in range(len(parts) - 3):
        if lower_parts[index] == "raw" and lower_parts[index + 1] == "color":
            filename = parts[-1]
            if PurePosixPath(filename).suffix.lower() not in image_extensions:
                return None

            return {
                "zip_path": normalized_path,
                "classe": parts[index + 2],
                "filename": filename,
            }

    return None


def _validate_columns(metadata: pd.DataFrame) -> None:
    missing_columns = [column for column in METADATA_COLUMNS if column not in metadata]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"Missing required metadata columns: {joined_columns}")


def _source_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.fillna("")


def _leaf_id_group_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["leaf_id", "quantidade_imagens", "quantidade_classes", "leaf_id_source"]
    if metadata.empty:
        return pd.DataFrame(columns=columns)

    return (
        metadata.groupby("leaf_id", dropna=False)
        .agg(
            quantidade_imagens=("leaf_id", "size"),
            quantidade_classes=("classe", "nunique"),
            leaf_id_source=("leaf_id_source", _join_unique),
        )
        .reset_index()
        .sort_values(["quantidade_imagens", "leaf_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _leaf_id_detail(metadata: pd.DataFrame, leaf_id: str) -> pd.DataFrame:
    columns = [
        "filename",
        "classe",
        "leaf_lookup_key",
        "leaf_id",
        "quantidade_classes_no_grupo",
        "quantidade_imagens_na_classe",
    ]
    rows = metadata.loc[
        metadata["leaf_id"].eq(leaf_id),
        ["filename", "classe", "leaf_lookup_key", "leaf_id"],
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)

    class_counts = (
        rows["classe"]
        .value_counts()
        .rename_axis("classe")
        .reset_index(name="quantidade_imagens_na_classe")
    )
    rows = rows.merge(class_counts, on="classe", how="left")
    rows["quantidade_classes_no_grupo"] = int(rows["classe"].nunique())
    return rows[columns].sort_values(["classe", "filename"]).reset_index(drop=True)


def _leaf_id_class_counts(metadata: pd.DataFrame, leaf_id: str) -> pd.DataFrame:
    rows = metadata.loc[metadata["leaf_id"].eq(leaf_id)]
    if rows.empty:
        return pd.DataFrame(columns=["classe", "quantidade"])

    return (
        rows["classe"]
        .value_counts()
        .rename_axis("classe")
        .reset_index(name="quantidade")
        .sort_values(["quantidade", "classe"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _fallbacks_multiclass_table(fallback_metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["leaf_id", "quantidade_imagens", "quantidade_classes", "classes"]
    if fallback_metadata.empty:
        return pd.DataFrame(columns=columns)

    group_summary = _leaf_id_group_summary(fallback_metadata)
    result = group_summary.loc[group_summary["quantidade_classes"] > 1].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)

    classes = (
        fallback_metadata.groupby("leaf_id", dropna=False)["classe"]
        .apply(_join_unique)
        .rename("classes")
        .reset_index()
    )
    return (
        result[["leaf_id", "quantidade_imagens", "quantidade_classes"]]
        .merge(classes, on="leaf_id", how="left")
        .sort_values(["quantidade_imagens", "leaf_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _join_unique(values: pd.Series) -> str:
    return ", ".join(sorted({str(value) for value in values.dropna()}))


__all__ = [
    "DEFAULT_IMAGE_EXTENSIONS",
    "METADATA_COLUMNS",
    "FALLBACK_PREFIX",
    "audit_metadata",
    "build_metadata_dataframe",
    "load_leaf_map",
    "normalize_image_identifier",
    "resolve_leaf_id",
    "save_metadata_csv",
    "split_class_name",
]
