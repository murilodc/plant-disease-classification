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
MISSING_LEAF_ID = "LEAF_ID_NOT_FOUND"

METADATA_COLUMNS = [
    "zip_path",
    "filename",
    "classe",
    "cultura",
    "doenca",
    "leaf_id",
    "leaf_id_found",
    "leaf_match_status",
    "leaf_lookup_key",
    "leaf_suggestions_count",
]


def normalize_image_identifier(filename: str) -> str:
    """Normalize an image filename using the PlantVillage loader logic."""
    image_identifier = PurePosixPath(str(filename).replace("\\", "/")).name
    image_identifier = image_identifier.replace("_final_masked", "")

    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]

    image_identifier = image_identifier.split("copy")[0]

    suffix = PurePosixPath(image_identifier).suffix
    if suffix:
        image_identifier = image_identifier[: -len(suffix)]

    return image_identifier.strip().lower()


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
    missing_leaf_id: str = MISSING_LEAF_ID,
) -> dict[str, Any]:
    """Find the PlantVillage leaf_id for one image.

    Missing or ambiguous matches are explicit; they are not converted into
    fallback identifiers.
    """
    lookup_key = normalize_image_identifier(filename)
    suggestions = leaf_map.get(lookup_key)

    if suggestions is None:
        return {
            "leaf_id": missing_leaf_id,
            "leaf_id_found": False,
            "leaf_match_status": "not_found",
            "leaf_lookup_key": lookup_key,
            "leaf_suggestions_count": 0,
        }

    suggestions = _coerce_suggestions(suggestions)
    if len(suggestions) == 1:
        return {
            "leaf_id": suggestions[0],
            "leaf_id_found": True,
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
                    "leaf_match_status": "matched_by_class",
                    "leaf_lookup_key": lookup_key,
                    "leaf_suggestions_count": len(suggestions),
                }

        return {
            "leaf_id": missing_leaf_id,
            "leaf_id_found": False,
            "leaf_match_status": "ambiguous_no_class_match",
            "leaf_lookup_key": lookup_key,
            "leaf_suggestions_count": len(suggestions),
        }

    return {
        "leaf_id": missing_leaf_id,
        "leaf_id_found": False,
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
    leaf_found = _as_bool_series(metadata["leaf_id_found"])
    missing_leaf_count = int((~leaf_found).sum())
    missing_leaf_percent = (
        round((missing_leaf_count / total_images) * 100, 4) if total_images else 0.0
    )

    leaf_counts = metadata.loc[leaf_found].groupby("leaf_id").size()
    leaves_with_multiple_images = leaf_counts[leaf_counts > 1]

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
                "numero_leaf_id_unicos": int(metadata.loc[leaf_found, "leaf_id"].nunique()),
                "imagens_sem_leaf_id": missing_leaf_count,
                "percentual_sem_leaf_id": missing_leaf_percent,
                "imagens_em_folhas_com_multiplas_imagens": int(
                    leaves_with_multiple_images.sum()
                ),
                "folhas_com_multiplas_imagens": int(len(leaves_with_multiple_images)),
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

    return {
        "resumo": summary,
        "imagens_por_classe": class_counts,
        "status_leaf_id": leaf_status_counts,
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


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "sim"}).fillna(False)


__all__ = [
    "DEFAULT_IMAGE_EXTENSIONS",
    "METADATA_COLUMNS",
    "MISSING_LEAF_ID",
    "audit_metadata",
    "build_metadata_dataframe",
    "load_leaf_map",
    "normalize_image_identifier",
    "resolve_leaf_id",
    "save_metadata_csv",
    "split_class_name",
]
