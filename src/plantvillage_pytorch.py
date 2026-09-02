"""PyTorch data pipeline for PlantVillage metadata splits."""

from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "validation", "test")
DEFAULT_IMAGE_ROOT = Path("/content/plantvillage_color")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def extract_raw_color_from_zip(
    zip_path: str | Path,
    output_dir: str | Path = DEFAULT_IMAGE_ROOT,
    overwrite: bool = False,
) -> dict[str, int | str]:
    """Extract only raw/color images to one local directory."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue

            try:
                relative_parts = _raw_color_relative_parts(member.filename)
            except ValueError:
                continue

            if PurePosixPath(relative_parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination = output_dir.joinpath(*relative_parts)
            _validate_inside_root(destination, output_root)

            if (
                not overwrite
                and destination.exists()
                and destination.stat().st_size == member.file_size
            ):
                skipped += 1
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1

    return {
        "output_dir": str(output_dir),
        "extracted": extracted,
        "skipped": skipped,
        "total": extracted + skipped,
    }


class PlantVillageDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path | pd.DataFrame,
        image_root: str | Path,
        split: str,
        class_to_idx: dict[str, int] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Split invalido: {split}. Use um de {SPLITS}.")

        metadata = load_metadata(metadata_csv)
        _validate_metadata_columns(metadata)

        self.split = split
        self.image_root = Path(image_root)
        self.class_to_idx = build_class_to_idx(metadata) if class_to_idx is None else class_to_idx
        self.idx_to_class = {index: classe for classe, index in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[index] for index in sorted(self.idx_to_class)]
        self.transform = build_image_transforms(split) if transform is None else transform

        split_metadata = metadata.loc[metadata["split"].eq(split)].copy().reset_index(drop=True)
        if split_metadata.empty:
            raise ValueError(f"Nenhuma imagem encontrada para o split {split}.")

        unknown_classes = sorted(set(split_metadata["classe"]) - set(self.class_to_idx))
        if unknown_classes:
            raise ValueError(f"Classes sem indice: {unknown_classes}")

        self.metadata = split_metadata
        self.labels = (
            self.metadata["classe"].map(self.class_to_idx).astype("int64").to_numpy()
        )
        self.image_paths = [
            self.image_root.joinpath(*_raw_color_relative_parts(zip_path))
            for zip_path in self.metadata["zip_path"]
        ]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        if not image_path.exists():
            raise FileNotFoundError(
                f"Imagem nao encontrada: {image_path}. "
                "Execute extract_raw_color_from_zip antes de criar as epocas."
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return image_tensor, int(self.labels[index])


def create_dataloaders(
    metadata_csv: str | Path | pd.DataFrame,
    image_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Create train, validation and test DataLoaders from metadata splits."""
    metadata = load_metadata(metadata_csv)
    class_to_idx = build_class_to_idx(metadata)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {}
    for split in SPLITS:
        dataset = PlantVillageDataset(
            metadata_csv=metadata,
            image_root=image_root,
            split=split,
            class_to_idx=class_to_idx,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    return dataloaders


def build_image_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if split in {"validation", "test"}:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    raise ValueError(f"Split invalido: {split}. Use um de {SPLITS}.")


def build_class_to_idx(
    metadata_csv: str | Path | pd.DataFrame,
    expected_classes: int = 38,
) -> dict[str, int]:
    metadata = load_metadata(metadata_csv)
    if "classe" not in metadata:
        raise ValueError("Coluna obrigatoria ausente: classe")

    classes = sorted(metadata["classe"].astype(str).unique())
    if len(classes) != expected_classes:
        raise ValueError(f"Esperadas {expected_classes} classes, encontradas {len(classes)}.")

    return {classe: index for index, classe in enumerate(classes)}


def load_metadata(metadata_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(metadata_csv, pd.DataFrame):
        return metadata_csv.copy()

    return pd.read_csv(metadata_csv)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _validate_metadata_columns(metadata: pd.DataFrame) -> None:
    required_columns = ["zip_path", "classe", "split"]
    missing = [column for column in required_columns if column not in metadata]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(missing)}")


def _raw_color_relative_parts(zip_path: str | Path) -> list[str]:
    normalized = str(zip_path).replace("\\", "/").lstrip("/")
    parts = [part for part in normalized.split("/") if part]
    lower_parts = [part.lower() for part in parts]

    for index in range(len(parts) - 2):
        if lower_parts[index] == "raw" and lower_parts[index + 1] == "color":
            relative_parts = parts[index + 2 :]
            if len(relative_parts) < 2:
                break
            if any(part in {"..", "."} for part in relative_parts):
                raise ValueError(f"Caminho inseguro no ZIP: {zip_path}")
            return relative_parts

    raise ValueError(f"Caminho fora de raw/color: {zip_path}")


def _validate_inside_root(path: Path, root: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Caminho inseguro para extracao: {path}") from exc


__all__ = [
    "DEFAULT_IMAGE_ROOT",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SPLITS",
    "PlantVillageDataset",
    "build_class_to_idx",
    "build_image_transforms",
    "create_dataloaders",
    "extract_raw_color_from_zip",
    "load_metadata",
]
