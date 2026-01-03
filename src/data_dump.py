"""
Flexible data-dump indexing utilities.

This module is intentionally lightweight: it indexes images from arbitrary
directory structures and provides simple, configurable dataset/label inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DataDumpItem:
    image_path: str
    dataset: str
    class_label: Optional[str]
    rel_path: str
    metadata: Dict[str, Any]


def _iter_image_files(root: Path, follow_symlinks: bool) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root, followlinks=follow_symlinks):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def _infer_from_relative_parts(
    rel_parts: Tuple[str, ...],
    root_name: str,
    *,
    dataset_level: int,
    label_level: Optional[int],
    label_default: Optional[str],
) -> Tuple[str, Optional[str]]:
    dataset: str
    if dataset_level == -1 or len(rel_parts) == 0:
        dataset = root_name
    elif dataset_level < len(rel_parts):
        dataset = rel_parts[dataset_level]
    else:
        dataset = root_name

    label: Optional[str] = None
    if label_level is not None:
        if label_level == -1:
            label = None
        elif label_level < len(rel_parts):
            label = rel_parts[label_level]
        else:
            label = None

        if label is None and label_default is not None:
            label = label_default

    return dataset, label


def index_data_dump(
    roots: Sequence[str],
    *,
    dataset_level: int = 0,
    label_level: Optional[int] = 1,
    label_default: Optional[str] = "unlabeled",
    follow_symlinks: bool = False,
    max_items: Optional[int] = None,
) -> List[DataDumpItem]:
    """
    Index image files under one or more roots.

    Dataset/label inference is based on relative path segments:
      - dataset_level=0 => first folder under the root
      - label_level=1   => second folder under the root
      - use -1 for dataset_level to use root basename
      - use label_level=None to disable labels entirely
    """
    items: List[DataDumpItem] = []

    for root_str in roots:
        root = Path(root_str).expanduser().resolve()
        if not root.exists():
            continue

        for image_path in _iter_image_files(root, follow_symlinks=follow_symlinks):
            rel = image_path.relative_to(root)
            rel_parts = rel.parts
            dataset, label = _infer_from_relative_parts(
                rel_parts,
                root.name,
                dataset_level=dataset_level,
                label_level=label_level,
                label_default=label_default,
            )

            items.append(
                DataDumpItem(
                    image_path=str(image_path),
                    dataset=str(dataset),
                    class_label=label,
                    rel_path=str(rel),
                    metadata={"root": str(root)},
                )
            )

            if max_items is not None and len(items) >= max_items:
                return items

    # Stable ordering helps caching and reproducibility.
    items.sort(key=lambda x: (x.dataset, x.class_label or "", x.rel_path))
    return items

