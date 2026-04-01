from collections.abc import Sequence
from typing import Literal

from .circle import circle_dataset
from .dino import (
    DATASAURUS_SHAPES,
    NUM_DATASAURUS_SHAPES,
    SHAPE_DESCRIPTIONS,
    SHAPE_TO_INDEX,
    DatasaurusShape,
    dino_dataset,
)
from .line import line_dataset
from .moons import moons_dataset
from .quickdraw import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_TO_INDEX,
    NUM_QUICKDRAW_CATEGORIES,
    QUICKDRAW_CATEGORIES,
    quickdraw_dataset,
)

__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "CATEGORY_TO_INDEX",
    "DATASAURUS_SHAPES",
    "NUM_DATASAURUS_SHAPES",
    "NUM_QUICKDRAW_CATEGORIES",
    "QUICKDRAW_CATEGORIES",
    "SHAPE_DESCRIPTIONS",
    "SHAPE_TO_INDEX",
    "DatasaurusShape",
    "circle_dataset",
    "dino_dataset",
    "get_dataset",
    "get_dataset_info",
    "line_dataset",
    "moons_dataset",
    "quickdraw_dataset",
]

type DatasetName = Literal["moons", "dino", "line", "circle", "quickdraw"]


def get_dataset(name: DatasetName, n: int = 8000, quickdraw_num_categories: int = 0):
    match name:
        case "moons":
            return moons_dataset(n)
        case "dino":
            return dino_dataset(n)
        case "line":
            return line_dataset(n)
        case "circle":
            return circle_dataset(n)
        case "quickdraw":
            cats = QUICKDRAW_CATEGORIES[:quickdraw_num_categories] if quickdraw_num_categories > 0 else None
            return quickdraw_dataset(categories=cats)


def get_dataset_info(
    name: DatasetName, quickdraw_num_categories: int = 0
) -> tuple[Sequence[str], dict[str, str]] | None:
    """Return (category_names, descriptions) for labeled datasets, None for unlabeled ones."""
    match name:
        case "dino":
            return (DATASAURUS_SHAPES, SHAPE_DESCRIPTIONS)
        case "quickdraw":
            cats = QUICKDRAW_CATEGORIES[:quickdraw_num_categories] if quickdraw_num_categories > 0 else QUICKDRAW_CATEGORIES
            descs = {c: CATEGORY_DESCRIPTIONS[c] for c in cats}
            return (cats, descs)
        case _:
            return None
