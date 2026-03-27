from typing import Literal

from .circle import circle_dataset
from .dino import DATASAURUS_SHAPES, NUM_DATASAURUS_SHAPES, SHAPE_TO_INDEX, DatasaurusShape, dino_dataset
from .line import line_dataset
from .moons import moons_dataset

__all__ = [
    "DATASAURUS_SHAPES",
    "NUM_DATASAURUS_SHAPES",
    "SHAPE_TO_INDEX",
    "DatasaurusShape",
    "circle_dataset",
    "dino_dataset",
    "get_dataset",
    "line_dataset",
    "moons_dataset",
]

type DatasetName = Literal["moons", "dino", "line", "circle"]


def get_dataset(name: DatasetName, n: int = 8000):
    match name:
        case "moons":
            return moons_dataset(n)
        case "dino":
            return dino_dataset(n)
        case "line":
            return line_dataset(n)
        case "circle":
            return circle_dataset(n)
