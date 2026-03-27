from pathlib import Path
from typing import Literal

import polars as pl
import torch
from torch.utils.data import TensorDataset

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "input" / "dinosaur"

type DatasaurusShape = Literal[
    "away",
    "bullseye",
    "circle",
    "dino",
    "dots",
    "h_lines",
    "high_lines",
    "slant_down",
    "slant_up",
    "star",
    "v_lines",
    "wide_lines",
    "x_shape",
]


def dino_dataset(n: int = 8000, shapes: list[DatasaurusShape] | None = None):
    df = pl.read_csv(_DATA_DIR / "datasaurus-dozen.tsv", separator="\t")
    if shapes is not None:
        df = df.filter(pl.col("dataset").is_in(shapes))

    gen = torch.Generator().manual_seed(42)
    ix = torch.randint(0, len(df), (n,), generator=gen)
    x_vals = torch.tensor(df["x"].to_list(), dtype=torch.float32)[ix]
    y_vals = torch.tensor(df["y"].to_list(), dtype=torch.float32)[ix]
    x_vals = x_vals + torch.randn(n, generator=gen) * 0.15
    y_vals = y_vals + torch.randn(n, generator=gen) * 0.15
    x_vals = (x_vals / 54 - 1) * 4
    y_vals = (y_vals / 48 - 1) * 4
    X = torch.stack((x_vals, y_vals), dim=1)
    return TensorDataset(X)
