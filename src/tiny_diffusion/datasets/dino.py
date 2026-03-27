from pathlib import Path

import polars as pl
import torch
from torch.utils.data import TensorDataset

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "input" / "dinosaur"


def dino_dataset(n: int = 8000):
    df = pl.read_csv(_DATA_DIR / "datasaurus-dozen.tsv", separator="\t")
    df = df.filter(pl.col("dataset") == "dino")

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
