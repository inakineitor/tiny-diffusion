import torch
from torch.utils.data import TensorDataset


def line_dataset(n: int = 8000):
    gen = torch.Generator().manual_seed(42)
    x = torch.rand(n, generator=gen) - 0.5
    y = torch.rand(n, generator=gen) * 2 - 1
    X = torch.stack((x, y), dim=1) * 4
    return TensorDataset(X)
