import torch
from torch.utils.data import TensorDataset


def circle_dataset(n: int = 8000):
    gen = torch.Generator().manual_seed(42)
    x = (torch.rand(n, generator=gen) - 0.5) / 2
    x = (x * 10).round() / 10 * 2
    y = (torch.rand(n, generator=gen) - 0.5) / 2
    y = (y * 10).round() / 10 * 2
    norm = torch.sqrt(x**2 + y**2) + 1e-10
    x = x / norm
    y = y / norm
    theta = 2 * torch.pi * torch.rand(n, generator=gen)
    r = torch.rand(n, generator=gen) * 0.03
    x = x + r * torch.cos(theta)
    y = y + r * torch.sin(theta)
    X = torch.stack((x, y), dim=1) * 3
    return TensorDataset(X)
