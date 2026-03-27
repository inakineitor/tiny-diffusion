from typing import Any, cast

import numpy.typing as npt
import torch
from sklearn.datasets import make_moons  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
from torch.utils.data import TensorDataset


def moons_dataset(n: int = 8000):
    X_numpy, _ = cast(tuple[npt.NDArray[Any], Any], make_moons(n_samples=n, random_state=42, noise=0.03))  # pyright: ignore[reportExplicitAny, reportAny]
    X = torch.from_numpy(X_numpy).float()  # pyright: ignore[reportUnknownMemberType]
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(X)
