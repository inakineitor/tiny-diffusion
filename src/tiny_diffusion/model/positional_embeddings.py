"""Different methods for positional embeddings.

These are not essential for understanding DDPMs, but are relevant for the ablation study.
"""

from typing import Literal, cast, override

import torch


class PositionalEmbeddingLayer(torch.nn.Module):
    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @override
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


class SinusoidalEmbedding(PositionalEmbeddingLayer):
    num_embedding_dims: int
    scale: float

    def __init__(self, num_embedding_dims: int, scale: float = 1.0):
        super().__init__()
        self.num_embedding_dims = num_embedding_dims
        self.scale = scale

    @property
    @override
    def output_dim(self) -> int:
        return self.num_embedding_dims

    @override
    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.num_embedding_dims // 2
        emb = torch.log(torch.tensor([10000.0], device=x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size, device=x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class LinearEmbedding(PositionalEmbeddingLayer):
    num_embedding_dims: int
    scale: float

    def __init__(self, num_embedding_dims: int, scale: float = 1.0):
        super().__init__()
        self.num_embedding_dims = num_embedding_dims
        self.scale = scale

    @property
    @override
    def output_dim(self) -> int:
        return 1

    @override
    def forward(self, x: torch.Tensor):
        x = x / self.num_embedding_dims * self.scale
        return x.unsqueeze(-1)


class LearnableEmbedding(PositionalEmbeddingLayer):
    num_embedding_dims: int
    linear: torch.nn.Linear

    def __init__(self, num_embedding_dims: int):
        super().__init__()
        self.num_embedding_dims = num_embedding_dims
        self.linear = torch.nn.Linear(1, num_embedding_dims)

    @property
    @override
    def output_dim(self) -> int:
        return self.num_embedding_dims

    @override
    def forward(self, x: torch.Tensor):
        return cast(
            torch.Tensor,
            self.linear(x.unsqueeze(-1).float() / self.num_embedding_dims),
        )


class IdentityEmbedding(PositionalEmbeddingLayer):
    def __init__(self):
        super().__init__()

    @property
    @override
    def output_dim(self) -> int:
        return 1

    @override
    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)


class ZeroEmbedding(PositionalEmbeddingLayer):
    def __init__(self):
        super().__init__()

    @property
    @override
    def output_dim(self) -> int:
        return 1

    @override
    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0


type EmbeddingType = Literal["sinusoidal", "linear", "learnable", "zero", "identity"]


def make_positional_embedding(
    embedding_type: EmbeddingType,
    num_embedding_dims: int = 0,
    scale: float = 0,
):
    match embedding_type:
        case "sinusoidal":
            return SinusoidalEmbedding(num_embedding_dims, scale)
        case "linear":
            return LinearEmbedding(num_embedding_dims, scale)
        case "learnable":
            return LearnableEmbedding(num_embedding_dims)
        case "zero":
            return ZeroEmbedding()
        case "identity":
            return IdentityEmbedding()
