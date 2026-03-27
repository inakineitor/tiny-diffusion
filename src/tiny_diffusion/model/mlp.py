from typing import cast, override

import torch

from .positional_embeddings import (
    EmbeddingType,
    PositionalEmbeddingLayer,
    make_positional_embedding,
)


class MLP(torch.nn.Module):
    ff: torch.nn.Linear
    act: torch.nn.GELU

    def __init__(self, size: int):
        super().__init__()

        self.ff = torch.nn.Linear(size, size)
        self.act = torch.nn.GELU()

    @override
    def forward(self, x: torch.Tensor):
        mlp_output = cast(torch.Tensor, self.act(self.ff(x)))
        return x + mlp_output

    @override
    def __call__(self, x: torch.Tensor):
        return cast(torch.Tensor, super().__call__(x))


class Block(torch.nn.Module):
    time_mlp: PositionalEmbeddingLayer
    input_mlp_x1: PositionalEmbeddingLayer
    input_mlp_x2: PositionalEmbeddingLayer
    joint_mlp: torch.nn.Sequential

    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        embedding_size: int = 128,
        time_embedding_type: EmbeddingType = "sinusoidal",
        embedding_type: EmbeddingType = "sinusoidal",
    ):
        super().__init__()

        self.time_mlp = make_positional_embedding(time_embedding_type, embedding_size)
        self.input_mlp_x1 = make_positional_embedding(embedding_type, embedding_size, scale=25.0)
        self.input_mlp_x2 = make_positional_embedding(embedding_type, embedding_size, scale=25.0)

        concat_size = self.time_mlp.output_dim + self.input_mlp_x1.output_dim + self.input_mlp_x2.output_dim
        self.joint_mlp = torch.nn.Sequential(
            torch.nn.Linear(concat_size, hidden_size),
            torch.nn.GELU(),
            *(MLP(hidden_size) for _ in range(hidden_layers)),
            torch.nn.Linear(hidden_size, 2),
        )

    @override
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x1_embedding = self.input_mlp_x1(x[:, 0])
        x2_embedding = self.input_mlp_x2(x[:, 1])
        time_embedding = self.time_mlp(t)
        x = torch.cat((x1_embedding, x2_embedding, time_embedding), dim=-1)
        x = cast(torch.Tensor, self.joint_mlp(x))
        return x
