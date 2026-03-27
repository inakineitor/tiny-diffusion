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
    num_classes: int
    cfg_dropout_prob: float

    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        embedding_size: int = 128,
        time_embedding_type: EmbeddingType = "sinusoidal",
        embedding_type: EmbeddingType = "sinusoidal",
        num_classes: int = 0,
        cfg_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob

        self.time_mlp = make_positional_embedding(time_embedding_type, embedding_size)
        self.input_mlp_x1 = make_positional_embedding(embedding_type, embedding_size, scale=25.0)
        self.input_mlp_x2 = make_positional_embedding(embedding_type, embedding_size, scale=25.0)

        concat_size = self.time_mlp.output_dim + self.input_mlp_x1.output_dim + self.input_mlp_x2.output_dim

        if num_classes > 0:
            # +1 for the unconditional/null class used in CFG
            self.class_embedding = torch.nn.Embedding(num_classes + 1, embedding_size)
            concat_size += embedding_size

        self.joint_mlp = torch.nn.Sequential(
            torch.nn.Linear(concat_size, hidden_size),
            torch.nn.GELU(),
            *(MLP(hidden_size) for _ in range(hidden_layers)),
            torch.nn.Linear(hidden_size, 2),
        )

    @override
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label: torch.Tensor | None = None):
        x1_embedding = self.input_mlp_x1(x[:, 0])
        x2_embedding = self.input_mlp_x2(x[:, 1])
        time_embedding = self.time_mlp(t)
        embeddings = [x1_embedding, x2_embedding, time_embedding]

        if self.num_classes > 0:
            assert class_label is not None
            if self.training:
                # CFG dropout: replace some labels with the null class index
                mask = torch.rand(class_label.shape[0], device=class_label.device) < self.cfg_dropout_prob
                class_label = class_label.clone()
                class_label[mask] = self.num_classes  # null class
            embeddings.append(self.class_embedding(class_label))

        x = torch.cat(embeddings, dim=-1)
        x = cast(torch.Tensor, self.joint_mlp(x))
        return x
