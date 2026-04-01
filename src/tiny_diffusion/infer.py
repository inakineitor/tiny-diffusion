import os
from enum import StrEnum
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from tqdm.auto import tqdm

from .model import Block, NoiseScheduler


class TimeEmbedding(StrEnum):
    sinusoidal = "sinusoidal"
    learnable = "learnable"
    linear = "linear"
    zero = "zero"


class InputEmbedding(StrEnum):
    sinusoidal = "sinusoidal"
    learnable = "learnable"
    linear = "linear"
    identity = "identity"

app = typer.Typer()


@app.command()
def main(
    model_path: Annotated[str, typer.Argument(help="Path to the saved model checkpoint (.pth)")],
    num_samples: Annotated[int, typer.Option()] = 1000,
    num_timesteps: Annotated[int, typer.Option()] = 50,
    embedding_size: Annotated[int, typer.Option()] = 128,
    hidden_size: Annotated[int, typer.Option()] = 128,
    hidden_layers: Annotated[int, typer.Option()] = 3,
    output_dir: Annotated[str | None, typer.Option()] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes (0=unconditional, must match training)")] = 0,
    class_label: Annotated[int | None, typer.Option(help="Class index to generate (None=all classes equally)")] = None,
    guidance_scale: Annotated[float, typer.Option(help="CFG guidance scale (higher=more class fidelity)")] = 3.0,
    text_embedding_dim: Annotated[int, typer.Option(help="Text embedding dim (must match training, 0=off)")] = 0,
    text_model: Annotated[str, typer.Option(help="Sentence-transformers model name")] = "all-MiniLM-L6-v2",
    prompt: Annotated[str | None, typer.Option(help="Text prompt for generation")] = None,
    time_embedding: Annotated[TimeEmbedding, typer.Option()] = TimeEmbedding.sinusoidal,
    input_embedding: Annotated[InputEmbedding, typer.Option()] = InputEmbedding.sinusoidal,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), "inference_outputs")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = Block(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        embedding_size=embedding_size,
        time_embedding_type=time_embedding.value,
        embedding_type=input_embedding.value,
        num_classes=num_classes,
        text_embedding_dim=text_embedding_dim,
    )
    _ = model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    _ = model.to(device)
    _ = model.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps).to(device)

    sample = torch.randn(num_samples, 2, device=device)

    # Prepare conditioning
    labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    null_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
    text_emb: torch.Tensor | None = None
    null_text: torch.Tensor | None = None

    if text_embedding_dim > 0:
        from sentence_transformers import SentenceTransformer

        assert prompt is not None, "Must provide --prompt when using text conditioning"
        print(f"Encoding prompt: '{prompt}'")
        encoder = SentenceTransformer(text_model)
        prompt_embedding = encoder.encode([prompt], convert_to_tensor=True).to(device)
        text_emb = prompt_embedding.expand(num_samples, -1)
        null_text = model.null_text_embedding.unsqueeze(0).expand(num_samples, -1)
    elif num_classes > 0:
        if class_label is not None:
            labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        else:
            labels = torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]

    timesteps = list(range(len(noise_scheduler)))[::-1]
    for t in tqdm(timesteps, desc="Sampling"):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            if text_embedding_dim > 0 and text_emb is not None and null_text is not None:
                noise_cond = model(sample, t_batch, text_embedding=text_emb)
                noise_uncond = model(sample, t_batch, text_embedding=null_text)
                residual = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            elif num_classes > 0:
                noise_cond = model(sample, t_batch, class_label=labels)
                noise_uncond = model(sample, t_batch, class_label=null_labels)
                residual = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                residual = model(sample, t_batch)
        sample = noise_scheduler.step(residual, t_batch[0], sample)

    os.makedirs(output_dir, exist_ok=True)

    _ = plt.figure(figsize=(10, 10))
    points = sample.cpu().numpy()
    _ = plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.7)
    if prompt is not None:
        plt.title(prompt)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig(f"{output_dir}/samples.png")
    plt.close()

    np.save(f"{output_dir}/samples.npy", points)
    print(f"Saved {num_samples} samples to {output_dir}/")


if __name__ == "__main__":
    app()
