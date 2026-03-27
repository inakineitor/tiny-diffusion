import os
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from tqdm.auto import tqdm

from .model import Block, NoiseScheduler

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
        num_classes=num_classes,
    )
    _ = model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    _ = model.to(device)
    _ = model.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps).to(device)

    sample = torch.randn(num_samples, 2, device=device)

    # Prepare class labels for conditional generation
    labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    null_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
    if num_classes > 0:
        if class_label is not None:
            labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        else:
            labels = torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]

    timesteps = list(range(len(noise_scheduler)))[::-1]
    for t in tqdm(timesteps, desc="Sampling"):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            if num_classes > 0:
                noise_cond = model(sample, t_batch, labels)
                noise_uncond = model(sample, t_batch, null_labels)
                residual = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                residual = model(sample, t_batch)
        sample = noise_scheduler.step(residual, t_batch[0], sample)

    os.makedirs(output_dir, exist_ok=True)

    _ = plt.figure(figsize=(10, 10))
    points = sample.cpu().numpy()
    if num_classes > 0:
        labels_np = labels.cpu().numpy()
        _ = plt.scatter(points[:, 0], points[:, 1], c=labels_np, cmap="tab20", s=5, alpha=0.7)
    else:
        _ = plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig(f"{output_dir}/samples.png")
    plt.close()

    np.save(f"{output_dir}/samples.npy", points)
    print(f"Saved {num_samples} samples to {output_dir}/")


if __name__ == "__main__":
    app()
