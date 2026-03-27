import os
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from tqdm.auto import tqdm

from .model import MLP, NoiseScheduler

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
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), "inference_outputs")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLP(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        embedding_size=embedding_size,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps).to(device)

    sample = torch.randn(num_samples, 2, device=device)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    for t in tqdm(timesteps, desc="Sampling"):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            residual = model(sample, t_batch)
        sample = noise_scheduler.step(residual, t_batch[0], sample)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    points = sample.cpu().numpy()
    plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig(f"{output_dir}/samples.png")
    plt.close()

    np.save(f"{output_dir}/samples.npy", points)
    print(f"Saved {num_samples} samples to {output_dir}/")


if __name__ == "__main__":
    app()
