import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm.auto import tqdm
from typing_extensions import Annotated

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
    output_dir: Annotated[str, typer.Option()] = "data/output/infer",
):
    model = MLP(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        emb_size=embedding_size,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)

    sample = torch.randn(num_samples, 2)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    for t in tqdm(timesteps, desc="Sampling"):
        t_batch = torch.from_numpy(np.repeat(t, num_samples)).long()
        with torch.no_grad():
            residual = model(sample, t_batch)
        sample = noise_scheduler.step(residual, t_batch[0], sample)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    points = sample.numpy()
    plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig(f"{output_dir}/samples.png")
    plt.close()

    np.save(f"{output_dir}/samples.npy", points)
    print(f"Saved {num_samples} samples to {output_dir}/")


if __name__ == "__main__":
    app()
