import os
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import typer
from typing_extensions import Annotated

from . import datasets
from .model import MLP, NoiseScheduler


class Dataset(str, Enum):
    circle = "circle"
    dino = "dino"
    line = "line"
    moons = "moons"


class BetaSchedule(str, Enum):
    linear = "linear"
    quadratic = "quadratic"


class TimeEmbedding(str, Enum):
    sinusoidal = "sinusoidal"
    learnable = "learnable"
    linear = "linear"
    zero = "zero"


class InputEmbedding(str, Enum):
    sinusoidal = "sinusoidal"
    learnable = "learnable"
    linear = "linear"
    identity = "identity"


app = typer.Typer()


@app.command()
def main(
    experiment_name: Annotated[str, typer.Option()] = "base",
    dataset: Annotated[Dataset, typer.Option()] = Dataset.dino,
    train_batch_size: Annotated[int, typer.Option()] = 32,
    eval_batch_size: Annotated[int, typer.Option()] = 1000,
    num_epochs: Annotated[int, typer.Option()] = 200,
    learning_rate: Annotated[float, typer.Option()] = 1e-3,
    num_timesteps: Annotated[int, typer.Option()] = 50,
    beta_schedule: Annotated[BetaSchedule, typer.Option()] = BetaSchedule.linear,
    embedding_size: Annotated[int, typer.Option()] = 128,
    hidden_size: Annotated[int, typer.Option()] = 128,
    hidden_layers: Annotated[int, typer.Option()] = 3,
    time_embedding: Annotated[TimeEmbedding, typer.Option()] = TimeEmbedding.sinusoidal,
    input_embedding: Annotated[
        InputEmbedding, typer.Option()
    ] = InputEmbedding.sinusoidal,
    save_images_step: Annotated[int, typer.Option()] = 1,
):
    ds = datasets.get_dataset(dataset.value)
    dataloader = DataLoader(
        ds, batch_size=train_batch_size, shuffle=True, drop_last=True
    )

    model = MLP(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        emb_size=embedding_size,
        time_emb=time_embedding.value,
        input_emb=input_embedding.value,
    )

    noise_scheduler = NoiseScheduler(
        num_timesteps=num_timesteps, beta_schedule=beta_schedule.value
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0]
            noise = torch.randn(batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % save_images_step == 0 or epoch == num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(eval_batch_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, eval_batch_size)).long()
                with torch.no_grad():
                    residual = model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.numpy())

    print("Saving model...")
    outdir = f"data/output/{experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)


if __name__ == "__main__":
    app()
