import os
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import datasets
from .model import Block, NoiseScheduler


class Dataset(StrEnum):
    circle = "circle"
    dino = "dino"
    line = "line"
    moons = "moons"
    quickdraw = "quickdraw"


class BetaSchedule(StrEnum):
    linear = "linear"
    quadratic = "quadratic"


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


def _precompute_text_embeddings(
    text_model_name: str,
    device: torch.device,
    categories: Sequence[str],
    descriptions: dict[str, str],
) -> tuple[torch.Tensor, int]:
    """Precompute text embeddings for all category descriptions + null string.

    Returns (embeddings_tensor, text_dim) where embeddings_tensor has shape
    [num_categories + 1, text_dim] — last row is the null/empty embedding.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(text_model_name)
    desc_list = [descriptions[name] for name in categories]
    desc_list.append("")  # null embedding for CFG
    embeddings = encoder.encode(desc_list, convert_to_tensor=True)
    return embeddings.to(device), embeddings.shape[1]


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
    input_embedding: Annotated[InputEmbedding, typer.Option()] = InputEmbedding.sinusoidal,
    save_images_step: Annotated[int, typer.Option()] = 1,
    num_classes: Annotated[int, typer.Option(help="Number of classes for conditional generation (0=uncond)")] = 0,
    cfg_dropout: Annotated[float, typer.Option(help="Classifier-free guidance dropout probability")] = 0.1,
    guidance_scale: Annotated[float, typer.Option(help="CFG guidance scale for eval sampling")] = 3.0,
    text_conditioned: Annotated[bool, typer.Option(help="Use text embeddings instead of class indices")] = False,
    text_model: Annotated[str, typer.Option(help="Sentence-transformers model name")] = "all-MiniLM-L6-v2",
    quickdraw_num_categories: Annotated[int, typer.Option(help="Number of Quick, Draw! categories (0=all)")] = 0,
    wandb_enabled: Annotated[bool, typer.Option("--wandb", help="Log to Weights & Biases")] = False,
    wandb_project: Annotated[str, typer.Option(help="W&B project name")] = "tiny-diffusion",
    resume: Annotated[str | None, typer.Option(help="Path to checkpoint.pth to resume training from")] = None,
):
    ds = datasets.get_dataset(dataset.value, quickdraw_num_categories=quickdraw_num_categories)
    has_labels = len(ds[0]) > 1
    dataset_info = datasets.get_dataset_info(dataset.value, quickdraw_num_categories=quickdraw_num_categories)

    # Text conditioning setup
    text_embedding_dim = 0
    text_embeddings_table: torch.Tensor | None = None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if text_conditioned:
        if not has_labels or dataset_info is None:
            print(f"Warning: --text-conditioned but dataset '{dataset.value}' has no labels. Ignoring.")
            text_conditioned = False
        else:
            categories, descriptions = dataset_info
            print(f"Loading text encoder: {text_model}")
            text_embeddings_table, text_embedding_dim = _precompute_text_embeddings(
                text_model, device, categories, descriptions
            )
            print(f"Text embedding dim: {text_embedding_dim}")
            num_classes = 0  # disable class-index mode
    elif num_classes > 0 and not has_labels:
        print(f"Warning: --num-classes={num_classes} but dataset '{dataset.value}' has no labels. Ignoring.")
        num_classes = 0

    if wandb_enabled:
        import wandb

        wandb.init(
            project=wandb_project,
            name=experiment_name,
            dir="data",
            config={
                "dataset": dataset.value,
                "train_batch_size": train_batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "num_timesteps": num_timesteps,
                "beta_schedule": beta_schedule.value,
                "embedding_size": embedding_size,
                "hidden_size": hidden_size,
                "hidden_layers": hidden_layers,
                "time_embedding": time_embedding.value,
                "input_embedding": input_embedding.value,
                "num_classes": num_classes,
                "cfg_dropout": cfg_dropout,
                "guidance_scale": guidance_scale,
                "text_conditioned": text_conditioned,
            },
        )

    dataloader = DataLoader(ds, batch_size=train_batch_size, shuffle=True, drop_last=True)

    model = Block(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        embedding_size=embedding_size,
        time_embedding_type=time_embedding.value,
        embedding_type=input_embedding.value,
        num_classes=num_classes,
        text_embedding_dim=text_embedding_dim,
        cfg_dropout_prob=cfg_dropout,
    )

    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps, beta_schedule=beta_schedule.value)

    print(f"Using device: {device}")
    if text_conditioned:
        print(f"Text-conditioned: True (model={text_model}, dim={text_embedding_dim})")
    else:
        print(f"Class-conditional: {num_classes > 0} (num_classes={num_classes})")
    model = model.to(device)
    noise_scheduler = noise_scheduler.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # Precompute eval labels for visualization
    n_cond = len(dataset_info[0]) if text_conditioned and dataset_info is not None else max(num_classes, 1)
    eval_labels = torch.arange(n_cond, device=device).repeat(eval_batch_size // n_cond + 1)[:eval_batch_size]

    start_epoch = 0
    global_step = 0
    ema_loss: float | None = None
    frames: list[np.ndarray] = []
    losses: list[float] = []

    if resume is not None:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        losses = checkpoint["losses"]
        frames = [f.numpy() for f in checkpoint["frames"]]
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")
        if start_epoch >= num_epochs:
            print(f"Checkpoint already at epoch {start_epoch - 1}, but --num-epochs is {num_epochs}. Nothing to do.")
            return

    print("Training model...")
    for epoch in range(start_epoch, num_epochs):
        _ = model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for _step, batch in enumerate(dataloader):
            coords = batch[0].to(device)

            # Build conditioning
            class_label: torch.Tensor | None = None
            text_emb: torch.Tensor | None = None
            if text_conditioned and text_embeddings_table is not None:
                label_indices = batch[1].to(device)
                text_emb = text_embeddings_table[label_indices]
            elif num_classes > 0 and has_labels:
                class_label = batch[1].to(device)

            noise = torch.randn(coords.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (coords.shape[0],), device=device).long()

            noisy = noise_scheduler.add_noise(coords, noise, timesteps)
            noise_pred = model(noisy, timesteps, class_label=class_label, text_embedding=text_emb)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            loss_val = loss.detach().item()
            losses.append(loss_val)
            ema_loss = loss_val if ema_loss is None else 0.99 * ema_loss + 0.01 * loss_val
            if wandb_enabled:
                wandb.log({"loss": loss_val, "loss_ema": ema_loss}, step=global_step)
            progress_bar.set_postfix(loss=loss.detach().item(), step=global_step)
            global_step += 1
        progress_bar.close()

        if epoch % save_images_step == 0 or epoch == num_epochs - 1:
            _ = model.eval()
            sample = torch.randn(eval_batch_size, 2, device=device)

            timesteps = list(range(len(noise_scheduler)))[::-1]
            for _i, t in enumerate(tqdm(timesteps)):
                t_batch = torch.full((eval_batch_size,), t, dtype=torch.long, device=device)
                with torch.no_grad():
                    if text_conditioned and text_embeddings_table is not None:
                        eval_text = text_embeddings_table[eval_labels]
                        null_text = model.null_text_embedding.unsqueeze(0).expand(eval_batch_size, -1)
                        noise_cond = model(sample, t_batch, text_embedding=eval_text)
                        noise_uncond = model(sample, t_batch, text_embedding=null_text)
                        residual = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                    elif num_classes > 0:
                        null_labels = torch.full((eval_batch_size,), num_classes, dtype=torch.long, device=device)
                        noise_cond = model(sample, t_batch, class_label=eval_labels)
                        noise_uncond = model(sample, t_batch, class_label=null_labels)
                        residual = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                    else:
                        residual = model(sample, t_batch)
                sample = noise_scheduler.step(residual, t_batch[0], sample)
            frame_np = sample.cpu().numpy()
            frames.append(frame_np)

            # Save intermediate checkpoint and eval image immediately
            outdir = f"data/output/experiments/{experiment_name}"
            imgdir = f"{outdir}/images"
            os.makedirs(imgdir, exist_ok=True)
            _ = plt.figure(figsize=(10, 10))
            if num_classes > 0 or text_conditioned:
                labels_np = eval_labels.cpu().numpy()
                _ = plt.scatter(frame_np[:, 0], frame_np[:, 1], c=labels_np, cmap="tab20", s=5, alpha=0.7)
            else:
                _ = plt.scatter(frame_np[:, 0], frame_np[:, 1])
            plt.xlim(-6, 6)
            plt.ylim(-6, 6)
            plt.title(f"Epoch {epoch}")
            plt.savefig(f"{imgdir}/epoch_{epoch:04d}.png")
            plt.close()
            if wandb_enabled:
                wandb.log({"samples": wandb.Image(f"{imgdir}/epoch_{epoch:04d}.png")}, step=global_step)
            torch.save(model.state_dict(), f"{outdir}/model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "losses": losses,
                    "frames": [torch.from_numpy(f) for f in frames],
                },
                f"{outdir}/checkpoint.pth",
            )

    print("Saving final model...")
    outdir = f"data/output/experiments/{experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs - 1,
            "global_step": global_step,
            "losses": losses,
            "frames": [torch.from_numpy(f) for f in frames],
        },
        f"{outdir}/checkpoint.pth",
    )

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", np.stack(frames))

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    app()
