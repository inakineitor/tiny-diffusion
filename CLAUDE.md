# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minimal PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for 2D datasets. Designed for educational clarity and hyperparameter ablation studies.

## Commands

```bash
# Install dependencies
uv sync

# Train a model
uv run model-train --experiment-name "my_exp" --dataset dino --num-epochs 200

# Resume training from a checkpoint
uv run model-train --experiment-name "my_exp" --dataset dino --num-epochs 400 --resume data/output/experiments/my_exp/checkpoint.pth

# Run inference on a trained model
uv run model-infer data/output/experiments/my_exp/model.pth --num-samples 1000

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Type check
uv run basedpyright src/
```

## Architecture

**Training pipeline:** Dataset → DataLoader → add noise at random timestep → MLP predicts noise → MSE loss → gradient clipping → backprop.

**Inference pipeline:** Start from Gaussian noise → iteratively denoise using trained model and noise scheduler → generated 2D samples.

**MLP model** (`model/mlp.py`): Three parallel embedding branches (timestep, x-coord, y-coord) → concatenate → FC layer → N residual blocks (Linear→GELU→residual) → 2D noise prediction output. No attention — pure MLP.

**Noise scheduler** (`model/noise_scheduler.py`): Manages forward process (`add_noise`) and reverse process (`step`). Supports linear and quadratic beta schedules.

**Positional embeddings** (`model/positional_embeddings.py`): Five strategies — sinusoidal, learnable, linear, identity, zero — selectable via CLI enums for ablation studies.

**Datasets** (`datasets/`): Four 2D distributions (dino, moons, circle, line), all returning `TensorDataset` with shape (N, 2).

## Code Style

- Line length: 120 characters
- Ruff rules: E, W, F, I, UP, B, A, SIM, C4, PIE, RUF, PERF, NPY, FURB
- Python ≥ 3.14
- CLI via Typer with entry points `model-train` and `model-infer`

## Key Conventions

- Training outputs go to `data/output/experiments/{experiment_name}/` containing `model.pth`, `checkpoint.pth`, `loss.npy`, `frames.npy`, and `images/`
- Dino dataset reads from `data/input/dinosaur/datasaurus-dozen.tsv`
- Device auto-detection: CUDA → MPS → CPU
- All datasets use seeded random generators for reproducibility
