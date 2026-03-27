from typing import Literal, cast

import torch


class NoiseScheduler(torch.nn.Module):
    num_timesteps: int

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "quadratic"] = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        match beta_schedule:
            case "linear":
                betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
            case "quadratic":
                betas = (
                    torch.linspace(
                        cast(float, beta_start**0.5),
                        cast(float, beta_end**0.5),
                        num_timesteps,
                        dtype=torch.float32,
                    )
                    ** 2
                )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # required for self.add_noise
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # required for reconstruct_x0 (derived from above instead of recomputing)
        sqrt_inv_alphas_cumprod = 1.0 / sqrt_alphas_cumprod
        sqrt_inv_alphas_cumprod_minus_one = sqrt_one_minus_alphas_cumprod / sqrt_alphas_cumprod

        # required for q_posterior
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_inv_alphas_cumprod", sqrt_inv_alphas_cumprod)
        self.register_buffer("sqrt_inv_alphas_cumprod_minus_one", sqrt_inv_alphas_cumprod_minus_one)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def reconstruct_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        if t == 0:
            return torch.tensor(0.0)

        variance = self.betas[t] * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
        return variance.clamp(min=1e-20)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        if t > 0:
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + self.get_variance(t).sqrt() * noise

        return pred_prev_sample

    def add_noise(self, x_start: torch.Tensor, x_noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self) -> int:
        return self.num_timesteps
