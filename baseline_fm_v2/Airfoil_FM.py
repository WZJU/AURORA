"""Conditional flow matching on one complete normalized geometry vector.

There are no section embeddings, VAE latents, pose channels or section losses.
Time runs from data (0) to Gaussian noise (1); generation integrates backwards.
"""

import math

import torch
from torch import Tensor, nn
import pytorch_lightning as pl


class GeometryFlow(pl.LightningModule):
    def __init__(self, geometry_dim: int, context_dim: int, hidden_dim: int,
                 depth: int, learning_rate: float, step_size: int, gamma: float) -> None:
        super().__init__()
        if geometry_dim != 125 or context_dim < 8 or (context_dim - 1) % 7:
            raise ValueError("Expected 125 geometry coordinates and AR/aerodynamic context with a mask.")
        if hidden_dim < 8 or depth < 1 or learning_rate <= 0 or step_size < 1 or not 0 < gamma <= 1:
            raise ValueError("Invalid network dimensions or optimizer configuration.")
        self.save_hyperparameters()
        self.geometry_dim = geometry_dim
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 32)))
        self.input_layer = nn.Linear(geometry_dim + context_dim + 2 + 64, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                          nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(depth)
        ])
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, geometry_dim))

    def forward(self, x: Tensor, time: Tensor, context: Tensor, available: Tensor) -> Tensor:
        phase = time[:, None] * self.frequencies[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        hidden = self.input_layer(torch.cat((x, context, available, embedding), dim=1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_layer(hidden)

    def flow_loss(self, x: Tensor, context: Tensor, time: Tensor, noise: Tensor) -> Tensor:
        """Average unconditional, AR-only, aerodynamic-only and joint conditions."""
        flags = x.new_tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).repeat_interleave(len(x), dim=0)
        repeated_context = context.repeat(4, 1)
        masked_context = torch.cat((repeated_context[:, :1] * flags[:, :1],
                                    repeated_context[:, 1:] * flags[:, 1:]), dim=1)
        interpolated = (1 - time[:, None]) * x + time[:, None] * noise
        predicted = self(interpolated.repeat(4, 1), time.repeat(4), masked_context, flags)
        return (predicted - (noise - x).repeat(4, 1)).square().mean()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, context = batch
        loss = self.flow_loss(x, context, torch.rand(len(x), device=x.device), torch.randn_like(x))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_idx}.")
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(x))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, context = batch
        generator = torch.Generator(device=x.device).manual_seed(62000 + batch_idx)
        time = torch.rand(len(x), generator=generator, device=x.device)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        loss = self.flow_loss(x, context, time, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss at batch {batch_idx}.")
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(x))

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.StepLR]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, context: Tensor, steps: int) -> Tensor:
        """Generate normalized whole-wing vectors conditioned on both AR and aerodynamics."""
        if steps < 1 or context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("Sampling requires positive steps and context [N, context_dim].")
        x = torch.randn((len(context), self.geometry_dim), device=context.device, dtype=context.dtype)
        available = context.new_ones((len(context), 2))
        for step in range(steps, 0, -1):
            time = context.new_full((len(context),), step / steps)
            x = x - self(x, time, context, available) / steps
        if not torch.isfinite(x).all():
            raise FloatingPointError("FM integration produced non-finite geometry.")
        return x
