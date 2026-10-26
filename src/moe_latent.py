"""Small image-to-latent mixture-of-experts policy for the Jetbot task."""

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class MoELatentPolicy(nn.Module):
    """Learn a compact image code, then mix specialist action predictors.

    Input is a float RGB tensor in [0, 1], shaped N x 3 x 64 x 64. The learned
    bottleneck is 16-D. Each expert predicts all actions; a learned gate chooses
    how much each expert contributes for the current latent state.
    """

    def __init__(self, latent_dim=16, num_experts=4, num_classes=4, dropout=0.10, spatial_pool=1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_experts = int(num_experts)
        self.num_classes = int(num_classes)
        self.spatial_pool = int(spatial_pool)
        if not 1 <= self.spatial_pool <= 4:
            raise ValueError("spatial_pool must be between 1 and 4")
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48, stride=2),
            nn.AdaptiveAvgPool2d((self.spatial_pool, self.spatial_pool)),
            nn.Flatten(),
        )
        self.to_latent = nn.Sequential(
            nn.Linear(48 * self.spatial_pool * self.spatial_pool, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.latent_dim),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.SiLU(),
            nn.Linear(16, self.num_experts),
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.latent_dim, 24),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(24, self.num_classes),
            )
            for _ in range(self.num_experts)
        )

    def forward(self, images, return_aux=False):
        latent = self.to_latent(self.image_encoder(images))
        gate_weights = torch.softmax(self.gate(latent), dim=-1)
        expert_probs = torch.stack(
            [torch.softmax(expert(latent), dim=-1) for expert in self.experts], dim=1
        )
        mixture_probs = torch.sum(gate_weights.unsqueeze(-1) * expert_probs, dim=1)
        log_probs = torch.log(mixture_probs.clamp_min(1e-8))
        if return_aux:
            return log_probs, gate_weights, latent
        return log_probs


def gate_balance_loss(gate_weights):
    """Small load-balancing penalty to prevent the gate abandoning experts."""
    mean_usage = gate_weights.mean(dim=0)
    target_usage = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return ((mean_usage - target_usage) ** 2).sum() * mean_usage.numel()
