"""Camera-only latent MoE policy with continuous outputs.

Two RGB streams (a fixed scene camera and a wrist/onboard camera), each with
the current and previous frame stacked, are compressed into one small latent
vector (16-D by default). Proprioception and an optional task command join the
latent; a learned gate mixes a few expert heads. Each head predicts a short
chunk of future actions: continuous deltas plus binary (gripper-style) logits.
An auxiliary head predicts the teacher's task phase from the latent alone,
which also gives a readable "what the model thinks it is doing" signal.

Must stay compatible with torch 2.7 (Isaac Sim's bundled build) for inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SpatialSoftmax(nn.Module):
    """Expected (x, y) image coordinate of each feature channel."""

    def forward(self, x):
        n, c, h, w = x.shape
        flat = x.reshape(n, c, h * w)
        probs = torch.softmax(flat, dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid_y = ys.view(h, 1).expand(h, w).reshape(1, 1, h * w)
        grid_x = xs.view(1, w).expand(h, w).reshape(1, 1, h * w)
        ex = (probs * grid_x).sum(-1)
        ey = (probs * grid_y).sum(-1)
        return torch.cat([ex, ey], dim=1)


class StreamEncoder(nn.Module):
    def __init__(self, in_frames=2, width=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3 * in_frames, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            DSBlock(32, 48, stride=2),
            DSBlock(48, width, stride=1),
            DSBlock(width, width, stride=1),
        )
        self.keypoints = SpatialSoftmax()
        self.out_dim = width * 3

    def forward(self, x):
        f = self.trunk(x)
        return torch.cat([self.keypoints(f), f.mean(dim=(2, 3))], dim=1)


class LatentMoEPolicy(nn.Module):
    def __init__(self, act_dim=7, bin_dim=1, proprio_dim=8, command_dim=0, latent_dim=16,
                 num_experts=4, chunk=4, num_phases=10, frames=2, dropout=0.05):
        super().__init__()
        self.config = dict(act_dim=act_dim, bin_dim=bin_dim, proprio_dim=proprio_dim, command_dim=command_dim,
                           latent_dim=latent_dim, num_experts=num_experts, chunk=chunk, num_phases=num_phases,
                           frames=frames, dropout=dropout)
        self.scene_enc = StreamEncoder(frames)
        self.hand_enc = StreamEncoder(frames)
        self.to_latent = nn.Sequential(
            nn.Linear(self.scene_enc.out_dim + self.hand_enc.out_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.Tanh(),
        )
        ctx_dim = latent_dim + 64
        self.proprio = nn.Sequential(nn.Linear(proprio_dim + command_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(ctx_dim, 32), nn.SiLU(), nn.Linear(32, num_experts))
        out = chunk * (act_dim + bin_dim)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(ctx_dim, 160), nn.SiLU(), nn.Dropout(dropout), nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, out))
            for _ in range(num_experts)
        )
        self.phase_head = nn.Linear(latent_dim, num_phases)

    def forward(self, scene_img, hand_img, proprio, command=None):
        """Images: float N x (3*frames) x H x W in [0, 1]. Returns dict of tensors."""
        cfg = self.config
        z = self.to_latent(torch.cat([self.scene_enc(scene_img), self.hand_enc(hand_img)], dim=1))
        p_in = proprio if command is None or cfg["command_dim"] == 0 else torch.cat([proprio, command], dim=1)
        ctx = torch.cat([z, self.proprio(p_in)], dim=1)
        gate = torch.softmax(self.gate(ctx), dim=-1)
        outs = torch.stack([e(ctx) for e in self.experts], dim=1)            # N x E x out
        mixed = (gate.unsqueeze(-1) * outs).sum(dim=1)
        mixed = mixed.view(-1, cfg["chunk"], cfg["act_dim"] + cfg["bin_dim"])
        return {
            "act": torch.tanh(mixed[..., : cfg["act_dim"]]),                  # normalized to [-1, 1]
            "bin_logit": mixed[..., cfg["act_dim"]:],
            "phase_logit": self.phase_head(z),
            "gate": gate,
            "latent": z,
        }


def gate_balance(gate):
    usage = gate.mean(dim=0)
    return ((usage - 1.0 / usage.numel()) ** 2).sum() * usage.numel()


def policy_loss(out, act_t, bin_t, phase_t, valid, bin_weight=0.5, phase_weight=0.1, balance_weight=0.01):
    """act_t: N x K x A in [-1,1]; bin_t: N x K x B in {0,1}; valid: N x K mask for chunk steps."""
    m = valid.unsqueeze(-1)
    act_loss = (F.smooth_l1_loss(out["act"], act_t, reduction="none", beta=0.05) * m).sum() / (m.sum() * act_t.shape[-1]).clamp_min(1)
    bin_loss = (F.binary_cross_entropy_with_logits(out["bin_logit"], bin_t, reduction="none") * m).sum() / (m.sum() * bin_t.shape[-1]).clamp_min(1)
    phase_loss = F.cross_entropy(out["phase_logit"], phase_t)
    bal = gate_balance(out["gate"])
    total = act_loss + bin_weight * bin_loss + phase_weight * phase_loss + balance_weight * bal
    return total, {"act": float(act_loss.detach()), "bin": float(bin_loss.detach()), "phase": float(phase_loss.detach()), "balance": float(bal.detach())}
