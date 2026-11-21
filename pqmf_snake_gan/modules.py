"""Core building blocks for PQMF-Snake-AudioGAN (pure time-domain, zero spectrograms)."""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


def apply_weight_norm(module: nn.Module) -> nn.Module:
    return weight_norm(module)


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    return spectral_norm(module)


class SnakeBeta(nn.Module):
    """Learnable periodic activation from BigVGAN / Ziyin et al.:

        SnakeBeta(x) = x + (1 / beta) * sin^2(alpha * x)

    Parameterized in log-space so alpha, beta > 0 strictly. Gives 1D audio
    convolutions a built-in periodic inductive bias for speech harmonics.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        init_val = float(torch.tensor(alpha_init).log().item())
        self.log_alpha = nn.Parameter(torch.full((1, channels, 1), init_val))
        self.log_beta = nn.Parameter(torch.full((1, channels, 1), init_val))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp exponents to avoid overflow in fp16/bf16 AMP
        alpha = self.log_alpha.clamp(-4.0, 4.0).exp()
        inv_beta = (-self.log_beta.clamp(-4.0, 4.0)).exp()
        return x + inv_beta * torch.sin(alpha * x).pow(2)


class FiLM1d(nn.Module):
    """Feature-wise Linear Modulation for injecting digit class conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels, affine=False)
        self.proj = nn.Linear(cond_dim, channels * 2)
        # Initialize as identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm(x)
        if cond is None:
            return h
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class MRFSnakeBlock(nn.Module):
    """Multi-Receptive-Field dilated 1D residual block with SnakeBeta + FiLM."""

    def __init__(
        self,
        channels: int,
        cond_dim: int = 0,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()
        self.use_cond = cond_dim > 0
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.acts2 = nn.ModuleList()
        self.films = nn.ModuleList()

        for d in dilations:
            pad1 = (kernel_size - 1) * d // 2
            pad2 = (kernel_size - 1) // 2
            self.acts1.append(SnakeBeta(channels))
            self.convs1.append(
                apply_weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad1)
                )
            )
            self.acts2.append(SnakeBeta(channels))
            self.convs2.append(
                apply_weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=pad2)
                )
            )
            self.films.append(FiLM1d(channels, cond_dim) if self.use_cond else nn.Identity())

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        for a1, c1, a2, c2, film in zip(
            self.acts1, self.convs1, self.acts2, self.convs2, self.films
        ):
            h = film(x, cond) if self.use_cond else x
            h = c1(a1(h))
            h = c2(a2(h))
            x = x + h
        return x


class PhaseShuffle1d(nn.Module):
    """WaveGAN-style random phase shuffle for 1D discriminator feature maps.

    Randomly shifts feature maps by [-rad, +rad] samples with reflection
    padding so the discriminator cannot exploit exact sample-index alignment.
    """

    def __init__(self, rad: int = 2):
        super().__init__()
        self.rad = rad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rad <= 0 or not self.training:
            return x
        shift = int(torch.randint(-self.rad, self.rad + 1, (1,), device=x.device).item())
        if shift == 0:
            return x
        if shift > 0:
            x = F.pad(x, (shift, 0), mode="reflect")[..., :-shift]
        else:
            s = -shift
            x = F.pad(x, (0, s), mode="reflect")[..., s:]
        return x


class DiffAugment1d(nn.Module):
    """Differentiable time-domain augmentation (shift + gain jitter) for GANs.

    Applied identically to real and fake waveforms entering the discriminator
    to prevent discriminator memorization and improve sample diversity.
    """

    def __init__(self, max_shift: int = 128, gain_range: float = 0.12):
        super().__init__()
        self.max_shift = max_shift
        self.gain_range = gain_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # 1. Random per-sample gain scaling (preserves zero-crossings / silence)
        if self.gain_range > 0:
            scale = 1.0 + (2.0 * torch.rand(x.size(0), 1, 1, device=x.device, dtype=x.dtype) - 1.0) * self.gain_range
            x = x * scale
        # 2. Random batch-level circular/zero-padded time shift
        if self.max_shift > 0:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x.device).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        return x


class EMA:
    """Exponential Moving Average of generator weights for cleaner inference."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(d).add_(m_p.data, alpha=1.0 - d)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.data.copy_(m_b.data)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow.load_state_dict(state["shadow"])
