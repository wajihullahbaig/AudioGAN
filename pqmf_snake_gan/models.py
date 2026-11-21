"""Multi-Band PQMF + Snake-Activation Conditional Generator & Time-Domain Discriminators."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    MRFSnakeBlock,
    PhaseShuffle1d,
    SnakeBeta,
    apply_spectral_norm,
    apply_weight_norm,
)
from .pqmf import PQMF


class PQMFGenerator(nn.Module):
    """Fast 1D Conditional Generator producing M PQMF subbands -> 16 kHz audio.

    Architecture (default for 16,384 samples @ 16 kHz, M=4 subbands):
        (z, digit) -> (B, 256, 16)
                   -> 4x Upsample + Conv1d + MRFSnakeBlock (16 -> 64 -> 256 -> 1024 -> 4096)
                   -> Conv1d -> tanh -> 4 subbands (B, 4, 4096)
                   -> Frozen PQMF.synthesis -> (B, 1, 16384)
    """

    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 10,
        cond_dim: int = 64,
        base_channels: int = 256,
        subbands: int = 4,
        pqmf_taps: int = 64,
        init_length: int = 16,
        upsample_rates: Sequence[int] = (4, 4, 4, 4),
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.cond_dim = cond_dim if num_classes > 0 else 0
        self.subbands = subbands
        self.init_length = init_length
        self.base_channels = base_channels

        # Expected subband and full waveform lengths
        up_prod = 1
        for r in upsample_rates:
            up_prod *= r
        self.subband_length = init_length * up_prod
        self.audio_length = self.subband_length * subbands

        if self.num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, cond_dim)
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.class_emb = None
            self.cond_mlp = None

        self.input_proj = apply_weight_norm(
            nn.Linear(z_dim + self.cond_dim, base_channels * init_length)
        )

        # Channel schedule tapers cleanly as temporal resolution grows
        ch_schedule = [base_channels]
        ch = base_channels
        for i in range(len(upsample_rates)):
            ch = max(32, ch // 2 if i > 0 else int(ch * 0.75))
            # Make divisible by 8 for GroupNorm inside FiLM1d
            ch = max(32, (ch // 8) * 8)
            ch_schedule.append(ch)

        self.upsample_rates = list(upsample_rates)
        self.up_convs = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()

        for i, r in enumerate(self.upsample_rates):
            in_ch = ch_schedule[i]
            out_ch = ch_schedule[i + 1]
            k = 2 * r + 1
            self.up_convs.append(
                apply_weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=r))
            )
            self.mrf_blocks.append(
                MRFSnakeBlock(
                    channels=out_ch,
                    cond_dim=self.cond_dim,
                    kernel_size=5,
                    dilations=(1, 3, 5),
                )
            )

        last_ch = ch_schedule[-1]
        self.out_act = SnakeBeta(last_ch)
        self.out_conv = apply_weight_norm(
            nn.Conv1d(last_ch, subbands, kernel_size=7, padding=3)
        )
        self.pqmf = PQMF(subbands=subbands, taps=pqmf_taps)

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_subbands: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.num_classes > 0:
            if labels is None:
                raise ValueError("`labels` must be provided when `num_classes > 0`.")
            cond = self.cond_mlp(self.class_emb(labels))
            h_in = torch.cat([z, cond], dim=-1)
        else:
            cond = None
            h_in = z

        h = self.input_proj(h_in).view(z.size(0), self.base_channels, self.init_length)

        for r, up_conv, mrf in zip(self.upsample_rates, self.up_convs, self.mrf_blocks):
            h = F.interpolate(h, scale_factor=r, mode="nearest")
            h = up_conv(h)
            h = mrf(h, cond)

        subbands = torch.tanh(self.out_conv(self.out_act(h)))
        waveform = self.pqmf.synthesis(subbands)

        if return_subbands:
            return waveform, subbands
        return waveform


class _PeriodSubDiscriminator(nn.Module):
    """Single-period 2D time-domain discriminator with projection conditioning."""

    def __init__(self, period: int, num_classes: int = 10, base_ch: int = 16):
        super().__init__()
        self.period = period
        self.num_classes = num_classes

        chs = [1, base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        strides = [3, 3, 3, 3, 1]
        self.convs = nn.ModuleList(
            [
                apply_spectral_norm(
                    nn.Conv2d(chs[i], chs[i + 1], kernel_size=(5, 1), stride=(strides[i], 1), padding=(2, 0))
                )
                for i in range(len(strides))
            ]
        )
        last_ch = chs[-1]
        self.conv_post = apply_spectral_norm(
            nn.Conv2d(last_ch, 1, kernel_size=(3, 1), padding=(1, 0))
        )
        if num_classes > 0:
            self.class_proj = apply_spectral_norm(nn.Embedding(num_classes, last_ch))
        else:
            self.class_proj = None

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, 1, T) -> pad to multiple of period -> (B, 1, T // p, p)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)

        # Global mean score + projection conditioning inner product
        score = self.conv_post(x).mean(dim=(1, 2, 3))
        if self.class_proj is not None and labels is not None:
            pooled = x.mean(dim=(2, 3))  # (B, last_ch)
            proj = (pooled * self.class_proj(labels)).sum(dim=-1) / (pooled.size(-1) ** 0.5)
            score = score + proj
        return score


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator operating purely on 1D waveforms (no STFT)."""

    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        num_classes: int = 10,
        base_ch: int = 16,
    ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [_PeriodSubDiscriminator(p, num_classes=num_classes, base_ch=base_ch) for p in periods]
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        return [d(x, labels) for d in self.sub_discriminators]


class _Band1DSubDiscriminator(nn.Module):
    """1D strided Conv discriminator with PhaseShuffle + projection conditioning."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        base_ch: int = 32,
        phase_shuffle_rad: int = 2,
    ):
        super().__init__()
        chs = [in_channels, base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        kernels = [15, 15, 15, 9, 5]
        strides = [4, 4, 4, 2, 2]
        groups = [1, 2, 4, 4, 1]

        self.convs = nn.ModuleList()
        self.shuffles = nn.ModuleList()
        for i in range(len(strides)):
            self.convs.append(
                apply_spectral_norm(
                    nn.Conv1d(
                        chs[i],
                        chs[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=kernels[i] // 2,
                        groups=groups[i],
                    )
                )
            )
            # Apply phase shuffle on early high-rate layers
            self.shuffles.append(PhaseShuffle1d(phase_shuffle_rad if i < 2 else 0))

        last_ch = chs[-1]
        self.conv_post = apply_spectral_norm(nn.Conv1d(last_ch, 1, kernel_size=3, padding=1))
        if num_classes > 0:
            self.class_proj = apply_spectral_norm(nn.Embedding(num_classes, last_ch))
        else:
            self.class_proj = None

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv, shuf in zip(self.convs, self.shuffles):
            x = shuf(F.leaky_relu(conv(x), 0.2))
        score = self.conv_post(x).mean(dim=(1, 2))
        if self.class_proj is not None and labels is not None:
            pooled = x.mean(dim=-1)
            proj = (pooled * self.class_proj(labels)).sum(dim=-1) / (pooled.size(-1) ** 0.5)
            score = score + proj
        return score


class MultiBandDiscriminator(nn.Module):
    """Pure time-domain PQMF subband + macro-envelope 1D discriminators."""

    def __init__(
        self,
        subbands: int = 4,
        pqmf_taps: int = 64,
        num_classes: int = 10,
        base_ch: int = 32,
        phase_shuffle_rad: int = 2,
    ):
        super().__init__()
        self.pqmf = PQMF(subbands=subbands, taps=pqmf_taps)
        self.subband_disc = _Band1DSubDiscriminator(
            in_channels=subbands,
            num_classes=num_classes,
            base_ch=base_ch,
            phase_shuffle_rad=phase_shuffle_rad,
        )
        self.envelope_disc = _Band1DSubDiscriminator(
            in_channels=1,
            num_classes=num_classes,
            base_ch=base_ch,
            phase_shuffle_rad=phase_shuffle_rad,
        )
        self.pool = nn.AvgPool1d(kernel_size=subbands, stride=subbands)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        subbands = self.pqmf.analysis(x)
        envelope = self.pool(x)
        return [
            self.subband_disc(subbands, labels),
            self.envelope_disc(envelope, labels),
        ]


class CompositeDiscriminator(nn.Module):
    """Combines Multi-Period (pitch/harmonics) + Multi-Band (subbands/envelope)."""

    def __init__(
        self,
        subbands: int = 4,
        pqmf_taps: int = 64,
        num_classes: int = 10,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        mpd_base_ch: int = 16,
        mbd_base_ch: int = 32,
        phase_shuffle_rad: int = 2,
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            periods=periods, num_classes=num_classes, base_ch=mpd_base_ch
        )
        self.mbd = MultiBandDiscriminator(
            subbands=subbands,
            pqmf_taps=pqmf_taps,
            num_classes=num_classes,
            base_ch=mbd_base_ch,
            phase_shuffle_rad=phase_shuffle_rad,
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        return self.mpd(x, labels) + self.mbd(x, labels)
