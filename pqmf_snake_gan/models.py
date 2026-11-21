"""Pure Time-Domain PQMF-Snake Generator + Multi-Resolution STFT / Time-Domain Discriminators.

Generator stays 100% pure time-domain (zero spectrograms):
    (z, digit) -> class temporal blueprint + 1D self-attention (16 -> 64)
               -> linear upsampling + MRFSnakeBlock (silence-preserving StyleGate at high rates)
               -> 4 PQMF subbands -> Frozen PQMF.synthesis -> 1D waveform (8 kHz or 16 kHz).

Discriminator (training-only) combines:
    1. Multi-Resolution STFT 2D Discriminator (MRD) with temporal-preserving word pooling
       and an Auxiliary 10-class Digit Classifier head (exposes formants, fricatives & phoneme order).
    2. Multi-Band Time-Domain Discriminator (PQMF subbands + rectified amplitude envelope)
       with temporal-preserving pooling.
    3. Multi-Period Discriminator (MPD) for crisp pitch harmonics & phase coherence.

All sub-discriminators support single-pass matching-aware wrong-label scoring (zero extra
conv FLOPs) and un-attenuated Projection Conditioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    MRFSnakeBlock,
    PhaseShuffle1d,
    SnakeBeta,
    TemporalSelfAttention1d,
    apply_spectral_norm,
    apply_weight_norm,
)
from .pqmf import PQMF


def default_upsample_rates(
    audio_length: int = 16384,
    init_length: int = 16,
    subbands: int = 4,
) -> Tuple[int, ...]:
    """Select clean upsampling factors for standard 8 kHz (8192) or 16 kHz (16384) lengths."""
    if audio_length % (init_length * subbands) != 0:
        raise ValueError(
            f"audio_length ({audio_length}) must be divisible by init_length*subbands ({init_length * subbands})."
        )
    factor = audio_length // (init_length * subbands)
    rates: List[int] = []
    rem = factor
    while rem >= 4 and rem % 4 == 0:
        rates.append(4)
        rem //= 4
    while rem >= 2 and rem % 2 == 0:
        rates.append(2)
        rem //= 2
    if rem != 1:
        rates.append(rem)
    return tuple(rates)


def default_stft_resolutions(sample_rate: int = 16000) -> Tuple[Tuple[int, int, int], ...]:
    """Return 3 STFT resolutions (n_fft, hop_length, win_length) matched to ~16/32/64 ms windows."""
    if sample_rate <= 8000:
        return (
            (128, 32, 128),
            (256, 64, 256),
            (512, 128, 512),
        )
    return (
        (256, 64, 256),
        (512, 128, 512),
        (1024, 256, 1024),
    )


class PQMFGenerator(nn.Module):
    """100% Time-Domain Conditional Generator producing M PQMF subbands -> waveform.

    Upgrades over v1:
      - Learned per-class temporal blueprint (`class_time_emb`) so each digit starts
        with its own multi-syllable/phoneme temporal plan.
      - Lightweight 1D Temporal Self-Attention at lengths 16 and 64 for global word coherence.
      - Linear upsampling (no nearest-neighbor staircase aliasing) + wider late channels.
      - Semantic FiLM at length 64, silence-preserving multiplicative StyleGate at high rates.
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
        self.style_dim = z_dim + self.cond_dim
        self.subbands = subbands
        self.init_length = init_length
        self.base_channels = base_channels

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
            # Dedicated temporal skeleton per digit class (B, base_channels, init_length)
            self.class_time_emb = nn.Parameter(
                torch.randn(num_classes, base_channels, init_length) * 0.1
            )
        else:
            self.class_emb = None
            self.cond_mlp = None
            self.class_time_emb = None

        self.input_proj = apply_weight_norm(
            nn.Linear(self.style_dim, base_channels * init_length)
        )
        self.init_attn = TemporalSelfAttention1d(base_channels, num_heads=4)

        # Wider channel schedule than v1: e.g. 256 -> 256 -> 192 -> 128 -> 64
        ratios = [1.0, 0.75, 0.5, 0.25]
        ch_schedule = [base_channels]
        for i in range(len(upsample_rates)):
            ratio = ratios[i] if i < len(ratios) else max(0.25, ratios[-1] / (2 ** (i - len(ratios) + 1)))
            ch = max(32, int(round(base_channels * ratio)))
            ch = max(32, (ch // 8) * 8)
            ch_schedule.append(ch)

        self.upsample_rates = list(upsample_rates)
        self.up_convs = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()
        self.mid_attn = TemporalSelfAttention1d(ch_schedule[1], num_heads=4)

        for i, r in enumerate(self.upsample_rates):
            in_ch = ch_schedule[i]
            out_ch = ch_schedule[i + 1]
            k = 2 * r + 1
            self.up_convs.append(
                apply_weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=r))
            )
            # Semantic FiLM only at stage 0 (length 64); silence-preserving StyleGate at higher rates
            self.mrf_blocks.append(
                MRFSnakeBlock(
                    channels=out_ch,
                    cond_dim=self.style_dim,
                    kernel_size=5,
                    dilations=(1, 3, 5),
                    cond_mode="film" if i == 0 else "gate",
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
            style = torch.cat([z, cond], dim=-1)
        else:
            style = z

        h = self.input_proj(style).view(z.size(0), self.base_channels, self.init_length)
        if self.class_time_emb is not None and labels is not None:
            h = h + self.class_time_emb[labels]
        h = self.init_attn(h)

        for i, (r, up_conv, mrf) in enumerate(zip(self.upsample_rates, self.up_convs, self.mrf_blocks)):
            h = F.interpolate(h, scale_factor=r, mode="linear", align_corners=False)
            h = up_conv(h)
            h = mrf(h, style)
            if i == 0:
                h = self.mid_attn(h)

        subbands = torch.tanh(self.out_conv(self.out_act(h)))
        waveform = self.pqmf.synthesis(subbands)

        if return_subbands:
            return waveform, subbands
        return waveform


class _STFTSubDiscriminator(nn.Module):
    """2D Conv discriminator on log-magnitude STFT with temporal-preserving word head & classifier."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_classes: int = 10,
        base_ch: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_classes = num_classes
        self.register_buffer("window", torch.hann_window(win_length))

        chs = [1, base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        kernels = [(3, 9), (3, 7), (3, 5), (3, 5), (3, 3)]
        strides = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]

        self.convs = nn.ModuleList(
            [
                apply_spectral_norm(
                    nn.Conv2d(
                        chs[i],
                        chs[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=(kernels[i][0] // 2, kernels[i][1] // 2),
                    )
                )
                for i in range(len(strides))
            ]
        )
        last_ch = chs[-1]
        self.conv_post = apply_spectral_norm(nn.Conv2d(last_ch, 1, kernel_size=3, padding=1))

        # Preserve (2 frequency zones x 4 ordered temporal quadrants) instead of global mean pooling
        self.word_conv = apply_spectral_norm(nn.Conv2d(last_ch, last_ch, kernel_size=(2, 4)))
        if num_classes > 0:
            self.class_proj = apply_spectral_norm(nn.Embedding(num_classes, last_ch))
            self.cls_head = nn.Linear(last_ch, num_classes)
        else:
            self.class_proj = None
            self.cls_head = None

    def _compute_log_mag(self, x: torch.Tensor) -> torch.Tensor:
        # Run STFT in float32 with epsilon inside sqrt for 100% AMP & silence-gradient stability
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = x.squeeze(1).float()
            spec = torch.stft(
                x32,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.float(),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-5)
            log_mag = torch.log1p(mag).unsqueeze(1)
        return log_mag.to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self._compute_log_mag(x)
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.2)

        score_uncond = self.conv_post(h).mean(dim=(1, 2, 3))

        # Temporally ordered word embedding (B, last_ch)
        grid = F.adaptive_avg_pool2d(h, (2, 4))
        h_word = F.leaky_relu(self.word_conv(grid).flatten(1), 0.2)

        score = score_uncond
        score_wrong = None
        cls_logits = None

        if self.class_proj is not None and labels is not None:
            # Un-attenuated Miyato & Koyama projection (no / sqrt(C) shrinkage!)
            score = score_uncond + (h_word * self.class_proj(labels)).sum(dim=-1)
            if wrong_labels is not None:
                score_wrong = score_uncond + (h_word * self.class_proj(wrong_labels)).sum(dim=-1)
        if self.cls_head is not None:
            cls_logits = self.cls_head(h_word)

        return score, score_wrong, cls_logits


class MultiResolutionSTFTDiscriminator(nn.Module):
    """Multi-Resolution STFT 2D Discriminator (training-only) with Auxiliary Digit Classifiers."""

    def __init__(
        self,
        sample_rate: int = 16000,
        resolutions: Optional[Sequence[Tuple[int, int, int]]] = None,
        num_classes: int = 10,
        base_ch: int = 32,
    ):
        super().__init__()
        if resolutions is None:
            resolutions = default_stft_resolutions(sample_rate)
        self.sub_discriminators = nn.ModuleList(
            [
                _STFTSubDiscriminator(
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    num_classes=num_classes,
                    base_ch=base_ch,
                )
                for n_fft, hop, win in resolutions
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        scores: List[torch.Tensor] = []
        wrong_scores: List[torch.Tensor] = []
        cls_logits: List[torch.Tensor] = []
        for d in self.sub_discriminators:
            s, sw, lg = d(x, labels, wrong_labels)
            scores.append(s)
            if sw is not None:
                wrong_scores.append(sw)
            if lg is not None:
                cls_logits.append(lg)
        return scores, wrong_scores, cls_logits


class _PeriodSubDiscriminator(nn.Module):
    """Single-period 2D time-domain discriminator with un-attenuated projection conditioning."""

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

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)

        score_uncond = self.conv_post(x).mean(dim=(1, 2, 3))
        score = score_uncond
        score_wrong = None
        if self.class_proj is not None and labels is not None:
            pooled = x.mean(dim=(2, 3))
            score = score_uncond + (pooled * self.class_proj(labels)).sum(dim=-1)
            if wrong_labels is not None:
                score_wrong = score_uncond + (pooled * self.class_proj(wrong_labels)).sum(dim=-1)
        return score, score_wrong


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator operating purely on 1D waveforms for pitch/harmonics."""

    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7),
        num_classes: int = 10,
        base_ch: int = 16,
    ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [_PeriodSubDiscriminator(p, num_classes=num_classes, base_ch=base_ch) for p in periods]
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        scores: List[torch.Tensor] = []
        wrong_scores: List[torch.Tensor] = []
        for d in self.sub_discriminators:
            s, sw = d(x, labels, wrong_labels)
            scores.append(s)
            if sw is not None:
                wrong_scores.append(sw)
        return scores, wrong_scores


class _Band1DSubDiscriminator(nn.Module):
    """1D strided Conv discriminator with PhaseShuffle + temporal-preserving word projection."""

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
            self.shuffles.append(PhaseShuffle1d(phase_shuffle_rad if i < 2 else 0))

        last_ch = chs[-1]
        self.conv_post = apply_spectral_norm(nn.Conv1d(last_ch, 1, kernel_size=3, padding=1))
        # Preserve 6 ordered temporal slots instead of destructive global mean pooling
        self.word_conv = apply_spectral_norm(nn.Conv1d(last_ch, last_ch, kernel_size=6))
        if num_classes > 0:
            self.class_proj = apply_spectral_norm(nn.Embedding(num_classes, last_ch))
        else:
            self.class_proj = None

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for conv, shuf in zip(self.convs, self.shuffles):
            x = shuf(F.leaky_relu(conv(x), 0.2))
        score_uncond = self.conv_post(x).mean(dim=(1, 2))
        score = score_uncond
        score_wrong = None
        if self.class_proj is not None and labels is not None:
            t_grid = F.adaptive_avg_pool1d(x, 6)
            h_word = F.leaky_relu(self.word_conv(t_grid).squeeze(-1), 0.2)
            score = score_uncond + (h_word * self.class_proj(labels)).sum(dim=-1)
            if wrong_labels is not None:
                score_wrong = score_uncond + (h_word * self.class_proj(wrong_labels)).sum(dim=-1)
        return score, score_wrong


class MultiBandDiscriminator(nn.Module):
    """Time-domain PQMF subband + rectified amplitude-envelope 1D discriminators."""

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

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        subbands = self.pqmf.analysis(x)
        # Rectified smooth amplitude envelope (float32-safe sqrt for FP16 AMP stability)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            env_f = torch.sqrt(x.float().pow(2) + 1e-5)
        envelope = self.pool(env_f.to(x.dtype))
        s1, w1 = self.subband_disc(subbands, labels, wrong_labels)
        s2, w2 = self.envelope_disc(envelope, labels, wrong_labels)
        scores = [s1, s2]
        wrong_scores = [w for w in (w1, w2) if w is not None]
        return scores, wrong_scores


class CompositeDiscriminator(nn.Module):
    """Balanced Multi-Resolution STFT (0.50) + Multi-Band (0.25) + Multi-Period (0.25) Ensemble."""

    def __init__(
        self,
        sample_rate: int = 16000,
        subbands: int = 4,
        pqmf_taps: int = 64,
        num_classes: int = 10,
        periods: Sequence[int] = (2, 3, 5, 7),
        mpd_base_ch: int = 16,
        mbd_base_ch: int = 32,
        mrd_base_ch: int = 32,
        phase_shuffle_rad: int = 2,
    ):
        super().__init__()
        self.mrd = MultiResolutionSTFTDiscriminator(
            sample_rate=sample_rate, num_classes=num_classes, base_ch=mrd_base_ch
        )
        self.mbd = MultiBandDiscriminator(
            subbands=subbands,
            pqmf_taps=pqmf_taps,
            num_classes=num_classes,
            base_ch=mbd_base_ch,
            phase_shuffle_rad=phase_shuffle_rad,
        )
        self.mpd = MultiPeriodDiscriminator(
            periods=periods, num_classes=num_classes, base_ch=mpd_base_ch
        )

        # Balanced group weights so MRD (formants/phonemes) leads and MPD doesn't drown it out
        n_mrd = len(self.mrd.sub_discriminators)
        n_mbd = 2
        n_mpd = len(self.mpd.sub_discriminators)
        self.head_weights: List[float] = (
            [0.50 / n_mrd] * n_mrd
            + [0.25 / n_mbd] * n_mbd
            + [0.25 / n_mpd] * n_mpd
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wrong_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        mrd_s, mrd_w, cls_logits = self.mrd(x, labels, wrong_labels)
        mbd_s, mbd_w = self.mbd(x, labels, wrong_labels)
        mpd_s, mpd_w = self.mpd(x, labels, wrong_labels)
        return {
            "scores": mrd_s + mbd_s + mpd_s,
            "wrong_scores": mrd_w + mbd_w + mpd_w,
            "cls_logits": cls_logits,
        }
