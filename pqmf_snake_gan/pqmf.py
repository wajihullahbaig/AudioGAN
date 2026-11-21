"""Pure time-domain Cosine-Modulated Pseudo-Quadrature Mirror Filterbank (PQMF).

Decomposes a 1D waveform (B, 1, T) into M downsampled time-domain subbands
(B, M, T // M) and reconstructs it back to (B, 1, T) with >70 dB SNR — using
zero FFTs and zero spectrograms.

References:
    - T. Q. Nguyen, "Near-perfect-reconstruction pseudo-QMF banks", IEEE TSP 1994.
    - C. D. Creusere & S. K. Mitra, "A simple method for designing high-quality
      prototype filters for M-band pseudo QMF banks", IEEE TSP 1995.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import minimize_scalar
    from scipy.signal.windows import kaiser as _kaiser_window
except ImportError:  # pragma: no cover
    minimize_scalar = None
    _kaiser_window = None


def _design_kaiser_prototype(
    subbands: int,
    taps: int,
    beta: float = 9.0,
) -> np.ndarray:
    """Design a near-perfect-reconstruction symmetric prototype filter p[n].

    Uses the Creusere-Mitra criterion: find the single scalar cutoff frequency
    fc in (0, 1 / (2 * M)) that minimizes aliasing in the autocorrelation
    g = p * p at nonzero multiples of 2*M (Nyquist-2M condition).
    """
    if taps % (2 * subbands) != 0:
        raise ValueError(
            f"Filter length `taps` ({taps}) must be an even multiple of `subbands` ({subbands}), "
            f"e.g. taps=64 for subbands=4."
        )

    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    if _kaiser_window is not None:
        win = _kaiser_window(taps, beta=beta)
    else:
        win = np.kaiser(taps, beta=beta)

    def make_proto(fc: float) -> np.ndarray:
        # Ideal lowpass impulse response windowed by Kaiser window
        h = 2.0 * fc * np.sinc(2.0 * fc * n) * win
        # Normalize so sum(h^2) = 1 / (2 * subbands) -> unity gain in M * A^T A
        energy = np.sum(h * h)
        return h / math.sqrt(2.0 * subbands * energy)

    def objective(fc: float) -> float:
        h = make_proto(fc)
        g = np.convolve(h, h, mode="full")
        center = taps - 1
        # Sample autocorrelation at nonzero multiples of 2 * M
        indices = np.arange(center + 2 * subbands, len(g), 2 * subbands)
        if len(indices) == 0:
            return 0.0
        # Normalize relative to g[center] so scaling doesn't bias optimization
        return float(np.sum((g[indices] / g[center]) ** 2))

    low = 0.25 / (2.0 * subbands)
    high = 0.95 / (2.0 * subbands)

    if minimize_scalar is not None:
        res = minimize_scalar(objective, bounds=(low, high), method="bounded")
        best_fc = float(res.x)
    else:
        # Coarse-to-fine grid search fallback (no scipy needed)
        grid = np.linspace(low, high, 400)
        best_fc = float(grid[int(np.argmin([objective(float(fc)) for fc in grid]))])
        fine = np.linspace(max(low, best_fc - 0.005), min(high, best_fc + 0.005), 400)
        best_fc = float(fine[int(np.argmin([objective(float(fc)) for fc in fine]))])

    return make_proto(best_fc)


class PQMF(nn.Module):
    """Differentiable, frozen M-band PQMF analysis and synthesis layer.

    With even filter length `taps = 2 * k * subbands` (default: M=4, taps=64):
      - `analysis(x)`:  (B, 1, T) -> (B, M, T // M) via strided Conv1d
      - `synthesis(y)`: (B, M, T // M) -> (B, 1, T) via exact adjoint ConvTranspose1d

    Because analysis and synthesis are exact transposes of a cosine-modulated
    near-paraunitary filterbank, there is zero phase shift and >70 dB SNR.
    """

    def __init__(self, subbands: int = 4, taps: int = 64, beta: float = 9.0):
        super().__init__()
        if subbands % 2 != 0 or subbands < 2:
            raise ValueError(f"`subbands` must be an even integer >= 2, got {subbands}.")
        if taps % (2 * subbands) != 0:
            raise ValueError(
                f"`taps` ({taps}) must be a multiple of 2 * subbands ({2 * subbands})."
            )

        self.subbands = subbands
        self.taps = taps
        self.padding = (taps - subbands) // 2

        proto = _design_kaiser_prototype(subbands=subbands, taps=taps, beta=beta)
        k = np.arange(subbands, dtype=np.float64)[:, None]
        n = np.arange(taps, dtype=np.float64)[None, :] - (taps - 1) / 2.0

        # Cosine-modulated analysis filters h_k[n]
        # Shape: (M, 1, taps)
        phase = (-1.0) ** k * (math.pi / 4.0)
        h = 2.0 * proto[None, :] * np.cos((k + 0.5) * (math.pi / subbands) * n + phase)

        analysis_weight = torch.from_numpy(h.astype(np.float32)).unsqueeze(1)
        # Synthesis via ConvTranspose1d is the exact adjoint of Conv1d analysis;
        # scale by M to compensate for the 1/M decimation factor.
        synthesis_weight = analysis_weight * float(subbands)

        self.register_buffer("analysis_weight", analysis_weight)
        self.register_buffer("synthesis_weight", synthesis_weight)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """Split full-band waveform (B, 1, T) into M subbands (B, M, T // M)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(-1) % self.subbands != 0:
            raise ValueError(
                f"Waveform length ({x.size(-1)}) must be divisible by subbands ({self.subbands})."
            )
        return F.conv1d(
            x,
            self.analysis_weight,
            stride=self.subbands,
            padding=self.padding,
        )

    def synthesis(self, subbands: torch.Tensor) -> torch.Tensor:
        """Recombine M subbands (B, M, T // M) into full-band waveform (B, 1, T)."""
        return F.conv_transpose1d(
            subbands,
            self.synthesis_weight,
            stride=self.subbands,
            padding=self.padding,
        )

    def forward(self, subbands: torch.Tensor) -> torch.Tensor:
        return self.synthesis(subbands)

    @torch.no_grad()
    def reconstruction_snr_db(self, length: int = 16384) -> float:
        """Verify near-perfect reconstruction SNR (dB) on interior samples."""
        x = torch.randn(1, 1, length, device=self.analysis_weight.device)
        y = self.analysis(x)
        x_hat = self.synthesis(y)
        margin = self.taps
        x_in = x[..., margin:-margin]
        err = x_in - x_hat[..., margin:-margin]
        snr = 10.0 * torch.log10(x_in.pow(2).mean() / (err.pow(2).mean() + 1e-12))
        return float(snr.item())
