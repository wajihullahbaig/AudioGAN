"""AudioMNIST & raw 1D waveform dataset with clean resampling and zero-preserving normalization.

Fixes all preprocessing issues from the legacy notebooks:
  1. Preserves native WAV sample rate (AudioMNIST is 48 kHz, not 44.1 kHz) and
     polyphase-resamples cleanly to 16 kHz (16,384 samples ~= 1.024 s).
  2. Symmetric peak normalization around 0.0 (no DC offset on silence!).
  3. Parses digit labels (0..9) from AudioMNIST filenames (`{digit}_{speaker}_{idx}.wav/npy`).
  4. Handles 2D `(N, 1)` arrays from `recorder.ipynb` automatically.
  5. Optional one-time `.npz` cache for blazing-fast epoch loading.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset

_DIGIT_PATTERN = re.compile(r"^([0-9])_")


def _extract_digit_label(filename: str) -> Optional[int]:
    m = _DIGIT_PATTERN.match(os.path.basename(filename))
    return int(m.group(1)) if m else None


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    """Convert raw PCM int or float array of shape (N,) or (N, C) to 1D float32."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        # Handle (N, 1) from recorder.ipynb or stereo (N, 2)
        arr = arr.mean(axis=1) if arr.shape[1] > 1 else arr[:, 0]
    elif arr.ndim > 2:
        arr = arr.reshape(-1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32)
    return arr


def _resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or waveform.size == 0:
        return waveform
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(waveform, up, down).astype(np.float32)


def _normalize_and_fit(
    waveform: np.ndarray,
    target_length: int,
    peak: float = 0.95,
    random_pad: bool = False,
) -> np.ndarray:
    """Remove DC offset on active signal, peak-normalize symmetrically, and fit length."""
    if waveform.size == 0:
        return np.zeros(target_length, dtype=np.float32)

    # Remove microphone DC offset before padding so silence stays strictly at 0.0
    waveform = waveform - float(np.mean(waveform))
    max_abs = float(np.max(np.abs(waveform)))
    if max_abs > 1e-7:
        waveform = (waveform / max_abs) * peak

    if waveform.size < target_length:
        pad_total = target_length - waveform.size
        if random_pad:
            # Randomize speech onset inside the window so D can't memorize start index
            pad_left = int(np.random.randint(0, pad_total + 1))
        else:
            pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), mode="constant", constant_values=0.0)
    elif waveform.size > target_length:
        if random_pad:
            start = int(np.random.randint(0, waveform.size - target_length + 1))
        else:
            start = (waveform.size - target_length) // 2
        waveform = waveform[start : start + target_length]

    return waveform.astype(np.float32)


def _collect_audio_files(data_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith(".wav") or low.endswith(".npy"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def load_single_audio(
    file_path: str,
    target_sr: int = 16000,
    npy_source_sr: int = 48000,
) -> np.ndarray:
    """Load a `.wav` or `.npy` file and resample to `target_sr`."""
    if file_path.lower().endswith(".wav"):
        sr, data = wavfile.read(file_path)
        wav = _to_float_mono(data)
        return _resample_audio(wav, orig_sr=int(sr), target_sr=target_sr)
    data = np.load(file_path)
    wav = _to_float_mono(data)
    return _resample_audio(wav, orig_sr=npy_source_sr, target_sr=target_sr)


def build_dataset_cache(
    data_dir: str,
    cache_path: str,
    target_sr: int = 16000,
    audio_length: int = 16384,
    npy_source_sr: int = 48000,
) -> str:
    """Preprocess an entire audio folder once into a compact int16 `.npz` cache.

    Storing 30,000 AudioMNIST clips at 16,384 samples in int16 takes ~980 MB
    and loads epochs ~20-50x faster than resampling `.wav`/`.npy` on the fly.
    """
    files = _collect_audio_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .wav or .npy files found in '{data_dir}'.")

    waveforms = np.zeros((len(files), audio_length), dtype=np.int16)
    labels = np.zeros((len(files),), dtype=np.int64)
    has_labels = True

    for i, fp in enumerate(files):
        wav = load_single_audio(fp, target_sr=target_sr, npy_source_sr=npy_source_sr)
        wav = _normalize_and_fit(wav, target_length=audio_length, peak=0.95, random_pad=False)
        waveforms[i] = np.clip(np.round(wav * 32767.0), -32768, 32767).astype(np.int16)
        lbl = _extract_digit_label(fp)
        if lbl is None:
            has_labels = False
            labels[i] = 0
        else:
            labels[i] = lbl

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(
        cache_path,
        waveforms=waveforms,
        labels=labels,
        has_labels=np.array(has_labels),
        sample_rate=np.array(target_sr),
    )
    return cache_path


class AudioMNISTDataset(Dataset):
    """Fast 16 kHz dataset supporting AudioMNIST `.wav`, `.npy`, or `.npz` cache."""

    def __init__(
        self,
        data_path: str,
        target_sr: int = 16000,
        audio_length: int = 16384,
        npy_source_sr: int = 48000,
        random_shift: bool = True,
        max_shift: int = 512,
    ):
        self.data_path = data_path
        self.target_sr = target_sr
        self.audio_length = audio_length
        self.npy_source_sr = npy_source_sr
        self.random_shift = random_shift
        self.max_shift = max_shift

        self._cached_waveforms: Optional[np.ndarray] = None
        self._cached_labels: Optional[np.ndarray] = None
        self.files: List[str] = []
        self.labels: List[int] = []
        self.has_labels: bool = False

        path_obj = Path(data_path)
        if path_obj.is_file() and path_obj.suffix.lower() == ".npz":
            npz = np.load(data_path)
            cached_waves = npz["waveforms"]
            self._cached_labels = npz["labels"]
            self.has_labels = bool(npz["has_labels"].item())
            cached_sr = int(npz["sample_rate"].item()) if "sample_rate" in npz else target_sr

            # Automatically adapt cached .npz if target_sr or audio_length changed (e.g. fsdd_16k -> 8k)
            if cached_sr != target_sr or cached_waves.shape[1] != audio_length:
                adapted = np.zeros((cached_waves.shape[0], audio_length), dtype=np.int16)
                for i in range(cached_waves.shape[0]):
                    w = cached_waves[i].astype(np.float32) / 32767.0
                    if cached_sr != target_sr:
                        w = _resample_audio(w, orig_sr=cached_sr, target_sr=target_sr)
                    w = _normalize_and_fit(w, target_length=audio_length, peak=0.95, random_pad=False)
                    adapted[i] = np.clip(np.round(w * 32767.0), -32768, 32767).astype(np.int16)
                self._cached_waveforms = adapted
            else:
                self._cached_waveforms = cached_waves
        elif path_obj.is_dir():
            self.files = _collect_audio_files(data_path)
            if not self.files:
                raise FileNotFoundError(f"No .wav or .npy files found under '{data_path}'.")
            parsed = [_extract_digit_label(f) for f in self.files]
            self.has_labels = all(p is not None for p in parsed)
            self.labels = [p if p is not None else 0 for p in parsed]
        else:
            raise FileNotFoundError(f"Dataset path not found: '{data_path}'")

    def __len__(self) -> int:
        if self._cached_waveforms is not None:
            return int(self._cached_waveforms.shape[0])
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_waveforms is not None:
            wav = self._cached_waveforms[idx].astype(np.float32) / 32767.0
            if wav.size != self.audio_length:
                wav = _normalize_and_fit(wav, self.audio_length, random_pad=self.random_shift)
            elif self.random_shift and self.max_shift > 0:
                shift = int(np.random.randint(-self.max_shift, self.max_shift + 1))
                if shift > 0:
                    wav = np.pad(wav[:-shift], (shift, 0), mode="constant")
                elif shift < 0:
                    wav = np.pad(wav[-shift:], (0, -shift), mode="constant")
            label = int(self._cached_labels[idx])
        else:
            wav = load_single_audio(
                self.files[idx],
                target_sr=self.target_sr,
                npy_source_sr=self.npy_source_sr,
            )
            wav = _normalize_and_fit(
                wav,
                target_length=self.audio_length,
                peak=0.95,
                random_pad=self.random_shift,
            )
            label = self.labels[idx]

        # Return shape (1, T) float32 waveform and int64 digit label
        return torch.from_numpy(wav).unsqueeze(0), torch.tensor(label, dtype=torch.long)
