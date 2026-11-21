"""Generate spoken-digit WAVs from a trained PQMF-Snake-AudioGAN checkpoint.

Usage:
    python -m pqmf_snake_gan.generate --checkpoint runs/pqmf_snake/checkpoint_latest.pt --digits 0 1 2 3 4 5 6 7 8 9
    python -m pqmf_snake_gan.generate --checkpoint runs/pqmf_snake/checkpoint_latest.pt --digits 7 7 7 --out-dir generated
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .models import PQMFGenerator
from .train import save_wav_pcm16, select_device


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Generate audio with PQMF-Snake-AudioGAN")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--digits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--out-dir", type=str, default="generated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--use-raw-gen", action="store_true", help="Use raw generator instead of EMA weights")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    gen = PQMFGenerator(
        z_dim=cfg["z_dim"],
        num_classes=cfg["num_classes"],
        base_channels=cfg["base_channels"],
    ).to(device)

    if not args.use_raw_gen and "ema" in ckpt:
        gen.load_state_dict(ckpt["ema"]["shadow"])
    else:
        gen.load_state_dict(ckpt["gen"])
    gen.eval()

    n = len(args.digits)
    z = torch.randn(n, cfg["z_dim"], device=device)
    labels = (
        torch.tensor(args.digits, dtype=torch.long, device=device)
        if cfg["num_classes"] > 0
        else None
    )

    waves = gen(z, labels).squeeze(1).cpu().numpy()
    sr = int(cfg.get("sample_rate", 16000))
    os.makedirs(args.out_dir, exist_ok=True)

    for idx, (digit, w) in enumerate(zip(args.digits, waves)):
        stem = f"digit_{digit}_{idx:02d}" if cfg["num_classes"] > 0 else f"sample_{idx:02d}"
        wav_path = os.path.join(args.out_dir, f"{stem}.wav")
        npy_path = os.path.join(args.out_dir, f"{stem}.npy")
        save_wav_pcm16(wav_path, w, sample_rate=sr)
        np.save(npy_path, w.astype(np.float32))
        print(f"Saved {wav_path}")


if __name__ == "__main__":
    main()
