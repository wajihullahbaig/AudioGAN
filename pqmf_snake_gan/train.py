"""Training script for Multi-Band PQMF + Snake-Activation Conditional Audio GAN.

Usage:
    # 1. Verify installation, PQMF SNR, and gradient flow (no dataset required):
    python -m pqmf_snake_gan.train --smoke-test

    # 2. (Optional) Pre-cache AudioMNIST once for fast training:
    python -m pqmf_snake_gan.train --data-dir audioMNIST --build-cache data_cache/audiomnist_16k.npz

    # 3. Train on AudioMNIST (.wav directory, .npy directory, or .npz cache):
    python -m pqmf_snake_gan.train --data-dir audioMNIST --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io.wavfile import write as write_wav
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioMNISTDataset, build_dataset_cache
from .models import CompositeDiscriminator, PQMFGenerator
from .modules import DiffAugment1d, EMA


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hinge_discriminator_loss(
    real_scores: List[torch.Tensor],
    fake_scores: List[torch.Tensor],
) -> torch.Tensor:
    loss = torch.zeros((), device=real_scores[0].device)
    for r, f in zip(real_scores, fake_scores):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / len(real_scores)


def hinge_generator_loss(fake_scores: List[torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros((), device=fake_scores[0].device)
    for f in fake_scores:
        loss = loss - f.mean()
    return loss / len(fake_scores)


def save_wav_pcm16(path: str, waveform: np.ndarray, sample_rate: int = 16000) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        wav = wav / peak
    pcm16 = np.clip(np.round(wav * 32767.0), -32768, 32767).astype(np.int16)
    write_wav(path, sample_rate, pcm16)


@torch.no_grad()
def export_sample_grid(
    generator: PQMFGenerator,
    device: torch.device,
    out_dir: str,
    epoch: int,
    sample_rate: int = 16000,
    fixed_z: torch.Tensor | None = None,
) -> None:
    """Save 1 generated WAV per digit (0..9) plus a concatenated reel."""
    generator.eval()
    n_samples = generator.num_classes if generator.num_classes > 0 else 4
    if fixed_z is None:
        z = torch.randn(n_samples, generator.z_dim, device=device)
    else:
        z = fixed_z.to(device)
    labels = (
        torch.arange(n_samples, device=device, dtype=torch.long)
        if generator.num_classes > 0
        else None
    )
    waves = generator(z, labels).squeeze(1).cpu().numpy()

    epoch_dir = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    gap = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    reel = []
    for i, w in enumerate(waves):
        tag = f"digit_{i}" if generator.num_classes > 0 else f"sample_{i}"
        save_wav_pcm16(os.path.join(epoch_dir, f"{tag}.wav"), w, sample_rate=sample_rate)
        reel.append(w)
        reel.append(gap)
    save_wav_pcm16(
        os.path.join(out_dir, f"latest_reel_epoch_{epoch:04d}.wav"),
        np.concatenate(reel),
        sample_rate=sample_rate,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_smoke_test(device: torch.device) -> None:
    """Verify PQMF reconstruction SNR, shapes, and non-zero G & D gradients."""
    print(f"[Smoke Test] Device: {device}")
    gen = PQMFGenerator(z_dim=128, num_classes=10, base_channels=192).to(device)
    disc = CompositeDiscriminator(num_classes=10).to(device)
    diff_aug = DiffAugment1d().to(device)

    snr = gen.pqmf.reconstruction_snr_db()
    print(f"[Smoke Test] PQMF 4-band reconstruction SNR: {snr:.2f} dB (target > 60 dB)")
    assert snr > 55.0, f"PQMF SNR unexpectedly low: {snr:.2f} dB"

    print(
        f"[Smoke Test] Generator params: {count_parameters(gen):,} | "
        f"Discriminator params: {count_parameters(disc):,}"
    )

    B = 4
    real = torch.randn(B, 1, gen.audio_length, device=device).clamp(-1.0, 1.0)
    labels = torch.randint(0, 10, (B,), device=device)
    z = torch.randn(B, gen.z_dim, device=device)

    # 1. Discriminator forward & backward
    fake = gen(z, labels)
    assert fake.shape == real.shape, f"Shape mismatch: {fake.shape} vs {real.shape}"

    d_real = disc(diff_aug(real), labels)
    d_fake = disc(diff_aug(fake.detach()), labels)
    d_loss = hinge_discriminator_loss(d_real, d_fake)
    disc.zero_grad(set_to_none=True)
    d_loss.backward()
    d_grad_norm = sum(
        p.grad.norm().item() for p in disc.parameters() if p.grad is not None
    )
    assert d_grad_norm > 0.0, "Discriminator received zero gradients!"

    # 2. Generator forward & backward (MUST NOT detach `fake`!)
    for p in disc.parameters():
        p.requires_grad_(False)
    d_fake_for_g = disc(diff_aug(fake), labels)
    g_loss = hinge_generator_loss(d_fake_for_g) + 0.1 * fake.mean(dim=-1).pow(2).mean()
    gen.zero_grad(set_to_none=True)
    g_loss.backward()
    for p in disc.parameters():
        p.requires_grad_(True)

    g_grad_norm = sum(
        p.grad.norm().item() for p in gen.parameters() if p.grad is not None
    )
    assert g_grad_norm > 0.0, "Generator received zero gradients!"

    print(
        f"[Smoke Test] PASSED! d_loss={d_loss.item():.4f} (grad={d_grad_norm:.3f}), "
        f"g_loss={g_loss.item():.4f} (grad={g_grad_norm:.3f})"
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)

    if args.smoke_test:
        run_smoke_test(device)
        return

    if args.build_cache:
        print(f"Building 16 kHz cache from '{args.data_dir}' -> '{args.build_cache}' ...")
        t0 = time.time()
        build_dataset_cache(
            data_dir=args.data_dir,
            cache_path=args.build_cache,
            target_sr=args.sample_rate,
            audio_length=16384,
            npy_source_sr=args.npy_source_sr,
        )
        print(f"Cache built in {time.time() - t0:.1f}s.")
        if args.epochs <= 0:
            return
        data_source = args.build_cache
    else:
        data_source = args.data_dir

    dataset = AudioMNISTDataset(
        data_path=data_source,
        target_sr=args.sample_rate,
        audio_length=16384,
        npy_source_sr=args.npy_source_sr,
        random_shift=True,
    )
    num_classes = 10 if (dataset.has_labels and not args.unconditional) else 0
    print(
        f"Loaded {len(dataset):,} clips on {device} | "
        f"conditioning={'digits 0-9' if num_classes == 10 else 'unconditional'}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    gen = PQMFGenerator(
        z_dim=args.z_dim,
        num_classes=num_classes,
        base_channels=args.base_channels,
    ).to(device)
    disc = CompositeDiscriminator(num_classes=num_classes).to(device)
    diff_aug = DiffAugment1d(max_shift=128, gain_range=0.12).to(device)
    ema = EMA(gen, decay=args.ema_decay)

    print(
        f"PQMF SNR: {gen.pqmf.reconstruction_snr_db():.1f} dB | "
        f"G params: {count_parameters(gen):,} | D params: {count_parameters(disc):,}"
    )

    # TTUR (Two Time-Scale Update Rule) with AdamW betas=(0.0, 0.9) for Hinge GAN
    opt_g = torch.optim.AdamW(
        gen.parameters(), lr=args.lr_g, betas=(0.0, 0.9), weight_decay=1e-4
    )
    opt_d = torch.optim.AdamW(
        disc.parameters(), lr=args.lr_d, betas=(0.0, 0.9), weight_decay=1e-4
    )

    use_amp = device.type == "cuda" and not args.no_amp
    scaler_g = torch.amp.GradScaler("cuda") if use_amp else None
    scaler_d = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["gen"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from '{args.resume}' at epoch {start_epoch}.")

    os.makedirs(args.out_dir, exist_ok=True)
    fixed_z = torch.randn(max(num_classes, 4), args.z_dim, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        gen.train()
        disc.train()
        diff_aug.train()

        d_loss_sum = 0.0
        g_loss_sum = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d}", leave=True)
        for real, labels in pbar:
            real = real.to(device, non_blocking=True)
            lbl = labels.to(device, non_blocking=True) if num_classes > 0 else None
            bsz = real.size(0)

            # ------------------ 1. Update Discriminator ------------------ #
            z = torch.randn(bsz, args.z_dim, device=device)
            opt_d.zero_grad(set_to_none=True)
            if use_amp and scaler_d is not None:
                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        fake_detached = gen(z, lbl)
                    d_real = disc(diff_aug(real), lbl)
                    d_fake = disc(diff_aug(fake_detached), lbl)
                    d_loss = hinge_discriminator_loss(d_real, d_fake)
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                with torch.no_grad():
                    fake_detached = gen(z, lbl)
                d_real = disc(diff_aug(real), lbl)
                d_fake = disc(diff_aug(fake_detached), lbl)
                d_loss = hinge_discriminator_loss(d_real, d_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0)
                opt_d.step()

            # ------------------ 2. Update Generator ---------------------- #
            # Freeze D params during G update (saves compute & avoids stale D grads)
            for p in disc.parameters():
                p.requires_grad_(False)

            z = torch.randn(bsz, args.z_dim, device=device)
            opt_g.zero_grad(set_to_none=True)
            if use_amp and scaler_g is not None:
                with torch.amp.autocast("cuda"):
                    fake = gen(z, lbl)  # NOTE: NOT detached! Gradients flow to G.
                    d_fake_for_g = disc(diff_aug(fake), lbl)
                    adv_loss = hinge_generator_loss(d_fake_for_g)
                    # Tiny DC penalty prevents constant offset drift in raw audio
                    dc_penalty = fake.mean(dim=-1).pow(2).mean()
                    g_loss = adv_loss + 0.1 * dc_penalty
                scaler_g.scale(g_loss).backward()
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                fake = gen(z, lbl)  # NOTE: NOT detached! Gradients flow to G.
                d_fake_for_g = disc(diff_aug(fake), lbl)
                adv_loss = hinge_generator_loss(d_fake_for_g)
                dc_penalty = fake.mean(dim=-1).pow(2).mean()
                g_loss = adv_loss + 0.1 * dc_penalty
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)
                opt_g.step()

            for p in disc.parameters():
                p.requires_grad_(True)

            ema.update(gen)

            d_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            steps += 1
            pbar.set_postfix(
                d_loss=f"{d_loss_sum / steps:.3f}",
                g_loss=f"{g_loss_sum / steps:.3f}",
            )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            export_sample_grid(
                ema.shadow,
                device=device,
                out_dir=os.path.join(args.out_dir, "samples"),
                epoch=epoch,
                sample_rate=args.sample_rate,
                fixed_z=fixed_z,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch:04d}.pt")
            latest_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
            payload = {
                "epoch": epoch,
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "ema": ema.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "config": {
                    "z_dim": args.z_dim,
                    "num_classes": num_classes,
                    "base_channels": args.base_channels,
                    "sample_rate": args.sample_rate,
                },
            }
            torch.save(payload, ckpt_path)
            torch.save(payload, latest_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PQMF-Snake Conditional AudioGAN")
    p.add_argument("--data-dir", type=str, default="audioMNIST", help="Folder of .wav/.npy or path to .npz cache")
    p.add_argument("--build-cache", type=str, default="", help="Optional path to save a preprocessed 16 kHz .npz cache")
    p.add_argument("--out-dir", type=str, default="runs/pqmf_snake", help="Output directory for checkpoints & WAVs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=256, help="Use 128-192 for CPU / low-VRAM GPUs")
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--npy-source-sr", type=int, default=48000, help="Original SR of .npy files (AudioMNIST=48000)")
    p.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--unconditional", action="store_true", help="Disable digit 0-9 conditioning")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true", help="Run fast self-check and exit")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
