"""Training script for Multi-Band PQMF + Snake-Activation Conditional Audio GAN.

Usage:
    # 1. Verify PQMF SNR, 8 kHz & 16 kHz shapes, AC/matching-aware losses, and R1 gradients:
    python -m pqmf_snake_gan.train --smoke-test

    # 2. Train on FSDD (native 8 kHz, 3k clips -> auto-adapts fsdd_16k.npz to 8 kHz / 8192 samples):
    python -m pqmf_snake_gan.train --data-dir data_cache/fsdd_16k.npz --out-dir runs/pqmf_fsdd_8k --sample-rate 8000 --epochs 230 --batch-size 32

    # 3. Train on full AudioMNIST (30k clips @ 16 kHz / 16384 samples):
    python -m pqmf_snake_gan.train --data-dir audioMNIST --build-cache data_cache/audiomnist_16k.npz --epochs 100 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io.wavfile import write as write_wav
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioMNISTDataset, build_dataset_cache
from .models import CompositeDiscriminator, PQMFGenerator, default_upsample_rates
from .modules import DiffAugment1d, EMA


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weighted_hinge_discriminator_loss(
    real_scores: List[torch.Tensor],
    fake_scores: List[torch.Tensor],
    weights: Sequence[float],
    wrong_scores: Optional[List[torch.Tensor]] = None,
    wrong_weight: float = 0.5,
) -> torch.Tensor:
    loss = torch.zeros((), device=real_scores[0].device)
    has_wrong = wrong_scores is not None and len(wrong_scores) == len(real_scores) and wrong_weight > 0
    fake_w = 1.0 - wrong_weight if has_wrong else 1.0
    for i, (r, f, w) in enumerate(zip(real_scores, fake_scores, weights)):
        head_loss = F.relu(1.0 - r).mean() + fake_w * F.relu(1.0 + f).mean()
        if has_wrong and wrong_scores is not None:
            head_loss = head_loss + wrong_weight * F.relu(1.0 + wrong_scores[i]).mean()
        loss = loss + w * head_loss
    return loss


def weighted_hinge_generator_loss(
    fake_scores: List[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    loss = torch.zeros((), device=fake_scores[0].device)
    for f, w in zip(fake_scores, weights):
        loss = loss - w * f.mean()
    return loss


def auxiliary_cls_loss(
    cls_logits: List[torch.Tensor],
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if not cls_logits:
        return torch.zeros((), device=labels.device)
    loss = torch.zeros((), device=labels.device)
    for logits in cls_logits:
        loss = loss + F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    return loss / len(cls_logits)


@torch.no_grad()
def auxiliary_cls_accuracy(cls_logits: List[torch.Tensor], labels: torch.Tensor) -> float:
    if not cls_logits:
        return 0.0
    avg_logits = sum(cls_logits) / len(cls_logits)
    preds = avg_logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def sample_wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    offsets = torch.randint(1, num_classes, labels.shape, device=labels.device)
    return (labels + offsets) % num_classes


def compute_r1_penalty(
    disc: CompositeDiscriminator,
    real: torch.Tensor,
    labels: Optional[torch.Tensor],
) -> torch.Tensor:
    """Lazy R1 gradient penalty on real audio to prevent D memorization on small datasets."""
    with torch.amp.autocast(device_type=real.device.type, enabled=False):
        real_f = real.detach().float().requires_grad_(True)
        out = disc(real_f, labels)
        total_score = sum(w * s.sum() for w, s in zip(disc.head_weights, out["scores"]))
        grad_real = torch.autograd.grad(
            outputs=total_score,
            inputs=real_f,
            create_graph=True,
            only_inputs=True,
        )[0]
        r1 = grad_real.pow(2).reshape(real_f.size(0), -1).sum(dim=1).mean()
    return r1


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


def resolve_audio_length(sample_rate: int, audio_length: int) -> int:
    if audio_length > 0:
        return audio_length
    return 8192 if sample_rate <= 8000 else 16384


def run_smoke_test(device: torch.device) -> None:
    """Verify PQMF SNR, 8 kHz & 16 kHz shapes, matching-aware/AC losses, R1, and G/D grads."""
    print(f"[Smoke Test] Device: {device}")
    for sr, length in [(8000, 8192), (16000, 16384)]:
        up_rates = default_upsample_rates(length, init_length=16, subbands=4)
        gen = PQMFGenerator(
            z_dim=128,
            num_classes=10,
            base_channels=192,
            upsample_rates=up_rates,
        ).to(device)
        disc = CompositeDiscriminator(sample_rate=sr, num_classes=10).to(device)
        diff_aug = DiffAugment1d(max_shift=64 if sr <= 8000 else 128).to(device)

        snr = gen.pqmf.reconstruction_snr_db(length=length)
        assert snr > 55.0, f"PQMF SNR unexpectedly low: {snr:.2f} dB"

        B = 4
        real = torch.randn(B, 1, gen.audio_length, device=device).clamp(-1.0, 1.0)
        labels = torch.randint(0, 10, (B,), device=device)
        wrong_labels = sample_wrong_labels(labels, 10)
        z = torch.randn(B, gen.z_dim, device=device)

        # 1. Discriminator forward & backward (+ wrong-label + AC + R1)
        fake = gen(z, labels)
        assert fake.shape == real.shape, f"Shape mismatch: {fake.shape} vs {real.shape}"

        d_real_out = disc(diff_aug(real), labels, wrong_labels=wrong_labels)
        d_fake_out = disc(diff_aug(fake.detach()), labels)
        d_adv = weighted_hinge_discriminator_loss(
            d_real_out["scores"],
            d_fake_out["scores"],
            disc.head_weights,
            wrong_scores=d_real_out["wrong_scores"],
        )
        d_cls = auxiliary_cls_loss(d_real_out["cls_logits"], labels, label_smoothing=0.05)
        r1 = compute_r1_penalty(disc, real, labels)
        d_loss = d_adv + d_cls + 0.5 * r1
        disc.zero_grad(set_to_none=True)
        d_loss.backward()
        d_grad_norm = sum(
            p.grad.norm().item() for p in disc.parameters() if p.grad is not None
        )
        assert d_grad_norm > 0.0 and np.isfinite(d_grad_norm), "Invalid Discriminator gradients!"

        # 2. Generator forward & backward
        for p in disc.parameters():
            p.requires_grad_(False)
        d_fake_for_g = disc(diff_aug(fake), labels)
        g_adv = weighted_hinge_generator_loss(d_fake_for_g["scores"], disc.head_weights)
        g_cls = auxiliary_cls_loss(d_fake_for_g["cls_logits"], labels)
        g_loss = g_adv + 0.5 * g_cls + 0.1 * fake.mean(dim=-1).pow(2).mean()
        gen.zero_grad(set_to_none=True)
        g_loss.backward()
        for p in disc.parameters():
            p.requires_grad_(True)

        g_grad_norm = sum(
            p.grad.norm().item() for p in gen.parameters() if p.grad is not None
        )
        assert g_grad_norm > 0.0 and np.isfinite(g_grad_norm), "Invalid Generator gradients!"

        print(
            f"[Smoke Test {sr} Hz / {length}] PASSED! PQMF SNR={snr:.1f} dB | "
            f"G={count_parameters(gen):,} D={count_parameters(disc):,} | "
            f"d_loss={d_loss.item():.3f} (grad={d_grad_norm:.2f}) | "
            f"g_loss={g_loss.item():.3f} (grad={g_grad_norm:.2f})"
        )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)

    if args.smoke_test:
        run_smoke_test(device)
        return

    audio_length = resolve_audio_length(args.sample_rate, args.audio_length)
    upsample_rates = default_upsample_rates(audio_length, init_length=16, subbands=4)

    if args.build_cache:
        print(f"Building {args.sample_rate} Hz cache from '{args.data_dir}' -> '{args.build_cache}' ...")
        t0 = time.time()
        build_dataset_cache(
            data_dir=args.data_dir,
            cache_path=args.build_cache,
            target_sr=args.sample_rate,
            audio_length=audio_length,
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
        audio_length=audio_length,
        npy_source_sr=args.npy_source_sr,
        random_shift=True,
        max_shift=256 if args.sample_rate <= 8000 else 512,
    )
    num_classes = 10 if (dataset.has_labels and not args.unconditional) else 0
    print(
        f"Loaded {len(dataset):,} clips on {device} | "
        f"{args.sample_rate} Hz ({audio_length} samples, up={upsample_rates}) | "
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
        upsample_rates=upsample_rates,
    ).to(device)
    disc = CompositeDiscriminator(
        sample_rate=args.sample_rate,
        num_classes=num_classes,
    ).to(device)
    diff_aug = DiffAugment1d(
        max_shift=64 if args.sample_rate <= 8000 else 128,
        gain_range=0.15,
        polarity_flip=True,
    ).to(device)
    ema = EMA(gen, decay=args.ema_decay)

    print(
        f"PQMF SNR: {gen.pqmf.reconstruction_snr_db(length=audio_length):.1f} dB | "
        f"G params: {count_parameters(gen):,} | D params: {count_parameters(disc):,}"
    )

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

    global_step = (start_epoch - 1) * len(loader)

    for epoch in range(start_epoch, args.epochs + 1):
        gen.train()
        disc.train()
        diff_aug.train()

        d_loss_sum = 0.0
        g_loss_sum = 0.0
        d_acc_sum = 0.0
        g_acc_sum = 0.0
        margin_sum = 0.0
        steps = 0

        # Smoothly warm up auxiliary classifier guidance on G while D learns digits
        if args.cls_warmup_epochs > 0:
            g_cls_scale = min(1.0, epoch / float(args.cls_warmup_epochs)) * args.cls_weight_g
        else:
            g_cls_scale = args.cls_weight_g

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d}", leave=True)
        for real, labels in pbar:
            global_step += 1
            real = real.to(device, non_blocking=True)
            lbl = labels.to(device, non_blocking=True) if num_classes > 0 else None
            wrong_lbl = sample_wrong_labels(lbl, num_classes) if lbl is not None else None
            bsz = real.size(0)

            # ------------------ 1. Update Discriminator ------------------ #
            z = torch.randn(bsz, args.z_dim, device=device)
            opt_d.zero_grad(set_to_none=True)
            if use_amp and scaler_d is not None:
                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        fake_detached = gen(z, lbl)
                    d_real_out = disc(diff_aug(real), lbl, wrong_labels=wrong_lbl)
                    d_fake_out = disc(diff_aug(fake_detached), lbl)
                    d_adv = weighted_hinge_discriminator_loss(
                        d_real_out["scores"],
                        d_fake_out["scores"],
                        disc.head_weights,
                        wrong_scores=d_real_out["wrong_scores"],
                        wrong_weight=args.wrong_label_weight,
                    )
                    d_cls = (
                        auxiliary_cls_loss(d_real_out["cls_logits"], lbl, label_smoothing=0.05)
                        if lbl is not None
                        else torch.zeros((), device=device)
                    )
                    d_loss = d_adv + args.cls_weight_d * d_cls
                scaler_d.scale(d_loss).backward()

                # Lazy R1 gradient penalty in FP32 every r1_every steps
                if args.r1_gamma > 0 and global_step % args.r1_every == 0:
                    r1_loss = 0.5 * args.r1_gamma * args.r1_every * compute_r1_penalty(disc, real, lbl)
                    scaler_d.scale(r1_loss).backward()

                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                with torch.no_grad():
                    fake_detached = gen(z, lbl)
                d_real_out = disc(diff_aug(real), lbl, wrong_labels=wrong_lbl)
                d_fake_out = disc(diff_aug(fake_detached), lbl)
                d_adv = weighted_hinge_discriminator_loss(
                    d_real_out["scores"],
                    d_fake_out["scores"],
                    disc.head_weights,
                    wrong_scores=d_real_out["wrong_scores"],
                    wrong_weight=args.wrong_label_weight,
                )
                d_cls = (
                    auxiliary_cls_loss(d_real_out["cls_logits"], lbl, label_smoothing=0.05)
                    if lbl is not None
                    else torch.zeros((), device=device)
                )
                d_loss = d_adv + args.cls_weight_d * d_cls
                d_loss.backward()

                if args.r1_gamma > 0 and global_step % args.r1_every == 0:
                    r1_loss = 0.5 * args.r1_gamma * args.r1_every * compute_r1_penalty(disc, real, lbl)
                    r1_loss.backward()

                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0)
                opt_d.step()

            # ------------------ 2. Update Generator ---------------------- #
            for p in disc.parameters():
                p.requires_grad_(False)

            z = torch.randn(bsz, args.z_dim, device=device)
            opt_g.zero_grad(set_to_none=True)
            if use_amp and scaler_g is not None:
                with torch.amp.autocast("cuda"):
                    fake = gen(z, lbl)
                    d_fake_for_g = disc(diff_aug(fake), lbl)
                    adv_loss = weighted_hinge_generator_loss(d_fake_for_g["scores"], disc.head_weights)
                    g_cls = (
                        auxiliary_cls_loss(d_fake_for_g["cls_logits"], lbl)
                        if lbl is not None
                        else torch.zeros((), device=device)
                    )
                    dc_penalty = fake.mean(dim=-1).pow(2).mean()
                    g_loss = adv_loss + g_cls_scale * g_cls + 0.1 * dc_penalty
                scaler_g.scale(g_loss).backward()
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                fake = gen(z, lbl)
                d_fake_for_g = disc(diff_aug(fake), lbl)
                adv_loss = weighted_hinge_generator_loss(d_fake_for_g["scores"], disc.head_weights)
                g_cls = (
                    auxiliary_cls_loss(d_fake_for_g["cls_logits"], lbl)
                    if lbl is not None
                    else torch.zeros((), device=device)
                )
                dc_penalty = fake.mean(dim=-1).pow(2).mean()
                g_loss = adv_loss + g_cls_scale * g_cls + 0.1 * dc_penalty
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)
                opt_g.step()

            for p in disc.parameters():
                p.requires_grad_(True)

            ema.update(gen)

            d_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            if lbl is not None:
                d_acc_sum += auxiliary_cls_accuracy(d_real_out["cls_logits"], lbl)
                g_acc_sum += auxiliary_cls_accuracy(d_fake_for_g["cls_logits"], lbl)
                if d_real_out["wrong_scores"]:
                    m = sum(
                        w * (r - rw).mean().item()
                        for w, r, rw in zip(disc.head_weights, d_real_out["scores"], d_real_out["wrong_scores"])
                    )
                    margin_sum += float(m)
            steps += 1
            postfix = {
                "d": f"{d_loss_sum / steps:.3f}",
                "g": f"{g_loss_sum / steps:.3f}",
            }
            if lbl is not None:
                postfix["d_acc"] = f"{100.0 * d_acc_sum / steps:.1f}%"
                postfix["g_acc"] = f"{100.0 * g_acc_sum / steps:.1f}%"
                postfix["gap"] = f"{margin_sum / steps:+.2f}"
            pbar.set_postfix(**postfix)

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
                    "arch_version": 2,
                    "z_dim": args.z_dim,
                    "num_classes": num_classes,
                    "base_channels": args.base_channels,
                    "sample_rate": args.sample_rate,
                    "audio_length": audio_length,
                    "subbands": 4,
                    "upsample_rates": list(upsample_rates),
                },
            }
            torch.save(payload, ckpt_path)
            torch.save(payload, latest_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PQMF-Snake Conditional AudioGAN (v2)")
    p.add_argument("--data-dir", type=str, default="audioMNIST", help="Folder of .wav/.npy or path to .npz cache")
    p.add_argument("--build-cache", type=str, default="", help="Optional path to save a preprocessed .npz cache")
    p.add_argument("--out-dir", type=str, default="runs/pqmf_snake_v2", help="Output directory for checkpoints & WAVs")
    p.add_argument("--epochs", type=int, default=230)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=256, help="Use 128-192 for CPU / low-VRAM GPUs")
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--cls-weight-d", type=float, default=1.0, help="Auxiliary digit CE weight on D (real)")
    p.add_argument("--cls-weight-g", type=float, default=0.5, help="Auxiliary digit CE weight on G (fake)")
    p.add_argument("--cls-warmup-epochs", type=int, default=5, help="Warmup epochs for G auxiliary CE loss")
    p.add_argument("--wrong-label-weight", type=float, default=0.5, help="Fraction of fake D hinge assigned to (real, wrong_y)")
    p.add_argument("--r1-gamma", type=float, default=1.0, help="Lazy R1 gradient penalty weight (0 disables)")
    p.add_argument("--r1-every", type=int, default=16, help="Apply R1 every N steps")
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--sample-rate", type=int, default=16000, help="Use 8000 for native FSDD or 16000 for AudioMNIST")
    p.add_argument("--audio-length", type=int, default=0, help="0 = auto (8192 @ 8kHz, 16384 @ 16kHz)")
    p.add_argument("--npy-source-sr", type=int, default=48000, help="Original SR of .npy files (AudioMNIST=48000)")
    p.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--unconditional", action="store_true", help="Disable digit 0-9 conditioning")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true", help="Run fast self-check and exit")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
