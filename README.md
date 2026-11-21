# AudioGAN — Multi-Band PQMF + Snake-Activation Conditional Audio GAN

A **pure time-domain generator** (zero-spectrogram inference) conditional GAN for generating spoken digits (`0–9`) on [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset) (8 kHz) or [AudioMNIST](https://github.com/soerenab/AudioMNIST) (16 kHz).

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Inputs["Conditioning Inputs"]
        Z["Latent Noise z (B, 128)"]
        Y["Digit Label y in 0..9"]
        Y --> EMB["class_emb + cond_mlp (B, 64)"]
        Z --> STYLE["style = concat(z, cond) (B, 192)"]
        EMB --> STYLE
    end

    subgraph GEN["100% Pure Time-Domain Generator (PQMFGenerator)"]
        STYLE --> PROJ["input_proj -> (B, 256, 16)"]
        Y --> TBLUE["class_time_emb[y] (B, 256, 16)"]
        PROJ --> ADD["Add Temporal Blueprint"]
        TBLUE --> ADD
        ADD --> ATTN1["TemporalSelfAttention1d (L = 16)"]
        ATTN1 --> S0["Stage 0: 4x Linear + Conv1d -> MRFSnakeBlock + FiLM1d -> TemporalSelfAttention1d (256 x 64)"]
        S0 --> S1["Stage 1: 4x Linear + Conv1d -> MRFSnakeBlock + StyleGate1d (192 x 256)"]
        S1 --> S2["Stage 2: 4x Linear + Conv1d -> MRFSnakeBlock + StyleGate1d (128 x 1024)"]
        S2 --> S3["Stage 3: 2x or 4x Linear + Conv1d -> MRFSnakeBlock + StyleGate1d (64 x 2048 or 4096)"]
        S3 --> OUTC["SnakeBeta + Conv1d + tanh -> 4 Subbands"]
        OUTC --> PQMF_S["Frozen 4-Band PQMF.synthesis (>64 dB SNR)"]
        PQMF_S --> FAKE["Generated 1D Waveform (8,192 @ 8 kHz or 16,384 @ 16 kHz)"]
    end

    subgraph TRAIN_D["Training-Only Augmentation & CompositeDiscriminator"]
        REAL["Real 1D Waveform"] --> AUG["DiffAugment1d (Gain + Polarity Flip + Zero-Pad Shift)"]
        FAKE --> AUG

        AUG --> MRD["Multi-Resolution STFT Discriminator (3 Scales, Weight 0.50)\n2D Conv -> 2x4 Temporal Word Pool"]
        AUG --> MBD["Multi-Band Time-Domain Discriminator (Weight 0.25)\n4-Band PQMF + Rectified Envelope -> 6-Slot Word Pool"]
        AUG --> MPD["Multi-Period Discriminator (Periods 2, 3, 5, 7, Weight 0.25)\n2D Period-Folded Conv"]

        MRD --> HINGE["Weighted Matching-Aware Hinge Loss\n(real, y) vs (fake, y) vs (real, wrong_y)"]
        MBD --> HINGE
        MPD --> HINGE
        MRD --> ACL["Auxiliary 10-Class Digit Classifier\nCross-Entropy Loss (d_acc / g_acc)"]
        REAL --> R1["Lazy R1 Gradient Penalty (Every 16 Steps)"]
    end

    STYLE -. "Semantic FiLM (Stage 0) & Silence-Preserving StyleGate (Stages 1-3)" .-> S0
```

## Architecture Highlights
- **100% Time-Domain PQMF-Snake Generator** ([models.py](pqmf_snake_gan/models.py), [pqmf.py](pqmf_snake_gan/pqmf.py)):
  - Per-class learned temporal blueprints + lightweight 1D self-attention (`16 -> 64`) for multi-syllable word planning.
  - Linear upsampling + wider late channels + `SnakeBeta` periodic activations + silence-preserving multiplicative `StyleGate1d` at high rates.
  - Frozen 4-band cosine-modulated PQMF synthesis (>64 dB SNR) producing 8 kHz (`8,192` samples) or 16 kHz (`16,384` samples) waveforms directly.
- **Training-Only Multi-Resolution STFT + Multi-Band + Multi-Period Discriminator** ([models.py](pqmf_snake_gan/models.py)):
  - **Multi-Resolution STFT 2D Discriminator (MRD, weight 0.50)** with temporal-preserving word pooling and an **Auxiliary 10-class Digit Classifier** to enforce formants, fricatives, and phoneme order.
  - **Multi-Band Time-Domain Discriminator (MBD, weight 0.25)** on 4 PQMF subbands + rectified amplitude envelope.
  - **Multi-Period Discriminator (MPD, weight 0.25)** on periods `2, 3, 5, 7` for clean pitch harmonics.
- **Anti-Collapse Conditioning & Stability** ([train.py](pqmf_snake_gan/train.py)):
  - Single-pass **Matching-Aware Hinge Loss** (penalizing `(real, wrong_digit)`), **Auxiliary Digit Cross-Entropy**, un-attenuated Projection Conditioning, lazy **R1 gradient penalty**, zero-padded `DiffAugment1d` (with polarity flip), AMP, and EMA generator weights.

## Quickstart

```bash
# 1. Verify PQMF SNR, 8 kHz & 16 kHz shapes, AC/matching-aware losses, and R1 gradients
python -m pqmf_snake_gan.train --smoke-test

# 2a. Train on FSDD (native 8 kHz, 3k clips; auto-adapts fsdd_16k.npz to 8 kHz / 8,192 samples)
python -m pqmf_snake_gan.train --data-dir data_cache/fsdd_16k.npz --out-dir runs/pqmf_fsdd_8k_v2 --sample-rate 8000 --epochs 350 --batch-size 32

# 2b. Or train on full AudioMNIST (30k clips @ 16 kHz / 16,384 samples)
python -m pqmf_snake_gan.train --data-dir audioMNIST --build-cache data_cache/audiomnist_16k.npz --out-dir runs/pqmf_audiomnist_16k --sample-rate 16000 --epochs 100 --batch-size 64

# 3. Generate spoken digits 0-9 using EMA weights
python -m pqmf_snake_gan.generate --checkpoint runs/pqmf_fsdd_8k_v2/checkpoint_latest.pt --digits 0 1 2 3 4 5 6 7 8 9
```

For interactive exploration and inline audio playback, open [PQMF_Snake_AudioGAN.ipynb](PQMF_Snake_AudioGAN.ipynb).
