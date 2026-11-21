# AudioGAN — Multi-Band PQMF + Snake-Activation Conditional Audio GAN

A **pure time-domain generator** (zero-spectrogram inference) conditional GAN for generating spoken digits (`0–9`) on [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset) (8 kHz) or [AudioMNIST](https://github.com/soerenab/AudioMNIST) (16 kHz).

## Architecture Diagram

```mermaid
flowchart TD
    %% UML-style palette (high contrast in both GitHub light & dark modes)
    classDef inputNode fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#f8fafc
    classDef condNode fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    classDef genNode fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5
    classDef attnNode fill:#0f766e,stroke:#2dd4bf,stroke-width:2px,color:#f0fdfa
    classDef pqmfNode fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#fffbeb
    classDef waveNode fill:#1f2937,stroke:#f59e0b,stroke-width:2.5px,color:#fef3c7
    classDef augNode fill:#4c1d95,stroke:#c084fc,stroke-width:2px,color:#faf5ff
    classDef discNode fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fef2f2
    classDef lossNode fill:#831843,stroke:#f472b6,stroke-width:2px,color:#fdf2f8

    subgraph Inputs["«package» Conditioning Inputs"]
        direction TB
        Z(["«input»\nLatent Noise z\n(B, 128)"]):::inputNode
        Y(["«input»\nDigit Label y in 0..9\n(B,)"]):::inputNode
        EMB["«embedding»\nclass_emb + cond_mlp\n(B, 64)"]:::condNode
        STYLE{{"«style vector»\nconcat(z, cond)\n(B, 192)"}}:::condNode
        Y --> EMB
        Z --> STYLE
        EMB --> STYLE
    end

    subgraph GEN["«component» 100% Pure Time-Domain Generator (PQMFGenerator — Inference & Training)"]
        direction TB
        PROJ["«projection»\ninput_proj\n(B, 256, 16)"]:::genNode
        TBLUE["«blueprint»\nclass_time_emb[y]\n(B, 256, 16)"]:::condNode
        ADD((" + ")):::genNode
        ATTN1[["«self-attention»\nTemporalSelfAttention1d\n(L = 16, SAGAN gamma)"]]:::attnNode
        S0["«stage 0»\n4x Linear + Conv1d -> MRFSnakeBlock + FiLM1d\n(256 x 64)"]:::genNode
        ATTN2[["«self-attention»\nTemporalSelfAttention1d\n(L = 64, SAGAN gamma)"]]:::attnNode
        S1["«stage 1»\n4x Linear + Conv1d -> MRFSnakeBlock + StyleGate1d\n(192 x 256)"]:::genNode
        S2["«stage 2»\n4x Linear + Conv1d -> MRFSnakeBlock + StyleGate1d\n(128 x 1024)"]:::genNode
        S3["«stage 3»\n2x/4x Linear + Conv1d -> MRFSnakeBlock + StyleGate1d\n(64 x 2048 or 4096)"]:::genNode
        OUTC["«head»\nSnakeBeta + Conv1d + tanh\n4 Subbands"]:::genNode
        PQMF_S[["«filterbank»\nFrozen 4-Band PQMF.synthesis\n>64 dB SNR, Zero FFT"]]:::pqmfNode
        FAKE(["«artifact»\nGenerated 1D Waveform\n8,192 @ 8 kHz or 16,384 @ 16 kHz"]):::waveNode

        STYLE --> PROJ
        Y --> TBLUE
        PROJ --> ADD
        TBLUE --> ADD
        ADD --> ATTN1
        ATTN1 --> S0
        S0 --> ATTN2
        ATTN2 --> S1
        S1 --> S2
        S2 --> S3
        S3 --> OUTC
        OUTC --> PQMF_S
        PQMF_S --> FAKE
    end

    subgraph TRAIN_D["«component» Training-Only Augmentation, CompositeDiscriminator & Losses"]
        direction TB
        REAL(["«dataset»\nReal 1D Waveform\nFSDD 8 kHz / AudioMNIST 16 kHz"]):::waveNode
        AUG[["«augment»\nDiffAugment1d\nGain + Polarity Flip + Zero-Pad Shift"]]:::augNode

        MRD["«discriminator — w = 0.50»\nMulti-Resolution STFT (3 Scales)\n2D Conv -> 2x4 Temporal Word Pool"]:::discNode
        MBD["«discriminator — w = 0.25»\nMulti-Band Time-Domain\n4-Band PQMF + Rectified Env -> 6-Slot Pool"]:::discNode
        MPD["«discriminator — w = 0.25»\nMulti-Period (2, 3, 5, 7)\n2D Period-Folded Conv"]:::discNode

        HINGE{{"«loss»\nWeighted Matching-Aware Hinge\n(real, y) vs (fake, y) vs (real, wrong_y)"}}:::lossNode
        ACL{{"«loss»\nAuxiliary 10-Class Digit Classifier\nLabel-Smoothed CE (d_acc / g_acc)"}}:::lossNode
        R1{{"«regularizer»\nLazy R1 Gradient Penalty\nEvery 16 Steps"}}:::lossNode

        REAL --> AUG
        FAKE --> AUG
        AUG --> MRD
        AUG --> MBD
        AUG --> MPD
        MRD --> HINGE
        MBD --> HINGE
        MPD --> HINGE
        MRD --> ACL
        REAL --> R1
    end

    STYLE -. "«modulate» FiLM1d (Stage 0)" .-> S0
    STYLE -. "«gate» Silence-Preserving StyleGate1d (Stages 1–3)" .-> S2

    style Inputs fill:#0f172a,stroke:#60a5fa,stroke-width:1.5px,color:#e2e8f0
    style GEN fill:#022c22,stroke:#34d399,stroke-width:2px,color:#d1fae5
    style TRAIN_D fill:#1c1917,stroke:#f87171,stroke-width:1.5px,stroke-dasharray:6 4,color:#fee2e2
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

## Design Heritage & Building Blocks

Most papers using PQMF, `SnakeBeta`, MPD, and MRD are **neural vocoders** (paired Mel-spectrogram → waveform reconstruction). This project adapts those modern vocoder inductive biases into a **single-stage, direct `(noise z, class label) → raw waveform` GAN** (zero spectrograms at inference, no separate vocoder stage):

| Block / Technique                                           | Origin / Inspiration                                                                                                                                                                                                              | How It Is Used & Adapted Here                                                                                                                                                                                                            |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1D Waveform GAN + `PhaseShuffle1d`**                      | [WaveGAN](https://arxiv.org/abs/1802.04208) (Donahue et al., 2019)                                                                                                                                                                | Core single-stage `(z, digit) → waveform` setting; `PhaseShuffle1d` prevents the 1D time-domain discriminators from exploiting exact sample-index alignment.                                                                             |
| **Frozen Cosine-Modulated PQMF Bank**                       | [Multi-Band MelGAN](https://arxiv.org/abs/2005.05106) (Yang et al., 2020), [RAVE](https://arxiv.org/abs/2111.05011) (Caillon & Esling, 2021), Creusere & Mitra (1995)                                                             | Generator predicts 4 subbands at `1/4` sample rate and synthesizes full-band audio with a frozen >64 dB SNR filterbank ([pqmf.py](pqmf_snake_gan/pqmf.py)), making synthesis ~4× faster without FFTs.                                    |
| **MRF Dilated Residual Blocks & MPD**                       | [HiFi-GAN](https://arxiv.org/abs/2010.05646) (Kong et al., 2020)                                                                                                                                                                  | Multi-receptive-field dilated Conv1d stacks (`dilations = 1, 3, 5`) model overlapping harmonic/phoneme scales; `MultiPeriodDiscriminator` (`periods = 2, 3, 5, 7`, weight `0.25`) enforces clean voiced pitch periodicity.               |
| **`SnakeBeta` & Multi-Resolution STFT Discriminator (MRD)** | [BigVGAN](https://arxiv.org/abs/2206.04658) (Lee et al., 2023), [Snake](https://arxiv.org/abs/2006.08195) (Ziyin et al., 2020), [UnivNet](https://arxiv.org/abs/2106.07889) (Jang et al., 2021)                                   | Tightly clamped `SnakeBeta` gives 1D convolutions a periodic inductive bias without excessive subband aliasing; training-only 3-scale `MRD` (weight `0.50`) exposes formants and fricatives while keeping inference 100% time-domain.    |
| **Zero-Init 1D Self-Attention (`gamma`)**                   | [SAGAN](https://arxiv.org/abs/1805.08318) (Zhang et al., 2019)                                                                                                                                                                    | `TemporalSelfAttention1d` at low temporal lengths (`16` and `64`) starts as identity and learns global cross-syllable coherence (e.g., _ze-ro_, _se-ven_) at negligible cost.                                                            |
| **Per-Class Temporal Blueprints & `StyleGate1d`**           | Custom adaptation for raw-audio cGANs                                                                                                                                                                                             | `class_time_emb[y]` supplies each digit with a learned temporal skeleton; `FiLM1d` is restricted to stage 0 (`L = 64`), while high-rate stages use multiplicative-only `StyleGate1d` (`0 * (1 + g) = 0`) so silence stays strictly zero. |
| **Projection + Matching-Aware + AC-GAN Losses**             | [Projection Discriminator](https://arxiv.org/abs/1802.05637) (Miyato & Koyama, 2018), [Matching-Aware GAN](https://arxiv.org/abs/1605.05396) (Reed et al., 2016), [AC-GAN](https://arxiv.org/abs/1610.09585) (Odena et al., 2017) | Temporal-preserving word pooling (`2×4` in MRD, `6`-slot in MBD) feeds un-attenuated class projection, single-pass `(real, wrong_digit)` rejection, and an auxiliary 10-class digit classifier to prevent class collapse.                |
| **DiffAugment & Lazy R1 Penalty**                           | [DiffAugment](https://arxiv.org/abs/2006.10738) (Zhao et al., 2020), [StyleGAN2](https://arxiv.org/abs/1912.04958) (Karras et al., 2020)                                                                                          | Time-domain gain jitter, polarity flip, and zero-padded shift + lazy R1 regularization (every 16 steps) prevent discriminator memorization on small datasets like FSDD (3,000 clips).                                                    |

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
