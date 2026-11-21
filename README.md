# AudioGAN — Multi-Band PQMF + Snake-Activation Conditional Audio GAN

A **100% time-domain** (zero-spectrogram) conditional GAN for generating 16 kHz spoken digits on [AudioMNIST](https://github.com/soerenab/AudioMNIST).

## Architecture Highlights
- **4-Band PQMF Filterbank** ([pqmf.py](pqmf_snake_gan/pqmf.py)): Generates 4 time-domain subbands (`4 × 4,096` @ 4 kHz) and synthesizes crisp 16 kHz waveforms (`16,384` samples) in a single pass — **4× faster** than full-rate 1D GANs with >70 dB reconstruction SNR.
- **SnakeBeta Periodic Activations** ([modules.py](pqmf_snake_gan/modules.py)): $x + \frac{1}{\beta}\sin^2(\alpha x)$ gives 1D convolutions a built-in harmonic inductive bias for speech.
- **Multi-Period + Multi-Band Time-Domain Discriminators** ([models.py](pqmf_snake_gan/models.py)): Captures pitch periodicity (periods `2, 3, 5, 7, 11`), cross-band coherence, and word envelope purely in the time domain, with **Projection Conditioning** for digits `0–9`.
- **Stable Training** ([train.py](pqmf_snake_gan/train.py)): Hinge GAN loss, Spectral Normalization, TTUR (`AdamW`, `betas=(0.0, 0.9)`), `DiffAugment1d`, DC-drift regularization, AMP, and EMA generator weights.

## Quickstart

```bash
# 1. Verify PQMF reconstruction SNR & G/D gradient flow
python -m pqmf_snake_gan.train --smoke-test

# 2. (Optional) Cache AudioMNIST (.wav or .npy) once at 16 kHz and train
python -m pqmf_snake_gan.train --data-dir audioMNIST --build-cache data_cache/audiomnist_16k.npz --epochs 50 --batch-size 64

# 3. Generate spoken digits 0-9 using EMA weights
python -m pqmf_snake_gan.generate --checkpoint runs/pqmf_snake/checkpoint_latest.pt --digits 0 1 2 3 4 5 6 7 8 9
```

For interactive exploration and inline audio playback, open [PQMF_Snake_AudioGAN.ipynb](PQMF_Snake_AudioGAN.ipynb).
