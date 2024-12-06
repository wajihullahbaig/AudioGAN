import torch.optim as optim
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic EEG signals
def generate_synthetic_eeg(num_samples=1000, signal_length=1024, noise_level=0.1):
    # Create an empty array to hold the EEG signals
    eeg_signals = np.zeros((num_samples, signal_length))
    
    # Generate synthetic EEG signals
    for i in range(num_samples):
        # Create a base signal (sine wave + random noise)
        time = np.linspace(0, 1, signal_length)
        frequency = np.random.uniform(5, 30)  # Random frequency between 5 and 30 Hz
        sine_wave = np.sin(2 * np.pi * frequency * time)  # Sine wave
        noise = noise_level * np.random.randn(signal_length)  # Gaussian noise
        eeg_signals[i] = sine_wave + noise  # Combine sine wave and noise
    
    # Normalize the signals to be between -1 and 1
    eeg_signals = (eeg_signals - np.min(eeg_signals)) / (np.max(eeg_signals) - np.min(eeg_signals)) * 2 - 1
    # Convert to PyTorch tensor with float32 type
    eeg_signals = torch.tensor(eeg_signals, dtype=torch.float32)
    return eeg_signals

# Generate a synthetic EEG dataset
num_samples = 1000
signal_length = 1024
synthetic_eeg_data = generate_synthetic_eeg(num_samples, signal_length)

# Plot a few samples to visualize
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(synthetic_eeg_data[i], label=f'Sample {i+1}')
plt.title('Synthetic EEG Signals')
plt.xlabel('Sample Point')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Assuming 'eeg_data' is a list of 1D EEG signals, each 1024 samples long
dataset = EEGDataset(synthetic_eeg_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=8, out_dim=1024):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
class Discriminator(nn.Module):
    def __init__(self, in_dim=1024):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# Hyperparameters
latent_dim = 32
lr = 0.0002
b1 = 0.5
b2 = 0.999
num_epochs = 50
# Initialize generator and discriminator
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
#optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
#optimizer_G = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.9)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=lr, momentum=0.9)
# Lists to store losses
g_losses = []
d_losses = []

for epoch in range(num_epochs):
    for i, real_samples in enumerate(dataloader):
        # Train Discriminator
        real_samples = real_samples
        z = torch.randn(real_samples.size(0), latent_dim)
        fake_samples = generator(z)
        
        real_output = discriminator(real_samples)
        fake_output = discriminator(fake_samples)
        
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        d_loss = 0.5 * (real_loss + fake_loss)
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        z = torch.randn(real_samples.size(0), latent_dim)
        fake_samples = generator(z)
        fake_output = discriminator(fake_samples)
        
        g_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        # Store losses
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        

# Plot the losses
plt.figure(figsize=(12, 6))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.title('GAN Training Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()        
        