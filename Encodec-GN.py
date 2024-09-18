# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:37:17 2024

@author: Acer
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:15:26 2024

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate a circular signal (as previously defined)
def generate_circular_signal(radius, num_samples):
    signals = []
    for _ in range(num_samples):
        signal = [[radius * np.cos(np.deg2rad(theta)), radius * np.sin(np.deg2rad(theta))] for theta in range(360)]
        signals.append(signal)
    return np.array(signals)

# Parameters
radius = 1.0
num_samples = 1  # For visualization, we only need one sample

# Generate dataset
data = generate_circular_signal(radius, num_samples)

# Extract x and y coordinates for plotting
x_values = data[0][:, 0]  # x-coordinates
y_values = data[0][:, 1]  # y-coordinates

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(x_values, y_values, label='Circular Signal', color='blue')
plt.title('Circular Signal')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
plt.grid()
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.legend()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
num_epochs = 1000
batch_size = 64
learning_rate = 0.0001

# Create a synthetic circular signal dataset
def generate_circular_signal(radius, num_samples):
    signals = []
    for _ in range(num_samples):
        signal = [[radius * np.cos(np.deg2rad(theta)), radius * np.sin(np.deg2rad(theta))] for theta in range(360)]
        signals.append(signal)
    return np.array(signals)

# Generate dataset
radius = 1.0
num_samples = 1000
data = generate_circular_signal(radius, num_samples)
data_tensor = torch.tensor(data, dtype=torch.float32)

# Create DataLoader
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, latent_dim)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # Use the last output
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64)  # Increase hidden size for better learning
        self.lstm1 = nn.LSTM(64, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, output_size, batch_first=True)
        
    def forward(self, z):
        z = self.fc(z).unsqueeze(1).repeat(1, 360, 1)  # Repeat for sequence length
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        return z  # Shape: (batch_size, 360, output_size)



# Hyperparameters
input_size = 2   # (x,y) coordinates
latent_dim = 16
output_size = input_size


# Initialize models and optimizers
encoder = Encoder(input_size=2, latent_dim=16)
decoder = Decoder(latent_dim=16, output_size=2)

optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.00001)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.00001)

# Loss functions
criterion_reconstruction = nn.MSELoss()
criterion_gan = nn.BCELoss()
re_losses = []
# Training loop
for epoch in range(num_epochs):
    batch_count = 0
    for real_signals in dataloader:
        real_signals = real_signals[0]  # Get the batch of real signals

    
        # ============================
        # Train the VAE Encoder
        # ============================
        
        # Latent space representation from encoder for reconstruction loss
        z_encoded = encoder(real_signals)
        
        optimizer_encoder.zero_grad()  
        optimizer_encoder.step()          
        
        optimizer_decoder.zero_grad()        
        # Reconstruct signals from latent space representation
        reconstructed_signals = decoder(z_encoded)

        # Reconstruction loss (VAE part)
        loss_reconstruction = criterion_reconstruction(reconstructed_signals, real_signals)
        
        loss_reconstruction.backward()
        optimizer_decoder.step()
        
        if batch_count % 15 == 0:  # Print every 100 epochs
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Reconstruction Loss: {loss_reconstruction.item():.4f}, ')
                  
            re_losses.append(loss_reconstruction.item())            
        batch_count += 1
        
# Plot the losses
plt.figure(figsize=(12, 6))
plt.plot(re_losses, label='Reconstruction Loss')
plt.title('VAE-GAN Training Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()       
# After training: Generate a new circular signal using the generator

with torch.no_grad():
    num_samples_to_generate = 5   # Number of samples you want to generate 
    z_random_sampled = torch.randn(num_samples_to_generate, 16) 
    generated_signals_sampled = decoder(z_random_sampled)   # Shape: (num_samples_to_generate ,360 ,2 )

# Plotting generated signals

plt.figure(figsize=(10 ,10))
for i in range(num_samples_to_generate):
    plt.subplot(3 ,2 ,i + 1)   # Adjust subplot grid as needed 
    plt.plot(generated_signals_sampled[i][: ,0].cpu().numpy() ,generated_signals_sampled[i][: ,1].cpu().numpy())
    plt.title(f'Generated Circular Signal {i + 1}')
    plt.xlim(-1.5 ,1.5 )
    plt.ylim(-1.5 ,1.5 )
    plt.gca().set_aspect('equal' ,adjustable='box')
    plt.grid()
plt.tight_layout()
plt.show()        
        
        