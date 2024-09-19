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
batch_size = 8
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
num_samples = 100
data = generate_circular_signal(radius, num_samples)
data_tensor = torch.tensor(data, dtype=torch.float32)

# Create DataLoader
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch
# Define the Generator for GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)  # Increase hidden size for better learning
        self.fc2 = nn.Linear(128, 64)  # Increase hidden size for better learning
        self.lstm1 = nn.LSTM(64, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, output_size, batch_first=True)        
        

    def forward(self, z):
        z = self.fc1(z).unsqueeze(1).repeat(1, 360, 1)  # Repeat for sequence length
        z = self.fc2(z)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        return torch.tanh(z)  # Shape: (batch_size, 360, output_size)

# Define the Discriminator for GAN
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, 32)
        #self.fc1 = nn.Linear(32, 16)
        #self.fc2 = nn.Linear(16, 1)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(x[:, -1])  # Use the last output
        #x = torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
        x = torch.sigmoid(self.fc2(torch.relu(x)))
        return x
   

# Hyperparameters
input_size = 2   # (x,y) coordinates
latent_dim = 64
output_size = input_size


# Initialize models and optimizers
generator = Generator(latent_dim=latent_dim, output_size=2)
discriminator = Discriminator(input_size=2)

optimizer_generator = optim.Adam(generator.parameters(), lr=0.0001)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.00001)

# Loss functions
criterion_reconstruction = nn.MSELoss()
criterion_gan = nn.BCELoss()
d_losses = []
g_losses = []
re_losses = []
# Training loop
for epoch in range(num_epochs):
    batch_count = 0
    for real_signals in dataloader:
        real_signals = real_signals[0]  # Get the batch of real signals

        # ============================
        # Train the GAN Discriminator
        # ============================

        # Generate random latent vectors
        batch_size_current = real_signals.size(0)
        z_random = torch.randn(batch_size_current, latent_dim)  # Random noise for generator

        # Generate fake signals
        fake_signals = generator(z_random)

        # Labels for real and fake signals
        real_labels = torch.ones(batch_size_current, 1)
        fake_labels = torch.zeros(batch_size_current, 1)

        # Discriminator loss on real and fake signals
        output_real = discriminator(real_signals)
        output_fake = discriminator(fake_signals.detach())

        loss_real = criterion_gan(output_real, real_labels)
        loss_fake = criterion_gan(output_fake, fake_labels)
        
        loss_discriminator_gan = (loss_real + loss_fake)/2.0
        
        optimizer_discriminator.zero_grad()
        loss_discriminator_gan.backward()
        optimizer_discriminator.step()

        # ============================
        # Train the GAN Generator
        # ============================
                
        # Generate fake signals again (for generator's training)
        fake_signals = generator(z_random)

        # Generator loss (we want the discriminator to think these are real)
        output_fake_for_generator = discriminator(fake_signals)
        loss_generator_gan = criterion_gan(output_fake_for_generator, real_labels)
        
        optimizer_generator.zero_grad()
        loss_generator_gan.backward()
        optimizer_discriminator.step()
        


        if batch_count % 15 == 0:  # Print every 100 epochs
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Discriminator Loss: {loss_discriminator_gan.item():.4f}, '
                  f'Generator Loss: {loss_generator_gan.item():.4f} ')
            d_losses.append(loss_discriminator_gan.item())
            g_losses.append(loss_generator_gan.item())            
            
        batch_count += 1
        
# Plot the losses
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Losses')
plt.title('VAE-GAN Training Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()       
# After training: Generate a new circular signal using the generator

with torch.no_grad():
    num_samples_to_generate = 5   # Number of samples you want to generate 
    z_random_sampled = torch.randn(num_samples_to_generate, latent_dim) 
    generated_signals_sampled = generator(z_random_sampled)   # Shape: (num_samples_to_generate ,360 ,2 )

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
        
        