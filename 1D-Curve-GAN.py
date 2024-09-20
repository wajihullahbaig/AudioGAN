# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:13:28 2024

@author: Acer
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Function to generate real data points based on the function y = mx^2 + c
def generate_real_data(n):
    x = np.random.uniform(-1, 1, n).astype(np.float32)
    m = 1.0  
    c = 0.0  
    y = m * (x ** 2) + c
    return torch.tensor(x).view(-1, 1), torch.tensor(y).view(-1, 1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 15),
            nn.ReLU(),
            nn.Linear(15, 1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(epochs, batch_size):
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        x_real, y_real = generate_real_data(batch_size)
        real_data = torch.cat((x_real, y_real), dim=1)

        noise = torch.randn(batch_size, 1)
        noise = noise/torch.max(noise)
        y_fake = generator(noise)

        y_discriminator_real = torch.ones(batch_size, 1)
        y_discriminator_fake = torch.zeros(batch_size, 1)

        discriminator.zero_grad()
        
        d_loss_real = criterion(discriminator(real_data), y_discriminator_real)
        
        fake_data = torch.cat((noise, y_fake), dim=1)  
        d_loss_fake = criterion(discriminator(fake_data), y_discriminator_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        generator.zero_grad()
        
        noise_for_gan = torch.randn(batch_size, 1)
        g_loss = criterion(discriminator(torch.cat((noise_for_gan, generator(noise_for_gan)), dim=1)), y_discriminator_real)
        
        g_loss.backward()
        optimizer_g.step()

        if epoch % (epochs // 10) == 0:
            print(f'Epoch: {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

epochs = 10000
batch_size = 32

generator = Generator()
discriminator = Discriminator()

optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

train_gan(epochs, batch_size)

x_test = torch.linspace(-1, 1, 100).view(-1, 1)
y_generated = generator(x_test).detach()

plt.figure(figsize=(10,5))
plt.plot(x_test.numpy(), y_generated.numpy(), label='Generated Curve', color='blue')
plt.scatter(*generate_real_data(100), label='Real Data', color='red', alpha=0.5)
plt.title('GAN Generated vs Real Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()