# Zelig & Dekel (2023) - https://arxiv.org/abs/2302.05322
# u(x,t) = sum_k c_k(0) * exp(-lam_k * t) * sin(k*pi*x/L)

import torch
import torch.nn as nn
import numpy as np

class SpectralPINN(nn.Module):
    def __init__(self, num_modes, hidden_dim, input_dim, L=1.0, nu=0.01):
        super().__init__()
        self.L = L
        self.nu = nu

        # maps u0 samples -> spectral coefficients c_k(0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, num_modes)
        )
        self.register_buffer('k', torch.arange(1, num_modes + 1).float())

    def forward(self, u0, x, t):
        c0 = self.net(u0).unsqueeze(1)         
        lam = self.nu * (self.k * np.pi / self.L) ** 2
        phi = torch.sin(x * (self.k * np.pi / self.L))  
        decay = torch.exp(-lam.unsqueeze(0).unsqueeze(0) * t)
        return (c0 * decay * phi).sum(-1, keepdim=True)

    def coeffs(self, u0):
        return self.net(u0)
