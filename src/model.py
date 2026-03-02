# Zelig & Dekel (2023) - https://arxiv.org/abs/2302.05322

import torch
import torch.nn as nn
import numpy as np


class LinearSpectralPINN(nn.Module):
    """
    Linear PDEs on [0,L] with Dirichlet BCs.
    Supported: 'heat', 'wave', 'reaction_diffusion'
    Time evolution is baked in analytically -- PDE satisfied by construction.
    """
    def __init__(self, pde, num_modes, hidden_dim, input_dim, L=1.0, **kw):
        super().__init__()
        self.pde, self.L, self.kw = pde, L, kw
        self.net = nn.Linear(input_dim, num_modes)
        self.register_buffer('k', torch.arange(1, num_modes + 1).float())

    def _time_factor(self, t):
        k, L = self.k, self.L
        freq = 2 * k * np.pi / L      # paper: sin(2πkx/L) basis
        if self.pde == 'heat':
            return torch.exp(-self.kw['nu'] * freq ** 2 * t)
        elif self.pde == 'wave':
            return torch.cos(self.kw['c'] * freq * t)
        elif self.pde == 'reaction_diffusion':
            lam = self.kw['nu'] * freq ** 2 - self.kw['r']
            return torch.exp(-lam * t)

    def forward(self, u0, x, t):
        # t: (B, N, 1)
        c0 = self.net(u0).unsqueeze(1)                          # (B, 1, K)
        T = self._time_factor(t)                               # (B, N, K)
        phi = torch.sin(x * (2 * self.k * np.pi / self.L))      # (B, N, K)
        return (c0 * T * phi).sum(-1, keepdim=True)
