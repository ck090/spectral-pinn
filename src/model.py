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
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, num_modes)
        )
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
        c0  = self.net(u0).unsqueeze(1)                          # (B, 1, K)
        T   = self._time_factor(t)                               # (B, N, K)
        phi = torch.sin(x * (2 * self.k * np.pi / self.L))      # (B, N, K)
        return (c0 * T * phi).sum(-1, keepdim=True)


class BurgersSpectralPINN(nn.Module):
    """
    Burgers equation: u_t + u*u_x = nu*u_xx on [0,L] with Dirichlet BCs.
    Spatial basis is the sine series; coefficient evolution is learned.
    Loss = IC reconstruction + weighted PDE residual.
    """
    def __init__(self, num_modes, hidden_dim, input_dim, L=1.0, nu=0.01):
        super().__init__()
        self.L, self.nu = L, nu
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # takes (u0_encoding, t) -> spectral coefficients c_k(t)
        self.coeff_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, num_modes)
        )
        self.register_buffer('k', torch.arange(1, num_modes + 1).float())

    def forward(self, u0, x, t):
        # t: (B, 1)
        h = self.encoder(u0)
        c = self.coeff_net(torch.cat([h, t], dim=-1)).unsqueeze(1)      # (B, 1, K)
        phi = torch.sin(x * (2 * self.k * np.pi / self.L))               # (B, N, K)
        return (c * phi).sum(-1, keepdim=True)

    def residual(self, u0, x, t):
        # x: (B, N, 1), t: (B, 1) requires_grad=True
        h    = self.encoder(u0)
        c    = self.coeff_net(torch.cat([h, t], dim=-1))  # (B, K)
        freq = 2 * self.k * np.pi / self.L      # paper: sin(2πkx/L) basis
        phi    = torch.sin(x * freq)
        phi_x  = torch.cos(x * freq) * freq
        phi_xx = -phi * freq ** 2
        c_e  = c.unsqueeze(1)
        u    = (c_e * phi).sum(-1)
        u_x  = (c_e * phi_x).sum(-1)
        u_xx = (c_e * phi_xx).sum(-1)
        # dc_k/dt via autograd, one pass per mode
        dc_dt = torch.stack([
            torch.autograd.grad(c[:, ki].sum(), t, create_graph=True)[0].squeeze(-1)
            for ki in range(c.shape[1])
        ], dim=-1)  # (B, K)
        u_t = (dc_dt.unsqueeze(1) * phi).sum(-1)
        return u_t + u * u_x - self.nu * u_xx  # (B, N)
