import torch
import numpy as np


def sample_u0(batch_size, n_pts, L=1.0, mode='mix_sine'):
    x = torch.linspace(0, L, n_pts).unsqueeze(0).expand(batch_size, -1)

    if mode == 'gaussian':
        u0 = torch.zeros_like(x)
        for _ in range(3):
            a = torch.rand(batch_size, 1) * 2 - 1
            w = torch.rand(batch_size, 1) * 0.2 + 0.05
            c = torch.rand(batch_size, 1) * L
            u0 += a * torch.exp(-((x - c) ** 2) / (2 * w ** 2))

    elif mode == 'mix_sine':
        # Paper W_20: all K=20 modes, coefficients unit-normalised (Σcₖ²=1)
        K = 20
        c = torch.rand(batch_size, K) * 2 - 1
        c = c / c.norm(dim=1, keepdim=True)          # enforce Σcₖ²=1
        u0 = torch.zeros_like(x)
        for k in range(1, K + 1):
            u0 += c[:, k-1:k] * torch.sin(2 * k * np.pi * x / L)

    return x, u0
