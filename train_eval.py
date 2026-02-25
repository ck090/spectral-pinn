# Zelig & Dekel (2023) - https://arxiv.org/abs/2302.05322

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import SpectralPINN
from src.data import sample_u0

torch.manual_seed(42)
np.random.seed(42)


def fourier_coeffs(u0_fn, L=1.0, n_modes=20, n_quad=1000):
    x = np.linspace(0, L, n_quad)
    u0 = u0_fn(x)
    return np.array([(2/L) * np.trapezoid(u0 * np.sin(k * np.pi * x / L), x)
                     for k in range(1, n_modes + 1)])

def analytical(x, t, ck, L=1.0, nu=0.01):
    u = np.zeros_like(x)
    for k, c in enumerate(ck, 1):
        u += c * np.exp(-nu * (k * np.pi / L) ** 2 * t) * np.sin(k * np.pi * x / L)
    return u

def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-10)


if __name__ == "__main__":
    L, nu = 1.0, 0.01
    n_modes, hdim, n_pts = 20, 50, 50
    bs, epochs, lr = 32, 1000, 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralPINN(n_modes, hdim, n_pts, L=L, nu=nu).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    x_fix = torch.linspace(0, L, n_pts).unsqueeze(0).expand(bs, -1).unsqueeze(-1).to(device)
    t0 = torch.zeros(bs, n_pts, 1, device=device)

    # --- train ---
    print("training...")
    for ep in range(epochs):
        opt.zero_grad()
        _, u0 = sample_u0(bs, n_pts, L=L)
        u0 = u0.to(device)
        loss = nn.MSELoss()(model(u0, x_fix, t0).squeeze(-1), u0)
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            print(f"  [{ep:4d}] loss {loss.item():.5f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/spectral_pinn.pth')
    print("done.\n")

    # --- eval ---
    model.eval()
    x_np = np.linspace(0, L, n_pts)
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    cases = {
        "sin(pi*x)":              lambda x: np.sin(np.pi * x),
        "sin(2*pi*x)":            lambda x: np.sin(2 * np.pi * x),
        "sin(pi*x)+0.5sin(3pi*x)":lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x),
        "gaussian (center=0.5)":  lambda x: np.exp(-50 * (x - 0.5) ** 2) * np.sin(np.pi * x),
        "x*(1-x)":                lambda x: x * (1 - x),
    }
    t_vals = [0.0, 0.5, 1.0, 2.0]

    print(f"{'Initial Condition':<35}" + "".join(f"  t={t:.1f}     " for t in t_vals))
    print("-" * 90)
    for name, fn in cases.items():
        ck = fourier_coeffs(fn, L=L, n_modes=n_modes)
        u0_t = torch.tensor(fn(x_np), dtype=torch.float32).unsqueeze(0)
        errs = []
        for t in t_vals:
            with torch.no_grad():
                pred = model(u0_t, x_t, torch.full((1, n_pts, 1), t)).squeeze().numpy()
            errs.append(rel_l2(pred, analytical(x_np, t, ck, L=L, nu=nu)))
        print(f"{name:<35}" + "".join(f"  {e:.4e}  " for e in errs))
