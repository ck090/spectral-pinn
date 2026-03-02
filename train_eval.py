# Zelig & Dekel (2023) - https://arxiv.org/abs/2302.05322
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import LinearSpectralPINN
from src.data import sample_u0
torch.manual_seed(42)
np.random.seed(42)

L = 1.0
n_modes, hdim, n_pts = 20, 64, 101   # paper: L=101 equispaced pts, K=20 modes
bs, epochs, lr = 32, 1000, 1e-3

# ICs from paper's function space W = span{sin(2πkx), k=1..K}
test_cases = {
    "sin(2pi*x)": lambda x: np.sin(2*np.pi*x),
    "sin(4pi*x)": lambda x: np.sin(4*np.pi*x),
    "sin(2pi*x)+0.5sin(4pi*x)": lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x),
    "sin(2pi*x)+0.5sin(6pi*x)": lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x),
    "3-mode mix (k=1,2,4)": lambda x: 0.6*np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.4*np.sin(8*np.pi*x),
}


def fourier_coeffs(fn, n_modes=n_modes, n_quad=1000):
    """Project fn onto paper basis {sin(2πkx/L), k=1..K}."""
    x = np.linspace(0, L, n_quad)
    return np.array([(2/L) * np.trapezoid(fn(x) * np.sin(2*k*np.pi*x/L), x)
                     for k in range(1, n_modes+1)])

def analytical_linear(x, t, ck, pde, **kw):
    """Analytical solution using paper basis {sin(2πkx/L)}."""
    u = np.zeros_like(x)
    for ki, c in enumerate(ck, 1):
        freq = 2 * ki * np.pi / L
        if pde == 'heat':
            T = np.exp(-kw['nu'] * freq**2 * t)
        elif pde == 'wave':
            T = np.cos(kw['c'] * freq * t)
        elif pde == 'reaction_diffusion':
            T = np.exp(-(kw['nu'] * freq**2 - kw['r']) * t)
        u += c * T * np.sin(freq * x)
    return u

def rel_l2(pred, ref):
    denom = np.linalg.norm(ref)
    if denom < 0.05:  # near-zero reference (e.g. wave passing through 0)
        return np.linalg.norm(pred - ref)
    return np.linalg.norm(pred - ref) / denom

def pretrain_transform(model, device, epochs=500):
    """Pretrain linear transform block C~: R¹⁰¹→R²⁰ to extract sine coefficients.
    Paper §4.1: pretrained separately on 1000 ICs before full model training.
    """
    opt = optim.Adam(model.net.parameters(), lr=1e-2)
    x_np = np.linspace(0, L, n_pts)
    freqs = torch.tensor([2*k*np.pi/L for k in range(1, n_modes+1)],
                         dtype=torch.float32, device=device)
    phi0 = torch.sin(torch.tensor(x_np, dtype=torch.float32, device=device)
                     .unsqueeze(-1) * freqs)          # (n_pts, K)
    for _ in range(epochs):
        c = torch.rand(bs, n_modes, device=device) * 2 - 1
        c = c / c.norm(dim=1, keepdim=True)           # unit-normalised ICs
        u0 = c @ phi0.T                               # (bs, n_pts)
        opt.zero_grad()
        nn.MSELoss()(model.net(u0), c).backward()
        opt.step()


def train_linear(pde, kw, device):
    model = LinearSpectralPINN(pde, n_modes, hdim, n_pts, L=L, **kw).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    x_fix = torch.linspace(0,L,n_pts).unsqueeze(0).expand(bs,-1).unsqueeze(-1).to(device)
    t0 = torch.zeros(bs, n_pts, 1, device=device)
    pretrain_transform(model, device)     # §4.1: pretrain transform block first
    print(f"\n[{pde}]")
    for ep in range(epochs):
        opt.zero_grad()
        _, u0 = sample_u0(bs, n_pts)
        u0 = u0.to(device)
        loss = nn.MSELoss()(model(u0, x_fix, t0).squeeze(-1), u0)
        loss.backward(); opt.step()
        if ep % 200 == 0:
            print(f"  [{ep:4d}] residual loss: {loss.item():.5f}")
    return model


def eval_linear(model, pde, kw, t_vals):
    x_np = np.linspace(0, L, n_pts)
    x_t  = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    print(f"\n  {'IC':<32}" + "".join(f"t={t:.1f}       " for t in t_vals))
    print("  " + "-"*78)
    for name, fn in test_cases.items():
        ck   = fourier_coeffs(fn)
        u0_t = torch.tensor(fn(x_np), dtype=torch.float32).unsqueeze(0)
        errs = []
        for t in t_vals:
            with torch.no_grad():
                pred = model(u0_t, x_t, torch.full((1,n_pts,1), t)).squeeze().numpy()
            errs.append(rel_l2(pred, analytical_linear(x_np, t, ck, pde, **kw)))
        print(f"  {name:<32}" + "".join(f"{e:.4e}  " for e in errs))


if __name__ == "__main__":
    device = "cpu"
    os.makedirs('checkpoints', exist_ok=True)

    # Paper §4.2: training and testing on t∈[0,0.5], α=0.01
    configs = [
        ('heat',               {'nu': 0.01},            [0.0, 0.1, 0.25, 0.5]),
        ('wave',               {'c':  1.0},             [0.0, 0.1, 0.25, 0.5]),
        ('reaction_diffusion', {'nu': 0.01, 'r': 0.05}, [0.0, 0.1, 0.25, 0.5]),
    ]

    for pde, kw, t_vals in configs:
        m = train_linear(pde, kw, device)
        torch.save(m.state_dict(), f'checkpoints/{pde}.pth')
        print(f"\n── {pde} (rel L2 vs analytical) ──")
        eval_linear(m, pde, kw, t_vals)

