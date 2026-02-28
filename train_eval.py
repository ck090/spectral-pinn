# Zelig & Dekel (2023) - https://arxiv.org/abs/2302.05322

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import LinearSpectralPINN, BurgersSpectralPINN
from src.data import sample_u0
torch.manual_seed(42)
np.random.seed(42)

L = 1.0
n_modes, hdim, n_pts = 20, 64, 50
bs, epochs, lr = 32, 1000, 1e-3

# ICs from paper's function space W = span{sin(2πkx), k=1..K}
test_cases = {
    "sin(2pi*x)":               lambda x: np.sin(2*np.pi*x),
    "sin(4pi*x)":               lambda x: np.sin(4*np.pi*x),
    "sin(2pi*x)+0.5sin(4pi*x)": lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x),
    "sin(2pi*x)+0.5sin(6pi*x)": lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x),
    "3-mode mix (k=1,2,4)":     lambda x: 0.6*np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.4*np.sin(8*np.pi*x),
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

def fd_burgers(fn, t_final, nu=0.01, nx=300):
    """finite difference reference for Burgers, Dirichlet BCs"""
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    dt = min(dx**2 / (4*nu), dx * 0.4)
    nt = int(t_final / dt) + 1
    dt = t_final / nt
    u = fn(x).copy(); u[0] = u[-1] = 0
    for _ in range(nt):
        u_x  = np.zeros_like(u)
        u_xx = np.zeros_like(u)
        u_x[1:-1]  = (u[2:] - u[:-2]) / (2*dx)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        u[1:-1] += dt * (-u[1:-1]*u_x[1:-1] + nu*u_xx[1:-1])
        u[0] = u[-1] = 0
    return x, u

def rel_l2(pred, ref):
    denom = np.linalg.norm(ref)
    if denom < 0.05:  # near-zero reference (e.g. wave passing through 0)
        return np.linalg.norm(pred - ref)
    return np.linalg.norm(pred - ref) / denom

def train_linear(pde, kw, device):
    model = LinearSpectralPINN(pde, n_modes, hdim, n_pts, L=L, **kw).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    x_fix = torch.linspace(0,L,n_pts).unsqueeze(0).expand(bs,-1).unsqueeze(-1).to(device)
    t0    = torch.zeros(bs, n_pts, 1, device=device)
    print(f"\n[{pde}]")
    for ep in range(epochs):
        opt.zero_grad()
        _, u0 = sample_u0(bs, n_pts)
        u0 = u0.to(device)
        loss = nn.MSELoss()(model(u0, x_fix, t0).squeeze(-1), u0)
        loss.backward(); opt.step()
        if ep % 200 == 0:
            print(f"  [{ep:4d}] loss {loss.item():.5f}")
    return model


def train_burgers(nu, device):
    T_max, N_col = 0.5, 30
    n_epochs = 2000
    model = BurgersSpectralPINN(n_modes, hdim, n_pts, L=L, nu=nu).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    x_fix = torch.linspace(0,L,n_pts).unsqueeze(0).expand(bs,-1).unsqueeze(-1).to(device)
    print(f"\n[burgers]")
    for ep in range(n_epochs):
        opt.zero_grad()
        _, u0 = sample_u0(bs, n_pts)
        u0 = u0.to(device)
        # IC loss at t=0
        ic_loss = nn.MSELoss()(
            model(u0, x_fix, torch.zeros(bs, 1, device=device)).squeeze(-1), u0
        )
        # PDE residual at random interior times (down-weighted early on)
        t_r   = (torch.rand(bs, 1, device=device) * T_max).detach().requires_grad_(True)
        x_col = torch.rand(bs, N_col, 1, device=device)
        pde_loss = model.residual(u0.detach(), x_col, t_r).pow(2).mean()
        w_pde = min(ep / 500, 1.0) * 0.1  # ramp up pde weight
        loss = ic_loss + w_pde * pde_loss
        loss.backward(); opt.step()
        if ep % 400 == 0:
            print(f"  [{ep:4d}] ic {ic_loss.item():.5f}  pde {pde_loss.item():.5f}")
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


def eval_burgers(model, nu, t_vals):
    x_np = np.linspace(0, L, n_pts)
    x_t  = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    print(f"\n  {'IC':<32}" + "".join(f"t={t:.2f}      " for t in t_vals))
    print("  " + "-"*78)
    for name, fn in test_cases.items():
        u0_t = torch.tensor(fn(x_np), dtype=torch.float32).unsqueeze(0)
        errs = []
        for t in t_vals:
            with torch.no_grad():
                pred = model(u0_t, x_t, torch.full((1,1), t)).squeeze().numpy()
            if t == 0:
                ref = fn(x_np)
            else:
                x_fd, u_fd = fd_burgers(fn, t, nu=nu)
                ref = np.interp(x_np, x_fd, u_fd)
            errs.append(rel_l2(pred, ref))
        print(f"  {name:<32}" + "".join(f"{e:.4e}  " for e in errs))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints', exist_ok=True)

    configs = [
        ('heat',               {'nu': 0.01},           [0.0, 0.5, 1.0, 2.0]),
        ('wave',               {'c':  1.0},            [0.0, 0.25, 0.75, 1.5]),
        ('reaction_diffusion', {'nu': 0.01, 'r': 0.05},[0.0, 0.5, 1.0, 2.0]),
    ]

    for pde, kw, t_vals in configs:
        m = train_linear(pde, kw, device)
        torch.save(m.state_dict(), f'checkpoints/{pde}.pth')
        print(f"\n── {pde} (rel L2 vs analytical) ──")
        eval_linear(m, pde, kw, t_vals)

    m = train_burgers(nu=0.05, device=device)
    torch.save(m.state_dict(), 'checkpoints/burgers.pth')
    print("\n── burgers nu=0.05 (rel L2 vs FD reference) ──")
    eval_burgers(m, nu=0.05, t_vals=[0.0, 0.1, 0.2, 0.4])
