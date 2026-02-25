# Reference: Numerical Methods For PDEs Over Manifolds Using Spectral Physics Informed Neural Networks
# https://www.alphaxiv.org/abs/2302.05322

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.model import SpectralPINN
from src.data import sample_u0

if __name__ == "__main__":
    L, nu = 1.0, 0.01
    n_modes, hdim, n_pts = 20, 50, 50
    bs, epochs, lr = 32, 1000, 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralPINN(n_modes, hdim, n_pts, L=L, nu=nu).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []

    x_fix = torch.linspace(0, L, n_pts).unsqueeze(0).expand(bs, -1).unsqueeze(-1).to(device)
    t0 = torch.zeros(bs, n_pts, 1, device=device)

    print("training...")
    for ep in range(epochs):
        opt.zero_grad()
        _, u0 = sample_u0(bs, n_pts, L=L)
        u0 = u0.to(device)
        pred = model(u0, x_fix, t0)
        loss = nn.MSELoss()(pred.squeeze(-1), u0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if ep % 100 == 0:
            print(f"  [{ep:4d}] loss {loss.item():.5f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/spectral_pinn.pth')
    print("done.")

    plt.figure()
    plt.semilogy(losses)
    plt.xlabel('epoch'); plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('loss_curve.png')
