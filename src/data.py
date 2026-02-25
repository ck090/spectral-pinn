import torch
import numpy as np

def generate_initial_condition(batch_size, num_points, L=1.0, type='gaussian'):
    """
    Generates random initial conditions u0(x).
    Returns sample points x_samp and u0 values at those points.
    """
    x = torch.linspace(0, L, num_points).unsqueeze(0).repeat(batch_size, 1) # (Batch, num_points)
    
    if type == 'gaussian':
        # Random gaussians sum
        u0 = torch.zeros_like(x)
        for _ in range(3):
            amp = torch.rand(batch_size, 1) * 2 - 1
            width = torch.rand(batch_size, 1) * 0.2 + 0.05
            center = torch.rand(batch_size, 1) * L
            u0 += amp * torch.exp(-((x - center)**2) / (2 * width**2))
            
    elif type == 'mix_sine':
        # Random sum of first few sine modes
        u0 = torch.zeros_like(x)
        for k in range(1, 5):
            amp = torch.rand(batch_size, 1) * 2 - 1
            u0 += amp * torch.sin(k * np.pi * x / L)
            
    # Enforce Dirichlet BCs at 0 and L approximately by multiplying with a window or sine
    # For simplicity in this demo, we multiply by sin(pi * x / L) to ensure 0 at boundaries
    u0 = u0 * torch.sin(np.pi * x / L) 
    
    return x, u0

def analytical_solution_heat(u0_fn, x_query, t_query, L=1.0, nu=0.01, n_terms=50):
    """
    Computes analytical solution for Heat Equation using Fourier Series.
    u(x,t) = sum c_k * exp(-nu * k^2 * pi^2 * t / L^2) * sin(k * pi * x / L)
    c_k = (2/L) * integral(u0(x) * sin(k * pi * x / L) dx)
    """
    # This requires knowing u0 in functional form or integrating numerically.
    # For training, we might rely on the PINN loss (residual) so we don't strictly need this 
    # except for validation.
    pass
