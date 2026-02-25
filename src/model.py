import torch
import torch.nn as nn
import numpy as np

class SpectralPINN(nn.Module):
    """
    Spectral Physics-Informed Neural Network (SPINN) for Operator Learning.
    
    This model takes a discretized initial condition u0(x) as input and implicitly learns
    the spectral coefficients to reconstruct the solution u(x,t).
    
    For the 1D Heat Equation on [0, L] with Dirichlet BCs:
    u(x,t) = sum_{k=1}^N c_k(t) * phi_k(x)
    phi_k(x) = sin(k * pi * x / L)
    
    The network predicts the initial spectral coefficients c_k(0) based on the input u0.
    The time evolution is handled either analytically (for linear heat) or by a secondary network.
    
    Here, we implement a version where the network predicts c_k(0) and we use the known
    decay for the heat equation: c_k(t) = c_k(0) * exp(-nu * (k*pi/L)^2 * t).
    
    This effectively learns the spectral decomposition of the initial condition 
    in a way that minimizes the PDE residual (or data loss if available).
    """
    def __init__(self, num_modes, hidden_dim, input_dim, L=1.0, nu=0.01):
        super(SpectralPINN, self).__init__()
        self.num_modes = num_modes
        self.L = L
        self.nu = nu
        
        # Branch network: Maps discretized initial condition u0 to spectral coefficients c_k(0)
        self.branch_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_modes) # Predicts c_1(0), ..., c_N(0)
        )
        
        # Pre-compute wavenumbers for efficiency (k * pi / L)
        # We start from k=1 for Dirichlet sine series
        self.register_buffer('k_vec', torch.arange(1, num_modes + 1).float())
        
    def forward(self, u0, x, t):
        """
        Forward pass.
        
        Args:
            u0: (Batch, Input_Dim) - Discretized initial condition samples
            x: (Batch, N_points, 1) or (Batch, 1) - Spatial coordinates
            t: (Batch, N_points, 1) or (Batch, 1) - Time coordinates
            
        Returns:
            u_pred: (Batch, N_points, 1) - Predicted solution at (x,t)
        """
        batch_size = u0.shape[0]
        
        # 1. Predict initial spectral coefficients c_k(0)
        # Shape: (Batch, num_modes)
        c0 = self.branch_net(u0) 
        
        # 2. Compute time-dependent coefficients analytical decay
        # c_k(t) = c_k(0) * exp(-nu * (k*pi/L)^2 * t)
        
        # Prepare terms for broadcasting
        # k_vec: (num_modes)
        # t: (Batch, N_points, 1) -> we need t to broadcast with k
        
        # Eigenvalues lambda_k = nu * (k * pi / L)^2
        lambda_k = self.nu * (self.k_vec * np.pi / self.L) ** 2 # (num_modes,)
        
        # Expand dims for broadcasting
        # c0: (Batch, 1, num_modes)
        c0_expanded = c0.unsqueeze(1)
        
        # lambda_k: (1, 1, num_modes)
        lambda_k_expanded = lambda_k.unsqueeze(0).unsqueeze(0)
        
        # t: (Batch, N_points, 1)
        # exp_term: (Batch, N_points, num_modes)
        exp_term = torch.exp(-lambda_k_expanded * t)
        
        # 3. Compute spatial basis functions phi_k(x) = sin(k * pi * x / L)
        # x: (Batch, N_points, 1)
        # argument: (Batch, N_points, num_modes) = x * (k * pi / L)
        spatial_freq = (self.k_vec * np.pi / self.L).unsqueeze(0).unsqueeze(0)
        phi_x = torch.sin(x * spatial_freq)
        
        # 4. Combine: u(x,t) = sum_k c_k(0) * exp(...) * phi_k(x)
        # Sum over the mode dimension (last dim)
        # (Batch, N_points, num_modes) * (Batch, N_points, num_modes) * (Batch, 1, num_modes)
        u_pred = (c0_expanded * exp_term * phi_x).sum(dim=-1, keepdim=True)
        
        return u_pred

    def predict_coefficients(self, u0):
        return self.branch_net(u0)
