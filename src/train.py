import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.model import SpectralPINN
from src.data import generate_initial_condition
import os

def train():
    # Configuration
    L = 1.0
    NU = 0.01
    NUM_MODES = 20  # Number of spectral modes
    HIDDEN_DIM = 50
    U0_SAMPLES = 50 # Number of points to sample input u(x,0)
    BATCH_SIZE = 32
    EPOCHS = 1000
    LR = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = SpectralPINN(num_modes=NUM_MODES, 
                         hidden_dim=HIDDEN_DIM, 
                         input_dim=U0_SAMPLES, 
                         L=L, 
                         nu=NU).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    loss_history = []
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Data Generation (On the fly)
        # We need u0 samples for the branch net input
        _, u0_input = generate_initial_condition(BATCH_SIZE, U0_SAMPLES, L=L, type='mix_sine')
        u0_input = u0_input.to(device) # (Batch, Input_Dim)
        
        # We need query points (x, t) to evaluate the PDE residual
        # Random collocation points in domain [0, L] x [0, 1]
        N_collocation = 100
        x_col = torch.rand(BATCH_SIZE, N_collocation, 1).to(device) * L
        t_col = torch.rand(BATCH_SIZE, N_collocation, 1).to(device) # Time t in [0,1]
        
        # Require grad for PID derivatives (though here we use analytical time evol, 
        # so maybe we don't strictly need automatic differentiation for the residual 
        # if we trust the construction? 
        # Actually, the construction guarantees the equation is satisfied IF the coefficients decay correctly.
        # Since I hardcoded the decay `exp(-lambda_k * t)`, the function `u_pred` BY DEFINITION
        # satisfies the heat equation layer-wise.
        
        # The only thing the network needs to learn is to match the INITIAL CONDITION.
        # u(x, 0) predicted by model should match the input u0 samples.
        
        # Let's verify this claim:
        # u = sum c_k * exp(-lambda_k t) * phi_k(x)
        # u_t = sum c_k * (-lambda_k) * exp(...) * phi_k
        # u_xx = sum c_k * exp(...) * (-k^2 pi^2 / L^2) * phi_k
        # lambda_k = nu * k^2 pi^2 / L^2
        # So u_t - nu * u_xx = 0 is automatically satisfied by the architecture!
        
        # This is the beauty of the spectral method described. 
        # We don't need a PDE residual loss (u_t - u_xx)^2.
        # We ONLY need a DATA loss at t=0 to match the initial condition u0.
        
        # So the loss is simply Reconstruction Loss at t=0.
        
        t_zero = torch.zeros_like(x_col)
        
        # Predict u at t=0
        u_pred_0 = model(u0_input, x_col, t_zero) # (Batch, N_col, 1)
        
        # Evaluating the ground truth u0 at x_col
        # NOTE: u0_input serves as the "representation" of the function. 
        # But to compute loss, we need the actual value of that function at x_col.
        # Since our generate_initial_condition creates random functions, we need to know 
        # the value at x_col corresponding to the u0_input.
        
        # Let's slightly refactor data generation to give us the function handle or sufficient info.
        # Or simpler for this demo: use the same x grid for input and loss evaluation?
        # That's easiest. Let's assume x_col are the fixed sensor locations for now.
        
        # Refined Data Generation for training loop:
        # Fix x_sensors to be linspace.
        x_sensors = torch.linspace(0, L, U0_SAMPLES).unsqueeze(0).repeat(BATCH_SIZE, 1).to(device) # (Batch, U0_SAMPLES)
        x_query = x_sensors.unsqueeze(-1) # (Batch, U0_SAMPLES, 1)
        
        # We still generate random u0 values at these x_sensors
        _, u0_vals = generate_initial_condition(BATCH_SIZE, U0_SAMPLES, L=L, type='mix_sine')
        u0_vals = u0_vals.to(device)
        u0_input = u0_vals # The network sees the values at sensors
        
        # Network prediction at t=0 and x=x_sensors
        t_zeros = torch.zeros(BATCH_SIZE, U0_SAMPLES, 1).to(device)
        u_rec_0 = model(u0_input, x_query, t_zeros) # Should match u0_vals
        
        loss = nn.MSELoss()(u_rec_0.squeeze(-1), u0_vals)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Save model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/spectral_pinn.pth')
    print("Training finished and model saved.")
    
    # Plotting results
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Loss')
    plt.savefig('loss_curve.png')

if __name__ == "__main__":
    train()
