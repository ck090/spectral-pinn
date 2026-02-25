Spectral PINN Implementation
===========================

This repository contains an implementation of a Spectral Physics-Informed Neural Network (SPINN) for solving the 1D Heat Equation, inspired by the paper "Numerical Methods For PDEs Over Manifolds Using Spectral Physics Informed Neural Networks".

Overview
--------
The architecture leverages the spectral decomposition of the solution. For the heat equation:
$u(x,t) = \sum c_k(t) \phi_k(x)$

The network learns to map an initial condition $u(x,0)$ to the spectral coefficients $c_k$. 
By hardcoding the analytical time evolution of the coefficients in the decoder ($c_k(t) = c_k(0)e^{-\lambda_k t}$), the network automatically satisfies the PDE, and training reduces to learning the decomposition of the initial condition.

Structure
---------
- `src/model.py`: Neural network architecture.
- `src/train.py`: Training script.
- `src/data.py`: Data generation utilities.

Usage
-----
1. Install dependencies:
   ```
   pip install torch numpy matplotlib
   ```

2. Run training:
   ```
   python src/train.py
   ```
