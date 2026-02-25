Spectral PINN Implementation
===========================

Reference: "Numerical Methods For PDEs Over Manifolds Using Spectral Physics Informed Neural Networks"  
https://www.alphaxiv.org/abs/2302.05322

Overview
--------
The architecture leverages the spectral decomposition of the solution. For the heat equation:
$u(x,t) = \sum c_k(t) \phi_k(x)$

The network learns to map an initial condition $u(x,0)$ to the spectral coefficients $c_k$. 
By hardcoding the analytical time evolution of the coefficients ($c_k(t) = c_k(0)e^{-\lambda_k t}$), the network automatically satisfies the PDE.

Files
-----

- `train.py`: Training script
- `src/model.py`: Neural network architecture
- `src/data.py`: Data generation utilities
- `src/__init__.py`: Module initialization

Usage
-----

**Dependencies:** torch, numpy, matplotlib

**Run from the root directory:**

```bash
conda activate apebench  # or your environment
KMP_DUPLICATE_LIB_OK=TRUE python train.py
```
