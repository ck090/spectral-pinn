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

- `train_eval.py`: Training and evaluation
- `src/model.py`: Neural network architecture
- `src/data.py`: Data generation utilities
- `src/__init__.py`: Module initialization

Usage
-----

**Dependencies:** torch, numpy, matplotlib

**Run from the root directory:**

```bash
conda activate apebench  # or your environment
KMP_DUPLICATE_LIB_OK=TRUE python train_eval.py
```

Results
-------

Relative L2 error between the model prediction and the analytical Fourier solution across time, evaluated on 5 test initial conditions (not seen during training).

| Initial Condition | t=0.0 | t=0.5 | t=1.0 | t=2.0 |
|---|---|---|---|---|
| sin(πx) | 9.04e-02 | 4.58e-02 | 3.82e-02 | 3.50e-02 |
| sin(2πx) | 1.90e-01 | 8.99e-02 | 7.24e-02 | 6.28e-02 |
| sin(πx) + 0.5·sin(3πx) | 2.28e-01 | 1.16e-01 | 8.65e-02 | 6.89e-02 |
| Gaussian (center=0.5) | 2.15e-01 | 1.10e-01 | 7.82e-02 | 5.48e-02 |
| x(1−x) | 1.41e-01 | 7.38e-02 | 6.11e-02 | 5.45e-02 |

Error generally decreases over time as the solution smooths out, consistent with the diffusive nature of the heat equation.
