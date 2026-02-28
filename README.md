Spectral PINN Implementation
===========================

Reference: "Numerical Methods For PDEs Over Manifolds Using Spectral Physics Informed Neural Networks"  
https://www.alphaxiv.org/abs/2302.05322

Overview
--------
The architecture leverages the spectral decomposition of the solution. The network maps
a discretized initial condition $u(x,0)$ to spectral coefficients $c_k$, then
applies the known eigenfunction basis $\phi_k(x) = \sin(k\pi x/L)$ with the
appropriate time factor for the target PDE.

For linear PDEs the time evolution is baked in analytically — meaning the PDE is 
satisfied by construction and training reduces to fitting the initial condition.
For Burgers (nonlinear), time-dependent coefficients are learned with a PDE residual loss.

Files
-----

- `train_eval.py`: Training and evaluation for all PDEs
- `src/model.py`: `LinearSpectralPINN` and `BurgersSpectralPINN`
- `src/data.py`: Initial condition sampling
- `src/__init__.py`: Module initialization

Usage
-----

**Dependencies:** torch, numpy

**Run from the root directory:**

```bash
conda activate apebench  # or your environment
KMP_DUPLICATE_LIB_OK=TRUE python train_eval.py
```

Results
-------

Relative L2 error vs analytical (linear PDEs) or finite-difference reference (Burgers).
Seed fixed to 42 for reproducibility.

**Heat equation** — $u_t = \nu u_{xx}$, $\nu=0.01$

| Initial Condition | t=0.0 | t=0.5 | t=1.0 | t=2.0 |
|---|---|---|---|---|
| sin(πx) | 1.16e-01 | 4.59e-02 | 2.72e-02 | 1.54e-02 |
| sin(2πx) | 1.58e-01 | 7.79e-02 | 8.05e-02 | 1.04e-01 |
| sin(πx) + 0.5·sin(3πx) | 2.77e-01 | 1.28e-01 | 8.03e-02 | 4.06e-02 |
| Gaussian (center=0.5) | 2.46e-01 | 1.29e-01 | 1.12e-01 | 1.06e-01 |
| x(1−x) | 1.88e-01 | 1.30e-01 | 1.22e-01 | 1.20e-01 |

Errors drop with time for smooth ICs — diffusion kills high-frequency content, which is exactly what the model struggles with most.

---

**Wave equation** — $u_{tt} = c^2 u_{xx}$, $c=1.0$ (reformulated as $u_t + cu_x = 0$ in spectral form)

| Initial Condition | t=0.0 | t=0.25 | t=0.75 | t=1.5 |
|---|---|---|---|---|
| sin(πx) | 1.35e-01 | 1.27e-01 | 1.27e-01 | 3.59e-01 |
| sin(2πx) | 1.74e-01 | 4.62e-01 | 4.62e-01 | 1.57e-01 |
| sin(πx) + 0.5·sin(3πx) | 3.22e-01 | 3.07e-01 | 3.07e-01 | 8.78e-01 |
| Gaussian (center=0.5) | 2.61e-01 | 2.61e-01 | 2.61e-01 | 2.21e-01 |
| x(1−x) | 2.11e-01 | 2.06e-01 | 2.06e-01 | 1.26e-01 |

Higher errors than heat — no dissipation means the network can't lean on smoothing. Errors at t=0.25 and t=0.75 are identical for most cases because $|\cos(k\pi t)|$ is symmetric around the quarter period.

---

**Reaction-diffusion** — $u_t = \nu u_{xx} + ru$, $\nu=0.01$, $r=0.05$

| Initial Condition | t=0.0 | t=0.5 | t=1.0 | t=2.0 |
|---|---|---|---|---|
| sin(πx) | 1.13e-01 | 4.84e-02 | 3.56e-02 | 3.13e-02 |
| sin(2πx) | 1.74e-01 | 8.70e-02 | 8.26e-02 | 8.52e-02 |
| sin(πx) + 0.5·sin(3πx) | 2.74e-01 | 1.23e-01 | 8.85e-02 | 7.50e-02 |
| Gaussian (center=0.5) | 2.69e-01 | 1.09e-01 | 7.60e-02 | 5.72e-02 |
| x(1−x) | 1.50e-01 | 6.85e-02 | 5.49e-02 | 5.10e-02 |

Similar to heat, the reaction term shifts the eigenvalue decay rate but doesn't change the overall trend.

---

**Burgers equation** — $u_t + uu_x = \nu u_{xx}$, $\nu=0.05$ (nonlinear, vs FD reference)

| Initial Condition | t=0.0 | t=0.1 | t=0.2 | t=0.4 |
|---|---|---|---|---|
| sin(πx) | 2.42e-01 | 2.61e-01 | 3.00e-01 | 3.71e-01 |
| sin(2πx) | 4.01e-01 | 2.94e-01 | 2.09e-01 | 4.71e-01 |
| sin(πx) + 0.5·sin(3πx) | 4.81e-01 | 3.65e-01 | 2.88e-01 | 2.79e-01 |
| Gaussian (center=0.5) | 5.43e-01 | 4.04e-01 | 3.71e-01 | 3.96e-01 |
| x(1−x) | 2.46e-01 | 2.10e-01 | 1.69e-01 | 1.10e-01 |

Larger errors as expected — Burgers is nonlinear so the clean spectral decomposition doesn't hold. The model uses a learned time-dependent coefficient network with a PDE residual loss, which is a rougher approximation.
