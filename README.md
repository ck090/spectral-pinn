Spectral PINN Implementation
===========================

Reference: "Numerical Methods For PDEs Over Manifolds Using Spectral Physics Informed Neural Networks"  
https://www.alphaxiv.org/abs/2302.05322

Overview

--------
The architecture leverages the spectral decomposition of the solution. The network maps a discretized initial condition $u(x,0)$ to spectral coefficients $c_k$, then applies the known eigenfunction basis $\phi_k(x) = \sin(2k\pi x/L)$ with the appropriate time factor for the target PDE.

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

| Initial Condition | t=0.0 | t=0.1 | t=0.2 | t=0.5 |
|---|---|---|---|---|
| sin(2πx) | 1.30e-02 | 1.46e-03 | 1.02e-03 | 7.34e-04 |
| sin(4πx) | 1.30e-02 | 4.75e-03 | 4.47e-03 | 5.30e-03 |
| sin(2πx)+0.5·sin(4πx) | 1.52e-02 | 2.58e-03 | 1.67e-03 | 1.13e-03 |
| sin(2πx)+0.5·sin(6πx) | 1.31e-02 | 1.85e-03 | 1.26e-03 | 7.87e-04 |
| 3-mode mix (k=1,2,4) | 1.87e-02 | 4.28e-03 | 3.27e-03 | 2.91e-03 |

Errors drop with time — diffusion damps high-frequency content, reducing the approximation error.

---

**Wave equation** — $u_{tt} = c^2 u_{xx}$, $c=1.0$

| Initial Condition | t=0.0 | t=0.1 | t=0.2 | t=0.5 |
|---|---|---|---|---|
| sin(2πx) | 2.64e-02 | 1.49e-02 | 9.22e-02 | 2.64e-02 |
| sin(4πx) | 2.52e-02 | 5.71e-02 | 1.50e-02 | 2.52e-02 |
| sin(2πx)+0.5·sin(4πx) | 2.92e-02 | 2.06e-02 | 3.00e-02 | 2.92e-02 |
| sin(2πx)+0.5·sin(6πx) | 3.35e-02 | 2.38e-02 | 1.47e-01 | 3.35e-02 |
| 3-mode mix (k=1,2,4) | 2.71e-02 | 2.70e-02 | 2.39e-02 | 2.71e-02 |

No dissipation — errors are governed by the accuracy of the learned coefficient extraction, not time.

---

**Reaction-diffusion** — $u_t = \nu u_{xx} + ru$, $\nu=0.01$, $r=0.05$

| Initial Condition | t=0.0 | t=0.1 | t=0.2 | t=0.5 |
|---|---|---|---|---|
| sin(2πx) | 2.11e-02 | 3.44e-03 | 2.47e-03 | 1.76e-03 |
| sin(4πx) | 1.77e-02 | 9.42e-03 | 9.98e-03 | 1.22e-02 |
| sin(2πx)+0.5·sin(4πx) | 2.02e-02 | 5.91e-03 | 5.19e-03 | 4.42e-03 |
| sin(2πx)+0.5·sin(6πx) | 2.83e-02 | 5.40e-03 | 4.49e-03 | 3.37e-03 |
| 3-mode mix (k=1,2,4) | 2.04e-02 | 5.56e-03 | 4.76e-03 | 3.84e-03 |

Similar to heat — the reaction term $r$ shifts the effective eigenvalue decay rate but doesn't alter the structure.
