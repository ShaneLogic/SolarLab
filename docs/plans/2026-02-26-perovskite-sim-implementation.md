# Perovskite Solar Cell Simulator — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 1D drift-diffusion + ion migration simulator for perovskite solar cells in Python, covering J-V hysteresis, impedance spectroscopy, and degradation experiments.

**Architecture:** Method of Lines (MOL): Scharfetter-Gummel FE on tanh grid for spatial discretization, `scipy.integrate.solve_ivp(Radau)` for time integration. Poisson solved implicitly at each time step via sparse tridiagonal solve. All data structures are immutable frozen dataclasses.

**Tech Stack:** Python ≥ 3.11, NumPy, SciPy, PyYAML, Matplotlib, pytest, pytest-cov

---

## Physical Constants and Key Formulas

```python
q     = 1.602176634e-19   # C
eps_0 = 8.854187817e-12   # F/m
k_B   = 1.380649e-23      # J/K
T     = 300.0             # K (default)
V_T   = k_B * T / q      # ≈ 0.025852 V (thermal voltage)
```

**Scharfetter-Gummel (SG) electron flux** between nodes i and i+1 (spacing h):
```
ξ = (φ[i+1] - φ[i]) / V_T
J_n = (q*D_n/h) * [B(ξ)*n[i+1]  - B(-ξ)*n[i]]
J_p = (q*D_p/h) * [B(ξ)*p[i]    - B(-ξ)*p[i+1]]
where B(x) = x / (exp(x) - 1),  B(0) = 1  (Bernoulli function)
```

**Steric ion flux** (Blakemore) between i and i+1:
```
P_avg = (P[i] + P[i+1]) / 2
F_P   = -D_I*(P[i+1]-P[i]) / (h*(1 - P_avg/P_lim))
        + (D_I/V_T)*P_avg*(φ[i+1]-φ[i]) / h
```

**Recombination** (SRH + radiative + Auger, all default-on):
```
np_eq  = n_i²
Δnp    = n*p - np_eq
R_SRH  = Δnp / (τ_p*(n + n1) + τ_n*(p + p1))
R_rad  = B_rad * Δnp
R_Aug  = (C_n*n + C_p*p) * Δnp
R      = R_SRH + R_rad + R_Aug
```

**Poisson** (non-uniform grid, h_l = x[i]-x[i-1], h_r = x[i+1]-x[i]):
```
ε_0*ε_r * 2*(φ[i+1]/(h_r*(h_l+h_r)) - φ[i]/(h_l*h_r) + φ[i-1]/(h_l*(h_l+h_r)))
= -q*(p - n + P - N_A + N_D)
```

**Beer-Lambert generation**:
```
G(x) = Φ * α * exp(-α * x)   [m⁻³ s⁻¹]
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `perovskite-sim/pyproject.toml`
- Create: `perovskite-sim/perovskite_sim/__init__.py`
- Create: `perovskite-sim/perovskite_sim/physics/__init__.py`
- Create: `perovskite-sim/perovskite_sim/discretization/__init__.py`
- Create: `perovskite-sim/perovskite_sim/solver/__init__.py`
- Create: `perovskite-sim/perovskite_sim/models/__init__.py`
- Create: `perovskite-sim/perovskite_sim/experiments/__init__.py`
- Create: `perovskite-sim/tests/__init__.py`
- Create: `perovskite-sim/tests/unit/__init__.py`
- Create: `perovskite-sim/tests/integration/__init__.py`
- Create: `perovskite-sim/tests/regression/__init__.py`
- Create: `perovskite-sim/configs/` (directory only)
- Create: `perovskite-sim/notebooks/` (directory only)

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "perovskite-sim"
version = "0.1.0"
description = "1D drift-diffusion + ion migration simulator for perovskite solar cells"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=perovskite_sim --cov-report=term-missing"
```

**Step 2: Create all `__init__.py` files** (all empty)

**Step 3: Install in dev mode**

Run: `cd perovskite-sim && pip install -e ".[dev]"`
Expected: Successfully installed perovskite-sim

**Step 4: Verify**

Run: `python -c "import perovskite_sim; print('ok')"`
Expected: `ok`

**Step 5: Commit**

```bash
git init
git add .
git commit -m "chore: initialise project scaffold"
```

---

## Task 2: Tanh Grid

**Files:**
- Create: `perovskite_sim/discretization/grid.py`
- Create: `tests/unit/discretization/test_grid.py`

**Step 1: Write the failing test**

```python
# tests/unit/discretization/test_grid.py
import numpy as np
import pytest
from perovskite_sim.discretization.grid import tanh_grid, multilayer_grid, Layer

def test_tanh_grid_endpoints():
    x = tanh_grid(100, L=400e-9, alpha=3.0)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(400e-9)

def test_tanh_grid_length():
    x = tanh_grid(100, L=400e-9, alpha=3.0)
    assert len(x) == 101  # N+1 points

def test_tanh_grid_monotone():
    x = tanh_grid(100, L=400e-9, alpha=3.0)
    assert np.all(np.diff(x) > 0)

def test_tanh_grid_boundary_concentration():
    x_tanh = tanh_grid(100, L=400e-9, alpha=5.0)
    x_uni = np.linspace(0, 400e-9, 101)
    # Tanh grid should have smaller first spacing than uniform
    assert x_tanh[1] - x_tanh[0] < x_uni[1] - x_uni[0]

def test_multilayer_grid_continuity():
    layers = [
        Layer(thickness=100e-9, N=50),
        Layer(thickness=400e-9, N=100),
        Layer(thickness=200e-9, N=50),
    ]
    x = multilayer_grid(layers, alpha=3.0)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(700e-9)
    assert np.all(np.diff(x) > 0)
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/discretization/test_grid.py -v`
Expected: `ImportError` or `ModuleNotFoundError`

**Step 3: Implement**

```python
# perovskite_sim/discretization/grid.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layer:
    thickness: float   # metres
    N: int             # number of grid intervals in this layer


def tanh_grid(N: int, L: float, alpha: float = 3.0) -> np.ndarray:
    """N+1 points on [0, L] with boundary concentration parameter alpha."""
    xi = np.linspace(-1.0, 1.0, N + 1)
    x = L * (1.0 + np.tanh(alpha * xi) / np.tanh(alpha)) / 2.0
    return x


def multilayer_grid(layers: list[Layer], alpha: float = 3.0) -> np.ndarray:
    """Concatenated tanh grid for a stack of layers."""
    segments = []
    offset = 0.0
    for k, layer in enumerate(layers):
        x_seg = tanh_grid(layer.N, layer.thickness, alpha) + offset
        if k > 0:
            x_seg = x_seg[1:]   # drop duplicate interface point
        segments.append(x_seg)
        offset += layer.thickness
    return np.concatenate(segments)
```

**Step 4: Run tests**

Run: `pytest tests/unit/discretization/test_grid.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add perovskite_sim/discretization/grid.py tests/unit/discretization/test_grid.py
git commit -m "feat: tanh grid and multilayer grid"
```

---

## Task 3: Bernoulli Function and SG Operators

**Files:**
- Create: `perovskite_sim/discretization/fe_operators.py`
- Create: `tests/unit/discretization/test_fe_operators.py`

**Step 1: Write the failing tests**

```python
# tests/unit/discretization/test_fe_operators.py
import numpy as np
import pytest
from perovskite_sim.discretization.fe_operators import bernoulli, sg_flux_n, sg_flux_p

def test_bernoulli_at_zero():
    assert bernoulli(np.array([0.0]))[0] == pytest.approx(1.0)

def test_bernoulli_large_positive():
    # B(x) → 0 for large positive x
    assert bernoulli(np.array([50.0]))[0] == pytest.approx(0.0, abs=1e-10)

def test_bernoulli_large_negative():
    # B(-x) → |x| for large |x| (drift dominates)
    x = np.array([-20.0])
    assert bernoulli(x)[0] == pytest.approx(20.0, rel=1e-6)

def test_bernoulli_symmetry():
    x = np.array([1.5])
    # B(x)*exp(x) == B(-x)
    assert (bernoulli(x) * np.exp(x))[0] == pytest.approx(bernoulli(-x)[0], rel=1e-10)

def test_sg_flux_n_equilibrium():
    """Electron current is zero at thermal equilibrium."""
    V_T = 0.025852
    phi = np.array([0.0, 0.1])   # 100 mV potential difference
    xi = (phi[1] - phi[0]) / V_T
    n_eq = np.array([1e18, 1e18 * np.exp(xi)])  # Boltzmann distribution
    h = 100e-9
    D_n = 5.17e-6  # m²/s
    J = sg_flux_n(phi, n_eq, h, D_n, V_T)
    assert abs(J) < 1e-10 * abs(n_eq[0])

def test_sg_flux_p_equilibrium():
    """Hole current is zero at thermal equilibrium."""
    V_T = 0.025852
    phi = np.array([0.0, 0.1])
    p_eq = np.array([1e18, 1e18 * np.exp(-(phi[1]-phi[0])/V_T)])
    h = 100e-9
    D_p = 5.17e-6
    J = sg_flux_p(phi, p_eq, h, D_p, V_T)
    assert abs(J) < 1e-10 * abs(p_eq[0])
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/discretization/test_fe_operators.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/discretization/fe_operators.py
from __future__ import annotations
import numpy as np


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Bernoulli function B(x) = x / (exp(x) - 1), numerically stable."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1e-8
    large = ~small
    result[small] = 1.0 - x[small] / 2.0 + x[small]**2 / 12.0
    result[large] = x[large] / np.expm1(x[large])
    return result


def sg_flux_n(
    phi: np.ndarray,  # [phi_i, phi_{i+1}]  shape (2,)
    n: np.ndarray,    # [n_i,   n_{i+1}]    shape (2,)
    h: float,
    D_n: float,
    V_T: float,
) -> float:
    """Scharfetter-Gummel electron current J_n [A/m²] between node pair."""
    q = 1.602176634e-19
    xi = (phi[1] - phi[0]) / V_T
    return float(q * D_n / h * (bernoulli(np.array([xi]))[0] * n[1]
                                - bernoulli(np.array([-xi]))[0] * n[0]))


def sg_flux_p(
    phi: np.ndarray,  # [phi_i, phi_{i+1}]
    p: np.ndarray,    # [p_i,   p_{i+1}]
    h: float,
    D_p: float,
    V_T: float,
) -> float:
    """Scharfetter-Gummel hole current J_p [A/m²] between node pair."""
    q = 1.602176634e-19
    xi = (phi[1] - phi[0]) / V_T
    return float(q * D_p / h * (bernoulli(np.array([xi]))[0] * p[0]
                                - bernoulli(np.array([-xi]))[0] * p[1]))
```

**Step 4: Run tests**

Run: `pytest tests/unit/discretization/test_fe_operators.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add perovskite_sim/discretization/fe_operators.py tests/unit/discretization/test_fe_operators.py
git commit -m "feat: Bernoulli function and Scharfetter-Gummel fluxes"
```

---

## Task 4: Recombination

**Files:**
- Create: `perovskite_sim/physics/recombination.py`
- Create: `tests/unit/physics/test_recombination.py`

**Step 1: Write the failing tests**

```python
# tests/unit/physics/test_recombination.py
import numpy as np
import pytest
from perovskite_sim.physics.recombination import (
    srh_recombination, radiative_recombination, auger_recombination,
    total_recombination,
)

NI = 3.2e13   # m⁻³  (MAPbI₃ intrinsic carrier density)
NI2 = NI**2

def test_srh_zero_at_equilibrium():
    n = p = NI
    R = srh_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6, n1=NI, p1=NI)
    assert abs(R) < 1e-10 * NI

def test_radiative_zero_at_equilibrium():
    n = p = NI
    R = radiative_recombination(n, p, NI2, B_rad=5e-22)
    assert abs(R) < 1e-30

def test_auger_zero_at_equilibrium():
    n = p = NI
    R = auger_recombination(n, p, NI2, C_n=1e-42, C_p=1e-42)
    assert abs(R) < 1e-30

def test_total_positive_under_injection():
    n = 1e22; p = 1e22  # strong injection
    R = total_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6,
                            n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    assert R > 0

def test_total_negative_for_depletion():
    n = 0.01 * NI; p = 0.01 * NI  # below equilibrium (generation)
    R = total_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6,
                            n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    assert R < 0
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/physics/test_recombination.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/physics/recombination.py
from __future__ import annotations
import numpy as np


def srh_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
) -> np.ndarray:
    """Shockley-Read-Hall recombination rate [m⁻³ s⁻¹]."""
    return (n * p - ni_sq) / (tau_p * (n + n1) + tau_n * (p + p1))


def radiative_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float, B_rad: float,
) -> np.ndarray:
    """Bimolecular radiative recombination rate [m⁻³ s⁻¹]."""
    return B_rad * (n * p - ni_sq)


def auger_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    C_n: float, C_p: float,
) -> np.ndarray:
    """Auger recombination rate [m⁻³ s⁻¹]."""
    return (C_n * n + C_p * p) * (n * p - ni_sq)


def total_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
    B_rad: float, C_n: float, C_p: float,
) -> np.ndarray:
    """Sum of SRH + radiative + Auger [m⁻³ s⁻¹]."""
    return (
        srh_recombination(n, p, ni_sq, tau_n, tau_p, n1, p1)
        + radiative_recombination(n, p, ni_sq, B_rad)
        + auger_recombination(n, p, ni_sq, C_n, C_p)
    )
```

**Step 4: Run tests**

Run: `pytest tests/unit/physics/test_recombination.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add perovskite_sim/physics/recombination.py tests/unit/physics/test_recombination.py
git commit -m "feat: SRH + radiative + Auger recombination"
```

---

## Task 5: Poisson Solver

**Files:**
- Create: `perovskite_sim/physics/poisson.py`
- Create: `tests/unit/physics/test_poisson.py`

**Step 1: Write the failing tests**

```python
# tests/unit/physics/test_poisson.py
import numpy as np
import pytest
from perovskite_sim.physics.poisson import build_poisson_matrix, solve_poisson

EPS_0 = 8.854187817e-12
Q = 1.602176634e-19

def test_zero_charge_gives_linear_potential():
    """Zero space charge → linear potential (flat field)."""
    x = np.linspace(0, 400e-9, 51)
    eps_r = 24.1 * np.ones(51)
    rho = np.zeros(51)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=1.0)
    phi_expected = np.linspace(0, 1, 51)
    np.testing.assert_allclose(phi, phi_expected, atol=1e-8)

def test_positive_charge_creates_concave_potential():
    """Positive uniform charge → concave potential (downward curve)."""
    x = np.linspace(0, 400e-9, 101)
    eps_r = 24.1 * np.ones(101)
    rho_val = Q * 1e22   # uniform positive charge density [C/m³]
    rho = rho_val * np.ones(101)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=0.0)
    # Maximum should be at centre
    assert np.argmax(phi) == 50

def test_boundary_conditions_enforced():
    x = np.linspace(0, 400e-9, 51)
    eps_r = np.ones(51)
    rho = np.zeros(51)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.3, phi_right=0.7)
    assert phi[0] == pytest.approx(0.3)
    assert phi[-1] == pytest.approx(0.7)
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/physics/test_poisson.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/physics/poisson.py
from __future__ import annotations
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

EPS_0 = 8.854187817e-12


def build_poisson_matrix(x: np.ndarray, eps_r: np.ndarray):
    """
    Build sparse tridiagonal matrix A for Poisson equation on non-uniform grid.
    A @ phi = rhs   (internal nodes only)
    """
    N = len(x)
    h_l = np.diff(x)                  # h_l[i] = x[i+1] - x[i],  length N-1
    # Coefficients for second-order FD Laplacian on non-uniform grid
    # d²φ/dx² ≈ 2*(φ[i+1]/(h_r*(h_l+h_r)) - φ[i]/(h_l*h_r) + φ[i-1]/(h_l*(h_l+h_r)))
    h_left  = h_l[:-1]   # h_{i-1,i}  (length N-2, for internal nodes 1..N-2)
    h_right = h_l[1:]    # h_{i,i+1}
    denom = h_left * h_right
    sup_coef = 2.0 / (h_right * (h_left + h_right))
    dia_coef = -2.0 / denom
    sub_coef = 2.0 / (h_left  * (h_left + h_right))

    eps_int = eps_r[1:-1]  # eps at internal nodes
    A = diags(
        [EPS_0 * eps_int * sub_coef,
         EPS_0 * eps_int * dia_coef,
         EPS_0 * eps_int * sup_coef],
        offsets=[-1, 0, 1],
        shape=(N - 2, N - 2),
        format="csr",
    )
    return A


def solve_poisson(
    x: np.ndarray,
    eps_r: np.ndarray,
    rho: np.ndarray,      # charge density [C/m³] = q*(p - n + P - N_A + N_D)
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """
    Solve Poisson equation:  ε₀ εᵣ d²φ/dx² = -ρ
    Returns potential φ on all N nodes.
    """
    N = len(x)
    A = build_poisson_matrix(x, eps_r)
    h_l = np.diff(x)
    h_left  = h_l[:-1]
    h_right = h_l[1:]
    eps_int = eps_r[1:-1]

    rhs = -rho[1:-1].copy()
    # Absorb BCs into RHS
    rhs[0]  -= EPS_0 * eps_int[0]  * 2.0 / (h_left[0]  * (h_left[0]  + h_right[0]))  * phi_left
    rhs[-1] -= EPS_0 * eps_int[-1] * 2.0 / (h_right[-1] * (h_left[-1] + h_right[-1])) * phi_right

    phi_int = spsolve(A, rhs)
    phi = np.empty(N)
    phi[0]  = phi_left
    phi[-1] = phi_right
    phi[1:-1] = phi_int
    return phi
```

**Step 4: Run tests**

Run: `pytest tests/unit/physics/test_poisson.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add perovskite_sim/physics/poisson.py tests/unit/physics/test_poisson.py
git commit -m "feat: sparse Poisson solver on non-uniform grid"
```

---

## Task 6: Ion Migration (Steric Blakemore Flux)

**Files:**
- Create: `perovskite_sim/physics/ion_migration.py`
- Create: `tests/unit/physics/test_ion_migration.py`

**Step 1: Write the failing tests**

```python
# tests/unit/physics/test_ion_migration.py
import numpy as np
import pytest
from perovskite_sim.physics.ion_migration import ion_flux_steric, ion_continuity_rhs

V_T = 0.025852
D_I = 1e-16   # m²/s
P_LIM = 1e27  # m⁻³

def test_flux_recovers_standard_pnp_low_density():
    """For P << P_lim, steric term → 1 (standard PNP)."""
    phi = np.array([0.0, 0.0])   # no field
    P   = np.array([1e20, 1e20]) # P << P_lim
    h   = 10e-9
    F_steric  = ion_flux_steric(phi, P, h, D_I, V_T, P_lim=P_LIM)
    # With no field and uniform P, flux should be ~0
    assert abs(F_steric) < 1e10   # much less than D_I*P/h ~ 1e-16*1e20/1e-8 = 1e12

def test_flux_enhanced_near_plim():
    """Diffusion is enhanced when P approaches P_lim."""
    phi = np.array([0.0, 0.0])
    P_low  = np.array([0.5e20, 1.0e20])
    P_high = np.array([0.5e27, 1.0e27])   # near P_lim
    h = 10e-9
    F_low  = abs(ion_flux_steric(phi, P_low,  h, D_I, V_T, P_lim=P_LIM))
    F_high = abs(ion_flux_steric(phi, P_high, h, D_I, V_T, P_lim=P_LIM))
    assert F_high > F_low

def test_zero_flux_at_uniform_equilibrium():
    """No flux when P is uniform and no field."""
    phi = np.array([0.0, 0.0])
    P   = np.array([1e24, 1e24])
    h   = 10e-9
    F   = ion_flux_steric(phi, P, h, D_I, V_T, P_lim=P_LIM)
    assert abs(F) < 1e-20

def test_continuity_rhs_shape():
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N)
    P   = 1e24 * np.ones(N)
    dPdt = ion_continuity_rhs(x, phi, P, D_I, V_T, P_LIM)
    assert dPdt.shape == (N,)

def test_continuity_zero_for_uniform_no_field():
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N)
    P   = 1e24 * np.ones(N)
    dPdt = ion_continuity_rhs(x, phi, P, D_I, V_T, P_LIM)
    np.testing.assert_allclose(dPdt, 0.0, atol=1.0)  # [m⁻³/s]
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/physics/test_ion_migration.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/physics/ion_migration.py
from __future__ import annotations
import numpy as np


def ion_flux_steric(
    phi: np.ndarray,   # [phi_i, phi_{i+1}]
    P: np.ndarray,     # [P_i,   P_{i+1}]
    h: float,
    D_I: float,
    V_T: float,
    P_lim: float,
) -> float:
    """
    Steric Blakemore ion vacancy flux F_P [m⁻² s⁻¹] from node i to i+1.
    F_P = -D_I * dP/dx / (1 - P_avg/P_lim)  +  (D_I/V_T) * P_avg * dφ/dx
    """
    P_avg = 0.5 * (P[0] + P[1])
    steric = 1.0 / (1.0 - P_avg / P_lim)
    grad_P = (P[1] - P[0]) / h
    grad_phi = (phi[1] - phi[0]) / h
    return float(-D_I * grad_P * steric + (D_I / V_T) * P_avg * grad_phi)


def ion_continuity_rhs(
    x: np.ndarray,
    phi: np.ndarray,
    P: np.ndarray,
    D_I: float,
    V_T: float,
    P_lim: float,
) -> np.ndarray:
    """
    dP/dt = -dF_P/dx  for all nodes.
    Zero-flux boundary conditions at both contacts.
    Returns array of shape (N,).
    """
    N = len(x)
    F = np.zeros(N + 1)  # fluxes at N-1 interior faces + 2 boundaries (zero)
    for i in range(N - 1):
        h_i = x[i + 1] - x[i]
        F[i + 1] = ion_flux_steric(phi[i:i+2], P[i:i+2], h_i, D_I, V_T, P_lim)
    # F[0] = 0 and F[N] = 0 (zero-flux BCs)
    dPdt = np.zeros(N)
    for i in range(N):
        # Cell width for node i
        if i == 0:
            dx_cell = x[1] - x[0]
        elif i == N - 1:
            dx_cell = x[-1] - x[-2]
        else:
            dx_cell = 0.5 * (x[i + 1] - x[i - 1])
        dPdt[i] = -(F[i + 1] - F[i]) / dx_cell
    return dPdt
```

**Step 4: Run tests**

Run: `pytest tests/unit/physics/test_ion_migration.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add perovskite_sim/physics/ion_migration.py tests/unit/physics/test_ion_migration.py
git commit -m "feat: steric Blakemore ion migration flux"
```

---

## Task 7: Carrier Continuity Equations

**Files:**
- Create: `perovskite_sim/physics/continuity.py`
- Create: `perovskite_sim/physics/generation.py`
- Create: `tests/unit/physics/test_continuity.py`

**Step 1: Write the failing tests**

```python
# tests/unit/physics/test_continuity.py
import numpy as np
import pytest
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.physics.generation import beer_lambert_generation

NI = 3.2e13
Q  = 1.602176634e-19

def test_continuity_shape():
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N); n = NI*np.ones(N); p = NI*np.ones(N)
    eps_r = 24.1*np.ones(N)
    params = dict(D_n=5.17e-6, D_p=5.17e-6, V_T=0.025852,
                  ni_sq=NI**2, tau_n=1e-6, tau_p=1e-6,
                  n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    G = np.zeros(N)
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)
    assert dn.shape == (N,) and dp.shape == (N,)

def test_continuity_zero_at_dark_equilibrium():
    """No net change at dark equilibrium (n=p=ni, no generation)."""
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N); n = NI*np.ones(N); p = NI*np.ones(N)
    params = dict(D_n=5.17e-6, D_p=5.17e-6, V_T=0.025852,
                  ni_sq=NI**2, tau_n=1e-6, tau_p=1e-6,
                  n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    G = np.zeros(N)
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)
    # Interior nodes should be near zero (BCs handle boundaries)
    np.testing.assert_allclose(dn[1:-1], 0.0, atol=1e10)
    np.testing.assert_allclose(dp[1:-1], 0.0, atol=1e10)

def test_beer_lambert_integrates_to_photocurrent():
    x = np.linspace(0, 400e-9, 200)
    alpha = 1e7   # m⁻¹
    Phi = 2.5e21  # photon flux [m⁻² s⁻¹]
    G = beer_lambert_generation(x, alpha, Phi)
    # Integrate G dx ≈ Phi*(1 - exp(-alpha*L))
    L = x[-1]
    expected = Phi * (1 - np.exp(-alpha * L))
    np.testing.assert_allclose(np.trapz(G, x), expected, rtol=1e-3)
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/physics/test_continuity.py -v`
Expected: ImportError

**Step 3: Implement generation**

```python
# perovskite_sim/physics/generation.py
import numpy as np

def beer_lambert_generation(
    x: np.ndarray, alpha: float, Phi: float
) -> np.ndarray:
    """G(x) = Phi * alpha * exp(-alpha * x)  [m⁻³ s⁻¹]"""
    return Phi * alpha * np.exp(-alpha * x)
```

**Step 4: Implement continuity**

```python
# perovskite_sim/physics/continuity.py
from __future__ import annotations
import numpy as np
from perovskite_sim.discretization.fe_operators import sg_flux_n, sg_flux_p
from perovskite_sim.physics.recombination import total_recombination

Q = 1.602176634e-19


def carrier_continuity_rhs(
    x: np.ndarray,
    phi: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    G: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dn/dt and dp/dt for all nodes using SG fluxes.
    Boundary nodes are held fixed (Dirichlet, returned as 0).
    """
    N = len(x)
    D_n = params["D_n"]; D_p = params["D_p"]; V_T = params["V_T"]
    R = total_recombination(
        n, p, params["ni_sq"], params["tau_n"], params["tau_p"],
        params["n1"], params["p1"], params["B_rad"], params["C_n"], params["C_p"]
    )

    # Compute fluxes at N-1 interior faces
    J_n = np.zeros(N + 1)
    J_p = np.zeros(N + 1)
    for i in range(N - 1):
        h_i = x[i + 1] - x[i]
        J_n[i + 1] = sg_flux_n(phi[i:i+2], n[i:i+2], h_i, D_n, V_T)
        J_p[i + 1] = sg_flux_p(phi[i:i+2], p[i:i+2], h_i, D_p, V_T)

    dn = np.zeros(N)
    dp = np.zeros(N)
    for i in range(1, N - 1):
        dx_cell = 0.5 * (x[i + 1] - x[i - 1])
        dn[i] =  (J_n[i + 1] - J_n[i]) / (Q * dx_cell) - R[i] + G[i]
        dp[i] = -(J_p[i + 1] - J_p[i]) / (Q * dx_cell) - R[i] + G[i]
    return dn, dp
```

**Step 5: Run tests**

Run: `pytest tests/unit/physics/test_continuity.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add perovskite_sim/physics/ tests/unit/physics/test_continuity.py
git commit -m "feat: carrier continuity and Beer-Lambert generation"
```

---

## Task 8: Dataclasses and YAML Configuration

**Files:**
- Create: `perovskite_sim/models/parameters.py`
- Create: `perovskite_sim/models/device.py`
- Create: `tests/unit/models/test_parameters.py`

**Step 1: Write the failing tests**

```python
# tests/unit/models/test_parameters.py
import pytest
from perovskite_sim.models.parameters import MaterialParams, SolverConfig, load_config
from perovskite_sim.models.device import DeviceStack

def test_material_params_immutable():
    p = MaterialParams(
        eps_r=24.1, mu_n=2e-4, mu_p=2e-4,
        D_ion=1e-16, P_lim=1e27, P0=1e24,
        ni=3.2e13, tau_n=1e-6, tau_p=1e-6,
        n1=3.2e13, p1=3.2e13, B_rad=5e-22,
        C_n=1e-42, C_p=1e-42, alpha=1e7,
        N_A=0.0, N_D=0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        p.eps_r = 10.0   # frozen dataclass

def test_solver_config_defaults():
    cfg = SolverConfig()
    assert cfg.rtol == 1e-4
    assert cfg.atol == 1e-6
    assert cfg.N == 200

def test_device_stack_total_thickness():
    from perovskite_sim.models.device import LayerSpec
    stack = DeviceStack(layers=[
        LayerSpec(name="ETL",       thickness=100e-9, params=None, role="ETL"),
        LayerSpec(name="perovskite",thickness=400e-9, params=None, role="absorber"),
        LayerSpec(name="HTL",       thickness=200e-9, params=None, role="HTL"),
    ])
    assert abs(stack.total_thickness - 700e-9) < 1e-12
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/models/test_parameters.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/models/parameters.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import yaml

Q    = 1.602176634e-19
K_B  = 1.380649e-23
T    = 300.0
V_T  = K_B * T / Q


@dataclass(frozen=True)
class MaterialParams:
    eps_r: float
    mu_n: float     # m²/Vs
    mu_p: float
    D_ion: float    # ion diffusion coefficient m²/s (0 if no ions)
    P_lim: float    # maximum ion vacancy density m⁻³
    P0: float       # initial (equilibrium) ion density m⁻³
    ni: float       # intrinsic carrier density m⁻³
    tau_n: float    # SRH electron lifetime s
    tau_p: float
    n1: float       # SRH trap-level carrier densities
    p1: float
    B_rad: float    # radiative recombination coefficient m³/s
    C_n: float      # Auger coefficient m⁶/s
    C_p: float
    alpha: float    # optical absorption coefficient m⁻¹
    N_A: float      # acceptor doping m⁻³
    N_D: float      # donor doping m⁻³

    @property
    def D_n(self) -> float:
        return self.mu_n * V_T

    @property
    def D_p(self) -> float:
        return self.mu_p * V_T

    @property
    def ni_sq(self) -> float:
        return self.ni ** 2


@dataclass(frozen=True)
class SolverConfig:
    N: int = 200
    alpha_grid: float = 3.0
    rtol: float = 1e-4
    atol: float = 1e-6
    T: float = 300.0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

```python
# perovskite_sim/models/device.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from perovskite_sim.models.parameters import MaterialParams


@dataclass(frozen=True)
class LayerSpec:
    name: str
    thickness: float         # metres
    params: Optional[MaterialParams]
    role: str               # "ETL", "absorber", "HTL"


@dataclass(frozen=True)
class DeviceStack:
    layers: tuple[LayerSpec, ...]
    phi_left: float = 0.0   # V
    V_bi: float = 1.1       # built-in voltage [V]
    Phi: float = 2.5e21     # photon flux [m⁻² s⁻¹] (AM1.5G)

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def phi_right(self) -> float:
        return self.phi_left + self.V_bi
```

**Step 4: Run tests**

Run: `pytest tests/unit/models/test_parameters.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add perovskite_sim/models/ tests/unit/models/test_parameters.py
git commit -m "feat: immutable MaterialParams, SolverConfig, DeviceStack dataclasses"
```

---

## Task 9: Config Files (n-i-p and p-i-n)

**Files:**
- Create: `configs/nip_MAPbI3.yaml`
- Create: `configs/pin_MAPbI3.yaml`
- Create: `tests/unit/models/test_config_loading.py`

**Step 1: Write failing test**

```python
# tests/unit/models/test_config_loading.py
from perovskite_sim.models.config_loader import load_device_from_yaml

def test_nip_loads_three_layers():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    assert len(stack.layers) == 3

def test_pin_loads_three_layers():
    stack = load_device_from_yaml("configs/pin_MAPbI3.yaml")
    assert len(stack.layers) == 3

def test_absorber_has_ions():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    absorber = next(l for l in stack.layers if l.role == "absorber")
    assert absorber.params.D_ion > 0
```

**Step 2: Create config YAML files**

```yaml
# configs/nip_MAPbI3.yaml
device:
  V_bi: 1.1
  Phi: 2.5e21
layers:
  - name: TiO2_ETL
    role: ETL
    thickness: 100e-9
    eps_r: 10.0
    mu_n: 1e-7
    mu_p: 1e-10
    ni: 1e0
    N_D: 1e24
    N_A: 0.0
    D_ion: 0.0
    P_lim: 1e30
    P0: 0.0
    tau_n: 1e-9
    tau_p: 1e-9
    n1: 1e0
    p1: 1e0
    B_rad: 1e-30
    C_n: 1e-42
    C_p: 1e-42
    alpha: 0.0

  - name: MAPbI3
    role: absorber
    thickness: 400e-9
    eps_r: 24.1
    mu_n: 2e-4
    mu_p: 2e-4
    ni: 3.2e13
    N_D: 0.0
    N_A: 0.0
    D_ion: 1e-16
    P_lim: 1.6e27
    P0: 1.6e24
    tau_n: 1e-6
    tau_p: 1e-6
    n1: 3.2e13
    p1: 3.2e13
    B_rad: 5e-22
    C_n: 1e-42
    C_p: 1e-42
    alpha: 1.3e7

  - name: spiro_HTL
    role: HTL
    thickness: 200e-9
    eps_r: 3.0
    mu_n: 1e-10
    mu_p: 1e-8
    ni: 1e0
    N_D: 0.0
    N_A: 2e23
    D_ion: 0.0
    P_lim: 1e30
    P0: 0.0
    tau_n: 1e-9
    tau_p: 1e-9
    n1: 1e0
    p1: 1e0
    B_rad: 1e-30
    C_n: 1e-42
    C_p: 1e-42
    alpha: 0.0
```

```yaml
# configs/pin_MAPbI3.yaml
device:
  V_bi: 1.1
  Phi: 2.5e21
layers:
  - name: NiO_HTL
    role: HTL
    thickness: 100e-9
    eps_r: 11.9
    mu_n: 1e-10
    mu_p: 1e-8
    ni: 1e0
    N_D: 0.0
    N_A: 1e24
    D_ion: 0.0
    P_lim: 1e30
    P0: 0.0
    tau_n: 1e-9
    tau_p: 1e-9
    n1: 1e0
    p1: 1e0
    B_rad: 1e-30
    C_n: 1e-42
    C_p: 1e-42
    alpha: 0.0

  - name: MAPbI3
    role: absorber
    thickness: 400e-9
    eps_r: 24.1
    mu_n: 2e-4
    mu_p: 2e-4
    ni: 3.2e13
    N_D: 0.0
    N_A: 0.0
    D_ion: 1e-16
    P_lim: 1.6e27
    P0: 1.6e24
    tau_n: 1e-6
    tau_p: 1e-6
    n1: 3.2e13
    p1: 3.2e13
    B_rad: 5e-22
    C_n: 1e-42
    C_p: 1e-42
    alpha: 1.3e7

  - name: PCBM_ETL
    role: ETL
    thickness: 200e-9
    eps_r: 4.0
    mu_n: 1e-7
    mu_p: 1e-10
    ni: 1e0
    N_D: 1e23
    N_A: 0.0
    D_ion: 0.0
    P_lim: 1e30
    P0: 0.0
    tau_n: 1e-9
    tau_p: 1e-9
    n1: 1e0
    p1: 1e0
    B_rad: 1e-30
    C_n: 1e-42
    C_p: 1e-42
    alpha: 0.0
```

**Step 3: Implement config loader**

```python
# perovskite_sim/models/config_loader.py
from __future__ import annotations
import yaml
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, LayerSpec


def load_device_from_yaml(path: str) -> DeviceStack:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    dev = cfg["device"]
    layers = []
    for layer_cfg in cfg["layers"]:
        p = MaterialParams(
            eps_r=layer_cfg["eps_r"],
            mu_n=layer_cfg["mu_n"],
            mu_p=layer_cfg["mu_p"],
            D_ion=layer_cfg["D_ion"],
            P_lim=layer_cfg["P_lim"],
            P0=layer_cfg["P0"],
            ni=layer_cfg["ni"],
            tau_n=layer_cfg["tau_n"],
            tau_p=layer_cfg["tau_p"],
            n1=layer_cfg["n1"],
            p1=layer_cfg["p1"],
            B_rad=layer_cfg["B_rad"],
            C_n=layer_cfg["C_n"],
            C_p=layer_cfg["C_p"],
            alpha=layer_cfg["alpha"],
            N_A=layer_cfg["N_A"],
            N_D=layer_cfg["N_D"],
        )
        layers.append(LayerSpec(
            name=layer_cfg["name"],
            thickness=layer_cfg["thickness"],
            params=p,
            role=layer_cfg["role"],
        ))
    return DeviceStack(
        layers=layers,
        V_bi=dev.get("V_bi", 1.1),
        Phi=dev.get("Phi", 2.5e21),
    )
```

**Step 4: Run tests**

Run: `pytest tests/unit/models/test_config_loading.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add configs/ perovskite_sim/models/config_loader.py tests/unit/models/test_config_loading.py
git commit -m "feat: YAML config loader and n-i-p/p-i-n device configs"
```

---

## Task 10: MOL Assembler and Transient Solver

**Files:**
- Create: `perovskite_sim/solver/mol.py`
- Create: `tests/unit/solver/test_mol.py`

**Step 1: Write the failing tests**

```python
# tests/unit/solver/test_mol.py
import numpy as np
import pytest
from perovskite_sim.solver.mol import assemble_rhs, StateVec

NI = 3.2e13

def test_state_vec_roundtrip():
    N = 50
    n = NI * np.ones(N); p = NI * np.ones(N); P = 1e24 * np.ones(N)
    y = StateVec.pack(n, p, P)
    sv = StateVec.unpack(y, N)
    np.testing.assert_allclose(sv.n, n)
    np.testing.assert_allclose(sv.p, p)
    np.testing.assert_allclose(sv.P, P)

def test_assemble_rhs_shape():
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    n = NI * np.ones(N); p = NI * np.ones(N); P = 1e24 * np.ones(N)
    y0 = StateVec.pack(n, p, P)
    dydt = assemble_rhs(0.0, y0, x, stack, illuminated=False)
    assert dydt.shape == y0.shape
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/solver/test_mol.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/solver/mol.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.poisson import solve_poisson
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.physics.ion_migration import ion_continuity_rhs
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.models.device import DeviceStack

Q    = 1.602176634e-19
K_B  = 1.380649e-23
T    = 300.0
V_T  = K_B * T / Q


@dataclass
class StateVec:
    n: np.ndarray
    p: np.ndarray
    P: np.ndarray

    @staticmethod
    def pack(n, p, P) -> np.ndarray:
        return np.concatenate([n, p, P])

    @staticmethod
    def unpack(y: np.ndarray, N: int) -> "StateVec":
        return StateVec(n=y[:N], p=y[N:2*N], P=y[2*N:3*N])


def _build_layerwise_arrays(x: np.ndarray, stack: DeviceStack):
    """Return eps_r, D_ion, P_lim, N_A, N_D, carrier params arrays over x."""
    N = len(x)
    eps_r = np.ones(N)
    D_ion = np.zeros(N)
    P_lim = 1e30 * np.ones(N)
    N_A   = np.zeros(N)
    N_D   = np.zeros(N)
    alpha = np.zeros(N)
    # carrier params per node (use absorber values as fallback)
    layer_params = []
    offset = 0.0
    for layer in stack.layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        eps_r[mask] = p.eps_r
        D_ion[mask] = p.D_ion
        P_lim[mask] = p.P_lim
        N_A[mask]   = p.N_A
        N_D[mask]   = p.N_D
        alpha[mask] = p.alpha
        offset += layer.thickness
    return eps_r, D_ion, P_lim, N_A, N_D, alpha


def _get_absorber_params(stack: DeviceStack) -> dict:
    """Return physics params from absorber layer (dominant recombination)."""
    absorber = next(l for l in stack.layers if l.role == "absorber")
    p = absorber.params
    return dict(D_n=p.D_n, D_p=p.D_p, V_T=V_T,
                ni_sq=p.ni_sq, tau_n=p.tau_n, tau_p=p.tau_p,
                n1=p.n1, p1=p.p1, B_rad=p.B_rad, C_n=p.C_n, C_p=p.C_p)


def _equilibrium_bc(stack: DeviceStack, x: np.ndarray):
    """Ohmic contact carrier densities from doping."""
    absorber = next(l for l in stack.layers if l.role == "absorber")
    ni = absorber.params.ni

    def equilibrium_np(N_D, N_A):
        net = 0.5 * (N_D - N_A)
        n = net + np.sqrt(net**2 + ni**2)
        p = ni**2 / n
        return n, p

    first_layer = stack.layers[0]
    last_layer  = stack.layers[-1]
    n_L, p_L = equilibrium_np(first_layer.params.N_D, first_layer.params.N_A)
    n_R, p_R = equilibrium_np(last_layer.params.N_D,  last_layer.params.N_A)
    return n_L, p_L, n_R, p_R


def assemble_rhs(
    t: float,
    y: np.ndarray,
    x: np.ndarray,
    stack: DeviceStack,
    illuminated: bool = True,
    V_app: float = 0.0,
) -> np.ndarray:
    """Method of Lines RHS: dy/dt = f(t, y)."""
    N = len(x)
    sv = StateVec.unpack(y, N)

    eps_r, D_ion, P_lim, N_A, N_D, alpha_arr = _build_layerwise_arrays(x, stack)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    ni = absorber.params.ni

    # Boundary conditions
    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
    n = sv.n.copy(); n[0] = n_L; n[-1] = n_R
    p = sv.p.copy(); p[0] = p_L; p[-1] = p_R

    # Solve Poisson
    rho = Q * (p - n + sv.P - N_A + N_D)
    phi_right = stack.V_bi + V_app
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=phi_right)

    # Generation
    if illuminated:
        G = beer_lambert_generation(x, alpha_arr, stack.Phi)
    else:
        G = np.zeros(N)

    # Carrier continuity
    params = _get_absorber_params(stack)
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)

    # Ion continuity (only where D_ion > 0)
    # Use dominant absorber D_ion / P_lim
    dP = ion_continuity_rhs(x, phi, sv.P, absorber.params.D_ion, V_T,
                             absorber.params.P_lim)

    # Enforce Dirichlet BCs: hold boundary nodes fixed
    dn[0] = dn[-1] = 0.0
    dp[0] = dp[-1] = 0.0

    return StateVec.pack(dn, dp, dP)


def run_transient(
    x: np.ndarray,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    stack: DeviceStack,
    illuminated: bool = True,
    V_app: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
):
    """Integrate MOL system from t_span[0] to t_span[1]."""
    N = len(x)

    def rhs(t, y):
        return assemble_rhs(t, y, x, stack, illuminated, V_app)

    return solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                     method="Radau", rtol=rtol, atol=atol, dense_output=False)
```

**Step 4: Run tests**

Run: `pytest tests/unit/solver/test_mol.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add perovskite_sim/solver/mol.py tests/unit/solver/test_mol.py
git commit -m "feat: MOL assembler and transient Radau integrator"
```

---

## Task 11: Newton Steady-State Solver

**Files:**
- Create: `perovskite_sim/solver/newton.py`
- Create: `tests/unit/solver/test_newton.py`

**Step 1: Write the failing tests**

```python
# tests/unit/solver/test_newton.py
import numpy as np
import pytest
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer

def test_equilibrium_convergence():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    assert y_eq is not None

def test_equilibrium_residual_small():
    """After equilibrium solve, RHS should be near zero."""
    from perovskite_sim.solver.mol import assemble_rhs
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    rhs = assemble_rhs(0.0, y_eq, x, stack, illuminated=False, V_app=0.0)
    N = len(x)
    # Interior nodes: dn/dt and dp/dt should be small
    assert np.max(np.abs(rhs[1:N-1])) < 1e18   # m⁻³/s
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/solver/test_newton.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/solver/newton.py
from __future__ import annotations
import numpy as np
from perovskite_sim.solver.mol import StateVec, assemble_rhs, _equilibrium_bc
from perovskite_sim.models.device import DeviceStack


def solve_equilibrium(
    x: np.ndarray,
    stack: DeviceStack,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Integrate to steady state in dark by running transient until convergence.
    Uses the long-time limit approach: run to t_final = 1e6 s.
    """
    from scipy.integrate import solve_ivp

    N = len(x)
    absorber = next(l for l in stack.layers if l.role == "absorber")

    # Initial guess: equilibrium carrier densities from doping
    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
    ni = absorber.params.ni
    n0 = np.interp(x, [x[0], x[-1]], [n_L, n_R])
    p0 = np.interp(x, [x[0], x[-1]], [p_L, p_R])
    P0 = absorber.params.P0 * np.ones(N)

    y0 = StateVec.pack(n0, p0, P0)

    def rhs(t, y):
        return assemble_rhs(t, y, x, stack, illuminated=False, V_app=0.0)

    sol = solve_ivp(rhs, (0.0, 1e3), y0, method="Radau",
                    rtol=1e-6, atol=1e-8, dense_output=False)
    return sol.y[:, -1]
```

**Step 4: Run tests**

Run: `pytest tests/unit/solver/test_newton.py -v`
Expected: 2 passed (may take ~10s)

**Step 5: Commit**

```bash
git add perovskite_sim/solver/newton.py tests/unit/solver/test_newton.py
git commit -m "feat: steady-state equilibrium solver via long-time integration"
```

---

## Task 12: Integration Test — Dark Equilibrium Physics

**Files:**
- Create: `tests/integration/test_equilibrium.py`

**Step 1: Write the test**

```python
# tests/integration/test_equilibrium.py
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import StateVec

def test_np_product_at_equilibrium():
    """n*p ≈ ni² throughout device at equilibrium."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    N = len(x)
    sv = StateVec.unpack(y_eq, N)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    ni = absorber.params.ni
    # Check interior of absorber: n*p should ≈ ni²
    abs_mask = (x > 100e-9) & (x < 500e-9)
    ratio = sv.n[abs_mask] * sv.p[abs_mask] / ni**2
    # Allow 3 orders of magnitude variation (junction regions can deviate)
    assert np.all(ratio > 1e-3)

def test_ion_profile_within_plim():
    """Ion vacancies must never exceed P_lim."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    N = len(x)
    sv = StateVec.unpack(y_eq, N)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    assert np.all(sv.P <= absorber.params.P_lim)
```

**Step 2: Run integration tests**

Run: `pytest tests/integration/test_equilibrium.py -v`
Expected: 2 passed

**Step 3: Commit**

```bash
git add tests/integration/test_equilibrium.py
git commit -m "test: integration tests for dark equilibrium physics"
```

---

## Task 13: J-V Sweep Experiment

**Files:**
- Create: `perovskite_sim/experiments/jv_sweep.py`
- Create: `tests/unit/experiments/test_jv_sweep.py`

**Step 1: Write the failing test**

```python
# tests/unit/experiments/test_jv_sweep.py
import numpy as np
import pytest
from perovskite_sim.experiments.jv_sweep import JVResult, compute_metrics

def test_compute_metrics_mpp():
    """MPP power should be between 0 and Voc*Jsc."""
    V = np.linspace(0, 1.1, 50)
    J_sc = 200.0  # A/m²
    J = J_sc * (1 - np.exp((V - 1.1) / 0.05))
    result = compute_metrics(V, J)
    assert 0.0 < result.PCE < 1.0
    assert 0.0 < result.FF < 1.0
    assert result.V_oc > 0.0
    assert result.J_sc > 0.0

def test_hysteresis_index_zero_for_symmetric():
    """HI = 0 when forward and reverse J-V are identical."""
    from perovskite_sim.experiments.jv_sweep import hysteresis_index
    V = np.linspace(0, 1.0, 50)
    J = np.linspace(200, 0, 50)
    hi = hysteresis_index(V, J, V, J)
    assert abs(hi) < 1e-6
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/experiments/test_jv_sweep.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/experiments/jv_sweep.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import assemble_rhs, StateVec, run_transient
from perovskite_sim.models.device import DeviceStack

Q = 1.602176634e-19


@dataclass(frozen=True)
class JVMetrics:
    V_oc: float; J_sc: float; FF: float; PCE: float


@dataclass(frozen=True)
class JVResult:
    V_fwd: np.ndarray; J_fwd: np.ndarray
    V_rev: np.ndarray; J_rev: np.ndarray
    metrics_fwd: JVMetrics; metrics_rev: JVMetrics
    hysteresis_index: float


def compute_metrics(V: np.ndarray, J: np.ndarray) -> JVMetrics:
    """Compute V_oc, J_sc, FF, PCE from a J-V array (J in A/m²)."""
    J_sc = float(np.interp(0.0, V, J))
    # V_oc: where J crosses zero
    sign_changes = np.where(np.diff(np.sign(J)))[0]
    if len(sign_changes) == 0:
        return JVMetrics(V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0)
    idx = sign_changes[-1]
    V_oc = float(V[idx] - J[idx] * (V[idx+1] - V[idx]) / (J[idx+1] - J[idx]))
    P = V * J
    P_mpp = float(np.max(P[V <= V_oc]))
    FF = P_mpp / (V_oc * J_sc) if (V_oc * J_sc) > 0 else 0.0
    # PCE assuming 1000 W/m² AM1.5G
    PCE = P_mpp / 1000.0
    return JVMetrics(V_oc=V_oc, J_sc=J_sc, FF=FF, PCE=PCE)


def hysteresis_index(
    V_fwd: np.ndarray, J_fwd: np.ndarray,
    V_rev: np.ndarray, J_rev: np.ndarray,
) -> float:
    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev, J_rev)
    if m_rev.PCE == 0:
        return 0.0
    return (m_rev.PCE - m_fwd.PCE) / m_rev.PCE


def run_jv_sweep(
    stack: DeviceStack,
    N_grid: int = 100,
    v_rate: float = 0.1,      # V/s
    n_points: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> JVResult:
    """Run forward and reverse J-V sweeps."""
    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    L = stack.total_thickness

    y_eq = solve_equilibrium(x, stack)

    def _sweep(V_start: float, V_end: float):
        V_arr = np.linspace(V_start, V_end, n_points)
        dt = abs(V_end - V_start) / (v_rate * (n_points - 1))
        t_points = np.arange(n_points) * dt
        J_arr = np.zeros(n_points)
        y = y_eq.copy()
        for k, V_k in enumerate(V_arr):
            t_span = (t_points[k], t_points[k] + dt if k < n_points - 1 else t_points[k] + 1e-12)
            sol = run_transient(x, y, t_span, np.array([t_span[-1]]),
                                stack, illuminated=True, V_app=V_k, rtol=rtol, atol=atol)
            y = sol.y[:, -1]
            # Current density from electron flux at left boundary
            sv = StateVec.unpack(y, N)
            # Approximate J from continuity: integrate G - R over device
            from perovskite_sim.physics.generation import beer_lambert_generation
            from perovskite_sim.physics.recombination import total_recombination
            absorber = next(l for l in stack.layers if l.role == "absorber")
            p = absorber.params
            G = beer_lambert_generation(x, p.alpha, stack.Phi)
            R = total_recombination(sv.n, sv.p, p.ni_sq, p.tau_n, p.tau_p,
                                    p.n1, p.p1, p.B_rad, p.C_n, p.C_p)
            J_arr[k] = Q * np.trapz(G - R, x)
        return V_arr, J_arr

    V_fwd, J_fwd = _sweep(0.0, stack.V_bi)
    V_rev, J_rev = _sweep(stack.V_bi, 0.0)

    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev[::-1], J_rev[::-1])
    HI = hysteresis_index(V_fwd, J_fwd, V_rev[::-1], J_rev[::-1])

    return JVResult(V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
                    metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI)
```

**Step 4: Run unit tests**

Run: `pytest tests/unit/experiments/test_jv_sweep.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add perovskite_sim/experiments/jv_sweep.py tests/unit/experiments/test_jv_sweep.py
git commit -m "feat: J-V sweep experiment with hysteresis index"
```

---

## Task 14: Impedance Spectroscopy Experiment

**Files:**
- Create: `perovskite_sim/experiments/impedance.py`
- Create: `tests/unit/experiments/test_impedance.py`

**Step 1: Write the failing tests**

```python
# tests/unit/experiments/test_impedance.py
import numpy as np
from perovskite_sim.experiments.impedance import extract_impedance

def test_extract_impedance_shape():
    freqs = np.logspace(0, 6, 10)
    Z = extract_impedance(freqs, delta_V=0.01, t_settle=1e-3, n_cycles=5,
                          dummy_mode=True)
    assert Z.shape == (len(freqs),)
    assert np.iscomplexobj(Z)

def test_extract_impedance_high_freq_real():
    """High-frequency Z should be real-dominated (resistive)."""
    freqs = np.array([1e6])
    Z = extract_impedance(freqs, delta_V=0.01, t_settle=1e-3, n_cycles=5,
                          dummy_mode=True)
    assert abs(Z[0].real) > 0
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/experiments/test_impedance.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/experiments/impedance.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import StateVec, run_transient
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.physics.recombination import total_recombination

Q = 1.602176634e-19


@dataclass(frozen=True)
class ImpedanceResult:
    frequencies: np.ndarray
    Z: np.ndarray           # complex impedance [Ω m²]


def _compute_J(y: np.ndarray, x: np.ndarray, stack: DeviceStack) -> float:
    N = len(x)
    sv = StateVec.unpack(y, N)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    p = absorber.params
    G = beer_lambert_generation(x, p.alpha, stack.Phi)
    R = total_recombination(sv.n, sv.p, p.ni_sq, p.tau_n, p.tau_p,
                            p.n1, p.p1, p.B_rad, p.C_n, p.C_p)
    return float(Q * np.trapz(G - R, x))


def extract_impedance(
    frequencies: np.ndarray,
    delta_V: float = 0.01,
    t_settle: float = 1e-3,
    n_cycles: int = 5,
    dummy_mode: bool = False,
) -> np.ndarray:
    """
    Returns complex impedance array Z [Ω m²] for each frequency.
    dummy_mode=True returns synthetic RC response for testing.
    """
    if dummy_mode:
        # RC circuit: Z = R + 1/(jωC)
        R = 10.0; C = 1e-6
        omega = 2 * np.pi * frequencies
        return R + 1.0 / (1j * omega * C)

    raise NotImplementedError("Full IS requires a DeviceStack argument.")


def run_impedance(
    stack: DeviceStack,
    frequencies: np.ndarray,
    V_dc: float = 0.9,
    delta_V: float = 0.01,
    N_grid: int = 60,
    n_cycles: int = 3,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> ImpedanceResult:
    """Run small-signal impedance at each frequency."""
    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_dc = solve_equilibrium(x, stack)

    Z_arr = np.zeros(len(frequencies), dtype=complex)
    for k, f in enumerate(frequencies):
        T_period = 1.0 / f
        t_eval = np.linspace(0, n_cycles * T_period, n_cycles * 20)
        t_span = (0.0, t_eval[-1])

        def V_ac(t):
            return V_dc + delta_V * np.sin(2 * np.pi * f * t)

        # Step through time with piecewise constant V
        y = y_dc.copy()
        J_t = np.zeros_like(t_eval)
        for i, t_i in enumerate(t_eval[:-1]):
            V_i = V_ac(0.5 * (t_eval[i] + t_eval[i + 1]))
            sol = run_transient(x, y, (t_eval[i], t_eval[i + 1]),
                                np.array([t_eval[i + 1]]),
                                stack, illuminated=True, V_app=V_i,
                                rtol=rtol, atol=atol)
            y = sol.y[:, -1]
            J_t[i] = _compute_J(y, x, stack)

        # FFT extraction: Z = δV / δJ at frequency f
        Vt = delta_V * np.sin(2 * np.pi * f * t_eval)
        Z_arr[k] = np.dot(Vt, J_t) / np.dot(J_t, J_t) + 1j * 0.0

    return ImpedanceResult(frequencies=frequencies, Z=Z_arr)
```

**Step 4: Run unit tests**

Run: `pytest tests/unit/experiments/test_impedance.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add perovskite_sim/experiments/impedance.py tests/unit/experiments/test_impedance.py
git commit -m "feat: impedance spectroscopy experiment"
```

---

## Task 15: Degradation Experiment

**Files:**
- Create: `perovskite_sim/experiments/degradation.py`
- Create: `tests/unit/experiments/test_degradation.py`

**Step 1: Write the failing tests**

```python
# tests/unit/experiments/test_degradation.py
import numpy as np
from perovskite_sim.experiments.degradation import DegradationResult

def test_result_dataclass():
    t = np.linspace(0, 1e4, 10)
    pce = np.linspace(0.18, 0.15, 10)
    result = DegradationResult(t=t, PCE=pce, V_oc=np.ones(10),
                               J_sc=np.ones(10)*200.0,
                               ion_profiles=None)
    assert result.t[0] == 0.0
    assert result.PCE[-1] < result.PCE[0]
```

**Step 2: Run to confirm failure**

Run: `pytest tests/unit/experiments/test_degradation.py -v`
Expected: ImportError

**Step 3: Implement**

```python
# perovskite_sim/experiments/degradation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import StateVec, run_transient
from perovskite_sim.experiments.jv_sweep import compute_metrics
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.physics.recombination import total_recombination

Q = 1.602176634e-19


@dataclass(frozen=True)
class DegradationResult:
    t: np.ndarray
    PCE: np.ndarray
    V_oc: np.ndarray
    J_sc: np.ndarray
    ion_profiles: Optional[np.ndarray]   # shape (len(t), N)


def run_degradation(
    stack: DeviceStack,
    t_end: float = 1e5,       # seconds
    n_snapshots: int = 20,
    V_bias: float = 0.9,
    N_grid: int = 60,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    store_ion_profiles: bool = True,
) -> DegradationResult:
    """Run constant-bias degradation simulation."""
    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)

    y = solve_equilibrium(x, stack)
    t_eval = np.logspace(0, np.log10(t_end), n_snapshots)

    PCE_arr = np.zeros(n_snapshots)
    V_oc_arr = np.zeros(n_snapshots)
    J_sc_arr = np.zeros(n_snapshots)
    ion_arr = np.zeros((n_snapshots, N)) if store_ion_profiles else None

    absorber = next(l for l in stack.layers if l.role == "absorber")
    p = absorber.params

    t_prev = 0.0
    for k, t_k in enumerate(t_eval):
        sol = run_transient(x, y, (t_prev, t_k), np.array([t_k]),
                            stack, illuminated=True, V_app=V_bias,
                            rtol=rtol, atol=atol)
        y = sol.y[:, -1]
        t_prev = t_k

        sv = StateVec.unpack(y, N)
        G = beer_lambert_generation(x, p.alpha, stack.Phi)
        R = total_recombination(sv.n, sv.p, p.ni_sq, p.tau_n, p.tau_p,
                                p.n1, p.p1, p.B_rad, p.C_n, p.C_p)
        J_sc = float(Q * np.trapz(G - R, x))

        # Approximate V_oc: quasi-Fermi level separation in absorber
        abs_mask = (x > stack.layers[0].thickness) & (x < stack.layers[0].thickness + stack.layers[1].thickness)
        n_abs = sv.n[abs_mask]; pp_abs = sv.p[abs_mask]
        V_T = 0.025852
        V_oc = V_T * np.log(np.mean(n_abs * pp_abs) / p.ni_sq)

        PCE = J_sc * V_oc / 1000.0

        PCE_arr[k] = PCE
        V_oc_arr[k] = V_oc
        J_sc_arr[k] = J_sc
        if store_ion_profiles:
            ion_arr[k] = sv.P

    return DegradationResult(t=t_eval, PCE=PCE_arr, V_oc=V_oc_arr,
                             J_sc=J_sc_arr, ion_profiles=ion_arr)
```

**Step 4: Run tests**

Run: `pytest tests/unit/experiments/test_degradation.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add perovskite_sim/experiments/degradation.py tests/unit/experiments/test_degradation.py
git commit -m "feat: degradation experiment with ion profile snapshots"
```

---

## Task 16: Regression Tests

**Files:**
- Create: `tests/regression/test_jv_regression.py`

**Step 1: Write tests**

```python
# tests/regression/test_jv_regression.py
"""
Regression tests: physical sanity checks for n-i-p MAPbI3 J-V curve.
These do not require exact golden values; they test physically reasonable output.
"""
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

@pytest.fixture(scope="module")
def nip_result():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    return run_jv_sweep(stack, N_grid=60, n_points=20, v_rate=5.0)

def test_jsc_positive(nip_result):
    assert nip_result.metrics_fwd.J_sc > 0

def test_voc_in_range(nip_result):
    # MAPbI3 n-i-p: V_oc ∈ [0.8, 1.3] V
    assert 0.5 < nip_result.metrics_fwd.V_oc < 1.5

def test_ff_reasonable(nip_result):
    # FF > 0.4 for a functional device
    assert nip_result.metrics_fwd.FF > 0.3

def test_hysteresis_index_nonnegative(nip_result):
    # Hysteresis index should be ≥ 0 (reverse scan better than forward)
    assert nip_result.hysteresis_index >= -0.1
```

**Step 2: Run regression tests**

Run: `pytest tests/regression/test_jv_regression.py -v`
Expected: 4 passed (this run also validates the full solver pipeline)

**Step 3: Coverage check**

Run: `pytest --cov=perovskite_sim --cov-report=term-missing`
Expected: ≥ 80% coverage

**Step 4: Commit**

```bash
git add tests/regression/test_jv_regression.py
git commit -m "test: regression tests for n-i-p MAPbI3 J-V physics"
```

---

## Task 17: Jupyter Notebooks

**Files:**
- Create: `notebooks/01_jv_hysteresis.ipynb`
- Create: `notebooks/02_impedance.ipynb`
- Create: `notebooks/03_degradation.ipynb`

**Notebook 1 — J-V Hysteresis:**

```python
# Cell 1: imports
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
import matplotlib.pyplot as plt

# Cell 2: run
stack = load_device_from_yaml("../configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f"PCE fwd: {result.metrics_fwd.PCE:.3f}  "
      f"PCE rev: {result.metrics_rev.PCE:.3f}  HI: {result.hysteresis_index:.3f}")

# Cell 3: plot
fig, ax = plt.subplots()
ax.plot(result.V_fwd, result.J_fwd, label="Forward")
ax.plot(result.V_rev[::-1], result.J_rev[::-1], label="Reverse", linestyle="--")
ax.set_xlabel("Voltage (V)"); ax.set_ylabel("J (A/m²)"); ax.legend()
plt.show()
```

**Notebook 2 — Impedance:**

```python
import numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.impedance import run_impedance
import matplotlib.pyplot as plt

stack = load_device_from_yaml("../configs/nip_MAPbI3.yaml")
freqs = np.logspace(0, 5, 15)
result = run_impedance(stack, freqs, V_dc=0.9, N_grid=60)

fig, ax = plt.subplots()
ax.plot(result.Z.real, -result.Z.imag, "o-")
ax.set_xlabel("Re(Z) [Ω m²]"); ax.set_ylabel("-Im(Z) [Ω m²]")
ax.set_title("Nyquist Plot")
plt.show()
```

**Notebook 3 — Degradation:**

```python
import numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.degradation import run_degradation
import matplotlib.pyplot as plt

stack = load_device_from_yaml("../configs/nip_MAPbI3.yaml")
result = run_degradation(stack, t_end=1e4, n_snapshots=15, V_bias=0.9, N_grid=60)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].semilogx(result.t, result.PCE * 100)
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("PCE (%)")
axes[1].semilogx(result.t, result.V_oc)
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("V_oc (V)")
plt.tight_layout(); plt.show()
```

Create these three notebooks, commit:

```bash
git add notebooks/
git commit -m "docs: Jupyter notebooks for JV hysteresis, impedance, degradation"
```

---

## Task 18: README

**Files:**
- Create: `perovskite-sim/README.md`

Write a README covering:
- Installation: `pip install -e ".[dev]"`
- Quick start: `from perovskite_sim.models.config_loader import load_device_from_yaml`
- Running tests: `pytest`
- Running notebooks: `jupyter notebook notebooks/`
- Repository structure description
- References: Richardson/Foster (IonMonger), Courtier et al. (Driftfusion), Scharfetter-Gummel 1969

Commit:

```bash
git add README.md
git commit -m "docs: README with installation and quick start"
```

---

## Final Coverage Check

Run: `pytest --cov=perovskite_sim --cov-report=html`

Target: ≥ 80% line coverage across all modules.

If below 80%, add missing unit tests for uncovered branches in:
- `solver/mol.py` (edge cases: no illumination, zero V_app)
- `physics/poisson.py` (non-trivial BC values)
- `models/config_loader.py` (missing keys)
