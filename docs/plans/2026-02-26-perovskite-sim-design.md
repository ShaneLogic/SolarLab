# Perovskite Solar Cell Simulator — Design Document

**Date:** 2026-02-26
**Status:** Approved
**Language:** Python (NumPy / SciPy)
**Scope:** 1D production-quality; 2D deferred

---

## Goals

Simulate perovskite solar cell devices with mobile ion migration, covering:

1. **J-V hysteresis** — forward/reverse scan at configurable sweep rate
2. **Impedance spectroscopy** — Nyquist plot with low-frequency ionic features
3. **Degradation / stability** — long constant-bias ion redistribution

Default test cases: n-i-p (Glass/ITO/TiO₂/MAPbI₃/spiro-OMeTAD/Au) and p-i-n (Glass/ITO/NiO/MAPbI₃/PC₆₁BM/Ag).

---

## Repository Layout

```
perovskite-sim/
├── perovskite_sim/
│   ├── physics/
│   │   ├── poisson.py           # Poisson residual (sparse tridiag)
│   │   ├── continuity.py        # Electron/hole continuity + FE fluxes
│   │   ├── ion_migration.py     # PNP + steric Blakemore flux
│   │   └── recombination.py     # SRH, radiative, Auger (all default-on)
│   ├── discretization/
│   │   ├── grid.py              # tanh_grid(N, x0, x1, alpha)
│   │   ├── fe_operators.py      # Scharfetter-Gummel FE fluxes
│   │   └── boundary.py          # Ohmic / selective contact BCs
│   ├── solver/
│   │   ├── mol.py               # assemble_rhs() → solve_ivp(Radau)
│   │   └── newton.py            # Steady-state Newton loop (dark equilibrium)
│   ├── models/
│   │   ├── device.py            # DeviceStack dataclass
│   │   └── parameters.py        # MaterialParams, SolverConfig dataclasses
│   └── experiments/
│       ├── jv_sweep.py          # Forward/reverse scan, hysteresis index
│       ├── impedance.py         # Small-signal AC, FFT extraction
│       └── degradation.py       # Long-time constant-bias run
├── configs/
│   ├── nip_MAPbI3.yaml
│   └── pin_MAPbI3.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── notebooks/
│   ├── 01_jv_hysteresis.ipynb
│   ├── 02_impedance.ipynb
│   └── 03_degradation.ipynb
├── docs/plans/
├── pyproject.toml
└── README.md
```

---

## Physics Model

### State vector

`[φ, n, p, P]` at each of N grid points.

### Governing PDEs

| Equation  | Form |
|-----------|------|
| Poisson   | `ε ∂²φ/∂x² = -q(p - n + P - N_A + N_D)` |
| Electrons | `∂n/∂t = (1/q) ∂J_n/∂x - R + G` |
| Holes     | `∂p/∂t = -(1/q) ∂J_p/∂x - R + G` |
| Ions      | `∂P/∂t = -∂F_P/∂x` |

### Carrier currents — Scharfetter-Gummel

Exponentially fitted FE scheme; handles sharp gradients near junctions without numerical oscillation.

### Ion flux — steric Blakemore

```
F_P = -D_I · (∂P/∂x) / (1 - P/P_lim)  +  (qP / k_BT) · (∂φ/∂x)
```

- Standard PNP recovered when `P << P_lim`
- Diverging diffusion prevents `P > P_lim`
- Typical: `P_lim = 1e27 m⁻³` for MAPbI₃

### Recombination (all default-on)

- **SRH**: `R_SRH = (np - ni²) / [τ_p(n + n₁) + τ_n(p + p₁)]`
- **Radiative**: `R_rad = B(np - ni²)`
- **Auger**: `R_Aug = (C_n·n + C_p·p)(np - ni²)`

### Optical generation

Beer-Lambert: `G(x) = G₀ α exp(-αx)` where `G₀` integrates to `J_sc` under AM1.5G.

---

## Spatial Discretization

- **Grid**: tanh grid, N=200 default, concentrates points near interfaces
- **Scheme**: 1D Finite Element (Scharfetter-Gummel fluxes)
- **Performance**: ~50× faster than finite difference at equivalent accuracy (per benchmarking literature)

---

## Temporal Integration

- **Method**: `scipy.integrate.solve_ivp(method='Radau', rtol=1e-4, atol=1e-6)`
- **Initialisation**: Newton solver for dark equilibrium before time integration
- **Stability**: rtol=1e-4 and rtol=1e-5 yield overlapping results (confirmed convergence)

---

## Experiments

### J-V Sweep
- Ramp `V_app` at `v_rate` [V/s], forward then reverse
- Record `J(V)`, compute `V_oc`, `J_sc`, `FF`, `PCE`
- Output hysteresis index: `HI = (PCE_rev - PCE_fwd) / PCE_rev`

### Impedance Spectroscopy
- Apply `V_dc + δV·sin(2πft)` for each frequency f
- Extract `Z(f) = δV / δJ` via FFT
- Return Nyquist (`-Im(Z)` vs `Re(Z)`) and Bode plot data

### Degradation
- Constant bias / constant illumination for configurable duration
- Record `V_oc(t)`, `J_sc(t)`, `PCE(t)`, ion profile `P(x,t)`

---

## Configuration

YAML files parsed into immutable dataclasses:

```python
@dataclass(frozen=True)
class MaterialParams:
    eps_r: float
    mu_n: float; mu_p: float
    D_ion: float; P_lim: float
    tau_n: float; tau_p: float
    B_rad: float
    C_n: float; C_p: float
    ...
```

All runs are pure functions: `result = run_jv(device, params, solver_config)`.

---

## Testing Strategy

| Layer | Coverage |
|-------|----------|
| Unit | Each physics function independently |
| Integration | Full equilibrium solve: depletion width, flat quasi-Fermi levels |
| Regression | J-V golden outputs for n-i-p and p-i-n; HI within ±1% |
| Convergence | Error vs N grid refinement test |
| Constraints | `P ≤ P_lim`, `n·p ≥ ni²` at open circuit |

Target: **≥ 80% line coverage** via `pytest` + `pytest-cov`.

---

## 2D Extension Path (deferred)

Replace `discretization/` module with FEniCSx 2D triangular FEM backend. All physics modules remain unchanged. Enables grain-boundary ion migration and lateral non-uniformity studies.

---

## Key Design Principles

- **Immutable data**: all params/results are frozen dataclasses; no in-place mutation
- **Pure functions**: physics residuals are stateless, testable in isolation
- **Modular spatial ops**: swapping grid/FE scheme does not touch physics code
- **YAML-driven**: experiments fully reproducible from config files
