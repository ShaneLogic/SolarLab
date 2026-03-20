# Perovskite Solar Cell Simulator

A 1D drift-diffusion + ion migration simulator for perovskite solar cells, implementing J-V hysteresis, impedance spectroscopy, and degradation experiments.

## Architecture

**Method of Lines (MOL):** Scharfetter-Gummel finite elements on a tanh grid for spatial discretization, `scipy.integrate.solve_ivp(Radau)` for time integration. Poisson equation solved implicitly at each time step via sparse tridiagonal solve. All data structures are immutable frozen dataclasses.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

# Load n-i-p MAPbI3 device
stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")

# Run J-V sweep with hysteresis
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f"PCE: {result.metrics_fwd.PCE:.3f}")
print(f"V_oc: {result.metrics_fwd.V_oc:.3f} V")
print(f"J_sc: {result.metrics_fwd.J_sc:.1f} A/m²")
print(f"FF: {result.metrics_fwd.FF:.3f}")
print(f"Hysteresis Index: {result.hysteresis_index:.3f}")
```

## Running Tests

```bash
pytest
```

Or with coverage report:

```bash
pytest --cov=perovskite_sim --cov-report=term-missing
```

## Running Notebooks

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `01_jv_hysteresis.ipynb` — J-V sweep with hysteresis
- `02_impedance.ipynb` — Impedance spectroscopy (Nyquist plot)
- `03_degradation.ipynb` — Long-term degradation simulation

## Repository Structure

```
perovskite-sim/
├── perovskite_sim/
│   ├── discretization/
│   │   ├── grid.py          # Tanh grid and multilayer grid
│   │   └── fe_operators.py  # Bernoulli function, SG fluxes
│   ├── physics/
│   │   ├── poisson.py       # Sparse Poisson solver
│   │   ├── continuity.py    # Carrier continuity equations
│   │   ├── recombination.py # SRH + radiative + Auger
│   │   ├── ion_migration.py # Steric Blakemore ion flux
│   │   └── generation.py    # Beer-Lambert optical generation
│   ├── models/
│   │   ├── parameters.py    # MaterialParams, SolverConfig
│   │   ├── device.py        # DeviceStack, LayerSpec
│   │   └── config_loader.py # YAML device loader
│   ├── solver/
│   │   ├── mol.py           # MOL assembler, transient solver
│   │   └── newton.py        # Steady-state equilibrium solver
│   └── experiments/
│       ├── jv_sweep.py      # J-V sweep + hysteresis
│       ├── impedance.py     # Impedance spectroscopy
│       └── degradation.py   # Degradation simulation
├── configs/
│   ├── nip_MAPbI3.yaml      # n-i-p perovskite device
│   └── pin_MAPbI3.yaml      # p-i-n perovskite device
├── notebooks/
│   ├── 01_jv_hysteresis.ipynb
│   ├── 02_impedance.ipynb
│   └── 03_degradation.ipynb
└── tests/
    ├── unit/                 # Unit tests per module
    ├── integration/          # Integration tests
    └── regression/           # Physical sanity regression tests
```

## Physics

- **Spatial discretization:** Scharfetter-Gummel finite elements on tanh-clustered grid
- **Time integration:** Radau implicit solver (stiff ODEs)
- **Poisson equation:** Sparse tridiagonal solve on non-uniform grid
- **Recombination:** SRH + radiative (bimolecular) + Auger
- **Ion migration:** Steric Blakemore flux with excluded-volume correction
- **Optical generation:** Beer-Lambert with constant absorption coefficient

## References

- Richardson, G. et al. — IonMonger (ion migration in perovskites)
- Courtier, N. et al. — Driftfusion (1D drift-diffusion)
- Scharfetter, D.L. & Gummel, H.K. (1969) — Large-signal analysis of a silicon Read diode oscillator
