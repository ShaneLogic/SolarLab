# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Perovskite Solar Cell Simulator — a 1D drift-diffusion + ion migration simulator implementing J-V hysteresis, impedance spectroscopy, and degradation experiments.

**Core Method:** Method of Lines (MOL) with Scharfetter-Gummel finite elements on a tanh grid, `scipy.integrate.solve_ivp(Radau)` for time integration, sparse tridiagonal solve for Poisson equation. All data structures are immutable frozen dataclasses.

## Common Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=perovskite_sim --cov-report=term-missing

# Run slow tests
pytest -m slow

# Run a specific test file
pytest tests/unit/test_grid.py

# Run notebooks
jupyter notebook notebooks/
```

## Architecture

```
perovskite_sim/
├── discretization/
│   ├── grid.py          # Tanh grid and multilayer grid
│   └── fe_operators.py  # Bernoulli function, SG fluxes
├── physics/
│   ├── poisson.py       # Sparse Poisson solver
│   ├── continuity.py    # Carrier continuity equations
│   ├── recombination.py # SRH + radiative + Auger
│   ├── ion_migration.py # Steric Blakemore ion flux
│   └── generation.py    # Beer-Lambert optical generation
├── models/
│   ├── parameters.py    # MaterialParams, SolverConfig (frozen dataclasses)
│   ├── device.py        # DeviceStack, LayerSpec
│   └── config_loader.py # YAML device loader
├── solver/
│   ├── mol.py           # MOL assembler, transient solver (Radau)
│   └── newton.py        # Steady-state equilibrium solver
└── experiments/
    ├── jv_sweep.py      # J-V sweep + hysteresis
    ├── impedance.py     # Impedance spectroscopy
    └── degradation.py   # Degradation simulation
```

### Key Physics

- **Spatial discretization:** Scharfetter-Gummel finite elements on tanh-clustered grid
- **Time integration:** Radau implicit solver (stiff ODEs)
- **Poisson equation:** Sparse tridiagonal solve on non-uniform grid
- **Recombination:** SRH + radiative (bimolecular) + Auger
- **Ion migration:** Steric Blakemore flux with excluded-volume correction
- **Optical generation:** Beer-Lambert with constant absorption coefficient

### Data Model

- `MaterialParams`: All material properties (frozen dataclass)
- `SolverConfig`: Grid size, tolerances, temperature (frozen dataclass)
- `DeviceStack`: Complete device with all layers and contacts
- Configuration via YAML files in `configs/` (nip_MAPbI3.yaml, pin_MAPbI3.yaml)

### Test Structure

```
tests/
├── unit/          # Unit tests per module
├── integration/   # Integration tests
└── regression/    # Physical sanity regression tests
```

## Quick Start

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f"PCE: {result.metrics_fwd.PCE:.3f}")
```
