# perovskite-sim

The Python simulation package, FastAPI backend, and Vite/TypeScript
frontend that make up the SolarLab simulator.

> 📖 **Start here:** the [root README](../README.md) covers installation,
> physics, equations, UI walkthrough, and shipped presets. This file is
> just a short orientation to the `perovskite-sim/` subtree.

## Layout

```
perovskite-sim/
├── perovskite_sim/   Python simulation library (drift-diffusion + ions + TMM)
├── backend/          FastAPI HTTP wrapper — see backend/README.md
├── frontend/         Vite + TypeScript + Plotly single-page UI
├── configs/          Shipped YAML device presets
├── tests/            pytest suite (unit / integration / regression)
└── notebooks/        Exploratory benchmarks
```

## Quick install

```bash
pip install -e ".[dev]"        # Python package in editable mode
cd frontend && npm install     # frontend dependencies
```

## Tests

```bash
pytest                                                  # default unit + integration (~15 s)
pytest -m slow                                          # physics regression (~30 s, BLAS pinned)
pytest --cov=perovskite_sim --cov-report=term-missing   # with coverage
```

## Notebooks

Interactive notebooks under `notebooks/`:

- `01_jv_hysteresis.ipynb` — J–V sweep with hysteresis
- `02_impedance.ipynb` — impedance spectroscopy (Nyquist plot)
- `03_degradation.ipynb` — long-term degradation simulation

Benchmark scripts (`.py`, run with `python`):

- `04_ionmonger_benchmark.py`
- `05_comprehensive_benchmark.py`
- `06_e2e_notebook_vs_api.py`

## Python-only quick start

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f"PCE: {result.metrics_fwd.PCE*100:.2f} %")
```
