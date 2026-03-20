# SolarLab

Solar cell research and simulation project.

## Structure

- **`perovskite-sim/`** — Perovskite solar cell drift-diffusion simulation (Python)
- **`docs/`** — Documentation and research plans
- **`.claude/`** — OMC skills and configuration

## Quick Start

```bash
cd perovskite-sim
pip install -e .
```

## perovskite-sim

A numerical drift-diffusion simulator for perovskite solar cells.

```bash
# Run simulations
python -m perovskite_sim

# Run tests
pytest tests/
```
