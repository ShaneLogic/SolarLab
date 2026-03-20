# SolarLab

Solar cell research and simulation project.

## Structure

```
SolarLab/                    # Root directory
├── perovskite-sim/          # Perovskite solar cell simulation code (Python)
├── docs/                   # Research documentation, plans, raw files
└── .claude/                # OMC skills and configuration
```

## Quick Start

### Perovskite Simulator

```bash
cd perovskite-sim
pip install -e ".[dev]"

# Run a simulation
python -c "
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml('configs/nip_MAPbI3.yaml')
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f'PCE: {result.metrics_fwd.PCE:.3f}')
"

# Run tests
pytest
```

### Documentation

See `docs/README.md` for research documentation and plans.
