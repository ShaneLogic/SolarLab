"""Test support: legacy preset loaders for the field-mobility parity tests.

Only tests import this module; it reads legacy presets through the legacy
loaders, which stay the frozen reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LEGACY_ROOT = Path(__file__).resolve().parents[4] / "perovskite-sim"
CONFIGS = LEGACY_ROOT / "tests" / "fixtures" / "configs"


def load_stack(name: str):
    from perovskite_sim.models.config_loader import load_device_from_yaml

    return load_device_from_yaml(str(CONFIGS / name))


def electrical_grid(stack, n_grid: int = 60) -> np.ndarray:
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.device import electrical_layers

    elec = electrical_layers(stack)
    return multilayer_grid([Layer(layer.thickness, n) for layer, n in zip(elec, _layer_node_counts(stack, n_grid))])


def legacy_material_arrays(x, stack):
    from perovskite_sim.solver.mol import build_material_arrays

    return build_material_arrays(x, stack)
