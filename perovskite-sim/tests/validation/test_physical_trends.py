"""Physics trend validation: asserts that the drift-diffusion solver reproduces
well-established device-physics scaling laws.

Invoke with: pytest -m validation
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import linregress

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.experiments.jv_sweep import run_jv_sweep, JVResult

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def baseline_stack() -> DeviceStack:
    """Beer-Lambert n-i-p MAPbI3 preset — FULL tier (default), all physics live.

    Uses the BL preset rather than TMM because Trends 1 & 5 vary absorber Eg
    and need optical generation to respond. TMM n,k data is fixed per
    optical_material key and does not shift with Eg.
    """
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def _vary_absorber_param(
    stack: DeviceStack, param_name: str, values: list[float],
) -> list[DeviceStack]:
    """Return new DeviceStacks with the absorber layer's MaterialParams field
    ``param_name`` set to each value in ``values``.

    Preserves the role-tag scan so ``role: absorber`` is the target layer.
    """
    absorber_idx = next(
        i for i, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    layer = stack.layers[absorber_idx]
    assert layer.params is not None, "absorber layer must have MaterialParams"

    stacks: list[DeviceStack] = []
    for v in values:
        new_params = replace(layer.params, **{param_name: v})
        new_layer = replace(layer, params=new_params)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _vary_absorber_thickness(
    stack: DeviceStack, thicknesses: list[float],
) -> list[DeviceStack]:
    """Return new DeviceStacks with the absorber layer thickness varied."""
    absorber_idx = next(
        i for i, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    layer = stack.layers[absorber_idx]
    stacks: list[DeviceStack] = []
    for t in thicknesses:
        new_layer = replace(layer, thickness=t)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _run_jv(stack: DeviceStack) -> JVResult:
    """Run a J-V sweep with settings matching the regression suite."""
    return run_jv_sweep(
        stack, N_grid=60, n_points=20, v_rate=5.0,
        V_max=1.5,
    )
