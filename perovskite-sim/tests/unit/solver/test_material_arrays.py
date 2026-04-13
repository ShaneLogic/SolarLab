"""Tests for build_material_arrays substrate handling and electrical_layers."""
import numpy as np

from perovskite_sim.models.device import (
    DeviceStack,
    LayerSpec,
    electrical_layers,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer


def _glass_params() -> MaterialParams:
    return MaterialParams(
        eps_r=2.25, mu_n=0.0, mu_p=0.0, D_ion=0.0,
        P_lim=1e30, P0=0.0,
        ni=1.0, tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
        B_rad=0.0, C_n=0.0, C_p=0.0, alpha=0.0, N_A=0.0, N_D=0.0,
        optical_material="glass", incoherent=True,
    )


def test_substrate_role_excluded_from_electrical_grid():
    """role: substrate must not appear in the electrical grid x-range."""
    real = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    absorber_params = real.layers[1].params  # MAPbI3

    stack = DeviceStack(layers=(
        LayerSpec("glass", 1.0e-3, _glass_params(), "substrate"),
        LayerSpec("MAPbI3", 400e-9, absorber_params, "absorber"),
    ))

    elec = electrical_layers(stack)
    assert len(elec) == 1
    assert elec[0].name == "MAPbI3"

    layers_grid = [Layer(l.thickness, 30) for l in elec]
    x = multilayer_grid(layers_grid)
    assert x[0] >= 0.0
    assert x[-1] <= 400e-9 + 1e-15


def test_electrical_layers_preserves_order_and_filters_only_substrate():
    """Non-substrate roles pass through in original order."""
    params = _glass_params()
    stack = DeviceStack(layers=(
        LayerSpec("glass", 1e-3, params, "substrate"),
        LayerSpec("ETL", 50e-9, params, "ETL"),
        LayerSpec("abs", 400e-9, params, "absorber"),
        LayerSpec("HTL", 200e-9, params, "HTL"),
    ))
    elec = electrical_layers(stack)
    assert [l.name for l in elec] == ["ETL", "abs", "HTL"]
