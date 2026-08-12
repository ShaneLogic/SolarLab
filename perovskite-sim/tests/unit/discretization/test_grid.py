import dataclasses

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, K_B, Q
from perovskite_sim.discretization.grid import (
    Layer,
    interface_grid_diagnostics,
    multilayer_grid,
    tanh_grid,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


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


def test_multilayer_grid_accepts_one_alpha_per_layer():
    layers = [
        Layer(thickness=300e-9, N=40),
        Layer(thickness=180e-6, N=160),
    ]

    actual = multilayer_grid(layers, alpha=(2.0, 3.0))
    expected = np.concatenate([
        tanh_grid(40, 300e-9, alpha=2.0),
        tanh_grid(160, 180e-6, alpha=3.0)[1:] + 300e-9,
    ])

    np.testing.assert_array_equal(actual, expected)


def test_multilayer_grid_scalar_alpha_remains_identical():
    layers = [
        Layer(thickness=100e-9, N=30),
        Layer(thickness=400e-9, N=50),
    ]

    scalar = multilayer_grid(layers, alpha=3.0)
    repeated = multilayer_grid(layers, alpha=(3.0, 3.0))
    historical_formula = np.concatenate([
        tanh_grid(30, 100e-9, alpha=3.0),
        tanh_grid(50, 400e-9, alpha=3.0)[1:] + 100e-9,
    ])

    np.testing.assert_array_equal(scalar, historical_formula)
    np.testing.assert_array_equal(scalar, repeated)


def test_multilayer_grid_rejects_alpha_count_mismatch():
    layers = [
        Layer(thickness=100e-9, N=30),
        Layer(thickness=400e-9, N=50),
    ]

    with pytest.raises(ValueError, match="one value per layer"):
        multilayer_grid(layers, alpha=(3.0,))


def _diagnostic(stack, n_grid, *, layer_name, side):
    x = build_electrical_grid(stack, n_grid)
    item = next(
        diagnostic for diagnostic in interface_grid_diagnostics(x, stack)
        if diagnostic.layer_name == layer_name and diagnostic.side == side
    )
    return x, item


def _carrier_debye_length(*, eps_r, temperature, ni, net_doping):
    carrier_sum = np.hypot(net_doping, 2.0 * ni)
    return np.sqrt(eps_r * EPS_0 * K_B * temperature / (Q * Q * carrier_sum))


def test_interface_diagnostic_uses_solver_temperature_mode():
    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    layers = list(stack.layers)
    base = layers[1]
    params = dataclasses.replace(base.params, N_A=0.0, N_D=0.0)
    layers[1] = dataclasses.replace(base, params=params)

    full = dataclasses.replace(stack, layers=tuple(layers), T=350.0, mode="full")
    x_full, diagnostic_full = _diagnostic(
        full, 100, layer_name="p_base", side="right",
    )
    interface = int(np.argmin(np.abs(x_full - layers[0].thickness)))
    mat_full = build_material_arrays(x_full, full)
    expected_full = _carrier_debye_length(
        eps_r=params.eps_r,
        temperature=350.0,
        ni=np.sqrt(mat_full.ni_sq[interface]),
        net_doping=0.0,
    )
    assert diagnostic_full.debye_length == pytest.approx(expected_full, rel=1e-12)

    legacy = dataclasses.replace(full, mode="legacy")
    x_legacy, diagnostic_legacy = _diagnostic(
        legacy, 100, layer_name="p_base", side="right",
    )
    mat_legacy = build_material_arrays(x_legacy, legacy)
    expected_legacy = _carrier_debye_length(
        eps_r=params.eps_r,
        temperature=300.0,
        ni=np.sqrt(mat_legacy.ni_sq[interface]),
        net_doping=0.0,
    )
    assert diagnostic_legacy.debye_length == pytest.approx(expected_legacy, rel=1e-12)
    assert diagnostic_full.debye_length != pytest.approx(
        diagnostic_legacy.debye_length,
        rel=1e-3,
    )


def test_interface_diagnostic_uses_solver_graded_endpoint_density():
    stack = load_device_from_yaml("configs/cigs_graded_notch.yaml")
    layers = list(stack.layers)
    absorber_index = next(
        index for index, layer in enumerate(layers) if layer.role == "absorber"
    )
    absorber = layers[absorber_index]
    params = dataclasses.replace(
        absorber.params,
        N_A=0.0,
        N_D=0.0,
        grading_direction="back_to_front",
    )
    layers[absorber_index] = dataclasses.replace(absorber, params=params)
    stack = dataclasses.replace(stack, layers=tuple(layers))

    x, diagnostic = _diagnostic(
        stack, 60, layer_name=absorber.name, side="right",
    )
    interface_position = sum(layer.thickness for layer in layers[:absorber_index])
    interface = int(np.argmin(np.abs(x - interface_position)))
    mat = build_material_arrays(x, stack)
    expected = _carrier_debye_length(
        eps_r=params.eps_r,
        temperature=stack.T,
        ni=np.sqrt(mat.ni_sq[interface]),
        net_doping=0.0,
    )
    assert diagnostic.debye_length == pytest.approx(expected, rel=1e-12)
