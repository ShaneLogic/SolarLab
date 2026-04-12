"""Integration tests for temperature scaling and trap profiles.

Covers Phase 4 of the physics accuracy upgrade:
- Temperature scaling flows through DeviceStack.T → MaterialArrays.V_T_device
- V_oc temperature coefficient is negative (~-2 mV/K is typical for MAPbI3)
- Position-dependent trap profiles increase recombination near interfaces and
  therefore reduce V_oc relative to a uniform-tau baseline.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.physics.temperature import thermal_voltage


pytestmark = pytest.mark.slow


# --- build_material_arrays temperature plumbing ---------------------------


def test_material_arrays_temperature_plumbing():
    """DeviceStack.T flows into MaterialArrays.T_device / V_T_device."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack_hot = dataclasses.replace(stack, T=350.0)

    x = np.linspace(0.0, stack_hot.total_thickness, 60)
    mat = build_material_arrays(x, stack_hot)

    assert mat.T_device == pytest.approx(350.0)
    assert mat.V_T_device == pytest.approx(thermal_voltage(350.0), rel=1e-12)
    # ni_sq should grow with T (Boltzmann factor dominates)
    mat300 = build_material_arrays(x, stack)
    abs_idx = len(x) // 2
    assert mat.ni_sq[abs_idx] > mat300.ni_sq[abs_idx]
    # mu_n (via D_n = mu * V_T) should drop at 350 K with gamma=-1.5
    assert mat.D_n_face.max() < mat300.D_n_face.max()


def test_material_arrays_trap_profile():
    """trap_N_t_interface > bulk shortens tau near both layer boundaries."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")

    # Inject a trap profile into the MAPbI3 absorber layer (index 1).
    absorber = stack.layers[1]
    p_new = dataclasses.replace(
        absorber.params,
        trap_N_t_interface=1.0e22,
        trap_N_t_bulk=1.0e20,
        trap_decay_length=20e-9,
    )
    layer_new = dataclasses.replace(absorber, params=p_new)
    layers = list(stack.layers)
    layers[1] = layer_new
    stack_trap = dataclasses.replace(stack, layers=tuple(layers))

    x = np.linspace(0.0, stack_trap.total_thickness, 200)
    mat_trap = build_material_arrays(x, stack_trap)
    mat_flat = build_material_arrays(x, stack)

    # Absorber range: 200 nm ≤ x ≤ 600 nm
    lo, hi = 200e-9, 600e-9
    mask = (x >= lo + 1e-12) & (x <= hi - 1e-12)

    # Near the HTL/absorber interface tau must be smaller than deep in bulk.
    left_edge = np.argmin(np.abs(x - (lo + 5e-9)))
    middle = np.argmin(np.abs(x - 0.5 * (lo + hi)))
    right_edge = np.argmin(np.abs(x - (hi - 5e-9)))
    assert mat_trap.tau_n[left_edge] < mat_trap.tau_n[middle]
    assert mat_trap.tau_n[right_edge] < mat_trap.tau_n[middle]

    # Deep in the bulk, the exponential edge contributions decay away, so tau
    # should match the uniform baseline to within ~1 %. (At 200 nm from the
    # nearest interface with L_d=20 nm the edge term is ~e^-10 ~= 5e-5 weighted
    # by (N_t_interface/N_t_bulk) = 100, giving ~1 % residual.)
    assert mat_trap.tau_n[middle] == pytest.approx(mat_flat.tau_n[middle], rel=2e-2)


# --- End-to-end effect on J–V ---------------------------------------------


@pytest.fixture(scope="module")
def jv_300K():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    return run_jv_sweep(stack, N_grid=40, n_points=15, v_rate=5.0)


@pytest.fixture(scope="module")
def jv_330K():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack_hot = dataclasses.replace(stack, T=330.0)
    return run_jv_sweep(stack_hot, N_grid=40, n_points=15, v_rate=5.0)


def test_voc_temperature_coefficient(jv_300K, jv_330K):
    """V_oc should decrease with temperature.

    For MAPbI3 the measured dV_oc/dT is roughly -1 to -3 mV/K. We only assert
    the sign and a generous magnitude bound to stay robust across solver
    settings.
    """
    V_oc_300 = jv_300K.metrics_rev.V_oc
    V_oc_330 = jv_330K.metrics_rev.V_oc
    dVoc_dT = (V_oc_330 - V_oc_300) / 30.0  # V/K
    assert dVoc_dT < 0.0, (
        f"Expected dV_oc/dT < 0 but got {dVoc_dT*1000:.3f} mV/K "
        f"(V_oc(300)={V_oc_300:.4f}, V_oc(330)={V_oc_330:.4f})"
    )
    # Sanity bound: should not exceed -10 mV/K in magnitude.
    assert dVoc_dT > -10.0e-3, f"|dV_oc/dT|={dVoc_dT*1000:.3f} mV/K unphysically large"


def test_trap_profile_reduces_voc():
    """Concentrating traps near interfaces reduces V_oc vs uniform baseline."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")

    absorber = stack.layers[1]
    p_trap = dataclasses.replace(
        absorber.params,
        tau_n=1e-7,
        tau_p=1e-7,
        trap_N_t_interface=1.0e22,
        trap_N_t_bulk=1.0e20,
        trap_decay_length=20e-9,
    )
    p_flat = dataclasses.replace(absorber.params, tau_n=1e-7, tau_p=1e-7)

    layers_trap = list(stack.layers)
    layers_trap[1] = dataclasses.replace(absorber, params=p_trap)
    stack_trap = dataclasses.replace(stack, layers=tuple(layers_trap))

    layers_flat = list(stack.layers)
    layers_flat[1] = dataclasses.replace(absorber, params=p_flat)
    stack_flat = dataclasses.replace(stack, layers=tuple(layers_flat))

    res_trap = run_jv_sweep(stack_trap, N_grid=40, n_points=12, v_rate=5.0)
    res_flat = run_jv_sweep(stack_flat, N_grid=40, n_points=12, v_rate=5.0)

    assert res_trap.metrics_rev.V_oc < res_flat.metrics_rev.V_oc, (
        f"Trap profile V_oc={res_trap.metrics_rev.V_oc:.4f} should be below "
        f"uniform-tau V_oc={res_flat.metrics_rev.V_oc:.4f}"
    )
