"""Integration tests for tiered simulation modes.

Verifies that mode='legacy' truly reverts the physics upgrades (TE, TMM,
dual ions, trap profile, temperature scaling), while mode='full' leaves
them enabled. The "full" path is already covered by the rest of the
integration suite; these tests concentrate on the difference.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.mode import LEGACY, FULL
from perovskite_sim.solver.mol import build_material_arrays


def _stack_with_all_upgrades():
    """IonMonger benchmark + trap profile + dual ions + T=330 K on absorber."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    absorber = stack.layers[1]
    p_new = dataclasses.replace(
        absorber.params,
        D_ion_neg=1.0e-18,
        P0_neg=1.0e24,
        P_lim_neg=1.0e27,
        trap_N_t_interface=1.0e22,
        trap_N_t_bulk=1.0e20,
        trap_decay_length=20e-9,
    )
    layers = list(stack.layers)
    layers[1] = dataclasses.replace(absorber, params=p_new)
    return dataclasses.replace(stack, layers=tuple(layers), T=330.0)


def test_legacy_mode_disables_thermionic_emission():
    stack = dataclasses.replace(_stack_with_all_upgrades(), mode="legacy")
    x = np.linspace(0.0, stack.total_thickness, 120)
    mat = build_material_arrays(x, stack)
    assert mat.interface_faces == ()
    assert mat.has_dual_ions is False
    assert mat.P_ion0_neg is None
    assert mat.D_ion_neg_face is None


def test_legacy_mode_ignores_temperature():
    """Even when stack.T=330 K, legacy mode uses V_T(300 K)."""
    stack = dataclasses.replace(_stack_with_all_upgrades(), mode="legacy")
    x = np.linspace(0.0, stack.total_thickness, 120)
    mat = build_material_arrays(x, stack)
    assert mat.T_device == pytest.approx(300.0)
    assert mat.V_T_device == pytest.approx(0.025852, abs=1e-5)


def test_legacy_mode_restores_uniform_tau():
    stack_all = _stack_with_all_upgrades()
    stack_legacy = dataclasses.replace(stack_all, mode="legacy")
    x = np.linspace(0.0, stack_all.total_thickness, 200)

    mat_full = build_material_arrays(x, dataclasses.replace(stack_all, mode="full"))
    mat_legacy = build_material_arrays(x, stack_legacy)

    # Near the absorber interface, full mode must have a shorter tau than
    # legacy because the trap profile is active.
    near_iface = np.argmin(np.abs(x - 205e-9))
    assert mat_full.tau_n[near_iface] < mat_legacy.tau_n[near_iface]
    # Legacy tau should be exactly the nominal absorber tau_n (3e-9).
    assert mat_legacy.tau_n[near_iface] == pytest.approx(3e-9, rel=1e-10)


def test_full_mode_keeps_all_upgrades():
    stack = dataclasses.replace(_stack_with_all_upgrades(), mode="full")
    x = np.linspace(0.0, stack.total_thickness, 120)
    mat = build_material_arrays(x, stack)
    # TE active (band offsets > 0.05 eV at HTL/absorber and absorber/ETL)
    assert len(mat.interface_faces) >= 1
    assert mat.has_dual_ions is True
    assert mat.T_device == pytest.approx(330.0)
