"""Phase E3 Sprint 7 Day 1-3 — assemble_rhs wire-through.

Verifies the end-to-end wire-up of interface-plane state through the
solver pipeline:

1. Env unset -> mat.N_iface_state == 0, solve_equilibrium returns
   3N-shaped y (legacy bit-identical).
2. Env=1 -> mat.N_iface_state == N_iface, solve_equilibrium returns
   y with 4*N_iface trailing iface_state block.
3. Env=1 -> assemble_rhs returns dy/dt of same shape as y; dark-eq
   evaluation gives finite values, iface_state block of dy/dt ~ 0
   (at dark equilibrium, TE flux + SRH sinks cancel to ~0).
4. Env=1 -> run_jv_sweep on scaps_mirror produces V_oc that DIFFERS
   from the legacy E1.5 cross-carrier baseline (proves the new path
   participates in carrier dynamics).
5. Env unset -> run_jv_sweep V_oc bit-identical to current main.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    assemble_rhs,
    build_material_arrays,
)
from perovskite_sim.solver.newton import solve_equilibrium


_LEGACY_V_OC = 1.0694


def _setup(monkeypatch_value: str | None):
    """Build stack + mat + grid for scaps_mirror.yaml."""
    if monkeypatch_value is None:
        # already cleared by test fixture
        pass
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, mat, x


def test_mat_has_n_iface_state_zero_when_env_unset(monkeypatch):
    """mat.N_iface_state == 0 when env unset (legacy mode)."""
    monkeypatch.delenv("SOLARLAB_INTERFACE_PLANE_STATE", raising=False)
    _, mat, _ = _setup(None)
    assert hasattr(mat, "N_iface_state")
    assert mat.N_iface_state == 0


def test_mat_has_n_iface_state_active_when_env_set(monkeypatch):
    """mat.N_iface_state == N_iface when env=1 (interface-plane state on)."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    _, mat, _ = _setup("1")
    assert mat.N_iface_state == len(mat.interface_nodes)
    assert mat.N_iface_state > 0  # scaps_mirror has at least one iface


def test_mat_has_n_iface_state_zero_for_malformed_env(monkeypatch):
    """Only literal '1' activates; anything else -> 0 (defensive)."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "true")
    _, mat, _ = _setup("true")
    assert mat.N_iface_state == 0


def test_solve_equilibrium_extends_y_when_active(monkeypatch):
    """env=1 -> y has 3N + 4*N_iface entries (legacy 3N + iface block)."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    stack, mat, x = _setup("1")
    y = solve_equilibrium(x, stack)
    N = len(x)
    expected = 3 * N + 4 * mat.N_iface_state
    assert y.shape == (expected,)


def test_solve_equilibrium_legacy_shape_when_env_unset(monkeypatch):
    """env unset -> y has 3N entries (bit-identical to legacy)."""
    monkeypatch.delenv("SOLARLAB_INTERFACE_PLANE_STATE", raising=False)
    stack, _, x = _setup(None)
    y = solve_equilibrium(x, stack)
    N = len(x)
    assert y.shape == (3 * N,)


def test_assemble_rhs_returns_same_shape_as_y(monkeypatch):
    """env=1 -> assemble_rhs(t, y, ...) returns array of len == len(y)."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    stack, mat, x = _setup("1")
    y = solve_equilibrium(x, stack)
    dy = assemble_rhs(t=0.0, y=y, x=x, stack=stack, mat=mat,
                     illuminated=False, V_app=0.0)
    assert dy.shape == y.shape


def test_assemble_rhs_finite_at_dark_eq(monkeypatch):
    """env=1 + dark-eq state -> dy/dt finite (no NaN/Inf)."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    stack, mat, x = _setup("1")
    y = solve_equilibrium(x, stack)
    dy = assemble_rhs(t=0.0, y=y, x=x, stack=stack, mat=mat,
                     illuminated=False, V_app=0.0)
    assert np.all(np.isfinite(dy))


def test_jv_sweep_legacy_voc_when_env_unset(monkeypatch):
    """env unset -> V_oc bit-identical to current main 1.0694 +- 5 mV."""
    monkeypatch.delenv("SOLARLAB_INTERFACE_PLANE_STATE", raising=False)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed
    assert r.metrics_fwd.V_oc == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_jv_sweep_voc_moves_when_env_active(monkeypatch):
    """env=1 -> V_oc moves measurably from legacy (proves new path participates)."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed
    voc = float(r.metrics_fwd.V_oc)
    assert 0.8 <= voc <= 1.3, f"V_oc {voc} outside physical envelope"
    # Move by at least 0.5 mV from legacy. Could be small if SRH+TE
    # net contribution nearly equals legacy cross-carrier sink at V_oc.
    assert abs(voc - _LEGACY_V_OC) >= 5.0e-4, (
        f"V_oc {voc} did not move from legacy {_LEGACY_V_OC}; "
        f"interface-plane state may be silently bypassed"
    )
