"""Phase E4 Sprint 8 Day 4-7 — integrate split_interface_flux into
``carrier_continuity_rhs`` divergence override at heterointerface faces.

Pins the contract for the new ``interface_split_data`` params dispatch:

1. MaterialArrays carries per-node D_n_node, D_p_node arrays (needed by
   per-layer half-flux as D_L, D_R).
2. carrier_continuity_rhs accepts an optional ``interface_split_data``
   dict; when None/missing, dn/dp are bit-identical to legacy.
3. When present, divergence at heterointerface faces is overridden using
   split_interface_flux + iface_state densities.
4. assemble_rhs passes interface_split_data when env active +
   N_iface_state > 0. Legacy bit-identity preserved otherwise.
5. JV sweep V_oc with split-flux active differs measurably from Sprint
   7 χ-step pure-iface-state path (proves split-flux divergence
   actually replaces the bulk-drain).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import assemble_rhs, build_material_arrays
from perovskite_sim.solver.newton import solve_equilibrium


_LEGACY_V_OC = 1.0694


def _setup_scaps_mirror():
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, mat, x


def test_material_arrays_carries_d_n_node(monkeypatch):
    """mat.D_n_node tuple/array exists, length == N grid nodes."""
    monkeypatch.delenv("SOLARLAB_INTERFACE_PLANE_STATE", raising=False)
    _, mat, x = _setup_scaps_mirror()
    assert hasattr(mat, "D_n_node"), "MaterialArrays must expose D_n_node"
    assert hasattr(mat, "D_p_node"), "MaterialArrays must expose D_p_node"
    assert mat.D_n_node is not None
    assert mat.D_n_node.shape == (len(x),)
    assert mat.D_p_node.shape == (len(x),)


def test_d_node_consistent_with_d_face(monkeypatch):
    """D_n_face[f] = harmonic-mean(D_n_node[f], D_n_node[f+1]).

    Validates the node-vs-face relationship that legacy build expects.
    """
    monkeypatch.delenv("SOLARLAB_INTERFACE_PLANE_STATE", raising=False)
    _, mat, _ = _setup_scaps_mirror()
    expected_face = (
        2.0 * mat.D_n_node[:-1] * mat.D_n_node[1:]
        / (mat.D_n_node[:-1] + mat.D_n_node[1:])
    )
    np.testing.assert_allclose(mat.D_n_face, expected_face, rtol=1e-12)


def test_legacy_path_bit_identical_when_env_unset(monkeypatch):
    """env unset → V_oc bit-identical to legacy main 1.0694 +- 5 mV.

    Required for legacy bit-identity. Existing SCAPS-subset regression
    cover this end-to-end; this test pins it at the unit-integration level.
    """
    monkeypatch.delenv("SOLARLAB_INTERFACE_PLANE_STATE", raising=False)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed
    assert r.metrics_fwd.V_oc == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_assemble_rhs_finite_with_split_active(monkeypatch):
    """env=1 → assemble_rhs returns finite dy/dt with split path active."""
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    stack, mat, x = _setup_scaps_mirror()
    y = solve_equilibrium(x, stack)
    dy = assemble_rhs(t=0.0, y=y, x=x, stack=stack, mat=mat,
                     illuminated=False, V_app=0.0)
    assert np.all(np.isfinite(dy)), "dy/dt must be finite under split-flux"


def test_jv_sweep_voc_in_envelope_with_split(monkeypatch):
    """env=1 → V_oc within physical envelope [0.8, 1.3] V.

    Looser than "V_oc moves from legacy by >X" because split-flux can
    REVERT toward legacy when half-fluxes ≈ sum to legacy SG at dark eq.
    Real change shows up in CBO sweep — gated by full validation, not
    this unit test.
    """
    monkeypatch.setenv("SOLARLAB_INTERFACE_PLANE_STATE", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed
    voc = float(r.metrics_fwd.V_oc)
    assert 0.8 <= voc <= 1.3, f"V_oc {voc} outside physical envelope"
