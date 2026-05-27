"""Phase E3 Sprint 6 Day 4-7 — thermionic-emission flux primitive.

Pure-function primitive for paper eq 14a/b (Pauwels-Vanhoutte 1978
J.Phys.D 11, 649-667):

  j_TE = v_th_eff · (density_bulk_projected − density_state)

Sign convention: positive flux means carriers flow INTO the
interface-plane state (state density grows). Units: m⁻³ × m/s =
m⁻² s⁻¹ (surface flux).

Plus a wrapper ``compute_interface_te_fluxes(mat, iface_state, V_app)``
that produces all 4·N_iface fluxes (n_1s, p_1s, n_2s, p_2s into each
interface) per RHS call:

  n_1s flux  = v_th_eff · (n_R_eq · exp(-V_1_app / V_T) − n_1s_state)
  p_1s flux  = v_th_eff · (p_R_eq · exp(+V_1_app / V_T) − p_1s_state)
  n_2s flux  = v_th_eff · (n_L_eq · exp(+V_2_app / V_T) − n_2s_state)
  p_2s flux  = v_th_eff · (p_L_eq · exp(-V_2_app / V_T) − p_2s_state)

where V_2_app = partition_left * (V_bi - V_app),
      V_1_app = (1 - partition_left) * (V_bi - V_app).

At dark equilibrium (V_app = 0), iface_state = dark-eq init values →
all fluxes ZERO (state already in equilibrium, no net TE current).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    _compute_iface_state_dark_eq,
    build_material_arrays,
)


def _scaps_mirror_mat():
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, mat


# ----------------- Pure primitive --------------------------------------


def test_te_flux_primitive_zero_at_equilibrium():
    """state == bulk_projected → flux = 0."""
    from perovskite_sim.physics.interface_plane import te_flux
    assert te_flux(1.0e22, 1.0e22, 1.0e5) == pytest.approx(0.0)


def test_te_flux_primitive_above_eq_drains():
    """state > bulk → negative flux (drains state)."""
    from perovskite_sim.physics.interface_plane import te_flux
    f = te_flux(1.0e22, 2.0e22, 1.0e5)  # bulk small, state large
    assert f < 0


def test_te_flux_primitive_below_eq_fills():
    """state < bulk → positive flux (fills state)."""
    from perovskite_sim.physics.interface_plane import te_flux
    f = te_flux(2.0e22, 1.0e22, 1.0e5)  # bulk large, state small
    assert f > 0


def test_te_flux_primitive_scales_linearly_with_v_th():
    """Doubling v_th doubles the flux."""
    from perovskite_sim.physics.interface_plane import te_flux
    f1 = te_flux(2.0e22, 1.0e22, 1.0e5)
    f2 = te_flux(2.0e22, 1.0e22, 2.0e5)
    assert f2 == pytest.approx(2.0 * f1)


# ----------------- Wrapper at dark equilibrium -------------------------


def test_wrapper_returns_4n_iface_shape():
    """compute_interface_te_fluxes returns shape (4 * N_iface,)."""
    from perovskite_sim.physics.interface_plane import compute_interface_te_fluxes
    _, mat = _scaps_mirror_mat()
    iface_eq = _compute_iface_state_dark_eq(mat)
    f = compute_interface_te_fluxes(mat, iface_eq, V_app=0.0)
    assert f.shape == (4 * len(mat.interface_nodes),)


def test_wrapper_zero_flux_at_dark_equilibrium():
    """V_app=0 with dark-eq iface_state → all 4*N_iface fluxes ~ 0.

    At thermal equilibrium the state values come from the SAME Boltzmann
    projection the TE flux primitive uses to compute bulk_projected, so
    the difference (bulk_projected − state) is exactly zero.
    """
    from perovskite_sim.physics.interface_plane import compute_interface_te_fluxes
    _, mat = _scaps_mirror_mat()
    iface_eq = _compute_iface_state_dark_eq(mat)
    f = compute_interface_te_fluxes(mat, iface_eq, V_app=0.0)
    # Allow tiny tolerance for float arithmetic.
    max_abs = float(np.max(np.abs(f)))
    # Reference scale: any non-trivial flux would be O(v_th * density).
    # If primitive is wired correctly, this is float-noise level.
    assert max_abs < 1.0e-3, f"max |flux| at dark eq = {max_abs} > 1e-3"


def test_wrapper_nonzero_flux_when_state_off_equilibrium():
    """If state densities are perturbed from dark-eq, fluxes nonzero."""
    from perovskite_sim.physics.interface_plane import compute_interface_te_fluxes
    _, mat = _scaps_mirror_mat()
    iface_eq = _compute_iface_state_dark_eq(mat)
    # Halve every state density (artificially drain).
    iface_perturbed = iface_eq * 0.5
    f = compute_interface_te_fluxes(mat, iface_perturbed, V_app=0.0)
    # Bulk_projected > state for all 4*N_iface entries → all fluxes positive.
    assert np.all(f > 0), "drained state must give positive (filling) flux"


def test_wrapper_handles_zero_v_app():
    """V_app=0 path executes without error + returns finite values."""
    from perovskite_sim.physics.interface_plane import compute_interface_te_fluxes
    _, mat = _scaps_mirror_mat()
    iface_eq = _compute_iface_state_dark_eq(mat)
    f = compute_interface_te_fluxes(mat, iface_eq, V_app=0.0)
    assert np.all(np.isfinite(f))


def test_wrapper_handles_forward_bias():
    """V_app > 0 path executes without error + returns finite values."""
    from perovskite_sim.physics.interface_plane import compute_interface_te_fluxes
    _, mat = _scaps_mirror_mat()
    iface_eq = _compute_iface_state_dark_eq(mat)
    f = compute_interface_te_fluxes(mat, iface_eq, V_app=1.0)
    assert np.all(np.isfinite(f))


def test_wrapper_empty_iface_state_returns_empty():
    """N_iface=0 → empty array."""
    from perovskite_sim.physics.interface_plane import compute_interface_te_fluxes
    from dataclasses import replace
    _, mat = _scaps_mirror_mat()
    # Synthetically zero out the per-interface caches.
    mat_empty = replace(
        mat,
        interface_V_partition_2=(),
        interface_n_L_eq=(),
        interface_p_L_eq=(),
        interface_n_R_eq=(),
        interface_p_R_eq=(),
        interface_nodes=(),
    )
    f = compute_interface_te_fluxes(
        mat_empty, np.zeros(0, dtype=float), V_app=0.0,
    )
    assert f.shape == (0,)
