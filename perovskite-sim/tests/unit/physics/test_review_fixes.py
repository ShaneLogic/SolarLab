"""Regression tests for the 2026-07 factual-review fixes (F07, F08, F18).

F07 - interface_recombination with a single zero capture velocity must
      return the blocked-cycle limit (0.0), not raise ZeroDivisionError.
F08 - the self-consistent radiative-reabsorption source must vanish at
      mass-action equilibrium (np = ni^2), preserving detailed balance.
F18 - compute_metrics accepts an input-power override so PCE is not
      hard-wired to the 1-sun 1000 W/m^2 denominator.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.experiments.jv_sweep import compute_metrics, _grid_node_count
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import (
    StateVec,
    assemble_rhs,
    build_material_arrays,
)
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers


# ---------------------------------------------------------------------------
# F07 - single-zero surface recombination velocity
# ---------------------------------------------------------------------------

def test_interface_recombination_zero_vn_blocks_cycle():
    r = interface_recombination(1e20, 1e20, 1e32, 1e9, 1e9, 0.0, 1e3)
    assert r == 0.0


def test_interface_recombination_zero_vp_blocks_cycle():
    r = interface_recombination(1e20, 1e20, 1e32, 1e9, 1e9, 1e3, 0.0)
    assert r == 0.0


def test_interface_recombination_both_zero_still_zero():
    assert interface_recombination(1e20, 1e20, 1e32, 1e9, 1e9, 0.0, 0.0) == 0.0


def test_interface_recombination_nonzero_pair_unchanged():
    r = interface_recombination(1e20, 1e18, 1e32, 1e9, 1e9, 1e3, 1e2)
    expected = (1e20 * 1e18 - 1e32) / ((1e20 + 1e9) / 1e2 + (1e18 + 1e9) / 1e3)
    assert np.isclose(r, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# F08 - reabsorption source detailed balance
# ---------------------------------------------------------------------------

def _mass_action_state(x, mat):
    """State with n = p = sqrt(ni^2) per node: np - ni^2 = 0 identically."""
    n = np.sqrt(mat.ni_sq)
    p = np.sqrt(mat.ni_sq)
    P = mat.P_ion0.copy()
    return StateVec.pack(n, p, P)


def test_reabsorption_source_vanishes_at_mass_action_equilibrium():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, 30 // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)

    # Paint a synthetic absorber-reabsorption cache onto the BL mat: the
    # middle third of the grid, P_esc = 0.5, physical thickness of the span.
    N = len(x)
    mask = np.zeros(N, dtype=bool)
    mask[N // 3: 2 * N // 3] = True
    thickness = float(x[2 * N // 3 - 1] - x[N // 3])
    # Nonzero B_rad on the masked nodes so the integrand is not trivially 0.
    B = mat.B_rad.copy()
    B[mask] = 1.0e-16
    mat_rr = dataclasses.replace(
        mat,
        B_rad=B,
        has_radiative_reabsorption=True,
        absorber_masks=(mask,),
        absorber_p_esc=(0.5,),
        absorber_thicknesses=(thickness,),
    )
    mat_off = dataclasses.replace(mat, B_rad=B)

    y = _mass_action_state(x, mat)
    dydt_on = assemble_rhs(0.0, y, x, stack, mat_rr, illuminated=False, V_app=0.0)
    dydt_off = assemble_rhs(0.0, y, x, stack, mat_off, illuminated=False, V_app=0.0)

    # At np = ni^2 the net emission is zero, so the reabsorption source must
    # add nothing: RHS identical with the hook on and off.
    np.testing.assert_allclose(dydt_on, dydt_off, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# F18 - PCE input-power override
# ---------------------------------------------------------------------------

def _toy_jv():
    V = np.linspace(0.0, 1.2, 25)
    J = 200.0 * (1.0 - np.expm1((V - 1.0) / 0.05) / np.expm1(0.2 / 0.05))
    return V, J


def test_compute_metrics_default_pin_is_one_sun():
    V, J = _toy_jv()
    m = compute_metrics(V, J)
    m_explicit = compute_metrics(V, J, P_in=1000.0)
    assert m.PCE == m_explicit.PCE


def test_compute_metrics_pce_scales_with_input_power():
    V, J = _toy_jv()
    m_1sun = compute_metrics(V, J)
    m_half = compute_metrics(V, J, P_in=500.0)
    assert np.isclose(m_half.PCE, 2.0 * m_1sun.PCE, rtol=1e-12)
    # Other metrics are input-power-independent.
    assert m_half.V_oc == m_1sun.V_oc
    assert m_half.J_sc == m_1sun.J_sc
    assert m_half.FF == m_1sun.FF
