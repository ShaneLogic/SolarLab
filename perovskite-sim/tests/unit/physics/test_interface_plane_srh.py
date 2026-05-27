"""Phase E3 Sprint 6 Day 8-10 — interface SRH on state-vec densities.

Two-sided Shockley-Read SRH evaluated on the interface-plane state-vec
block (NOT projected from bulk). Per heterointerface k, the SRH
consumes pairs:
  R_s1 = interface_recombination(n_1s, p_2s, ni_eff_sq, n_1, p_1, v_n, v_p)
  R_s2 = interface_recombination(n_2s, p_1s, ni_eff_sq, n_1, p_1, v_n, v_p)

Sinks apply to all four state densities in that pair:
  d(n_1s)/dt -= R_s1
  d(p_2s)/dt -= R_s1
  d(n_2s)/dt -= R_s2
  d(p_1s)/dt -= R_s2

Helper `_apply_interface_srh_on_state(iface_state, stack, mat)` returns
a 4*N_iface array of SRH-sink contributions (negative; subtract from
diface_state in the wiring step).

Contract pinned:
1. At dark equilibrium iface_state -> all sinks ~ 0 (np_eq ~ ni_eff_sq).
2. Perturbed state -> nonzero sinks.
3. Returns shape (4 * N_iface,).
4. Finite values under physical envelopes.
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


def _scaps_mirror():
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, mat


def test_srh_on_state_returns_4n_iface_shape():
    """Helper returns shape (4 * N_iface,)."""
    from perovskite_sim.physics.interface_plane import (
        compute_interface_srh_on_state,
    )
    stack, mat = _scaps_mirror()
    iface_eq = _compute_iface_state_dark_eq(mat)
    sinks = compute_interface_srh_on_state(iface_eq, stack, mat)
    assert sinks.shape == (4 * len(mat.interface_nodes),)


def test_srh_on_state_zero_at_dark_equilibrium():
    """Dark-eq iface_state -> all sinks ~ 0 (n*p = ni_eff_sq).

    At thermal equilibrium n_1s * p_2s = (n_R_eq * exp(-V_1)) * (p_L_eq
    * exp(-V_2)) = n_R_eq * p_L_eq * exp(-V_bi/V_T). Detailed balance
    sets ni_eff_sq = n_R_eq * p_L_eq * exp(-V_bi/V_T) so the SRH
    numerator vanishes.

    NOTE: this requires the SRH helper to use the same ni_eff_sq the
    rest of the solver uses. Test pins that consistency.
    """
    from perovskite_sim.physics.interface_plane import (
        compute_interface_srh_on_state,
    )
    stack, mat = _scaps_mirror()
    iface_eq = _compute_iface_state_dark_eq(mat)
    sinks = compute_interface_srh_on_state(iface_eq, stack, mat)
    # Equilibrium np product should match ni_eff_sq tightly; small numerical
    # residuals are OK. Use the largest legacy interface SRV as a scale.
    max_abs = float(np.max(np.abs(sinks)))
    # If sinks were on the order of v_n * n_eq (~1e22 * 1e-2 = 1e20), test
    # would fail; finding 1e15 or less indicates detailed-balance cancellation.
    assert max_abs < 1.0e18, (
        f"max |SRH sink| at dark eq = {max_abs:.3e} -- detailed balance "
        f"should make np_eq match ni_eff_sq closely"
    )


def test_srh_on_state_nonzero_when_perturbed():
    """Inflate all state densities 10x -> np >> ni_eff_sq -> negative sinks."""
    from perovskite_sim.physics.interface_plane import (
        compute_interface_srh_on_state,
    )
    stack, mat = _scaps_mirror()
    iface_eq = _compute_iface_state_dark_eq(mat)
    iface_perturbed = iface_eq * 10.0
    sinks = compute_interface_srh_on_state(iface_perturbed, stack, mat)
    # At least one sink should be measurably negative (carrier loss).
    assert float(np.min(sinks)) < 0.0


def test_srh_on_state_finite_under_dark_eq():
    """All sinks finite at dark equilibrium."""
    from perovskite_sim.physics.interface_plane import (
        compute_interface_srh_on_state,
    )
    stack, mat = _scaps_mirror()
    iface_eq = _compute_iface_state_dark_eq(mat)
    sinks = compute_interface_srh_on_state(iface_eq, stack, mat)
    assert np.all(np.isfinite(sinks))


def test_srh_on_state_block_layout():
    """For interface k, sinks at indices 4k, 4k+1, 4k+2, 4k+3 correspond to
    (n_1s, p_1s, n_2s, p_2s) in that order. Pairs: (n_1s, p_2s) get R_s1
    so sinks[4k] == sinks[4k+3] in magnitude; (n_2s, p_1s) get R_s2 so
    sinks[4k+2] == sinks[4k+1] in magnitude.
    """
    from perovskite_sim.physics.interface_plane import (
        compute_interface_srh_on_state,
    )
    stack, mat = _scaps_mirror()
    iface_eq = _compute_iface_state_dark_eq(mat)
    iface_perturbed = iface_eq * 100.0  # large perturbation, well above ni_eff
    sinks = compute_interface_srh_on_state(iface_perturbed, stack, mat)
    for k in range(len(mat.interface_nodes)):
        n1s, p1s, n2s, p2s = sinks[4*k:4*k+4]
        # n_1s sink and p_2s sink share R_s1 magnitude.
        assert n1s == pytest.approx(p2s, rel=1e-9, abs=1e-3)
        # n_2s sink and p_1s sink share R_s2 magnitude.
        assert n2s == pytest.approx(p1s, rel=1e-9, abs=1e-3)


def test_srh_on_state_empty_iface_returns_empty():
    """N_iface=0 -> empty array."""
    from perovskite_sim.physics.interface_plane import (
        compute_interface_srh_on_state,
    )
    from dataclasses import replace
    stack, mat = _scaps_mirror()
    mat_empty = replace(
        mat,
        interface_V_partition_2=(),
        interface_n_L_eq=(),
        interface_p_L_eq=(),
        interface_n_R_eq=(),
        interface_p_R_eq=(),
        interface_nodes=(),
        interface_n1=(),
        interface_p1=(),
        interface_ni_sq_eff=(),
        interface_eval_node_n=(),
        interface_eval_node_p=(),
        interface_calibration_factor=(),
    )
    sinks = compute_interface_srh_on_state(
        np.zeros(0, dtype=float), stack, mat_empty,
    )
    assert sinks.shape == (0,)
