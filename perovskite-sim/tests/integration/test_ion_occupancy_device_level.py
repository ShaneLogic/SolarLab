"""Review section 7 case 4 — finite ion occupancy, at DEVICE level.

``tests/unit/physics/test_ion_steric_dual_shared_site.py`` already covers
the flux-level half of this case: total ion number is conserved, and the
rate is finite and stable at prescribed occupancies up to theta = 0.8.

What it does not cover is the half the review actually asks for -- that
during a real sweep to strong forward bias the occupancy the SOLVER
reaches stays in [0, 1). Those unit tests impose P and check the RHS; this
one lets the device decide P and checks the result. A formulation can pass
the first and still drive P past P_lim, which is where the legacy
whole-flux factor 1/(1 - P/P_lim) becomes singular.

Runs on ionmonger_benchmark, the shipped preset with genuinely mobile ions
(D_ion = 1e-17 m^2/s in the absorber), swept past flat band so the ionic
redistribution is at its largest.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts, run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import build_material_arrays

_CONFIG = "configs/ionmonger_benchmark.yaml"
_N_GRID = 30


@pytest.fixture(scope="module")
def swept():
    """Full sweep past flat band, keeping every spatial snapshot."""
    stack = load_device_from_yaml(_CONFIG)
    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, _N_GRID))
    ])
    mat = build_material_arrays(x, stack)
    r = run_jv_sweep(
        stack, N_grid=_N_GRID, n_points=20, v_rate=5.0, V_max=1.4,
        save_snapshots=True,
    )
    assert r.snapshots_fwd, "sweep returned no snapshots"
    return x, mat, r


@pytest.mark.slow
def test_occupancy_stays_strictly_below_the_site_limit(swept):
    """0 <= P/P_lim < 1 at every node and every voltage.

    The upper bound is the physical one: the legacy steric factor
    1/(1 - P/P_lim) is singular AT the limit, so reaching it is not merely
    unphysical but numerically fatal.
    """
    x, mat, r = swept
    P_lim = np.asarray(mat.P_lim_node, dtype=float)
    assert np.all(P_lim > 0.0), "P_lim_node is not positive everywhere"

    worst_theta, worst_V = -1.0, None
    for snap, V in zip(r.snapshots_fwd, r.V_fwd):
        P = np.asarray(snap.P, dtype=float)
        assert np.all(np.isfinite(P)), f"non-finite ion density at V={V:.4f}"
        assert np.all(P >= 0.0), (
            f"negative ion density at V={V:.4f}: min {P.min():.3e}"
        )
        theta = float(np.max(P / P_lim))
        if theta > worst_theta:
            worst_theta, worst_V = theta, float(V)

    assert worst_theta < 1.0, (
        f"ion occupancy reached the site limit: P/P_lim = {worst_theta:.6f} "
        f"at V = {worst_V:.4f} V"
    )
    # Report the headroom -- the shipped presets are dilute, and knowing by
    # how much is the point of the measurement.
    print(f"\n[ion] peak occupancy P/P_lim = {worst_theta:.4e} "
          f"at V = {worst_V:.4f} V")


def test_total_ion_number_is_conserved_across_the_sweep(swept):
    """Ions redistribute; they are neither created nor destroyed.

    The continuity equation is in divergence form with zero-flux boundaries,
    so the dual-cell-weighted sum is invariant. Checked against the FIRST
    snapshot rather than the nominal P_ion0 so that the assertion is about
    the sweep, not about the seeding.
    """
    x, mat, r = swept
    w = np.asarray(mat.dx_cell, dtype=float)
    totals = [
        float(np.sum(np.asarray(s.P, dtype=float) * w))
        for s in r.snapshots_fwd
    ]
    ref = totals[0]
    assert ref > 0.0, "no ions present -- this preset cannot test occupancy"
    drift = max(abs(t - ref) for t in totals) / ref
    assert drift < 1e-6, (
        f"total ion number drifted by {drift:.3e} of its initial value "
        f"across the sweep (min {min(totals):.6e}, max {max(totals):.6e})"
    )
