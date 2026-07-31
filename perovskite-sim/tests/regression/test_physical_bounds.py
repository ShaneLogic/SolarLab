"""Figure-of-merit bounds anchored OUTSIDE the solve.

WHAT THIS COVERS THAT NOTHING ELSE DID
--------------------------------------
``test_jv_regression`` is the headline figure-of-merit gate, and on the two
quantities that matter here it asserts only::

    assert nip_result.metrics_fwd.J_sc > 0      # positivity ONLY
    assert nip_result.metrics_fwd.FF > 0.3      # floor ONLY, no upper bound

Measured against that: J_sc = 485.6 A/m^2 (2.18x the absorbed-photon budget)
passes, FF = 9.29 passes, FF = 1.043 passes.  The gate is blind to the entire
failure class, which is why it stayed green while the class was live.

``test_jv_branch_rejection`` DOES reject on a physical bound -- but on an
INTERNAL one.  Under illumination ``J(V) = J_sc - J_dark(V)`` with
``J_dark >= 0``, so no sample may rise above ``J_sc``; ``_J_BRANCH_EXCESS``
enforces exactly that.  The reference, however, is J_sc itself, and J_sc is
just the V=0 sample.  When THAT is the corrupted one the guard inverts:
measured at N_grid=41 / n_points=15 on ``ionmonger_benchmark``, J_sc came out
19.755, the ceiling became 19.755*1.05 = 20.743, and the sweep then emitted
RuntimeWarnings rejecting the physically correct 221.99 / 222.71 / 222.90
A/m^2 samples as "unconverged" -- every one of them within 0.5 % of the true
photon budget.  One bad datum invalidated the whole reverse sweep, loudly and
backwards.

So the missing bound is an EXTERNAL anchor on J_sc.  The absorbed-photon
budget ``q * sum(G * dx_cell)`` is exactly that: it is the areal generation
the RHS integrates, computed from the optics alone, and no drift-diffusion
solution may collect more carriers than there were photons absorbed to make
them.  ``test_tmm_photon_conservation`` already applies this bound to the
OPTICAL profile (``got <= incident * (1 + 1e-12)``); it was simply never
applied to the SOLVED terminal current.

WHY IT MATTERS -- THE SAFE REGION IS NOT AN AXIS
------------------------------------------------
Measured on ``ionmonger_benchmark`` at n_points=15, BLAS pinned, sweeping
N_grid (J_sc/budget, FF)::

    N=25  0.996 / 0.817  OK      N=45  1.059 / 0.770  J_sc>budget
    N=30  0.785 / 1.043  FF>1    N=50  1.145 / 0.716  J_sc>budget
    N=31  0.785 / 1.043  FF>1    N=55  2.096 / 0.394  J_sc>budget
    N=35  0.966 / 0.843  OK      N=61  1.033 / 0.798  J_sc>budget
    N=40  0.089 / 9.292  FF>1    N=70  0.677 / 1.216  FF>1
    N=41  0.089 / 9.292  FF>1

Nine of eleven wrong, spanning 0.089x to 2.096x the budget.  It is not
monotone in N_grid and not monotone in n_points either -- N_grid=30 is clean
at n_points 8/10/12/20/30/50 and wrong at 15.  Bit-reproducible within one
environment (repeat runs agree to all printed digits) and BLAS-thread
dependent across environments, i.e. the wrong-branch class documented in
``test_jv_branch_rejection``.

Consequence: a user cannot steer into a safe region a priori, so these two
bounds are not merely a regression net -- they are the only available
detector.  FF > 1 is not an independent defect; it is the J_sc error made
visible, since FF = P_mpp/(V_oc*J_sc) inflates by exactly the factor J_sc is
under-read by (0.83 -> 9.29 at an 11.2x under-read).

WHAT IS ASSERTED
----------------
Only bounds that are thermodynamics, not tolerances:
  * ``J_sc <= q * sum(G * dx_cell)`` -- cannot collect more than absorbed
  * ``0 < FF <= 1``                  -- P_mpp <= V_oc * J_sc by definition
Settings are the shipped defaults, which are measured clean; this gate
protects that state rather than encoding the broken combinations.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    _layer_node_counts,
    compute_metrics,
    run_jv_sweep,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import (
    beer_lambert_generation,
    dual_cell_widths,
)
from perovskite_sim.solver.mol import build_material_arrays

# The solve is near-singular at flat band and its branch landing shifts with
# BLAS reduction order -- same sensitivity class as test_jv_branch_rejection,
# which pins threads itself for this reason rather than relying on the slow
# marker. Do the same here.
try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl is a dev dependency
    threadpool_limits = None


@pytest.fixture(scope="module", autouse=True)
def _pin_blas():
    if threadpool_limits is None:
        yield
        return
    with threadpool_limits(limits=1, user_api="blas"):
        yield


# Shipped presets at their DEFAULT sweep settings.
PRESETS = ["nip_MAPbI3", "ionmonger_benchmark"]


def _absorbed_photon_budget(stack, N_grid: int) -> float:
    """q * sum(G * dx_cell) [A/m^2] -- the areal generation the RHS integrates.

    Mirrors ``mol.assemble_rhs``: the cached TMM profile when the config has
    optical data, else Beer-Lambert. Uses the package's own
    ``dual_cell_widths`` so this cannot drift from the solver's quadrature
    convention -- G is a CELL-AVERAGED quantity and must not be integrated
    with trapezoid (see physics/generation.py).
    """
    from perovskite_sim.constants import Q

    elec = electrical_layers(stack)
    x = multilayer_grid(
        [Layer(lyr.thickness, n)
         for lyr, n in zip(elec, _layer_node_counts(stack, N_grid))]
    )
    mat = build_material_arrays(x, stack)
    G = mat.G_optical
    if G is None:
        G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    return float(Q * np.sum(np.asarray(G, dtype=float) * dual_cell_widths(x)))


@pytest.fixture(scope="module")
def swept():
    """One default-settings sweep per preset, shared across assertions."""
    out = {}
    for name in PRESETS:
        stack = load_device_from_yaml(f"configs/{name}.yaml")
        N_grid = 100          # run_jv_sweep default
        result = run_jv_sweep(stack, N_grid=N_grid)
        metrics = compute_metrics(
            np.asarray(result.V_fwd), np.asarray(result.J_fwd)
        )
        out[name] = (metrics, _absorbed_photon_budget(stack, N_grid))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("name", PRESETS)
def test_jsc_cannot_exceed_the_absorbed_photon_budget(swept, name):
    """No solution may collect more carriers than photons absorbed.

    This is the EXTERNAL anchor the branch-rejection guard lacks: the budget
    comes from the optics, so unlike J_sc it cannot be corrupted by the solve.
    """
    metrics, budget = swept[name]
    assert budget > 0.0, f"{name}: photon budget is zero; fixture is not lit"
    assert metrics.J_sc <= budget, (
        f"{name}: J_sc = {metrics.J_sc:.3f} A/m^2 exceeds the absorbed-photon "
        f"budget {budget:.3f} A/m^2 (ratio {metrics.J_sc / budget:.4f}). The "
        "device cannot create more electron-hole pairs than there were photons "
        "absorbed to make them, so this is a solver or optics defect, not a "
        "tolerance to widen."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", PRESETS)
def test_jsc_is_not_grossly_under_read(swept, name):
    """Catch the other tail: a wrong-branch V=0 sample reads J_sc far too LOW.

    Measured failures ran to J_sc = 0.089x budget. Collection is never perfect
    so no tight lower bound is physical, but an order-of-magnitude shortfall on
    a working preset is a defect -- and it is the tail that inflates FF.
    """
    metrics, budget = swept[name]
    assert metrics.J_sc > 0.5 * budget, (
        f"{name}: J_sc = {metrics.J_sc:.3f} A/m^2 is only "
        f"{metrics.J_sc / budget:.4f} of the absorbed-photon budget "
        f"{budget:.3f} A/m^2. On a working preset that indicates the V=0 sample "
        "landed on the wrong branch, which also inflates FF by the same factor."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", PRESETS)
def test_fill_factor_is_thermodynamically_possible(swept, name):
    """FF = P_mpp/(V_oc*J_sc) <= 1 by definition; FF > 1 says P_mpp > V_oc*J_sc.

    Not a tolerance. Measured values of 1.04, 1.22 and 9.29 were all returned
    without error while `assert FF > 0.3` -- the only FF assertion in the
    suite -- passed on every one of them.
    """
    metrics, _ = swept[name]
    assert 0.0 < metrics.FF <= 1.0, (
        f"{name}: FF = {metrics.FF:.5f} is outside (0, 1]. FF > 1 means "
        f"P_mpp > V_oc * J_sc, which is impossible; it is the signature of a "
        "J_sc under-read at the V=0 sample rather than an independent defect."
    )
