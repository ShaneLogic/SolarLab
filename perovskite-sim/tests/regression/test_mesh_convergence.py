"""Mesh-convergence regression: refine the grid, watch the metrics converge.

Review item §7 #6 — "when the mesh near the interfaces is refined, V_oc / J_sc /
FF and the interface recombination current must converge."  Nothing in the suite
checked this before; the closest thing was ``_compute_current_ss_with_spread``'s
docstring, which explicitly punts the question ("use mesh convergence of the
extracted metrics for the convergence question it cannot answer").

What is asserted
----------------
Let ``m(N_grid)`` be an observable on the grid ``multilayer_grid`` builds with
``N_grid // n_elec`` tanh-clustered intervals per electrical layer.  Along the
refinement ladder ``N_grid = 30 -> 60 -> 120`` (31 / 61 / 121 nodes) every
observable must be **Cauchy and contracting**::

    |m(120) - m(60)|  <  |m(60) - m(30)|

and the Richardson geometric-tail estimate of the residual discretisation error
at the finest level::

    rho = |D32| / |D21|          (contraction factor per N_grid doubling)
    E   = |D32| * rho / (1 - rho)

must sit under the accuracy bar at which this repository states physics results.
**No absolute value of any metric is asserted** — the tests are about the
*sequence*, not the number.

Tolerances (every one traced to a source outside this measurement)
------------------------------------------------------------------
* **5 mV on V_oc** — CLAUDE.md's own statement of the precision at which this
  repo reports V_oc ("SS J-V matches the frozen-ion transient within 5 mV V_oc /
  1% J_sc").  Cross-check: 5 mV ~ (kT/q)/5 with kT/q = 25.852 mV, a fifth of the
  thermal voltage that sets the diode's exponential scale.  *Measured 0.30 mV.*
* **1% on J_sc / terminal current** — same CLAUDE.md sentence.  It is 2.4x
  tighter than the shipped TMM J_sc pin (+/-5 A/m2 on ~215, see
  ``test_tmm_baseline.py``).  *Measured 0.44%.*
* **0.5% on FF** — propagated, not chosen.  FF = P_mpp/(V_oc*J_sc); with V_oc to
  5 mV on ~1.2 V (0.42%) and J_sc to 1%, the natural bar on a ratio built from
  them is ~1%.  0.5% is the tighter half, taken because FF is a shape factor and
  is empirically far less mesh-sensitive than either factor.  *Measured 0.036%.*
* **0.1% of J_sc on the interface-recombination current** — one full order
  inside the 1% J_sc bar, so a residual mesh error in this loss channel provably
  cannot move the terminal current by a reportable amount.  Expressed against
  J_sc rather than against J_iface itself because J_iface is small (0.75-5.05
  A/m2) and a relative bar on it would be arbitrary.  *Measured 0.011-0.014%.*
* **0.5% of J_sc on the spike guard** — derived from the FF bar: an isolated
  upward jump dJ at the sample nearest MPP raises P_mpp by ~dJ*V_mpp, hence
  dFF/FF ~ dJ/J_sc, so setting the spike bound equal to the FF tolerance
  guarantees a surviving spike cannot by itself break the FF test.  This is 10x
  tighter than the shipped guard in ``test_jv_regression.py`` (5% of J_sc).
* **factor 2.0 on interface-cell refinement** — not a tolerance; a property of
  the tanh map, whose edge spacing scales asymptotically as 1/N from above.
  *Measured 2.81 then 2.35 per doubling.*

Why the Richardson residual instead of the raw last difference: for rho ~ 0.26
the last difference *overstates* the remaining error by ~2.8x, and the raw
finest-pair J_sc change (1.23%) would spuriously breach the 1% bar while the
actual residual (0.435%) does not.

Why ``n_points = 81`` (a PROTOCOL parameter derived a priori, not a tolerance)
-----------------------------------------------------------------------------
``compute_metrics`` takes V_oc from a LINEAR interpolation between the two
samples bracketing J = 0 (``jv_sweep.py:114-117``).  For J ~ J_sc - J_0*exp(V/
(m*V_T)) the chord-vs-curve gap gives a V_oc bias ~ dV^2/(8*m*V_T); with
m*V_T ~ 30 mV that is 5.1 mV at n_points = 41 and 1.28 mV at n_points = 81.
The mesh signal being measured is 0.8-2.9 mV, so 41 points is
*extraction-dominated* and 81 is not.

This matters, and it is the honest caveat on this whole file.  At the natural
n_points = 41 (dV = 35 mV) the same ladder gives::

    V_oc = 1.200253, 1.195410, 1.201021 V   (D = -4.84, +5.61 mV; rho = 1.16)
    FF   = 0.795473, 0.798500, 0.794541     (D = +3.03e-3, -3.96e-3; rho = 1.31)

— differences GROW and the sign flips, i.e. a "successive differences shrink"
assertion FAILS outright there.  Diagnosed, not assumed: at *fixed* mesh, moving
41 -> 81 points shifts V_oc by +1.49 mV (N=31) and +3.44 mV (N=61), i.e. by more
than the 60->120 mesh signal itself, and the closed-form interpolation bias
predicts exactly that magnitude.  Two readings are defensible and both are
recorded here: (a) the mesh converges, and the test must sample V finely enough
to see it; (b) the shipped V_oc/FF extraction is not mesh-convergent at the
default sampling density — a real limitation of the reported metrics.

Preset
------
``configs/ionmonger_benchmark.yaml`` — the cheapest shipped preset that has real
heterojunction band offsets (so the thermionic-emission cap is genuinely active
at both interfaces), a live interface-SRH channel carrying ~5 A/m2 at 0.9 V, and
**Beer-Lambert** optics.  Beer-Lambert is the point: it is 3-5x cheaper than the
TMM presets *and* it avoids stacking an independent optical-interpolation error
on top of the transport error, so the transport/interface part is isolated.

Scope limits
------------
Single preset.  On ``configs/nip_MAPbI3.yaml`` (legacy, chi = Eg = 0, no band
offsets, no interface SRV) the 41-point ladder gives V_oc = 0.919292, 0.914809,
0.911575 V — differences do shrink but only with rho = 0.72, which would give a
Richardson residual of 8.3 mV and FAIL the 5 mV bar.  Whether that is real or
another extraction artifact is an open question this file does not answer.

The quoted rates are contraction factors per *doubling of N_grid*, not a formal
spatial order: the tanh map refines the interface cell 2.35-2.81x per doubling
against ~2x at layer centres, so "p ~ 1.9 in N_grid" is the honest statement and
"second-order in h" is not.

Run with: pytest -m slow tests/regression/test_mesh_convergence.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    _compute_current_ss_with_spread,
    _grid_node_count,
    _layer_node_counts,
    _state_fields,
    run_jv_sweep,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    _apply_interface_recombination,
    build_material_arrays,
)

pytestmark = pytest.mark.slow

CONFIG = "configs/ionmonger_benchmark.yaml"

#: Refinement ladder in ``N_grid`` (intervals across the whole electrical stack;
#: ``N_grid // 3`` per layer here, exact for all three rungs).  N_grid = 15 is
#: deliberately NOT the coarsest rung: it is demonstrably pre-asymptotic
#: (J_iface(V=0) goes 0.6976 -> 0.9068 -> 0.7827, non-monotone with a sign flip
#: in the first difference).  Widening the ladder downward will make the
#: interface-current test fail for a legitimate reason — being outside the
#: asymptotic range — rather than for a defect.
# Ladder re-chosen 2026-07-28 (was 30/60/120). N_grid >~ 115 on this config
# is an UNSTABLE illuminated-settle regime: measured at 1 BLAS thread on an
# idle box, seconds to solve at V=0,
#
#     N_grid    100    105    110    115     120
#     pre-fix   1.13   1.18   1.44   >150    1.32
#     current   0.94   1.24   1.16   1.74    >150
#
# i.e. each generation formula hangs at a DIFFERENT rung and is fine at the
# other -- a knife edge, not a property of either. The original 120 top rung
# hung the whole slow lane for four hours. Every rung below is fast for both
# formulas, so 25/50/100 keeps the exact doubling the convergence-order
# assertions need while staying inside the stable region.
LADDER = (25, 50, 100)

#: Second fixed bias: inside the power quadrant (V_mpp ~ 1.015 V) but below
#: V_bi = 1.1 V, so the solve stays out of the documented near-flat-band stiff
#: region.  Not tuned — 0.0 and 0.9 were the first two biases probed.
V_POWER = 0.9

N_POINTS = 81      # see the module docstring: derived a priori, not fitted
V_RATE = 5.0       # V/s — sweep lasts 0.28 s vs an ionic tau of ~0.2-0.8 s
V_MAX = 1.4

# --- tolerances (see module docstring for provenance) -----------------------
TOL_VOC_V = 5.0e-3          # CLAUDE.md steady-state-driver accuracy statement
TOL_J_REL = 0.01            # CLAUDE.md, same sentence
TOL_FF_REL = 0.005          # propagated from V_oc (0.42%) x J_sc (1%)
TOL_JIFACE_REL_JSC = 1.0e-3  # one order inside the J_sc bar
TOL_SPIKE_REL_JSC = 0.005   # equal to the FF bar, by the dFF/FF ~ dJ/J_sc chain
MIN_IFACE_REFINE = 2.0      # tanh edge spacing ~ 1/N, approached from above


# ---------------------------------------------------------------------------
# ladder arithmetic
# ---------------------------------------------------------------------------


def _diffs(vals: tuple[float, float, float]) -> tuple[float, float]:
    """Successive differences (m2 - m1, m3 - m2)."""
    return vals[1] - vals[0], vals[2] - vals[1]


def _rho(vals: tuple[float, float, float]) -> float:
    """Contraction factor |D32| / |D21| per doubling of N_grid.

    ``inf`` when the coarse difference vanishes (nothing to contract from).
    """
    d21, d32 = _diffs(vals)
    if d21 == 0.0:
        return math.inf if d32 != 0.0 else 0.0
    return abs(d32) / abs(d21)


def _richardson_residual(vals: tuple[float, float, float]) -> float:
    """Geometric-tail estimate of the error remaining at the finest level.

    Given an observed contraction factor ``rho`` per refinement step, the
    residual is the sum of the remaining geometric series::

        E = |D32| * (rho + rho^2 + ...) = |D32| * rho / (1 - rho)

    Standard, derived, not chosen.  Returns ``inf`` for a non-contracting
    sequence (rho >= 1), where the tail does not converge — in that case the
    contraction assertion has already failed and this number is diagnostic only.
    """
    _, d32 = _diffs(vals)
    rho = _rho(vals)
    if rho >= 1.0:
        return math.inf
    return abs(d32) * rho / (1.0 - rho)


def _report(name: str, vals: tuple[float, float, float], unit: str = "") -> str:
    """One-line ladder summary: values, differences, rho, exponent, residual."""
    d21, d32 = _diffs(vals)
    rho = _rho(vals)
    exponent = math.log2(1.0 / rho) if 0.0 < rho < math.inf else float("nan")
    res = _richardson_residual(vals)
    return (
        f"{name:<16s} N_grid={LADDER}: "
        f"{vals[0]:.6g}, {vals[1]:.6g}, {vals[2]:.6g} {unit} | "
        f"D21={d21:+.4g} D32={d32:+.4g} | "
        f"rho={rho:.4f} (err ~ N_grid^-{exponent:.2f}) | "
        f"E_richardson={res:.4g} {unit}"
    )


def _assert_contracting(name: str, vals: tuple[float, float, float], unit: str = "") -> None:
    """|m(120) - m(60)| < |m(60) - m(30)| — the Cauchy/contraction invariant."""
    d21, d32 = _diffs(vals)
    assert abs(d32) < abs(d21), (
        f"{name} does NOT contract under mesh refinement — successive "
        f"differences grew or held.\n  {_report(name, vals, unit)}\n"
        "This is a result, not a tolerance to widen: the discretisation is "
        "not converging for this observable on this ladder."
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack():
    return load_device_from_yaml(CONFIG)


def _grid(stack, N_grid: int) -> np.ndarray:
    """Rebuild exactly the electrical grid ``run_jv_sweep`` would build."""
    elec = electrical_layers(stack)
    layers_grid = [
        Layer(l.thickness, n)
        for l, n in zip(elec, _layer_node_counts(stack, N_grid))
    ]
    x = multilayer_grid(layers_grid)
    assert len(x) == _grid_node_count(stack, N_grid)
    return x


@pytest.fixture(scope="module")
def fixed_bias_ladder(stack):
    """Leg A — fixed-bias steady states at V = 0 and V = 0.9 on each rung.

    Cheap (~4 s total).  Carries the mesh diagnostics, the terminal current at
    two biases, and the interface-recombination current reconstructed by calling
    the solver's OWN sink, so whichever interface formulation is active is the
    one measured.
    """
    out: dict[str, list] = {
        "mats": [], "J0": [], "J09": [], "Jif0": [], "Jif09": [],
        "spread0": [], "spread09": [], "dx_iface": [],
    }
    for N_grid in LADDER:
        x = _grid(stack, N_grid)
        mat = build_material_arrays(x, stack)
        out["mats"].append(mat)
        out["dx_iface"].append(
            tuple(float(mat.dx_cell[i]) for i in mat.interface_nodes)
        )
        for V, jkey, ifkey, spkey in (
            (0.0, "J0", "Jif0", "spread0"),
            (V_POWER, "J09", "Jif09", "spread09"),
        ):
            y = solve_illuminated_ss(x, stack, V_app=V)
            J, spread = _compute_current_ss_with_spread(x, y, stack, V, mat=mat)
            n, p, phi, _sv = _state_fields(x, y, stack, V, mat)
            # Call the solver's own interface sink on zeroed derivative
            # buffers; the areal recombination current is then
            # q * sum(-dn * dx_cell) over the whole grid.
            dn = np.zeros_like(n)
            dp = np.zeros_like(p)
            _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)
            out[jkey].append(float(J))
            out[spkey].append(float(spread))
            out[ifkey].append(Q * float(np.sum(-dn * mat.dx_cell)))
    return {k: (tuple(v) if k != "mats" else v) for k, v in out.items()}


@pytest.fixture(scope="module")
def sweep_ladder(stack):
    """Leg B — a full quasi-static forward/reverse J-V on each rung (~3 min).

    ``run_jv_sweep`` rather than the cheaper public ``quasi_static_sweep``: the
    two are mathematically the same loop fed the same y_eq/mat/voltages and
    agree to 8-10 significant figures at N_grid = 30 and 120, but at N_grid = 60
    the ~1e-10 state difference is enough to flip whether the near-flat-band
    Radau spike fires, and ``quasi_static_sweep`` then reports FF = 0.8272
    instead of 0.7961 (+3.9%).  Identical mathematics, 3.9% different FF — so
    the guarded path is used.
    """
    results = []
    for N_grid in LADDER:
        results.append(
            run_jv_sweep(
                stack, N_grid=N_grid, n_points=N_POINTS,
                v_rate=V_RATE, V_max=V_MAX,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# GUARDS — pin what is being measured before believing any of it
# ---------------------------------------------------------------------------


def test_guard_interface_formulation_is_pinned(fixed_bias_ladder):
    """The convergence numbers describe ONE interface formulation — pin it.

    Legacy single-node cross-carrier SRH with the thermionic-emission cap
    active at both heterointerfaces.  If a default flips (plane projection,
    plane closure, shared occupancy, two-sided capture, interface states), this
    test must fail loudly rather than silently measure a different quantity
    and call it "mesh convergence".
    """
    for N_grid, mat in zip(LADDER, fixed_bias_ladder["mats"]):
        assert mat.iface_plane_projection is False, N_grid
        assert mat.iface_plane_closure is False, N_grid
        assert mat.iface_shared_occ is False, N_grid
        assert mat.iface_two_sided is False, N_grid
        assert mat.N_iface_state == 0, N_grid
        # legacy single-node form: both eval nodes sit ON the interface node
        assert tuple(mat.interface_eval_node_n) == tuple(mat.interface_nodes), N_grid
        assert tuple(mat.interface_eval_node_p) == tuple(mat.interface_nodes), N_grid
        # TE cap genuinely active at both heterointerfaces
        assert len(mat.interface_faces) == 2, (
            f"N_grid={N_grid}: expected the TE flux cap to be active at both "
            f"heterointerfaces, got interface_faces={mat.interface_faces}"
        )


def test_guard_interface_cells_actually_refine(fixed_bias_ladder):
    """Review §7 #6 asks about refining the mesh NEAR THE INTERFACES.

    Certify that premise: the dual-cell width AT each interface node must
    shrink by more than 2.0x per N_grid doubling.  The 2.0 is not a tolerance
    — the tanh map's edge spacing scales as 1/N and approaches 2x per doubling
    from above, so anything at or below 2.0 would mean the interface region is
    NOT being refined faster than the bulk.
    """
    dx = fixed_bias_ladder["dx_iface"]
    n_iface = len(dx[0])
    lines = []
    for k in range(n_iface):
        seq = tuple(d[k] for d in dx)
        r1 = seq[0] / seq[1]
        r2 = seq[1] / seq[2]
        lines.append(
            f"  interface {k}: dx_cell = "
            f"{seq[0]*1e9:.4f} -> {seq[1]*1e9:.4f} -> {seq[2]*1e9:.4f} nm "
            f"(x{r1:.2f}, x{r2:.2f} per N_grid doubling)"
        )
        assert r1 > MIN_IFACE_REFINE and r2 > MIN_IFACE_REFINE, (
            "interface cell did not refine faster than 2x per N_grid "
            "doubling — the premise of review item 7#6 is not met:\n"
            + "\n".join(lines)
        )
    print("\n[mesh] interface-cell refinement\n" + "\n".join(lines))


def test_guard_no_silent_dark_fallback(fixed_bias_ladder, stack):
    """``solve_illuminated_ss`` returns DARK equilibrium without raising.

    ``illuminated_ss.py:58-60`` — ``if not sol.success: return y_dark``.  Dark
    J ~ 0, so a settle failure would be read as a perfectly converged number.
    Guard: the short-circuit current must exceed half the incident photon
    current, and the power-quadrant current must be positive.
    """
    q_phi = Q * stack.Phi
    for N_grid, J0, J09 in zip(LADDER, fixed_bias_ladder["J0"], fixed_bias_ladder["J09"]):
        assert J0 > 0.5 * q_phi, (
            f"N_grid={N_grid}: J(V=0) = {J0:.4f} A/m2 is below half the "
            f"incident photon current ({0.5*q_phi:.2f}) — solve_illuminated_ss "
            "probably fell back to dark equilibrium."
        )
        assert J09 > 0.0, f"N_grid={N_grid}: J(V={V_POWER}) = {J09:.4f} <= 0"


def test_guard_voc_is_bracketed(sweep_ladder):
    """Without a J = 0 crossing, V_oc / FF / PCE are sentinel zeros."""
    for N_grid, res in zip(LADDER, sweep_ladder):
        assert res.metrics_fwd.voc_bracketed, (
            f"N_grid={N_grid}: forward sweep never crossed J = 0 at "
            f"V_max={V_MAX} — V_oc/FF/PCE are sentinel zeros and every "
            "comparison below is meaningless."
        )


def test_guard_no_flat_band_spike_at_any_mesh_level(sweep_ladder):
    """dJ/dV <= 0 on an illuminated forward scan, to 0.5% of J_sc.

    NOT cosmetic.  An isolated upward jump dJ at the sample nearest MPP raises
    P_mpp by ~dJ*V_mpp, i.e. moves FF by ~dJ/J_sc, so a surviving spike
    corrupts the FF ladder directly.  A 3.0%-of-J_sc spike that survives the
    shipped ``max_step = dt/20`` cap (``jv_sweep.py:539-547``, whose docstring
    says it "removes the spikes") WAS observed on this preset at N_grid = 60,
    n_points = 81, V = 0.9975 V via ``quasi_static_sweep``:
    J = 217.994 -> 224.680 -> 212.510 A/m2.  It is under the 5%-of-J_sc
    threshold of the shipped guard in ``test_jv_regression.py``, so nothing in
    the suite currently catches it.  If this trips, the verdict is "the solver
    emitted dJ/dV > 0 on an illuminated forward scan" — a defect to report, NOT
    a tolerance to widen.
    """
    for N_grid, res in zip(LADDER, sweep_ladder):
        J = np.asarray(res.J_fwd, dtype=float)
        J_sc = abs(float(res.metrics_fwd.J_sc))
        d = np.diff(J)
        spike = float(max(d.max(), 0.0))
        assert spike < TOL_SPIKE_REL_JSC * J_sc, (
            f"N_grid={N_grid}: forward sweep has a positive J jump of "
            f"{spike:.4f} A/m2 ({100*spike/J_sc:.2f}% of J_sc={J_sc:.2f}) at "
            f"V={res.V_fwd[int(np.argmax(d)) + 1]:.4f} V — non-physical "
            "dJ/dV > 0 near flat band; FF at this level is not trustworthy."
        )


# ---------------------------------------------------------------------------
# LEG A — fixed-bias terminal current and interface recombination current
# ---------------------------------------------------------------------------


def test_short_circuit_current_converges(fixed_bias_ladder):
    """J(V=0) — definitionally J_sc, with no interpolation/extraction step.

    The interior-face current spread from ``_compute_current_ss_with_spread``
    is REPORTED, never asserted — its own docstring says to report and not
    gate on it, and this test follows that.  It is printed because it is an
    unexplained observation worth a separate look: the spread GROWS with
    refinement on this preset even while the median current contracts
    cleanly at rho = 0.26.
    """
    vals = fixed_bias_ladder["J0"]
    line = _report("J(V=0)", vals, "A/m2")
    print("\n[mesh] " + line)
    print(
        "[mesh] "
        + _report("  J spread(V=0)", fixed_bias_ladder["spread0"], "A/m2")
    )
    _assert_contracting("J(V=0)", vals, "A/m2")
    res = _richardson_residual(vals)
    assert res < TOL_J_REL * abs(vals[-1]), (
        f"residual discretisation error in J(V=0) exceeds the 1% bar this "
        f"repo states J_sc at.\n  {line}"
    )


def test_power_quadrant_current_converges(fixed_bias_ladder):
    """J(V=0.9) — same test inside the power quadrant, below V_bi."""
    vals = fixed_bias_ladder["J09"]
    line = _report(f"J(V={V_POWER})", vals, "A/m2")
    print("\n[mesh] " + line)
    print(
        "[mesh] "
        + _report(f"  J spread(V={V_POWER})", fixed_bias_ladder["spread09"], "A/m2")
    )
    _assert_contracting(f"J(V={V_POWER})", vals, "A/m2")
    res = _richardson_residual(vals)
    assert res < TOL_J_REL * abs(vals[-1]), (
        f"residual discretisation error in J(V={V_POWER}) exceeds 1%.\n  {line}"
    )


def test_interface_recombination_current_converges(fixed_bias_ladder, stack):
    """The quantity review §7 #6 names explicitly, at both biases.

    Measured by calling the solver's own ``_apply_interface_recombination``
    sink on the converged state and integrating the resulting node sink over
    the dual cells — so this measures whatever interface formulation is
    actually active (pinned by the formulation guard above), not a
    re-implementation that can drift.
    """
    J_sc_fine = abs(fixed_bias_ladder["J0"][-1])
    lines = []
    for key, label in (("Jif0", "J_iface(V=0)"), ("Jif09", f"J_iface(V={V_POWER})")):
        vals = fixed_bias_ladder[key]
        line = _report(label, vals, "A/m2")
        lines.append(line)
        _assert_contracting(label, vals, "A/m2")
        _, d32 = _diffs(vals)
        assert abs(d32) < TOL_JIFACE_REL_JSC * J_sc_fine, (
            f"{label} finest-pair change {abs(d32):.5g} A/m2 = "
            f"{100*abs(d32)/J_sc_fine:.4f}% of J_sc exceeds the 0.1%-of-J_sc "
            f"bar.\n  {line}"
        )
    print("\n[mesh] " + "\n[mesh] ".join(lines))


# ---------------------------------------------------------------------------
# LEG B — the shipped extraction: V_oc, J_sc, FF from compute_metrics
# ---------------------------------------------------------------------------


def test_voc_converges(sweep_ladder):
    vals = tuple(float(r.metrics_fwd.V_oc) for r in sweep_ladder)
    line = _report("V_oc", vals, "V")
    print("\n[mesh] " + line)
    _assert_contracting("V_oc", vals, "V")
    res = _richardson_residual(vals)
    assert res < TOL_VOC_V, (
        f"residual discretisation error in V_oc ({res*1e3:.3f} mV) exceeds the "
        f"5 mV bar this repo states V_oc at.\n  {line}"
    )


def test_jsc_converges(sweep_ladder):
    vals = tuple(float(r.metrics_fwd.J_sc) for r in sweep_ladder)
    line = _report("J_sc", vals, "A/m2")
    print("\n[mesh] " + line)
    _assert_contracting("J_sc", vals, "A/m2")
    res = _richardson_residual(vals)
    assert res < TOL_J_REL * abs(vals[-1]), (
        f"residual discretisation error in J_sc exceeds 1%.\n  {line}"
    )


def test_ff_converges(sweep_ladder):
    """FF: a TOLERANCE test on the finest pair, deliberately not a ratio test.

    FF's mesh signal is close to the measurement floor.  Via an
    extraction-free route (Brent maximisation of V*J over fixed-bias steady
    states) FF moves only 0.788830 -> 0.788895 between N = 31 and N = 61, an
    8e-5 relative change, comparable to the Radau rtol = 1e-4 noise floor.  A
    difference-ratio test on FF would therefore largely be measuring noise:
    rho(FF) is 0.17 at n_points = 81 but 1.31 at n_points = 41 — not a stable
    statistic.  The contraction ratio is REPORTED, not asserted.  Anyone
    wanting a genuine ratio test on FF must run at rtol <= 1e-7 with a fine
    P(V) grid, at much higher cost.
    """
    vals = tuple(float(r.metrics_fwd.FF) for r in sweep_ladder)
    line = _report("FF", vals)
    print("\n[mesh] " + line)
    rel = abs(vals[2] - vals[1]) / abs(vals[2])
    assert rel < TOL_FF_REL, (
        f"FF finest-pair relative change {100*rel:.4f}% exceeds the 0.5% bar "
        f"propagated from the V_oc (0.42%) and J_sc (1%) bars.\n  {line}"
    )


# ---------------------------------------------------------------------------
# The optical half of the J_sc mesh error, isolated
# ---------------------------------------------------------------------------
#
# Not in the original design.  Added because the review's motivating
# observation was a ~2.5% J_sc shift between mesh levels, and the discrete
# generation integral can be evaluated exactly, in milliseconds, with no
# solver in the loop — which attributes essentially all of that shift and,
# on the way, turns up a hard physical violation (see below).


def _generation_integral(stack, N_grid: int) -> float:
    """q * sum(G(x) * dx_cell) — the photocurrent the mesh actually generates.

    This is precisely the quadrature ``assemble_rhs`` performs implicitly: it
    adds ``G`` from ``beer_lambert_generation`` (mol.py:1929) to each node's
    dn/dp, and each node owns ``dx_cell`` of volume.
    """
    x = _grid(stack, N_grid)
    mat = build_material_arrays(x, stack)
    G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    return Q * float(np.sum(G * mat.dx_cell))


def test_generation_integral_converges(stack):
    """The absorbed-photon budget is closed-form, so this is a TRUE error test.

    Beer-Lambert with alpha non-zero only in the absorber gives an exact
    absorbed-photon ceiling::

        J_max = q * Phi * (1 - exp(-alpha * d_abs)) = 223.0673 A/m2

    for this preset (alpha = 1.3e7 1/m, d = 400 nm, Phi = 1.4e21 1/m2/s).  The
    error of the discrete integral against it must shrink monotonically along
    the ladder and be inside the 1% J_sc bar at the finest level.
    """
    elec = electrical_layers(stack)
    optical_depth = sum(
        float(l.params.alpha) * float(l.thickness)
        for l in elec if l.params is not None
    )
    J_max = Q * stack.Phi * (1.0 - math.exp(-optical_depth))
    vals = tuple(_generation_integral(stack, N) for N in LADDER)
    errs = tuple(abs(v - J_max) for v in vals)
    line = (
        f"q*int(G dx)     N_grid={LADDER}: {vals[0]:.5f}, {vals[1]:.5f}, "
        f"{vals[2]:.5f} A/m2 | exact J_max={J_max:.5f} | "
        f"|err| = {errs[0]:.5f}, {errs[1]:.5f}, {errs[2]:.5f} "
        f"(x{errs[0]/errs[1]:.2f}, x{errs[1]/errs[2]:.2f} per doubling)"
    )
    print("\n[mesh] " + line)
    assert errs[1] < errs[0] and errs[2] < errs[1], (
        f"discrete generation integral does not converge to the closed-form "
        f"absorbed-photon budget.\n  {line}"
    )
    assert errs[2] < TOL_J_REL * J_max, (
        f"generation-integral error at the finest level exceeds 1%.\n  {line}"
    )


def test_generation_integral_respects_photon_flux_ceiling(stack):
    """HARD PHYSICS BOUND, not a convergence statement.

    The device cannot create more electron-hole pairs per second than there
    are incident photons: ``q * integral(G dx) <= q * Phi``, for ANY mesh.
    This is conservation, not accuracy — no tolerance is involved and no
    refinement argument can excuse a breach.

    ``beer_lambert_generation`` point-samples ``G = Phi*alpha*exp(-tau(x))`` at
    the nodes and the solver integrates it with node-centred ``dx_cell``
    weights.  That quadrature is not photon-conserving: on a mesh coarse
    relative to the absorption length (1/alpha = 76.9 nm here) it over-counts
    the sharply-peaked front-of-absorber generation.  A photon-conserving
    discretisation would instead assign each cell the exact absorbed fraction
    ``Phi * (exp(-tau_left) - exp(-tau_right))``, which telescopes to the exact
    budget on every mesh.
    """
    q_phi = Q * stack.Phi
    vals = tuple(_generation_integral(stack, N) for N in LADDER)
    lines = [
        f"  N_grid={N:3d}: q*int(G dx) = {v:.5f} A/m2 vs q*Phi = "
        f"{q_phi:.5f} ({100*(v-q_phi)/q_phi:+.3f}%)"
        for N, v in zip(LADDER, vals)
    ]
    print("\n[mesh] photon-flux ceiling\n" + "\n".join(lines))
    over = [(N, v) for N, v in zip(LADDER, vals) if v > q_phi]
    assert not over, (
        "discrete optical generation EXCEEDS the incident photon flux — the "
        "solver creates carriers that have no photons to come from:\n"
        + "\n".join(lines)
        + "\n\nThis is a conservation violation in the Beer-Lambert quadrature "
        "(perovskite_sim/physics/generation.py:24 point-sampling + "
        "perovskite_sim/solver/mol.py:1929 node-centred dx_cell weights), not "
        "a mesh-accuracy question. It is NOT confined to unrealistically "
        "coarse grids: measured +5.881% at N_grid=30, +1.111% at N_grid=60 "
        "(the resolution tests/regression/test_jv_regression.py runs at) and "
        "+0.064% at N_grid=100 (the run_jv_sweep default). Fix by integrating "
        "the absorbed fraction per cell, Phi*(exp(-tau_L)-exp(-tau_R)), "
        "instead of point-sampling G and multiplying by dx_cell."
    )
