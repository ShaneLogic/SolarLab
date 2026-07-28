"""V_oc must be read where the terminal current REACHES zero, not where its
SIGN flips.

The defect
----------
``compute_metrics`` used to take V_oc from the first positive→non-positive
sign change in J.  On a device whose photocurrent COLLAPSES before the diode
turns on, J does not descend smoothly through zero: it falls three orders of
magnitude in one voltage step and then sits on a residual plateau whose sign
is numerical noise, because past flat band the J-V curve is in the
near-singular-Jacobian region ``CLAUDE.md`` already documents.  The first
sign change can then land a volt past the true open-circuit point.

Measured 2026-07-27 on ``configs/scaps_mirror.yaml`` + Robin contacts
(``S_n_right=1e5, S_p_right=1e-4, S_p_left=1e5, S_n_left=1e-4``) with
ETL ``N_D = 1e18 m^-3``.  Absorber Eg = 1.53 eV, V_bi = 1.300 V, and
``D_ion = 0`` in every layer, so this is not ion history.  Forward branch,
N_grid=30, v_rate=5.0, dV = 75 mV::

    V = 1.350   J = 1.375e+02      still delivering
    V = 1.425   J = 2.140e-02      collapsed to 1e-4 of J_sc
    V = 1.500   J = 1.695e-03  ┐
    ...                        │ plateau at ~7e-6 of J_sc,
    V = 2.400   J = 1.312e-03  │ sign is noise
    V = 2.475   J =-1.614e-02  ┘ <- FIRST sign change
    V = 2.700   J = 1.321e-02      ...and back to positive
    V = 2.775   J =-6.012e+00      the real diode finally turns on

The sign-change rule reports V_oc = 2.4056 V on that sweep — above the
1.53 eV band gap, i.e. thermodynamically impossible — and the SAME device
reports "unbracketed", 1.4367 V or 2.4056 V depending only on V_max and the
voltage sampling.  It also mis-reports FF (0.4745 vs 0.8011).

The guard
---------
Two independent nets, both in ``experiments/jv_sweep.py``:

1. ``_J_ZERO_FRACTION_OF_JSC`` — a terminal-current resolution floor.  The
   bracket is the first pair with ``J[i] > J_tol >= J[i+1]``, with
   ``J_tol = 1e-3·|J_sc|``, and the J=0 interpolation is clamped to that
   bracket.  Provenance of the constant is metrology + diode slope, argued
   on the constant itself; it is NOT fitted to any observed residual.
2. ``thermodynamic_voc_ceiling`` — ``min(Eg)/q`` over the ELECTRICAL layers,
   refused rather than reported.  ``None`` (no ceiling) for the legacy
   ``chi = Eg = 0`` presets, which declare no gap.

What "voc_bracketed" means after this change
--------------------------------------------
Exactly one thing: an open-circuit point was resolved inside the window and
V_oc / FF / PCE are physical.  A collapsed-current device is ``True``: a cell
delivering less current than any real measurement can distinguish from zero
HAS reached open circuit, its J_sc and P_mpp are real, and reporting
``False`` would additionally invite ``run_jv_sweep(v_max_max_attempts=N)`` to
extend V_max further into the region whose sign is noise — which is measured
above to produce 2.4056 V.  ``False`` is reserved for "no open-circuit point
in this window" and "the one found is impossible".

The synthetic tests below pin the extraction algebra exactly; the
``TestRealCollapsedDevice`` group pins that the shipped solver path actually
lands in the regime the algebra is for.
"""
from __future__ import annotations

import dataclasses
import glob

import numpy as np
import pytest

from perovskite_sim.experiments import jv_sweep as jv
from perovskite_sim.experiments.jv_sweep import (
    compute_metrics,
    run_jv_sweep,
    thermodynamic_voc_ceiling,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml

# scaps_mirror absorber gap; the ceiling for that stack is min(Eg)/q = 1.53 V.
EG_SCAPS_ABSORBER = 1.53

V_T_300K = 0.025852  # k_B*T/q at 300 K, V


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sign_change_metrics(V, J, **kw):
    """Metrics under the PRE-guard rule, for use as an A/B control.

    Setting the resolution floor to 0.0 recovers the old rule identically,
    not merely approximately: ``np.sign(J[:-1]) > 0`` is ``J[:-1] > 0`` and
    ``np.sign(J[1:]) <= 0`` is ``J[1:] <= 0``.  The bracket clamp is then a
    no-op, because for a genuine sign change the interpolated
    ``V[i] + J[i]/(J[i]-J[i+1]) * dV`` already lies in ``(V[i], V[i+1]]``.

    Restoring the module attribute afterwards keeps this usable without a
    monkeypatch fixture from inside a loop.
    """
    saved = jv._J_ZERO_FRACTION_OF_JSC
    jv._J_ZERO_FRACTION_OF_JSC = 0.0
    try:
        return compute_metrics(V, J, V_oc_max=None, **kw)
    finally:
        jv._J_ZERO_FRACTION_OF_JSC = saved


def collapsed_curve(V_max: float):
    """The measured collapsed-current shape, on a grid of FIXED resolution
    dV = 0.15 V truncated at ``V_max``.

    Fixing dV and varying only the RANGE is what makes the three windows
    comparable: they share every sample they have in common, so any
    difference in the extracted V_oc is the extractor's, not the grid's.
    """
    dV = 0.15
    n = int(round(V_max / dV)) + 1
    V = np.arange(n, dtype=float) * dV
    J = np.empty(n, dtype=float)
    # 0.00 .. 1.20 V: photocurrent, drooping into the knee.
    photo = np.array([220.0, 219.9, 219.6, 219.0, 218.0, 216.0,
                      213.0, 205.0, 180.0])[: min(n, 9)]
    J[: len(photo)] = photo
    if n > 9:
        # 1.35 .. 2.70 V: collapsed residual plateau. Positive, ~2e-5 of
        # J_sc, with the sign wandering — this is the measured noise floor.
        plateau = np.array([0.0054, 0.0031, 0.0056, 0.0021, 0.0044,
                            0.0019, -0.0052, 0.0028, 0.0056, 0.0013])
        J[9: min(n, 19)] = plateau[: min(n, 19) - 9]
    if n > 19:
        # 2.85 V on: the real diode finally turns on.
        J[19:] = np.array([-6.9, -8.4])[: n - 19]
    return V, J


def diode_curve(n_points=40, V_max=1.30, J_sc=220.0, m=1.5, V_oc_target=1.10):
    """A healthy single-diode J-V curve: J = J_sc - J_0*(exp(V/mV_T) - 1)."""
    J_0 = J_sc / (np.exp(V_oc_target / (m * V_T_300K)) - 1.0)
    V = np.linspace(0.0, V_max, n_points)
    return V, J_sc - J_0 * (np.expm1(V / (m * V_T_300K)))


# ---------------------------------------------------------------------------
# 1. the defect itself, on synthetic curves
# ---------------------------------------------------------------------------
def test_collapsed_plateau_does_not_report_a_super_bandgap_voc():
    """The headline invariant: q*V_oc is a quasi-Fermi-level splitting and
    cannot exceed the band gap, so no extraction may return V_oc > Eg/q."""
    V, J = collapsed_curve(3.0)
    m = compute_metrics(V, J, V_oc_max=EG_SCAPS_ABSORBER)

    assert m.voc_bracketed is True
    assert 0.0 < m.V_oc <= EG_SCAPS_ABSORBER, (
        f"V_oc={m.V_oc:.4f} V exceeds the {EG_SCAPS_ABSORBER} eV gap"
    )
    # ...and the rule it replaces does exceed it, so this test has teeth.
    old = sign_change_metrics(V, J)
    assert old.voc_bracketed is True
    assert old.V_oc > EG_SCAPS_ABSORBER, (
        "control failed: the sign-change rule should read V_oc off the "
        f"noise flip at ~2.7 V, got {old.V_oc:.4f}"
    )


def test_collapsed_voc_is_taken_at_the_collapse_not_the_noise_flip():
    """V_oc lands in the sample pair where the current actually died."""
    V, J = collapsed_curve(3.0)
    m = compute_metrics(V, J, V_oc_max=EG_SCAPS_ABSORBER)
    # Current is 205 A/m^2 at V=1.20 and 5.4 mA/m^2 at V=1.35: the device
    # reaches open circuit inside that pair and nowhere else.
    assert 1.20 <= m.V_oc <= 1.35 + 1e-12


def test_collapsed_voc_is_independent_of_v_max():
    """Same device, same voltage RESOLUTION, three different sweep RANGES →
    one V_oc.  Exact equality is the right assertion: the three grids share
    every sample below 1.5 V, and V_oc is determined by a bracket that lies
    inside that shared prefix, so no float can differ."""
    got = {}
    for V_max in (1.5, 2.1, 3.0):
        V, J = collapsed_curve(V_max)
        got[V_max] = compute_metrics(V, J, V_oc_max=EG_SCAPS_ABSORBER)

    assert all(m.voc_bracketed for m in got.values())
    assert got[1.5].V_oc == got[2.1].V_oc == got[3.0].V_oc, (
        "V_oc moved with V_max: "
        + ", ".join(f"{k}->{v.V_oc!r}" for k, v in got.items())
    )


def test_sign_change_rule_is_v_max_dependent_on_the_same_curve():
    """Control for the test above: the rule being replaced gives a different
    answer in each window, which is the whole reason for the change."""
    old = {V_max: sign_change_metrics(*collapsed_curve(V_max))
           for V_max in (1.5, 2.1, 3.0)}
    outcomes = {(round(m.V_oc, 9), m.voc_bracketed) for m in old.values()}
    assert len(outcomes) > 1, (
        f"expected the sign-change rule to be V_max-dependent, got {outcomes}"
    )


# ---------------------------------------------------------------------------
# 2. inertness on curves that were never broken
# ---------------------------------------------------------------------------
def test_healthy_diode_curve_is_bit_identical_to_the_sign_change_rule():
    """A curve with a genuine crossing must be untouched — every metric
    equal as a float, not merely close."""
    V, J = diode_curve()
    new = compute_metrics(V, J, V_oc_max=1.6)
    old = sign_change_metrics(V, J)
    assert new.voc_bracketed == old.voc_bracketed is True
    assert new.V_oc == old.V_oc
    assert new.J_sc == old.J_sc
    assert new.FF == old.FF
    assert new.PCE == old.PCE


@pytest.mark.parametrize("n_points", [12, 25, 40, 77, 128])
def test_healthy_curve_inert_across_sampling_densities(n_points):
    """The window ``(0, J_tol]`` is 1e-3 of J_sc wide while the diode drops
    ~0.6*J_sc per sample near V_oc, so a sample landing inside it is a
    ~0.2 % event.  Sweep the sampling density to look for one; if it ever
    happens the displacement is still bounded by eps*m*V_T = 52 uV."""
    V, J = diode_curve(n_points=n_points)
    new = compute_metrics(V, J, V_oc_max=1.6)
    old = sign_change_metrics(V, J)
    assert new.V_oc == pytest.approx(old.V_oc, abs=1e-3 * 1.5 * V_T_300K)


def test_dark_sweep_degenerates_to_the_sign_change_rule():
    """J_sc = 0 ⇒ J_tol = 0 ⇒ the new bracket condition IS the old one."""
    V = np.linspace(0.0, 1.3, 40)
    J = -1e-8 * np.expm1(V / (1.5 * V_T_300K))     # dark diode, J(0) = 0
    new = compute_metrics(V, J, V_oc_max=1.6)
    old = sign_change_metrics(V, J)
    assert new.J_sc == 0.0
    assert new.voc_bracketed == old.voc_bracketed is False
    assert new.V_oc == old.V_oc == 0.0


def test_2d_sign_convention_still_flips_before_extraction():
    """``assume_jsc_positive=False`` must flip J BEFORE the floor is applied,
    or the 2D/grain-sweep callers get J_tol from a negative J_sc."""
    V, J = collapsed_curve(3.0)
    m_pos = compute_metrics(V, J, V_oc_max=EG_SCAPS_ABSORBER)
    m_neg = compute_metrics(V, -J, assume_jsc_positive=False,
                            V_oc_max=EG_SCAPS_ABSORBER)
    assert m_neg.V_oc == m_pos.V_oc
    assert m_neg.J_sc == m_pos.J_sc
    assert m_neg.voc_bracketed == m_pos.voc_bracketed


# ---------------------------------------------------------------------------
# 3. the interpolation clamp
# ---------------------------------------------------------------------------
def test_interpolation_is_clamped_to_its_own_bracket():
    """When ``J[i+1]`` is a small POSITIVE residual instead of a sign flip,
    the linear interpolation to J=0 extrapolates OUTSIDE the bracket, and
    the overshoot is unbounded as ``J[i] → J_tol⁺``.  Constructed here to
    overshoot by ~200 V; the reported V_oc must be the bracket edge."""
    V = np.array([0.0, 1.0, 2.0, 3.0])
    J = np.array([220.0, 0.2201, 0.2190, 0.5])   # J_tol = 0.220
    raw = 1.0 - 0.2201 * (2.0 - 1.0) / (0.2190 - 0.2201)
    assert raw > 100.0, "fixture no longer exercises the overshoot"

    m = compute_metrics(V, J)
    assert m.V_oc == 2.0


def test_clamp_is_a_noop_for_a_genuine_sign_crossing():
    """For J[i] > 0 >= J[i+1] the interpolant is already inside the bracket,
    so the clamp cannot move it — this is why healthy curves are inert."""
    V = np.array([0.0, 1.0, 2.0])
    J = np.array([220.0, 100.0, -50.0])
    m = compute_metrics(V, J)
    assert m.V_oc == pytest.approx(1.0 + 100.0 / 150.0, rel=0, abs=1e-15)
    assert V[1] < m.V_oc <= V[2]


# ---------------------------------------------------------------------------
# 4. the thermodynamic ceiling
# ---------------------------------------------------------------------------
def test_voc_above_the_ceiling_is_refused_with_sentinel_zeros():
    V = np.array([0.0, 1.0, 2.0])
    J = np.array([220.0, 100.0, -50.0])          # crossing at 1.6667 V
    free = compute_metrics(V, J)
    assert free.voc_bracketed is True and free.V_oc > EG_SCAPS_ABSORBER

    refused = compute_metrics(V, J, V_oc_max=EG_SCAPS_ABSORBER)
    assert refused.voc_bracketed is False
    assert refused.V_oc == 0.0 and refused.FF == 0.0 and refused.PCE == 0.0
    # J_sc survives the refusal — it is measured at V=0, far from the doubt.
    assert refused.J_sc == free.J_sc


def test_voc_exactly_at_the_ceiling_is_accepted():
    """The bound is V_oc <= Eg/q, so equality is admissible; only a strict
    excess is impossible."""
    V = np.array([0.0, 1.0, 2.0])
    J = np.array([220.0, 100.0, -50.0])
    exact = compute_metrics(V, J).V_oc
    m = compute_metrics(V, J, V_oc_max=exact)
    assert m.voc_bracketed is True and m.V_oc == exact


def test_ceiling_none_disables_the_check():
    V = np.array([0.0, 1.0, 2.0])
    J = np.array([220.0, 100.0, -50.0])
    assert compute_metrics(V, J, V_oc_max=None).voc_bracketed is True


def test_ceiling_is_the_absorber_gap_on_a_heterostack():
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert thermodynamic_voc_ceiling(stack) == EG_SCAPS_ABSORBER


def test_ceiling_is_none_for_legacy_zero_gap_presets():
    """``nip_MAPbI3``/``pin_MAPbI3`` set chi = Eg = 0 on every layer.  A
    ``min()`` over those would be 0 and would refuse every V_oc, so the
    ceiling must be absent, not zero."""
    for cfg in ("configs/nip_MAPbI3.yaml", "configs/pin_MAPbI3.yaml"):
        stack = load_device_from_yaml(cfg)
        assert all(l.params.Eg == 0.0 for l in electrical_layers(stack))
        assert thermodynamic_voc_ceiling(stack) is None, cfg


def test_ceiling_ignores_optical_only_substrate_layers():
    """A ``role: substrate`` layer carries no carriers, so its gap (or lack
    of one) must not enter the bound."""
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    assert len(stack.layers) == len(electrical_layers(stack)) + 1
    doped = dataclasses.replace(
        stack,
        layers=(
            dataclasses.replace(
                stack.layers[0],
                params=dataclasses.replace(stack.layers[0].params, Eg=0.2),
            ),
        ) + tuple(stack.layers[1:]),
    )
    assert stack.layers[0].role == "substrate"
    assert thermodynamic_voc_ceiling(doped) == thermodynamic_voc_ceiling(stack)


def _device_configs():
    paths = sorted(glob.glob("configs/*.yaml")) + sorted(
        glob.glob("configs/twod/*.yaml"))
    out = []
    for p in paths:
        for loader in (load_device_from_yaml, load_scaps_yaml):
            try:
                out.append((p, loader(p)))
                break
            except Exception:                     # noqa: BLE001, PERF203
                continue
    return out


def test_ceiling_equals_the_absorber_gap_on_every_shipped_preset():
    """``thermodynamic_voc_ceiling`` uses ``min(Eg)`` over the electrical
    layers, but the bound that is a theorem is the ABSORBER's gap.  The two
    agree only while no transport layer is narrower than the absorber.  This
    pins that they agree on everything shipped; if it ever fires, the
    ceiling has become tighter than the thermodynamic bound and can refuse a
    valid V_oc, so the DEFINITION must be revisited — do not relax this."""
    checked = 0
    for path, stack in _device_configs():
        ceiling = thermodynamic_voc_ceiling(stack)
        absorbers = [float(l.params.Eg) for l in electrical_layers(stack)
                     if l.role == "absorber" and float(l.params.Eg) > 0.0]
        if ceiling is None:
            assert not absorbers, f"{path}: absorber declares a gap but the " \
                                  f"ceiling is None"
            continue
        assert absorbers, f"{path}: gaps declared but no role:absorber layer"
        assert ceiling == max(absorbers), (
            f"{path}: ceiling {ceiling} is below the absorber gap "
            f"{max(absorbers)} — a transport layer is now the narrowest "
            f"electrical gap, so min(Eg) is no longer the thermodynamic bound"
        )
        checked += 1
    assert checked >= 5, f"only {checked} gap-declaring configs found"


def test_hysteresis_index_forwards_the_ceiling_to_both_branches():
    """``hysteresis_index`` divides by the reverse-branch PCE, so if only one
    branch were ceiling-checked the index would be computed from a refused
    V_oc on one side and a physical one on the other."""
    V = np.array([0.0, 1.0, 2.0])
    J = np.array([220.0, 100.0, -50.0])          # crossing at 1.6667 V
    assert jv.hysteresis_index(V, J, V, J) == 0.0            # symmetric
    # With both branches refused, PCE is 0 on the reverse branch and the
    # helper returns its documented 0.0 rather than dividing by zero.
    assert jv.hysteresis_index(V, J, V, J,
                               V_oc_max=EG_SCAPS_ABSORBER) == 0.0


# ---------------------------------------------------------------------------
# 5. the shipped solver path
# ---------------------------------------------------------------------------
def _spy_on_compute_metrics(monkeypatch):
    """Record the ``V_oc_max`` every ``compute_metrics`` call receives.

    ``run_jv_sweep`` and ``hysteresis_index`` both resolve the module global,
    so one patch catches all four calls.
    """
    seen: list = []
    real = jv.compute_metrics

    def spy(*a, **kw):
        seen.append(kw.get("V_oc_max", "MISSING"))
        return real(*a, **kw)

    monkeypatch.setattr(jv, "compute_metrics", spy)
    return seen


def test_run_jv_sweep_plumbs_the_ceiling_from_the_stack(monkeypatch):
    """The ``V_oc_max`` kwarg must not be dead code: ``run_jv_sweep`` is the
    caller that owns a ``DeviceStack``, so it must derive the ceiling and
    pass it to every metrics call — including the hysteresis index."""
    seen = _spy_on_compute_metrics(monkeypatch)
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    run_jv_sweep(stack, N_grid=6, n_points=3, v_rate=50.0, V_max=1.0)

    assert seen, "compute_metrics was never called"
    assert set(seen) == {1.6}, (
        f"expected every metrics call to carry the 1.6 eV ceiling, got {seen}"
    )
    # Forward + reverse at minimum; today it is 4 because hysteresis_index
    # recomputes both branches. The count is deliberately not pinned — the
    # invariant is that EVERY call carries the ceiling, asserted above.
    assert len(seen) >= 2, f"only {len(seen)} metrics call(s)"


def test_run_jv_sweep_passes_no_ceiling_for_a_legacy_zero_gap_stack(
        monkeypatch):
    """A legacy chi = Eg = 0 preset declares no gap, so it must get
    ``V_oc_max=None`` — a 0.0 ceiling would refuse every V_oc."""
    seen = _spy_on_compute_metrics(monkeypatch)
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    run_jv_sweep(stack, N_grid=6, n_points=3, v_rate=50.0, V_max=1.0)
    assert seen and set(seen) == {None}, seen



def _collapsed_stack():
    """scaps_mirror + Robin contacts + ETL N_D = 1e18 m^-3 — the device the
    defect was diagnosed on.  Same construction as
    ``test_run_jv_sweep_auto_extend_v_max._scaps_mirror_robin_low_etl``."""
    base = load_scaps_yaml("configs/scaps_mirror.yaml")
    robin = dataclasses.replace(
        base, mode="full",
        S_n_right=1.0e5, S_p_right=1.0e-4,
        S_p_left=1.0e5, S_n_left=1.0e-4,
    )
    layers = list(robin.layers)
    etl = layers[-1]
    layers[-1] = dataclasses.replace(
        etl, params=dataclasses.replace(etl.params, N_D=1.0e18))
    return dataclasses.replace(robin, layers=tuple(layers))


@pytest.fixture(scope="module")
def collapsed_sweeps():
    """Three sweeps over the same device at a FIXED voltage resolution of
    75 mV, differing only in range.  Escalating the range at fixed
    resolution (rather than at fixed n_points) is what isolates
    V_max-dependence from grid-refinement effects."""
    stack = _collapsed_stack()
    return stack, {
        V_max: run_jv_sweep(stack, N_grid=30, n_points=n, v_rate=5.0,
                            V_max=V_max)
        for V_max, n in ((1.5, 21), (2.1, 29), (3.0, 41))
    }


def test_real_collapsed_device_stays_below_its_band_gap(collapsed_sweeps):
    stack, res = collapsed_sweeps
    ceiling = thermodynamic_voc_ceiling(stack)
    assert ceiling == EG_SCAPS_ABSORBER
    for V_max, r in res.items():
        m = r.metrics_fwd
        assert m.voc_bracketed is True, f"V_max={V_max} failed to bracket"
        assert 0.0 < m.V_oc <= ceiling, (
            f"V_max={V_max}: V_oc={m.V_oc:.4f} V above the {ceiling} eV gap"
        )


def test_real_collapsed_device_voc_is_v_max_independent(collapsed_sweeps):
    """The sharpest form of the invariant on the real solver path.

    The forward branch is integrated from V=0 upward, so the three sweeps
    share their whole prefix below 1.5 V and V_oc is decided inside it —
    where the sweep STOPPED must not enter the answer at all.

    Asserted to 1 mV rather than to exact float equality.  The shared
    prefix argument holds in exact arithmetic, but each sweep is an
    ADAPTIVE Radau transient whose total t_span scales with V_max
    (v_rate is fixed), and an adaptive integrator does not guarantee a
    bit-identical trajectory when the span changes.  Measured spread is
    0.027 mV — 1.425000 / 1.424973 / 1.425000 V at V_max = 1.5 / 2.1 / 3.0.

    The 1 mV bound is the resolution V_oc is quoted at in this repo
    (CLAUDE.md pins the SS-vs-transient agreement at 5 mV), so it carries
    ~37x margin over the observed spread while still being three orders
    below the defect it exists to catch: the sign-change rule reports
    unbracketed / unbracketed / 2.4056 V on these same three sweeps, i.e.
    a ~1 V swing.  Relaxing exact equality to 1 mV therefore costs the
    test none of its discriminating power.
    """
    _stack, res = collapsed_sweeps
    voc = {V_max: r.metrics_fwd.V_oc for V_max, r in res.items()}
    spread = max(voc.values()) - min(voc.values())
    assert spread < 1e-3, f"V_oc moved with V_max by {spread*1e3:.4f} mV: {voc}"

    old = {V_max: sign_change_metrics(r.V_fwd, r.J_fwd)
           for V_max, r in res.items()}
    old_outcomes = {(round(m.V_oc, 9), m.voc_bracketed) for m in old.values()}
    assert len(old_outcomes) > 1, (
        "control failed: the sign-change rule should still be V_max-dependent "
        f"on this device, got {old_outcomes}"
    )


def test_real_collapsed_device_keeps_jsc_and_reports_a_sane_ff(
        collapsed_sweeps):
    """J_sc is measured at V=0, nowhere near the collapse, so it must be
    identical across the three windows; FF is the metric the old rule
    corrupted (it divides P_mpp by V_oc*J_sc)."""
    _stack, res = collapsed_sweeps
    jsc = {V_max: r.metrics_fwd.J_sc for V_max, r in res.items()}
    assert len(set(jsc.values())) == 1, f"J_sc moved with V_max: {jsc}"
    for V_max, r in res.items():
        assert 0.0 < r.metrics_fwd.FF < 1.0, f"V_max={V_max}"
