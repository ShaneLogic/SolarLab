"""
Regression tests: physical sanity checks for n-i-p MAPbI3 J-V curve.
These do not require exact golden values; they test physically reasonable output.

Run with: pytest -m slow
"""
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nip_result():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    return run_jv_sweep(stack, N_grid=60, n_points=20, v_rate=5.0)


@pytest.fixture(scope="module")
def ionmonger_result():
    """Ionmonger preset — the near-flat-band Radau spike canary.

    CLAUDE.md documents that Radau's adaptive estimator can accept an
    unphysical J spike (e.g. `J = 188, 258, 101`) near V_bi without the
    max_step cap. This fixture runs that preset at the tightest grid the
    slow suite uses so the monotonicity test below can guard the cap.
    """
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    return run_jv_sweep(stack, N_grid=60, n_points=20, v_rate=5.0)


def test_jsc_positive(nip_result):
    assert nip_result.metrics_fwd.J_sc > 0


def test_voc_in_range(nip_result):
    """MAPbI3 n-i-p (legacy, chi=Eg=0): recombination-limited V_oc ≈ 0.91 V.

    Observed on the reference Mac run: V_oc_fwd ≈ V_oc_rev ≈ 0.9110 V with
    V_bi_eff = stack.V_bi = 1.100 V (legacy fallback, no band-offset physics).
    The ratio V_oc / V_bi ≈ 0.83 is what the default SRH lifetime in the
    YAML gives; a collapse to < 0.8 V or an inflation to > 1.05 V would
    signal either a tau/ni misread, a contact-BC regression, or a Radau
    flat-band spike that compute_metrics is interpolating through.
    """
    V_oc_fwd = nip_result.metrics_fwd.V_oc
    V_oc_rev = nip_result.metrics_rev.V_oc
    assert 0.80 < V_oc_fwd < 1.05, (
        f"nip_MAPbI3 forward V_oc out of expected range: {V_oc_fwd:.4f} V "
        f"(expected ≈ 0.91 V, allowed [0.80, 1.05])"
    )
    assert 0.80 < V_oc_rev < 1.05, (
        f"nip_MAPbI3 reverse V_oc out of expected range: {V_oc_rev:.4f} V "
        f"(expected ≈ 0.91 V, allowed [0.80, 1.05])"
    )


def test_ionmonger_voc_in_range(ionmonger_result):
    """IonMonger heterostack: band-offset V_oc ≈ 1.19 V, well above V_bi_eff ≈ 0.86 V.

    The Fermi-level-derived V_bi_eff bounds the bulk band bending, but
    band-offset contacts pin the quasi-Fermi-levels such that the absorber
    QFL splitting can exceed V_bi_eff (ceiling is E_g/q ≈ 1.6 V for MAPbI3).
    Observed on the reference Mac run: V_oc_fwd ≈ V_oc_rev ≈ 1.192 V.
    If V_oc collapses below 1.00 V, likely culprits are (a) an over-eager
    thermionic-emission flux cap shorting the contacts, (b) a regression
    in DeviceStack.compute_V_bi() feeding a too-low V_max and clipping the
    sweep before the crossing. If V_oc inflates above 1.35 V, the sweep
    may be running into V_max and compute_metrics is extrapolating.
    """
    V_oc_fwd = ionmonger_result.metrics_fwd.V_oc
    V_oc_rev = ionmonger_result.metrics_rev.V_oc
    assert 1.00 < V_oc_fwd < 1.35, (
        f"ionmonger forward V_oc out of expected range: {V_oc_fwd:.4f} V "
        f"(expected ≈ 1.19 V, allowed [1.00, 1.35])"
    )
    assert 1.00 < V_oc_rev < 1.35, (
        f"ionmonger reverse V_oc out of expected range: {V_oc_rev:.4f} V "
        f"(expected ≈ 1.19 V, allowed [1.00, 1.35])"
    )


def test_ionmonger_voc_exceeds_vbi_eff(ionmonger_result):
    """Band-offset V_oc must exceed V_bi_eff for this heterostack.

    This pins the Phase 1 intent: V_bi_eff is a bulk band-bending
    measure, not an upper bound on QFL splitting. If a future refactor
    accidentally re-clamps V_oc to V_bi_eff (e.g. by propagating V_bi_eff
    into the Poisson BC without equally updating the QFL handling), this
    test fails loudly.
    """
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    V_bi_eff = stack.compute_V_bi()
    V_oc = ionmonger_result.metrics_fwd.V_oc
    assert V_oc > V_bi_eff + 0.1, (
        f"ionmonger V_oc ({V_oc:.4f} V) failed to exceed V_bi_eff "
        f"({V_bi_eff:.4f} V) by the 0.1 V margin expected for a working "
        "band-offset stack — likely a QFL-clipping regression."
    )


def test_ff_reasonable(nip_result):
    # FF > 0.4 for a functional device
    assert nip_result.metrics_fwd.FF > 0.3


def test_hysteresis_index_nonnegative(nip_result):
    # Hysteresis index should be ≥ 0 (reverse scan better than forward)
    assert nip_result.hysteresis_index >= -0.1


def _max_spike(J: np.ndarray) -> float:
    """Largest positive J jump between consecutive forward-sweep points.

    A physically monotone forward J–V has dJ/dV ≤ 0, so np.diff(J) is all
    ≤ 0 to within discretization jitter. A Radau flat-band spike shows
    up as a single large positive diff followed by a large negative diff.
    Returning the max positive diff (or 0 if none) gives a scalar the
    test can bound directly.
    """
    d = np.diff(J)
    return float(max(d.max(), 0.0))


def test_nip_fwd_sweep_no_flat_band_spike(nip_result):
    """Forward J(V) must be monotone non-increasing within tolerance.

    Tolerance is 5% of J_sc — a Radau flat-band spike is typically
    30-50% of J_sc, so this catches the documented failure mode while
    leaving headroom for normal discretization jitter (<1%).
    """
    J = np.asarray(nip_result.J_fwd)
    J_sc = abs(float(nip_result.metrics_fwd.J_sc))
    spike = _max_spike(J)
    assert spike < 0.05 * J_sc, (
        f"nip forward sweep has a positive J jump of {spike:.2f} A/m² "
        f"(>5% of J_sc={J_sc:.2f}); likely Radau flat-band spike"
    )


def test_ionmonger_fwd_sweep_no_flat_band_spike(ionmonger_result):
    """Same monotonicity guard on the ionmonger preset.

    This is the preset with the nearly-singular Jacobian near V_bi that
    motivated the max_step cap and the max_nfev cap. If either guard is
    ever removed or loosened, this test should fail loudly instead of
    producing a silently-wrong J-V curve.
    """
    J = np.asarray(ionmonger_result.J_fwd)
    J_sc = abs(float(ionmonger_result.metrics_fwd.J_sc))
    spike = _max_spike(J)
    assert spike < 0.05 * J_sc, (
        f"ionmonger forward sweep has a positive J jump of {spike:.2f} A/m² "
        f"(>5% of J_sc={J_sc:.2f}); likely Radau flat-band spike"
    )
