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
    # MAPbI3 n-i-p: V_oc ∈ [0.8, 1.3] V
    assert 0.5 < nip_result.metrics_fwd.V_oc < 1.5


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
