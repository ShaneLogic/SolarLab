"""Tests for `experiments/mott_schottky.py` — C-V + depletion-doping fit.

Two layers of coverage:

1. Pure-math unit tests on synthetic Mott-Schottky data (no solver) —
   these tightly bound the V_bi and N_eff the fitter must recover when
   given a curve that is guaranteed-linear in 1/C² vs V by construction.
2. A wrapper test that supplies an analytic capacitive impedance and checks
   the complete Z -> Y -> C(V) -> fit -> MottSchottkyResult path.
3. A real c-Si guard test proving that the old N_grid=30 protocol is rejected
   before integration. Its first p-base cell is comparable to the entire
   depletion width, so its nearly geometric C(V) cannot support a physical
   Mott-Schottky fit.

The real c-Si claim is covered separately by the QF frequency-domain
N=200/300/400 grid/frequency/derivative-step protocol. The general transient
path still requires its own amplitude/cycle certificate.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import K_B, Q
from perovskite_sim.discretization.grid import GridResolutionError
from perovskite_sim.experiments.impedance import ImpedanceResult
from perovskite_sim.experiments.mott_schottky import (
    EPS_0,
    MottSchottkyResult,
    _fit_mott_schottky,
    _resolve_eps_r,
    _select_ms_window,
    run_mott_schottky,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


# ---------------------------------------------------------------------------
# Pure-math unit tests — synthetic Mott-Schottky data, no solver.
# ---------------------------------------------------------------------------

T_TEST = 300.0


def _synthetic_cv(V, V_bi, N, eps_r, T=T_TEST):
    """Build a synthetic C(V) from the Mott-Schottky formula.

    ``C(V) = sqrt(q·ε·ε_0·N / (2·(V_bi − V − kT/q)))``. The ``kT/q``
    majority-carrier tail term is the one ``_fit_mott_schottky`` inverts
    (review F-16), so the round-trip recovers ``V_bi`` exactly. Only
    valid for ``V < V_bi − kT/q``.
    """
    return np.sqrt(
        Q * eps_r * EPS_0 * N / (2.0 * (V_bi - V - K_B * T / Q))
    )


def test_fit_recovers_known_vbi_and_doping():
    """Round-trip: synthetic C(V) → V_bi_fit and N_eff_fit."""
    V = np.linspace(-0.3, 0.4, 20)
    V_bi_true = 0.9
    N_true = 1e22  # m⁻³
    eps_r = 11.7
    C = _synthetic_cv(V, V_bi_true, N_true, eps_r)

    V_bi_fit, N_fit, V_lo, V_hi = _fit_mott_schottky(V, C, eps_r, T_TEST)
    assert abs(V_bi_fit - V_bi_true) < 0.01, (
        f"V_bi_fit={V_bi_fit:.4f} off from {V_bi_true}"
    )
    assert abs(np.log10(N_fit) - np.log10(N_true)) < 0.02, (
        f"N_eff_fit={N_fit:.3e} off from {N_true:.3e}"
    )
    assert V_lo < V_hi


def test_fit_rejects_non_linear_tail():
    """A strongly-curved tail must not pull V_bi_fit or N_eff_fit off.

    Build a pure-MS curve for V in [-0.2, 0.5] V and prepend 3 points on
    a curved branch (``1/C²`` 9x higher than MS would predict at those
    V — simulates a freeze-out / fully-depleted tail that bends sharply
    upward). The window selector should land on the linear segment so
    the extracted (V_bi, N_eff) are close to the true MS values.
    """
    V_good = np.linspace(-0.2, 0.5, 15)
    V_bad = np.array([-0.6, -0.5, -0.4])
    V = np.concatenate([V_bad, V_good])

    V_bi, N, eps_r = 1.0, 1e22, 11.7
    C_good = _synthetic_cv(V_good, V_bi, N, eps_r)
    # Bad tail: C suppressed by 3x → 1/C² inflated by 9x relative to the
    # MS prediction at these biases. A line through the bad+good data
    # has much worse RMS than one through the good segment alone.
    C_bad = _synthetic_cv(V_bad, V_bi, N, eps_r) / 3.0
    C = np.concatenate([C_bad, C_good])

    V_bi_fit, N_fit, V_lo, V_hi = _fit_mott_schottky(V, C, eps_r, T_TEST)
    assert abs(V_bi_fit - V_bi) < 0.03, (
        f"V_bi_fit={V_bi_fit:.3f} off from {V_bi:.3f} — tail leaked "
        "into window"
    )
    assert abs(np.log10(N_fit) - np.log10(N)) < 0.05, (
        f"N_eff_fit={N_fit:.3e} off from {N:.3e} — tail leaked into "
        "window"
    )
    # And the window must sit strictly past the bad tail.
    assert V_lo >= V_good[0] - 1e-9, (
        f"fit window V_lo={V_lo:.3f} started inside bad tail"
    )


def test_window_excludes_smooth_forward_injection_tail():
    """A mildly curved tail must not bias an otherwise linear intercept."""
    V = np.linspace(-0.3, 0.4, 8)
    V_bi = 0.9
    N = 1.0e22
    eps_r = 11.7
    C = _synthetic_cv(V, V_bi, N, eps_r)
    # Smooth forward-injection contamination: only the last point is changed,
    # but a 10%-of-span RMS gate accepted the full window and shifted V_bi.
    C[-1] *= 1.12

    V_bi_fit, N_fit, V_lo, V_hi = _fit_mott_schottky(
        V,
        C,
        eps_r,
        T_TEST,
    )

    assert V_lo == pytest.approx(V[0])
    assert V_hi <= V[-2]
    assert V_bi_fit == pytest.approx(V_bi, abs=0.01)
    assert np.log10(N_fit) == pytest.approx(np.log10(N), abs=0.02)


def test_resolve_eps_r_picks_absorber_layer():
    """With an explicit 'absorber' role, that layer's ε_r must win."""
    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    eps_r = _resolve_eps_r(stack)
    # c-Si homojunction: absorber is p_base with eps_r=11.7.
    assert eps_r == pytest.approx(11.7, rel=1e-3)


def test_fit_returns_nan_on_flat_curve():
    """Flat 1/C² (no information) must return NaN rather than blow up."""
    V = np.linspace(0.0, 0.3, 8)
    C = np.full_like(V, 1e-4)  # constant
    V_bi_fit, N_fit, *_ = _fit_mott_schottky(V, C, eps_r=11.7, T=T_TEST)
    assert not np.isfinite(V_bi_fit) or abs(V_bi_fit) > 1e6
    # Either NaN or absurdly large — both are acceptable "this fit is
    # meaningless" signals. An absurd V_bi also fails downstream
    # sanity checks.


def test_fit_returns_nan_for_non_depletion_slope():
    """A positive 1/C² slope is outside this diode depletion convention."""
    V = np.linspace(-0.2, 0.2, 8)
    one_over_c2 = 2.0e12 + 5.0e11 * V
    C = 1.0 / np.sqrt(one_over_c2)

    V_bi_fit, N_fit, *_ = _fit_mott_schottky(
        V, C, eps_r=11.7, T=T_TEST
    )

    assert np.isnan(V_bi_fit)
    assert np.isnan(N_fit)


def test_fit_returns_nan_when_no_linear_window_is_identifiable():
    V = np.array([-0.3, -0.2, -0.1, 0.0])
    one_over_c2 = np.array([10.0, 9.0, 8.0, 1.0]) * 1.0e7
    C = 1.0 / np.sqrt(one_over_c2)

    assert _select_ms_window(V, one_over_c2) is None
    V_bi_fit, N_fit, V_lo, V_hi = _fit_mott_schottky(
        V,
        C,
        eps_r=11.7,
        T=T_TEST,
    )

    assert np.isnan(V_bi_fit)
    assert np.isnan(N_fit)
    assert V_lo == pytest.approx(V[0])
    assert V_hi == pytest.approx(V[-1])


def test_cv_rejects_noncapacitive_admittance(monkeypatch):
    """Do not turn inductive/numerically invalid Im(Y) into C with abs()."""
    import perovskite_sim.experiments.mott_schottky as ms_module

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    monkeypatch.setattr(
        ms_module,
        "run_impedance",
        lambda *args, **kwargs: type(
            "Result", (), {"Z": np.array([1.0j])}
        )(),
    )

    with pytest.raises(RuntimeError, match="positive capacitive susceptance"):
        run_mott_schottky(stack, V_range=[-0.1, 0.0, 0.1])


# ---------------------------------------------------------------------------
# Wrapper contract and real-grid rejection.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csi_stack():
    return load_device_from_yaml("configs/cSi_homojunction.yaml")


def test_cv_wrapper_recovers_analytic_capacitive_impedance(
    monkeypatch, csi_stack
):
    """The public wrapper preserves a physical C-V curve and its fit."""
    import perovskite_sim.experiments.mott_schottky as ms_module

    V = np.linspace(-0.3, 0.4, 8)
    frequency = 1.0e6
    omega = 2.0 * np.pi * frequency
    V_bi_true = 0.9
    N_true = 1.0e22
    C_expected = _synthetic_cv(V, V_bi_true, N_true, 11.7)
    methods = []

    def analytic_impedance(*args, V_dc, **kwargs):
        methods.append(kwargs["method"])
        C_value = float(_synthetic_cv(V_dc, V_bi_true, N_true, 11.7))
        return ImpedanceResult(
            frequencies=np.array([frequency]),
            Z=np.array([1.0 / (1j * omega * C_value)]),
        )

    monkeypatch.setattr(ms_module, "run_impedance", analytic_impedance)
    r = run_mott_schottky(
        csi_stack,
        V_range=V,
        frequency=frequency,
        impedance_method="quasi_fermi_frequency",
    )

    assert isinstance(r, MottSchottkyResult)
    assert r.V.shape == r.C.shape == r.one_over_C2.shape
    np.testing.assert_allclose(r.C, C_expected, rtol=1e-12)
    assert np.all(np.isfinite(r.one_over_C2))
    assert r.frequency == pytest.approx(frequency)
    np.testing.assert_allclose(r.one_over_C2, 1.0 / (r.C * r.C), rtol=1e-12)
    assert r.V_bi_fit == pytest.approx(V_bi_true, abs=0.01)
    assert np.log10(r.N_eff_fit) == pytest.approx(np.log10(N_true), abs=0.02)
    assert r.V_fit_lo <= r.V_fit_hi
    assert r.V[0] - 1e-9 <= r.V_fit_lo <= r.V_fit_hi <= r.V[-1] + 1e-9
    assert methods == ["quasi_fermi_frequency"] * V.size


def test_csi_cv_rejects_underresolved_grid(csi_stack):
    with pytest.raises(GridResolutionError, match="under-resolved"):
        run_mott_schottky(
            csi_stack,
            V_range=[-0.2, 0.0, 0.2],
            frequency=1.0e5,
            N_grid=30,
            n_cycles=3,
            n_extract=1,
        )


def test_rejects_sparse_v_range(csi_stack):
    """Need at least 3 V points for a meaningful fit."""
    with pytest.raises(ValueError, match="at least 3"):
        run_mott_schottky(
            csi_stack, V_range=[0.0, 0.1], N_grid=30, n_cycles=3, n_extract=1,
        )


def test_rejects_nonpositive_frequency(csi_stack):
    with pytest.raises(ValueError, match="frequency"):
        run_mott_schottky(
            csi_stack, V_range=[-0.1, 0.0, 0.1], frequency=0.0,
            N_grid=30, n_cycles=3, n_extract=1,
        )
