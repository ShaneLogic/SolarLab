"""Unit tests for ``perovskite_sim.physics.traps`` (Phase 4a)."""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.traps import (
    exponential_edge_profile,
    gaussian_edge_profile,
    tau_from_trap_density,
    has_trap_profile_params,
)


THICKNESS = 400e-9  # 400 nm absorber


@pytest.fixture
def x_local():
    return np.linspace(0.0, THICKNESS, 201)


class TestExponentialEdgeProfile:
    def test_endpoints_equal_two_decayed_contributions(self, x_local):
        """At x=0 one decay term is 1 (left interface) and the other is
        exp(-d/L_d). The profile value there is
        N_t_bulk + (N_t_interface - N_t_bulk) * (1 + exp(-d/L_d)).
        """
        L_d = 20e-9
        N_t = exponential_edge_profile(
            x_local, THICKNESS, 1e22, 1e20, L_d,
        )
        expected_left = 1e20 + (1e22 - 1e20) * (1.0 + np.exp(-THICKNESS / L_d))
        assert N_t[0] == pytest.approx(expected_left, rel=1e-10)
        # Symmetric around the midpoint.
        assert N_t[0] == pytest.approx(N_t[-1], rel=1e-10)

    def test_midpoint_far_from_both_interfaces_recovers_bulk(self, x_local):
        """At the midpoint the smaller exponential is exp(-d/2L_d);
        with L_d = d/20 each decay term is ~4e-5 so N_t ~ N_t_bulk."""
        L_d = THICKNESS / 40  # each half-layer is 20 L_d
        N_t = exponential_edge_profile(
            x_local, THICKNESS, 1e22, 1e20, L_d,
        )
        mid = len(x_local) // 2
        # Bulk recovery to within the edge residual (< 1 %).
        assert N_t[mid] == pytest.approx(1e20, rel=1e-2)

    def test_monotonic_decay_from_interface(self, x_local):
        """Moving from the left interface into the bulk, N_t decreases
        monotonically in the left half of the layer."""
        N_t = exponential_edge_profile(
            x_local, THICKNESS, 1e22, 1e20, 20e-9,
        )
        half = len(x_local) // 2
        diffs = np.diff(N_t[:half])
        # All negative — monotone decrease.
        assert np.all(diffs < 0.0)

    def test_zero_decay_length_yields_flat_bulk(self, x_local):
        """L_d = 0 is used as a degenerate 'no profile' sentinel."""
        N_t = exponential_edge_profile(
            x_local, THICKNESS, 1e22, 1e20, 0.0,
        )
        assert np.all(N_t == pytest.approx(1e20))

    def test_equal_bulk_and_interface_yields_flat_profile(self, x_local):
        N_t = exponential_edge_profile(
            x_local, THICKNESS, 1e20, 1e20, 20e-9,
        )
        assert np.all(N_t == pytest.approx(1e20))


class TestGaussianEdgeProfile:
    def test_endpoints_have_full_interface_contribution(self, x_local):
        """At x=0 the left Gaussian is 1 and the right is exp(-(d/sigma)²),
        which is tiny when sigma ≪ d."""
        sigma = 10e-9
        N_t = gaussian_edge_profile(x_local, THICKNESS, 1e22, 1e20, sigma)
        right_tail = np.exp(-((THICKNESS / sigma) ** 2))
        expected_left = 1e20 + (1e22 - 1e20) * (1.0 + right_tail)
        assert N_t[0] == pytest.approx(expected_left, rel=1e-10)

    def test_gaussian_decays_faster_than_exponential_in_tail(self, x_local):
        """For the same length parameter the Gaussian tail at ~3-sigma is
        ~1e-4 while the exponential tail at ~3 L_d is ~0.05.
        """
        length = 20e-9
        N_t_exp = exponential_edge_profile(
            x_local, THICKNESS, 1e22, 1e20, length,
        )
        N_t_gauss = gaussian_edge_profile(
            x_local, THICKNESS, 1e22, 1e20, length,
        )
        # In the tail (3× length from nearest interface), Gaussian must
        # be much closer to N_t_bulk than the exponential.
        tail_idx = np.argmin(np.abs(x_local - 3 * length))
        delta_exp = N_t_exp[tail_idx] - 1e20
        delta_gauss = N_t_gauss[tail_idx] - 1e20
        assert delta_gauss < delta_exp

    def test_zero_sigma_yields_flat_bulk(self, x_local):
        N_t = gaussian_edge_profile(x_local, THICKNESS, 1e22, 1e20, 0.0)
        assert np.all(N_t == pytest.approx(1e20))


class TestTauFromTrapDensity:
    def test_uniform_N_t_equals_bulk_yields_unchanged_tau(self):
        tau_bulk = 1e-6 * np.ones(10)
        N_t = 1e20 * np.ones(10)
        tau = tau_from_trap_density(tau_bulk, N_t, 1e20)
        assert np.all(tau == pytest.approx(1e-6))

    def test_N_t_tenfold_higher_shortens_tau_by_tenfold(self):
        tau_bulk = 1e-6 * np.ones(10)
        N_t = 1e21 * np.ones(10)
        tau = tau_from_trap_density(tau_bulk, N_t, 1e20)
        assert np.all(tau == pytest.approx(1e-7, rel=1e-12))

    def test_passivation_regime_floors_at_bulk(self):
        """N_t below N_t_bulk is treated as the bulk value (tau unchanged)
        — a user who wants the passivation regime should compute the
        ratio themselves to avoid a non-physical boost."""
        tau_bulk = 1e-6 * np.ones(5)
        N_t = 1e19 * np.ones(5)
        tau = tau_from_trap_density(tau_bulk, N_t, 1e20)
        assert np.all(tau == pytest.approx(1e-6))


class TestHasTrapProfileParams:
    def _mk(self, **kw):
        class _P:
            pass
        p = _P()
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def test_all_three_set_returns_true(self):
        p = self._mk(
            trap_N_t_interface=1e22,
            trap_N_t_bulk=1e20,
            trap_decay_length=20e-9,
        )
        assert has_trap_profile_params(p) is True

    def test_any_missing_returns_false(self):
        assert has_trap_profile_params(self._mk()) is False
        assert has_trap_profile_params(self._mk(
            trap_N_t_interface=1e22,
        )) is False
        assert has_trap_profile_params(self._mk(
            trap_N_t_interface=1e22,
            trap_N_t_bulk=1e20,
        )) is False
