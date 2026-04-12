"""Tests for thermionic emission capping in carrier_continuity_rhs."""
import numpy as np
import pytest
from perovskite_sim.physics.continuity import carrier_continuity_rhs

NI = 3.2e13
Q = 1.602176634e-19


def _make_base_params(N: int) -> dict:
    """Build a minimal params dict for carrier_continuity_rhs."""
    return dict(
        D_n=5.17e-6 * np.ones(N - 1),
        D_p=5.17e-6 * np.ones(N - 1),
        V_T=0.025852,
        ni_sq=NI**2 * np.ones(N),
        tau_n=1e-6 * np.ones(N),
        tau_p=1e-6 * np.ones(N),
        n1=NI * np.ones(N),
        p1=NI * np.ones(N),
        B_rad=5e-22 * np.ones(N),
        C_n=1e-42 * np.ones(N),
        C_p=1e-42 * np.ones(N),
    )


def test_continuity_rhs_with_thermionic_capping():
    """With a chi discontinuity at midpoint, TE capping should produce finite dn/dp."""
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N)
    n = NI * np.ones(N)
    p = NI * np.ones(N)
    G = np.zeros(N)

    # Create chi discontinuity at midpoint (0.3 eV step)
    chi = np.zeros(N)
    chi[N // 2:] = 0.3  # 0.3 eV CB offset at midpoint

    Eg = 1.6 * np.ones(N)

    params = _make_base_params(N)
    params["chi"] = chi
    params["Eg"] = Eg

    # Add TE data: face index just before the midpoint
    mid_face = N // 2 - 1
    params["interface_faces"] = [mid_face]
    params["A_star_n"] = 1.2017e6 * np.ones(N)
    params["A_star_p"] = 1.2017e6 * np.ones(N)
    params["T"] = 300.0

    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)

    assert dn.shape == (N,)
    assert dp.shape == (N,)
    assert np.all(np.isfinite(dn))
    assert np.all(np.isfinite(dp))


def test_continuity_rhs_no_interfaces_unchanged():
    """Without interface_faces, result should be identical to standard call."""
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N)
    n = NI * np.ones(N)
    p = NI * np.ones(N)
    G = np.zeros(N)

    chi = np.zeros(N)
    Eg = 1.6 * np.ones(N)

    params = _make_base_params(N)
    params["chi"] = chi
    params["Eg"] = Eg

    # Standard call (no interface_faces key)
    dn_std, dp_std = carrier_continuity_rhs(x, phi, n, p, G, params)

    # Call with empty interface_faces (should take same code path)
    params_with_empty = dict(params)
    params_with_empty["interface_faces"] = []
    dn_empty, dp_empty = carrier_continuity_rhs(x, phi, n, p, G, params_with_empty)

    np.testing.assert_array_equal(dn_std, dn_empty)
    np.testing.assert_array_equal(dp_std, dp_empty)
