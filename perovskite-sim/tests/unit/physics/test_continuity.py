import numpy as np
import pytest
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.physics.generation import beer_lambert_generation

NI = 3.2e13
Q  = 1.602176634e-19


def test_continuity_shape():
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N); n = NI*np.ones(N); p = NI*np.ones(N)
    eps_r = 24.1*np.ones(N)
    params = dict(D_n=5.17e-6, D_p=5.17e-6, V_T=0.025852,
                  ni_sq=NI**2, tau_n=1e-6, tau_p=1e-6,
                  n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    G = np.zeros(N)
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)
    assert dn.shape == (N,) and dp.shape == (N,)


def test_continuity_zero_at_dark_equilibrium():
    """No net change at dark equilibrium (n=p=ni, no generation)."""
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N); n = NI*np.ones(N); p = NI*np.ones(N)
    params = dict(D_n=5.17e-6, D_p=5.17e-6, V_T=0.025852,
                  ni_sq=NI**2, tau_n=1e-6, tau_p=1e-6,
                  n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    G = np.zeros(N)
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)
    # Interior nodes should be near zero (BCs handle boundaries)
    np.testing.assert_allclose(dn[1:-1], 0.0, atol=1e10)
    np.testing.assert_allclose(dp[1:-1], 0.0, atol=1e10)


def test_beer_lambert_integrates_to_photocurrent():
    x = np.linspace(0, 400e-9, 200)
    alpha = 1e7   # m⁻¹
    Phi = 2.5e21  # photon flux [m⁻² s⁻¹]
    G = beer_lambert_generation(x, alpha, Phi)
    # Integrate G dx ≈ Phi*(1 - exp(-alpha*L))
    L = x[-1]
    expected = Phi * (1 - np.exp(-alpha * L))
    np.testing.assert_allclose(np.trapezoid(G, x), expected, rtol=1e-3)
