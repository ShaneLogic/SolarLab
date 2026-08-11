import numpy as np
import pytest
import perovskite_sim.physics.continuity as continuity
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.physics.generation import (
    beer_lambert_generation,
    dual_cell_widths,
)

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
    """G integrates to the absorbed-photon budget under the SOLVER's rule.

    The quadrature that matters is the one ``carrier_continuity_rhs``
    actually performs: it adds ``G[i]`` to ``dn[i]`` and weights node ``i``
    by ``dx_cell[i]``, i.e. a node-centred RECTANGLE rule -- not a
    trapezoid.  ``beer_lambert_generation`` is built to make that rule
    exact on every mesh, so this asserts equality to round-off rather than
    the 1e-3 a convergent-but-inexact quadrature would need.

    This used to be written as ``trapezoid(G, x)``, which passed only
    because the old point-sampled G happened to be a smooth pointwise
    rate.  Trapezoid is now off by ~5e-3 here, for a benign reason worth
    recording: the two boundary nodes are deliberately HALF-loaded
    (``G[0] = absorbed_in_half_cell / dx[0]``) so that multiplying by the
    solver's full-width ``dx_cell[0] = dx[0]`` returns the right number of
    photons.  Trapezoid then halves those endpoints a second time.
    """
    x = np.linspace(0, 400e-9, 200)
    alpha = 1e7   # m⁻¹
    Phi = 2.5e21  # photon flux [m⁻² s⁻¹]
    G = beer_lambert_generation(x, alpha, Phi)
    L = x[-1]
    expected = Phi * (1 - np.exp(-alpha * L))

    absorbed = float(np.sum(G * dual_cell_widths(x)))
    np.testing.assert_allclose(absorbed, expected, rtol=1e-12)

    # Hard physical ceiling: the device cannot absorb more photons than
    # arrive, on ANY mesh.  Coarse grids are where the old point-sampled
    # quadrature broke this (+20.8 % at 15 nodes).
    for n_nodes in (5, 10, 15, 30, 200):
        xs = np.linspace(0, L, n_nodes)
        Gs = beer_lambert_generation(xs, alpha, Phi)
        total = float(np.sum(Gs * dual_cell_widths(xs)))
        assert total <= Phi * (1.0 + 1e-12), (
            f"generation exceeds the incident photon flux on a {n_nodes}-node "
            f"mesh: {total:.6e} > Phi = {Phi:.6e}"
        )


def test_exclusive_interface_face_removes_only_the_declared_sg_bypass(monkeypatch):
    x = np.linspace(0.0, 4.0e-9, 5)
    state = np.full(5, NI)
    params = dict(
        D_n=np.ones(4),
        D_p=np.ones(4),
        V_T=0.025852,
        ni_sq=NI**2,
        tau_n=1.0e30,
        tau_p=1.0e30,
        n1=NI,
        p1=NI,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
    )
    monkeypatch.setattr(
        continuity,
        "sg_fluxes_n",
        lambda *args: np.array([1.0, 2.0, 3.0, 4.0]),
    )
    monkeypatch.setattr(
        continuity,
        "sg_fluxes_p",
        lambda *args: np.array([5.0, 6.0, 7.0, 8.0]),
    )
    baseline_n, baseline_p = carrier_continuity_rhs(
        x,
        np.zeros(5),
        state,
        state,
        np.zeros(5),
        params,
    )
    exclusive_n, exclusive_p = carrier_continuity_rhs(
        x,
        np.zeros(5),
        state,
        state,
        np.zeros(5),
        {**params, "exclusive_interface_faces": (1,)},
    )

    np.testing.assert_array_equal(exclusive_n[[0, 3, 4]], baseline_n[[0, 3, 4]])
    np.testing.assert_array_equal(exclusive_p[[0, 3, 4]], baseline_p[[0, 3, 4]])
    assert exclusive_n[1] != baseline_n[1]
    assert exclusive_n[2] != baseline_n[2]
    assert exclusive_p[1] != baseline_p[1]
    assert exclusive_p[2] != baseline_p[2]
