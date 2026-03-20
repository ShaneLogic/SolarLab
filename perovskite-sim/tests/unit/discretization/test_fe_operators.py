import numpy as np
import pytest
from perovskite_sim.discretization.fe_operators import bernoulli, sg_flux_n, sg_flux_p


def test_bernoulli_at_zero():
    assert bernoulli(np.array([0.0]))[0] == pytest.approx(1.0)


def test_bernoulli_large_positive():
    # B(x) → 0 for large positive x
    assert bernoulli(np.array([50.0]))[0] == pytest.approx(0.0, abs=1e-10)


def test_bernoulli_large_negative():
    # B(-x) → |x| for large |x| (drift dominates)
    x = np.array([-20.0])
    assert bernoulli(x)[0] == pytest.approx(20.0, rel=1e-6)


def test_bernoulli_symmetry():
    x = np.array([1.5])
    # B(x)*exp(x) == B(-x)
    assert (bernoulli(x) * np.exp(x))[0] == pytest.approx(bernoulli(-x)[0], rel=1e-10)


def test_sg_flux_n_equilibrium():
    """Electron current is zero at thermal equilibrium."""
    V_T = 0.025852
    phi = np.array([0.0, 0.1])   # 100 mV potential difference
    xi = (phi[1] - phi[0]) / V_T
    n_eq = np.array([1e18, 1e18 * np.exp(xi)])  # Boltzmann distribution
    h = 100e-9
    D_n = 5.17e-6  # m²/s
    J = sg_flux_n(phi, n_eq, h, D_n, V_T)
    assert abs(J) < 1e-10 * abs(n_eq[0])


def test_sg_flux_p_equilibrium():
    """Hole current is zero at thermal equilibrium."""
    V_T = 0.025852
    phi = np.array([0.0, 0.1])
    p_eq = np.array([1e18, 1e18 * np.exp(-(phi[1]-phi[0])/V_T)])
    h = 100e-9
    D_p = 5.17e-6
    J = sg_flux_p(phi, p_eq, h, D_p, V_T)
    assert abs(J) < 1e-10 * abs(p_eq[0])
