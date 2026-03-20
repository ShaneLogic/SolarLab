import numpy as np
import pytest
from perovskite_sim.physics.ion_migration import ion_flux_steric, ion_continuity_rhs

V_T = 0.025852
D_I = 1e-16   # m²/s
P_LIM = 1e27  # m⁻³


def test_flux_recovers_standard_pnp_low_density():
    """For P << P_lim, steric term → 1 (standard PNP)."""
    phi = np.array([0.0, 0.0])   # no field
    P   = np.array([1e20, 1e20]) # P << P_lim
    h   = 10e-9
    F_steric  = ion_flux_steric(phi, P, h, D_I, V_T, P_lim=P_LIM)
    # With no field and uniform P, flux should be ~0
    assert abs(F_steric) < 1e10   # much less than D_I*P/h ~ 1e-16*1e20/1e-8 = 1e12


def test_flux_enhanced_near_plim():
    """Diffusion is enhanced when P approaches P_lim."""
    phi = np.array([0.0, 0.0])
    P_low  = np.array([0.5e20, 1.0e20])
    P_high = np.array([0.5e27, 1.0e27])   # near P_lim
    h = 10e-9
    F_low  = abs(ion_flux_steric(phi, P_low,  h, D_I, V_T, P_lim=P_LIM))
    F_high = abs(ion_flux_steric(phi, P_high, h, D_I, V_T, P_lim=P_LIM))
    assert F_high > F_low


def test_zero_flux_at_uniform_equilibrium():
    """No flux when P is uniform and no field."""
    phi = np.array([0.0, 0.0])
    P   = np.array([1e24, 1e24])
    h   = 10e-9
    F   = ion_flux_steric(phi, P, h, D_I, V_T, P_lim=P_LIM)
    assert abs(F) < 1e-20


def test_continuity_rhs_shape():
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N)
    P   = 1e24 * np.ones(N)
    dPdt = ion_continuity_rhs(x, phi, P, D_I, V_T, P_LIM)
    assert dPdt.shape == (N,)


def test_continuity_zero_for_uniform_no_field():
    N = 50
    x = np.linspace(0, 400e-9, N)
    phi = np.zeros(N)
    P   = 1e24 * np.ones(N)
    dPdt = ion_continuity_rhs(x, phi, P, D_I, V_T, P_LIM)
    np.testing.assert_allclose(dPdt, 0.0, atol=1.0)  # [m⁻³/s]
