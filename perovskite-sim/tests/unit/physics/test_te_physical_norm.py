"""Physical thermionic-emission normalization (review F02).

The legacy TE bound is density-weighted (A*T^2 * n, units A/m^5) and near-
inert; the physical form divides by the band-edge DOS to recover the
dimensionally correct emission-velocity current. These tests pin: the
primitive's two forms, exact equilibrium preservation, the bit-identical
default, and bit-identity on DOS-free configs even with the flag on.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep


def test_primitive_legacy_form_unchanged():
    # N_dos=None -> historical density-weighted value, exactly.
    j = thermionic_emission_flux(1e22, 1e21, 0.2, 300.0, 1.2017e6)
    v_t = 1.380649e-23 * 300.0 / 1.602176634e-19
    expected = 1.2017e6 * 300.0**2 * (
        1e22 * np.exp(-0.2 / v_t) - 1e21 * np.exp(0.0)
    )
    assert np.isclose(j, expected, rtol=1e-12)


def test_primitive_physical_form_divides_by_dos():
    N_C = 1e25
    j_legacy = thermionic_emission_flux(1e22, 1e21, 0.2, 300.0, 1.2017e6)
    j_phys = thermionic_emission_flux(1e22, 1e21, 0.2, 300.0, 1.2017e6, N_dos=N_C)
    assert np.isclose(j_phys, j_legacy / N_C, rtol=1e-12)
    # Physical magnitude lands ~25 orders below the density-weighted bound.
    assert abs(j_phys) < 1e-20 * abs(j_legacy)


def test_physical_form_preserves_equilibrium():
    # At equilibrium the two legs cancel: n_L e^(-dE/Vt) == n_R (dE>0).
    v_t = 1.380649e-23 * 300.0 / 1.602176634e-19
    dE = 0.2
    n_R = 1e18
    n_L = n_R * np.exp(dE / v_t)   # makes the bracket exactly zero
    j = thermionic_emission_flux(n_L, n_R, dE, 300.0, 1.2017e6, N_dos=1e25)
    assert abs(j) < 1e-6 * abs(1.2017e6 * 300.0**2 * n_L / 1e25)


def test_ionmonger_bit_identical_no_dos():
    # ionmonger_benchmark has no Nc300/Nv300 -> the flag is a no-op.
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    on = dataclasses.replace(base, te_physical_norm=True)
    common = dict(N_grid=30, n_points=12, v_rate=5.0)
    r_off = run_jv_sweep(base, **common)
    r_on = run_jv_sweep(on, **common)
    np.testing.assert_array_equal(r_off.J_fwd, r_on.J_fwd)
    np.testing.assert_array_equal(r_off.J_rev, r_on.J_rev)


# ---------------------------------------------------------------------------
# Density-unit invariance (2026-07 review, suggested validation case 1).
# ---------------------------------------------------------------------------
#
# A change of density unit (m^-3 -> cm^-3, say) rescales every carrier
# density AND the band-edge DOS by the same factor. A dimensionally sound
# interface-limited current depends on those only through the occupancy
# ratio n/N_dos, so it must be invariant under that rescaling. These two
# tests pin exactly that contrast: the emission-velocity form is invariant,
# the legacy density-weighted bound is not — which is the concrete content
# of the "A*T^2 already is a current density" objection, expressed as a
# behavioural property rather than a units annotation.

_LAMBDA = 1e-6  # m^-3 -> cm^-3


def test_physical_form_is_invariant_under_density_unit_rescaling():
    n_L, n_R, N_dos = 1e22, 1e21, 2.2e24
    j_si = thermionic_emission_flux(
        n_L, n_R, 0.2, 300.0, 1.2017e6, N_dos=N_dos,
    )
    j_rescaled = thermionic_emission_flux(
        n_L * _LAMBDA, n_R * _LAMBDA, 0.2, 300.0, 1.2017e6,
        N_dos=N_dos * _LAMBDA,
    )
    assert np.isclose(j_si, j_rescaled, rtol=1e-12), (
        f"emission-velocity TE flux changed under a pure unit rescaling: "
        f"{j_si:.6e} -> {j_rescaled:.6e}"
    )


def test_legacy_form_scales_with_the_density_unit():
    """The legacy bound is NOT unit-invariant — pinned so the defect stays
    visible and any future 'fix' that silently changes its magnitude is
    caught. See manual Ch17 and the te_physical_norm opt-in."""
    n_L, n_R = 1e22, 1e21
    j_si = thermionic_emission_flux(n_L, n_R, 0.2, 300.0, 1.2017e6)
    j_rescaled = thermionic_emission_flux(
        n_L * _LAMBDA, n_R * _LAMBDA, 0.2, 300.0, 1.2017e6,
    )
    assert np.isclose(j_rescaled, j_si * _LAMBDA, rtol=1e-12), (
        "legacy density-weighted TE bound should scale linearly with the "
        "density unit; it no longer does"
    )
