"""Analytic checks of trap profiles and the energy-resolved bulk-trap closure."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.physics.recombination import (
    BulkTrapDistribution,
    TrapProfile,
    build_bulk_trap_quadrature,
    evaluate_bulk_trap_state,
    exponential_edge_profile,
    gaussian_edge_profile,
    solve_bulk_trap_charge_neutrality,
    tau_from_trap_density,
)

Q = 1.602176634e-19
K_B = 1.380649e-23


# --- position-dependent trap profiles -----------------------------------------


@pytest.mark.parametrize("profile", [exponential_edge_profile, gaussian_edge_profile])
def test_profile_saturates_at_the_interface_density_and_recovers_the_bulk(profile):
    d = 500e-9
    x = np.linspace(0.0, d, 2001)
    density = profile(x, d, 1e22, 1e20, 10e-9)
    assert density[0] == pytest.approx(1e22, rel=1e-12)
    assert density[-1] == pytest.approx(1e22, rel=1e-12)
    assert density[1000] == pytest.approx(1e20, rel=1e-6)
    assert np.all(np.diff(density[:1000]) <= 0.0)


def test_thin_layer_profile_never_overshoots_the_interface_density():
    """F-05: the two-face sum is normalised by its closed-form supremum."""

    for ratio in (1e-2, 0.1, 0.64, 1.0, 10.0, 100.0):
        d = 100e-9
        x = np.linspace(0.0, d, 4001)
        exponential = exponential_edge_profile(x, d, 1e22, 1e20, ratio * d)
        gaussian = gaussian_edge_profile(x, d, 1e22, 1e20, ratio * d)
        assert np.max(exponential) <= 1e22 * (1.0 + 1e-12)
        # The Gaussian candidate pair under-estimates the true peak by < 3 %.
        assert np.max(gaussian) <= 1.03e22


def test_gaussian_tail_decays_faster_than_exponential():
    d, length = 1e-6, 20e-9
    x = np.array([3.0 * length])
    assert gaussian_edge_profile(x, d, 1e22, 0.0, length, edge="left")[0] < (
        exponential_edge_profile(x, d, 1e22, 0.0, length, edge="left")[0]
    )


def test_single_face_profiles_attach_to_one_face_only():
    d = 400e-9
    x = np.array([0.0, d])
    left = exponential_edge_profile(x, d, 1e22, 1e20, 5e-9, edge="left")
    right = exponential_edge_profile(x, d, 1e22, 1e20, 5e-9, edge="right")
    assert left[0] == 1e22 and right[-1] == 1e22
    assert left[-1] == pytest.approx(1e20, rel=1e-12) and right[0] == pytest.approx(1e20, rel=1e-12)


def test_zero_decay_length_is_the_uniform_bulk_sentinel():
    x = np.linspace(0.0, 1e-7, 11)
    assert np.all(exponential_edge_profile(x, 1e-7, 1e22, 1e20, 0.0) == 1e20)
    assert np.all(gaussian_edge_profile(x, 1e-7, 1e22, 1e20, 0.0) == 1e20)


def test_inverse_density_rule_and_its_passivation_floor():
    tau = tau_from_trap_density(1e-6, np.array([1e20, 1e21, 1e22, 1e19]), 1e20)
    np.testing.assert_allclose(tau[:3], [1e-6, 1e-7, 1e-8], rtol=1e-15)
    assert tau[3] == 1e-6  # N_t below the bulk never lengthens the lifetime


@pytest.mark.parametrize("kwargs", [dict(shape="cubic"), dict(edge="middle"), dict(decay_length_m=-1.0)])
def test_trap_profile_declaration_rejects_unknown_choices(kwargs):
    base = dict(interface_density_m3=1e22, bulk_density_m3=1e20, decay_length_m=1e-8)
    base.update(kwargs)
    with pytest.raises(ValueError):
        TrapProfile(**base)


def test_binding_returns_read_only_per_node_lifetimes():
    binding = TrapProfile(1e22, 1e20, 1e-8).bind(np.linspace(0, 1e-7, 5), 1e-7, 1e-6, 2e-6)
    assert binding.tau_n_s.shape == (5,) and binding.tau_p_s[2] > binding.tau_p_s[0]
    with pytest.raises(ValueError):
        binding.tau_n_s[0] = 0.0


# --- energy-resolved bulk traps -------------------------------------------------


def _single(transition="acceptor"):
    return BulkTrapDistribution("single_level", 1e21, 0.9, 1e-19, 3e-20, 1e5, transition)


def test_single_level_state_matches_the_closed_form():
    gap, Nc, Nv, T = 1.5, 2e24, 3e25, 300.0
    V_T = K_B * T / Q
    n, p = 1e18, 4e13
    state = evaluate_bulk_trap_state(n, p, _single(), band_gap_eV=gap, effective_conduction_dos_m3=Nc,
                                     effective_valence_dos_m3=Nv, temperature_K=T)
    cn, cp = 1e-19 * 1e5, 3e-20 * 1e5
    n1 = Nc * math.exp(-(gap - 0.9) / V_T)
    p1 = Nv * math.exp(-0.9 / V_T)
    denominator = cn * (n + n1) + cp * (p + p1)
    occupancy = (cn * n + cp * p1) / denominator
    ni_sq = Nc * Nv * math.exp(-gap / V_T)
    rate = 1e21 * cn * cp * (n * p - ni_sq) / denominator
    assert float(state.occupancy) == pytest.approx(occupancy, rel=1e-14)
    assert float(state.recombination_rate_m3_s) == pytest.approx(rate, rel=1e-14)
    assert float(state.charge_density_C_m3) == pytest.approx(-Q * 1e21 * occupancy, rel=1e-14)
    donor = evaluate_bulk_trap_state(n, p, _single("donor"), band_gap_eV=gap, effective_conduction_dos_m3=Nc,
                                     effective_valence_dos_m3=Nv, temperature_K=T)
    assert float(donor.signed_charge_number_density_m3) == pytest.approx(1e21 * (1.0 - occupancy), rel=1e-12)


def test_rate_and_charge_come_from_one_population_and_vanish_at_mass_action():
    gap, Nc, Nv, T = 1.5, 2e24, 3e25, 300.0
    V_T = K_B * T / Q
    ni_sq = Nc * Nv * math.exp(-gap / V_T)
    n = 1e15
    p = ni_sq / n
    state = evaluate_bulk_trap_state(n, p, _single(), band_gap_eV=gap, effective_conduction_dos_m3=Nc,
                                     effective_valence_dos_m3=Nv, temperature_K=T)
    doubled = evaluate_bulk_trap_state(n, 2.0 * p, _single(), band_gap_eV=gap, effective_conduction_dos_m3=Nc,
                                       effective_valence_dos_m3=Nv, temperature_K=T)
    # R is proportional to n p - ni^2: at mass action only the rounding of
    # p = ni^2 / n survives, a few ulp of the rate at a unit excess.
    eps = np.finfo(float).eps
    assert abs(float(state.recombination_rate_m3_s)) <= 8 * eps * abs(float(doubled.recombination_rate_m3_s))
    assert float(state.charge_density_C_m3) == pytest.approx(-Q * float(state.occupied_density_m3), rel=1e-15)


def test_gaussian_quadrature_is_normalised_and_confined_to_the_gap():
    distribution = BulkTrapDistribution("gaussian", 2e22, 0.75, 1e-19, 1e-19, 1e5, "acceptor", 0.4)
    quadrature = build_bulk_trap_quadrature(distribution, band_gap_eV=1.5, order=64)
    assert math.fsum(quadrature.density_weights_m3) == pytest.approx(2e22, rel=1e-14)
    assert min(quadrature.energy_levels_eV) >= 0.0 and max(quadrature.energy_levels_eV) <= 1.5


def test_narrow_gaussian_reduces_to_the_single_level():
    kwargs = dict(band_gap_eV=1.5, effective_conduction_dos_m3=2e24, effective_valence_dos_m3=3e25,
                  temperature_K=300.0)
    single = evaluate_bulk_trap_state(1e17, 1e15, _single(), **kwargs)
    narrow = evaluate_bulk_trap_state(
        1e17, 1e15, BulkTrapDistribution("gaussian", 1e21, 0.9, 1e-19, 3e-20, 1e5, "acceptor", 1e-5), **kwargs
    )
    assert float(narrow.recombination_rate_m3_s) == pytest.approx(float(single.recombination_rate_m3_s), rel=1e-3)
    assert float(narrow.occupancy) == pytest.approx(float(single.occupancy), rel=1e-3)


def test_charge_and_rate_tangents_match_finite_differences():
    kwargs = dict(band_gap_eV=1.5, effective_conduction_dos_m3=2e24, effective_valence_dos_m3=3e25,
                  temperature_K=300.0)
    distribution = BulkTrapDistribution("gaussian", 2e21, 0.6, 1e-19, 3e-20, 1e5, "donor", 0.08)
    n, p = 3e17, 2e16
    base = evaluate_bulk_trap_state(n, p, distribution, **kwargs)
    for name, rate_d, charge_d in (("n", "recombination_derivative_n_s", "charge_number_derivative_n"),
                                   ("p", "recombination_derivative_p_s", "charge_number_derivative_p")):
        h = 1e-6 * (n if name == "n" else p)
        plus = evaluate_bulk_trap_state(n + h if name == "n" else n, p + h if name == "p" else p, distribution, **kwargs)
        minus = evaluate_bulk_trap_state(n - h if name == "n" else n, p - h if name == "p" else p, distribution, **kwargs)
        rate_fd = (float(plus.recombination_rate_m3_s) - float(minus.recombination_rate_m3_s)) / (2 * h)
        charge_fd = (float(plus.signed_charge_number_density_m3) - float(minus.signed_charge_number_density_m3)) / (2 * h)
        assert float(getattr(base, rate_d)) == pytest.approx(rate_fd, rel=1e-7)
        assert float(getattr(base, charge_d)) == pytest.approx(charge_fd, rel=1e-6)


@pytest.mark.parametrize("doping", [(0.0, 0.0), (1e22, 0.0), (0.0, 3e23)])
def test_trap_aware_neutrality_closes_the_charge_balance(doping):
    distribution = BulkTrapDistribution("gaussian", 5e21, 0.7, 1e-19, 1e-19, 1e5, "acceptor", 0.1)
    result = solve_bulk_trap_charge_neutrality(
        temperature_K=300.0, band_gap_eV=1.6, effective_conduction_dos_m3=2.2e24, effective_valence_dos_m3=1.8e25,
        acceptor_density_m3=doping[0], donor_density_m3=doping[1], distribution=distribution,
    )
    residual = (result.hole_density_m3 - result.electron_density_m3 + doping[1] - doping[0]
                + float(result.trap_state.signed_charge_number_density_m3))
    scale = max(result.electron_density_m3, result.hole_density_m3, *doping, 5e21, 1.0)
    assert abs(residual) / scale <= 1e-12
    assert result.normalized_charge_residual <= 1e-12
    V_T = K_B * 300.0 / Q
    assert result.reduced_electron_fermi_level + result.reduced_hole_fermi_level == pytest.approx(-1.6 / V_T, rel=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(distribution="box"),
        dict(charge_transition="neutral"),
        dict(total_density_m3=0.0),
        dict(center_eV_above_vb=-0.1),
        dict(energy_sigma_eV=0.1),
    ],
)
def test_distribution_rejects_invalid_declarations(kwargs):
    base = dict(distribution="single_level", total_density_m3=1e21, center_eV_above_vb=0.8, sigma_n_m2=1e-19,
                sigma_p_m2=1e-19, thermal_velocity_m_s=1e5, charge_transition="acceptor")
    base.update(kwargs)
    with pytest.raises(ValueError):
        BulkTrapDistribution(**base)


def test_state_rejects_negative_densities_and_levels_outside_the_gap():
    kwargs = dict(effective_conduction_dos_m3=2e24, effective_valence_dos_m3=3e25, temperature_K=300.0)
    with pytest.raises(ValueError):
        evaluate_bulk_trap_state(-1.0, 1.0, _single(), band_gap_eV=1.5, **kwargs)
    with pytest.raises(ValueError):
        evaluate_bulk_trap_state(1.0, 1.0, _single(), band_gap_eV=0.5, **kwargs)
