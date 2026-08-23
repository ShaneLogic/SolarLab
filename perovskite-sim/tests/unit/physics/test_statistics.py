"""Bulk Maxwell-Boltzmann/Fermi-Dirac constitutive tests."""

from __future__ import annotations

import math

import pytest

from perovskite_sim.physics.statistics import (
    FERMI_DIRAC,
    MAXWELL_BOLTZMANN,
    carrier_density_derivative_reduced_fermi_level,
    carrier_density_from_reduced_fermi_level,
    carrier_logarithmic_compressibility,
    carrier_occupation,
    generalized_einstein_factor,
    normalize_carrier_statistics,
    reduced_fermi_level_from_density,
    solve_fully_ionized_charge_neutrality,
)


def test_statistics_identifier_is_strict_and_normalized():
    assert normalize_carrier_statistics(" Fermi_Dirac ") == FERMI_DIRAC
    assert normalize_carrier_statistics(MAXWELL_BOLTZMANN) == MAXWELL_BOLTZMANN
    with pytest.raises(ValueError, match="carrier statistics"):
        normalize_carrier_statistics("fd")
    with pytest.raises(ValueError, match="string"):
        normalize_carrier_statistics(None)


@pytest.mark.parametrize("eta", [-30.0, -3.0, 0.0, 4.0])
def test_maxwell_boltzmann_density_inverse_and_derivative_are_exact(eta):
    density_of_states = 2.8e25
    density = carrier_density_from_reduced_fermi_level(
        eta,
        density_of_states,
    )
    assert density == pytest.approx(density_of_states * math.exp(eta))
    assert reduced_fermi_level_from_density(
        density,
        density_of_states,
    ) == pytest.approx(eta, abs=2.0e-15)
    assert carrier_logarithmic_compressibility(eta) == 1.0
    assert generalized_einstein_factor(eta) == 1.0
    assert carrier_density_derivative_reduced_fermi_level(
        eta,
        density_of_states,
    ) == pytest.approx(density)


@pytest.mark.parametrize(
    ("eta", "reference"),
    [
        (-10.0, 4.5399201053e-5),
        (0.0, 0.7651470246),
        (2.0, 2.8237212774),
        (10.0, 24.0846569646),
    ],
)
def test_bulk_fermi_dirac_occupation_matches_reference_integral(eta, reference):
    assert carrier_occupation(eta, statistics=FERMI_DIRAC) == pytest.approx(
        reference,
        rel=2.0e-7,
    )


@pytest.mark.parametrize("eta", [-25.0, -8.0, 0.0, 5.0, 30.0])
def test_bulk_fermi_dirac_density_inverse_round_trip(eta):
    density_of_states = 1.04e25
    density = carrier_density_from_reduced_fermi_level(
        eta,
        density_of_states,
        statistics=FERMI_DIRAC,
    )
    recovered = reduced_fermi_level_from_density(
        density,
        density_of_states,
        statistics=FERMI_DIRAC,
    )
    assert recovered == pytest.approx(eta, abs=6.0e-6)


def test_fermi_dirac_recovers_dilute_mb_and_has_degenerate_einstein_factor():
    eta_dilute = -15.0
    assert carrier_occupation(
        eta_dilute,
        statistics=FERMI_DIRAC,
    ) == pytest.approx(math.exp(eta_dilute), rel=2.0e-7)
    assert generalized_einstein_factor(
        eta_dilute,
        statistics=FERMI_DIRAC,
    ) == pytest.approx(1.0, rel=2.0e-6)
    assert generalized_einstein_factor(8.0, statistics=FERMI_DIRAC) > 4.0


@pytest.mark.parametrize("eta", [-12.3, -2.7, 0.123, 4.321, 30.0])
def test_fermi_dirac_density_derivative_matches_constitutive_difference(eta):
    density_of_states = 2.8e25
    step = 1.0e-6
    finite_difference = (
        carrier_density_from_reduced_fermi_level(
            eta + step,
            density_of_states,
            statistics=FERMI_DIRAC,
        )
        - carrier_density_from_reduced_fermi_level(
            eta - step,
            density_of_states,
            statistics=FERMI_DIRAC,
        )
    ) / (2.0 * step)
    analytic = carrier_density_derivative_reduced_fermi_level(
        eta,
        density_of_states,
        statistics=FERMI_DIRAC,
    )
    assert analytic == pytest.approx(finite_difference, rel=2.0e-6)


@pytest.mark.parametrize(
    ("acceptors", "donors"),
    [(0.0, 0.0), (0.0, 1.0e22), (3.0e23, 0.0), (2.0e23, 2.0e23)],
)
def test_mb_charge_neutrality_matches_mass_action_solution(acceptors, donors):
    temperature = 300.0
    band_gap = 1.124
    conduction_dos = 2.8e25
    valence_dos = 1.04e25
    state = solve_fully_ionized_charge_neutrality(
        temperature_K=temperature,
        band_gap_eV=band_gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        acceptor_density_m3=acceptors,
        donor_density_m3=donors,
    )
    intrinsic_product = (
        conduction_dos
        * valence_dos
        * math.exp(-band_gap / state.thermal_voltage_V)
    )
    assert state.electron_density_m3 * state.hole_density_m3 == pytest.approx(
        intrinsic_product,
        rel=2.0e-14,
    )
    charge_scale = max(
        state.electron_density_m3,
        state.hole_density_m3,
        acceptors,
        donors,
        1.0,
    )
    assert (
        state.hole_density_m3
        - state.electron_density_m3
        + donors
        - acceptors
    ) == pytest.approx(0.0, abs=2.0e-13 * charge_scale)
    assert state.normalized_charge_residual < 2.0e-13


def test_symmetric_intrinsic_fermi_dirac_level_is_midgap():
    state = solve_fully_ionized_charge_neutrality(
        temperature_K=300.0,
        band_gap_eV=1.0,
        effective_conduction_dos_m3=2.0e25,
        effective_valence_dos_m3=2.0e25,
        statistics=FERMI_DIRAC,
    )
    assert state.reduced_electron_fermi_level == pytest.approx(
        -0.5 * 1.0 / state.thermal_voltage_V,
        abs=2.0e-13,
    )
    assert state.electron_density_m3 == pytest.approx(state.hole_density_m3)
    assert state.normalized_charge_residual < 2.0e-13


@pytest.mark.parametrize(
    ("acceptors", "donors"),
    [(0.0, 1.0e27), (8.0e26, 0.0)],
)
def test_degenerate_charge_neutrality_closes_for_high_doping(acceptors, donors):
    state = solve_fully_ionized_charge_neutrality(
        temperature_K=300.0,
        band_gap_eV=1.124,
        effective_conduction_dos_m3=2.8e25,
        effective_valence_dos_m3=1.04e25,
        acceptor_density_m3=acceptors,
        donor_density_m3=donors,
        statistics=FERMI_DIRAC,
    )
    scale = max(acceptors, donors)
    assert state.normalized_charge_residual < 2.0e-13
    assert abs(
        state.hole_density_m3
        - state.electron_density_m3
        + donors
        - acceptors
    ) < 2.0e-13 * scale
    if donors > acceptors:
        assert state.electron_density_m3 / donors == pytest.approx(1.0, rel=1.0e-8)
        assert state.reduced_electron_fermi_level > 1.0
    else:
        assert state.hole_density_m3 / acceptors == pytest.approx(1.0, rel=1.0e-8)
        assert state.reduced_hole_fermi_level > 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature_K": 0.0},
        {"band_gap_eV": -1.0},
        {"effective_conduction_dos_m3": 0.0},
        {"effective_valence_dos_m3": math.inf},
        {"acceptor_density_m3": -1.0},
        {"donor_density_m3": math.nan},
    ],
)
def test_charge_neutrality_rejects_nonphysical_inputs(kwargs):
    inputs = {
        "temperature_K": 300.0,
        "band_gap_eV": 1.1,
        "effective_conduction_dos_m3": 2.8e25,
        "effective_valence_dos_m3": 1.04e25,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        solve_fully_ionized_charge_neutrality(**inputs)
