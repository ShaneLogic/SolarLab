"""Constitutive tests for energy-resolved charged bulk traps."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from perovskite_sim.constants import Q
from perovskite_sim.physics.bulk_traps import (
    BulkTrapDistribution,
    build_bulk_trap_quadrature,
    bulk_trap_distribution_from_mapping,
    evaluate_bulk_trap_state,
    solve_bulk_trap_charge_neutrality,
)
from perovskite_sim.physics.temperature import thermal_voltage


GAP_EV = 1.124
NC_M3 = 2.8e25
NV_M3 = 1.04e25
TEMPERATURE_K = 300.0


def _distribution(**updates) -> BulkTrapDistribution:
    values = {
        "distribution": "gaussian",
        "total_density_m3": 1.0e22,
        "center_eV_above_vb": 0.562,
        "energy_sigma_eV": 0.08,
        "sigma_n_m2": 1.0e-19,
        "sigma_p_m2": 2.0e-19,
        "thermal_velocity_m_s": 1.0e5,
        "charge_transition": "acceptor",
    }
    values.update(updates)
    return BulkTrapDistribution(**values)


def _evaluate(n, p, distribution, order=32):
    return evaluate_bulk_trap_state(
        n,
        p,
        distribution,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        quadrature_order=order,
    )


def test_standard_mapping_is_strict_and_normalized():
    mapping = {
        "distribution": "gaussian",
        "total_density_m3": 2.0e22,
        "center_eV_above_vb": 0.5,
        "energy_sigma_eV": 0.1,
        "sigma_n_m2": 1.0e-19,
        "sigma_p_m2": 2.0e-19,
        "thermal_velocity_m_s": 1.0e5,
        "charge_transition": "DONOR",
    }
    distribution = bulk_trap_distribution_from_mapping(mapping)
    quadrature = build_bulk_trap_quadrature(
        distribution,
        band_gap_eV=GAP_EV,
        order=24,
    )

    assert distribution.charge_transition == "donor"
    assert sum(quadrature.density_weights_m3) == pytest.approx(2.0e22)
    assert all(0.0 <= energy <= GAP_EV for energy in quadrature.energy_levels_eV)
    with pytest.raises(ValueError, match="unknown=.*peak_density"):
        bulk_trap_distribution_from_mapping({**mapping, "peak_density": 1.0})
    with pytest.raises(ValueError, match="missing=.*sigma_p_m2"):
        bulk_trap_distribution_from_mapping(
            {key: value for key, value in mapping.items() if key != "sigma_p_m2"}
        )


def test_single_level_matches_closed_form_occupancy_rate_and_charge():
    distribution = _distribution(
        distribution="single_level",
        energy_sigma_eV=None,
        center_eV_above_vb=0.45,
        total_density_m3=3.0e21,
    )
    n = 4.0e19
    p = 7.0e18
    state = _evaluate(n, p, distribution)
    thermal = thermal_voltage(TEMPERATURE_K)
    n1 = NC_M3 * math.exp(-(GAP_EV - 0.45) / thermal)
    p1 = NV_M3 * math.exp(-0.45 / thermal)
    capture_n = distribution.sigma_n_m2 * distribution.thermal_velocity_m_s
    capture_p = distribution.sigma_p_m2 * distribution.thermal_velocity_m_s
    denominator = capture_n * (n + n1) + capture_p * (p + p1)
    expected_occupancy = (capture_n * n + capture_p * p1) / denominator
    ni_sq = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal)
    expected_rate = (
        distribution.total_density_m3
        * capture_n
        * capture_p
        * (n * p - ni_sq)
        / denominator
    )

    assert state.occupancy.item() == pytest.approx(expected_occupancy)
    assert state.recombination_rate_m3_s.item() == pytest.approx(expected_rate)
    assert state.charge_density_C_m3.item() == pytest.approx(
        -Q * distribution.total_density_m3 * expected_occupancy
    )


def test_donor_and_acceptor_neutral_references_differ_by_one_site_charge():
    acceptor = _distribution()
    donor = replace(acceptor, charge_transition="donor")
    acceptor_state = _evaluate(3.0e20, 5.0e18, acceptor)
    donor_state = _evaluate(3.0e20, 5.0e18, donor)

    assert donor_state.occupancy.item() == pytest.approx(
        acceptor_state.occupancy.item()
    )
    assert (
        donor_state.signed_charge_number_density_m3.item()
        - acceptor_state.signed_charge_number_density_m3.item()
    ) == pytest.approx(acceptor.total_density_m3)
    assert donor_state.charge_number_derivative_n.item() == pytest.approx(
        acceptor_state.charge_number_derivative_n.item()
    )
    assert donor_state.charge_number_derivative_p.item() == pytest.approx(
        acceptor_state.charge_number_derivative_p.item()
    )


def test_gaussian_delta_limit_recovers_the_single_level_model():
    single = _distribution(
        distribution="single_level",
        energy_sigma_eV=None,
        center_eV_above_vb=0.5,
    )
    narrow = _distribution(
        center_eV_above_vb=0.5,
        energy_sigma_eV=1.0e-6,
    )
    single_state = _evaluate(2.0e20, 3.0e19, single)
    narrow_state = _evaluate(2.0e20, 3.0e19, narrow, order=16)

    assert narrow_state.occupancy.item() == pytest.approx(
        single_state.occupancy.item(),
        rel=2.0e-9,
    )
    assert narrow_state.recombination_rate_m3_s.item() == pytest.approx(
        single_state.recombination_rate_m3_s.item(),
        rel=2.0e-9,
    )
    assert narrow_state.charge_density_C_m3.item() == pytest.approx(
        single_state.charge_density_C_m3.item(),
        rel=2.0e-9,
    )


def test_recombination_and_charge_tangents_match_centered_differences():
    distribution = _distribution()
    n = 2.0e20
    p = 8.0e18
    state = _evaluate(n, p, distribution, order=40)
    relative_step = 1.0e-6
    n_plus = _evaluate(n * (1.0 + relative_step), p, distribution, order=40)
    n_minus = _evaluate(n * (1.0 - relative_step), p, distribution, order=40)
    rate_derivative_n = (
        n_plus.recombination_rate_m3_s.item()
        - n_minus.recombination_rate_m3_s.item()
    ) / (2.0 * relative_step * n)
    p_plus = _evaluate(n, p * (1.0 + relative_step), distribution, order=40)
    p_minus = _evaluate(n, p * (1.0 - relative_step), distribution, order=40)
    rate_derivative_p = (
        p_plus.recombination_rate_m3_s.item()
        - p_minus.recombination_rate_m3_s.item()
    ) / (2.0 * relative_step * p)

    assert state.recombination_derivative_n_s.item() == pytest.approx(
        rate_derivative_n,
        rel=2.0e-8,
    )
    assert state.recombination_derivative_p_s.item() == pytest.approx(
        rate_derivative_p,
        rel=2.0e-8,
    )
    thermal = thermal_voltage(TEMPERATURE_K)
    potential_step = 1.0e-7
    plus = _evaluate(
        n * math.exp(potential_step / thermal),
        p * math.exp(-potential_step / thermal),
        distribution,
        order=40,
    )
    minus = _evaluate(
        n * math.exp(-potential_step / thermal),
        p * math.exp(potential_step / thermal),
        distribution,
        order=40,
    )
    charge_derivative = (
        plus.signed_charge_number_density_m3.item()
        - minus.signed_charge_number_density_m3.item()
    ) / (2.0 * potential_step)
    assert state.charge_number_derivative_potential_m3_V.item() == pytest.approx(
        charge_derivative,
        rel=2.0e-8,
    )


def test_equilibrium_detailed_balance_and_trap_aware_neutrality_close():
    distribution = _distribution()
    result = solve_bulk_trap_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=0.0,
        donor_density_m3=5.0e22,
        distribution=distribution,
        quadrature_order=32,
    )

    assert result.neutrality.normalized_charge_residual < 1.0e-12
    assert result.trap_state.minimum_level_occupancy >= 0.0
    assert result.trap_state.maximum_level_occupancy <= 1.0
    ni_sq = NC_M3 * NV_M3 * math.exp(
        -GAP_EV / thermal_voltage(TEMPERATURE_K)
    )
    assert (
        result.neutrality.electron_density_m3
        * result.neutrality.hole_density_m3
    ) == pytest.approx(ni_sq, rel=2.0e-14)
    capture_scale = (
        distribution.total_density_m3
        * distribution.sigma_n_m2
        * distribution.thermal_velocity_m_s
        * result.neutrality.electron_density_m3
    )
    assert abs(result.trap_state.recombination_rate_m3_s.item()) <= (
        1.0e-13 * capture_scale
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"total_density_m3": 0.0}, "positive"),
        ({"center_eV_above_vb": -0.1}, "non-negative"),
        ({"sigma_n_m2": math.inf}, "positive"),
        ({"charge_transition": "amphoteric"}, "acceptor.*donor"),
        ({"distribution": "uniform"}, "single_level.*gaussian"),
        ({"distribution": "single_level"}, "forbidden"),
    ),
)
def test_distribution_contract_rejects_ambiguous_or_nonphysical_inputs(
    updates,
    message,
):
    with pytest.raises(ValueError, match=message):
        _distribution(**updates)
