from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    ENERGY_ABOVE_VALENCE_BAND,
    INTEGRATED_TOTAL,
    LAYER_AVERAGE_UNITY,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    NORMALIZED_LAYER_COORDINATE,
    PIECEWISE_LINEAR,
    SINGLE_LEVEL,
    UNIFORM,
    WIDTH_UNIFORM_FULL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpatialKnot,
    BulkDefectSpatialProfile,
    BulkDefectSpecies,
    bulk_defect_species_at_layer_position,
)
from perovskite_sim.models.interface_defects import (
    InterfaceDefectDocument,
    InterfaceDefectKinetics,
)
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_defect_closure,
    evaluate_monovalent_source_defect_closure,
)
from perovskite_sim.physics.trap_population_response import (
    INTERFACE_CAPTURE_FLUX_ORDER,
    INTERFACE_RESERVOIR_ORDER,
    LOCAL_POPULATION_BINDING_SCOPE,
    TrapPopulationCertificationError,
    TrapPopulationResponseError,
    solve_bulk_defect_population_frequency_response,
    solve_two_sided_interface_trap_population_frequency_response,
)
from perovskite_sim.physics.temperature import thermal_voltage
from perovskite_sim.physics.two_sided_interface import (
    TwoSidedInterfacePhysics,
    shared_trap_capture_flux,
)


GAP_EV = 1.5
NC_M3 = 2.4e25
NV_M3 = 1.1e25
TEMPERATURE_K = 300.0
N_M3 = 4.0e19
P_M3 = 7.0e18
FREQUENCIES_HZ = np.array([1.0e-5, 1.0, 1.0e5])
DN_M3_V = np.array([1.0e19, 2.0e19, 3.0e19])
DP_M3_V = np.array([-2.0e18, -3.0e18, -4.0e18])


def _bulk_species(
    *,
    transition: str = ACCEPTOR,
    distribution_kind: str = SINGLE_LEVEL,
    spatial_profile: BulkDefectSpatialProfile | None = None,
) -> BulkDefectSpecies:
    distribution_kwargs: dict[str, object] = {
        "kind": distribution_kind,
        "normalization": INTEGRATED_TOTAL,
        "total_density_m3": 3.0e21,
        "center_eV_above_vb": 0.62,
    }
    if distribution_kind == UNIFORM:
        distribution_kwargs |= {
            "energy_reference": ENERGY_ABOVE_VALENCE_BAND,
            "width_eV": 0.24,
            "width_convention": WIDTH_UNIFORM_FULL,
        }
    neutral_reference = {
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
    }[transition]
    return BulkDefectSpecies(
        name="bulk-source",
        distribution=BulkDefectDistribution(**distribution_kwargs),
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.3e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        spatial_profile=spatial_profile,
    )


def _solve_bulk(species: BulkDefectSpecies, **updates):
    values = {
        "source_species": species,
        "electron_density_m3": N_M3,
        "hole_density_m3": P_M3,
        "frequencies_Hz": FREQUENCIES_HZ,
        "electron_density_response_m3_per_V": DN_M3_V,
        "hole_density_response_m3_per_V": DP_M3_V,
        "band_gap_eV": GAP_EV,
        "effective_conduction_dos_m3": NC_M3,
        "effective_valence_dos_m3": NV_M3,
        "temperature_K": TEMPERATURE_K,
    }
    values.update(updates)
    return solve_bulk_defect_population_frequency_response(**values)


def _interface_document(**kinetics_updates: float) -> InterfaceDefectDocument:
    document = InterfaceDefectDocument.from_scaps_cgs(
        sigma_n_cm2=3.0e-20,
        sigma_p_cm2=5.0e-20,
        thermal_velocity_cm_s=2.0e7,
        total_density_cm2=4.0e12,
        trap_depth_eV_below_cb=0.55,
    )
    if kinetics_updates:
        kinetics = replace(document.kinetics, **kinetics_updates)
        document = replace(document, kinetics=kinetics)
    return document


def _interface_physics(
    document: InterfaceDefectDocument,
    **updates: float,
) -> TwoSidedInterfacePhysics:
    values = {
        "thermal_voltage_V": 0.02585,
        "temperature_K": 300.0,
        "D_n_left_m2_s": 1.0e-4,
        "D_n_right_m2_s": 1.0e-4,
        "D_p_left_m2_s": 1.0e-4,
        "D_p_right_m2_s": 1.0e-4,
        "N_C_left_m3": 1.0e25,
        "N_C_right_m3": 1.0e25,
        "N_V_left_m3": 1.0e25,
        "N_V_right_m3": 1.0e25,
        "richardson_n_A_m2_K2": 1.0e6,
        "richardson_p_A_m2_K2": 1.0e6,
        "surface_recombination_velocity_n_m_s": (document.capture_velocity_n_m_s),
        "surface_recombination_velocity_p_m_s": (document.capture_velocity_p_m_s),
        "n1_left_m3": 2.0e17,
        "n1_right_m3": 5.0e17,
        "p1_left_m3": 8.0e16,
        "p1_right_m3": 3.0e17,
    }
    values.update(updates)
    return TwoSidedInterfacePhysics(**values)


INTERFACE_STATE_M3 = np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20])
INTERFACE_DN_M3_V = np.array([1.0e19, 2.0e19])
INTERFACE_DP_M3_V = np.array([-3.0e18, -4.0e18])


def _solve_interface(
    document: InterfaceDefectDocument | None = None,
    physics: TwoSidedInterfacePhysics | None = None,
    **updates,
):
    resolved_document = document or _interface_document()
    resolved_physics = physics or _interface_physics(resolved_document)
    values = {
        "document": resolved_document,
        "physics": resolved_physics,
        "state_m3": INTERFACE_STATE_M3,
        "frequencies_Hz": FREQUENCIES_HZ,
        "electron_density_response_m3_per_V": INTERFACE_DN_M3_V,
        "hole_density_response_m3_per_V": INTERFACE_DP_M3_V,
    }
    values.update(updates)
    return solve_two_sided_interface_trap_population_frequency_response(**values)


def test_single_level_bulk_population_matches_existing_dc_and_static_tangent():
    species = _bulk_species()
    result = _solve_bulk(species)
    closure = evaluate_monovalent_defect_closure(
        N_M3,
        P_M3,
        (species,),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )
    point = result.operating_points[0]
    population = species.distribution.total_density_m3
    expected_static = (
        closure.total_recombination_derivative_n_s1 * DN_M3_V
        + closure.total_recombination_derivative_p_s1 * DP_M3_V
    )

    assert result.population_binding_certified
    assert result.certification_scope == LOCAL_POPULATION_BINDING_SCOPE
    assert result.quadrature.order == 1
    assert point.occupancy == pytest.approx(closure.occupancy[0].item())
    assert point.relaxation_rate_s1 == pytest.approx(
        closure.kinetic_denominator_s1[0].item()
    )
    assert population * point.electron_capture_rates_s1[0] == pytest.approx(
        closure.recombination_rate_m3_s[0].item()
    )
    assert population * point.hole_capture_rates_s1[0] == pytest.approx(
        closure.recombination_rate_m3_s[0].item()
    )
    np.testing.assert_allclose(
        result.total_quasistatic_recombination_response_m3_s_V,
        expected_static,
        rtol=3.0e-15,
    )
    assert result.maximum_dc_closure_relative_error < 1.0e-14
    assert result.maximum_quasistatic_tangent_relative_error < 1.0e-14
    assert result.maximum_local_balance_relative_error < 1.0e-14


def test_bulk_charge_response_uses_same_differential_sign_for_acceptor_and_donor():
    acceptor = _solve_bulk(_bulk_species(transition=ACCEPTOR))
    donor = _solve_bulk(_bulk_species(transition=DONOR))
    neutral = _solve_bulk(_bulk_species(transition=NEUTRAL))

    np.testing.assert_array_equal(
        acceptor.total_charge_density_response_C_m3_V,
        donor.total_charge_density_response_C_m3_V,
    )
    np.testing.assert_array_equal(
        neutral.total_charge_density_response_C_m3_V,
        np.zeros(FREQUENCIES_HZ.size, dtype=complex),
    )


def test_distributed_bulk_population_preserves_quadrature_and_integrated_tangent():
    species = _bulk_species(distribution_kind=UNIFORM)
    result = _solve_bulk(species, energy_quadrature_order=12)
    closure = evaluate_monovalent_source_defect_closure(
        N_M3,
        P_M3,
        (species,),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        energy_quadrature_order=12,
    )
    expected = (
        closure.total_recombination_derivative_n_s1 * DN_M3_V
        + closure.total_recombination_derivative_p_s1 * DP_M3_V
    )

    assert result.population_binding_certified
    assert result.quadrature.order == 12
    assert result.quadrature.integrated_density_m3 == (
        species.distribution.total_density_m3
    )
    assert len(result.node_responses) == 12
    np.testing.assert_allclose(
        result.total_quasistatic_recombination_response_m3_s_V,
        expected,
        rtol=4.0e-15,
    )


def test_spatial_bulk_source_must_be_localized_before_frequency_response():
    profile = BulkDefectSpatialProfile(
        coordinate=NORMALIZED_LAYER_COORDINATE,
        interpolation=PIECEWISE_LINEAR,
        density_normalization=LAYER_AVERAGE_UNITY,
        knots=(
            BulkDefectSpatialKnot(0.0, 0.5),
            BulkDefectSpatialKnot(1.0, 1.5),
        ),
    )
    source = _bulk_species(spatial_profile=profile)
    with pytest.raises(TrapPopulationResponseError, match="position-resolved"):
        _solve_bulk(source)

    local = bulk_defect_species_at_layer_position((source,), 0.25)[0]
    result = _solve_bulk(local)
    assert result.population_binding_certified
    assert result.quadrature.total_density_m3 == pytest.approx(0.75 * 3.0e21)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"frequencies_Hz": [1.0, 1.0]}, "strictly increasing"),
        (
            {"electron_density_response_m3_per_V": np.ones((3, 2))},
            "frequency-by-reservoir",
        ),
        ({"electron_density_m3": -1.0}, "non-negative"),
    ],
)
def test_bulk_population_rejects_ambiguous_local_inputs(updates, message):
    with pytest.raises(TrapPopulationResponseError, match=message):
        _solve_bulk(_bulk_species(), **updates)


def test_bulk_population_certificate_fails_closed_without_hiding_result():
    species = _bulk_species()
    with pytest.raises(TrapPopulationCertificationError) as caught:
        _solve_bulk(species, max_crosscheck_relative_error=1.0e-20)
    assert not caught.value.result.population_binding_certified
    observed = _solve_bulk(
        species,
        max_crosscheck_relative_error=1.0e-20,
        require_certified=False,
    )
    assert not observed.population_binding_certified


def test_bulk_population_crosscheck_is_stable_at_zero_net_recombination():
    intrinsic_product = NC_M3 * NV_M3 * np.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    result = _solve_bulk(
        _bulk_species(),
        hole_density_m3=intrinsic_product / N_M3,
    )
    assert result.population_binding_certified
    assert result.maximum_dc_closure_relative_error < 1.0e-14


def test_population_response_arrays_are_immutable():
    result = _solve_bulk(_bulk_species())
    with pytest.raises(ValueError):
        result.total_charge_density_response_C_m3_V[0] = 0.0
    with pytest.raises(ValueError):
        result.electron_capture_response_m3_s_V[0, 0] = 0.0


def test_interface_population_matches_existing_capture_flux_with_explicit_order():
    document = _interface_document()
    physics = _interface_physics(document)
    result = _solve_interface(document, physics)
    population = document.total_density_m2
    expected_population_flux = np.array(
        [
            population * result.operating_point.electron_capture_rates_s1[0],
            population * result.operating_point.hole_capture_rates_s1[0],
            population * result.operating_point.electron_capture_rates_s1[1],
            population * result.operating_point.hole_capture_rates_s1[1],
        ]
    )

    assert result.population_binding_certified
    assert result.certification_scope == LOCAL_POPULATION_BINDING_SCOPE
    assert result.document_sha256 == document.sha256
    assert result.reservoir_order == INTERFACE_RESERVOIR_ORDER
    assert result.capture_flux_order == INTERFACE_CAPTURE_FLUX_ORDER
    np.testing.assert_array_equal(
        result.dc_capture_flux_m2_s,
        shared_trap_capture_flux(INTERFACE_STATE_M3, physics),
    )
    np.testing.assert_allclose(
        result.dc_capture_flux_m2_s,
        expected_population_flux,
        rtol=3.0e-15,
    )
    np.testing.assert_array_equal(
        result.electron_capture_flux_response_m2_s_V,
        population * result.response.electron_capture_response_s1_per_V,
    )
    np.testing.assert_array_equal(
        result.hole_capture_flux_response_m2_s_V,
        population * result.response.hole_capture_response_s1_per_V,
    )


def test_interface_sheet_charge_and_capture_obey_local_charge_conservation():
    document = _interface_document()
    result = _solve_interface(document)
    expected_charge = (
        -Q * document.total_density_m2 * result.response.occupancy_response_per_V
    )
    omega = 2.0 * np.pi * result.frequencies_Hz
    residual = (
        Q
        * (
            np.sum(result.electron_capture_flux_response_m2_s_V, axis=1)
            - np.sum(result.hole_capture_flux_response_m2_s_V, axis=1)
        )
        + 1j * omega * result.sheet_charge_response_C_m2_V
    )

    np.testing.assert_allclose(
        result.sheet_charge_response_C_m2_V,
        expected_charge,
        rtol=3.0e-16,
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-12)
    assert result.maximum_local_charge_conservation_relative_error < 1.0e-14


def test_interface_binding_hash_covers_reference_densities():
    document = _interface_document()
    first = _solve_interface(document, _interface_physics(document))
    second = _solve_interface(
        document,
        _interface_physics(document, n1_left_m3=2.1e17),
    )
    assert first.trap_binding_sha256 != second.trap_binding_sha256


def test_interface_rejects_surface_velocity_drift_from_microscopic_document():
    document = _interface_document()
    physics = _interface_physics(
        document,
        surface_recombination_velocity_n_m_s=(
            np.nextafter(document.capture_velocity_n_m_s, np.inf)
        ),
    )
    with pytest.raises(TrapPopulationResponseError, match=r"sigma\*v_th\*N_t"):
        _solve_interface(document, physics)


def test_interface_rejects_unsupported_degeneracy_and_capture_off_limit():
    document = _interface_document()
    with pytest.raises(TrapPopulationResponseError, match="degeneracy=1.0"):
        _solve_interface(replace(document, degeneracy=2.0))

    off = replace(
        document,
        kinetics=InterfaceDefectKinetics(
            sigma_n_m2=0.0,
            sigma_p_m2=0.0,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=1.0e5,
        ),
    )
    with pytest.raises(TrapPopulationResponseError, match="both capture legs"):
        _solve_interface(off, _interface_physics(off))


@pytest.mark.parametrize("zero_field", ["sigma_n_m2", "sigma_p_m2"])
def test_interface_population_supports_one_capture_leg_off(zero_field):
    document = _interface_document(**{zero_field: 0.0})
    result = _solve_interface(document, _interface_physics(document))
    assert result.population_binding_certified
    assert result.maximum_dc_closure_relative_error < 1.0e-14
    assert result.maximum_local_charge_conservation_relative_error < 1.0e-14


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"state_m3": np.ones(3)}, "n_left"),
        (
            {"electron_density_response_m3_per_V": np.ones((3, 3))},
            "frequency-by-reservoir",
        ),
    ],
)
def test_interface_population_rejects_order_or_shape_ambiguity(updates, message):
    with pytest.raises(TrapPopulationResponseError, match=message):
        _solve_interface(**updates)


def test_interface_population_certificate_fails_closed_and_arrays_are_immutable():
    with pytest.raises(TrapPopulationCertificationError) as caught:
        _solve_interface(max_crosscheck_relative_error=1.0e-20)
    result = caught.value.result
    assert not result.population_binding_certified
    with pytest.raises(ValueError):
        result.sheet_charge_response_C_m2_V[0] = 0.0
    with pytest.raises(ValueError):
        result.dc_capture_flux_m2_s[0] = 0.0
