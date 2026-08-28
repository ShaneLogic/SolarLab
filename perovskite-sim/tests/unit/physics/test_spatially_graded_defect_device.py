"""D3-E4b spatially graded explicit-defect constitutive closure tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.models.defects import (
    ACCEPTOR,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    LAYER_AVERAGE_UNITY,
    NEUTRAL_WHEN_EMPTY,
    NORMALIZED_LAYER_COORDINATE,
    PIECEWISE_LINEAR,
    SINGLE_LEVEL,
    WIDTH_GAUSSIAN_SIGMA,
    BulkDefectDistribution,
    BulkDefectDocument,
    BulkDefectKinetics,
    BulkDefectSpatialKnot,
    BulkDefectSpatialProfile,
    BulkDefectSpecies,
)
from perovskite_sim.physics.defect_closure import (
    MonovalentBulkDefectModel,
    MonovalentDefectClosureCapabilityError,
    MonovalentDefectRegion,
    evaluate_monovalent_bulk_defects,
    evaluate_monovalent_defect_closure,
)
from perovskite_sim.physics.temperature import thermal_voltage


GAP_EV = 1.2
NC_M3 = 2.0e24
NV_M3 = 1.5e24
TEMPERATURE_K = 300.0
COORDINATES = np.asarray([0.0, 0.5, 1.0])
ACTIVE = np.asarray([True, True, True])


def _profile(
    front: float = 0.5,
    middle: float = 1.0,
    back: float = 1.5,
) -> BulkDefectSpatialProfile:
    return BulkDefectSpatialProfile(
        coordinate=NORMALIZED_LAYER_COORDINATE,
        interpolation=PIECEWISE_LINEAR,
        density_normalization=LAYER_AVERAGE_UNITY,
        knots=tuple(
            BulkDefectSpatialKnot(position, multiplier)
            for position, multiplier in zip(
                (0.0, 0.5, 1.0),
                (front, middle, back),
                strict=True,
            )
        ),
    )


def _species(
    profile: BulkDefectSpatialProfile | None,
) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name="graded_acceptor",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=4.0e21,
            center_eV_above_vb=0.55,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
        charge_transition=ACCEPTOR,
        neutral_reference=NEUTRAL_WHEN_EMPTY,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
        spatial_profile=profile,
    )


def _gaussian_species(
    profile: BulkDefectSpatialProfile | None,
) -> BulkDefectSpecies:
    return replace(
        _species(profile),
        distribution=BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=4.0e21,
            center_eV_above_vb=0.55,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            width_eV=0.08,
            width_convention=WIDTH_GAUSSIAN_SIGMA,
            support_width_multiplier=6.0,
        ),
    )


def _region(
    profile: BulkDefectSpatialProfile,
    *,
    local_gap: np.ndarray | None = None,
    source: BulkDefectSpecies | None = None,
) -> MonovalentDefectRegion:
    source = _species(profile) if source is None else source
    gap = (
        np.full(3, GAP_EV)
        if local_gap is None
        else np.asarray(local_gap, dtype=float)
    )
    multipliers = np.asarray(
        [[profile.density_multiplier_at(value) for value in COORDINATES]]
    )
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(source,),
    )
    return MonovalentDefectRegion(
        identifier="absorber",
        document_sha256=document.sha256,
        active_nodes=ACTIVE,
        band_gap_eV=float(gap[0]),
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=(source,),
        schema_version=EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
        energy_quadrature_order=16,
        normalized_layer_coordinates=COORDINATES,
        local_band_gap_eV=gap,
        local_effective_conduction_dos_m3=np.full(3, NC_M3),
        local_effective_valence_dos_m3=np.full(3, NV_M3),
        source_density_multipliers=multipliers,
    )


def _evaluate(region: MonovalentDefectRegion):
    n = np.asarray([2.0e19, 4.0e20, 8.0e21])
    p = np.asarray([7.0e21, 3.0e20, 5.0e19])
    model = MonovalentBulkDefectModel(regions=(region,))
    return n, p, model, evaluate_monovalent_bulk_defects(n, p, model)


def test_uniform_v3_profile_recovers_the_v2_single_level_closure_exactly():
    spatial_source = _species(_profile(1.0, 1.0, 1.0))
    uniform_source = replace(spatial_source, spatial_profile=None)
    spatial_region = _region(_profile(1.0, 1.0, 1.0))
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(uniform_source,),
    )
    uniform_region = MonovalentDefectRegion(
        identifier="absorber",
        document_sha256=document.sha256,
        active_nodes=ACTIVE,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=(uniform_source,),
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        energy_quadrature_order=16,
    )
    n, p, _, spatial = _evaluate(spatial_region)
    uniform = evaluate_monovalent_bulk_defects(
        n,
        p,
        MonovalentBulkDefectModel(regions=(uniform_region,)),
    )

    for field in (
        "kinetic_denominator_s1",
        "occupancy",
        "occupied_density_m3",
        "charge_density_C_m3",
        "recombination_rate_m3_s",
        "recombination_derivative_n_s1",
        "recombination_derivative_p_s1",
        "charge_derivative_fixed_qf_C_m3_V",
    ):
        np.testing.assert_array_equal(
            getattr(spatial, field),
            getattr(uniform, field),
        )


def test_uniform_v3_profile_recovers_v2_gaussian_source_closure():
    profile = _profile(1.0, 1.0, 1.0)
    spatial_source = _gaussian_species(profile)
    uniform_source = replace(spatial_source, spatial_profile=None)
    spatial_region = _region(profile, source=spatial_source)
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(uniform_source,),
    )
    uniform_region = MonovalentDefectRegion(
        identifier="absorber",
        document_sha256=document.sha256,
        active_nodes=ACTIVE,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=(uniform_source,),
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        energy_quadrature_order=16,
    )
    n, p, _, spatial = _evaluate(spatial_region)
    uniform = evaluate_monovalent_bulk_defects(
        n,
        p,
        MonovalentBulkDefectModel(regions=(uniform_region,)),
    )

    for field in (
        "kinetic_denominator_s1",
        "occupancy",
        "occupied_density_m3",
        "charge_density_C_m3",
        "recombination_rate_m3_s",
        "recombination_derivative_n_s1",
        "recombination_derivative_p_s1",
        "charge_derivative_fixed_qf_C_m3_V",
    ):
        np.testing.assert_allclose(
            getattr(spatial, field),
            getattr(uniform, field),
            rtol=2.0e-15,
            atol=0.0,
        )


def test_density_profile_scales_sources_but_not_occupancy_at_fixed_local_bands():
    n, p, _, graded = _evaluate(_region(_profile()))
    uniform = evaluate_monovalent_defect_closure(
        n,
        p,
        (_species(None),),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )
    multipliers = np.asarray([0.5, 1.0, 1.5])

    np.testing.assert_array_equal(graded.occupancy[0], uniform.occupancy[0])
    for graded_field, local_field in (
        ("occupied_density_m3", "occupied_density_m3"),
        ("charge_density_C_m3", "charge_density_C_m3"),
        ("recombination_rate_m3_s", "recombination_rate_m3_s"),
        ("recombination_derivative_n_s1", "recombination_derivative_n_s1"),
        ("recombination_derivative_p_s1", "recombination_derivative_p_s1"),
        (
            "charge_derivative_fixed_qf_C_m3_V",
            "charge_derivative_fixed_qf_C_m3_V",
        ),
    ):
        np.testing.assert_allclose(
            getattr(graded, graded_field)[0],
            getattr(uniform, local_field)[0] * multipliers,
            rtol=2.0e-15,
            atol=0.0,
        )


def test_spatial_closure_analytic_carrier_and_fixed_qf_tangents():
    profile = _profile()
    n, p, model, baseline = _evaluate(
        _region(
            profile,
            local_gap=np.asarray([1.2, 1.1, 1.0]),
            source=_gaussian_species(profile),
        )
    )
    relative_step = 1.0e-6
    n_plus = evaluate_monovalent_bulk_defects(
        n * (1.0 + relative_step), p, model
    )
    n_minus = evaluate_monovalent_bulk_defects(
        n * (1.0 - relative_step), p, model
    )
    finite_n = (
        n_plus.total_recombination_rate_m3_s
        - n_minus.total_recombination_rate_m3_s
    ) / (2.0 * relative_step * n)
    p_plus = evaluate_monovalent_bulk_defects(
        n, p * (1.0 + relative_step), model
    )
    p_minus = evaluate_monovalent_bulk_defects(
        n, p * (1.0 - relative_step), model
    )
    finite_p = (
        p_plus.total_recombination_rate_m3_s
        - p_minus.total_recombination_rate_m3_s
    ) / (2.0 * relative_step * p)
    np.testing.assert_allclose(
        baseline.total_recombination_derivative_n_s1,
        finite_n,
        rtol=3.0e-7,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        baseline.total_recombination_derivative_p_s1,
        finite_p,
        rtol=3.0e-7,
        atol=1.0e-10,
    )

    voltage_step = 1.0e-7
    thermal = thermal_voltage(TEMPERATURE_K)
    charge_plus = evaluate_monovalent_bulk_defects(
        n * np.exp(voltage_step / thermal),
        p * np.exp(-voltage_step / thermal),
        model,
    ).total_charge_density_C_m3
    charge_minus = evaluate_monovalent_bulk_defects(
        n * np.exp(-voltage_step / thermal),
        p * np.exp(voltage_step / thermal),
        model,
    ).total_charge_density_C_m3
    finite_fixed_qf = (charge_plus - charge_minus) / (2.0 * voltage_step)
    np.testing.assert_allclose(
        baseline.total_charge_derivative_fixed_qf_C_m3_V,
        finite_fixed_qf,
        rtol=2.0e-8,
        atol=1.0e-5,
    )


def test_spatial_model_identity_and_diagnostics_bind_profile_evidence():
    _, _, forward_model, forward = _evaluate(_region(_profile()))
    _, _, reverse_model, _ = _evaluate(_region(_profile(1.5, 1.0, 0.5)))

    assert forward_model.identity_sha256 != reverse_model.identity_sha256
    assert forward.spatial_profile_sha256s == (
        forward_model.spatial_profile_sha256s
    )
    assert forward.minimum_density_multipliers == (0.5,)
    assert forward.maximum_density_multipliers == (1.5,)
    np.testing.assert_array_equal(
        forward.source_density_multiplier[0],
        np.asarray([0.5, 1.0, 1.5]),
    )
    payload = forward.to_dict()
    assert payload["spatial_closure"] == "layer-density-profile-v1"
    assert payload["minimum_density_multipliers"] == [0.5]
    assert payload["maximum_density_multipliers"] == [1.5]


def test_uncompiled_local_closure_rejects_spatial_profile():
    with pytest.raises(
        MonovalentDefectClosureCapabilityError,
        match="compiled v3",
    ):
        evaluate_monovalent_defect_closure(
            1.0e20,
            1.0e20,
            (_species(_profile()),),
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            temperature_K=TEMPERATURE_K,
        )
