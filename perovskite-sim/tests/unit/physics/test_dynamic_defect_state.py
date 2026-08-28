"""D5-E2 dynamic bulk-trap state layout and constitutive closure."""

from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    LAYER_AVERAGE_UNITY,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
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
    MonovalentDefectRegion,
    evaluate_monovalent_bulk_defects,
)
from perovskite_sim.physics.dynamic_defect_state import (
    DynamicBulkTrapStateError,
    compile_dynamic_bulk_trap_layout,
    evaluate_dynamic_bulk_traps,
    evaluate_dynamic_bulk_traps_about_qss,
    occupancy_from_logit_increment,
    occupancy_logit,
    quasi_steady_bulk_trap_occupancy,
)


GAP_EV = 1.2
NC_M3 = 2.0e24
NV_M3 = 1.0e24
TEMPERATURE_K = 300.0


def _species(
    name: str,
    transition: str,
    *,
    distributed: bool = False,
    explicit_energy_reference: bool = False,
    spatial_profile: BulkDefectSpatialProfile | None = None,
) -> BulkDefectSpecies:
    neutral_reference = {
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
    }[transition]
    distribution = (
        BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=3.0e21,
            center_eV_above_vb=0.58,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            width_eV=0.06,
            width_convention=WIDTH_GAUSSIAN_SIGMA,
            support_width_multiplier=5.0,
        )
        if distributed
        else BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=3.0e21,
            center_eV_above_vb=0.58,
            energy_reference=(
                ENERGY_ABOVE_VALENCE_BAND if explicit_energy_reference else None
            ),
        )
    )
    return BulkDefectSpecies(
        name=name,
        distribution=distribution,
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.1e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
        spatial_profile=spatial_profile,
    )


def _model(*, spatial: bool = False) -> MonovalentBulkDefectModel:
    profile = (
        BulkDefectSpatialProfile(
            coordinate=NORMALIZED_LAYER_COORDINATE,
            interpolation=PIECEWISE_LINEAR,
            density_normalization=LAYER_AVERAGE_UNITY,
            knots=(
                BulkDefectSpatialKnot(0.0, 0.5),
                BulkDefectSpatialKnot(1.0, 1.5),
            ),
        )
        if spatial
        else None
    )
    species = (
        _species(
            "acceptor",
            ACCEPTOR,
            explicit_energy_reference=True,
            spatial_profile=profile,
        ),
        _species("donor_gaussian", DONOR, distributed=True, spatial_profile=profile),
        _species(
            "neutral",
            NEUTRAL,
            explicit_energy_reference=True,
            spatial_profile=profile,
        ),
    )
    schema = (
        EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION
        if spatial
        else EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION
    )
    document = BulkDefectDocument(
        schema_version=schema,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=species,
    )
    values: dict[str, object] = {}
    if spatial:
        coordinates = np.array([0.0, 0.5, 1.0])
        values = {
            "normalized_layer_coordinates": coordinates,
            "local_band_gap_eV": np.full(3, GAP_EV),
            "local_effective_conduction_dos_m3": np.full(3, NC_M3),
            "local_effective_valence_dos_m3": np.full(3, NV_M3),
            "source_density_multipliers": np.asarray(
                [
                    [
                        source.spatial_profile.density_multiplier_at(x)
                        for x in coordinates
                    ]
                    for source in species
                ]
            ),
        }
    region = MonovalentDefectRegion(
        identifier="absorber",
        document_sha256=document.sha256,
        active_nodes=np.array([True, True, True, False]),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=species,
        schema_version=schema,
        energy_quadrature_order=7,
        **values,
    )
    return MonovalentBulkDefectModel(regions=(region,))


def _single_model(transition: str = ACCEPTOR) -> MonovalentBulkDefectModel:
    species = (_species("single", transition),)
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=species,
    )
    return MonovalentBulkDefectModel(
        regions=(
            MonovalentDefectRegion(
                identifier="single-region",
                document_sha256=document.sha256,
                active_nodes=np.array([True, True, True]),
                band_gap_eV=GAP_EV,
                effective_conduction_dos_m3=NC_M3,
                effective_valence_dos_m3=NV_M3,
                temperature_K=TEMPERATURE_K,
                species=species,
            ),
        )
    )


@pytest.mark.parametrize("spatial", [False, True])
def test_layout_qss_recovers_compiled_charge_capture_and_occupied_density(spatial):
    model = _model(spatial=spatial)
    layout = compile_dynamic_bulk_trap_layout(model)
    n = np.array([2.0e18, 7.0e20, 4.0e22, 3.0e19])
    p = np.array([8.0e22, 3.0e20, 5.0e17, 2.0e19])
    occupancy = quasi_steady_bulk_trap_occupancy(n, p, layout)
    dynamic = evaluate_dynamic_bulk_traps(n, p, occupancy, layout)
    established = evaluate_monovalent_bulk_defects(n, p, model)

    source_shape = established.recombination_rate_m3_s.shape
    occupied = np.zeros(source_shape)
    electron = np.zeros(source_shape)
    hole = np.zeros(source_shape)
    np.add.at(
        occupied,
        (layout.source_indices, layout.device_node_indices),
        dynamic.occupied_storage_m3,
    )
    np.add.at(
        electron,
        (layout.source_indices, layout.device_node_indices),
        dynamic.electron_capture_rate_m3_s,
    )
    np.add.at(
        hole,
        (layout.source_indices, layout.device_node_indices),
        dynamic.hole_capture_rate_m3_s,
    )

    np.testing.assert_allclose(
        occupied,
        established.occupied_density_m3,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        electron,
        established.recombination_rate_m3_s,
        rtol=2.0e-14,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        hole,
        established.recombination_rate_m3_s,
        rtol=2.0e-14,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        dynamic.total_charge_density_C_m3,
        established.total_charge_density_C_m3,
        rtol=3.0e-15,
        atol=0.0,
    )
    assert np.max(np.abs(dynamic.trap_storage_rate_m3_s)) < 1.0e7
    assert dynamic.maximum_local_charge_balance_relative_error < 1.0e-15


def test_layout_order_identity_mask_and_logit_round_trip_are_explicit():
    model = _model()
    full = compile_dynamic_bulk_trap_layout(model)
    interior = compile_dynamic_bulk_trap_layout(
        model,
        dynamic_node_mask=np.array([False, True, True, False]),
    )

    assert full.size == 3 * (1 + 7 + 1)
    assert interior.size == 2 * (1 + 7 + 1)
    assert full.identity_sha256 != interior.identity_sha256
    assert tuple(interior.device_node_indices[:9]) == (1,) * 9
    assert tuple(interior.source_identifiers[:9]) == (
        "acceptor",
        *("donor_gaussian",) * 7,
        "neutral",
    )
    assert not full.population_density_m3.flags.writeable

    n = np.full(model.node_count, 2.0e20)
    p = np.full(model.node_count, 3.0e20)
    occupancy = quasi_steady_bulk_trap_occupancy(n, p, interior)
    reference = occupancy_logit(occupancy, interior)
    recovered = occupancy_from_logit_increment(
        reference,
        np.zeros(interior.size),
        interior,
    )
    np.testing.assert_allclose(recovered, occupancy, rtol=2.0e-15, atol=0.0)


@pytest.mark.parametrize("transition", [ACCEPTOR, DONOR, NEUTRAL])
def test_non_qss_capture_and_storage_obey_local_population_and_charge_balance(
    transition,
):
    model = _single_model(transition)
    layout = compile_dynamic_bulk_trap_layout(model)
    n = np.array([1.0e18, 2.0e20, 4.0e22])
    p = np.array([5.0e22, 3.0e20, 7.0e17])
    occupancy = np.array([0.15, 0.45, 0.85])
    value = evaluate_dynamic_bulk_traps(n, p, occupancy, layout)

    np.testing.assert_allclose(
        value.trap_storage_rate_m3_s,
        value.electron_capture_rate_m3_s - value.hole_capture_rate_m3_s,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert value.maximum_local_charge_balance_relative_error < 1.0e-15
    if transition == NEUTRAL:
        np.testing.assert_array_equal(value.charge_density_C_m3, 0.0)
    elif transition == ACCEPTOR:
        assert np.all(value.charge_density_C_m3 < 0.0)
    else:
        assert np.all(value.charge_density_C_m3 > 0.0)


def test_layout_and_state_fail_closed_on_ambiguous_or_unphysical_inputs():
    model = _single_model()
    with pytest.raises(DynamicBulkTrapStateError, match="match the compiled"):
        compile_dynamic_bulk_trap_layout(
            model,
            dynamic_node_mask=np.array([True, False]),
        )
    with pytest.raises(DynamicBulkTrapStateError, match="selects no"):
        compile_dynamic_bulk_trap_layout(
            model,
            dynamic_node_mask=np.zeros(model.node_count, dtype=bool),
        )
    layout = compile_dynamic_bulk_trap_layout(model)
    n = np.full(model.node_count, 1.0e20)
    p = np.full(model.node_count, 1.0e20)
    with pytest.raises(DynamicBulkTrapStateError, match=r"\[0, 1\]"):
        evaluate_dynamic_bulk_traps(n, p, np.full(layout.size, 1.01), layout)
    with pytest.raises(DynamicBulkTrapStateError, match="inside"):
        occupancy_logit(np.zeros(layout.size), layout)
    with pytest.raises(DynamicBulkTrapStateError, match="saturated"):
        occupancy_from_logit_increment(
            np.zeros(layout.size),
            np.full(layout.size, 1.0e4),
            layout,
        )


def test_reference_increment_form_is_nonlinearly_equivalent_and_qss_exact():
    model = _single_model()
    layout = compile_dynamic_bulk_trap_layout(model)
    n0 = np.array([1.0e18, 2.0e20, 4.0e22])
    p0 = np.array([5.0e22, 3.0e20, 7.0e17])
    f0 = quasi_steady_bulk_trap_occupancy(n0, p0, layout)
    base = evaluate_dynamic_bulk_traps_about_qss(
        n0,
        p0,
        f0,
        layout,
        reference_electron_density_m3=n0,
        reference_hole_density_m3=p0,
        reference_occupancy=f0,
    )
    np.testing.assert_array_equal(base.trap_storage_rate_m3_s, 0.0)

    n = n0 * np.array([1.02, 0.97, 1.01])
    p = p0 * np.array([0.96, 1.03, 1.04])
    f = occupancy_from_logit_increment(
        occupancy_logit(f0, layout),
        np.array([1.0e-2, -2.0e-2, 3.0e-2]),
        layout,
    )
    direct = evaluate_dynamic_bulk_traps(n, p, f, layout)
    incremental = evaluate_dynamic_bulk_traps_about_qss(
        n,
        p,
        f,
        layout,
        reference_electron_density_m3=n0,
        reference_hole_density_m3=p0,
        reference_occupancy=f0,
    )
    for name in (
        "electron_capture_rate_m3_s",
        "hole_capture_rate_m3_s",
        "trap_storage_rate_m3_s",
        "total_charge_density_C_m3",
    ):
        np.testing.assert_allclose(
            getattr(incremental, name),
            getattr(direct, name),
            rtol=2.0e-10,
            atol=0.0,
        )
    with pytest.raises(DynamicBulkTrapStateError, match="exact QSS"):
        evaluate_dynamic_bulk_traps_about_qss(
            n,
            p,
            f,
            layout,
            reference_electron_density_m3=n0,
            reference_hole_density_m3=p0,
            reference_occupancy=f0 + 1.0e-6,
        )
