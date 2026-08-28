"""D3-E1/E2 independent energy-order refinement evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    ENERGY_ABOVE_VALENCE_BAND,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.validation.defect_energy_refinement import (
    DEFECT_ENERGY_REFINEMENT_VERSION,
    DefectEnergyRefinementComparison,
    assess_defect_energy_order_refinement,
)


GAP_EV = 1.5
NC_M3 = 2.4e25
NV_M3 = 1.1e25
TEMPERATURE_K = 300.0


def _species(name: str, kind: str, transition: str) -> BulkDefectSpecies:
    values: dict[str, object] = {
        "kind": kind,
        "normalization": INTEGRATED_TOTAL,
        "total_density_m3": 3.0e21,
        "center_eV_above_vb": 0.72,
        "energy_reference": ENERGY_ABOVE_VALENCE_BAND,
    }
    if kind == GAUSSIAN:
        values |= {
            "width_eV": 0.08,
            "width_convention": WIDTH_GAUSSIAN_SIGMA,
            "support_width_multiplier": 8.0,
        }
    elif kind == UNIFORM:
        values |= {
            "center_eV_above_vb": 0.82,
            "width_eV": 0.4,
            "width_convention": WIDTH_UNIFORM_FULL,
        }
    elif kind == CONDUCTION_BAND_TAIL:
        values |= {
            "center_eV_above_vb": 1.45,
            "width_eV": 0.1,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 7.0,
        }
    elif kind == VALENCE_BAND_TAIL:
        values |= {
            "center_eV_above_vb": 0.05,
            "width_eV": 0.1,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 7.0,
        }
    neutral_reference = (
        NEUTRAL_WHEN_EMPTY
        if transition == ACCEPTOR
        else NEUTRAL_WHEN_FILLED
    )
    return BulkDefectSpecies(
        name=name,
        distribution=BulkDefectDistribution(**values),
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.3e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
    )


def _assess(*species, orders=(8, 16, 32), threshold=5.0e-3):
    return assess_defect_energy_order_refinement(
        np.asarray([2.0e17, 6.0e20, 4.0e22]),
        np.asarray([8.0e21, 2.0e19, 7.0e16]),
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        energy_orders=orders,
        threshold=threshold,
    )


def test_report_passes_and_contains_only_the_energy_refinement_axis():
    report = _assess(
        _species("gaussian", GAUSSIAN, ACCEPTOR),
        _species("uniform", UNIFORM, DONOR),
    )
    payload = report.to_dict()

    assert report.passed
    assert report.energy_orders == (8, 16, 32)
    assert report.source_identifiers == ("gaussian", "uniform")
    assert report.distribution_kinds == (GAUSSIAN, UNIFORM)
    assert len(set(report.closure_identity_sha256)) == 3
    assert payload["refinement"] == DEFECT_ENERGY_REFINEMENT_VERSION
    assert payload["passed"] is True
    assert "spatial_grid" not in payload
    assert "solver_tolerance" not in payload
    for comparison in report.comparisons:
        assert comparison.passed
        assert comparison.maximum_source_occupancy_absolute_change < 5.0e-3
        assert comparison.maximum_source_charge_normalized_change < 5.0e-3
        assert comparison.maximum_source_recombination_relative_change < 5.0e-3
        assert comparison.maximum_source_tangent_relative_change < 5.0e-3


def test_report_identity_binds_state_species_orders_and_threshold():
    gaussian = _species("gaussian", GAUSSIAN, ACCEPTOR)
    base = _assess(gaussian)
    changed_species = _assess(
        replace(
            gaussian,
            distribution=replace(
                gaussian.distribution,
                total_density_m3=4.0e21,
            ),
        )
    )
    changed_orders = _assess(gaussian, orders=(16, 32, 64))
    changed_threshold = _assess(gaussian, threshold=1.0e-2)
    changed_state = assess_defect_energy_order_refinement(
        np.asarray([3.0e17, 6.0e20, 4.0e22]),
        np.asarray([8.0e21, 2.0e19, 7.0e16]),
        (gaussian,),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    assert len(
        {
            base.input_identity_sha256,
            changed_species.input_identity_sha256,
            changed_orders.input_identity_sha256,
            changed_threshold.input_identity_sha256,
            changed_state.input_identity_sha256,
        }
    ) == 5


def test_tight_threshold_reports_failure_without_changing_the_closure():
    gaussian = _species("gaussian", GAUSSIAN, ACCEPTOR)
    normal = _assess(gaussian)
    strict = _assess(gaussian, threshold=1.0e-18)

    assert normal.passed
    assert not strict.passed
    assert normal.closure_identity_sha256 == strict.closure_identity_sha256
    assert any(not item.passed for item in strict.comparisons)


def test_input_identity_uses_the_resolved_broadcast_carrier_state():
    gaussian = _species("gaussian", GAUSSIAN, ACCEPTOR)
    common = {
        "band_gap_eV": GAP_EV,
        "effective_conduction_dos_m3": NC_M3,
        "effective_valence_dos_m3": NV_M3,
        "temperature_K": TEMPERATURE_K,
    }
    scalar = assess_defect_energy_order_refinement(
        2.0e20,
        np.asarray([3.0e19, 7.0e19]),
        (gaussian,),
        **common,
    )
    expanded = assess_defect_energy_order_refinement(
        np.asarray([2.0e20, 2.0e20]),
        np.asarray([3.0e19, 7.0e19]),
        (gaussian,),
        **common,
    )

    assert scalar.input_identity_sha256 == expanded.input_identity_sha256


@pytest.mark.parametrize(
    "orders",
    ((8,), (8, 24), (8, 16, 16), (1, 2), (8.0, 16)),
)
def test_energy_order_ladder_is_strict(orders):
    with pytest.raises(ValueError, match="energy refinement orders"):
        _assess(
            _species("gaussian", GAUSSIAN, ACCEPTOR),
            orders=orders,
        )


def test_single_only_input_cannot_claim_energy_refinement():
    with pytest.raises(ValueError, match="distributed species"):
        _assess(_species("single", SINGLE_LEVEL, ACCEPTOR))


@pytest.mark.parametrize(
    "kind",
    (CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_tail_input_has_independent_energy_order_evidence(kind):
    report = _assess(
        _species("tail", kind, ACCEPTOR),
        orders=(16, 32, 64),
    )

    assert report.passed
    assert report.source_identifiers == ("tail",)
    assert report.distribution_kinds == (kind,)
    assert all(item.passed for item in report.comparisons)


def test_combined_tail_refinement_fixture_freezes_documented_evidence():
    report = _assess(
        _species("cb_tail", CONDUCTION_BAND_TAIL, ACCEPTOR),
        _species("vb_tail", VALENCE_BAND_TAIL, DONOR),
        orders=(16, 32, 64),
    )
    terminal = report.comparisons[-1]

    assert report.input_identity_sha256 == (
        "9a5529321d8dc0848fd27214904b2bf6efb780405d559bb4a191fc432fa4614a"
    )
    assert terminal.maximum_source_occupancy_absolute_change == pytest.approx(
        4.62427221514794e-09,
        rel=2.0e-14,
    )
    assert terminal.maximum_source_charge_normalized_change == pytest.approx(
        4.624272260686276e-09,
        rel=2.0e-14,
    )
    assert terminal.maximum_source_recombination_relative_change == (
        pytest.approx(3.92845990776778e-08, rel=2.0e-14)
    )
    assert terminal.maximum_source_tangent_relative_change == pytest.approx(
        2.541153515274365e-07,
        rel=2.0e-14,
    )


def test_comparison_pass_flag_is_fail_closed():
    with pytest.raises(ValueError, match="pass flag"):
        DefectEnergyRefinementComparison(
            coarse_order=8,
            fine_order=16,
            maximum_source_occupancy_absolute_change=1.0e-4,
            maximum_source_charge_normalized_change=1.0e-4,
            maximum_source_recombination_relative_change=1.0e-4,
            maximum_source_tangent_relative_change=1.0e-4,
            threshold=5.0e-3,
            passed=False,
        )
