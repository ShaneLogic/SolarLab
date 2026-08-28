"""D3-E0 normalized explicit-defect energy-distribution contracts."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    ENERGY_ABOVE_VALENCE_BAND,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    NEUTRAL_WHEN_EMPTY,
    SINGLE_LEVEL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
    ExplicitDefectSchemaError,
)
from perovskite_sim.physics.defect_distributions import (
    build_defect_energy_quadrature,
    distribution_shape_integral_eV,
    expand_bulk_defect_species_energy,
    integrated_density_from_peak_density,
    peak_density_from_integrated_density,
)


GAP_EV = 1.5
TOTAL_DENSITY_M3 = 2.5e22


def _distribution(kind: str, **updates: object) -> BulkDefectDistribution:
    common: dict[str, object] = {
        "kind": kind,
        "normalization": INTEGRATED_TOTAL,
        "total_density_m3": TOTAL_DENSITY_M3,
        "center_eV_above_vb": 0.75,
        "energy_reference": ENERGY_ABOVE_VALENCE_BAND,
    }
    if kind == GAUSSIAN:
        common |= {
            "width_eV": 0.08,
            "width_convention": WIDTH_GAUSSIAN_SIGMA,
            "support_width_multiplier": 8.0,
        }
    elif kind == UNIFORM:
        common |= {
            "width_eV": 0.4,
            "width_convention": WIDTH_UNIFORM_FULL,
        }
    elif kind == CONDUCTION_BAND_TAIL:
        common |= {
            "center_eV_above_vb": 1.45,
            "width_eV": 0.1,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 7.0,
        }
    elif kind == VALENCE_BAND_TAIL:
        common |= {
            "center_eV_above_vb": 0.05,
            "width_eV": 0.1,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 7.0,
        }
    common.update(updates)
    return BulkDefectDistribution(**common)


def _species(distribution: BulkDefectDistribution) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name="acceptor_distribution",
        distribution=distribution,
        charge_transition=ACCEPTOR,
        neutral_reference=NEUTRAL_WHEN_EMPTY,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=1.0e-19,
            sigma_p_m2=2.0e-19,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=2.0e5,
        ),
    )


@pytest.mark.parametrize(
    "kind",
    (
        SINGLE_LEVEL,
        GAUSSIAN,
        UNIFORM,
        CONDUCTION_BAND_TAIL,
        VALENCE_BAND_TAIL,
    ),
)
def test_all_five_distributions_recover_declared_integrated_density(kind):
    distribution = _distribution(kind)

    quadrature = build_defect_energy_quadrature(
        distribution,
        band_gap_eV=GAP_EV,
        order=24,
    )

    assert quadrature.integrated_density_m3 == pytest.approx(
        TOTAL_DENSITY_M3,
        rel=8.0e-16,
    )
    assert all(
        quadrature.support_lower_eV_above_vb <= energy
        <= quadrature.support_upper_eV_above_vb
        for energy in quadrature.energy_levels_eV_above_vb
    )
    assert quadrature.order == (1 if kind == SINGLE_LEVEL else 24)


def test_single_level_quadrature_and_expansion_are_exact_not_approximate():
    species = _species(_distribution(SINGLE_LEVEL))

    coarse = expand_bulk_defect_species_energy(
        species,
        band_gap_eV=GAP_EV,
        order=2,
    )
    fine = expand_bulk_defect_species_energy(
        species,
        band_gap_eV=GAP_EV,
        order=512,
    )

    assert coarse.node_species == fine.node_species == (species,)
    assert coarse.node_species[0] is species
    assert fine.node_species[0] is species
    assert coarse.quadrature.energy_levels_eV_above_vb == (0.75,)
    assert coarse.quadrature.density_weights_m3 == (TOTAL_DENSITY_M3,)


@pytest.mark.parametrize(
    ("distribution", "expected_integral_eV"),
    (
        (_distribution(UNIFORM), 0.4),
        (
            _distribution(GAUSSIAN),
            0.08
            * math.sqrt(2.0 * math.pi)
            * math.erf(8.0 / (2.0 * math.sqrt(2.0))),
        ),
        (
            _distribution(
                GAUSSIAN,
                width_convention=WIDTH_SCAPS_CHARACTERISTIC,
                support_width_multiplier=6.0,
            ),
            0.08 * math.sqrt(math.pi) * math.erf(3.0),
        ),
        (
            _distribution(CONDUCTION_BAND_TAIL),
            0.1 * (1.0 - math.exp(-7.0)),
        ),
        (
            _distribution(VALENCE_BAND_TAIL),
            0.1 * (1.0 - math.exp(-7.0)),
        ),
    ),
)
def test_peak_and_integrated_density_conversion_uses_exact_shape_integral(
    distribution,
    expected_integral_eV,
):
    integral = distribution_shape_integral_eV(distribution)
    peak = peak_density_from_integrated_density(distribution)

    assert integral == pytest.approx(expected_integral_eV, rel=2.0e-15)
    assert integrated_density_from_peak_density(
        distribution,
        peak,
    ) == pytest.approx(TOTAL_DENSITY_M3, rel=2.0e-15)


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_distributed_species_expansion_is_auditable_and_conservative(kind):
    species = _species(_distribution(kind))

    expansion = expand_bulk_defect_species_energy(
        species,
        band_gap_eV=GAP_EV,
        order=12,
    )

    assert len(expansion.node_species) == 12
    assert [item.name for item in expansion.node_species] == [
        f"acceptor_distribution::energy[{index:03d}]" for index in range(12)
    ]
    assert all(
        item.distribution.kind == SINGLE_LEVEL
        and item.kinetics is species.kinetics
        and item.charge_transition == species.charge_transition
        for item in expansion.node_species
    )
    assert math.fsum(
        item.distribution.total_density_m3
        for item in expansion.node_species
    ) == pytest.approx(TOTAL_DENSITY_M3, rel=8.0e-16)


def test_gaussian_delta_limit_concentrates_on_the_exact_single_level():
    center = 0.73
    narrow = _distribution(
        GAUSSIAN,
        center_eV_above_vb=center,
        width_eV=1.0e-10,
        support_width_multiplier=8.0,
    )

    quadrature = build_defect_energy_quadrature(
        narrow,
        band_gap_eV=GAP_EV,
        order=24,
    )

    assert max(
        abs(energy - center)
        for energy in quadrature.energy_levels_eV_above_vb
    ) < 4.0e-10
    assert quadrature.integrated_density_m3 == pytest.approx(
        TOTAL_DENSITY_M3,
        rel=8.0e-16,
    )


def test_legacy_or_out_of_gap_distributed_inputs_fail_closed():
    legacy_gaussian = replace(
        _distribution(GAUSSIAN),
        energy_reference=None,
        support_width_multiplier=None,
    )
    with pytest.raises(ExplicitDefectSchemaError, match="complete v2"):
        build_defect_energy_quadrature(
            legacy_gaussian,
            band_gap_eV=GAP_EV,
        )

    outside_gap = _distribution(
        CONDUCTION_BAND_TAIL,
        center_eV_above_vb=0.5,
        width_eV=0.1,
        support_width_multiplier=7.0,
    )
    with pytest.raises(ExplicitDefectSchemaError, match="support"):
        build_defect_energy_quadrature(
            outside_gap,
            band_gap_eV=GAP_EV,
        )


def test_exact_band_edge_support_is_not_rejected_by_binary_roundoff():
    conduction_tail = _distribution(
        CONDUCTION_BAND_TAIL,
        center_eV_above_vb=0.7,
        width_eV=0.1,
        support_width_multiplier=7.0,
    )
    valence_tail = _distribution(
        VALENCE_BAND_TAIL,
        center_eV_above_vb=0.8,
        width_eV=0.1,
        support_width_multiplier=7.0,
    )

    conduction = build_defect_energy_quadrature(
        conduction_tail,
        band_gap_eV=GAP_EV,
        order=8,
    )
    valence = build_defect_energy_quadrature(
        valence_tail,
        band_gap_eV=GAP_EV,
        order=8,
    )

    assert conduction.support_lower_eV_above_vb == 0.0
    assert valence.support_upper_eV_above_vb == GAP_EV


@pytest.mark.parametrize("order", (1, 513, 8.0, True))
def test_distributed_quadrature_order_is_strict(order):
    with pytest.raises(ValueError, match="quadrature order"):
        build_defect_energy_quadrature(
            _distribution(UNIFORM),
            band_gap_eV=GAP_EV,
            order=order,
        )


def test_peak_density_is_undefined_for_a_delta_level():
    with pytest.raises(ExplicitDefectSchemaError, match="delta-like"):
        peak_density_from_integrated_density(_distribution(SINGLE_LEVEL))
