"""Normalized energy supports for canonical explicit bulk defects.

The canonical density is always the finite-support integral in ``m^-3``.
Distributed peak densities in ``m^-3 eV^-1`` are conversion inputs only. This
module is carrier independent: it creates immutable energy nodes and density
weights, but does not enable a distributed defect in a production solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import ndtr, ndtri

from perovskite_sim.models.defects import (
    CONDUCTION_BAND_TAIL,
    ENERGY_ABOVE_VALENCE_BAND,
    GAUSSIAN,
    SINGLE_LEVEL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_SCAPS_CHARACTERISTIC,
    BulkDefectDistribution,
    BulkDefectSpecies,
    ExplicitDefectSchemaError,
)


DEFECT_ENERGY_QUADRATURE_VERSION = "normalized-finite-support-v1"
DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER = 32


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _quadrature_order(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("defect energy quadrature order must be an integer")
    order = int(value)
    if order < 2 or order > 512:
        raise ValueError("defect energy quadrature order must be in [2, 512]")
    return order


def _density_weights(probability_weights: np.ndarray, total: float) -> np.ndarray:
    weights = np.asarray(probability_weights, dtype=float) * total
    weights *= total / math.fsum(float(value) for value in weights)
    correction = total - math.fsum(float(value) for value in weights)
    weights[-1] += correction
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise FloatingPointError("defect energy density weights are invalid")
    return weights


@dataclass(frozen=True, slots=True)
class DefectEnergyQuadrature:
    """Finite energy nodes and volume-density weights for one species."""

    distribution_kind: str
    energy_reference: str | None
    energy_levels_eV_above_vb: tuple[float, ...]
    density_weights_m3: tuple[float, ...]
    support_lower_eV_above_vb: float
    support_upper_eV_above_vb: float
    total_density_m3: float
    shape_integral_eV: float | None

    def __post_init__(self) -> None:
        energies = tuple(float(value) for value in self.energy_levels_eV_above_vb)
        weights = tuple(float(value) for value in self.density_weights_m3)
        if not energies or len(energies) != len(weights):
            raise ValueError("defect energy nodes and weights must be non-empty")
        lower = float(self.support_lower_eV_above_vb)
        upper = float(self.support_upper_eV_above_vb)
        total = _finite_positive(self.total_density_m3, "total_density_m3")
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower < 0.0
            or upper < lower
            or any(not math.isfinite(value) for value in energies)
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or any(value < lower or value > upper for value in energies)
        ):
            raise ValueError("defect energy quadrature support is invalid")
        integrated = math.fsum(weights)
        if not math.isclose(integrated, total, rel_tol=8.0e-16, abs_tol=0.0):
            raise ValueError("defect energy weights do not recover total density")
        shape_integral = self.shape_integral_eV
        if self.distribution_kind == SINGLE_LEVEL:
            if len(energies) != 1 or shape_integral is not None:
                raise ValueError("single-level quadrature must contain one delta node")
        elif shape_integral is None or not math.isfinite(shape_integral) or shape_integral <= 0.0:
            raise ValueError("distributed quadrature requires a shape integral")
        object.__setattr__(self, "energy_levels_eV_above_vb", energies)
        object.__setattr__(self, "density_weights_m3", weights)
        object.__setattr__(self, "support_lower_eV_above_vb", lower)
        object.__setattr__(self, "support_upper_eV_above_vb", upper)
        object.__setattr__(self, "total_density_m3", total)

    @property
    def order(self) -> int:
        return len(self.energy_levels_eV_above_vb)

    @property
    def integrated_density_m3(self) -> float:
        return math.fsum(self.density_weights_m3)

    def to_dict(self) -> dict[str, object]:
        return {
            "quadrature": DEFECT_ENERGY_QUADRATURE_VERSION,
            "distribution_kind": self.distribution_kind,
            "energy_reference": self.energy_reference,
            "energy_levels_eV_above_vb": list(self.energy_levels_eV_above_vb),
            "density_weights_m3": list(self.density_weights_m3),
            "support_lower_eV_above_vb": self.support_lower_eV_above_vb,
            "support_upper_eV_above_vb": self.support_upper_eV_above_vb,
            "total_density_m3": self.total_density_m3,
            "shape_integral_eV": self.shape_integral_eV,
        }


@dataclass(frozen=True, slots=True)
class DefectSpeciesEnergyExpansion:
    """One canonical species expanded into auditable single-level nodes."""

    source_species: BulkDefectSpecies
    quadrature: DefectEnergyQuadrature
    node_species: tuple[BulkDefectSpecies, ...]

    def __post_init__(self) -> None:
        nodes = tuple(self.node_species)
        if not isinstance(self.source_species, BulkDefectSpecies):
            raise TypeError("source_species must be a BulkDefectSpecies")
        if not isinstance(self.quadrature, DefectEnergyQuadrature):
            raise TypeError("quadrature must be a DefectEnergyQuadrature")
        if len(nodes) != self.quadrature.order or not all(
            isinstance(item, BulkDefectSpecies) for item in nodes
        ):
            raise ValueError("node_species must match the energy quadrature")
        object.__setattr__(self, "node_species", nodes)


def distribution_shape_integral_eV(
    distribution: BulkDefectDistribution,
) -> float:
    """Return ``integral shape(E) dE`` for a v2 distributed species.

    The peak of every supported shape is one. Multiplying this integral by a
    peak density in ``m^-3 eV^-1`` therefore yields the canonical integrated
    density in ``m^-3``.
    """

    if not isinstance(distribution, BulkDefectDistribution):
        raise TypeError("distribution must be a BulkDefectDistribution")
    if distribution.kind == SINGLE_LEVEL:
        raise ExplicitDefectSchemaError(
            "a delta-like single level has no finite peak-density integral"
        )
    if not distribution.v2_ready:
        raise ExplicitDefectSchemaError(
            "distributed density conversion requires the complete v2 contract"
        )
    width = float(distribution.width_eV)
    if distribution.kind == UNIFORM:
        return width
    multiplier = float(distribution.support_width_multiplier)
    if distribution.kind == GAUSSIAN:
        if distribution.width_convention == WIDTH_GAUSSIAN_SIGMA:
            return (
                width
                * math.sqrt(2.0 * math.pi)
                * math.erf(multiplier / (2.0 * math.sqrt(2.0)))
            )
        if distribution.width_convention == WIDTH_SCAPS_CHARACTERISTIC:
            return (
                width
                * math.sqrt(math.pi)
                * math.erf(0.5 * multiplier)
            )
        raise ExplicitDefectSchemaError(
            "gaussian width convention is not executable"
        )
    return width * -math.expm1(-multiplier)


def integrated_density_from_peak_density(
    distribution: BulkDefectDistribution,
    peak_density_m3_eV: object,
) -> float:
    """Convert a distributed peak density to canonical integrated density."""

    peak = _finite_positive(peak_density_m3_eV, "peak_density_m3_eV")
    return peak * distribution_shape_integral_eV(distribution)


def peak_density_from_integrated_density(
    distribution: BulkDefectDistribution,
) -> float:
    """Return the peak density implied by the canonical integrated density."""

    return (
        float(distribution.total_density_m3)
        / distribution_shape_integral_eV(distribution)
    )


def build_defect_energy_quadrature(
    distribution: BulkDefectDistribution,
    *,
    band_gap_eV: object,
    order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
) -> DefectEnergyQuadrature:
    """Build normalized nodes without enabling distributed solver physics."""

    if not isinstance(distribution, BulkDefectDistribution):
        raise TypeError("distribution must be a BulkDefectDistribution")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    distribution.validate_band_gap(gap)
    support = distribution.support_bounds_eV()
    if distribution.kind == SINGLE_LEVEL:
        energy = float(distribution.center_eV_above_vb)
        density = float(distribution.total_density_m3)
        return DefectEnergyQuadrature(
            distribution_kind=SINGLE_LEVEL,
            energy_reference=distribution.energy_reference,
            energy_levels_eV_above_vb=(energy,),
            density_weights_m3=(density,),
            support_lower_eV_above_vb=energy,
            support_upper_eV_above_vb=energy,
            total_density_m3=density,
            shape_integral_eV=None,
        )
    if not distribution.v2_ready or support is None:
        raise ExplicitDefectSchemaError(
            "distributed quadrature requires the complete v2 contract"
        )

    resolved_order = _quadrature_order(order)
    nodes, legendre_weights = np.polynomial.legendre.leggauss(resolved_order)
    probability = 0.5 * (nodes + 1.0)
    probability_weights = 0.5 * legendre_weights
    # Normalize only arithmetic roundoff at an exactly declared band edge.
    lower = max(0.0, support[0])
    upper = min(gap, support[1])
    center = float(distribution.center_eV_above_vb)
    width = float(distribution.width_eV)

    if distribution.kind == UNIFORM:
        energy = lower + probability * (upper - lower)
    elif distribution.kind == GAUSSIAN:
        sigma = (
            width
            if distribution.width_convention == WIDTH_GAUSSIAN_SIGMA
            else width / math.sqrt(2.0)
        )
        lower_cdf = float(ndtr((lower - center) / sigma))
        upper_cdf = float(ndtr((upper - center) / sigma))
        mass = upper_cdf - lower_cdf
        if not math.isfinite(mass) or mass <= np.finfo(float).eps:
            raise FloatingPointError(
                "gaussian defect support has no resolvable probability mass"
            )
        energy = center + sigma * ndtri(
            lower_cdf + probability * mass
        )
    else:
        multiplier = float(distribution.support_width_multiplier)
        cutoff = math.exp(-multiplier)
        if distribution.kind == CONDUCTION_BAND_TAIL:
            energy = center + width * np.log(
                cutoff + probability * (1.0 - cutoff)
            )
        elif distribution.kind == VALENCE_BAND_TAIL:
            energy = center - width * np.log(
                1.0 - probability * (1.0 - cutoff)
            )
        else:
            raise ExplicitDefectSchemaError(
                f"unsupported defect distribution {distribution.kind!r}"
            )

    if (
        not np.all(np.isfinite(energy))
        or np.any(energy < lower)
        or np.any(energy > upper)
        or np.any(energy < 0.0)
        or np.any(energy > gap)
    ):
        raise FloatingPointError("defect energy nodes are outside their support")
    weights = _density_weights(
        probability_weights,
        float(distribution.total_density_m3),
    )
    return DefectEnergyQuadrature(
        distribution_kind=distribution.kind,
        energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        energy_levels_eV_above_vb=tuple(float(value) for value in energy),
        density_weights_m3=tuple(float(value) for value in weights),
        support_lower_eV_above_vb=lower,
        support_upper_eV_above_vb=upper,
        total_density_m3=float(distribution.total_density_m3),
        shape_integral_eV=distribution_shape_integral_eV(distribution),
    )


def expand_bulk_defect_species_energy(
    species: BulkDefectSpecies,
    *,
    band_gap_eV: object,
    order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
) -> DefectSpeciesEnergyExpansion:
    """Expand one species; a single level is returned by object identity."""

    if not isinstance(species, BulkDefectSpecies):
        raise TypeError("species must be a BulkDefectSpecies")
    quadrature = build_defect_energy_quadrature(
        species.distribution,
        band_gap_eV=band_gap_eV,
        order=order,
    )
    if species.distribution.kind == SINGLE_LEVEL:
        nodes = (species,)
    else:
        if species.name is None:
            raise ExplicitDefectSchemaError(
                "distributed energy expansion requires a named species"
            )
        nodes = tuple(
            replace(
                species,
                name=f"{species.name}::energy[{index:03d}]",
                distribution=BulkDefectDistribution(
                    kind=SINGLE_LEVEL,
                    normalization=species.distribution.normalization,
                    total_density_m3=density,
                    center_eV_above_vb=energy,
                    energy_reference=species.distribution.energy_reference,
                ),
            )
            for index, (energy, density) in enumerate(
                zip(
                    quadrature.energy_levels_eV_above_vb,
                    quadrature.density_weights_m3,
                    strict=True,
                )
            )
        )
    return DefectSpeciesEnergyExpansion(
        source_species=species,
        quadrature=quadrature,
        node_species=nodes,
    )


__all__ = [
    "DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER",
    "DEFECT_ENERGY_QUADRATURE_VERSION",
    "DefectEnergyQuadrature",
    "DefectSpeciesEnergyExpansion",
    "build_defect_energy_quadrature",
    "distribution_shape_integral_eV",
    "expand_bulk_defect_species_energy",
    "integrated_density_from_peak_density",
    "peak_density_from_integrated_density",
]
