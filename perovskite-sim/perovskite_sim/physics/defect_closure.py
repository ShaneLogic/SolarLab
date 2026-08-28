"""Maxwell-Boltzmann closure for canonical monovalent bulk defects.

The local primitive evaluates occupancy, recombination, charge, and their
analytic tangents from one state. Device-level helpers compile the same
primitive onto disjoint material regions so every downstream equation consumes
one constitutive interpretation of the explicit-defect document.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    NEUTRAL,
    SINGLE_LEVEL,
    BulkDefectDocument,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
    bulk_defect_species_at_layer_position,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    DefectEnergyQuadrature,
    DefectSpeciesEnergyExpansion,
    expand_bulk_defect_species_energy,
    validate_defect_energy_quadrature_order,
)
from perovskite_sim.physics.temperature import thermal_voltage


if TYPE_CHECKING:
    from perovskite_sim.physics.distributed_defect_closure import (
        EnergyDistributedDefectClosureResult,
    )
    from perovskite_sim.physics.statistics import BulkChargeNeutralityState


MONOVALENT_DEFECT_CLOSURE_VERSION = "monovalent-local-mb-v1"
MONOVALENT_BULK_DEFECT_MODEL_VERSION = "monovalent-device-mb-qf-dc-v1"


class MonovalentDefectClosureCapabilityError(ExplicitDefectCapabilityError):
    """A valid defect input requested physics outside the DEF-2 closure."""


@dataclass(frozen=True, slots=True)
class MonovalentDefectNeutralityResult:
    """Common-Fermi-level contact state using the same local defect closure."""

    neutrality: "BulkChargeNeutralityState"
    closure: MonovalentDefectClosureResult | EnergyDistributedDefectClosureResult


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


def _readonly(value: object, *, dtype: object = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MonovalentDefectClosureResult:
    """Per-species and total local quasi-steady defect observables."""

    closure_identity_sha256: str
    temperature_K: float
    thermal_voltage_V: float
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    intrinsic_product_m6: float
    species_identifiers: tuple[str, ...]
    charge_transitions: tuple[str, ...]
    n1_m3: np.ndarray
    p1_m3: np.ndarray
    capture_n_m3_s: np.ndarray
    capture_p_m3_s: np.ndarray
    kinetic_denominator_s1: np.ndarray
    occupancy: np.ndarray
    occupancy_derivative_n_m3: np.ndarray
    occupancy_derivative_p_m3: np.ndarray
    occupied_density_m3: np.ndarray
    signed_charge_number_density_m3: np.ndarray
    charge_density_C_m3: np.ndarray
    recombination_rate_m3_s: np.ndarray
    recombination_derivative_n_s1: np.ndarray
    recombination_derivative_p_s1: np.ndarray
    charge_derivative_n_C: np.ndarray
    charge_derivative_p_C: np.ndarray
    charge_derivative_fixed_qf_C_m3_V: np.ndarray
    recombination_derivative_fixed_qf_m3_s_V: np.ndarray
    total_charge_density_C_m3: np.ndarray
    total_recombination_rate_m3_s: np.ndarray
    total_recombination_derivative_n_s1: np.ndarray
    total_recombination_derivative_p_s1: np.ndarray
    total_charge_derivative_n_C: np.ndarray
    total_charge_derivative_p_C: np.ndarray
    total_charge_derivative_fixed_qf_C_m3_V: np.ndarray
    total_recombination_derivative_fixed_qf_m3_s_V: np.ndarray
    minimum_occupancy: float
    maximum_occupancy: float

    def __post_init__(self) -> None:
        digest = str(self.closure_identity_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("closure_identity_sha256 must be a SHA-256 hex")
        identifiers = tuple(self.species_identifiers)
        transitions = tuple(self.charge_transitions)
        if (
            not identifiers
            or len(identifiers) != len(set(identifiers))
            or len(transitions) != len(identifiers)
        ):
            raise ValueError("closure result requires unique species identifiers")
        if any(value not in {NEUTRAL, ACCEPTOR, DONOR} for value in transitions):
            raise ValueError("closure result has an unsupported charge transition")
        for name in (
            "temperature_K",
            "thermal_voltage_V",
            "band_gap_eV",
            "effective_conduction_dos_m3",
            "effective_valence_dos_m3",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        intrinsic_product = float(self.intrinsic_product_m6)
        if not math.isfinite(intrinsic_product) or intrinsic_product < 0.0:
            raise ValueError("intrinsic_product_m6 must be finite and non-negative")
        object.__setattr__(self, "intrinsic_product_m6", intrinsic_product)
        species_count = len(identifiers)
        occupancy = np.asarray(self.occupancy, dtype=float)
        if occupancy.ndim < 1 or occupancy.shape[0] != species_count:
            raise ValueError("closure occupancy must start with the species axis")
        species_shape = occupancy.shape
        state_shape = occupancy.shape[1:]
        reference_names = (
            "n1_m3",
            "p1_m3",
            "capture_n_m3_s",
            "capture_p_m3_s",
        )
        for name in reference_names:
            if np.asarray(getattr(self, name)).shape != (species_count,):
                raise ValueError(f"{name} must have one value per species")
        species_names = (
            "kinetic_denominator_s1",
            "occupancy",
            "occupancy_derivative_n_m3",
            "occupancy_derivative_p_m3",
            "occupied_density_m3",
            "signed_charge_number_density_m3",
            "charge_density_C_m3",
            "recombination_rate_m3_s",
            "recombination_derivative_n_s1",
            "recombination_derivative_p_s1",
            "charge_derivative_n_C",
            "charge_derivative_p_C",
            "charge_derivative_fixed_qf_C_m3_V",
            "recombination_derivative_fixed_qf_m3_s_V",
        )
        for name in species_names:
            if np.asarray(getattr(self, name)).shape != species_shape:
                raise ValueError(f"{name} must match the per-species state shape")
        total_names = (
            "total_charge_density_C_m3",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
            "total_charge_derivative_n_C",
            "total_charge_derivative_p_C",
            "total_charge_derivative_fixed_qf_C_m3_V",
            "total_recombination_derivative_fixed_qf_m3_s_V",
        )
        for name in total_names:
            if np.asarray(getattr(self, name)).shape != state_shape:
                raise ValueError(f"{name} must match the carrier state shape")
        for name in reference_names + species_names + total_names:
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, _readonly(value))
        minimum = float(self.minimum_occupancy)
        maximum = float(self.maximum_occupancy)
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < 0.0
            or maximum > 1.0
            or minimum > maximum
            or minimum != float(np.min(occupancy))
            or maximum != float(np.max(occupancy))
        ):
            raise ValueError("closure occupancy bounds are inconsistent")
        object.__setattr__(self, "minimum_occupancy", minimum)
        object.__setattr__(self, "maximum_occupancy", maximum)
        object.__setattr__(self, "closure_identity_sha256", digest)
        object.__setattr__(self, "species_identifiers", identifiers)
        object.__setattr__(self, "charge_transitions", transitions)

    def to_dict(self) -> dict[str, object]:
        """Return the local closure evidence as JSON-compatible data."""

        return {
            "closure": MONOVALENT_DEFECT_CLOSURE_VERSION,
            "closure_identity_sha256": self.closure_identity_sha256,
            "statistics": "maxwell_boltzmann",
            "temperature_K": self.temperature_K,
            "thermal_voltage_V": self.thermal_voltage_V,
            "band_gap_eV": self.band_gap_eV,
            "effective_conduction_dos_m3": self.effective_conduction_dos_m3,
            "effective_valence_dos_m3": self.effective_valence_dos_m3,
            "intrinsic_product_m6": self.intrinsic_product_m6,
            "species_identifiers": list(self.species_identifiers),
            "charge_transitions": list(self.charge_transitions),
            "n1_m3": self.n1_m3.tolist(),
            "p1_m3": self.p1_m3.tolist(),
            "capture_n_m3_s": self.capture_n_m3_s.tolist(),
            "capture_p_m3_s": self.capture_p_m3_s.tolist(),
            "kinetic_denominator_s1": self.kinetic_denominator_s1.tolist(),
            "occupancy": self.occupancy.tolist(),
            "occupancy_derivative_n_m3": self.occupancy_derivative_n_m3.tolist(),
            "occupancy_derivative_p_m3": self.occupancy_derivative_p_m3.tolist(),
            "occupied_density_m3": self.occupied_density_m3.tolist(),
            "signed_charge_number_density_m3": (
                self.signed_charge_number_density_m3.tolist()
            ),
            "charge_density_C_m3": self.charge_density_C_m3.tolist(),
            "recombination_rate_m3_s": self.recombination_rate_m3_s.tolist(),
            "recombination_derivative_n_s1": (
                self.recombination_derivative_n_s1.tolist()
            ),
            "recombination_derivative_p_s1": (
                self.recombination_derivative_p_s1.tolist()
            ),
            "charge_derivative_n_C": self.charge_derivative_n_C.tolist(),
            "charge_derivative_p_C": self.charge_derivative_p_C.tolist(),
            "charge_derivative_fixed_qf_C_m3_V": (
                self.charge_derivative_fixed_qf_C_m3_V.tolist()
            ),
            "recombination_derivative_fixed_qf_m3_s_V": (
                self.recombination_derivative_fixed_qf_m3_s_V.tolist()
            ),
            "total_charge_density_C_m3": self.total_charge_density_C_m3.tolist(),
            "total_recombination_rate_m3_s": (
                self.total_recombination_rate_m3_s.tolist()
            ),
            "total_recombination_derivative_n_s1": (
                self.total_recombination_derivative_n_s1.tolist()
            ),
            "total_recombination_derivative_p_s1": (
                self.total_recombination_derivative_p_s1.tolist()
            ),
            "total_charge_derivative_n_C": self.total_charge_derivative_n_C.tolist(),
            "total_charge_derivative_p_C": self.total_charge_derivative_p_C.tolist(),
            "total_charge_derivative_fixed_qf_C_m3_V": (
                self.total_charge_derivative_fixed_qf_C_m3_V.tolist()
            ),
            "total_recombination_derivative_fixed_qf_m3_s_V": (
                self.total_recombination_derivative_fixed_qf_m3_s_V.tolist()
            ),
            "minimum_occupancy": self.minimum_occupancy,
            "maximum_occupancy": self.maximum_occupancy,
        }


@dataclass(frozen=True, slots=True)
class MonovalentDefectRegion:
    """One canonical defect document compiled onto a disjoint node region."""

    identifier: str
    document_sha256: str
    active_nodes: np.ndarray
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    temperature_K: float
    species: tuple[BulkDefectSpecies, ...]
    schema_version: str = EXPLICIT_DEFECT_SCHEMA_VERSION
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER
    normalized_layer_coordinates: np.ndarray | None = None
    local_band_gap_eV: np.ndarray | None = None
    local_effective_conduction_dos_m3: np.ndarray | None = None
    local_effective_valence_dos_m3: np.ndarray | None = None
    source_density_multipliers: np.ndarray | None = None
    source_expansions: tuple[DefectSpeciesEnergyExpansion, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    source_quadratures: tuple[DefectEnergyQuadrature, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    execution_species: tuple[BulkDefectSpecies, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    source_node_ranges: tuple[tuple[int, int], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise ValueError("monovalent defect region identifier must be non-empty")
        digest = str(self.document_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("defect region document_sha256 must be a SHA-256 hex")
        active = np.asarray(self.active_nodes, dtype=bool)
        if active.ndim != 1 or not np.any(active):
            raise ValueError("defect region active_nodes must be a non-empty 1D mask")
        gap = _finite_positive(self.band_gap_eV, "band_gap_eV")
        conduction_dos = _finite_positive(
            self.effective_conduction_dos_m3,
            "effective_conduction_dos_m3",
        )
        valence_dos = _finite_positive(
            self.effective_valence_dos_m3,
            "effective_valence_dos_m3",
        )
        temperature = _finite_positive(self.temperature_K, "temperature_K")
        spatial = self.schema_version == EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION
        spatial_values = (
            self.normalized_layer_coordinates,
            self.local_band_gap_eV,
            self.local_effective_conduction_dos_m3,
            self.local_effective_valence_dos_m3,
            self.source_density_multipliers,
        )
        if spatial and any(value is None for value in spatial_values):
            raise ValueError(
                "v3 defect regions require coordinates, local band data, and "
                "source density multipliers"
            )
        if not spatial and any(value is not None for value in spatial_values):
            raise ValueError("spatial defect region arrays require the v3 schema")
        coordinates = None
        local_gap = None
        local_nc = None
        local_nv = None
        density_multipliers = None
        if spatial:
            active_count = int(np.count_nonzero(active))
            coordinates = np.asarray(
                self.normalized_layer_coordinates,
                dtype=float,
            )
            local_gap = np.asarray(self.local_band_gap_eV, dtype=float)
            local_nc = np.asarray(
                self.local_effective_conduction_dos_m3,
                dtype=float,
            )
            local_nv = np.asarray(
                self.local_effective_valence_dos_m3,
                dtype=float,
            )
            density_multipliers = np.asarray(
                self.source_density_multipliers,
                dtype=float,
            )
            for name, values in (
                ("normalized_layer_coordinates", coordinates),
                ("local_band_gap_eV", local_gap),
                ("local_effective_conduction_dos_m3", local_nc),
                ("local_effective_valence_dos_m3", local_nv),
            ):
                if values.shape != (active_count,) or not np.all(
                    np.isfinite(values)
                ):
                    raise ValueError(f"{name} must be finite and match active nodes")
            if (
                np.any(coordinates < 0.0)
                or np.any(coordinates > 1.0)
                or np.any(np.diff(coordinates) <= 0.0)
            ):
                raise ValueError(
                    "normalized layer coordinates must increase within [0, 1]"
                )
            if (
                np.any(local_gap <= 0.0)
                or np.any(local_nc <= 0.0)
                or np.any(local_nv <= 0.0)
            ):
                raise ValueError("local band data must be strictly positive")
            if density_multipliers.shape != (
                len(self.species),
                active_count,
            ) or not np.all(np.isfinite(density_multipliers)):
                raise ValueError(
                    "source density multipliers must match species and active nodes"
                )
            if np.any(density_multipliers <= 0.0):
                raise ValueError("source density multipliers must be positive")
            expected_multipliers = np.asarray(
                [
                    [
                        1.0
                        if source.spatial_profile is None
                        else source.spatial_profile.density_multiplier_at(position)
                        for position in coordinates
                    ]
                    for source in self.species
                ],
                dtype=float,
            )
            if not np.array_equal(density_multipliers, expected_multipliers):
                raise ValueError(
                    "source density multipliers do not match spatial profiles"
                )
            if (
                gap != float(local_gap[0])
                or conduction_dos != float(local_nc[0])
                or valence_dos != float(local_nv[0])
            ):
                raise ValueError(
                    "v3 scalar band references must equal the first local node"
                )
            validation_gap = float(np.min(local_gap))
        else:
            validation_gap = gap
        species = _validate_source_species(
            self.species,
            band_gap_eV=validation_gap,
            allow_spatial_profiles=spatial,
        )
        energy_order = validate_defect_energy_quadrature_order(
            self.energy_quadrature_order
        )
        expected_digest = BulkDefectDocument(
            schema_version=self.schema_version,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=species,
        ).sha256
        if digest != expected_digest:
            raise ValueError(
                "defect region document_sha256 does not match its species"
            )
        expansions: list[DefectSpeciesEnergyExpansion] = []
        quadratures: list[DefectEnergyQuadrature] = []
        execution_species: list[BulkDefectSpecies] = []
        source_node_ranges: list[tuple[int, int]] = []
        for source in species:
            expansion = expand_bulk_defect_species_energy(
                source,
                band_gap_eV=validation_gap,
                order=energy_order,
            )
            expansions.append(expansion)
            start = len(execution_species)
            execution_species.extend(expansion.node_species)
            source_node_ranges.append((start, len(execution_species)))
            quadratures.append(expansion.quadrature)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "document_sha256", digest)
        object.__setattr__(self, "active_nodes", _readonly(active, dtype=bool))
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(
            self,
            "effective_conduction_dos_m3",
            conduction_dos,
        )
        object.__setattr__(self, "effective_valence_dos_m3", valence_dos)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "energy_quadrature_order", energy_order)
        object.__setattr__(self, "source_expansions", tuple(expansions))
        object.__setattr__(self, "source_quadratures", tuple(quadratures))
        object.__setattr__(
            self,
            "execution_species",
            tuple(execution_species),
        )
        object.__setattr__(
            self,
            "source_node_ranges",
            tuple(source_node_ranges),
        )
        if spatial:
            object.__setattr__(
                self,
                "normalized_layer_coordinates",
                _readonly(coordinates),
            )
            object.__setattr__(self, "local_band_gap_eV", _readonly(local_gap))
            object.__setattr__(
                self,
                "local_effective_conduction_dos_m3",
                _readonly(local_nc),
            )
            object.__setattr__(
                self,
                "local_effective_valence_dos_m3",
                _readonly(local_nv),
            )
            object.__setattr__(
                self,
                "source_density_multipliers",
                _readonly(density_multipliers),
            )

    @property
    def has_distributed_species(self) -> bool:
        return any(
            item.distribution.kind != SINGLE_LEVEL for item in self.species
        )

    @property
    def has_spatial_profiles(self) -> bool:
        return self.schema_version == EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION

    @property
    def spatial_profile_sha256s(self) -> tuple[str | None, ...]:
        return tuple(
            None if item.spatial_profile is None else item.spatial_profile.sha256
            for item in self.species
        )

    @property
    def source_density_multiplier_bounds(
        self,
    ) -> tuple[tuple[float, float], ...]:
        if not self.has_spatial_profiles:
            return tuple((1.0, 1.0) for _ in self.species)
        multipliers = np.asarray(self.source_density_multipliers)
        return tuple(
            (float(np.min(values)), float(np.max(values)))
            for values in multipliers
        )

    def species_at_local_node(self, index: int) -> tuple[BulkDefectSpecies, ...]:
        if not self.has_spatial_profiles:
            return self.species
        coordinates = np.asarray(self.normalized_layer_coordinates)
        if index < 0 or index >= coordinates.size:
            raise IndexError("local defect node index is outside the region")
        return bulk_defect_species_at_layer_position(
            self.species,
            float(coordinates[index]),
        )

    @property
    def distribution_kinds(self) -> tuple[str, ...]:
        return tuple(item.distribution.kind for item in self.species)

    @property
    def source_energy_orders(self) -> tuple[int, ...]:
        return tuple(item.order for item in self.source_quadratures)

    @property
    def source_node_identifiers(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(
                str(item.name)
                for item in self.execution_species[start:stop]
            )
            for start, stop in self.source_node_ranges
        )


@dataclass(frozen=True, slots=True)
class MonovalentBulkDefectModel:
    """Disjoint device regions governed by the monovalent local closure."""

    regions: tuple[MonovalentDefectRegion, ...]

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions or not all(
            isinstance(item, MonovalentDefectRegion) for item in regions
        ):
            raise ValueError("monovalent bulk-defect model requires regions")
        identifiers = [item.identifier for item in regions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("monovalent defect region identifiers must be unique")
        node_count = regions[0].active_nodes.size
        if any(item.active_nodes.size != node_count for item in regions):
            raise ValueError("monovalent defect regions must share one grid")
        occupancy = np.zeros(node_count, dtype=np.int8)
        for region in regions:
            occupancy += region.active_nodes.astype(np.int8)
        if np.any(occupancy > 1):
            raise ValueError("monovalent defect regions must not overlap")
        species_identifiers = [
            f"{region.identifier}/{species.name}"
            for region in regions
            for species in region.species
        ]
        if len(species_identifiers) != len(set(species_identifiers)):
            raise ValueError("compiled monovalent defect identifiers must be unique")
        object.__setattr__(self, "regions", regions)

    @property
    def node_count(self) -> int:
        return int(self.regions[0].active_nodes.size)

    @property
    def explicit_node_mask(self) -> np.ndarray:
        return np.logical_or.reduce(
            [region.active_nodes for region in self.regions]
        )

    @property
    def species_identifiers(self) -> tuple[str, ...]:
        return tuple(
            f"{region.identifier}/{species.name}"
            for region in self.regions
            for species in region.species
        )

    @property
    def charge_transitions(self) -> tuple[str, ...]:
        return tuple(
            species.charge_transition
            for region in self.regions
            for species in region.species
        )

    @property
    def has_distributed_species(self) -> bool:
        return any(region.has_distributed_species for region in self.regions)

    @property
    def has_spatial_profiles(self) -> bool:
        return any(region.has_spatial_profiles for region in self.regions)

    @property
    def spatial_profile_sha256s(self) -> tuple[str | None, ...]:
        return tuple(
            digest
            for region in self.regions
            for digest in region.spatial_profile_sha256s
        )

    @property
    def source_density_multiplier_bounds(
        self,
    ) -> tuple[tuple[float, float], ...]:
        return tuple(
            bounds
            for region in self.regions
            for bounds in region.source_density_multiplier_bounds
        )

    @property
    def distribution_kinds(self) -> tuple[str, ...]:
        return tuple(
            kind
            for region in self.regions
            for kind in region.distribution_kinds
        )

    @property
    def source_energy_orders(self) -> tuple[int, ...]:
        return tuple(
            order
            for region in self.regions
            for order in region.source_energy_orders
        )

    @property
    def source_node_identifiers(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(f"{region.identifier}/{name}" for name in identifiers)
            for region in self.regions
            for identifiers in region.source_node_identifiers
        )

    @property
    def identity_sha256(self) -> str:
        payload = {
            "model": MONOVALENT_BULK_DEFECT_MODEL_VERSION,
            "regions": [
                {
                    "identifier": region.identifier,
                    "document_sha256": region.document_sha256,
                    "active_node_indices": np.flatnonzero(
                        region.active_nodes
                    ).tolist(),
                    "band_gap_eV": region.band_gap_eV,
                    "effective_conduction_dos_m3": (
                        region.effective_conduction_dos_m3
                    ),
                    "effective_valence_dos_m3": (
                        region.effective_valence_dos_m3
                    ),
                    "temperature_K": region.temperature_K,
                    **(
                        {
                            "energy_quadrature_order": (
                                region.energy_quadrature_order
                            )
                        }
                        if region.has_distributed_species
                        else {}
                    ),
                    **(
                        {
                            "spatial_closure": "layer-density-profile-v1",
                            "normalized_layer_coordinates": (
                                region.normalized_layer_coordinates.tolist()
                            ),
                            "local_band_gap_eV": region.local_band_gap_eV.tolist(),
                            "local_effective_conduction_dos_m3": (
                                region.local_effective_conduction_dos_m3.tolist()
                            ),
                            "local_effective_valence_dos_m3": (
                                region.local_effective_valence_dos_m3.tolist()
                            ),
                            "source_density_multipliers": (
                                region.source_density_multipliers.tolist()
                            ),
                            "spatial_profile_sha256s": list(
                                region.spatial_profile_sha256s
                            ),
                        }
                        if region.has_spatial_profiles
                        else {}
                    ),
                }
                for region in self.regions
            ],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class MonovalentBulkDefectEvaluation:
    """Grid-aligned species diagnostics and total QF/DC source terms."""

    model_identity_sha256: str
    species_identifiers: tuple[str, ...]
    charge_transitions: tuple[str, ...]
    active_nodes: np.ndarray
    kinetic_denominator_s1: np.ndarray
    occupancy: np.ndarray
    occupied_density_m3: np.ndarray
    charge_density_C_m3: np.ndarray
    recombination_rate_m3_s: np.ndarray
    recombination_derivative_n_s1: np.ndarray
    recombination_derivative_p_s1: np.ndarray
    charge_derivative_fixed_qf_C_m3_V: np.ndarray
    total_charge_density_C_m3: np.ndarray
    total_recombination_rate_m3_s: np.ndarray
    total_recombination_derivative_n_s1: np.ndarray
    total_recombination_derivative_p_s1: np.ndarray
    total_charge_derivative_fixed_qf_C_m3_V: np.ndarray
    minimum_occupancy: float
    maximum_occupancy: float
    minimum_kinetic_denominator_s1: float
    distribution_kinds: tuple[str, ...] = ()
    source_energy_orders: tuple[int, ...] = ()
    source_node_identifiers: tuple[tuple[str, ...], ...] = ()
    source_density_multiplier: np.ndarray | None = None
    spatial_profile_sha256s: tuple[str | None, ...] = ()
    minimum_density_multipliers: tuple[float, ...] = ()
    maximum_density_multipliers: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        digest = str(self.model_identity_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("model_identity_sha256 must be a SHA-256 hex")
        identifiers = tuple(self.species_identifiers)
        transitions = tuple(self.charge_transitions)
        active = np.asarray(self.active_nodes, dtype=bool)
        if (
            active.ndim != 2
            or active.shape[0] != len(identifiers)
            or len(transitions) != len(identifiers)
            or not identifiers
            or len(identifiers) != len(set(identifiers))
            or np.any(~np.any(active, axis=1))
        ):
            raise ValueError("monovalent bulk-defect evaluation identity is invalid")
        if any(value not in {NEUTRAL, ACCEPTOR, DONOR} for value in transitions):
            raise ValueError("bulk-defect evaluation has an unsupported transition")
        species_fields = (
            "kinetic_denominator_s1",
            "occupancy",
            "occupied_density_m3",
            "charge_density_C_m3",
            "recombination_rate_m3_s",
            "recombination_derivative_n_s1",
            "recombination_derivative_p_s1",
            "charge_derivative_fixed_qf_C_m3_V",
        )
        for name in species_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != active.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and match active_nodes")
            object.__setattr__(self, name, _readonly(value))
        total_fields = (
            "total_charge_density_C_m3",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
            "total_charge_derivative_fixed_qf_C_m3_V",
        )
        for name in total_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != active.shape[1:] or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and match the device grid")
            object.__setattr__(self, name, _readonly(value))
        kinds = tuple(self.distribution_kinds)
        orders = tuple(self.source_energy_orders)
        node_identifiers = tuple(
            tuple(values) for values in self.source_node_identifiers
        )
        distributed_metadata = bool(kinds)
        if distributed_metadata:
            if (
                len(kinds) != len(identifiers)
                or len(orders) != len(identifiers)
                or len(node_identifiers) != len(identifiers)
                or any(order < 1 for order in orders)
                or any(
                    len(nodes) != order
                    for nodes, order in zip(
                        node_identifiers,
                        orders,
                        strict=True,
                    )
                )
                or any(
                    len(nodes) != len(set(nodes))
                    for nodes in node_identifiers
                )
            ):
                raise ValueError(
                    "distributed bulk-defect metadata is inconsistent"
                )
        elif orders or node_identifiers:
            raise ValueError("partial distributed bulk-defect metadata is invalid")
        profile_sha256s = tuple(self.spatial_profile_sha256s)
        minimum_multipliers = tuple(
            float(value) for value in self.minimum_density_multipliers
        )
        maximum_multipliers = tuple(
            float(value) for value in self.maximum_density_multipliers
        )
        multiplier = self.source_density_multiplier
        spatial_metadata = bool(profile_sha256s)
        if spatial_metadata:
            if (
                multiplier is None
                or len(profile_sha256s) != len(identifiers)
                or len(minimum_multipliers) != len(identifiers)
                or len(maximum_multipliers) != len(identifiers)
                or any(
                    value is not None
                    and (
                        len(str(value)) != 64
                        or any(c not in "0123456789abcdef" for c in str(value))
                    )
                    for value in profile_sha256s
                )
            ):
                raise ValueError("spatial bulk-defect metadata is inconsistent")
            multiplier_array = np.asarray(multiplier, dtype=float)
            if (
                multiplier_array.shape != active.shape
                or not np.all(np.isfinite(multiplier_array))
                or np.any(multiplier_array[active] <= 0.0)
                or np.any(multiplier_array[~active] != 0.0)
            ):
                raise ValueError(
                    "source_density_multiplier must be positive on active nodes"
                )
            for index, row_active in enumerate(active):
                values = multiplier_array[index, row_active]
                if (
                    minimum_multipliers[index] != float(np.min(values))
                    or maximum_multipliers[index] != float(np.max(values))
                ):
                    raise ValueError(
                        "spatial density multiplier bounds are inconsistent"
                    )
            object.__setattr__(
                self,
                "source_density_multiplier",
                _readonly(multiplier_array),
            )
        elif (
            multiplier is not None
            or minimum_multipliers
            or maximum_multipliers
        ):
            raise ValueError("partial spatial bulk-defect metadata is invalid")
        active_occupancy = np.asarray(self.occupancy)[active]
        active_denominator = np.asarray(self.kinetic_denominator_s1)[active]
        minimum = float(self.minimum_occupancy)
        maximum = float(self.maximum_occupancy)
        minimum_denominator = float(self.minimum_kinetic_denominator_s1)
        if (
            (
                not distributed_metadata
                and (
                    minimum != float(np.min(active_occupancy))
                    or maximum != float(np.max(active_occupancy))
                )
            )
            or (
                distributed_metadata
                and (
                    minimum > float(np.min(active_occupancy))
                    or maximum < float(np.max(active_occupancy))
                )
            )
            or minimum < 0.0
            or maximum > 1.0
            or not math.isfinite(minimum_denominator)
            or minimum_denominator <= 0.0
            or minimum_denominator != float(np.min(active_denominator))
        ):
            raise ValueError("bulk-defect evaluation extrema are inconsistent")
        object.__setattr__(self, "model_identity_sha256", digest)
        object.__setattr__(self, "species_identifiers", identifiers)
        object.__setattr__(self, "charge_transitions", transitions)
        object.__setattr__(self, "active_nodes", _readonly(active, dtype=bool))
        object.__setattr__(self, "distribution_kinds", kinds)
        object.__setattr__(self, "source_energy_orders", orders)
        object.__setattr__(self, "source_node_identifiers", node_identifiers)
        object.__setattr__(self, "spatial_profile_sha256s", profile_sha256s)
        object.__setattr__(
            self,
            "minimum_density_multipliers",
            minimum_multipliers,
        )
        object.__setattr__(
            self,
            "maximum_density_multipliers",
            maximum_multipliers,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible constitutive evidence payload."""

        payload = {
            "model": MONOVALENT_BULK_DEFECT_MODEL_VERSION,
            "model_identity_sha256": self.model_identity_sha256,
            "species_identifiers": list(self.species_identifiers),
            "charge_transitions": list(self.charge_transitions),
            "active_nodes": self.active_nodes.tolist(),
            "kinetic_denominator_s1": self.kinetic_denominator_s1.tolist(),
            "occupancy": self.occupancy.tolist(),
            "occupied_density_m3": self.occupied_density_m3.tolist(),
            "charge_density_C_m3": self.charge_density_C_m3.tolist(),
            "recombination_rate_m3_s": self.recombination_rate_m3_s.tolist(),
            "recombination_derivative_n_s1": (
                self.recombination_derivative_n_s1.tolist()
            ),
            "recombination_derivative_p_s1": (
                self.recombination_derivative_p_s1.tolist()
            ),
            "charge_derivative_fixed_qf_C_m3_V": (
                self.charge_derivative_fixed_qf_C_m3_V.tolist()
            ),
            "total_charge_density_C_m3": self.total_charge_density_C_m3.tolist(),
            "total_recombination_rate_m3_s": (
                self.total_recombination_rate_m3_s.tolist()
            ),
            "total_recombination_derivative_n_s1": (
                self.total_recombination_derivative_n_s1.tolist()
            ),
            "total_recombination_derivative_p_s1": (
                self.total_recombination_derivative_p_s1.tolist()
            ),
            "total_charge_derivative_fixed_qf_C_m3_V": (
                self.total_charge_derivative_fixed_qf_C_m3_V.tolist()
            ),
            "minimum_occupancy": self.minimum_occupancy,
            "maximum_occupancy": self.maximum_occupancy,
            "minimum_kinetic_denominator_s1": (
                self.minimum_kinetic_denominator_s1
            ),
        }
        if self.distribution_kinds:
            payload["distribution_kinds"] = list(self.distribution_kinds)
            payload["source_energy_orders"] = list(self.source_energy_orders)
            payload["source_node_identifiers"] = [
                list(values) for values in self.source_node_identifiers
            ]
        if self.spatial_profile_sha256s:
            payload["spatial_closure"] = "layer-density-profile-v1"
            payload["spatial_profile_sha256s"] = list(
                self.spatial_profile_sha256s
            )
            payload["source_density_multiplier"] = (
                self.source_density_multiplier.tolist()
            )
            payload["minimum_density_multipliers"] = list(
                self.minimum_density_multipliers
            )
            payload["maximum_density_multipliers"] = list(
                self.maximum_density_multipliers
            )
        return payload


def _validate_source_species(
    species: Sequence[BulkDefectSpecies],
    *,
    band_gap_eV: float,
    allow_spatial_profiles: bool = False,
) -> tuple[BulkDefectSpecies, ...]:
    resolved = tuple(species)
    if not resolved:
        raise ValueError("species must not be empty")
    if not all(isinstance(item, BulkDefectSpecies) for item in resolved):
        raise TypeError("species must contain BulkDefectSpecies values")
    names = [item.name for item in resolved]
    if any(name is None for name in names) or len(names) != len(set(names)):
        raise MonovalentDefectClosureCapabilityError(
            "DEF-2 requires unique named defect species"
        )
    for item in resolved:
        item.validate_band_gap(band_gap_eV)
        if item.spatial_profile is not None and not allow_spatial_profiles:
            raise MonovalentDefectClosureCapabilityError(
                "spatial profiles require the compiled v3 device closure"
            )
        if item.charge_transition not in {NEUTRAL, ACCEPTOR, DONOR}:
            raise MonovalentDefectClosureCapabilityError(
                "DEF-2 requires neutral, acceptor, or donor transitions"
            )
        if item.degeneracy != 1.0:
            raise MonovalentDefectClosureCapabilityError(
                "DEF-2 does not silently ignore degeneracy; use degeneracy=1.0"
            )
        kinetics = item.kinetics
        if kinetics.sigma_n_m2 == 0.0 and kinetics.sigma_p_m2 == 0.0:
            raise MonovalentDefectClosureCapabilityError(
                "DEF-2 occupancy is undefined when both capture legs are "
                f"zero: {item.name}"
            )
        if (
            item.distribution.kind != SINGLE_LEVEL
            and not (
                item.distributed_explicit_ready
                or (allow_spatial_profiles and item.spatial_explicit_ready)
            )
        ):
            raise MonovalentDefectClosureCapabilityError(
                "distributed production requires a complete normalized v2 "
                f"source: {item.name}"
            )
    return resolved


def _validate_species(
    species: Sequence[BulkDefectSpecies],
    *,
    band_gap_eV: float,
) -> tuple[BulkDefectSpecies, ...]:
    resolved = tuple(species)
    if not resolved:
        raise ValueError("species must not be empty")
    if not all(isinstance(item, BulkDefectSpecies) for item in resolved):
        raise TypeError("species must contain BulkDefectSpecies values")
    names = [item.name for item in resolved]
    if any(name is None for name in names) or len(names) != len(set(names)):
        raise MonovalentDefectClosureCapabilityError(
            "DEF-2 requires unique named defect species"
        )
    for item in resolved:
        item.validate_band_gap(band_gap_eV)
        if item.distribution.kind != SINGLE_LEVEL:
            raise MonovalentDefectClosureCapabilityError(
                "DEF-2 supports single-level species only"
            )
    return _validate_source_species(resolved, band_gap_eV=band_gap_eV)


def _closure_identity(
    species: tuple[BulkDefectSpecies, ...],
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> str:
    payload = {
        "closure": MONOVALENT_DEFECT_CLOSURE_VERSION,
        "statistics": "maxwell_boltzmann",
        "band_gap_eV": band_gap_eV,
        "effective_conduction_dos_m3": effective_conduction_dos_m3,
        "effective_valence_dos_m3": effective_valence_dos_m3,
        "temperature_K": temperature_K,
        "species": [item.to_dict() for item in species],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def evaluate_monovalent_defect_closure(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    species: Sequence[BulkDefectSpecies],
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> MonovalentDefectClosureResult:
    """Evaluate exact local occupancy, SRH recombination, charge, and tangents.

    The potential-direction derivatives hold the electron and hole
    quasi-Fermi levels fixed, so ``dn/dphi=n/V_T`` and ``dp/dphi=-p/V_T``.
    There is no explicit trap-potential derivative at fixed carrier density
    because the level energy is referenced to the local band edge.
    """

    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if (
        not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n < 0.0)
        or np.any(p < 0.0)
    ):
        raise ValueError("defect closure carrier densities must be finite and non-negative")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    temperature = _finite_positive(temperature_K, "temperature_K")
    resolved = _validate_species(species, band_gap_eV=gap)
    thermal = thermal_voltage(temperature)
    energies = np.asarray(
        [item.distribution.center_eV_above_vb for item in resolved],
        dtype=float,
    )
    density = np.asarray(
        [item.distribution.total_density_m3 for item in resolved],
        dtype=float,
    )
    capture_n = np.asarray(
        [
            item.kinetics.sigma_n_m2
            * item.kinetics.thermal_velocity_n_m_s
            for item in resolved
        ],
        dtype=float,
    )
    capture_p = np.asarray(
        [
            item.kinetics.sigma_p_m2
            * item.kinetics.thermal_velocity_p_m_s
            for item in resolved
        ],
        dtype=float,
    )
    n1 = conduction_dos * np.exp(-(gap - energies) / thermal)
    p1 = valence_dos * np.exp(-energies / thermal)
    if not all(
        np.all(np.isfinite(value)) and np.all(value >= 0.0)
        for value in (capture_n, capture_p, n1, p1)
    ):
        raise FloatingPointError("defect closure references are non-finite")

    expansion = (len(resolved),) + (1,) * n.ndim
    density_e = density.reshape(expansion)
    capture_n_e = capture_n.reshape(expansion)
    capture_p_e = capture_p.reshape(expansion)
    n1_e = n1.reshape(expansion)
    p1_e = p1.reshape(expansion)
    n_e = np.expand_dims(n, axis=0)
    p_e = np.expand_dims(p, axis=0)
    filled_numerator = capture_n_e * n_e + capture_p_e * p1_e
    empty_numerator = capture_n_e * n1_e + capture_p_e * p_e
    denominator = filled_numerator + empty_numerator
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise FloatingPointError("defect closure kinetic denominator is invalid")

    occupancy = filled_numerator / denominator
    one_minus_occupancy = 1.0 - occupancy
    occupancy_derivative_n = capture_n_e * one_minus_occupancy / denominator
    occupancy_derivative_p = -capture_p_e * occupancy / denominator
    occupied_density = density_e * occupancy

    intrinsic_product = conduction_dos * valence_dos * math.exp(-gap / thermal)
    excess_product = n_e * p_e - intrinsic_product
    rate_prefactor = density_e * capture_n_e * capture_p_e
    rate = rate_prefactor * excess_product / denominator
    derivative_n = (
        rate_prefactor
        * (p_e * denominator - excess_product * capture_n_e)
        / denominator**2
    )
    derivative_p = (
        rate_prefactor
        * (n_e * denominator - excess_product * capture_p_e)
        / denominator**2
    )

    signed_charge_number = np.zeros_like(occupancy)
    charged = np.zeros(len(resolved), dtype=bool)
    for index, item in enumerate(resolved):
        if item.charge_transition == ACCEPTOR:
            signed_charge_number[index] = -occupied_density[index]
            charged[index] = True
        elif item.charge_transition == DONOR:
            signed_charge_number[index] = density[index] - occupied_density[index]
            charged[index] = True
    charge_factor = (-Q * density * charged).reshape(expansion)
    charge_derivative_n = charge_factor * occupancy_derivative_n
    charge_derivative_p = charge_factor * occupancy_derivative_p
    charge_density = Q * signed_charge_number
    charge_derivative_fixed_qf = (
        charge_derivative_n * n_e - charge_derivative_p * p_e
    ) / thermal
    rate_derivative_fixed_qf = (
        derivative_n * n_e - derivative_p * p_e
    ) / thermal

    finite_outputs = (
        occupancy,
        occupancy_derivative_n,
        occupancy_derivative_p,
        occupied_density,
        signed_charge_number,
        charge_density,
        rate,
        derivative_n,
        derivative_p,
        charge_derivative_n,
        charge_derivative_p,
        charge_derivative_fixed_qf,
        rate_derivative_fixed_qf,
    )
    if not all(np.all(np.isfinite(value)) for value in finite_outputs):
        raise FloatingPointError("defect closure produced non-finite output")
    minimum = float(np.min(occupancy))
    maximum = float(np.max(occupancy))
    if minimum < 0.0 or maximum > 1.0:
        raise FloatingPointError("defect occupancy left [0, 1] without clipping")

    return MonovalentDefectClosureResult(
        closure_identity_sha256=_closure_identity(
            resolved,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
        ),
        temperature_K=temperature,
        thermal_voltage_V=thermal,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        intrinsic_product_m6=intrinsic_product,
        species_identifiers=tuple(str(item.name) for item in resolved),
        charge_transitions=tuple(item.charge_transition for item in resolved),
        n1_m3=n1,
        p1_m3=p1,
        capture_n_m3_s=capture_n,
        capture_p_m3_s=capture_p,
        kinetic_denominator_s1=denominator,
        occupancy=occupancy,
        occupancy_derivative_n_m3=occupancy_derivative_n,
        occupancy_derivative_p_m3=occupancy_derivative_p,
        occupied_density_m3=occupied_density,
        signed_charge_number_density_m3=signed_charge_number,
        charge_density_C_m3=charge_density,
        recombination_rate_m3_s=rate,
        recombination_derivative_n_s1=derivative_n,
        recombination_derivative_p_s1=derivative_p,
        charge_derivative_n_C=charge_derivative_n,
        charge_derivative_p_C=charge_derivative_p,
        charge_derivative_fixed_qf_C_m3_V=charge_derivative_fixed_qf,
        recombination_derivative_fixed_qf_m3_s_V=rate_derivative_fixed_qf,
        total_charge_density_C_m3=np.sum(charge_density, axis=0),
        total_recombination_rate_m3_s=np.sum(rate, axis=0),
        total_recombination_derivative_n_s1=np.sum(derivative_n, axis=0),
        total_recombination_derivative_p_s1=np.sum(derivative_p, axis=0),
        total_charge_derivative_n_C=np.sum(charge_derivative_n, axis=0),
        total_charge_derivative_p_C=np.sum(charge_derivative_p, axis=0),
        total_charge_derivative_fixed_qf_C_m3_V=np.sum(
            charge_derivative_fixed_qf,
            axis=0,
        ),
        total_recombination_derivative_fixed_qf_m3_s_V=np.sum(
            rate_derivative_fixed_qf,
            axis=0,
        ),
        minimum_occupancy=minimum,
        maximum_occupancy=maximum,
    )


def evaluate_monovalent_source_defect_closure(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    species: Sequence[BulkDefectSpecies],
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    energy_expansions: Sequence[DefectSpeciesEnergyExpansion] | None = None,
) -> (
    MonovalentDefectClosureResult
    | "EnergyDistributedDefectClosureResult"
):
    """Evaluate canonical source species, preserving the exact single lane."""

    resolved = _validate_source_species(
        species,
        band_gap_eV=_finite_positive(band_gap_eV, "band_gap_eV"),
    )
    order = validate_defect_energy_quadrature_order(
        energy_quadrature_order
    )
    if all(item.distribution.kind == SINGLE_LEVEL for item in resolved):
        return evaluate_monovalent_defect_closure(
            electron_density_m3,
            hole_density_m3,
            resolved,
            band_gap_eV=band_gap_eV,
            effective_conduction_dos_m3=effective_conduction_dos_m3,
            effective_valence_dos_m3=effective_valence_dos_m3,
            temperature_K=temperature_K,
        )
    from perovskite_sim.physics.distributed_defect_closure import (
        evaluate_energy_distributed_defect_closure,
    )

    return evaluate_energy_distributed_defect_closure(
        electron_density_m3,
        hole_density_m3,
        resolved,
        band_gap_eV=band_gap_eV,
        effective_conduction_dos_m3=effective_conduction_dos_m3,
        effective_valence_dos_m3=effective_valence_dos_m3,
        temperature_K=temperature_K,
        energy_quadrature_order=order,
        energy_expansions=energy_expansions,
    )


@dataclass(frozen=True, slots=True)
class _SpatialDefectRegionEvaluation:
    kinetic_denominator_s1: np.ndarray
    occupancy: np.ndarray
    occupied_density_m3: np.ndarray
    charge_density_C_m3: np.ndarray
    recombination_rate_m3_s: np.ndarray
    recombination_derivative_n_s1: np.ndarray
    recombination_derivative_p_s1: np.ndarray
    charge_derivative_fixed_qf_C_m3_V: np.ndarray
    minimum_occupancy: float
    maximum_occupancy: float
    minimum_kinetic_denominator_s1: float


def _evaluate_spatial_defect_region(
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    region: MonovalentDefectRegion,
) -> _SpatialDefectRegionEvaluation:
    """Evaluate a v3 region across energy and physical position in one pass."""

    if not region.has_spatial_profiles:
        raise ValueError("spatial region evaluation requires a v3 region")
    n = np.asarray(electron_density_m3, dtype=float)
    p = np.asarray(hole_density_m3, dtype=float)
    local_gap = np.asarray(region.local_band_gap_eV, dtype=float)
    local_nc = np.asarray(region.local_effective_conduction_dos_m3, dtype=float)
    local_nv = np.asarray(region.local_effective_valence_dos_m3, dtype=float)
    multipliers = np.asarray(region.source_density_multipliers, dtype=float)
    if (
        n.shape != local_gap.shape
        or p.shape != local_gap.shape
        or not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n < 0.0)
        or np.any(p < 0.0)
    ):
        raise ValueError(
            "spatial defect closure carrier densities must be finite, "
            "non-negative, and match the region"
        )

    source_count = len(region.species)
    local_count = n.size
    shape = (source_count, local_count)
    kinetic_denominator = np.empty(shape, dtype=float)
    mean_occupancy = np.empty(shape, dtype=float)
    occupied_density = np.empty(shape, dtype=float)
    charge_density = np.empty(shape, dtype=float)
    recombination_rate = np.empty(shape, dtype=float)
    derivative_n = np.empty(shape, dtype=float)
    derivative_p = np.empty(shape, dtype=float)
    charge_derivative_fixed_qf = np.empty(shape, dtype=float)
    thermal = thermal_voltage(region.temperature_K)
    intrinsic_product = local_nc * local_nv * np.exp(-local_gap / thermal)
    minimum_occupancies: list[float] = []
    maximum_occupancies: list[float] = []

    for source_index, (source, quadrature) in enumerate(
        zip(region.species, region.source_quadratures, strict=True)
    ):
        energies = np.asarray(
            quadrature.energy_levels_eV_above_vb,
            dtype=float,
        )[:, None]
        base_density = np.asarray(
            quadrature.density_weights_m3,
            dtype=float,
        )[:, None]
        density = base_density * multipliers[source_index][None, :]
        capture_n = (
            float(source.kinetics.sigma_n_m2)
            * float(source.kinetics.thermal_velocity_n_m_s)
        )
        capture_p = (
            float(source.kinetics.sigma_p_m2)
            * float(source.kinetics.thermal_velocity_p_m_s)
        )
        n1 = local_nc[None, :] * np.exp(
            -(local_gap[None, :] - energies) / thermal
        )
        p1 = local_nv[None, :] * np.exp(-energies / thermal)
        n_e = n[None, :]
        p_e = p[None, :]
        filled_numerator = capture_n * n_e + capture_p * p1
        empty_numerator = capture_n * n1 + capture_p * p_e
        denominator = filled_numerator + empty_numerator
        if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
            raise FloatingPointError(
                f"spatial defect kinetic denominator is invalid: {source.name}"
            )
        node_occupancy = filled_numerator / denominator
        occupancy_derivative_n = (
            capture_n * (1.0 - node_occupancy) / denominator
        )
        occupancy_derivative_p = (
            -capture_p * node_occupancy / denominator
        )
        node_occupied_density = density * node_occupancy
        excess_product = n_e * p_e - intrinsic_product[None, :]
        rate_prefactor = density * capture_n * capture_p
        node_rate = rate_prefactor * excess_product / denominator
        node_derivative_n = (
            rate_prefactor
            * (p_e * denominator - excess_product * capture_n)
            / denominator**2
        )
        node_derivative_p = (
            rate_prefactor
            * (n_e * denominator - excess_product * capture_p)
            / denominator**2
        )
        if source.charge_transition == ACCEPTOR:
            signed_charge_number = -node_occupied_density
            charge_factor = -Q * density
        elif source.charge_transition == DONOR:
            signed_charge_number = density - node_occupied_density
            charge_factor = -Q * density
        else:
            signed_charge_number = np.zeros_like(node_occupied_density)
            charge_factor = np.zeros_like(density)
        node_charge = Q * signed_charge_number
        node_charge_derivative_n = charge_factor * occupancy_derivative_n
        node_charge_derivative_p = charge_factor * occupancy_derivative_p
        node_charge_derivative_fixed_qf = (
            node_charge_derivative_n * n_e
            - node_charge_derivative_p * p_e
        ) / thermal

        def integrate(values: np.ndarray) -> np.ndarray:
            if quadrature.order == 1:
                return values[0]
            return np.sum(values, axis=0)

        kinetic_denominator[source_index] = np.min(denominator, axis=0)
        integrated_occupied = integrate(node_occupied_density)
        mean_occupancy[source_index] = (
            node_occupancy[0]
            if quadrature.order == 1
            else integrated_occupied
            / (
                float(source.distribution.total_density_m3)
                * multipliers[source_index]
            )
        )
        occupied_density[source_index] = integrated_occupied
        charge_density[source_index] = integrate(node_charge)
        recombination_rate[source_index] = integrate(node_rate)
        derivative_n[source_index] = integrate(node_derivative_n)
        derivative_p[source_index] = integrate(node_derivative_p)
        charge_derivative_fixed_qf[source_index] = integrate(
            node_charge_derivative_fixed_qf
        )
        minimum_occupancies.append(float(np.min(node_occupancy)))
        maximum_occupancies.append(float(np.max(node_occupancy)))

    finite_outputs = (
        kinetic_denominator,
        mean_occupancy,
        occupied_density,
        charge_density,
        recombination_rate,
        derivative_n,
        derivative_p,
        charge_derivative_fixed_qf,
    )
    if not all(np.all(np.isfinite(value)) for value in finite_outputs):
        raise FloatingPointError("spatial defect closure produced non-finite output")
    minimum = min(minimum_occupancies)
    maximum = max(maximum_occupancies)
    if minimum < 0.0 or maximum > 1.0:
        raise FloatingPointError(
            "spatial defect occupancy left [0, 1] without clipping"
        )
    return _SpatialDefectRegionEvaluation(
        kinetic_denominator_s1=kinetic_denominator,
        occupancy=mean_occupancy,
        occupied_density_m3=occupied_density,
        charge_density_C_m3=charge_density,
        recombination_rate_m3_s=recombination_rate,
        recombination_derivative_n_s1=derivative_n,
        recombination_derivative_p_s1=derivative_p,
        charge_derivative_fixed_qf_C_m3_V=charge_derivative_fixed_qf,
        minimum_occupancy=minimum,
        maximum_occupancy=maximum,
        minimum_kinetic_denominator_s1=float(np.min(kinetic_denominator)),
    )


def evaluate_monovalent_bulk_defects(
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    model: MonovalentBulkDefectModel,
) -> MonovalentBulkDefectEvaluation:
    """Evaluate one compiled device model on its full electrical grid."""

    if not isinstance(model, MonovalentBulkDefectModel):
        raise TypeError("model must be a MonovalentBulkDefectModel")
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if n.shape != (model.node_count,):
        raise ValueError("monovalent bulk-defect state must match the compiled grid")
    species_count = len(model.species_identifiers)
    shape = (species_count, model.node_count)
    active_nodes = np.zeros(shape, dtype=bool)
    kinetic_denominator = np.zeros(shape, dtype=float)
    occupancy = np.zeros(shape, dtype=float)
    occupied_density = np.zeros(shape, dtype=float)
    charge_density = np.zeros(shape, dtype=float)
    recombination_rate = np.zeros(shape, dtype=float)
    recombination_derivative_n = np.zeros(shape, dtype=float)
    recombination_derivative_p = np.zeros(shape, dtype=float)
    charge_derivative_fixed_qf = np.zeros(shape, dtype=float)
    source_density_multiplier = (
        np.zeros(shape, dtype=float) if model.has_spatial_profiles else None
    )
    total_charge_density = np.zeros(model.node_count, dtype=float)
    total_recombination_rate = np.zeros(model.node_count, dtype=float)
    total_recombination_derivative_n = np.zeros(
        model.node_count,
        dtype=float,
    )
    total_recombination_derivative_p = np.zeros(
        model.node_count,
        dtype=float,
    )
    total_charge_derivative_fixed_qf = np.zeros(
        model.node_count,
        dtype=float,
    )
    minimum_occupancies: list[float] = []
    maximum_occupancies: list[float] = []
    minimum_denominators: list[float] = []
    offset = 0
    for region in model.regions:
        mask = region.active_nodes
        if region.has_spatial_profiles:
            local = _evaluate_spatial_defect_region(n[mask], p[mask], region)
        else:
            local = evaluate_monovalent_source_defect_closure(
                n[mask],
                p[mask],
                region.species,
                band_gap_eV=region.band_gap_eV,
                effective_conduction_dos_m3=(
                    region.effective_conduction_dos_m3
                ),
                effective_valence_dos_m3=region.effective_valence_dos_m3,
                temperature_K=region.temperature_K,
                energy_quadrature_order=region.energy_quadrature_order,
                energy_expansions=region.source_expansions,
            )
        count = len(region.species)
        rows = slice(offset, offset + count)
        active_nodes[rows, mask] = True
        if source_density_multiplier is not None:
            source_density_multiplier[rows, mask] = (
                region.source_density_multipliers
                if region.has_spatial_profiles
                else 1.0
            )
        if isinstance(local, _SpatialDefectRegionEvaluation):
            kinetic_denominator[rows, mask] = local.kinetic_denominator_s1
            occupancy[rows, mask] = local.occupancy
            occupied_density[rows, mask] = local.occupied_density_m3
            charge_density[rows, mask] = local.charge_density_C_m3
            recombination_rate[rows, mask] = local.recombination_rate_m3_s
            recombination_derivative_n[rows, mask] = (
                local.recombination_derivative_n_s1
            )
            recombination_derivative_p[rows, mask] = (
                local.recombination_derivative_p_s1
            )
            charge_derivative_fixed_qf[rows, mask] = (
                local.charge_derivative_fixed_qf_C_m3_V
            )
        elif isinstance(local, MonovalentDefectClosureResult):
            kinetic_denominator[rows, mask] = local.kinetic_denominator_s1
            occupancy[rows, mask] = local.occupancy
            occupied_density[rows, mask] = local.occupied_density_m3
            charge_density[rows, mask] = local.charge_density_C_m3
            recombination_rate[rows, mask] = local.recombination_rate_m3_s
            recombination_derivative_n[rows, mask] = (
                local.recombination_derivative_n_s1
            )
            recombination_derivative_p[rows, mask] = (
                local.recombination_derivative_p_s1
            )
            charge_derivative_fixed_qf[rows, mask] = (
                local.charge_derivative_fixed_qf_C_m3_V
            )
        else:
            for source_index, source in enumerate(local.source_closures):
                row = offset + source_index
                kinetic_denominator[row, mask] = np.min(
                    source.node_closure.kinetic_denominator_s1,
                    axis=0,
                )
                occupancy[row, mask] = source.mean_occupancy
                occupied_density[row, mask] = source.occupied_density_m3
                charge_density[row, mask] = source.charge_density_C_m3
                recombination_rate[row, mask] = (
                    source.recombination_rate_m3_s
                )
                recombination_derivative_n[row, mask] = (
                    source.recombination_derivative_n_s1
                )
                recombination_derivative_p[row, mask] = (
                    source.recombination_derivative_p_s1
                )
                charge_derivative_fixed_qf[row, mask] = (
                    source.charge_derivative_fixed_qf_C_m3_V
                )
        if isinstance(local, _SpatialDefectRegionEvaluation):
            total_charge_density[mask] = np.sum(
                local.charge_density_C_m3,
                axis=0,
            )
            total_recombination_rate[mask] = np.sum(
                local.recombination_rate_m3_s,
                axis=0,
            )
            total_recombination_derivative_n[mask] = np.sum(
                local.recombination_derivative_n_s1,
                axis=0,
            )
            total_recombination_derivative_p[mask] = np.sum(
                local.recombination_derivative_p_s1,
                axis=0,
            )
            total_charge_derivative_fixed_qf[mask] = np.sum(
                local.charge_derivative_fixed_qf_C_m3_V,
                axis=0,
            )
        else:
            total_charge_density[mask] = local.total_charge_density_C_m3
            total_recombination_rate[mask] = (
                local.total_recombination_rate_m3_s
            )
            total_recombination_derivative_n[mask] = (
                local.total_recombination_derivative_n_s1
            )
            total_recombination_derivative_p[mask] = (
                local.total_recombination_derivative_p_s1
            )
            total_charge_derivative_fixed_qf[mask] = (
                local.total_charge_derivative_fixed_qf_C_m3_V
            )
        minimum_occupancies.append(local.minimum_occupancy)
        maximum_occupancies.append(local.maximum_occupancy)
        if isinstance(local, MonovalentDefectClosureResult):
            minimum_denominators.append(
                float(np.min(local.kinetic_denominator_s1))
            )
        else:
            minimum_denominators.append(
                local.minimum_kinetic_denominator_s1
            )
        offset += count
    distributed_metadata = model.has_distributed_species
    return MonovalentBulkDefectEvaluation(
        model_identity_sha256=model.identity_sha256,
        species_identifiers=model.species_identifiers,
        charge_transitions=model.charge_transitions,
        active_nodes=active_nodes,
        kinetic_denominator_s1=kinetic_denominator,
        occupancy=occupancy,
        occupied_density_m3=occupied_density,
        charge_density_C_m3=charge_density,
        recombination_rate_m3_s=recombination_rate,
        recombination_derivative_n_s1=recombination_derivative_n,
        recombination_derivative_p_s1=recombination_derivative_p,
        charge_derivative_fixed_qf_C_m3_V=charge_derivative_fixed_qf,
        total_charge_density_C_m3=total_charge_density,
        total_recombination_rate_m3_s=total_recombination_rate,
        total_recombination_derivative_n_s1=(
            total_recombination_derivative_n
        ),
        total_recombination_derivative_p_s1=(
            total_recombination_derivative_p
        ),
        total_charge_derivative_fixed_qf_C_m3_V=(
            total_charge_derivative_fixed_qf
        ),
        minimum_occupancy=min(minimum_occupancies),
        maximum_occupancy=max(maximum_occupancies),
        minimum_kinetic_denominator_s1=min(minimum_denominators),
        distribution_kinds=(
            model.distribution_kinds if distributed_metadata else ()
        ),
        source_energy_orders=(
            model.source_energy_orders if distributed_metadata else ()
        ),
        source_node_identifiers=(
            model.source_node_identifiers if distributed_metadata else ()
        ),
        source_density_multiplier=source_density_multiplier,
        spatial_profile_sha256s=(
            model.spatial_profile_sha256s if model.has_spatial_profiles else ()
        ),
        minimum_density_multipliers=(
            tuple(item[0] for item in model.source_density_multiplier_bounds)
            if model.has_spatial_profiles
            else ()
        ),
        maximum_density_multipliers=(
            tuple(item[1] for item in model.source_density_multiplier_bounds)
            if model.has_spatial_profiles
            else ()
        ),
    )


def solve_monovalent_defect_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    acceptor_density_m3: float,
    donor_density_m3: float,
    species: Sequence[BulkDefectSpecies],
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    energy_expansions: Sequence[DefectSpeciesEnergyExpansion] | None = None,
) -> MonovalentDefectNeutralityResult:
    """Solve ``p-n+N_D-N_A+sum(N_defect_charge)=0`` in the MB limit."""

    from perovskite_sim.physics.statistics import (
        FULLY_IONIZED,
        MAXWELL_BOLTZMANN,
        BulkChargeNeutralityState,
        carrier_density_from_reduced_fermi_level,
    )

    temperature = _finite_positive(temperature_K, "temperature_K")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    acceptors = float(acceptor_density_m3)
    donors = float(donor_density_m3)
    if (
        not math.isfinite(acceptors)
        or acceptors < 0.0
        or not math.isfinite(donors)
        or donors < 0.0
    ):
        raise ValueError("contact dopant densities must be finite and non-negative")
    resolved = _validate_source_species(species, band_gap_eV=gap)
    energy_order = validate_defect_energy_quadrature_order(
        energy_quadrature_order
    )
    if energy_expansions is None and any(
        item.distribution.kind != SINGLE_LEVEL for item in resolved
    ):
        prepared_expansions: tuple[DefectSpeciesEnergyExpansion, ...] | None = tuple(
            expand_bulk_defect_species_energy(
                item,
                band_gap_eV=gap,
                order=energy_order,
            )
            for item in resolved
        )
    elif energy_expansions is None:
        prepared_expansions = None
    else:
        prepared_expansions = tuple(energy_expansions)
    thermal = thermal_voltage(temperature)
    reduced_gap = gap / thermal

    def evaluate(
        eta_n: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
        MonovalentDefectClosureResult | EnergyDistributedDefectClosureResult,
    ]:
        eta_p = -reduced_gap - eta_n
        electron = carrier_density_from_reduced_fermi_level(
            eta_n,
            conduction_dos,
            statistics=MAXWELL_BOLTZMANN,
        )
        hole = carrier_density_from_reduced_fermi_level(
            eta_p,
            valence_dos,
            statistics=MAXWELL_BOLTZMANN,
        )
        closure = evaluate_monovalent_source_defect_closure(
            electron,
            hole,
            resolved,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
            energy_quadrature_order=energy_order,
            energy_expansions=prepared_expansions,
        )
        if isinstance(closure, MonovalentDefectClosureResult):
            defect_charge_number = float(
                np.sum(closure.signed_charge_number_density_m3)
            )
        else:
            defect_charge_number = float(
                np.asarray(closure.total_charge_density_C_m3) / Q
            )
        residual = hole - electron + donors - acceptors + defect_charge_number
        return residual, electron, hole, eta_p, closure

    intrinsic_center = 0.5 * (
        -reduced_gap + math.log(valence_dos / conduction_dos)
    )
    half_width = max(32.0, 0.5 * reduced_gap + 8.0)
    lower = intrinsic_center - half_width
    upper = intrinsic_center + half_width
    for _ in range(32):
        lower_residual = evaluate(lower)[0]
        upper_residual = evaluate(upper)[0]
        if lower_residual >= 0.0 and upper_residual <= 0.0:
            break
        half_width *= 2.0
        lower = intrinsic_center - half_width
        upper = intrinsic_center + half_width
    else:
        raise RuntimeError("could not bracket explicit-defect charge neutrality")

    for _ in range(220):
        midpoint = 0.5 * (lower + upper)
        residual = evaluate(midpoint)[0]
        if residual > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= max(2.0e-14, 4.0 * math.ulp(midpoint)):
            break

    eta_n = 0.5 * (lower + upper)
    residual, electron, hole, eta_p, closure = evaluate(eta_n)
    scale = max(
        electron,
        hole,
        donors,
        acceptors,
        *(item.distribution.total_density_m3 for item in resolved),
        1.0,
    )
    normalized = abs(residual) / scale
    if not math.isfinite(normalized) or normalized > 1.0e-12:
        raise RuntimeError(
            "explicit-defect charge-neutrality residual exceeded gate"
        )
    neutrality = BulkChargeNeutralityState(
        statistics=MAXWELL_BOLTZMANN,
        temperature_K=temperature,
        thermal_voltage_V=thermal,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        acceptor_density_m3=acceptors,
        donor_density_m3=donors,
        reduced_electron_fermi_level=eta_n,
        reduced_hole_fermi_level=eta_p,
        electron_density_m3=electron,
        hole_density_m3=hole,
        normalized_charge_residual=normalized,
        dopant_ionization_model=FULLY_IONIZED,
        ionized_acceptor_density_m3=acceptors,
        ionized_donor_density_m3=donors,
        acceptor_ionized_fraction=1.0,
        donor_ionized_fraction=1.0,
    )
    return MonovalentDefectNeutralityResult(
        neutrality=neutrality,
        closure=closure,
    )


__all__ = [
    "MONOVALENT_BULK_DEFECT_MODEL_VERSION",
    "MONOVALENT_DEFECT_CLOSURE_VERSION",
    "MonovalentBulkDefectEvaluation",
    "MonovalentBulkDefectModel",
    "MonovalentDefectClosureCapabilityError",
    "MonovalentDefectNeutralityResult",
    "MonovalentDefectClosureResult",
    "MonovalentDefectRegion",
    "evaluate_monovalent_bulk_defects",
    "evaluate_monovalent_defect_closure",
    "evaluate_monovalent_source_defect_closure",
    "solve_monovalent_defect_charge_neutrality",
]
