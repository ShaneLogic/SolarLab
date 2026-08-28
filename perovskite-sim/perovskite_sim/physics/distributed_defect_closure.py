"""Local quasi-steady closure for energy-distributed bulk defects.

Each canonical source species is expanded into normalized single-level energy
nodes and evaluated by the D2 monovalent closure. This module only aggregates
that exact primitive. It does not connect distributed defects to contacts,
Poisson, continuity, or a production experiment path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from perovskite_sim.models.defects import (
    CONDUCTION_BAND_TAIL,
    GAUSSIAN,
    SINGLE_LEVEL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.physics.defect_closure import (
    MonovalentDefectClosureCapabilityError,
    MonovalentDefectClosureResult,
    evaluate_monovalent_defect_closure,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    DefectEnergyQuadrature,
    DefectSpeciesEnergyExpansion,
    expand_bulk_defect_species_energy,
    validate_defect_energy_quadrature_order,
)


ENERGY_DISTRIBUTED_DEFECT_CLOSURE_VERSION = (
    "monovalent-energy-distributed-mb-v1"
)
_LOCAL_DISTRIBUTIONS = frozenset(
    {
        SINGLE_LEVEL,
        GAUSSIAN,
        UNIFORM,
        CONDUCTION_BAND_TAIL,
        VALENCE_BAND_TAIL,
    }
)


class EnergyDistributedDefectClosureCapabilityError(
    ExplicitDefectCapabilityError
):
    """A valid defect input requested unsupported local defect physics."""


class EnergyDistributedDefectClosureError(RuntimeError):
    """A named source species failed during energy-resolved evaluation."""


def _readonly(value: object) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


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


_INTEGRATED_ARRAY_FIELDS = (
    "mean_occupancy",
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
_NODE_FIELD_BY_INTEGRATED_FIELD = {
    "occupied_density_m3": "occupied_density_m3",
    "signed_charge_number_density_m3": "signed_charge_number_density_m3",
    "charge_density_C_m3": "charge_density_C_m3",
    "recombination_rate_m3_s": "recombination_rate_m3_s",
    "recombination_derivative_n_s1": "recombination_derivative_n_s1",
    "recombination_derivative_p_s1": "recombination_derivative_p_s1",
    "charge_derivative_n_C": "charge_derivative_n_C",
    "charge_derivative_p_C": "charge_derivative_p_C",
    "charge_derivative_fixed_qf_C_m3_V": (
        "charge_derivative_fixed_qf_C_m3_V"
    ),
    "recombination_derivative_fixed_qf_m3_s_V": (
        "recombination_derivative_fixed_qf_m3_s_V"
    ),
}


@dataclass(frozen=True, slots=True)
class EnergyResolvedSpeciesClosure:
    """Node-resolved evidence and integrated observables for one source."""

    source_species: BulkDefectSpecies
    quadrature: DefectEnergyQuadrature
    node_closure: MonovalentDefectClosureResult
    mean_occupancy: np.ndarray
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
    minimum_occupancy: float
    maximum_occupancy: float
    minimum_kinetic_denominator_s1: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_species, BulkDefectSpecies):
            raise TypeError("source_species must be a BulkDefectSpecies")
        if self.source_species.name is None:
            raise ValueError("energy-resolved source species must be named")
        if not isinstance(self.quadrature, DefectEnergyQuadrature):
            raise TypeError("quadrature must be a DefectEnergyQuadrature")
        if not isinstance(self.node_closure, MonovalentDefectClosureResult):
            raise TypeError("node_closure must be MonovalentDefectClosureResult")
        if self.quadrature.order != len(
            self.node_closure.species_identifiers
        ):
            raise ValueError("node closure does not match quadrature order")
        if (
            self.quadrature.distribution_kind
            != self.source_species.distribution.kind
            or self.quadrature.total_density_m3
            != self.source_species.distribution.total_density_m3
        ):
            raise ValueError("quadrature does not match its source distribution")
        expected_shape = np.asarray(
            self.node_closure.total_recombination_rate_m3_s
        ).shape
        for name in _INTEGRATED_ARRAY_FIELDS:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} must be finite and match the carrier state shape"
                )
            object.__setattr__(self, name, _readonly(value))
        mean_occupancy = np.asarray(self.mean_occupancy)
        if np.any(mean_occupancy < 0.0) or np.any(mean_occupancy > 1.0):
            raise ValueError("integrated mean occupancy must lie in [0, 1]")
        minimum = float(self.minimum_occupancy)
        maximum = float(self.maximum_occupancy)
        minimum_denominator = float(self.minimum_kinetic_denominator_s1)
        if (
            minimum != self.node_closure.minimum_occupancy
            or maximum != self.node_closure.maximum_occupancy
            or minimum_denominator
            != float(np.min(self.node_closure.kinetic_denominator_s1))
            or not 0.0 <= minimum <= maximum <= 1.0
            or not math.isfinite(minimum_denominator)
            or minimum_denominator <= 0.0
        ):
            raise ValueError("energy-resolved source extrema are inconsistent")
        for integrated_field, node_field in (
            _NODE_FIELD_BY_INTEGRATED_FIELD.items()
        ):
            node_values = np.asarray(getattr(self.node_closure, node_field))
            expected = (
                node_values[0]
                if self.quadrature.order == 1
                else np.sum(node_values, axis=0)
            )
            if not np.array_equal(
                np.asarray(getattr(self, integrated_field)),
                expected,
            ):
                raise ValueError(
                    f"{integrated_field} is not the exact node aggregate"
                )
        expected_mean = (
            np.asarray(self.node_closure.occupancy)[0]
            if self.quadrature.order == 1
            else np.asarray(self.occupied_density_m3)
            / float(self.source_species.distribution.total_density_m3)
        )
        if not np.array_equal(np.asarray(self.mean_occupancy), expected_mean):
            raise ValueError("mean_occupancy is not the density-weighted mean")

    @property
    def source_identifier(self) -> str:
        return str(self.source_species.name)

    @property
    def distribution_kind(self) -> str:
        return self.source_species.distribution.kind

    def to_dict(self) -> dict[str, object]:
        return {
            "source_identifier": self.source_identifier,
            "source_species": self.source_species.to_dict(),
            "distribution_kind": self.distribution_kind,
            "quadrature": self.quadrature.to_dict(),
            "node_closure": self.node_closure.to_dict(),
            "mean_occupancy": self.mean_occupancy.tolist(),
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
            "minimum_occupancy": self.minimum_occupancy,
            "maximum_occupancy": self.maximum_occupancy,
            "minimum_kinetic_denominator_s1": (
                self.minimum_kinetic_denominator_s1
            ),
        }


@dataclass(frozen=True, slots=True)
class EnergyDistributedDefectClosureResult:
    """Integrated source-species and total local defect observables."""

    closure_identity_sha256: str
    temperature_K: float
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    source_closures: tuple[EnergyResolvedSpeciesClosure, ...]
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
    minimum_kinetic_denominator_s1: float

    def __post_init__(self) -> None:
        digest = str(self.closure_identity_sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("closure_identity_sha256 must be a SHA-256 hex")
        sources = tuple(self.source_closures)
        if not sources or not all(
            isinstance(item, EnergyResolvedSpeciesClosure) for item in sources
        ):
            raise ValueError("distributed closure requires source closures")
        identifiers = [item.source_identifier for item in sources]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("distributed closure source identifiers must be unique")
        for name in (
            "temperature_K",
            "band_gap_eV",
            "effective_conduction_dos_m3",
            "effective_valence_dos_m3",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name),
            )
        total_fields = (
            "total_charge_density_C_m3",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
            "total_charge_derivative_n_C",
            "total_charge_derivative_p_C",
            "total_charge_derivative_fixed_qf_C_m3_V",
            "total_recombination_derivative_fixed_qf_m3_s_V",
        )
        expected_shape = np.asarray(sources[0].mean_occupancy).shape
        for name in total_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} must be finite and match the carrier state shape"
                )
            object.__setattr__(self, name, _readonly(value))
            expected = _stable_source_sum(sources, name.removeprefix("total_"))
            if not np.array_equal(value, expected):
                raise ValueError(f"{name} is not the stable source sum")
        minimum = float(self.minimum_occupancy)
        maximum = float(self.maximum_occupancy)
        minimum_denominator = float(self.minimum_kinetic_denominator_s1)
        if (
            minimum != min(item.minimum_occupancy for item in sources)
            or maximum != max(item.maximum_occupancy for item in sources)
            or minimum_denominator
            != min(item.minimum_kinetic_denominator_s1 for item in sources)
            or not 0.0 <= minimum <= maximum <= 1.0
            or not math.isfinite(minimum_denominator)
            or minimum_denominator <= 0.0
        ):
            raise ValueError("distributed closure extrema are inconsistent")
        object.__setattr__(self, "closure_identity_sha256", digest)
        object.__setattr__(self, "source_closures", sources)
        expected_identity = _closure_identity(
            sources,
            temperature_K=self.temperature_K,
            band_gap_eV=self.band_gap_eV,
            effective_conduction_dos_m3=self.effective_conduction_dos_m3,
            effective_valence_dos_m3=self.effective_valence_dos_m3,
        )
        if digest != expected_identity:
            raise ValueError("distributed closure identity is inconsistent")

    @property
    def source_identifiers(self) -> tuple[str, ...]:
        return tuple(item.source_identifier for item in self.source_closures)

    @property
    def distribution_kinds(self) -> tuple[str, ...]:
        return tuple(item.distribution_kind for item in self.source_closures)

    @property
    def energy_orders(self) -> tuple[int, ...]:
        return tuple(item.quadrature.order for item in self.source_closures)

    def to_dict(self) -> dict[str, object]:
        return {
            "closure": ENERGY_DISTRIBUTED_DEFECT_CLOSURE_VERSION,
            "closure_identity_sha256": self.closure_identity_sha256,
            "statistics": "maxwell_boltzmann",
            "temperature_K": self.temperature_K,
            "band_gap_eV": self.band_gap_eV,
            "effective_conduction_dos_m3": (
                self.effective_conduction_dos_m3
            ),
            "effective_valence_dos_m3": self.effective_valence_dos_m3,
            "source_identifiers": list(self.source_identifiers),
            "distribution_kinds": list(self.distribution_kinds),
            "energy_orders": list(self.energy_orders),
            "source_closures": [
                item.to_dict() for item in self.source_closures
            ],
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
            "total_charge_derivative_n_C": (
                self.total_charge_derivative_n_C.tolist()
            ),
            "total_charge_derivative_p_C": (
                self.total_charge_derivative_p_C.tolist()
            ),
            "total_charge_derivative_fixed_qf_C_m3_V": (
                self.total_charge_derivative_fixed_qf_C_m3_V.tolist()
            ),
            "total_recombination_derivative_fixed_qf_m3_s_V": (
                self.total_recombination_derivative_fixed_qf_m3_s_V.tolist()
            ),
            "minimum_occupancy": self.minimum_occupancy,
            "maximum_occupancy": self.maximum_occupancy,
            "minimum_kinetic_denominator_s1": (
                self.minimum_kinetic_denominator_s1
            ),
        }


def _integrate_source(
    source: BulkDefectSpecies,
    node_closure: MonovalentDefectClosureResult,
    quadrature: DefectEnergyQuadrature,
) -> EnergyResolvedSpeciesClosure:
    def integrate(field: str) -> np.ndarray:
        values = np.asarray(getattr(node_closure, field))
        if quadrature.order == 1:
            return values[0]
        return np.sum(values, axis=0)

    occupied_density = integrate("occupied_density_m3")
    total_density = float(source.distribution.total_density_m3)
    mean_occupancy = (
        np.asarray(node_closure.occupancy)[0]
        if quadrature.order == 1
        else occupied_density / total_density
    )
    return EnergyResolvedSpeciesClosure(
        source_species=source,
        quadrature=quadrature,
        node_closure=node_closure,
        mean_occupancy=mean_occupancy,
        occupied_density_m3=occupied_density,
        signed_charge_number_density_m3=integrate(
            "signed_charge_number_density_m3"
        ),
        charge_density_C_m3=integrate("charge_density_C_m3"),
        recombination_rate_m3_s=integrate("recombination_rate_m3_s"),
        recombination_derivative_n_s1=integrate(
            "recombination_derivative_n_s1"
        ),
        recombination_derivative_p_s1=integrate(
            "recombination_derivative_p_s1"
        ),
        charge_derivative_n_C=integrate("charge_derivative_n_C"),
        charge_derivative_p_C=integrate("charge_derivative_p_C"),
        charge_derivative_fixed_qf_C_m3_V=integrate(
            "charge_derivative_fixed_qf_C_m3_V"
        ),
        recombination_derivative_fixed_qf_m3_s_V=integrate(
            "recombination_derivative_fixed_qf_m3_s_V"
        ),
        minimum_occupancy=node_closure.minimum_occupancy,
        maximum_occupancy=node_closure.maximum_occupancy,
        minimum_kinetic_denominator_s1=float(
            np.min(node_closure.kinetic_denominator_s1)
        ),
    )


def _stable_source_sum(
    sources: tuple[EnergyResolvedSpeciesClosure, ...],
    field: str,
) -> np.ndarray:
    ordered = sorted(sources, key=lambda item: item.source_identifier)
    if len(ordered) == 1:
        return np.asarray(getattr(ordered[0], field))
    return np.sum(
        np.stack([np.asarray(getattr(item, field)) for item in ordered]),
        axis=0,
    )


def _closure_identity(
    sources: tuple[EnergyResolvedSpeciesClosure, ...],
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
) -> str:
    payload = {
        "closure": ENERGY_DISTRIBUTED_DEFECT_CLOSURE_VERSION,
        "statistics": "maxwell_boltzmann",
        "temperature_K": temperature_K,
        "band_gap_eV": band_gap_eV,
        "effective_conduction_dos_m3": effective_conduction_dos_m3,
        "effective_valence_dos_m3": effective_valence_dos_m3,
        "sources": [
            {
                "source_species": item.source_species.to_dict(),
                "quadrature": item.quadrature.to_dict(),
                "node_closure_identity_sha256": (
                    item.node_closure.closure_identity_sha256
                ),
            }
            for item in sources
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


def evaluate_energy_distributed_defect_closure(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    species: Sequence[BulkDefectSpecies],
    *,
    band_gap_eV: object,
    effective_conduction_dos_m3: object,
    effective_valence_dos_m3: object,
    temperature_K: object,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    energy_expansions: Sequence[DefectSpeciesEnergyExpansion] | None = None,
) -> EnergyDistributedDefectClosureResult:
    """Evaluate and integrate D2 node closures for canonical distributions."""

    resolved = tuple(species)
    if not resolved:
        raise ValueError("distributed defect species must not be empty")
    if not all(isinstance(item, BulkDefectSpecies) for item in resolved):
        raise TypeError("species must contain BulkDefectSpecies values")
    identifiers = [item.name for item in resolved]
    if any(name is None for name in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise EnergyDistributedDefectClosureCapabilityError(
            "energy-distributed closure requires unique named source species"
        )
    unsupported = [
        f"{item.name}:{item.distribution.kind}"
        for item in resolved
        if item.distribution.kind not in _LOCAL_DISTRIBUTIONS
    ]
    if unsupported:
        raise EnergyDistributedDefectClosureCapabilityError(
            f"unsupported energy distributions: {unsupported}"
        )

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
    resolved_order = validate_defect_energy_quadrature_order(
        energy_quadrature_order
    )
    if energy_expansions is None:
        prepared_expansions: tuple[DefectSpeciesEnergyExpansion | None, ...] = (
            (None,) * len(resolved)
        )
    else:
        supplied = tuple(energy_expansions)
        if len(supplied) != len(resolved) or not all(
            isinstance(item, DefectSpeciesEnergyExpansion)
            for item in supplied
        ):
            raise ValueError(
                "energy_expansions must align with the source species"
            )
        for source, expansion in zip(resolved, supplied, strict=True):
            expected_order = (
                1
                if source.distribution.kind == SINGLE_LEVEL
                else resolved_order
            )
            if (
                expansion.source_species != source
                or expansion.quadrature.order != expected_order
            ):
                raise ValueError(
                    "energy_expansions do not match the source/order protocol"
                )
        prepared_expansions = supplied
    source_results: list[EnergyResolvedSpeciesClosure] = []
    for source, prepared in zip(
        resolved,
        prepared_expansions,
        strict=True,
    ):
        try:
            expansion = (
                expand_bulk_defect_species_energy(
                    source,
                    band_gap_eV=gap,
                    order=resolved_order,
                )
                if prepared is None
                else prepared
            )
            node_closure = evaluate_monovalent_defect_closure(
                electron_density_m3,
                hole_density_m3,
                expansion.node_species,
                band_gap_eV=gap,
                effective_conduction_dos_m3=conduction_dos,
                effective_valence_dos_m3=valence_dos,
                temperature_K=temperature,
            )
        except MonovalentDefectClosureCapabilityError as exc:
            raise EnergyDistributedDefectClosureCapabilityError(
                "energy-distributed source closure is unsupported for "
                f"{source.name!r} ({source.distribution.kind}): {exc}"
            ) from exc
        except (FloatingPointError, ValueError) as exc:
            raise EnergyDistributedDefectClosureError(
                "energy-distributed source evaluation failed for "
                f"{source.name!r} ({source.distribution.kind}, "
                f"requested_order={resolved_order}): {exc}"
            ) from exc
        source_results.append(
            _integrate_source(source, node_closure, expansion.quadrature)
        )
    sources = tuple(source_results)

    identity = _closure_identity(
        sources,
        temperature_K=temperature,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
    )
    return EnergyDistributedDefectClosureResult(
        closure_identity_sha256=identity,
        temperature_K=temperature,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        source_closures=sources,
        total_charge_density_C_m3=_stable_source_sum(
            sources,
            "charge_density_C_m3",
        ),
        total_recombination_rate_m3_s=_stable_source_sum(
            sources,
            "recombination_rate_m3_s",
        ),
        total_recombination_derivative_n_s1=_stable_source_sum(
            sources,
            "recombination_derivative_n_s1",
        ),
        total_recombination_derivative_p_s1=_stable_source_sum(
            sources,
            "recombination_derivative_p_s1",
        ),
        total_charge_derivative_n_C=_stable_source_sum(
            sources,
            "charge_derivative_n_C",
        ),
        total_charge_derivative_p_C=_stable_source_sum(
            sources,
            "charge_derivative_p_C",
        ),
        total_charge_derivative_fixed_qf_C_m3_V=_stable_source_sum(
            sources,
            "charge_derivative_fixed_qf_C_m3_V",
        ),
        total_recombination_derivative_fixed_qf_m3_s_V=_stable_source_sum(
            sources,
            "recombination_derivative_fixed_qf_m3_s_V",
        ),
        minimum_occupancy=min(item.minimum_occupancy for item in sources),
        maximum_occupancy=max(item.maximum_occupancy for item in sources),
        minimum_kinetic_denominator_s1=min(
            item.minimum_kinetic_denominator_s1 for item in sources
        ),
    )


__all__ = [
    "ENERGY_DISTRIBUTED_DEFECT_CLOSURE_VERSION",
    "EnergyDistributedDefectClosureCapabilityError",
    "EnergyDistributedDefectClosureError",
    "EnergyDistributedDefectClosureResult",
    "EnergyResolvedSpeciesClosure",
    "evaluate_energy_distributed_defect_closure",
]
