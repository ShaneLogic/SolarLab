"""Frozen-configuration device model for metastable bulk defects.

SCAPS establishes the donor/acceptor configuration distribution at an initial
working point and then *freezes* it for the measurement. This module carries
that frozen split: each region owns one immutable per-node donor fraction
``y`` produced by a preparation solve, and evaluates the measurement state as

    charge   = N_t * [ y * q_donor(n, p) + (1 - y) * q_acceptor(n, p) ]
    rate     = N_t * [ y * R_donor(n, p) + (1 - y) * R_acceptor(n, p) ]

where each configuration's internal charge-state populations still follow the
local carriers through the ordinary multivalent master equation. Only the
*configuration* split is frozen, not the charge-state distribution inside a
configuration.

Both configuration observables are linear in the defect density, so each
configuration is evaluated once at the full density and weighted afterwards;
no per-node species objects are constructed.

The frozen fraction is immutable and carries the preparation protocol and
state hashes, so a measurement cannot silently re-prepare itself: any change
to the preparation inputs changes the identity of this model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from perovskite_sim.models.multivalent_defects import (
    MetastableDefectDefinition,
    MultivalentBulkDefectSpecies,
)
from perovskite_sim.physics.multivalent_defect_closure import (
    MULTIVALENT_DEFECT_CLOSURE_VERSION,
    evaluate_multivalent_defect_closure,
)


FROZEN_METASTABLE_MODEL_VERSION = "metastable-frozen-configuration-device-v1"


class FrozenMetastableModelError(RuntimeError):
    """The frozen metastable inventory was invalid or inconsistent."""


def _readonly(value: object, *, dtype: object = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _digest(value: object, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return digest


def configuration_species(
    definition: MetastableDefectDefinition,
) -> tuple[MultivalentBulkDefectSpecies, MultivalentBulkDefectSpecies]:
    """Return the donor and acceptor configurations as full-density species."""

    return (
        MultivalentBulkDefectSpecies(
            name=f"{definition.name}/donor_configuration",
            total_density_m3=definition.total_density_m3,
            configuration=definition.donor_configuration,
        ),
        MultivalentBulkDefectSpecies(
            name=f"{definition.name}/acceptor_configuration",
            total_density_m3=definition.total_density_m3,
            configuration=definition.acceptor_configuration,
        ),
    )


@dataclass(frozen=True, slots=True)
class FrozenMetastableRegion:
    """One metastable defect frozen onto one uniform material region."""

    identifier: str
    definition_sha256: str
    preparation_protocol_sha256: str
    preparation_state_sha256: str
    active_nodes: np.ndarray
    donor_fraction: np.ndarray
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    temperature_K: float
    definition: MetastableDefectDefinition

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise ValueError("frozen metastable identifier must be non-empty")
        active = np.asarray(self.active_nodes, dtype=bool)
        if active.ndim != 1 or active.size < 1 or not np.any(active):
            raise ValueError("active_nodes must be a non-empty one-dimensional mask")
        fraction = np.asarray(self.donor_fraction, dtype=float)
        if fraction.shape != (int(np.count_nonzero(active)),):
            raise ValueError(
                "donor_fraction must supply one frozen value per active node"
            )
        if (
            not np.all(np.isfinite(fraction))
            or np.any(fraction < 0.0)
            or np.any(fraction > 1.0)
        ):
            raise ValueError("frozen donor_fraction must be finite and inside [0, 1]")
        if not isinstance(self.definition, MetastableDefectDefinition):
            raise TypeError("definition must be a MetastableDefectDefinition")
        gap = _positive(self.band_gap_eV, "band_gap_eV")
        self.definition.validate_band_gap(gap)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(
            self,
            "definition_sha256",
            _digest(self.definition_sha256, "definition_sha256"),
        )
        object.__setattr__(
            self,
            "preparation_protocol_sha256",
            _digest(self.preparation_protocol_sha256, "preparation_protocol_sha256"),
        )
        object.__setattr__(
            self,
            "preparation_state_sha256",
            _digest(self.preparation_state_sha256, "preparation_state_sha256"),
        )
        object.__setattr__(self, "active_nodes", _readonly(active, dtype=bool))
        object.__setattr__(self, "donor_fraction", _readonly(fraction))
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(
            self,
            "effective_conduction_dos_m3",
            _positive(self.effective_conduction_dos_m3, "effective_conduction_dos_m3"),
        )
        object.__setattr__(
            self,
            "effective_valence_dos_m3",
            _positive(self.effective_valence_dos_m3, "effective_valence_dos_m3"),
        )
        object.__setattr__(
            self, "temperature_K", _positive(self.temperature_K, "temperature_K")
        )

    @property
    def species_identifier(self) -> str:
        return f"{self.identifier}/{self.definition.name}"


@dataclass(frozen=True, slots=True)
class FrozenMetastableBulkDefectModel:
    """Disjoint frozen metastable regions compiled on one electrical grid."""

    regions: tuple[FrozenMetastableRegion, ...]

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions or not all(
            isinstance(item, FrozenMetastableRegion) for item in regions
        ):
            raise TypeError("regions must contain FrozenMetastableRegion values")
        node_count = regions[0].active_nodes.size
        if any(item.active_nodes.size != node_count for item in regions):
            raise ValueError("all frozen metastable regions must share one grid")
        ownership = np.zeros(node_count, dtype=np.int8)
        for item in regions:
            ownership += item.active_nodes.astype(np.int8)
        if np.any(ownership > 1):
            raise ValueError("frozen metastable region masks must be disjoint")
        identifiers = tuple(item.species_identifier for item in regions)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("frozen metastable species identifiers must be unique")
        object.__setattr__(self, "regions", regions)

    @property
    def node_count(self) -> int:
        return int(self.regions[0].active_nodes.size)

    @property
    def explicit_node_mask(self) -> np.ndarray:
        mask = np.logical_or.reduce([item.active_nodes for item in self.regions])
        return _readonly(mask, dtype=bool)

    @property
    def species_identifiers(self) -> tuple[str, ...]:
        return tuple(item.species_identifier for item in self.regions)

    @property
    def preparation_protocol_sha256s(self) -> tuple[str, ...]:
        return tuple(item.preparation_protocol_sha256 for item in self.regions)

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {
                "model": FROZEN_METASTABLE_MODEL_VERSION,
                "local_closure": MULTIVALENT_DEFECT_CLOSURE_VERSION,
                "regions": [
                    {
                        "identifier": region.identifier,
                        "definition_sha256": region.definition_sha256,
                        "preparation_protocol_sha256": (
                            region.preparation_protocol_sha256
                        ),
                        "preparation_state_sha256": region.preparation_state_sha256,
                        "active_node_indices": np.flatnonzero(
                            region.active_nodes
                        ).tolist(),
                        "donor_fraction": region.donor_fraction.tolist(),
                        "band_gap_eV": region.band_gap_eV,
                        "effective_conduction_dos_m3": (
                            region.effective_conduction_dos_m3
                        ),
                        "effective_valence_dos_m3": region.effective_valence_dos_m3,
                        "temperature_K": region.temperature_K,
                        "definition": region.definition.to_dict(),
                    }
                    for region in self.regions
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenMetastableEvaluation:
    """Measurement-state charge, recombination, and fixed-QF tangents."""

    model_identity_sha256: str
    species_identifiers: tuple[str, ...]
    active_nodes: np.ndarray
    donor_fraction: np.ndarray
    donor_state_probability: tuple[np.ndarray, ...]
    acceptor_state_probability: tuple[np.ndarray, ...]
    total_charge_density_C_m3: np.ndarray
    total_recombination_rate_m3_s: np.ndarray
    total_recombination_derivative_n_s1: np.ndarray
    total_recombination_derivative_p_s1: np.ndarray
    total_charge_derivative_fixed_qf_C_m3_V: np.ndarray
    minimum_state_probability: float
    maximum_state_probability: float
    maximum_probability_sum_error: float
    maximum_master_residual_s1: float
    minimum_transition_rate_s1: float


def evaluate_frozen_metastable_bulk_defects(
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    model: FrozenMetastableBulkDefectModel,
) -> FrozenMetastableEvaluation:
    """Evaluate one frozen metastable inventory on its electrical grid."""

    if not isinstance(model, FrozenMetastableBulkDefectModel):
        raise TypeError("model must be a FrozenMetastableBulkDefectModel")
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if n.shape != (model.node_count,):
        raise ValueError("frozen metastable state must match the compiled grid")
    if (
        not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n <= 0.0)
        or np.any(p <= 0.0)
    ):
        raise ValueError("frozen metastable carrier densities must be positive")

    node_count = model.node_count
    active = np.zeros((len(model.regions), node_count), dtype=bool)
    donor_fraction_full = np.full(node_count, np.nan, dtype=float)
    total_charge = np.zeros(node_count, dtype=float)
    total_rate = np.zeros(node_count, dtype=float)
    total_derivative_n = np.zeros(node_count, dtype=float)
    total_derivative_p = np.zeros(node_count, dtype=float)
    total_charge_fixed_qf = np.zeros(node_count, dtype=float)
    donor_probability: list[np.ndarray] = []
    acceptor_probability: list[np.ndarray] = []
    minimum_probabilities: list[float] = []
    maximum_probabilities: list[float] = []
    sum_errors: list[float] = []
    residuals: list[float] = []
    transition_rates: list[float] = []

    for index, region in enumerate(model.regions):
        mask = region.active_nodes
        active[index] = mask
        fraction = np.asarray(region.donor_fraction, dtype=float)
        donor_fraction_full[mask] = fraction
        donor_species, acceptor_species = configuration_species(region.definition)
        weights = (fraction, 1.0 - fraction)
        blocks = []
        for species, weight in zip(
            (donor_species, acceptor_species), weights, strict=True
        ):
            closure = evaluate_multivalent_defect_closure(
                n[mask],
                p[mask],
                species,
                band_gap_eV=region.band_gap_eV,
                effective_conduction_dos_m3=region.effective_conduction_dos_m3,
                effective_valence_dos_m3=region.effective_valence_dos_m3,
                temperature_K=region.temperature_K,
            )
            # Every observable below is linear in the species density, so the
            # frozen configuration weight applies after the closure.
            total_charge[mask] += weight * closure.charge_density_C_m3
            total_rate[mask] += weight * closure.total_recombination_rate_m3_s
            total_derivative_n[mask] += (
                weight * closure.total_recombination_derivative_n_s1
            )
            total_derivative_p[mask] += (
                weight * closure.total_recombination_derivative_p_s1
            )
            total_charge_fixed_qf[mask] += (
                weight * closure.charge_derivative_fixed_qf_C_m3_V
            )
            minimum_probabilities.append(closure.minimum_state_probability)
            maximum_probabilities.append(closure.maximum_state_probability)
            sum_errors.append(closure.maximum_probability_sum_error)
            residuals.append(closure.maximum_master_residual_s1)
            transition_rates.append(
                min(
                    float(np.min(closure.forward_state_rate_s1)),
                    float(np.min(closure.backward_state_rate_s1)),
                )
            )
            padded = np.full(
                (closure.state_probability.shape[0], node_count), np.nan, dtype=float
            )
            padded[:, mask] = closure.state_probability
            blocks.append(padded)
        donor_probability.append(blocks[0])
        acceptor_probability.append(blocks[1])

    return FrozenMetastableEvaluation(
        model_identity_sha256=model.identity_sha256,
        species_identifiers=model.species_identifiers,
        active_nodes=_readonly(active, dtype=bool),
        donor_fraction=_readonly(donor_fraction_full),
        donor_state_probability=tuple(_readonly(item) for item in donor_probability),
        acceptor_state_probability=tuple(
            _readonly(item) for item in acceptor_probability
        ),
        total_charge_density_C_m3=_readonly(total_charge),
        total_recombination_rate_m3_s=_readonly(total_rate),
        total_recombination_derivative_n_s1=_readonly(total_derivative_n),
        total_recombination_derivative_p_s1=_readonly(total_derivative_p),
        total_charge_derivative_fixed_qf_C_m3_V=_readonly(total_charge_fixed_qf),
        minimum_state_probability=min(minimum_probabilities),
        maximum_state_probability=max(maximum_probabilities),
        maximum_probability_sum_error=max(sum_errors),
        maximum_master_residual_s1=max(residuals),
        minimum_transition_rate_s1=min(transition_rates),
    )


__all__ = [
    "FROZEN_METASTABLE_MODEL_VERSION",
    "FrozenMetastableBulkDefectModel",
    "FrozenMetastableEvaluation",
    "FrozenMetastableModelError",
    "FrozenMetastableRegion",
    "configuration_species",
    "evaluate_frozen_metastable_bulk_defects",
]
