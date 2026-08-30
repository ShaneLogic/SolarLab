"""Compiled device closure for stationary multivalent bulk defects.

The local physics lives in :mod:`multivalent_defect_closure`.  This module
adds only device ownership: disjoint layer masks, multi-species aggregation,
contact charge neutrality, and immutable provenance.  A physical defect keeps
one shared density and one normalized charge-state distribution throughout;
it is never decomposed into independent monovalent SRH centers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.multivalent_defects import (
    MultivalentBulkDefectSpecies,
)
from perovskite_sim.physics.multivalent_defect_closure import (
    MULTIVALENT_DEFECT_CLOSURE_VERSION,
    MultivalentDefectClosureResult,
    evaluate_multivalent_defect_closure,
)
from perovskite_sim.physics.temperature import thermal_voltage

if TYPE_CHECKING:
    from perovskite_sim.physics.statistics import BulkChargeNeutralityState


MULTIVALENT_BULK_DEFECT_MODEL_VERSION = "multivalent-device-mb-qf-dc-v1"


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


@dataclass(frozen=True, slots=True)
class MultivalentDefectRegion:
    """One canonical v4 document compiled onto one uniform material region."""

    identifier: str
    document_sha256: str
    active_nodes: np.ndarray
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    temperature_K: float
    species: tuple[MultivalentBulkDefectSpecies, ...]

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise ValueError("multivalent region identifier must be non-empty")
        digest = str(self.document_sha256).strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("document_sha256 must be a SHA-256 hex digest")
        active = np.asarray(self.active_nodes, dtype=bool)
        if active.ndim != 1 or active.size < 1 or not np.any(active):
            raise ValueError("active_nodes must be a non-empty one-dimensional mask")
        resolved_species = tuple(self.species)
        if not resolved_species or not all(
            isinstance(item, MultivalentBulkDefectSpecies) for item in resolved_species
        ):
            raise TypeError("species must contain multivalent defect species")
        names = tuple(item.name for item in resolved_species)
        if len(set(names)) != len(names):
            raise ValueError("multivalent species names must be unique within a region")
        gap = _positive(self.band_gap_eV, "band_gap_eV")
        conduction_dos = _positive(
            self.effective_conduction_dos_m3,
            "effective_conduction_dos_m3",
        )
        valence_dos = _positive(
            self.effective_valence_dos_m3,
            "effective_valence_dos_m3",
        )
        temperature = _positive(self.temperature_K, "temperature_K")
        for item in resolved_species:
            item.validate_band_gap(gap)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "document_sha256", digest)
        object.__setattr__(self, "active_nodes", _readonly(active, dtype=bool))
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(self, "effective_conduction_dos_m3", conduction_dos)
        object.__setattr__(self, "effective_valence_dos_m3", valence_dos)
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "species", resolved_species)

    @property
    def species_identifiers(self) -> tuple[str, ...]:
        return tuple(f"{self.identifier}/{item.name}" for item in self.species)


@dataclass(frozen=True, slots=True)
class MultivalentBulkDefectModel:
    """Disjoint canonical v4 regions compiled on one electrical grid."""

    regions: tuple[MultivalentDefectRegion, ...]

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions or not all(
            isinstance(item, MultivalentDefectRegion) for item in regions
        ):
            raise TypeError("regions must contain MultivalentDefectRegion values")
        node_count = regions[0].active_nodes.size
        if any(item.active_nodes.size != node_count for item in regions):
            raise ValueError("all multivalent regions must share one device grid")
        ownership = np.zeros(node_count, dtype=np.int8)
        for item in regions:
            ownership += item.active_nodes.astype(np.int8)
        if np.any(ownership > 1):
            raise ValueError("multivalent region masks must be disjoint")
        identifiers = tuple(
            identifier
            for region in regions
            for identifier in region.species_identifiers
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("compiled multivalent species identifiers must be unique")
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
        return tuple(
            identifier
            for region in self.regions
            for identifier in region.species_identifiers
        )

    @property
    def state_counts(self) -> tuple[int, ...]:
        return tuple(
            len(item.configuration.charge_states_e)
            for region in self.regions
            for item in region.species
        )

    @property
    def layer_document_sha256(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (region.identifier, region.document_sha256) for region in self.regions
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {
                "model": MULTIVALENT_BULK_DEFECT_MODEL_VERSION,
                "local_closure": MULTIVALENT_DEFECT_CLOSURE_VERSION,
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
                        "effective_valence_dos_m3": (region.effective_valence_dos_m3),
                        "temperature_K": region.temperature_K,
                        "species": [item.to_dict() for item in region.species],
                    }
                    for region in self.regions
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class MultivalentSourceDefectClosureResult:
    """Aggregate of multiple physical multivalent species at common carriers."""

    species_names: tuple[str, ...]
    species_closures: tuple[MultivalentDefectClosureResult, ...]
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
    minimum_state_probability: float
    maximum_state_probability: float
    maximum_probability_sum_error: float
    maximum_master_residual_s1: float
    minimum_transition_rate_s1: float


def evaluate_multivalent_source_defect_closure(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    species: Sequence[MultivalentBulkDefectSpecies],
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> MultivalentSourceDefectClosureResult:
    """Evaluate and sum physical species without merging their state spaces."""

    resolved = tuple(species)
    if not resolved or not all(
        isinstance(item, MultivalentBulkDefectSpecies) for item in resolved
    ):
        raise TypeError("species must contain multivalent defect species")
    names = tuple(item.name for item in resolved)
    if len(set(names)) != len(names):
        raise ValueError("multivalent species names must be unique")
    closures = tuple(
        evaluate_multivalent_defect_closure(
            electron_density_m3,
            hole_density_m3,
            item,
            band_gap_eV=band_gap_eV,
            effective_conduction_dos_m3=effective_conduction_dos_m3,
            effective_valence_dos_m3=effective_valence_dos_m3,
            temperature_K=temperature_K,
        )
        for item in resolved
    )
    charge = np.stack([item.charge_density_C_m3 for item in closures], axis=0)
    rate = np.stack(
        [item.total_recombination_rate_m3_s for item in closures],
        axis=0,
    )
    derivative_n = np.stack(
        [item.total_recombination_derivative_n_s1 for item in closures],
        axis=0,
    )
    derivative_p = np.stack(
        [item.total_recombination_derivative_p_s1 for item in closures],
        axis=0,
    )
    charge_fixed_qf = np.stack(
        [item.charge_derivative_fixed_qf_C_m3_V for item in closures],
        axis=0,
    )
    transition_rates = tuple(
        value
        for item in closures
        for value in (
            float(np.min(item.forward_state_rate_s1)),
            float(np.min(item.backward_state_rate_s1)),
        )
    )
    return MultivalentSourceDefectClosureResult(
        species_names=names,
        species_closures=closures,
        charge_density_C_m3=_readonly(charge),
        recombination_rate_m3_s=_readonly(rate),
        recombination_derivative_n_s1=_readonly(derivative_n),
        recombination_derivative_p_s1=_readonly(derivative_p),
        charge_derivative_fixed_qf_C_m3_V=_readonly(charge_fixed_qf),
        total_charge_density_C_m3=_readonly(np.sum(charge, axis=0)),
        total_recombination_rate_m3_s=_readonly(np.sum(rate, axis=0)),
        total_recombination_derivative_n_s1=_readonly(np.sum(derivative_n, axis=0)),
        total_recombination_derivative_p_s1=_readonly(np.sum(derivative_p, axis=0)),
        total_charge_derivative_fixed_qf_C_m3_V=_readonly(
            np.sum(charge_fixed_qf, axis=0)
        ),
        minimum_state_probability=min(
            item.minimum_state_probability for item in closures
        ),
        maximum_state_probability=max(
            item.maximum_state_probability for item in closures
        ),
        maximum_probability_sum_error=max(
            item.maximum_probability_sum_error for item in closures
        ),
        maximum_master_residual_s1=max(
            item.maximum_master_residual_s1 for item in closures
        ),
        minimum_transition_rate_s1=min(transition_rates),
    )


@dataclass(frozen=True, slots=True)
class MultivalentBulkDefectEvaluation:
    """Full-grid multivalent charge, recombination, probabilities, and tangents."""

    model_identity_sha256: str
    species_identifiers: tuple[str, ...]
    state_counts: tuple[int, ...]
    active_nodes: np.ndarray
    state_probability: tuple[np.ndarray, ...]
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
    minimum_state_probability: float
    maximum_state_probability: float
    maximum_probability_sum_error: float
    maximum_master_residual_s1: float
    minimum_transition_rate_s1: float


def evaluate_multivalent_bulk_defects(
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    model: MultivalentBulkDefectModel,
) -> MultivalentBulkDefectEvaluation:
    """Evaluate one compiled multivalent model on its full electrical grid."""

    if not isinstance(model, MultivalentBulkDefectModel):
        raise TypeError("model must be a MultivalentBulkDefectModel")
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if n.shape != (model.node_count,):
        raise ValueError("multivalent bulk-defect state must match the compiled grid")
    if (
        not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n <= 0.0)
        or np.any(p <= 0.0)
    ):
        raise ValueError("multivalent carrier densities must be finite and positive")

    species_count = len(model.species_identifiers)
    shape = (species_count, model.node_count)
    active_nodes = np.zeros(shape, dtype=bool)
    charge = np.zeros(shape, dtype=float)
    rate = np.zeros(shape, dtype=float)
    derivative_n = np.zeros(shape, dtype=float)
    derivative_p = np.zeros(shape, dtype=float)
    charge_fixed_qf = np.zeros(shape, dtype=float)
    # Charge-state probabilities are an intensive, normalized quantity: off a
    # species' own region they are undefined, not zero. Zero-padding them
    # would publish columns that sum to 0 while the accompanying certificate
    # scalar reports a ~1e-16 normalization error. NaN follows the existing
    # band_diagram convention for ill-defined regions; the extensive charge
    # and rate arrays stay zero-padded because zero IS their value there.
    state_probability = [
        np.full((state_count, model.node_count), np.nan, dtype=float)
        for state_count in model.state_counts
    ]
    total_charge = np.zeros(model.node_count, dtype=float)
    total_rate = np.zeros(model.node_count, dtype=float)
    total_derivative_n = np.zeros(model.node_count, dtype=float)
    total_derivative_p = np.zeros(model.node_count, dtype=float)
    total_charge_fixed_qf = np.zeros(model.node_count, dtype=float)
    minimum_probabilities: list[float] = []
    maximum_probabilities: list[float] = []
    probability_sum_errors: list[float] = []
    master_residuals: list[float] = []
    transition_rates: list[float] = []

    offset = 0
    for region in model.regions:
        mask = region.active_nodes
        local = evaluate_multivalent_source_defect_closure(
            n[mask],
            p[mask],
            region.species,
            band_gap_eV=region.band_gap_eV,
            effective_conduction_dos_m3=region.effective_conduction_dos_m3,
            effective_valence_dos_m3=region.effective_valence_dos_m3,
            temperature_K=region.temperature_K,
        )
        count = len(region.species)
        rows = slice(offset, offset + count)
        active_nodes[rows, mask] = True
        charge[rows, mask] = local.charge_density_C_m3
        rate[rows, mask] = local.recombination_rate_m3_s
        derivative_n[rows, mask] = local.recombination_derivative_n_s1
        derivative_p[rows, mask] = local.recombination_derivative_p_s1
        charge_fixed_qf[rows, mask] = local.charge_derivative_fixed_qf_C_m3_V
        for local_index, closure in enumerate(local.species_closures):
            state_probability[offset + local_index][:, mask] = closure.state_probability
        total_charge[mask] = local.total_charge_density_C_m3
        total_rate[mask] = local.total_recombination_rate_m3_s
        total_derivative_n[mask] = local.total_recombination_derivative_n_s1
        total_derivative_p[mask] = local.total_recombination_derivative_p_s1
        total_charge_fixed_qf[mask] = local.total_charge_derivative_fixed_qf_C_m3_V
        minimum_probabilities.append(local.minimum_state_probability)
        maximum_probabilities.append(local.maximum_state_probability)
        probability_sum_errors.append(local.maximum_probability_sum_error)
        master_residuals.append(local.maximum_master_residual_s1)
        transition_rates.append(local.minimum_transition_rate_s1)
        offset += count

    return MultivalentBulkDefectEvaluation(
        model_identity_sha256=model.identity_sha256,
        species_identifiers=model.species_identifiers,
        state_counts=model.state_counts,
        active_nodes=_readonly(active_nodes, dtype=bool),
        state_probability=tuple(_readonly(value) for value in state_probability),
        charge_density_C_m3=_readonly(charge),
        recombination_rate_m3_s=_readonly(rate),
        recombination_derivative_n_s1=_readonly(derivative_n),
        recombination_derivative_p_s1=_readonly(derivative_p),
        charge_derivative_fixed_qf_C_m3_V=_readonly(charge_fixed_qf),
        total_charge_density_C_m3=_readonly(total_charge),
        total_recombination_rate_m3_s=_readonly(total_rate),
        total_recombination_derivative_n_s1=_readonly(total_derivative_n),
        total_recombination_derivative_p_s1=_readonly(total_derivative_p),
        total_charge_derivative_fixed_qf_C_m3_V=_readonly(total_charge_fixed_qf),
        minimum_state_probability=min(minimum_probabilities),
        maximum_state_probability=max(maximum_probabilities),
        maximum_probability_sum_error=max(probability_sum_errors),
        maximum_master_residual_s1=max(master_residuals),
        minimum_transition_rate_s1=min(transition_rates),
    )


@dataclass(frozen=True, slots=True)
class MultivalentDefectNeutralityResult:
    """Contact neutrality and the exact multivalent closure at its root."""

    neutrality: "BulkChargeNeutralityState"
    closure: MultivalentSourceDefectClosureResult


def solve_multivalent_defect_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    acceptor_density_m3: float,
    donor_density_m3: float,
    species: Sequence[MultivalentBulkDefectSpecies],
) -> MultivalentDefectNeutralityResult:
    """Solve MB contact neutrality with the same multivalent master equation."""

    from perovskite_sim.physics.statistics import (
        FULLY_IONIZED,
        MAXWELL_BOLTZMANN,
        BulkChargeNeutralityState,
        carrier_density_from_reduced_fermi_level,
    )

    temperature = _positive(temperature_K, "temperature_K")
    gap = _positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _positive(
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
    resolved = tuple(species)
    if not resolved or not all(
        isinstance(item, MultivalentBulkDefectSpecies) for item in resolved
    ):
        raise TypeError("species must contain multivalent defect species")
    for item in resolved:
        item.validate_band_gap(gap)
    thermal = thermal_voltage(temperature)
    reduced_gap = gap / thermal

    def evaluate(
        eta_n: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
        MultivalentSourceDefectClosureResult,
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
        closure = evaluate_multivalent_source_defect_closure(
            electron,
            hole,
            resolved,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
        )
        defect_charge_number = float(np.asarray(closure.total_charge_density_C_m3) / Q)
        residual = hole - electron + donors - acceptors + defect_charge_number
        return residual, electron, hole, eta_p, closure

    intrinsic_center = 0.5 * (-reduced_gap + math.log(valence_dos / conduction_dos))
    half_width = max(32.0, 0.5 * reduced_gap + 8.0)
    lower = intrinsic_center - half_width
    upper = intrinsic_center + half_width
    for _ in range(32):
        if evaluate(lower)[0] >= 0.0 and evaluate(upper)[0] <= 0.0:
            break
        half_width *= 2.0
        lower = intrinsic_center - half_width
        upper = intrinsic_center + half_width
    else:
        raise RuntimeError("could not bracket multivalent-defect charge neutrality")

    for _ in range(220):
        midpoint = 0.5 * (lower + upper)
        if evaluate(midpoint)[0] > 0.0:
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
        *(item.total_density_m3 for item in resolved),
        1.0,
    )
    normalized = abs(residual) / scale
    if not math.isfinite(normalized) or normalized > 1.0e-12:
        raise RuntimeError(
            "multivalent-defect charge-neutrality residual exceeded gate"
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
    return MultivalentDefectNeutralityResult(
        neutrality=neutrality,
        closure=closure,
    )


__all__ = [
    "MULTIVALENT_BULK_DEFECT_MODEL_VERSION",
    "MultivalentBulkDefectEvaluation",
    "MultivalentBulkDefectModel",
    "MultivalentDefectNeutralityResult",
    "MultivalentDefectRegion",
    "MultivalentSourceDefectClosureResult",
    "evaluate_multivalent_bulk_defects",
    "evaluate_multivalent_source_defect_closure",
    "solve_multivalent_defect_charge_neutrality",
]
