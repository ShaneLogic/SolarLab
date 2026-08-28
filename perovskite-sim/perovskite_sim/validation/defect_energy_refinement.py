"""Independent energy-order refinement for local distributed defects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import SINGLE_LEVEL, BulkDefectSpecies
from perovskite_sim.physics.distributed_defect_closure import (
    EnergyDistributedDefectClosureResult,
    evaluate_energy_distributed_defect_closure,
)


DEFECT_ENERGY_REFINEMENT_VERSION = "local-defect-energy-order-v1"
DEFAULT_ENERGY_REFINEMENT_ORDERS = (8, 16, 32)
DEFAULT_ENERGY_REFINEMENT_THRESHOLD = 5.0e-3


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


def _orders(values: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(values)
    if len(resolved) < 2:
        raise ValueError("energy refinement orders require at least two values")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in resolved
    ):
        raise ValueError("energy refinement orders must be integers")
    orders = tuple(int(value) for value in resolved)
    if any(value < 2 or value > 512 for value in orders):
        raise ValueError("energy refinement orders must lie in [2, 512]")
    if any(fine != 2 * coarse for coarse, fine in zip(orders, orders[1:])):
        raise ValueError("energy refinement orders must increase by exactly 2x")
    return orders


def _maximum_absolute_difference(left: object, right: object) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(right, dtype=float)
                - np.asarray(left, dtype=float)
            )
        )
    )


def _maximum_relative_difference(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(left_array))),
        float(np.max(np.abs(right_array))),
        1.0,
    )
    return _maximum_absolute_difference(left_array, right_array) / scale


@dataclass(frozen=True, slots=True)
class DefectEnergyRefinementComparison:
    """One adjacent energy-order comparison across every source species."""

    coarse_order: int
    fine_order: int
    maximum_source_occupancy_absolute_change: float
    maximum_source_charge_normalized_change: float
    maximum_source_recombination_relative_change: float
    maximum_source_tangent_relative_change: float
    threshold: float
    passed: bool

    def __post_init__(self) -> None:
        if self.fine_order != 2 * self.coarse_order:
            raise ValueError("comparison orders must increase by exactly 2x")
        threshold = _finite_positive(self.threshold, "threshold")
        metrics = (
            self.maximum_source_occupancy_absolute_change,
            self.maximum_source_charge_normalized_change,
            self.maximum_source_recombination_relative_change,
            self.maximum_source_tangent_relative_change,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in metrics
        ):
            raise ValueError("energy refinement metrics must be finite and non-negative")
        expected = all(float(value) <= threshold for value in metrics)
        if bool(self.passed) != expected:
            raise ValueError("energy refinement comparison pass flag is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "coarse_order": self.coarse_order,
            "fine_order": self.fine_order,
            "maximum_source_occupancy_absolute_change": (
                self.maximum_source_occupancy_absolute_change
            ),
            "maximum_source_charge_normalized_change": (
                self.maximum_source_charge_normalized_change
            ),
            "maximum_source_recombination_relative_change": (
                self.maximum_source_recombination_relative_change
            ),
            "maximum_source_tangent_relative_change": (
                self.maximum_source_tangent_relative_change
            ),
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class DefectEnergyRefinementReport:
    """Content-addressed local energy-order convergence evidence."""

    input_identity_sha256: str
    energy_orders: tuple[int, ...]
    closure_identity_sha256: tuple[str, ...]
    source_identifiers: tuple[str, ...]
    distribution_kinds: tuple[str, ...]
    comparisons: tuple[DefectEnergyRefinementComparison, ...]
    passed: bool

    def __post_init__(self) -> None:
        digest = str(self.input_identity_sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("input_identity_sha256 must be a SHA-256 hex")
        orders = _orders(self.energy_orders)
        identities = tuple(self.closure_identity_sha256)
        if len(identities) != len(orders) or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in identities
        ):
            raise ValueError("closure identities must match energy orders")
        identifiers = tuple(self.source_identifiers)
        kinds = tuple(self.distribution_kinds)
        if (
            not identifiers
            or len(identifiers) != len(set(identifiers))
            or len(kinds) != len(identifiers)
        ):
            raise ValueError("refinement report source metadata is invalid")
        comparisons = tuple(self.comparisons)
        if len(comparisons) != len(orders) - 1 or any(
            item.coarse_order != coarse or item.fine_order != fine
            for item, coarse, fine in zip(
                comparisons,
                orders[:-1],
                orders[1:],
                strict=True,
            )
        ):
            raise ValueError("refinement comparisons do not cover adjacent orders")
        expected = all(item.passed for item in comparisons)
        if bool(self.passed) != expected:
            raise ValueError("energy refinement report pass flag is inconsistent")
        object.__setattr__(self, "input_identity_sha256", digest)
        object.__setattr__(self, "energy_orders", orders)
        object.__setattr__(self, "closure_identity_sha256", identities)
        object.__setattr__(self, "source_identifiers", identifiers)
        object.__setattr__(self, "distribution_kinds", kinds)
        object.__setattr__(self, "comparisons", comparisons)

    @property
    def terminal_comparison(self) -> DefectEnergyRefinementComparison:
        return self.comparisons[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "refinement": DEFECT_ENERGY_REFINEMENT_VERSION,
            "input_identity_sha256": self.input_identity_sha256,
            "energy_orders": list(self.energy_orders),
            "closure_identity_sha256": list(self.closure_identity_sha256),
            "source_identifiers": list(self.source_identifiers),
            "distribution_kinds": list(self.distribution_kinds),
            "comparisons": [item.to_dict() for item in self.comparisons],
            "passed": self.passed,
        }


_SOURCE_TANGENT_FIELDS = (
    "recombination_derivative_n_s1",
    "recombination_derivative_p_s1",
    "charge_derivative_n_C",
    "charge_derivative_p_C",
    "charge_derivative_fixed_qf_C_m3_V",
    "recombination_derivative_fixed_qf_m3_s_V",
)


def _comparison(
    coarse: EnergyDistributedDefectClosureResult,
    fine: EnergyDistributedDefectClosureResult,
    *,
    threshold: float,
) -> DefectEnergyRefinementComparison:
    coarse_sources = {
        item.source_identifier: item for item in coarse.source_closures
    }
    fine_sources = {
        item.source_identifier: item for item in fine.source_closures
    }
    if set(coarse_sources) != set(fine_sources):
        raise ValueError("energy refinement source sets changed between orders")
    occupancy_change = 0.0
    charge_change = 0.0
    recombination_change = 0.0
    tangent_change = 0.0
    for identifier in sorted(coarse_sources):
        left = coarse_sources[identifier]
        right = fine_sources[identifier]
        occupancy_change = max(
            occupancy_change,
            _maximum_absolute_difference(
                left.mean_occupancy,
                right.mean_occupancy,
            ),
        )
        charge_scale = Q * float(
            right.source_species.distribution.total_density_m3
        )
        charge_change = max(
            charge_change,
            _maximum_absolute_difference(
                left.charge_density_C_m3,
                right.charge_density_C_m3,
            )
            / charge_scale,
        )
        recombination_change = max(
            recombination_change,
            _maximum_relative_difference(
                left.recombination_rate_m3_s,
                right.recombination_rate_m3_s,
            ),
        )
        tangent_change = max(
            tangent_change,
            *(
                _maximum_relative_difference(
                    getattr(left, field),
                    getattr(right, field),
                )
                for field in _SOURCE_TANGENT_FIELDS
            ),
        )
    metrics = (
        occupancy_change,
        charge_change,
        recombination_change,
        tangent_change,
    )
    return DefectEnergyRefinementComparison(
        coarse_order=max(coarse.energy_orders),
        fine_order=max(fine.energy_orders),
        maximum_source_occupancy_absolute_change=occupancy_change,
        maximum_source_charge_normalized_change=charge_change,
        maximum_source_recombination_relative_change=recombination_change,
        maximum_source_tangent_relative_change=tangent_change,
        threshold=threshold,
        passed=all(value <= threshold for value in metrics),
    )


def _input_identity(
    electron_density_m3: object,
    hole_density_m3: object,
    species: tuple[BulkDefectSpecies, ...],
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
    energy_orders: tuple[int, ...],
    threshold: float,
) -> str:
    payload = {
        "refinement": DEFECT_ENERGY_REFINEMENT_VERSION,
        "electron_density_m3": np.asarray(
            electron_density_m3,
            dtype=float,
        ).tolist(),
        "hole_density_m3": np.asarray(hole_density_m3, dtype=float).tolist(),
        "species": [item.to_dict() for item in species],
        "band_gap_eV": band_gap_eV,
        "effective_conduction_dos_m3": effective_conduction_dos_m3,
        "effective_valence_dos_m3": effective_valence_dos_m3,
        "temperature_K": temperature_K,
        "energy_orders": list(energy_orders),
        "threshold": threshold,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def assess_defect_energy_order_refinement(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    species: Sequence[BulkDefectSpecies],
    *,
    band_gap_eV: object,
    effective_conduction_dos_m3: object,
    effective_valence_dos_m3: object,
    temperature_K: object,
    energy_orders: Sequence[int] = DEFAULT_ENERGY_REFINEMENT_ORDERS,
    threshold: object = DEFAULT_ENERGY_REFINEMENT_THRESHOLD,
) -> DefectEnergyRefinementReport:
    """Evaluate only the local energy dimension at a fixed carrier state."""

    resolved_species = tuple(species)
    if not resolved_species or not all(
        isinstance(item, BulkDefectSpecies) for item in resolved_species
    ):
        raise ValueError("energy refinement requires bulk defect species")
    if all(
        item.distribution.kind == SINGLE_LEVEL for item in resolved_species
    ):
        raise ValueError(
            "energy refinement requires at least one distributed species"
        )
    resolved_orders = _orders(energy_orders)
    resolved_threshold = _finite_positive(threshold, "threshold")
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
    electron, hole = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    results = tuple(
        evaluate_energy_distributed_defect_closure(
            electron,
            hole,
            resolved_species,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
            energy_quadrature_order=order,
        )
        for order in resolved_orders
    )
    comparisons = tuple(
        _comparison(coarse, fine, threshold=resolved_threshold)
        for coarse, fine in zip(results[:-1], results[1:], strict=True)
    )
    first = results[0]
    return DefectEnergyRefinementReport(
        input_identity_sha256=_input_identity(
            electron,
            hole,
            resolved_species,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
            energy_orders=resolved_orders,
            threshold=resolved_threshold,
        ),
        energy_orders=resolved_orders,
        closure_identity_sha256=tuple(
            item.closure_identity_sha256 for item in results
        ),
        source_identifiers=first.source_identifiers,
        distribution_kinds=first.distribution_kinds,
        comparisons=comparisons,
        passed=all(item.passed for item in comparisons),
    )


__all__ = [
    "DEFAULT_ENERGY_REFINEMENT_ORDERS",
    "DEFAULT_ENERGY_REFINEMENT_THRESHOLD",
    "DEFECT_ENERGY_REFINEMENT_VERSION",
    "DefectEnergyRefinementComparison",
    "DefectEnergyRefinementReport",
    "assess_defect_energy_order_refinement",
]
