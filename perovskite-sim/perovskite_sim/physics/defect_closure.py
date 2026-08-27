"""Local Maxwell-Boltzmann closure for canonical monovalent bulk defects.

This module is solver-independent by design. It evaluates occupancy,
recombination, and charge from one shared local state, but it does not insert
charged defects into Poisson, contacts, or any production experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    NEUTRAL,
    SINGLE_LEVEL,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.physics.temperature import thermal_voltage


MONOVALENT_DEFECT_CLOSURE_VERSION = "monovalent-local-mb-v1"


class MonovalentDefectClosureCapabilityError(ExplicitDefectCapabilityError):
    """A valid defect input requested physics outside the DEF-2 closure."""


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


def _readonly(value: object) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
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
                f"DEF-2 occupancy is undefined when both capture legs are zero: {item.name}"
            )
    return resolved


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


__all__ = [
    "MONOVALENT_DEFECT_CLOSURE_VERSION",
    "MonovalentDefectClosureCapabilityError",
    "MonovalentDefectClosureResult",
    "evaluate_monovalent_defect_closure",
]
