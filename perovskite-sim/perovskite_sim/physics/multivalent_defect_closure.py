"""Local stationary master-equation closure for multivalent defects.

The implementation follows the charge-state recurrence of Decock, Khelifi,
and Burgelman, Thin Solid Films 519 (2011) 7481-7484,
doi:10.1016/j.tsf.2010.12.039.  One shared density is distributed over all
coupled states.  Detailed-balance emission rates include the ratio of adjacent
state degeneracies.  The analytic tangent is the normalized recurrence form of
the implicit-function derivative of the stationary master equation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.multivalent_defects import (
    MultivalentBulkDefectSpecies,
)
from perovskite_sim.physics.temperature import thermal_voltage


MULTIVALENT_DEFECT_CLOSURE_VERSION = "multivalent-local-mb-master-v1"


class MultivalentDefectClosureError(RuntimeError):
    """The local multivalent stationary system was invalid or non-finite."""


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


def _readonly(value: object) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _identity(
    species: MultivalentBulkDefectSpecies,
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> str:
    payload = {
        "closure": MULTIVALENT_DEFECT_CLOSURE_VERSION,
        "statistics": "maxwell_boltzmann_grand_partition",
        "source_model": {
            "doi": "10.1016/j.tsf.2010.12.039",
            "equations": [1, 2, 4, 6, 7, 8, 9],
        },
        "band_gap_eV": band_gap_eV,
        "effective_conduction_dos_m3": effective_conduction_dos_m3,
        "effective_valence_dos_m3": effective_valence_dos_m3,
        "temperature_K": temperature_K,
        "species": species.to_dict(),
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
class MultivalentDefectClosureResult:
    """Stationary probabilities, observables, and analytic local tangents."""

    closure_identity_sha256: str
    species_name: str
    temperature_K: float
    thermal_voltage_V: float
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    intrinsic_product_m6: float
    total_density_m3: float
    charge_states_e: tuple[int, ...]
    state_degeneracies: tuple[float, ...]
    transition_energies_eV_above_vb: np.ndarray
    capture_n_m3_s: np.ndarray
    capture_p_m3_s: np.ndarray
    emission_n_s1: np.ndarray
    emission_p_s1: np.ndarray
    forward_state_rate_s1: np.ndarray
    backward_state_rate_s1: np.ndarray
    state_probability: np.ndarray
    state_probability_derivative_n_m3: np.ndarray
    state_probability_derivative_p_m3: np.ndarray
    master_matrix_s1: np.ndarray
    master_matrix_derivative_n_m3_s1: np.ndarray
    master_matrix_derivative_p_m3_s1: np.ndarray
    master_residual_s1: np.ndarray
    transition_recombination_rate_m3_s: np.ndarray
    transition_recombination_derivative_n_s1: np.ndarray
    transition_recombination_derivative_p_s1: np.ndarray
    charge_number_density_m3: np.ndarray
    charge_density_C_m3: np.ndarray
    charge_derivative_n_C: np.ndarray
    charge_derivative_p_C: np.ndarray
    charge_derivative_fixed_qf_C_m3_V: np.ndarray
    total_recombination_rate_m3_s: np.ndarray
    total_recombination_derivative_n_s1: np.ndarray
    total_recombination_derivative_p_s1: np.ndarray
    recombination_derivative_fixed_qf_m3_s_V: np.ndarray
    minimum_state_probability: float
    maximum_state_probability: float
    maximum_probability_sum_error: float
    maximum_master_residual_s1: float

    def __post_init__(self) -> None:
        digest = str(self.closure_identity_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("closure_identity_sha256 must be a SHA-256 hex")
        if not isinstance(self.species_name, str) or not self.species_name:
            raise ValueError("species_name must be non-empty")
        for field in (
            "temperature_K",
            "thermal_voltage_V",
            "band_gap_eV",
            "effective_conduction_dos_m3",
            "effective_valence_dos_m3",
            "total_density_m3",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        intrinsic = float(self.intrinsic_product_m6)
        if not math.isfinite(intrinsic) or intrinsic < 0.0:
            raise ValueError("intrinsic_product_m6 must be finite and non-negative")
        object.__setattr__(self, "intrinsic_product_m6", intrinsic)

        charges = tuple(self.charge_states_e)
        degeneracies = tuple(float(value) for value in self.state_degeneracies)
        if not 2 <= len(charges) <= 5 or len(degeneracies) != len(charges):
            raise ValueError("closure requires aligned 2-5 state metadata")
        state_probability = np.asarray(self.state_probability, dtype=float)
        if state_probability.ndim < 1 or state_probability.shape[0] != len(charges):
            raise ValueError("state_probability must begin with the state axis")
        state_shape = state_probability.shape
        carrier_shape = state_shape[1:]
        transition_shape = (len(charges) - 1,) + carrier_shape
        matrix_shape = (len(charges), len(charges)) + carrier_shape

        transition_reference_fields = (
            "transition_energies_eV_above_vb",
            "capture_n_m3_s",
            "capture_p_m3_s",
            "emission_n_s1",
            "emission_p_s1",
        )
        for field in transition_reference_fields:
            if np.asarray(getattr(self, field)).shape != (len(charges) - 1,):
                raise ValueError(f"{field} must have one value per transition")
        transition_fields = (
            "forward_state_rate_s1",
            "backward_state_rate_s1",
            "transition_recombination_rate_m3_s",
            "transition_recombination_derivative_n_s1",
            "transition_recombination_derivative_p_s1",
        )
        for field in transition_fields:
            if np.asarray(getattr(self, field)).shape != transition_shape:
                raise ValueError(f"{field} must match the transition/state shape")
        state_fields = (
            "state_probability",
            "state_probability_derivative_n_m3",
            "state_probability_derivative_p_m3",
            "master_residual_s1",
        )
        for field in state_fields:
            if np.asarray(getattr(self, field)).shape != state_shape:
                raise ValueError(f"{field} must match the state shape")
        matrix_fields = (
            "master_matrix_s1",
            "master_matrix_derivative_n_m3_s1",
            "master_matrix_derivative_p_m3_s1",
        )
        for field in matrix_fields:
            if np.asarray(getattr(self, field)).shape != matrix_shape:
                raise ValueError(f"{field} must match the master-matrix shape")
        carrier_fields = (
            "charge_number_density_m3",
            "charge_density_C_m3",
            "charge_derivative_n_C",
            "charge_derivative_p_C",
            "charge_derivative_fixed_qf_C_m3_V",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
            "recombination_derivative_fixed_qf_m3_s_V",
        )
        for field in carrier_fields:
            if np.asarray(getattr(self, field)).shape != carrier_shape:
                raise ValueError(f"{field} must match the carrier shape")
        for field in (
            transition_reference_fields
            + transition_fields
            + state_fields
            + matrix_fields
            + carrier_fields
        ):
            value = np.asarray(getattr(self, field), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, _readonly(value))

        minimum = float(self.minimum_state_probability)
        maximum = float(self.maximum_state_probability)
        sum_error = float(self.maximum_probability_sum_error)
        residual = float(self.maximum_master_residual_s1)
        if (
            not all(
                math.isfinite(value)
                for value in (minimum, maximum, sum_error, residual)
            )
            or minimum < 0.0
            or maximum > 1.0
            or sum_error < 0.0
            or residual < 0.0
            or minimum != float(np.min(state_probability))
            or maximum != float(np.max(state_probability))
        ):
            raise ValueError("multivalent closure diagnostics are inconsistent")
        object.__setattr__(self, "minimum_state_probability", minimum)
        object.__setattr__(self, "maximum_state_probability", maximum)
        object.__setattr__(self, "maximum_probability_sum_error", sum_error)
        object.__setattr__(self, "maximum_master_residual_s1", residual)
        object.__setattr__(self, "closure_identity_sha256", digest)
        object.__setattr__(self, "charge_states_e", charges)
        object.__setattr__(self, "state_degeneracies", degeneracies)

    def to_dict(self) -> dict[str, object]:
        return {
            "closure": MULTIVALENT_DEFECT_CLOSURE_VERSION,
            "closure_identity_sha256": self.closure_identity_sha256,
            "species_name": self.species_name,
            "statistics": "maxwell_boltzmann_grand_partition",
            "temperature_K": self.temperature_K,
            "thermal_voltage_V": self.thermal_voltage_V,
            "band_gap_eV": self.band_gap_eV,
            "effective_conduction_dos_m3": self.effective_conduction_dos_m3,
            "effective_valence_dos_m3": self.effective_valence_dos_m3,
            "intrinsic_product_m6": self.intrinsic_product_m6,
            "total_density_m3": self.total_density_m3,
            "charge_states_e": list(self.charge_states_e),
            "state_degeneracies": list(self.state_degeneracies),
            "transition_energies_eV_above_vb": (
                self.transition_energies_eV_above_vb.tolist()
            ),
            "capture_n_m3_s": self.capture_n_m3_s.tolist(),
            "capture_p_m3_s": self.capture_p_m3_s.tolist(),
            "emission_n_s1": self.emission_n_s1.tolist(),
            "emission_p_s1": self.emission_p_s1.tolist(),
            "forward_state_rate_s1": self.forward_state_rate_s1.tolist(),
            "backward_state_rate_s1": self.backward_state_rate_s1.tolist(),
            "state_probability": self.state_probability.tolist(),
            "state_probability_derivative_n_m3": (
                self.state_probability_derivative_n_m3.tolist()
            ),
            "state_probability_derivative_p_m3": (
                self.state_probability_derivative_p_m3.tolist()
            ),
            "master_matrix_s1": self.master_matrix_s1.tolist(),
            "master_matrix_derivative_n_m3_s1": (
                self.master_matrix_derivative_n_m3_s1.tolist()
            ),
            "master_matrix_derivative_p_m3_s1": (
                self.master_matrix_derivative_p_m3_s1.tolist()
            ),
            "master_residual_s1": self.master_residual_s1.tolist(),
            "transition_recombination_rate_m3_s": (
                self.transition_recombination_rate_m3_s.tolist()
            ),
            "transition_recombination_derivative_n_s1": (
                self.transition_recombination_derivative_n_s1.tolist()
            ),
            "transition_recombination_derivative_p_s1": (
                self.transition_recombination_derivative_p_s1.tolist()
            ),
            "charge_number_density_m3": self.charge_number_density_m3.tolist(),
            "charge_density_C_m3": self.charge_density_C_m3.tolist(),
            "charge_derivative_n_C": self.charge_derivative_n_C.tolist(),
            "charge_derivative_p_C": self.charge_derivative_p_C.tolist(),
            "charge_derivative_fixed_qf_C_m3_V": (
                self.charge_derivative_fixed_qf_C_m3_V.tolist()
            ),
            "total_recombination_rate_m3_s": (
                self.total_recombination_rate_m3_s.tolist()
            ),
            "total_recombination_derivative_n_s1": (
                self.total_recombination_derivative_n_s1.tolist()
            ),
            "total_recombination_derivative_p_s1": (
                self.total_recombination_derivative_p_s1.tolist()
            ),
            "recombination_derivative_fixed_qf_m3_s_V": (
                self.recombination_derivative_fixed_qf_m3_s_V.tolist()
            ),
            "minimum_state_probability": self.minimum_state_probability,
            "maximum_state_probability": self.maximum_state_probability,
            "maximum_probability_sum_error": self.maximum_probability_sum_error,
            "maximum_master_residual_s1": self.maximum_master_residual_s1,
        }


def evaluate_multivalent_defect_closure(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    species: MultivalentBulkDefectSpecies,
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> MultivalentDefectClosureResult:
    """Solve one local coupled charge-state system without clipping.

    States are ordered from most positive to most negative.  Adjacent state
    rates are ``a_s = n*c_n + e_p`` and ``b_s = p*c_p + e_n``.  Stationary
    weights are accumulated in log space, so a strongly dominant charge state
    does not require subtracting nearly equal densities.
    """

    if not isinstance(species, MultivalentBulkDefectSpecies):
        raise TypeError("species must be MultivalentBulkDefectSpecies")
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if (
        not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n <= 0.0)
        or np.any(p <= 0.0)
    ):
        raise ValueError(
            "multivalent closure carrier densities must be finite and positive"
        )
    gap = _positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    temperature = _positive(temperature_K, "temperature_K")
    species.validate_band_gap(gap)
    configuration = species.configuration
    thermal = thermal_voltage(temperature)

    energies = np.asarray(
        configuration.energy_levels.transition_energies_eV_above_vb,
        dtype=float,
    )
    capture_n = np.asarray(
        [
            value.sigma_n_m2 * value.thermal_velocity_n_m_s
            for value in configuration.transition_kinetics
        ],
        dtype=float,
    )
    capture_p = np.asarray(
        [
            value.sigma_p_m2 * value.thermal_velocity_p_m_s
            for value in configuration.transition_kinetics
        ],
        dtype=float,
    )
    degeneracies = np.asarray(configuration.state_degeneracies, dtype=float)
    degeneracy_ratio = degeneracies[1:] / degeneracies[:-1]
    n1 = conduction_dos * np.exp(-(gap - energies) / thermal)
    p1 = valence_dos * np.exp(-energies / thermal)
    emission_n = capture_n * n1 / degeneracy_ratio
    emission_p = capture_p * p1 * degeneracy_ratio
    references = (capture_n, capture_p, n1, p1, emission_n, emission_p)
    if not all(
        np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in references
    ):
        raise MultivalentDefectClosureError(
            "multivalent detailed-balance rates are non-finite"
        )

    transition_count = len(energies)
    state_count = transition_count + 1
    transition_expansion = (transition_count,) + (1,) * n.ndim
    state_shape = (state_count,) + n.shape
    matrix_shape = (state_count, state_count) + n.shape
    capture_n_e = capture_n.reshape(transition_expansion)
    capture_p_e = capture_p.reshape(transition_expansion)
    emission_n_e = emission_n.reshape(transition_expansion)
    emission_p_e = emission_p.reshape(transition_expansion)
    n_e = np.expand_dims(n, axis=0)
    p_e = np.expand_dims(p, axis=0)
    forward = capture_n_e * n_e + emission_p_e
    backward = emission_n_e + capture_p_e * p_e
    if (
        not np.all(np.isfinite(forward))
        or not np.all(np.isfinite(backward))
        or np.any(forward <= 0.0)
        or np.any(backward <= 0.0)
    ):
        raise MultivalentDefectClosureError(
            "multivalent adjacent-state rates must be finite and positive"
        )

    log_ratio = np.log(forward) - np.log(backward)
    log_weight = np.zeros(state_shape, dtype=float)
    log_weight[1:] = np.cumsum(log_ratio, axis=0)
    log_weight -= np.max(log_weight, axis=0, keepdims=True)
    weight = np.exp(log_weight)
    probability = weight / np.sum(weight, axis=0, keepdims=True)

    log_weight_derivative_n = np.zeros(state_shape, dtype=float)
    log_weight_derivative_p = np.zeros(state_shape, dtype=float)
    log_weight_derivative_n[1:] = np.cumsum(capture_n_e / forward, axis=0)
    log_weight_derivative_p[1:] = np.cumsum(-capture_p_e / backward, axis=0)
    mean_log_derivative_n = np.sum(
        probability * log_weight_derivative_n,
        axis=0,
        keepdims=True,
    )
    mean_log_derivative_p = np.sum(
        probability * log_weight_derivative_p,
        axis=0,
        keepdims=True,
    )
    probability_derivative_n = probability * (
        log_weight_derivative_n - mean_log_derivative_n
    )
    probability_derivative_p = probability * (
        log_weight_derivative_p - mean_log_derivative_p
    )

    master = np.zeros(matrix_shape, dtype=float)
    master_derivative_n = np.zeros(matrix_shape, dtype=float)
    master_derivative_p = np.zeros(matrix_shape, dtype=float)
    for index in range(transition_count):
        forward_i = forward[index]
        backward_i = backward[index]
        capture_n_i = capture_n[index]
        capture_p_i = capture_p[index]
        master[index, index] -= forward_i
        master[index + 1, index] += forward_i
        master[index + 1, index + 1] -= backward_i
        master[index, index + 1] += backward_i
        master_derivative_n[index, index] -= capture_n_i
        master_derivative_n[index + 1, index] += capture_n_i
        master_derivative_p[index + 1, index + 1] -= capture_p_i
        master_derivative_p[index, index + 1] += capture_p_i
    master_residual = np.einsum("ij...,j...->i...", master, probability)

    intrinsic_product = conduction_dos * valence_dos * math.exp(-gap / thermal)
    pair_probability = probability[:-1] + probability[1:]
    pair_derivative_n = probability_derivative_n[:-1] + probability_derivative_n[1:]
    pair_derivative_p = probability_derivative_p[:-1] + probability_derivative_p[1:]
    capture_product_e = (capture_n * capture_p).reshape(transition_expansion)
    numerator = capture_product_e * (n_e * p_e - intrinsic_product)
    denominator = forward + backward
    rate_fraction = numerator / denominator
    numerator_derivative_n = capture_product_e * p_e
    numerator_derivative_p = capture_product_e * n_e
    fraction_derivative_n = (
        numerator_derivative_n * denominator - numerator * capture_n_e
    ) / denominator**2
    fraction_derivative_p = (
        numerator_derivative_p * denominator - numerator * capture_p_e
    ) / denominator**2
    density = species.total_density_m3
    transition_rate = density * pair_probability * rate_fraction
    transition_derivative_n = density * (
        pair_derivative_n * rate_fraction + pair_probability * fraction_derivative_n
    )
    transition_derivative_p = density * (
        pair_derivative_p * rate_fraction + pair_probability * fraction_derivative_p
    )
    total_rate = np.sum(transition_rate, axis=0)
    total_derivative_n = np.sum(transition_derivative_n, axis=0)
    total_derivative_p = np.sum(transition_derivative_p, axis=0)

    charges_e = np.asarray(configuration.charge_states_e, dtype=float).reshape(
        (state_count,) + (1,) * n.ndim
    )
    charge_number = density * np.sum(charges_e * probability, axis=0)
    charge_derivative_n = (
        Q
        * density
        * np.sum(
            charges_e * probability_derivative_n,
            axis=0,
        )
    )
    charge_derivative_p = (
        Q
        * density
        * np.sum(
            charges_e * probability_derivative_p,
            axis=0,
        )
    )
    charge_density = Q * charge_number
    charge_derivative_fixed_qf = (
        charge_derivative_n * n - charge_derivative_p * p
    ) / thermal
    rate_derivative_fixed_qf = (
        total_derivative_n * n - total_derivative_p * p
    ) / thermal

    outputs = (
        probability,
        probability_derivative_n,
        probability_derivative_p,
        master,
        master_derivative_n,
        master_derivative_p,
        master_residual,
        transition_rate,
        transition_derivative_n,
        transition_derivative_p,
        charge_number,
        charge_density,
        charge_derivative_n,
        charge_derivative_p,
        charge_derivative_fixed_qf,
        total_rate,
        total_derivative_n,
        total_derivative_p,
        rate_derivative_fixed_qf,
    )
    if not all(np.all(np.isfinite(value)) for value in outputs):
        raise MultivalentDefectClosureError(
            "multivalent stationary closure produced non-finite output"
        )
    minimum = float(np.min(probability))
    maximum = float(np.max(probability))
    sum_error = float(np.max(np.abs(np.sum(probability, axis=0) - 1.0)))
    residual = float(np.max(np.abs(master_residual)))
    if minimum < 0.0 or maximum > 1.0:
        raise MultivalentDefectClosureError(
            "multivalent state probability left [0, 1] without clipping"
        )

    return MultivalentDefectClosureResult(
        closure_identity_sha256=_identity(
            species,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
        ),
        species_name=species.name,
        temperature_K=temperature,
        thermal_voltage_V=thermal,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        intrinsic_product_m6=intrinsic_product,
        total_density_m3=density,
        charge_states_e=configuration.charge_states_e,
        state_degeneracies=configuration.state_degeneracies,
        transition_energies_eV_above_vb=energies,
        capture_n_m3_s=capture_n,
        capture_p_m3_s=capture_p,
        emission_n_s1=emission_n,
        emission_p_s1=emission_p,
        forward_state_rate_s1=forward,
        backward_state_rate_s1=backward,
        state_probability=probability,
        state_probability_derivative_n_m3=probability_derivative_n,
        state_probability_derivative_p_m3=probability_derivative_p,
        master_matrix_s1=master,
        master_matrix_derivative_n_m3_s1=master_derivative_n,
        master_matrix_derivative_p_m3_s1=master_derivative_p,
        master_residual_s1=master_residual,
        transition_recombination_rate_m3_s=transition_rate,
        transition_recombination_derivative_n_s1=transition_derivative_n,
        transition_recombination_derivative_p_s1=transition_derivative_p,
        charge_number_density_m3=charge_number,
        charge_density_C_m3=charge_density,
        charge_derivative_n_C=charge_derivative_n,
        charge_derivative_p_C=charge_derivative_p,
        charge_derivative_fixed_qf_C_m3_V=charge_derivative_fixed_qf,
        total_recombination_rate_m3_s=total_rate,
        total_recombination_derivative_n_s1=total_derivative_n,
        total_recombination_derivative_p_s1=total_derivative_p,
        recombination_derivative_fixed_qf_m3_s_V=rate_derivative_fixed_qf,
        minimum_state_probability=minimum,
        maximum_state_probability=maximum,
        maximum_probability_sum_error=sum_error,
        maximum_master_residual_s1=residual,
    )


__all__ = [
    "MULTIVALENT_DEFECT_CLOSURE_VERSION",
    "MultivalentDefectClosureError",
    "MultivalentDefectClosureResult",
    "evaluate_multivalent_defect_closure",
]
