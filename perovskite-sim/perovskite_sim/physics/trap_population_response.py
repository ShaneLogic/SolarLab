"""Population-scaled local trap frequency responses.

The primitives in :mod:`perovskite_sim.physics.trap_kinetics` are normalized
per trap. This module binds those primitives to canonical bulk volume density
or interface areal density and cross-checks the resulting DC quantities
against the established quasi-steady closures.

This remains a local constitutive layer. It does not insert trap charge into a
device Poisson equation, carrier continuity equation, or terminal current.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    NEUTRAL,
    BulkDefectSpecies,
)
from perovskite_sim.models.interface_defects import InterfaceDefectDocument
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_defect_closure,
    evaluate_monovalent_source_defect_closure,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    DefectEnergyQuadrature,
    expand_bulk_defect_species_energy,
)
from perovskite_sim.physics.trap_kinetics import (
    TrapDCOperatingPoint,
    TrapFrequencyResponse,
    TrapReservoirKinetics,
    TrapReservoirState,
    evaluate_trap_dc_operating_point,
    linearize_trap_kinetics,
    solve_trap_frequency_response,
)
from perovskite_sim.physics.two_sided_interface import (
    TwoSidedInterfacePhysics,
    shared_trap_capture_flux,
    shared_trap_occupancy,
)


TRAP_POPULATION_RESPONSE_VERSION = "local-trap-population-frequency-v1"
BULK_POPULATION_MEASURE = "volume_density_m3"
INTERFACE_POPULATION_MEASURE = "areal_density_m2"
LOCAL_POPULATION_BINDING_SCOPE = "local_population_binding_only"
INTERFACE_RESERVOIR_ORDER = ("left", "right")
INTERFACE_CAPTURE_FLUX_ORDER = ("n_left", "p_left", "n_right", "p_right")


class TrapPopulationResponseError(ValueError):
    """A population binding is incomplete or inconsistent."""


class TrapPopulationCertificationError(RuntimeError):
    """A finite population response failed its DC closure cross-check."""

    def __init__(
        self,
        message: str,
        result: (
            "BulkDefectPopulationFrequencyResponse"
            " | InterfaceTrapPopulationFrequencyResponse"
        ),
    ) -> None:
        self.result = result
        super().__init__(message)


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise TrapPopulationResponseError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise TrapPopulationResponseError(f"{name} must be positive")
    return number


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise TrapPopulationResponseError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _frequencies(value: object) -> np.ndarray:
    frequencies = np.asarray(value, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise TrapPopulationResponseError(
            "frequencies_Hz must be finite, positive, and strictly increasing"
        )
    return frequencies


def _density_response(
    value: object,
    *,
    frequency_count: int,
    reservoir_count: int,
    name: str,
) -> np.ndarray:
    response = np.asarray(value, dtype=complex)
    if response.ndim == 0 and reservoir_count == 1:
        response = np.full((frequency_count, 1), response, dtype=complex)
    elif response.shape == (frequency_count,) and reservoir_count == 1:
        response = response[:, np.newaxis]
    elif response.shape == (reservoir_count,):
        response = np.broadcast_to(
            response,
            (frequency_count, reservoir_count),
        )
    if response.shape != (frequency_count, reservoir_count) or not np.all(
        np.isfinite(response)
    ):
        raise TrapPopulationResponseError(
            f"{name} must be finite with frequency-by-reservoir shape"
        )
    return response


def _relative_error(left: object, right: object, *, scale: float = 0.0) -> float:
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    denominator = np.maximum(np.abs(left_values), np.abs(right_values))
    denominator = np.maximum(denominator, max(float(scale), np.finfo(float).tiny))
    return float(np.max(np.abs(left_values - right_values) / denominator))


def _trap_balance_relative_error(
    response: TrapFrequencyResponse,
    population: float,
) -> float:
    omega = 2.0 * np.pi * response.frequencies_Hz
    electron = population * np.sum(
        response.electron_capture_response_s1_per_V,
        axis=1,
    )
    hole = population * np.sum(
        response.hole_capture_response_s1_per_V,
        axis=1,
    )
    occupancy = population * response.occupancy_response_per_V
    residual = electron - hole - 1j * omega * occupancy
    scale = np.abs(electron) + np.abs(hole) + np.abs(1j * omega * occupancy)
    return float(
        np.max(
            np.divide(
                np.abs(residual),
                scale,
                out=np.zeros_like(scale, dtype=float),
                where=scale > 0.0,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class BulkDefectPopulationFrequencyResponse:
    """Energy-resolved and integrated response of one bulk defect species."""

    source_species_sha256: str
    source_identifier: str
    charge_transition: str
    quadrature: DefectEnergyQuadrature
    frequencies_Hz: np.ndarray
    kinetics: tuple[TrapReservoirKinetics, ...]
    states: tuple[TrapReservoirState, ...]
    operating_points: tuple[TrapDCOperatingPoint, ...]
    node_responses: tuple[TrapFrequencyResponse, ...]
    electron_capture_response_m3_s_V: np.ndarray
    hole_capture_response_m3_s_V: np.ndarray
    charge_density_response_C_m3_V: np.ndarray
    quasistatic_recombination_response_m3_s_V: np.ndarray
    total_electron_capture_response_m3_s_V: np.ndarray
    total_hole_capture_response_m3_s_V: np.ndarray
    total_charge_density_response_C_m3_V: np.ndarray
    total_quasistatic_recombination_response_m3_s_V: np.ndarray
    maximum_dc_closure_relative_error: float
    maximum_quasistatic_tangent_relative_error: float
    maximum_local_balance_relative_error: float
    population_binding_certified: bool
    certification_scope: str = LOCAL_POPULATION_BINDING_SCOPE
    population_measure: str = BULK_POPULATION_MEASURE
    response_version: str = TRAP_POPULATION_RESPONSE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_species_sha256",
            _sha256(self.source_species_sha256, "source_species_sha256"),
        )
        identifier = str(self.source_identifier).strip()
        if not identifier:
            raise TrapPopulationResponseError("source_identifier must be non-empty")
        object.__setattr__(self, "source_identifier", identifier)
        transition = str(self.charge_transition).strip().lower()
        if transition not in {NEUTRAL, ACCEPTOR, DONOR}:
            raise TrapPopulationResponseError(
                "bulk response requires a resolved charge transition"
            )
        object.__setattr__(self, "charge_transition", transition)
        if not isinstance(self.quadrature, DefectEnergyQuadrature):
            raise TypeError("quadrature must be DefectEnergyQuadrature")
        frequencies = _frequencies(self.frequencies_Hz)
        object.__setattr__(self, "frequencies_Hz", _readonly(frequencies, dtype=float))
        node_count = self.quadrature.order
        nested = (
            self.kinetics,
            self.states,
            self.operating_points,
            self.node_responses,
        )
        if any(len(values) != node_count for values in nested):
            raise TrapPopulationResponseError(
                "bulk population nested results must match energy quadrature"
            )
        typed_nested = (
            (self.kinetics, TrapReservoirKinetics),
            (self.states, TrapReservoirState),
            (self.operating_points, TrapDCOperatingPoint),
            (self.node_responses, TrapFrequencyResponse),
        )
        if any(
            not all(isinstance(item, expected_type) for item in values)
            for values, expected_type in typed_nested
        ):
            raise TypeError("bulk population nested result types are invalid")
        for kinetics, state, point, response in zip(
            self.kinetics,
            self.states,
            self.operating_points,
            self.node_responses,
            strict=True,
        ):
            if (
                kinetics.electron_reservoir_count != 1
                or kinetics.hole_reservoir_count != 1
            ):
                raise TrapPopulationResponseError(
                    "bulk energy nodes require one electron and one hole reservoir"
                )
            if (
                point.kinetics_sha256 != kinetics.sha256
                or point.state_sha256 != state.sha256
            ):
                raise TrapPopulationResponseError(
                    "bulk operating point does not match its kinetics and state"
                )
            if not np.array_equal(response.frequencies_Hz, frequencies):
                raise TrapPopulationResponseError(
                    "bulk node response frequencies do not match the aggregate"
                )
            if response.charge_transition != transition:
                raise TrapPopulationResponseError(
                    "bulk node response charge transition does not match source"
                )
        node_shape = (node_count, frequencies.size)
        for name in (
            "electron_capture_response_m3_s_V",
            "hole_capture_response_m3_s_V",
            "charge_density_response_C_m3_V",
            "quasistatic_recombination_response_m3_s_V",
        ):
            values = _readonly(getattr(self, name), dtype=complex)
            if values.shape != node_shape or not np.all(np.isfinite(values)):
                raise TrapPopulationResponseError(
                    f"{name} must match nodes and frequency"
                )
            object.__setattr__(self, name, values)
        for name in (
            "total_electron_capture_response_m3_s_V",
            "total_hole_capture_response_m3_s_V",
            "total_charge_density_response_C_m3_V",
            "total_quasistatic_recombination_response_m3_s_V",
        ):
            values = _readonly(getattr(self, name), dtype=complex)
            if values.shape != frequencies.shape or not np.all(np.isfinite(values)):
                raise TrapPopulationResponseError(f"{name} must match frequency")
            object.__setattr__(self, name, values)
        aggregates = (
            (
                self.total_electron_capture_response_m3_s_V,
                self.electron_capture_response_m3_s_V,
            ),
            (
                self.total_hole_capture_response_m3_s_V,
                self.hole_capture_response_m3_s_V,
            ),
            (
                self.total_charge_density_response_C_m3_V,
                self.charge_density_response_C_m3_V,
            ),
            (
                self.total_quasistatic_recombination_response_m3_s_V,
                self.quasistatic_recombination_response_m3_s_V,
            ),
        )
        if any(
            not np.array_equal(total, np.sum(nodes, axis=0))
            for total, nodes in aggregates
        ):
            raise TrapPopulationResponseError(
                "bulk population totals must equal the node sums"
            )
        for name in (
            "maximum_dc_closure_relative_error",
            "maximum_quasistatic_tangent_relative_error",
            "maximum_local_balance_relative_error",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise TrapPopulationResponseError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.population_binding_certified, (bool, np.bool_)):
            raise TypeError("population_binding_certified must be boolean")
        object.__setattr__(
            self,
            "population_binding_certified",
            bool(self.population_binding_certified),
        )
        if self.certification_scope != LOCAL_POPULATION_BINDING_SCOPE:
            raise TrapPopulationResponseError("unsupported certification scope")
        if self.population_measure != BULK_POPULATION_MEASURE:
            raise TrapPopulationResponseError("unsupported bulk population measure")
        if self.response_version != TRAP_POPULATION_RESPONSE_VERSION:
            raise TrapPopulationResponseError("unsupported population response version")


@dataclass(frozen=True, slots=True)
class InterfaceTrapPopulationFrequencyResponse:
    """Areal shared-occupancy response with explicit left/right reservoirs."""

    document_sha256: str
    trap_binding_sha256: str
    trap_density_m2: float
    frequencies_Hz: np.ndarray
    kinetics: TrapReservoirKinetics
    state: TrapReservoirState
    operating_point: TrapDCOperatingPoint
    response: TrapFrequencyResponse
    dc_capture_flux_m2_s: np.ndarray
    electron_capture_flux_response_m2_s_V: np.ndarray
    hole_capture_flux_response_m2_s_V: np.ndarray
    sheet_charge_response_C_m2_V: np.ndarray
    maximum_dc_closure_relative_error: float
    maximum_local_charge_conservation_relative_error: float
    population_binding_certified: bool
    certification_scope: str = LOCAL_POPULATION_BINDING_SCOPE
    reservoir_order: tuple[str, str] = INTERFACE_RESERVOIR_ORDER
    capture_flux_order: tuple[str, str, str, str] = INTERFACE_CAPTURE_FLUX_ORDER
    population_measure: str = INTERFACE_POPULATION_MEASURE
    response_version: str = TRAP_POPULATION_RESPONSE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_sha256",
            _sha256(self.document_sha256, "document_sha256"),
        )
        object.__setattr__(
            self,
            "trap_binding_sha256",
            _sha256(self.trap_binding_sha256, "trap_binding_sha256"),
        )
        object.__setattr__(
            self,
            "trap_density_m2",
            _positive(self.trap_density_m2, "trap_density_m2"),
        )
        frequencies = _frequencies(self.frequencies_Hz)
        object.__setattr__(self, "frequencies_Hz", _readonly(frequencies, dtype=float))
        if not isinstance(self.kinetics, TrapReservoirKinetics):
            raise TypeError("kinetics must be TrapReservoirKinetics")
        if not isinstance(self.state, TrapReservoirState):
            raise TypeError("state must be TrapReservoirState")
        if not isinstance(self.operating_point, TrapDCOperatingPoint):
            raise TypeError("operating_point must be TrapDCOperatingPoint")
        if not isinstance(self.response, TrapFrequencyResponse):
            raise TypeError("response must be TrapFrequencyResponse")
        if (
            self.kinetics.electron_reservoir_count != 2
            or self.kinetics.hole_reservoir_count != 2
        ):
            raise TrapPopulationResponseError(
                "interface response requires two electron and two hole reservoirs"
            )
        if (
            self.operating_point.kinetics_sha256 != self.kinetics.sha256
            or self.operating_point.state_sha256 != self.state.sha256
        ):
            raise TrapPopulationResponseError(
                "interface operating point does not match its kinetics and state"
            )
        if not np.array_equal(self.response.frequencies_Hz, frequencies):
            raise TrapPopulationResponseError(
                "interface response frequencies do not match the aggregate"
            )
        dc_flux = _readonly(self.dc_capture_flux_m2_s, dtype=float)
        if dc_flux.shape != (4,) or not np.all(np.isfinite(dc_flux)):
            raise TrapPopulationResponseError(
                "dc_capture_flux_m2_s must use [nL,pL,nR,pR]"
            )
        object.__setattr__(self, "dc_capture_flux_m2_s", dc_flux)
        for name in (
            "electron_capture_flux_response_m2_s_V",
            "hole_capture_flux_response_m2_s_V",
        ):
            values = _readonly(getattr(self, name), dtype=complex)
            if values.shape != (frequencies.size, 2) or not np.all(np.isfinite(values)):
                raise TrapPopulationResponseError(
                    f"{name} must use frequency-by-[left,right] order"
                )
            object.__setattr__(self, name, values)
        charge = _readonly(self.sheet_charge_response_C_m2_V, dtype=complex)
        if charge.shape != frequencies.shape or not np.all(np.isfinite(charge)):
            raise TrapPopulationResponseError(
                "sheet_charge_response_C_m2_V must match frequency"
            )
        object.__setattr__(self, "sheet_charge_response_C_m2_V", charge)
        for name in (
            "maximum_dc_closure_relative_error",
            "maximum_local_charge_conservation_relative_error",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise TrapPopulationResponseError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.population_binding_certified, (bool, np.bool_)):
            raise TypeError("population_binding_certified must be boolean")
        object.__setattr__(
            self,
            "population_binding_certified",
            bool(self.population_binding_certified),
        )
        if self.certification_scope != LOCAL_POPULATION_BINDING_SCOPE:
            raise TrapPopulationResponseError("unsupported certification scope")
        if tuple(self.reservoir_order) != INTERFACE_RESERVOIR_ORDER:
            raise TrapPopulationResponseError("unsupported interface reservoir order")
        if tuple(self.capture_flux_order) != INTERFACE_CAPTURE_FLUX_ORDER:
            raise TrapPopulationResponseError(
                "unsupported interface capture-flux order"
            )
        if self.population_measure != INTERFACE_POPULATION_MEASURE:
            raise TrapPopulationResponseError(
                "unsupported interface population measure"
            )
        if self.response_version != TRAP_POPULATION_RESPONSE_VERSION:
            raise TrapPopulationResponseError("unsupported population response version")


def solve_bulk_defect_population_frequency_response(
    source_species: BulkDefectSpecies,
    electron_density_m3: object,
    hole_density_m3: object,
    frequencies_Hz: object,
    electron_density_response_m3_per_V: object,
    hole_density_response_m3_per_V: object,
    *,
    band_gap_eV: object,
    effective_conduction_dos_m3: object,
    effective_valence_dos_m3: object,
    temperature_K: object,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    max_crosscheck_relative_error: float = 1.0e-12,
    require_certified: bool = True,
) -> BulkDefectPopulationFrequencyResponse:
    """Bind one canonical bulk source to per-volume trap response."""
    if not isinstance(source_species, BulkDefectSpecies):
        raise TypeError("source_species must be BulkDefectSpecies")
    if source_species.spatial_profile is not None:
        raise TrapPopulationResponseError(
            "local bulk response requires a position-resolved species without "
            "a spatial_profile"
        )
    if source_species.name is None:
        raise TrapPopulationResponseError("bulk response requires a named species")
    n = _finite(electron_density_m3, "electron_density_m3")
    p = _finite(hole_density_m3, "hole_density_m3")
    if n < 0.0 or p < 0.0:
        raise TrapPopulationResponseError("carrier densities must be non-negative")
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
    tolerance = _positive(
        max_crosscheck_relative_error,
        "max_crosscheck_relative_error",
    )
    frequencies = _frequencies(frequencies_Hz)
    electron_response = _density_response(
        electron_density_response_m3_per_V,
        frequency_count=frequencies.size,
        reservoir_count=1,
        name="electron_density_response_m3_per_V",
    )
    hole_response = _density_response(
        hole_density_response_m3_per_V,
        frequency_count=frequencies.size,
        reservoir_count=1,
        name="hole_density_response_m3_per_V",
    )
    expansion = expand_bulk_defect_species_energy(
        source_species,
        band_gap_eV=gap,
        order=energy_quadrature_order,
    )
    nodes = expansion.node_species
    closure = evaluate_monovalent_defect_closure(
        n,
        p,
        nodes,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        temperature_K=temperature,
    )
    source_closure = evaluate_monovalent_source_defect_closure(
        n,
        p,
        (source_species,),
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        temperature_K=temperature,
        energy_quadrature_order=energy_quadrature_order,
        energy_expansions=(expansion,),
    )

    kinetics_values: list[TrapReservoirKinetics] = []
    states: list[TrapReservoirState] = []
    points: list[TrapDCOperatingPoint] = []
    responses: list[TrapFrequencyResponse] = []
    electron_nodes: list[np.ndarray] = []
    hole_nodes: list[np.ndarray] = []
    charge_nodes: list[np.ndarray] = []
    quasistatic_nodes: list[np.ndarray] = []
    dc_errors: list[float] = []
    balance_errors: list[float] = []
    state = TrapReservoirState(
        electron_densities_m3=np.array([n]),
        hole_densities_m3=np.array([p]),
    )
    for index, node in enumerate(nodes):
        population = node.distribution.total_density_m3
        kinetics = TrapReservoirKinetics(
            identifier=f"bulk/{source_species.name}/energy[{index:03d}]",
            electron_capture_coefficients_m3_s=np.array(
                [closure.capture_n_m3_s[index]]
            ),
            hole_capture_coefficients_m3_s=np.array([closure.capture_p_m3_s[index]]),
            electron_reference_densities_m3=np.array([closure.n1_m3[index]]),
            hole_reference_densities_m3=np.array([closure.p1_m3[index]]),
        )
        point = evaluate_trap_dc_operating_point(
            kinetics,
            state,
            occupancy=float(closure.occupancy[index]),
        )
        response = solve_trap_frequency_response(
            kinetics,
            state,
            point,
            frequencies,
            electron_response,
            hole_response,
            charge_transition=source_species.charge_transition,
        )
        electron_capture = (
            population * response.electron_capture_response_s1_per_V[:, 0]
        )
        hole_capture = population * response.hole_capture_response_s1_per_V[:, 0]
        charge = population * response.charge_per_trap_response_C_per_V
        linearization = linearize_trap_kinetics(kinetics, state, point)
        quasistatic_electron = population * (
            electron_response[:, 0] * linearization.electron_density_forcing_m3_s[0]
            + response.quasistatic_occupancy_response_per_V
            * linearization.electron_capture_occupancy_derivative_s1[0]
        )
        quasistatic_hole = population * (
            hole_response[:, 0]
            * kinetics.hole_capture_coefficients_m3_s[0]
            * point.occupancy
            + response.quasistatic_occupancy_response_per_V
            * linearization.hole_capture_occupancy_derivative_s1[0]
        )
        population_scale = population * point.relaxation_rate_s1
        existing_rate = float(closure.recombination_rate_m3_s[index])
        dc_errors.extend(
            (
                _relative_error(
                    population * point.electron_capture_rates_s1[0],
                    existing_rate,
                    scale=population_scale,
                ),
                _relative_error(
                    population * point.hole_capture_rates_s1[0],
                    existing_rate,
                    scale=population_scale,
                ),
                _relative_error(
                    point.relaxation_rate_s1,
                    closure.kinetic_denominator_s1[index],
                    scale=point.relaxation_rate_s1,
                ),
                _relative_error(
                    point.occupancy,
                    closure.occupancy[index],
                    scale=1.0,
                ),
            )
        )
        dc_errors.append(
            _relative_error(
                quasistatic_electron,
                quasistatic_hole,
                scale=float(np.max(np.abs(quasistatic_electron))),
            )
        )
        balance_errors.append(_trap_balance_relative_error(response, population))
        kinetics_values.append(kinetics)
        states.append(state)
        points.append(point)
        responses.append(response)
        electron_nodes.append(electron_capture)
        hole_nodes.append(hole_capture)
        charge_nodes.append(charge)
        quasistatic_nodes.append(quasistatic_electron)

    electron_array = np.asarray(electron_nodes)
    hole_array = np.asarray(hole_nodes)
    charge_array = np.asarray(charge_nodes)
    quasistatic_array = np.asarray(quasistatic_nodes)
    total_electron = np.sum(electron_array, axis=0)
    total_hole = np.sum(hole_array, axis=0)
    total_charge = np.sum(charge_array, axis=0)
    total_quasistatic = np.sum(quasistatic_array, axis=0)
    expected_quasistatic = (
        float(source_closure.total_recombination_derivative_n_s1)
        * electron_response[:, 0]
        + float(source_closure.total_recombination_derivative_p_s1)
        * hole_response[:, 0]
    )
    tangent_error = _relative_error(
        total_quasistatic,
        expected_quasistatic,
        scale=float(np.max(np.abs(expected_quasistatic))),
    )
    maximum_dc_error = max(dc_errors, default=0.0)
    maximum_balance_error = max(balance_errors, default=0.0)
    certified = bool(
        maximum_dc_error <= tolerance
        and tangent_error <= tolerance
        and maximum_balance_error <= tolerance
        and all(point.certified for point in points)
    )
    result = BulkDefectPopulationFrequencyResponse(
        source_species_sha256=_canonical_sha256(source_species.to_dict()),
        source_identifier=source_species.name,
        charge_transition=source_species.charge_transition,
        quadrature=expansion.quadrature,
        frequencies_Hz=frequencies,
        kinetics=tuple(kinetics_values),
        states=tuple(states),
        operating_points=tuple(points),
        node_responses=tuple(responses),
        electron_capture_response_m3_s_V=electron_array,
        hole_capture_response_m3_s_V=hole_array,
        charge_density_response_C_m3_V=charge_array,
        quasistatic_recombination_response_m3_s_V=quasistatic_array,
        total_electron_capture_response_m3_s_V=total_electron,
        total_hole_capture_response_m3_s_V=total_hole,
        total_charge_density_response_C_m3_V=total_charge,
        total_quasistatic_recombination_response_m3_s_V=total_quasistatic,
        maximum_dc_closure_relative_error=maximum_dc_error,
        maximum_quasistatic_tangent_relative_error=tangent_error,
        maximum_local_balance_relative_error=maximum_balance_error,
        population_binding_certified=certified,
    )
    if require_certified and not certified:
        raise TrapPopulationCertificationError(
            "bulk trap population response failed closure cross-check",
            result,
        )
    return result


def _interface_binding_sha256(
    document: InterfaceDefectDocument,
    physics: TwoSidedInterfacePhysics,
) -> str:
    return _canonical_sha256(
        {
            "document_sha256": document.sha256,
            "surface_recombination_velocity_n_m_s": (
                physics.surface_recombination_velocity_n_m_s
            ),
            "surface_recombination_velocity_p_m_s": (
                physics.surface_recombination_velocity_p_m_s
            ),
            "n1_left_m3": physics.n1_left_m3,
            "n1_right_m3": physics.n1_right_m3,
            "p1_left_m3": physics.p1_left_m3,
            "p1_right_m3": physics.p1_right_m3,
            "reservoir_order": list(INTERFACE_RESERVOIR_ORDER),
            "capture_flux_order": list(INTERFACE_CAPTURE_FLUX_ORDER),
        }
    )


def solve_two_sided_interface_trap_population_frequency_response(
    document: InterfaceDefectDocument,
    physics: TwoSidedInterfacePhysics,
    state_m3: object,
    frequencies_Hz: object,
    electron_density_response_m3_per_V: object,
    hole_density_response_m3_per_V: object,
    *,
    max_crosscheck_relative_error: float = 1.0e-12,
    require_certified: bool = True,
) -> InterfaceTrapPopulationFrequencyResponse:
    """Bind a microscopic interface document to left/right trap response."""
    if not isinstance(document, InterfaceDefectDocument):
        raise TypeError("document must be InterfaceDefectDocument")
    if not isinstance(physics, TwoSidedInterfacePhysics):
        raise TypeError("physics must be TwoSidedInterfacePhysics")
    if document.degeneracy != 1.0:
        raise TrapPopulationResponseError(
            "interface trap response requires degeneracy=1.0"
        )
    expected_velocities = document.capture_velocities_m_s
    actual_velocities = (
        float(physics.surface_recombination_velocity_n_m_s),
        float(physics.surface_recombination_velocity_p_m_s),
    )
    if actual_velocities != expected_velocities:
        raise TrapPopulationResponseError(
            "interface surface velocities must exactly equal sigma*v_th*N_t"
        )
    if expected_velocities == (0.0, 0.0):
        raise TrapPopulationResponseError(
            "interface occupancy is undefined when both capture legs are zero"
        )
    tolerance = _positive(
        max_crosscheck_relative_error,
        "max_crosscheck_relative_error",
    )
    state_values = np.asarray(state_m3, dtype=float)
    if (
        state_values.shape != (4,)
        or not np.all(np.isfinite(state_values))
        or np.any(state_values < 0.0)
    ):
        raise TrapPopulationResponseError(
            "state_m3 must be finite [n_left,p_left,n_right,p_right]"
        )
    reference_values = np.array(
        [
            physics.n1_left_m3,
            physics.n1_right_m3,
            physics.p1_left_m3,
            physics.p1_right_m3,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(reference_values)) or np.any(reference_values < 0.0):
        raise TrapPopulationResponseError(
            "interface emission reference densities must be finite and non-negative"
        )
    frequencies = _frequencies(frequencies_Hz)
    electron_response = _density_response(
        electron_density_response_m3_per_V,
        frequency_count=frequencies.size,
        reservoir_count=2,
        name="electron_density_response_m3_per_V",
    )
    hole_response = _density_response(
        hole_density_response_m3_per_V,
        frequency_count=frequencies.size,
        reservoir_count=2,
        name="hole_density_response_m3_per_V",
    )
    capture_n = document.kinetics.capture_coefficient_n_m3_s
    capture_p = document.kinetics.capture_coefficient_p_m3_s
    kinetics = TrapReservoirKinetics(
        identifier=f"interface/{document.sha256}",
        electron_capture_coefficients_m3_s=np.array([capture_n, capture_n]),
        hole_capture_coefficients_m3_s=np.array([capture_p, capture_p]),
        electron_reference_densities_m3=reference_values[:2],
        hole_reference_densities_m3=reference_values[2:],
    )
    state = TrapReservoirState(
        electron_densities_m3=state_values[[0, 2]],
        hole_densities_m3=state_values[[1, 3]],
    )
    existing_occupancy = shared_trap_occupancy(state_values, physics)
    point = evaluate_trap_dc_operating_point(
        kinetics,
        state,
        occupancy=existing_occupancy,
    )
    response = solve_trap_frequency_response(
        kinetics,
        state,
        point,
        frequencies,
        electron_response,
        hole_response,
        charge_transition=ACCEPTOR,
    )
    population = document.total_density_m2
    existing_flux = shared_trap_capture_flux(state_values, physics)
    population_flux = np.array(
        [
            population * point.electron_capture_rates_s1[0],
            population * point.hole_capture_rates_s1[0],
            population * point.electron_capture_rates_s1[1],
            population * point.hole_capture_rates_s1[1],
        ]
    )
    dc_scale = population * point.relaxation_rate_s1
    dc_error = max(
        _relative_error(existing_occupancy, point.occupancy, scale=1.0),
        _relative_error(existing_flux, population_flux, scale=dc_scale),
    )
    electron_flux_response = population * response.electron_capture_response_s1_per_V
    hole_flux_response = population * response.hole_capture_response_s1_per_V
    sheet_charge_response = population * response.charge_per_trap_response_C_per_V
    omega = 2.0 * np.pi * frequencies
    conservation_residual = (
        Q
        * (np.sum(electron_flux_response, axis=1) - np.sum(hole_flux_response, axis=1))
        + 1j * omega * sheet_charge_response
    )
    conservation_scale = Q * (
        np.sum(np.abs(electron_flux_response), axis=1)
        + np.sum(np.abs(hole_flux_response), axis=1)
    ) + np.abs(1j * omega * sheet_charge_response)
    conservation_error = float(
        np.max(
            np.divide(
                np.abs(conservation_residual),
                conservation_scale,
                out=np.zeros_like(conservation_scale, dtype=float),
                where=conservation_scale > 0.0,
            )
        )
    )
    certified = bool(
        point.certified and dc_error <= tolerance and conservation_error <= tolerance
    )
    result = InterfaceTrapPopulationFrequencyResponse(
        document_sha256=document.sha256,
        trap_binding_sha256=_interface_binding_sha256(document, physics),
        trap_density_m2=document.total_density_m2,
        frequencies_Hz=frequencies,
        kinetics=kinetics,
        state=state,
        operating_point=point,
        response=response,
        dc_capture_flux_m2_s=existing_flux,
        electron_capture_flux_response_m2_s_V=electron_flux_response,
        hole_capture_flux_response_m2_s_V=hole_flux_response,
        sheet_charge_response_C_m2_V=sheet_charge_response,
        maximum_dc_closure_relative_error=dc_error,
        maximum_local_charge_conservation_relative_error=conservation_error,
        population_binding_certified=certified,
    )
    if require_certified and not certified:
        raise TrapPopulationCertificationError(
            "interface trap population response failed closure cross-check",
            result,
        )
    return result


__all__ = [
    "BULK_POPULATION_MEASURE",
    "INTERFACE_CAPTURE_FLUX_ORDER",
    "INTERFACE_POPULATION_MEASURE",
    "INTERFACE_RESERVOIR_ORDER",
    "LOCAL_POPULATION_BINDING_SCOPE",
    "TRAP_POPULATION_RESPONSE_VERSION",
    "BulkDefectPopulationFrequencyResponse",
    "InterfaceTrapPopulationFrequencyResponse",
    "TrapPopulationCertificationError",
    "TrapPopulationResponseError",
    "solve_bulk_defect_population_frequency_response",
    "solve_two_sided_interface_trap_population_frequency_response",
]
