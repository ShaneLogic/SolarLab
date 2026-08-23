"""Reference frequency-domain impedance about a certified mobile-ion DC state.

The adapter deliberately starts with central finite differences.  It uses
dimensionless log-density increments for every dynamic density, re-solves the
eliminated Poisson equation at every stencil point, and therefore retains the
global electrostatic derivative without pretending that Poisson is a dynamic
storage row.  The reference operator is expensive but explicit enough to
certify later analytic or structured Jacobians column by column.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Literal, Mapping, Self

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.impedance_frequency import (
    FrequencyWindowAssessment,
    assess_impedance_frequency_window,
)
from perovskite_sim.experiments.ion_aware_dc import (
    IonAwareDCResult,
    assess_ion_aware_dc_state,
    ion_aware_dc_state_sha256,
)
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    compute_ionic_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.solver.mol import (
    MaterialArrays,
    assemble_rhs,
    build_material_arrays,
    _harmonic_face_average,
)
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    SmallSignalLinearizationError,
    solve_frequency_domain,
)


ION_AWARE_IMPEDANCE_PROTOCOL_SCHEMA = "ion-aware-impedance-protocol-v2"
MAX_LINEAR_PERTURBATION_V = 0.02
DEFAULT_REFINEMENT_FACTORS = (1.0, 0.5, 0.25)
ProgressCallback = Callable[[str, int, int, str], None]


class IonAwareImpedanceCapabilityError(ValueError):
    """The requested state or topology cannot enter the reference lane."""


class IonAwareImpedanceError(SmallSignalLinearizationError):
    """The mobile-ion reference linearization could not be evaluated."""


class IonAwareImpedanceCertificationError(RuntimeError):
    """A finite response failed one or more declared numerical gates."""

    def __init__(self, message: str, result: "IonAwareImpedanceResult") -> None:
        self.result = result
        super().__init__(message)


def _finite(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _positive(value: object, field: str) -> float:
    number = _finite(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class IonAwareImpedanceProtocol:
    """Immutable DC identity, AC request, and numerical acceptance contract."""

    V_dc: float
    illuminated: bool
    temperature_K: float
    dc_protocol_sha256: str
    dc_state_sha256: str
    frequencies_Hz: tuple[float, ...]
    delta_V: float = 0.01
    state_step: float = 1.0e-5
    voltage_step: float = 1.0e-5
    refinement_factors: tuple[float, ...] = DEFAULT_REFINEMENT_FACTORS
    max_relative_face_spread: float = 5.0e-4
    max_backward_error: float = 1.0e-10
    max_impedance_magnitude_relative_change: float = 1.0e-2
    max_impedance_phase_change_deg: float = 0.5
    max_mass_matrix_relative_error: float = 1.0e-8
    max_ion_inventory_response_relative: float = 1.0e-8
    max_current_decomposition_relative_error: float = 1.0e-7
    frequency_branch_margin_decades: float = 1.0
    max_frequency_sampling_gap_decades: float = 0.5
    ion_boundary_condition: Literal["blocking"] = "blocking"
    coordinate_mode: Literal["log_density_increment"] = "log_density_increment"
    current_convention: Literal["passive"] = "passive"
    schema_version: Literal["ion-aware-impedance-protocol-v2"] = (
        ION_AWARE_IMPEDANCE_PROTOCOL_SCHEMA
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "V_dc", _finite(self.V_dc, "V_dc"))
        if not isinstance(self.illuminated, bool):
            raise TypeError("illuminated must be boolean")
        object.__setattr__(
            self,
            "temperature_K",
            _positive(self.temperature_K, "temperature_K"),
        )
        for name in ("dc_protocol_sha256", "dc_state_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        try:
            frequencies = tuple(
                _positive(value, f"frequencies_Hz[{index}]")
                for index, value in enumerate(self.frequencies_Hz)
            )
        except TypeError as exc:
            raise TypeError("frequencies_Hz must be an iterable") from exc
        if not frequencies or any(
            right <= left for left, right in zip(frequencies, frequencies[1:])
        ):
            raise ValueError("frequencies_Hz must be non-empty and increasing")
        object.__setattr__(self, "frequencies_Hz", frequencies)
        delta = _positive(self.delta_V, "delta_V")
        if delta >= MAX_LINEAR_PERTURBATION_V:
            raise ValueError("delta_V must be below the 20 mV small-signal limit")
        object.__setattr__(self, "delta_V", delta)
        for name in (
            "state_step",
            "voltage_step",
            "max_relative_face_spread",
            "max_backward_error",
            "max_impedance_magnitude_relative_change",
            "max_impedance_phase_change_deg",
            "max_mass_matrix_relative_error",
            "max_ion_inventory_response_relative",
            "max_current_decomposition_relative_error",
            "frequency_branch_margin_decades",
            "max_frequency_sampling_gap_decades",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        try:
            factors = tuple(
                _positive(value, f"refinement_factors[{index}]")
                for index, value in enumerate(self.refinement_factors)
            )
        except TypeError as exc:
            raise TypeError("refinement_factors must be an iterable") from exc
        if (
            len(factors) < 3
            or factors[0] != 1.0
            or any(right >= left for left, right in zip(factors, factors[1:]))
        ):
            raise ValueError(
                "refinement_factors must start at 1 and contain at least three "
                "strictly decreasing positive levels"
            )
        object.__setattr__(self, "refinement_factors", factors)
        if self.ion_boundary_condition != "blocking":
            raise ValueError("ion-aware impedance supports blocking ions only")
        if self.coordinate_mode != "log_density_increment":
            raise ValueError("unsupported ion-aware impedance coordinate mode")
        if self.current_convention != "passive":
            raise ValueError("ion-aware impedance uses the passive convention")
        if self.schema_version != ION_AWARE_IMPEDANCE_PROTOCOL_SCHEMA:
            raise ValueError("unsupported ion-aware impedance protocol schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["frequencies_Hz"] = list(self.frequencies_Hz)
        payload["refinement_factors"] = list(self.refinement_factors)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("ion-aware impedance protocol must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "ion-aware impedance protocol keys do not match schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        values = dict(payload)
        values["frequencies_Hz"] = tuple(values["frequencies_Hz"])
        values["refinement_factors"] = tuple(values["refinement_factors"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("ion-aware impedance protocol JSON must contain an object")
        return cls.from_dict(parsed)


def build_ion_aware_impedance_protocol(
    dc_state: IonAwareDCResult,
    frequencies_Hz: np.ndarray | tuple[float, ...],
    *,
    delta_V: float = 0.01,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    refinement_factors: tuple[float, ...] = DEFAULT_REFINEMENT_FACTORS,
    max_relative_face_spread: float = 5.0e-4,
    max_backward_error: float = 1.0e-10,
    max_impedance_magnitude_relative_change: float = 1.0e-2,
    max_impedance_phase_change_deg: float = 0.5,
    max_mass_matrix_relative_error: float = 1.0e-8,
    max_ion_inventory_response_relative: float = 1.0e-8,
    max_current_decomposition_relative_error: float = 1.0e-7,
    frequency_branch_margin_decades: float = 1.0,
    max_frequency_sampling_gap_decades: float = 0.5,
) -> IonAwareImpedanceProtocol:
    """Bind a small-signal request to one exact certified DC result."""
    if not isinstance(dc_state, IonAwareDCResult):
        raise TypeError("dc_state must be an IonAwareDCResult")
    return IonAwareImpedanceProtocol(
        V_dc=dc_state.protocol.V_dc,
        illuminated=dc_state.protocol.illuminated,
        temperature_K=dc_state.protocol.temperature_K,
        dc_protocol_sha256=dc_state.protocol_hash,
        dc_state_sha256=ion_aware_dc_state_sha256(dc_state.y),
        frequencies_Hz=tuple(np.asarray(frequencies_Hz, dtype=float).tolist()),
        delta_V=delta_V,
        state_step=state_step,
        voltage_step=voltage_step,
        refinement_factors=refinement_factors,
        max_relative_face_spread=max_relative_face_spread,
        max_backward_error=max_backward_error,
        max_impedance_magnitude_relative_change=(
            max_impedance_magnitude_relative_change
        ),
        max_impedance_phase_change_deg=max_impedance_phase_change_deg,
        max_mass_matrix_relative_error=max_mass_matrix_relative_error,
        max_ion_inventory_response_relative=(
            max_ion_inventory_response_relative
        ),
        max_current_decomposition_relative_error=(
            max_current_decomposition_relative_error
        ),
        frequency_branch_margin_decades=frequency_branch_margin_decades,
        max_frequency_sampling_gap_decades=(
            max_frequency_sampling_gap_decades
        ),
    )


@dataclass(frozen=True, slots=True)
class IonAwareStateCoordinateLayout:
    """Packed physical indices represented by log-density increments."""

    n_nodes: int
    electron_state_indices: tuple[int, ...]
    hole_state_indices: tuple[int, ...]
    positive_ion_state_indices: tuple[int, ...]
    negative_ion_state_indices: tuple[int, ...]

    @property
    def state_indices(self) -> tuple[int, ...]:
        return (
            self.electron_state_indices
            + self.hole_state_indices
            + self.positive_ion_state_indices
            + self.negative_ion_state_indices
        )

    @property
    def size(self) -> int:
        return len(self.state_indices)

    def coordinate_slice(self, species: str) -> slice:
        sizes = {
            "electron": len(self.electron_state_indices),
            "hole": len(self.hole_state_indices),
            "positive_ion": len(self.positive_ion_state_indices),
            "negative_ion": len(self.negative_ion_state_indices),
        }
        if species not in sizes:
            raise KeyError(species)
        order = ("electron", "hole", "positive_ion", "negative_ion")
        start = sum(sizes[name] for name in order[: order.index(species)])
        return slice(start, start + sizes[species])

    def node_indices(self, species: str) -> np.ndarray:
        state_indices = {
            "electron": self.electron_state_indices,
            "hole": self.hole_state_indices,
            "positive_ion": self.positive_ion_state_indices,
            "negative_ion": self.negative_ion_state_indices,
        }[species]
        return np.asarray(state_indices, dtype=int) % self.n_nodes


@dataclass(frozen=True, slots=True)
class PerturbationStepAssessment:
    coarse_factor: float
    fine_factor: float
    max_impedance_magnitude_relative_change: float
    max_impedance_phase_change_deg: float
    passed: bool


@dataclass(frozen=True, slots=True)
class IonAwareImpedanceCertificate:
    numerically_certified: bool
    thermodynamically_certified: bool
    certified: bool
    max_relative_face_spread: float
    max_backward_error: float
    minimum_reciprocal_condition: float
    max_mass_diagonal_relative_error: float
    max_mass_off_diagonal_relative: float
    max_ion_inventory_response_relative: float
    max_current_decomposition_relative_error: float
    frequency_window_certified: bool
    perturbation_assessments: tuple[PerturbationStepAssessment, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareImpedanceResult:
    frequencies: np.ndarray
    Z: np.ndarray
    Y: np.ndarray
    Y_faces: np.ndarray
    conduction_admittance_faces_S_m2: np.ndarray
    displacement_admittance_faces_S_m2: np.ndarray
    electron_admittance_faces_S_m2: np.ndarray
    hole_admittance_faces_S_m2: np.ndarray
    positive_ion_admittance_faces_S_m2: np.ndarray
    negative_ion_admittance_faces_S_m2: np.ndarray | None
    electron_storage_response_F_m2: np.ndarray
    hole_storage_response_F_m2: np.ndarray
    positive_ion_storage_response_F_m2: np.ndarray
    negative_ion_storage_response_F_m2: np.ndarray | None
    net_charge_storage_response_F_m2: np.ndarray
    state_response_per_V: np.ndarray
    storage_response_per_V: np.ndarray
    coordinate_layout: IonAwareStateCoordinateLayout
    reference_linearization: FrequencyDomainResult
    reference_linearizations: tuple[FrequencyDomainResult, ...]
    frequency_window: FrequencyWindowAssessment
    protocol: IonAwareImpedanceProtocol
    dc_state: IonAwareDCResult
    certificate: IonAwareImpedanceCertificate

    @property
    def protocol_hash(self) -> str:
        return self.protocol.protocol_hash


def _state_coordinate_layout(
    material: MaterialArrays,
    n_nodes: int,
) -> IonAwareStateCoordinateLayout:
    def carrier_indices(offset: int, left_free: bool, right_free: bool) -> tuple[int, ...]:
        selected = np.ones(n_nodes, dtype=bool)
        selected[0] = left_free
        selected[-1] = right_free
        return tuple((offset + np.flatnonzero(selected)).tolist())

    electron = carrier_indices(
        0,
        material.has_selective_contacts and material.S_n_L is not None,
        material.has_selective_contacts and material.S_n_R is not None,
    )
    hole = carrier_indices(
        n_nodes,
        material.has_selective_contacts and material.S_p_L is not None,
        material.has_selective_contacts and material.S_p_R is not None,
    )
    positive_active = (
        (np.asarray(material.P_ion0, dtype=float) > 0.0)
        & (np.asarray(material.D_ion_node, dtype=float) > 0.0)
    )
    positive = tuple((2 * n_nodes + np.flatnonzero(positive_active)).tolist())
    negative: tuple[int, ...] = ()
    if material.has_dual_ions:
        negative_active = (
            (np.asarray(material.P_ion0_neg, dtype=float) > 0.0)
            & (np.asarray(material.D_ion_neg_node, dtype=float) > 0.0)
        )
        negative = tuple((3 * n_nodes + np.flatnonzero(negative_active)).tolist())
    if not positive and not negative:
        raise IonAwareImpedanceCapabilityError(
            "ion-aware impedance requires an active mobile-ion coordinate"
        )
    return IonAwareStateCoordinateLayout(
        n_nodes=n_nodes,
        electron_state_indices=electron,
        hole_state_indices=hole,
        positive_ion_state_indices=positive,
        negative_ion_state_indices=negative,
    )


def _physical_state(
    coordinate: np.ndarray,
    base_state: np.ndarray,
    layout: IonAwareStateCoordinateLayout,
) -> np.ndarray:
    values = np.asarray(coordinate, dtype=float)
    if values.shape != (layout.size,) or not np.all(np.isfinite(values)):
        raise IonAwareImpedanceError(
            f"state coordinate must be a finite vector of length {layout.size}"
        )
    if np.any(values > np.log(np.finfo(float).max)):
        raise IonAwareImpedanceError("log-density coordinate overflow")
    physical = np.asarray(base_state, dtype=float).copy()
    indices = np.asarray(layout.state_indices, dtype=int)
    physical[indices] *= np.exp(values)
    if not np.all(np.isfinite(physical)) or np.any(physical[indices] <= 0.0):
        raise IonAwareImpedanceError(
            "log-density coordinate produced a non-positive or non-finite state"
        )
    return physical


def _stencil_occupancy_admissible(
    base_state: np.ndarray,
    layout: IonAwareStateCoordinateLayout,
    material: MaterialArrays,
    state_step: float,
) -> bool:
    n_nodes = layout.n_nodes
    positive = np.asarray(base_state[2 * n_nodes : 3 * n_nodes], dtype=float)
    negative = (
        np.asarray(base_state[3 * n_nodes : 4 * n_nodes], dtype=float)
        if material.has_dual_ions
        else None
    )
    multiplier = float(np.exp(state_step))
    for state_index in layout.positive_ion_state_indices:
        node = state_index - 2 * n_nodes
        trial = positive[node] * multiplier
        if material.ion_steric_shared_site and negative is not None:
            trial += negative[node]
            limit = min(material.P_lim_node[node], material.P_lim_neg_node[node])
        else:
            limit = material.P_lim_node[node]
        if trial > float(limit) * (1.0 + 1.0e-8):
            return False
    for state_index in layout.negative_ion_state_indices:
        node = state_index - 3 * n_nodes
        trial = negative[node] * multiplier
        if material.ion_steric_shared_site:
            trial += positive[node]
            limit = min(material.P_lim_node[node], material.P_lim_neg_node[node])
        else:
            limit = material.P_lim_neg_node[node]
        if trial > float(limit) * (1.0 + 1.0e-8):
            return False
    return True


def _storage_integral(
    response: np.ndarray,
    layout: IonAwareStateCoordinateLayout,
    species: str,
    widths: np.ndarray,
) -> np.ndarray | None:
    coordinate_slice = layout.coordinate_slice(species)
    if coordinate_slice.stop == coordinate_slice.start:
        return None
    nodes = layout.node_indices(species)
    return Q * (response[:, coordinate_slice] @ widths[nodes])


def _inventory_response_relative(
    response: np.ndarray,
    layout: IonAwareStateCoordinateLayout,
    species: str,
    widths: np.ndarray,
) -> float:
    coordinate_slice = layout.coordinate_slice(species)
    if coordinate_slice.stop == coordinate_slice.start:
        return 0.0
    values = response[:, coordinate_slice]
    node_widths = widths[layout.node_indices(species)]
    inventory = values @ node_widths
    scale = np.abs(values) @ node_widths
    ratio = np.divide(
        np.abs(inventory),
        scale,
        out=np.zeros_like(scale, dtype=float),
        where=scale > np.finfo(float).tiny,
    )
    return float(np.max(ratio))


def _perturbation_assessment(
    coarse_factor: float,
    fine_factor: float,
    coarse: FrequencyDomainResult,
    fine: FrequencyDomainResult,
    protocol: IonAwareImpedanceProtocol,
) -> PerturbationStepAssessment:
    magnitude_scale = np.maximum(np.abs(fine.impedance), np.finfo(float).tiny)
    magnitude_change = float(
        np.max(np.abs(np.abs(fine.impedance) - np.abs(coarse.impedance)) / magnitude_scale)
    )
    phase_change = float(
        np.max(np.abs(np.angle(fine.impedance / coarse.impedance, deg=True)))
    )
    passed = (
        magnitude_change <= protocol.max_impedance_magnitude_relative_change
        and phase_change <= protocol.max_impedance_phase_change_deg
    )
    return PerturbationStepAssessment(
        coarse_factor=coarse_factor,
        fine_factor=fine_factor,
        max_impedance_magnitude_relative_change=magnitude_change,
        max_impedance_phase_change_deg=phase_change,
        passed=passed,
    )


def _build_reference_evaluator(
    grid: np.ndarray,
    stack: DeviceStack,
    protocol: IonAwareImpedanceProtocol,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    base_state: np.ndarray,
):
    """Return the full-Poisson nonlinear callback used by the FD reference."""
    physical_indices = np.asarray(layout.state_indices, dtype=int)
    eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
    polarity = float(material.junction_polarity)

    def evaluate(coordinate: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        physical = _physical_state(coordinate, base_state, layout)
        rate = assemble_rhs(
            0.0,
            physical,
            grid,
            stack,
            material,
            illuminated=protocol.illuminated,
            V_app=voltage,
        )
        current = compute_current_components(
            grid,
            physical,
            stack,
            voltage,
            mat=material,
        )
        ionic = compute_ionic_current_components(
            grid,
            physical,
            stack,
            voltage,
            mat=material,
        )
        electron = -np.asarray(current.J_n, dtype=float)
        hole = -np.asarray(current.J_p, dtype=float)
        positive_ion = -np.asarray(ionic.J_positive, dtype=float)
        negative_ion = (
            None
            if ionic.J_negative is None
            else -np.asarray(ionic.J_negative, dtype=float)
        )
        components = [
            SmallSignalCurrentComponent("electron", electron),
            SmallSignalCurrentComponent("hole", hole),
            SmallSignalCurrentComponent("positive_ion", positive_ion),
        ]
        if negative_ion is not None:
            components.append(
                SmallSignalCurrentComponent("negative_ion", negative_ion)
            )
        conduction = sum(
            (component.current_faces for component in components),
            start=np.zeros_like(electron),
        )
        snapshot = extract_spatial_snapshot(
            grid,
            physical,
            stack,
            voltage,
            mat=material,
        )
        displacement_charge = polarity * eps_face * snapshot.E
        return SmallSignalEvaluation(
            storage=physical[physical_indices],
            rate=np.asarray(rate, dtype=float)[physical_indices],
            conduction_current_faces=conduction,
            displacement_charge_faces=displacement_charge,
            current_components=tuple(components),
        )

    return evaluate


def run_ion_aware_impedance(
    x: np.ndarray,
    stack: DeviceStack,
    protocol: IonAwareImpedanceProtocol,
    *,
    dc_state: IonAwareDCResult,
    mat: MaterialArrays | None = None,
    require_numerical_certificate: bool = True,
    require_contact_certificate: bool = False,
    progress: ProgressCallback | None = None,
) -> IonAwareImpedanceResult:
    """Build and solve a reference mobile-ion small-signal operator."""
    if not isinstance(protocol, IonAwareImpedanceProtocol):
        raise TypeError("protocol must be an IonAwareImpedanceProtocol")
    if not isinstance(dc_state, IonAwareDCResult):
        raise TypeError("dc_state must be an IonAwareDCResult")
    grid = np.asarray(x, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("x must be a finite, strictly increasing 1-D grid")
    if not np.array_equal(grid, np.asarray(dc_state.x, dtype=float)):
        raise IonAwareImpedanceCapabilityError(
            "impedance grid does not exactly match the certified DC grid"
        )
    if stack != dc_state.stack:
        raise IonAwareImpedanceCapabilityError(
            "impedance stack does not match the stack that produced the DC state"
        )
    if protocol.dc_protocol_sha256 != dc_state.protocol_hash:
        raise IonAwareImpedanceCapabilityError(
            "impedance protocol does not match the DC protocol hash"
        )
    if protocol.dc_state_sha256 != ion_aware_dc_state_sha256(dc_state.y):
        raise IonAwareImpedanceCapabilityError(
            "impedance protocol does not match the packed DC state hash"
        )
    for field, expected, actual in (
        ("V_dc", protocol.V_dc, dc_state.protocol.V_dc),
        ("temperature_K", protocol.temperature_K, dc_state.protocol.temperature_K),
    ):
        if not np.isclose(expected, actual, rtol=0.0, atol=0.0):
            raise IonAwareImpedanceCapabilityError(
                f"impedance {field} does not match the DC protocol"
            )
    if protocol.illuminated != dc_state.protocol.illuminated:
        raise IonAwareImpedanceCapabilityError(
            "impedance illumination does not match the DC protocol"
        )
    material = build_material_arrays(grid, stack) if mat is None else mat
    if material.N_iface_state:
        raise IonAwareImpedanceCapabilityError(
            "dynamic interface-state blocks remain outside ion-aware impedance"
        )
    frequencies = np.asarray(protocol.frequencies_Hz, dtype=float)
    frequency_window = assess_impedance_frequency_window(
        grid,
        material,
        frequencies,
        branch_margin_decades=protocol.frequency_branch_margin_decades,
        max_sampling_gap_decades=(
            protocol.max_frequency_sampling_gap_decades
        ),
    )
    reassessed = assess_ion_aware_dc_state(
        grid,
        dc_state.y,
        dc_state.initial_y,
        stack,
        dc_state.protocol,
        mat=material,
    )
    if not dc_state.numerically_certified or not reassessed.numerically_certified:
        raise IonAwareImpedanceCapabilityError(
            "frequency-domain ion-aware impedance requires a currently valid "
            "numerical DC certificate"
        )
    if require_contact_certificate and not reassessed.thermodynamically_certified:
        raise IonAwareImpedanceCapabilityError(
            "the requested strict physical lane requires certified contact "
            "thermodynamics"
        )

    base_state = np.asarray(dc_state.y, dtype=float)
    expected_size = (4 if material.has_dual_ions else 3) * grid.size
    if base_state.shape != (expected_size,) or not np.all(np.isfinite(base_state)):
        raise IonAwareImpedanceCapabilityError(
            f"DC state must be a finite packed vector of length {expected_size}"
        )
    layout = _state_coordinate_layout(material, grid.size)
    physical_indices = np.asarray(layout.state_indices, dtype=int)
    if np.any(base_state[physical_indices] <= 0.0):
        raise IonAwareImpedanceCapabilityError(
            "every active impedance density must be positive"
        )
    if not _stencil_occupancy_admissible(
        base_state,
        layout,
        material,
        protocol.state_step * protocol.refinement_factors[0],
    ):
        raise IonAwareImpedanceCapabilityError(
            "the declared state stencil crosses the ion site-occupancy limit"
        )

    face_weights = np.diff(grid) / float(grid[-1] - grid[0])
    evaluate = _build_reference_evaluator(
        grid,
        stack,
        protocol,
        material,
        layout,
        base_state,
    )

    levels: list[tuple[float, FrequencyDomainResult]] = []
    coordinate = np.zeros(layout.size, dtype=float)
    for level_index, factor in enumerate(protocol.refinement_factors):
        if progress is not None:
            progress(
                "ion_aware_impedance_refinement",
                level_index,
                len(protocol.refinement_factors),
                f"finite-difference factor {factor:g}",
            )
        try:
            response = solve_frequency_domain(
                evaluate,
                coordinate,
                protocol.V_dc,
                frequencies,
                state_step=protocol.state_step * factor,
                voltage_step=protocol.voltage_step * factor,
                face_weights=face_weights,
                progress=progress,
            )
        except SmallSignalLinearizationError as exc:
            raise IonAwareImpedanceError(
                f"ion-aware reference linearization failed at factor {factor:g}: {exc}"
            ) from exc
        levels.append((factor, response))
    if progress is not None:
        progress(
            "ion_aware_impedance_refinement",
            len(protocol.refinement_factors),
            len(protocol.refinement_factors),
            "finite-difference ladder complete",
        )

    assessments = tuple(
        _perturbation_assessment(
            coarse_factor,
            fine_factor,
            coarse,
            fine,
            protocol,
        )
        for (coarse_factor, coarse), (fine_factor, fine) in zip(levels, levels[1:])
    )
    final = levels[-1][1]
    expected_mass = np.asarray(final.storage_at_operating_point, dtype=float)
    normalized_mass = final.mass_matrix / expected_mass[:, None]
    diagonal = np.diag(normalized_mass)
    off_diagonal = normalized_mass.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    mass_diagonal_error = float(np.max(np.abs(diagonal - 1.0)))
    mass_off_diagonal = float(np.max(np.abs(off_diagonal)))
    widths = np.asarray(material.dx_cell, dtype=float)
    positive_inventory_response = _inventory_response_relative(
        final.storage_response,
        layout,
        "positive_ion",
        widths,
    )
    negative_inventory_response = _inventory_response_relative(
        final.storage_response,
        layout,
        "negative_ion",
        widths,
    )
    inventory_response = max(
        positive_inventory_response,
        negative_inventory_response,
    )
    max_spread = float(np.max(final.max_relative_face_spread))
    max_backward = float(np.max(final.backward_error))
    min_rcond = float(np.min(final.reciprocal_condition))
    component_map = {
        component.name: component.admittance_faces
        for component in final.current_components
    }
    component_sum = np.sum(np.stack(tuple(component_map.values())), axis=0)
    component_magnitude_sum = sum(
        (np.abs(value) for value in component_map.values()),
        start=np.zeros_like(final.conduction_admittance_faces.real),
    )
    decomposition_scale = np.maximum(
        np.maximum(
            np.abs(final.conduction_admittance_faces),
            component_magnitude_sum,
        ),
        np.finfo(float).tiny,
    )
    decomposition_error = float(
        np.max(
            np.abs(final.conduction_admittance_faces - component_sum)
            / decomposition_scale
        )
    )
    reasons: list[str] = []
    if max_spread > protocol.max_relative_face_spread:
        reasons.append("all_face_admittance_spread_exceeds_limit")
    if max_backward > protocol.max_backward_error:
        reasons.append("componentwise_backward_error_exceeds_limit")
    if not assessments[-1].passed:
        reasons.append("finite_difference_refinement_not_converged")
    if max(mass_diagonal_error, mass_off_diagonal) > (
        protocol.max_mass_matrix_relative_error
    ):
        reasons.append("mass_matrix_log_coordinate_identity_failed")
    if inventory_response > protocol.max_ion_inventory_response_relative:
        reasons.append("blocking_ion_inventory_response_exceeds_limit")
    if decomposition_error > protocol.max_current_decomposition_relative_error:
        reasons.append("current_decomposition_closure_exceeds_limit")
    numerical = not reasons
    thermodynamic = reassessed.thermodynamically_certified
    frequency_window_certified = (
        frequency_window.ionic_branch_covered is True
    )
    certificate = IonAwareImpedanceCertificate(
        numerically_certified=numerical,
        thermodynamically_certified=thermodynamic,
        certified=(
            numerical and thermodynamic and frequency_window_certified
        ),
        max_relative_face_spread=max_spread,
        max_backward_error=max_backward,
        minimum_reciprocal_condition=min_rcond,
        max_mass_diagonal_relative_error=mass_diagonal_error,
        max_mass_off_diagonal_relative=mass_off_diagonal,
        max_ion_inventory_response_relative=inventory_response,
        max_current_decomposition_relative_error=decomposition_error,
        frequency_window_certified=frequency_window_certified,
        perturbation_assessments=assessments,
        reasons=tuple(reasons),
    )
    electron_storage = _storage_integral(
        final.storage_response, layout, "electron", widths
    )
    hole_storage = _storage_integral(
        final.storage_response, layout, "hole", widths
    )
    positive_storage = _storage_integral(
        final.storage_response, layout, "positive_ion", widths
    )
    negative_storage = _storage_integral(
        final.storage_response, layout, "negative_ion", widths
    )
    if electron_storage is None or hole_storage is None:
        raise IonAwareImpedanceError(
            "ion-aware coordinate layout lost a carrier storage block"
        )
    if positive_storage is None and negative_storage is None:
        raise IonAwareImpedanceError(
            "ion-aware coordinate layout lost every mobile-ion storage block"
        )
    positive_storage_array = (
        np.zeros_like(electron_storage)
        if positive_storage is None
        else positive_storage
    )
    net_charge_storage = (
        -electron_storage + hole_storage + positive_storage_array
        - (np.zeros_like(electron_storage) if negative_storage is None else negative_storage)
    )
    result = IonAwareImpedanceResult(
        frequencies=final.frequencies,
        Z=final.impedance,
        Y=final.admittance,
        Y_faces=final.admittance_faces,
        conduction_admittance_faces_S_m2=final.conduction_admittance_faces,
        displacement_admittance_faces_S_m2=final.displacement_admittance_faces,
        electron_admittance_faces_S_m2=component_map["electron"],
        hole_admittance_faces_S_m2=component_map["hole"],
        positive_ion_admittance_faces_S_m2=component_map["positive_ion"],
        negative_ion_admittance_faces_S_m2=component_map.get("negative_ion"),
        electron_storage_response_F_m2=electron_storage,
        hole_storage_response_F_m2=hole_storage,
        positive_ion_storage_response_F_m2=positive_storage_array,
        negative_ion_storage_response_F_m2=negative_storage,
        net_charge_storage_response_F_m2=net_charge_storage,
        state_response_per_V=final.state_response,
        storage_response_per_V=final.storage_response,
        coordinate_layout=layout,
        reference_linearization=final,
        reference_linearizations=tuple(response for _, response in levels),
        frequency_window=frequency_window,
        protocol=protocol,
        dc_state=dc_state,
        certificate=certificate,
    )
    if require_numerical_certificate and not numerical:
        raise IonAwareImpedanceCertificationError(
            "ion-aware impedance numerical certificate failed: "
            + ", ".join(reasons),
            result,
        )
    return result


__all__ = [
    "DEFAULT_REFINEMENT_FACTORS",
    "FrequencyWindowAssessment",
    "ION_AWARE_IMPEDANCE_PROTOCOL_SCHEMA",
    "IonAwareImpedanceCapabilityError",
    "IonAwareImpedanceCertificate",
    "IonAwareImpedanceCertificationError",
    "IonAwareImpedanceError",
    "IonAwareImpedanceProtocol",
    "IonAwareImpedanceResult",
    "IonAwareStateCoordinateLayout",
    "MAX_LINEAR_PERTURBATION_V",
    "PerturbationStepAssessment",
    "build_ion_aware_impedance_protocol",
    "assess_impedance_frequency_window",
    "run_ion_aware_impedance",
]
