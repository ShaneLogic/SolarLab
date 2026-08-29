"""Protocol-bound production adapter for dynamic interface defects and ions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Callable, Literal, Self

import numpy as np

from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientCertificate,
    InterfaceDefectIonTransientError,
    InterfaceDefectIonTransientPolicy,
    run_interface_defect_ion_device_transient,
)
from perovskite_sim.experiments.ion_aware_impedance_grid import (
    ion_aware_impedance_grid_sha256,
)
from perovskite_sim.models.device import (
    DeviceStack,
    electrical_interface_defects,
    electrical_layers,
    require_uncalibrated_microscopic_interface_defects,
)
from perovskite_sim.models.defects import EXPLICIT_QUASI_STEADY
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
)
from perovskite_sim.physics.generation import dual_cell_widths


DYNAMIC_DEFECT_TRANSIENT_METHOD = "dynamic_defect_transient_certified"
DYNAMIC_DEFECT_TRANSIENT_SCHEMA = "dynamic-defect-transient-protocol-v1"
DYNAMIC_DEFECT_TRANSIENT_EVIDENCE = "dynamic-defect-transient-evidence-v1"
DYNAMIC_DEFECT_TRANSIENT_CAPABILITY = "interface_defect_plus_positive_ions"
DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE = (
    "dynamic-defect-ion-transient-timescale-reference-resolved-v5"
)
DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256 = (
    "9eab2f9e251b8d4c0f7f3f07e0baeea9bb6497126ef8d8111eba1803947e5beb"
)
ALLOWED_TIME_STEP_REFINEMENT_FACTORS = (1.0, 0.5, 0.25)
ProgressCallback = Callable[[str, int, int, str], None]

DynamicDefectTransientCapability = Literal["interface_defect_plus_positive_ions"]
VoltageInterpolation = Literal["right_continuous_step_and_hold"]
InterfaceCurrentObservation = Literal["symmetric_adjacent_physical_faces"]


class DynamicDefectTransientError(RuntimeError):
    """The production dynamic-defect transient contract failed closed."""


class DynamicDefectTransientCapabilityError(DynamicDefectTransientError):
    """The requested stack is outside the certified transient capability."""


class DynamicDefectTransientProtocolError(DynamicDefectTransientError):
    """A supplied transient protocol is malformed or mismatches execution."""


class DynamicDefectTransientCertificationError(DynamicDefectTransientError):
    """A dynamic-defect transient failed its numerical evidence gates."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _positive_integer(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise DynamicDefectTransientProtocolError(
            f"{label} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("dynamic-defect transient protocol is non-finite")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "dynamic-defect transient protocol contains unsupported value "
        f"{type(value).__name__}"
    )


def _readonly(value: object) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise DynamicDefectTransientCertificationError(
            "production projection contains non-finite data"
        )
    result.setflags(write=False)
    return result


def _trace_values(
    times_s: object,
    voltage_V: object,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        times = tuple(
            _finite(value, f"times_s[{index}]")
            for index, value in enumerate(times_s)  # type: ignore[arg-type]
        )
        voltage = tuple(
            _finite(value, f"voltage_V[{index}]")
            for index, value in enumerate(voltage_V)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise TypeError("times_s and voltage_V must be numeric sequences") from exc
    if len(times) < 2 or len(voltage) != len(times):
        raise ValueError("times_s and voltage_V must contain aligned traces")
    if times[0] != 0.0:
        raise ValueError("production transient history must start at t=0")
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("times_s must be strictly increasing")
    return times, voltage


def default_dynamic_defect_transient_policy(
    time_step_refinement_factor: float = 1.0,
) -> InterfaceDefectIonTransientPolicy:
    """Return the nonlinear policy frozen by the D6-E3c v5 certificate."""

    factor = _finite(time_step_refinement_factor, "time_step_refinement_factor")
    if factor not in ALLOWED_TIME_STEP_REFINEMENT_FACTORS:
        raise ValueError(
            "time_step_refinement_factor must be one of "
            f"{ALLOWED_TIME_STEP_REFINEMENT_FACTORS}"
        )
    multiplier = int(round(1.0 / factor))
    return replace(
        InterfaceDefectIonTransientPolicy(),
        maximum_newton_iterations=100,
        maximum_line_search_steps=40,
        maximum_near_acceptance_nonmonotone_steps=2,
        refinement_substeps=tuple(multiplier * value for value in (1, 2, 4)),
    )


def _policy_from_dict(payload: object) -> InterfaceDefectIonTransientPolicy:
    if not isinstance(payload, Mapping):
        raise TypeError("solver_policy must be a JSON object")
    expected = {
        field.name for field in dataclasses.fields(InterfaceDefectIonTransientPolicy)
    }
    _exact_keys(payload, expected, "InterfaceDefectIonTransientPolicy")
    values = dict(payload)
    values["refinement_substeps"] = tuple(values["refinement_substeps"])
    return InterfaceDefectIonTransientPolicy(**values)


@dataclass(frozen=True, slots=True)
class DynamicDefectTransientProtocol:
    schema_version: Literal["dynamic-defect-transient-protocol-v1"]
    method: Literal["dynamic_defect_transient_certified"]
    capability: DynamicDefectTransientCapability
    illuminated: bool
    times_s: tuple[float, ...]
    voltage_V: tuple[float, ...]
    voltage_interpolation: VoltageInterpolation
    requested_grid_intervals: int
    actual_grid_nodes: int
    grid_sha256: str
    stack_sha256: str
    interface_defect_document_sha256: tuple[str, ...]
    active_positive_ion_layer_indices: tuple[int, ...]
    defect_energy_quadrature_order: int
    interface_current_observation: InterfaceCurrentObservation
    time_step_refinement_factor: float
    solver_policy: InterfaceDefectIonTransientPolicy
    reference_lane_id: str
    reference_certificate_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_DEFECT_TRANSIENT_SCHEMA:
            raise ValueError("unsupported dynamic-defect transient schema")
        if self.method != DYNAMIC_DEFECT_TRANSIENT_METHOD:
            raise ValueError("unsupported dynamic-defect transient method")
        if self.capability != DYNAMIC_DEFECT_TRANSIENT_CAPABILITY:
            raise ValueError("unsupported dynamic-defect transient capability")
        if not isinstance(self.illuminated, bool):
            raise TypeError("illuminated must be boolean")
        if self.illuminated:
            raise ValueError("the v1 production transient capability is dark-only")
        times, voltage = _trace_values(self.times_s, self.voltage_V)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "voltage_V", voltage)
        if self.voltage_interpolation != "right_continuous_step_and_hold":
            raise ValueError("unsupported voltage interpolation")
        object.__setattr__(
            self,
            "requested_grid_intervals",
            _positive_integer(
                self.requested_grid_intervals,
                "requested_grid_intervals",
                minimum=4,
            ),
        )
        object.__setattr__(
            self,
            "actual_grid_nodes",
            _positive_integer(self.actual_grid_nodes, "actual_grid_nodes", minimum=4),
        )
        object.__setattr__(
            self, "grid_sha256", _sha256(self.grid_sha256, "grid_sha256")
        )
        object.__setattr__(
            self,
            "stack_sha256",
            _sha256(self.stack_sha256, "stack_sha256"),
        )
        documents = tuple(
            _sha256(value, f"interface_defect_document_sha256[{index}]")
            for index, value in enumerate(self.interface_defect_document_sha256)
        )
        if len(documents) != 1:
            raise ValueError(
                "v1 production transient requires exactly one interface defect"
            )
        object.__setattr__(self, "interface_defect_document_sha256", documents)
        indices = tuple(
            _positive_integer(
                value, f"active_positive_ion_layer_indices[{index}]", minimum=0
            )
            for index, value in enumerate(self.active_positive_ion_layer_indices)
        )
        if len(indices) != 1:
            raise ValueError(
                "v1 production transient requires one active positive-ion layer"
            )
        object.__setattr__(self, "active_positive_ion_layer_indices", indices)
        order = _positive_integer(
            self.defect_energy_quadrature_order,
            "defect_energy_quadrature_order",
        )
        if order != DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER:
            raise ValueError("v1 transient uses the frozen default defect energy order")
        object.__setattr__(self, "defect_energy_quadrature_order", order)
        if self.interface_current_observation != "symmetric_adjacent_physical_faces":
            raise ValueError(
                "v1 transient requires symmetric interface current observation"
            )
        factor = _finite(
            self.time_step_refinement_factor,
            "time_step_refinement_factor",
        )
        if factor not in ALLOWED_TIME_STEP_REFINEMENT_FACTORS:
            raise ValueError(
                "time_step_refinement_factor is outside the certified ladder"
            )
        object.__setattr__(self, "time_step_refinement_factor", factor)
        if not isinstance(self.solver_policy, InterfaceDefectIonTransientPolicy):
            raise TypeError("solver_policy has the wrong type")
        if self.solver_policy != default_dynamic_defect_transient_policy(factor):
            raise ValueError(
                "solver_policy does not match the frozen time-step refinement factor"
            )
        if self.reference_lane_id != DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE:
            raise ValueError("reference_lane_id does not match the certified v5 lane")
        object.__setattr__(
            self,
            "reference_certificate_sha256",
            _sha256(
                self.reference_certificate_sha256,
                "reference_certificate_sha256",
            ),
        )
        if (
            self.reference_certificate_sha256
            != DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256
        ):
            raise ValueError(
                "reference certificate does not match the certified v5 lane"
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("dynamic_defect_transient_protocol must be a JSON object")
        _exact_keys(
            payload, {field.name for field in dataclasses.fields(cls)}, cls.__name__
        )
        values = dict(payload)
        for field in (
            "times_s",
            "voltage_V",
            "interface_defect_document_sha256",
            "active_positive_ion_layer_indices",
        ):
            values[field] = tuple(values[field])
        values["solver_policy"] = _policy_from_dict(values["solver_policy"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError(
                "dynamic-defect transient protocol JSON must contain an object"
            )
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class DynamicDefectTransientEvidence:
    model: Literal["dynamic-defect-transient-evidence-v1"]
    protocol: DynamicDefectTransientProtocol
    protocol_sha256: str
    capability: DynamicDefectTransientCapability
    engine_scope: str
    engine_version: str
    state_sha256: str
    reference_lane_id: str
    reference_certificate_sha256: str
    engine_certificate: InterfaceDefectIonTransientCertificate
    dc_operating_point_certified: bool
    dark_reference_certified: bool
    microscopic_binding_certified: bool
    numerically_certified: bool
    public_projection_certified: bool
    certified: bool
    maximum_interface_occupancy_motion: float
    maximum_positive_ion_relative_motion: float
    maximum_positive_ion_centroid_shift_m: float
    maximum_integrated_charge_change_C_m2: float
    maximum_terminal_current_A_m2: float
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DynamicDefectTransientProductionResult:
    grid_m: np.ndarray
    times_s: np.ndarray
    voltage_V: np.ndarray
    terminal_total_current_A_m2: np.ndarray
    total_current_faces_A_m2: np.ndarray
    interface_total_current_A_m2: np.ndarray
    interface_occupancy: np.ndarray
    interface_occupancy_change: np.ndarray
    positive_ion_centroid_m: np.ndarray
    positive_ion_centroid_shift_m: np.ndarray
    integrated_charge_change_C_m2: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    positive_ion_density_m3: np.ndarray
    electrostatic_potential_V: np.ndarray
    protocol: DynamicDefectTransientProtocol
    evidence: DynamicDefectTransientEvidence

    def __post_init__(self) -> None:
        names = (
            "grid_m",
            "times_s",
            "voltage_V",
            "terminal_total_current_A_m2",
            "total_current_faces_A_m2",
            "interface_total_current_A_m2",
            "interface_occupancy",
            "interface_occupancy_change",
            "positive_ion_centroid_m",
            "positive_ion_centroid_shift_m",
            "integrated_charge_change_C_m2",
            "electron_density_m3",
            "hole_density_m3",
            "positive_ion_density_m3",
            "electrostatic_potential_V",
        )
        for name in names:
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def _bulk_defect_documents(stack: DeviceStack) -> tuple[str, ...]:
    documents: list[str] = []
    for layer in electrical_layers(stack):
        document = layer.params.defect_document
        if (
            document is not None
            and document.defect_model == EXPLICIT_QUASI_STEADY
            and document.bulk_defects
        ):
            documents.append(document.sha256)
    return tuple(documents)


def _interface_documents(stack: DeviceStack) -> tuple[str, ...]:
    return tuple(
        defect.microscopic_document.sha256
        for defect in electrical_interface_defects(stack)
        if defect is not None and defect.microscopic_document is not None
    )


def classify_dynamic_defect_transient_capability(
    stack: DeviceStack,
) -> DynamicDefectTransientCapability:
    """Admit only the narrow topology covered by the D6-E3c v5 evidence."""

    layers = tuple(electrical_layers(stack))
    if len(layers) != 2:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires exactly two electrical layers"
        )
    if _bulk_defect_documents(stack):
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient excludes simultaneous bulk explicit defects"
        )
    defects = tuple(
        defect for defect in electrical_interface_defects(stack) if defect is not None
    )
    if len(defects) != 1 or len(_interface_documents(stack)) != 1:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires exactly one microscopic interface defect"
        )
    try:
        require_uncalibrated_microscopic_interface_defects(
            stack,
            consumer="production dynamic-defect transient",
        )
    except (TypeError, ValueError) as exc:
        raise DynamicDefectTransientCapabilityError(str(exc)) from exc
    if getattr(stack, "interface_charge_closure", "off") != "equilibrium_referenced":
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires equilibrium-referenced interface charge"
        )
    if not bool(getattr(stack, "interface_charge_rebaseline_acknowledged", False)):
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires interface charge rebaseline acknowledgement"
        )
    active_positive = tuple(
        index
        for index, layer in enumerate(layers)
        if float(layer.params.D_ion) > 0.0 and float(layer.params.P0) > 0.0
    )
    active_negative = tuple(
        index
        for index, layer in enumerate(layers)
        if float(layer.params.D_ion_neg) > 0.0 and float(layer.params.P0_neg) > 0.0
    )
    if len(active_positive) != 1:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires exactly one active positive-ion layer"
        )
    if str(layers[active_positive[0]].role).lower() != "absorber":
        raise DynamicDefectTransientCapabilityError(
            "the active positive-ion layer must have role='absorber'"
        )
    if active_negative:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient excludes active negative ions"
        )
    return DYNAMIC_DEFECT_TRANSIENT_CAPABILITY


def _active_positive_indices(stack: DeviceStack) -> tuple[int, ...]:
    return tuple(
        index
        for index, layer in enumerate(electrical_layers(stack))
        if float(layer.params.D_ion) > 0.0 and float(layer.params.P0) > 0.0
    )


def build_dynamic_defect_transient_protocol(
    stack: DeviceStack,
    grid: np.ndarray,
    times_s: object,
    voltage_V: object,
    *,
    requested_grid_intervals: int,
    illuminated: bool = False,
    time_step_refinement_factor: float = 1.0,
) -> DynamicDefectTransientProtocol:
    capability = classify_dynamic_defect_transient_capability(stack)
    if not isinstance(illuminated, (bool, np.bool_)):
        raise TypeError("illuminated must be boolean")
    if bool(illuminated):
        raise DynamicDefectTransientCapabilityError(
            "the v1 certified dynamic-defect transient is dark-only"
        )
    times, voltage = _trace_values(times_s, voltage_V)
    actual_grid = np.asarray(grid, dtype=float)
    if (
        actual_grid.ndim != 1
        or actual_grid.size < 4
        or not np.all(np.isfinite(actual_grid))
        or np.any(np.diff(actual_grid) <= 0.0)
    ):
        raise ValueError("grid must be finite, one-dimensional, and increasing")
    return DynamicDefectTransientProtocol(
        schema_version=DYNAMIC_DEFECT_TRANSIENT_SCHEMA,
        method=DYNAMIC_DEFECT_TRANSIENT_METHOD,
        capability=capability,
        illuminated=False,
        times_s=times,
        voltage_V=voltage,
        voltage_interpolation="right_continuous_step_and_hold",
        requested_grid_intervals=requested_grid_intervals,
        actual_grid_nodes=int(actual_grid.size),
        grid_sha256=ion_aware_impedance_grid_sha256(actual_grid),
        stack_sha256=hashlib.sha256(repr(stack).encode("utf-8")).hexdigest(),
        interface_defect_document_sha256=_interface_documents(stack),
        active_positive_ion_layer_indices=_active_positive_indices(stack),
        defect_energy_quadrature_order=DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
        interface_current_observation="symmetric_adjacent_physical_faces",
        time_step_refinement_factor=time_step_refinement_factor,
        solver_policy=default_dynamic_defect_transient_policy(
            time_step_refinement_factor
        ),
        reference_lane_id=DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE,
        reference_certificate_sha256=(
            DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256
        ),
    )


def resolve_dynamic_defect_transient_protocol(
    supplied: DynamicDefectTransientProtocol | None,
    expected: DynamicDefectTransientProtocol,
) -> DynamicDefectTransientProtocol:
    if supplied is None:
        return expected
    if not isinstance(supplied, DynamicDefectTransientProtocol):
        raise TypeError("dynamic_defect_transient_protocol has the wrong type")
    if supplied != expected:
        raise DynamicDefectTransientProtocolError(
            "dynamic_defect_transient_protocol does not match the requested "
            "stack, grid, history, solver policy, or reference certificate"
        )
    return supplied


def _state_sha256(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in values:
        array = np.array(value, dtype="<f8", order="C", copy=True)
        array[array == 0.0] = 0.0
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _positive_ion_centroid(
    grid: np.ndarray,
    density: np.ndarray,
    active_nodes: tuple[int, ...],
) -> np.ndarray:
    nodes = np.asarray(active_nodes, dtype=int)
    if nodes.size == 0:
        raise DynamicDefectTransientCertificationError(
            "positive-ion centroid requires active ion nodes"
        )
    widths = dual_cell_widths(grid)[nodes]
    values = np.asarray(density, dtype=float)[:, nodes]
    inventory = np.sum(values * widths[None, :], axis=1)
    if np.any(~np.isfinite(inventory)) or np.any(inventory <= 0.0):
        raise DynamicDefectTransientCertificationError(
            "positive-ion centroid requires finite positive inventory"
        )
    return np.sum(values * grid[nodes][None, :] * widths[None, :], axis=1) / inventory


def _maximum_positive_ion_relative_motion(
    density: np.ndarray,
    active_nodes: tuple[int, ...],
) -> float:
    nodes = np.asarray(active_nodes, dtype=int)
    values = np.asarray(density, dtype=float)
    if nodes.size == 0:
        raise DynamicDefectTransientCertificationError(
            "positive-ion motion requires active ion nodes"
        )
    active = values[:, nodes]
    if active.ndim != 2 or not np.all(np.isfinite(active)) or np.any(active[0] <= 0.0):
        raise DynamicDefectTransientCertificationError(
            "positive-ion motion requires a finite positive reference"
        )
    return float(np.max(np.abs(active / active[0] - 1.0)))


def run_dynamic_defect_transient(
    grid: np.ndarray,
    stack: DeviceStack,
    protocol: DynamicDefectTransientProtocol,
    *,
    progress: ProgressCallback | None = None,
) -> DynamicDefectTransientProductionResult:
    """Run the exact dark interface-defect/positive-ion transient protocol."""

    if not isinstance(protocol, DynamicDefectTransientProtocol):
        raise TypeError("protocol must be DynamicDefectTransientProtocol")
    expected = build_dynamic_defect_transient_protocol(
        stack,
        grid,
        protocol.times_s,
        protocol.voltage_V,
        requested_grid_intervals=protocol.requested_grid_intervals,
        illuminated=protocol.illuminated,
        time_step_refinement_factor=protocol.time_step_refinement_factor,
    )
    resolve_dynamic_defect_transient_protocol(protocol, expected)
    if progress is not None:
        progress(
            "dynamic_defect_transient",
            0,
            1,
            "Solving certified interface-defect/positive-ion transient",
        )
    try:
        raw = run_interface_defect_ion_device_transient(
            np.asarray(grid, dtype=float),
            stack,
            protocol.times_s,
            protocol.voltage_V,
            illuminated=False,
            policy=protocol.solver_policy,
            require_certificate=True,
        )
    except InterfaceDefectIonTransientError as exc:
        raise DynamicDefectTransientCertificationError(str(exc)) from exc
    if progress is not None:
        progress(
            "dynamic_defect_transient",
            1,
            1,
            "Certified dynamic-defect transient complete",
        )
    actual_grid = np.asarray(grid, dtype=float)
    centroid = _positive_ion_centroid(
        actual_grid,
        raw.positive_ion_density_m3,
        raw.ion_layout.positive_nodes,
    )
    occupancy = np.asarray(raw.interface_occupancy, dtype=float)
    occupancy_change = occupancy - occupancy[0]
    centroid_shift = centroid - centroid[0]
    charge = np.asarray(raw.integrated_free_interface_ion_charge_C_m2, dtype=float)
    charge_change = charge - charge[0]
    terminal = np.asarray(raw.total_current_faces_A_m2, dtype=float)[:, 0]
    relative_ion_motion = _maximum_positive_ion_relative_motion(
        raw.positive_ion_density_m3,
        raw.ion_layout.positive_nodes,
    )
    projection_arrays = (
        occupancy_change,
        centroid,
        centroid_shift,
        charge_change,
        terminal,
    )
    projection_certified = all(
        np.all(np.isfinite(value)) for value in projection_arrays
    )
    reasons = list(raw.certificate.reasons)
    if not projection_certified:
        reasons.append("public_projection_nonfinite")
    evidence = DynamicDefectTransientEvidence(
        model=DYNAMIC_DEFECT_TRANSIENT_EVIDENCE,
        protocol=protocol,
        protocol_sha256=protocol.protocol_hash,
        capability=protocol.capability,
        engine_scope=raw.scope,
        engine_version=raw.version,
        state_sha256=_state_sha256(
            "dynamic-defect-transient-production-state-v1",
            raw.electron_density_m3,
            raw.hole_density_m3,
            raw.interface_occupancy,
            raw.positive_ion_density_m3,
            raw.electrostatic_potential_V,
            raw.total_current_faces_A_m2,
        ),
        reference_lane_id=protocol.reference_lane_id,
        reference_certificate_sha256=protocol.reference_certificate_sha256,
        engine_certificate=raw.certificate,
        dc_operating_point_certified=bool(raw.certificate.dc_operating_point_certified),
        dark_reference_certified=bool(raw.certificate.dark_reference_certified),
        microscopic_binding_certified=bool(
            raw.certificate.microscopic_binding_certified
        ),
        numerically_certified=bool(raw.certificate.certified),
        public_projection_certified=projection_certified,
        certified=bool(raw.certificate.certified and projection_certified),
        maximum_interface_occupancy_motion=float(np.max(np.abs(occupancy_change))),
        maximum_positive_ion_relative_motion=relative_ion_motion,
        maximum_positive_ion_centroid_shift_m=float(np.max(np.abs(centroid_shift))),
        maximum_integrated_charge_change_C_m2=float(np.max(np.abs(charge_change))),
        maximum_terminal_current_A_m2=float(np.max(np.abs(terminal))),
        reasons=tuple(reasons),
        limitations=(
            "internal numerical certification, not SCAPS transient parity",
            "dark two-layer single-interface positive-ion capability only",
            "no bulk dynamic defects, negative ions, distributions, or metastability",
        ),
    )
    if not evidence.certified:
        raise DynamicDefectTransientCertificationError(
            "dynamic-defect transient production projection did not certify: "
            + ", ".join(evidence.reasons)
        )
    return DynamicDefectTransientProductionResult(
        grid_m=actual_grid,
        times_s=raw.times_s,
        voltage_V=raw.voltage_V,
        terminal_total_current_A_m2=terminal,
        total_current_faces_A_m2=raw.total_current_faces_A_m2,
        interface_total_current_A_m2=raw.interface_total_current_A_m2,
        interface_occupancy=occupancy,
        interface_occupancy_change=occupancy_change,
        positive_ion_centroid_m=centroid,
        positive_ion_centroid_shift_m=centroid_shift,
        integrated_charge_change_C_m2=charge_change,
        electron_density_m3=raw.electron_density_m3,
        hole_density_m3=raw.hole_density_m3,
        positive_ion_density_m3=raw.positive_ion_density_m3,
        electrostatic_potential_V=raw.electrostatic_potential_V,
        protocol=protocol,
        evidence=evidence,
    )


__all__ = [
    "ALLOWED_TIME_STEP_REFINEMENT_FACTORS",
    "DYNAMIC_DEFECT_TRANSIENT_CAPABILITY",
    "DYNAMIC_DEFECT_TRANSIENT_EVIDENCE",
    "DYNAMIC_DEFECT_TRANSIENT_METHOD",
    "DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256",
    "DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE",
    "DYNAMIC_DEFECT_TRANSIENT_SCHEMA",
    "DynamicDefectTransientCapability",
    "DynamicDefectTransientCapabilityError",
    "DynamicDefectTransientCertificationError",
    "DynamicDefectTransientError",
    "DynamicDefectTransientEvidence",
    "DynamicDefectTransientProductionResult",
    "DynamicDefectTransientProtocol",
    "DynamicDefectTransientProtocolError",
    "build_dynamic_defect_transient_protocol",
    "classify_dynamic_defect_transient_capability",
    "default_dynamic_defect_transient_policy",
    "resolve_dynamic_defect_transient_protocol",
    "run_dynamic_defect_transient",
]
