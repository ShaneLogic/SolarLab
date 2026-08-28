"""Protocol-bound production adapter for dynamic explicit-defect impedance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Literal, Self

import numpy as np

from perovskite_sim.experiments.defect_aware_impedance import (
    BulkDefectDeviceACError,
    BulkDefectDeviceACResult,
    run_bulk_defect_device_impedance,
)
from perovskite_sim.experiments.defect_ion_combined_impedance import (
    DefectIonCombinedError,
    DefectIonCombinedResult,
    run_defect_ion_combined_impedance,
)
from perovskite_sim.experiments.interface_defect_aware_impedance import (
    InterfaceDefectDeviceACError,
    InterfaceDefectDeviceACResult,
    run_interface_defect_device_impedance,
)
from perovskite_sim.experiments.ion_aware_impedance_grid import (
    ion_aware_impedance_grid_sha256,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
)
from perovskite_sim.models.device import (
    DeviceStack,
    electrical_interface_defects,
    electrical_layers,
    require_uncalibrated_microscopic_interface_defects,
)
from perovskite_sim.models.defects import EXPLICIT_QUASI_STEADY
from perovskite_sim.physics.contacts import (
    ContactThermodynamicCertificate,
    assess_contact_thermodynamics,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    validate_defect_energy_quadrature_order,
)
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_OFF,
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    build_material_arrays,
)


DYNAMIC_DEFECT_IMPEDANCE_METHOD = "dynamic_defect_frequency_certified"
DYNAMIC_DEFECT_IMPEDANCE_SCHEMA = "dynamic-defect-impedance-protocol-v1"
DYNAMIC_DEFECT_IMPEDANCE_EVIDENCE = "dynamic-defect-impedance-evidence-v1"
DEFAULT_REFINEMENT_FACTORS = (1.0, 0.5, 0.25)
ProgressCallback = Callable[[str, int, int, str], None]

DynamicDefectCapability = Literal[
    "bulk_dynamic_defect",
    "interface_dynamic_defect",
    "bulk_defect_plus_ions",
    "interface_defect_plus_ions",
    "bulk_interface_defect_plus_ions",
]
InterfaceCurrentObservation = Literal[
    "ordinary_finite_volume_faces",
    "symmetric_adjacent_physical_faces",
]


class DynamicDefectImpedanceError(RuntimeError):
    """The production dynamic-defect impedance contract failed closed."""


class DynamicDefectImpedanceCapabilityError(DynamicDefectImpedanceError):
    """The requested material combination has no certified dynamic adapter."""


class DynamicDefectImpedanceProtocolError(DynamicDefectImpedanceError):
    """A supplied production protocol is malformed or mismatches execution."""


class DynamicDefectImpedanceCertificationError(DynamicDefectImpedanceError):
    """A production dynamic-defect calculation failed its evidence gates."""


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


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
        raise DynamicDefectImpedanceProtocolError(
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
            raise ValueError(
                "dynamic-defect impedance protocol contains non-finite data"
            )
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "dynamic-defect impedance protocol contains unsupported value "
        f"{type(value).__name__}"
    )


def _strict_float_tuple(
    value: object,
    field: str,
    *,
    minimum_size: int,
    decreasing: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence")
    try:
        result = tuple(
            _finite(item, f"{field}[{index}]", positive=True)
            for index, item in enumerate(value)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence") from exc
    if len(result) < minimum_size:
        raise ValueError(f"{field} must contain at least {minimum_size} values")
    if decreasing:
        if any(right >= left for left, right in zip(result, result[1:], strict=False)):
            raise ValueError(f"{field} must be strictly decreasing")
    elif any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError(f"{field} must be strictly increasing")
    return result


@dataclass(frozen=True, slots=True)
class DynamicDefectImpedanceGates:
    frequency_branch_margin_decades: float
    maximum_frequency_sampling_gap_decades: float
    maximum_dc_operator_match_error: float
    maximum_dc_normalized_residual: float
    maximum_dc_continuity_bound_A_m2: float
    maximum_dc_ionic_face_current_A_m2: float
    maximum_dc_inventory_error: float
    maximum_dc_poisson_residual: float
    maximum_dc_face_current_spread_A_m2: float
    maximum_qss_embedding_error: float
    maximum_local_interface_residual: float
    maximum_local_gauss_residual: float
    maximum_trap_balance_relative_error: float
    maximum_all_face_admittance_spread: float
    maximum_linear_solve_backward_error: float
    maximum_refinement_relative_change: float
    maximum_ion_inventory_response_relative: float
    maximum_current_decomposition_relative_error: float
    maximum_limit_relative_error: float
    dc_max_nfev: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "dc_max_nfev":
                object.__setattr__(
                    self,
                    field.name,
                    _positive_integer(value, field.name),
                )
            else:
                object.__setattr__(
                    self,
                    field.name,
                    _finite(value, field.name, positive=True),
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("dynamic-defect impedance gates must be an object")
        _exact_keys(
            payload,
            {field.name for field in dataclasses.fields(cls)},
            cls.__name__,
        )
        return cls(**dict(payload))


def default_dynamic_defect_impedance_gates(
    capability: DynamicDefectCapability,
) -> DynamicDefectImpedanceGates:
    interface_only = capability == "interface_dynamic_defect"
    combined = capability.endswith("_plus_ions")
    return DynamicDefectImpedanceGates(
        frequency_branch_margin_decades=1.0 if combined else 2.0,
        maximum_frequency_sampling_gap_decades=0.5,
        maximum_dc_operator_match_error=1.0e-10,
        maximum_dc_normalized_residual=(
            1.0e-8 if combined else (1.0e-7 if interface_only else 1.0e-10)
        ),
        maximum_dc_continuity_bound_A_m2=1.0e-4,
        maximum_dc_ionic_face_current_A_m2=1.0e-6,
        maximum_dc_inventory_error=1.0e-10,
        maximum_dc_poisson_residual=1.0e-8,
        maximum_dc_face_current_spread_A_m2=1.0e-4,
        maximum_qss_embedding_error=(
            1.0e-8 if combined else (1.0e-9 if interface_only else 1.0e-10)
        ),
        maximum_local_interface_residual=1.0e-7,
        maximum_local_gauss_residual=1.0e-7,
        maximum_trap_balance_relative_error=1.0e-3 if combined else 1.0e-4,
        maximum_all_face_admittance_spread=5.0e-4,
        maximum_linear_solve_backward_error=1.0e-10,
        maximum_refinement_relative_change=2.0e-3,
        maximum_ion_inventory_response_relative=1.0e-8,
        maximum_current_decomposition_relative_error=1.0e-7,
        maximum_limit_relative_error=3.0e-2,
        dc_max_nfev=1000,
    )


@dataclass(frozen=True, slots=True)
class DynamicDefectImpedanceProtocol:
    schema_version: Literal["dynamic-defect-impedance-protocol-v1"]
    method: Literal["dynamic_defect_frequency_certified"]
    capability: DynamicDefectCapability
    V_dc_V: float
    delta_V: float
    illuminated: bool
    frequencies_Hz: tuple[float, ...]
    requested_grid_intervals: int
    actual_grid_nodes: int
    grid_sha256: str
    stack_sha256: str
    bulk_defect_document_sha256: tuple[str, ...]
    interface_defect_document_sha256: tuple[str, ...]
    defect_energy_quadrature_order: int
    state_step: float
    voltage_step: float
    refinement_factors: tuple[float, ...]
    interface_current_observation: InterfaceCurrentObservation
    gates: DynamicDefectImpedanceGates

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_DEFECT_IMPEDANCE_SCHEMA:
            raise ValueError("unsupported dynamic-defect impedance schema")
        if self.method != DYNAMIC_DEFECT_IMPEDANCE_METHOD:
            raise ValueError("unsupported dynamic-defect impedance method")
        allowed = {
            "bulk_dynamic_defect",
            "interface_dynamic_defect",
            "bulk_defect_plus_ions",
            "interface_defect_plus_ions",
            "bulk_interface_defect_plus_ions",
        }
        if self.capability not in allowed:
            raise ValueError(f"unknown dynamic-defect capability {self.capability!r}")
        object.__setattr__(self, "V_dc_V", _finite(self.V_dc_V, "V_dc_V"))
        object.__setattr__(
            self,
            "delta_V",
            _finite(self.delta_V, "delta_V", positive=True),
        )
        if not isinstance(self.illuminated, bool):
            raise TypeError("illuminated must be boolean")
        object.__setattr__(
            self,
            "frequencies_Hz",
            _strict_float_tuple(
                self.frequencies_Hz,
                "frequencies_Hz",
                minimum_size=3,
            ),
        )
        object.__setattr__(
            self,
            "requested_grid_intervals",
            _positive_integer(
                self.requested_grid_intervals,
                "requested_grid_intervals",
                minimum=3,
            ),
        )
        object.__setattr__(
            self,
            "actual_grid_nodes",
            _positive_integer(self.actual_grid_nodes, "actual_grid_nodes", minimum=3),
        )
        object.__setattr__(
            self, "grid_sha256", _sha256(self.grid_sha256, "grid_sha256")
        )
        object.__setattr__(
            self, "stack_sha256", _sha256(self.stack_sha256, "stack_sha256")
        )
        for name in (
            "bulk_defect_document_sha256",
            "interface_defect_document_sha256",
        ):
            values = tuple(
                _sha256(item, f"{name}[{index}]")
                for index, item in enumerate(getattr(self, name))
            )
            object.__setattr__(self, name, values)
        if (
            not self.bulk_defect_document_sha256
            and not self.interface_defect_document_sha256
        ):
            raise ValueError(
                "dynamic-defect protocol must bind at least one defect document"
            )
        object.__setattr__(
            self,
            "defect_energy_quadrature_order",
            validate_defect_energy_quadrature_order(
                self.defect_energy_quadrature_order
            ),
        )
        object.__setattr__(
            self, "state_step", _finite(self.state_step, "state_step", positive=True)
        )
        object.__setattr__(
            self,
            "voltage_step",
            _finite(self.voltage_step, "voltage_step", positive=True),
        )
        object.__setattr__(
            self,
            "refinement_factors",
            _strict_float_tuple(
                self.refinement_factors,
                "refinement_factors",
                minimum_size=2,
                decreasing=True,
            ),
        )
        expected_observation = (
            "symmetric_adjacent_physical_faces"
            if "interface" in self.capability
            else "ordinary_finite_volume_faces"
        )
        if self.interface_current_observation != expected_observation:
            raise ValueError(
                "interface_current_observation does not match capability topology"
            )
        if not isinstance(self.gates, DynamicDefectImpedanceGates):
            raise TypeError("gates must be DynamicDefectImpedanceGates")

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
            raise TypeError("dynamic_defect_protocol must be a JSON object")
        _exact_keys(
            payload,
            {field.name for field in dataclasses.fields(cls)},
            cls.__name__,
        )
        values = dict(payload)
        for name in (
            "frequencies_Hz",
            "bulk_defect_document_sha256",
            "interface_defect_document_sha256",
            "refinement_factors",
        ):
            values[name] = tuple(values[name])
        values["gates"] = DynamicDefectImpedanceGates.from_dict(values["gates"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError(
                "dynamic-defect impedance protocol JSON must contain an object"
            )
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class DynamicDefectFrequencyWindowEvidence:
    requested_minimum_frequency_Hz: float
    requested_maximum_frequency_Hz: float
    maximum_sampling_gap_decades: float
    minimum_trap_relaxation_frequency_Hz: float | None
    maximum_trap_relaxation_frequency_Hz: float | None
    trap_low_frequency_limit_covered: bool
    trap_high_frequency_limit_covered: bool
    every_trap_relaxation_frequency_bracketed: bool
    ionic_frequency_window: Any | None
    certified: bool


@dataclass(frozen=True, slots=True)
class DynamicDefectImpedanceEvidence:
    model: Literal["dynamic-defect-impedance-evidence-v1"]
    protocol: DynamicDefectImpedanceProtocol
    protocol_sha256: str
    capability: DynamicDefectCapability
    engine_scope: str
    engine_version: str
    interface_current_observation: InterfaceCurrentObservation
    dc_state_sha256: str
    contact_thermodynamics: ContactThermodynamicCertificate
    dc_certificate: Any
    engine_certificate: Any
    dc_operating_point_certified: bool
    thermodynamically_certified: bool
    frequency_window_certified: bool
    numerically_certified: bool
    certified: bool
    qss_embedding_error: float
    maximum_bulk_trap_balance_relative_error: float | None
    maximum_interface_trap_balance_relative_error: float | None
    maximum_all_face_admittance_spread: float
    maximum_linear_solve_backward_error: float
    minimum_reciprocal_condition: float
    maximum_refinement_relative_change: float
    maximum_ion_inventory_response_relative: float | None
    maximum_current_decomposition_relative_error: float | None
    low_frequency_qss_relative_error: float
    high_frequency_frozen_relative_error: float
    frequency_window: DynamicDefectFrequencyWindowEvidence
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DynamicDefectProductionResult:
    frequencies_Hz: np.ndarray
    impedance_ohm_m2: np.ndarray
    admittance_S_m2: np.ndarray
    admittance_faces_S_m2: np.ndarray
    electron_admittance_faces_S_m2: np.ndarray
    hole_admittance_faces_S_m2: np.ndarray
    displacement_admittance_faces_S_m2: np.ndarray
    electron_storage_response_F_m2: np.ndarray
    hole_storage_response_F_m2: np.ndarray
    positive_ion_admittance_faces_S_m2: np.ndarray | None
    negative_ion_admittance_faces_S_m2: np.ndarray | None
    positive_ion_storage_response_F_m2: np.ndarray | None
    negative_ion_storage_response_F_m2: np.ndarray | None
    bulk_trap_charge_storage_response_F_m2: np.ndarray | None
    interface_sheet_charge_storage_response_F_m2: np.ndarray | None
    bulk_trap_occupancy_response_per_V: np.ndarray | None
    interface_occupancy_response_per_V: np.ndarray | None
    protocol: DynamicDefectImpedanceProtocol
    evidence: DynamicDefectImpedanceEvidence


def _bulk_documents(stack: DeviceStack) -> tuple[str, ...]:
    documents: list[str] = []
    for layer in electrical_layers(stack):
        params = layer.params
        document = None if params is None else params.defect_document
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


def _has_active_mobile_ions(stack: DeviceStack) -> bool:
    for layer in electrical_layers(stack):
        params = layer.params
        if params is None:
            continue
        if (float(params.D_ion) > 0.0 and float(params.P0) > 0.0) or (
            float(params.D_ion_neg) > 0.0 and float(params.P0_neg) > 0.0
        ):
            return True
    return False


def classify_dynamic_defect_capability(
    stack: DeviceStack,
) -> DynamicDefectCapability:
    """Classify only combinations backed by an independently certified adapter."""
    bulk = bool(_bulk_documents(stack))
    raw_interface_defects = tuple(
        defect for defect in electrical_interface_defects(stack) if defect is not None
    )
    interface = bool(raw_interface_defects)
    ions = _has_active_mobile_ions(stack)
    if not bulk and not interface:
        raise DynamicDefectImpedanceCapabilityError(
            "dynamic-defect impedance requires an explicit bulk or canonical "
            "interface defect population"
        )
    if interface:
        try:
            require_uncalibrated_microscopic_interface_defects(
                stack,
                consumer="production dynamic-defect impedance",
            )
        except (TypeError, ValueError) as exc:
            raise DynamicDefectImpedanceCapabilityError(str(exc)) from exc
    if ions:
        if bulk and interface:
            return "bulk_interface_defect_plus_ions"
        if bulk:
            return "bulk_defect_plus_ions"
        return "interface_defect_plus_ions"
    if bulk and interface:
        raise DynamicDefectImpedanceCapabilityError(
            "bulk + interface dynamic defects without active mobile ions do "
            "not yet have a jointly certified production operator"
        )
    return "bulk_dynamic_defect" if bulk else "interface_dynamic_defect"


def build_dynamic_defect_impedance_protocol(
    stack: DeviceStack,
    grid: np.ndarray,
    frequencies_Hz: object,
    *,
    requested_grid_intervals: int,
    V_dc: float,
    delta_V: float,
    illuminated: bool,
    defect_energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    refinement_factors: object = DEFAULT_REFINEMENT_FACTORS,
    gates: DynamicDefectImpedanceGates | None = None,
) -> DynamicDefectImpedanceProtocol:
    capability = classify_dynamic_defect_capability(stack)
    frequencies = _strict_float_tuple(
        frequencies_Hz,
        "frequencies_Hz",
        minimum_size=3,
    )
    factors = _strict_float_tuple(
        refinement_factors,
        "refinement_factors",
        minimum_size=2,
        decreasing=True,
    )
    actual_grid = np.asarray(grid, dtype=float)
    if (
        actual_grid.ndim != 1
        or actual_grid.size < 3
        or not np.all(np.isfinite(actual_grid))
        or np.any(np.diff(actual_grid) <= 0.0)
    ):
        raise ValueError("grid must be finite, one-dimensional, and increasing")
    return DynamicDefectImpedanceProtocol(
        schema_version=DYNAMIC_DEFECT_IMPEDANCE_SCHEMA,
        method=DYNAMIC_DEFECT_IMPEDANCE_METHOD,
        capability=capability,
        V_dc_V=V_dc,
        delta_V=delta_V,
        illuminated=illuminated,
        frequencies_Hz=frequencies,
        requested_grid_intervals=requested_grid_intervals,
        actual_grid_nodes=int(actual_grid.size),
        grid_sha256=ion_aware_impedance_grid_sha256(actual_grid),
        stack_sha256=hashlib.sha256(repr(stack).encode("utf-8")).hexdigest(),
        bulk_defect_document_sha256=_bulk_documents(stack),
        interface_defect_document_sha256=_interface_documents(stack),
        defect_energy_quadrature_order=defect_energy_quadrature_order,
        state_step=state_step,
        voltage_step=voltage_step,
        refinement_factors=factors,
        interface_current_observation=(
            "symmetric_adjacent_physical_faces"
            if "interface" in capability
            else "ordinary_finite_volume_faces"
        ),
        gates=default_dynamic_defect_impedance_gates(capability)
        if gates is None
        else gates,
    )


def resolve_dynamic_defect_impedance_protocol(
    supplied: DynamicDefectImpedanceProtocol | None,
    expected: DynamicDefectImpedanceProtocol,
) -> DynamicDefectImpedanceProtocol:
    if supplied is None:
        return expected
    if not isinstance(supplied, DynamicDefectImpedanceProtocol):
        raise TypeError("dynamic_defect_protocol has the wrong type")
    if supplied != expected:
        raise DynamicDefectImpedanceProtocolError(
            "dynamic_defect_protocol does not match the requested stack, grid, "
            "frequency sampling, perturbation controls, or evidence gates"
        )
    return supplied


def _state_sha256(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in values:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        array[array == 0.0] = 0.0
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _window_evidence(
    frequencies: np.ndarray,
    result: BulkDefectDeviceACResult
    | InterfaceDefectDeviceACResult
    | DefectIonCombinedResult,
) -> DynamicDefectFrequencyWindowEvidence:
    gap = float(np.max(np.diff(np.log10(frequencies))))
    if isinstance(result, DefectIonCombinedResult):
        window = result.certificate.frequency_window
        return DynamicDefectFrequencyWindowEvidence(
            requested_minimum_frequency_Hz=float(frequencies[0]),
            requested_maximum_frequency_Hz=float(frequencies[-1]),
            maximum_sampling_gap_decades=gap,
            minimum_trap_relaxation_frequency_Hz=float(
                window.minimum_trap_relaxation_frequency_Hz
            ),
            maximum_trap_relaxation_frequency_Hz=float(
                window.maximum_trap_relaxation_frequency_Hz
            ),
            trap_low_frequency_limit_covered=bool(
                window.trap_low_frequency_limit_covered
            ),
            trap_high_frequency_limit_covered=bool(
                window.trap_high_frequency_limit_covered
            ),
            every_trap_relaxation_frequency_bracketed=bool(
                window.every_trap_relaxation_frequency_bracketed
            ),
            ionic_frequency_window=window.ionic,
            certified=bool(window.certified),
        )
    window = result.certificate.frequency_window
    return DynamicDefectFrequencyWindowEvidence(
        requested_minimum_frequency_Hz=float(window.requested_minimum_frequency_Hz),
        requested_maximum_frequency_Hz=float(window.requested_maximum_frequency_Hz),
        maximum_sampling_gap_decades=float(window.maximum_sampling_gap_decades),
        minimum_trap_relaxation_frequency_Hz=float(
            window.minimum_relaxation_frequency_Hz
        ),
        maximum_trap_relaxation_frequency_Hz=float(
            window.maximum_relaxation_frequency_Hz
        ),
        trap_low_frequency_limit_covered=bool(window.low_frequency_limit_covered),
        trap_high_frequency_limit_covered=bool(window.high_frequency_limit_covered),
        every_trap_relaxation_frequency_bracketed=bool(
            window.every_relaxation_frequency_bracketed
        ),
        ionic_frequency_window=None,
        certified=bool(window.certified),
    )


def _contact_certificate(
    grid: np.ndarray,
    stack: DeviceStack,
    *,
    energy_order: int,
):
    material_stack = stack
    if getattr(stack, "interface_charge_closure", "off") != "off":
        material_stack = dataclasses.replace(
            stack,
            interface_charge_closure="off",
            interface_charge_rebaseline_acknowledged=False,
        )
    material = build_material_arrays(
        grid,
        material_stack,
        explicit_defect_charge_closure=(
            EXPLICIT_DEFECT_CHARGE_QF_DC
            if _bulk_documents(stack)
            else EXPLICIT_DEFECT_CHARGE_OFF
        ),
        defect_energy_quadrature_order=energy_order,
    )
    certificate = assess_contact_thermodynamics(stack, material)
    if not certificate.certified:
        raise DynamicDefectImpedanceCapabilityError(
            "production dynamic-defect impedance requires certified contact "
            f"thermodynamics, got {certificate.status}: {certificate.message}"
        )
    return material, certificate


def _project_bulk(
    result: BulkDefectDeviceACResult,
    protocol: DynamicDefectImpedanceProtocol,
    contact: ContactThermodynamicCertificate,
) -> DynamicDefectProductionResult:
    certificate = result.certificate
    dc = result.dc_state
    window = _window_evidence(result.frequencies_Hz, result)
    evidence = DynamicDefectImpedanceEvidence(
        model=DYNAMIC_DEFECT_IMPEDANCE_EVIDENCE,
        protocol=protocol,
        protocol_sha256=protocol.protocol_hash,
        capability=protocol.capability,
        engine_scope=result.scope,
        engine_version=result.version,
        interface_current_observation=protocol.interface_current_observation,
        dc_state_sha256=_state_sha256("bulk-dynamic-defect-dc-v1", dc.y, dc.phi),
        contact_thermodynamics=contact,
        dc_certificate={
            "certified": certificate.dc_operating_point_certified,
            "maximum_normalized_residual": certificate.dc_maximum_normalized_residual,
            "electron_continuity_bound_A_m2": certificate.dc_electron_continuity_bound_A_m2,
            "hole_continuity_bound_A_m2": certificate.dc_hole_continuity_bound_A_m2,
            "face_current_spread_A_m2": certificate.dc_face_current_spread_A_m2,
            "poisson_residual": certificate.dc_poisson_residual,
        },
        engine_certificate=certificate,
        dc_operating_point_certified=bool(certificate.dc_operating_point_certified),
        thermodynamically_certified=bool(contact.certified),
        frequency_window_certified=bool(window.certified),
        numerically_certified=bool(certificate.certified),
        certified=bool(certificate.certified and contact.certified),
        qss_embedding_error=float(certificate.qss_embedding_normalized_error),
        maximum_bulk_trap_balance_relative_error=float(
            certificate.maximum_local_trap_balance_relative_error
        ),
        maximum_interface_trap_balance_relative_error=None,
        maximum_all_face_admittance_spread=float(
            certificate.maximum_all_face_admittance_spread
        ),
        maximum_linear_solve_backward_error=float(
            certificate.maximum_linear_solve_backward_error
        ),
        minimum_reciprocal_condition=float(certificate.minimum_reciprocal_condition),
        maximum_refinement_relative_change=float(
            certificate.maximum_refinement_relative_change
        ),
        maximum_ion_inventory_response_relative=None,
        maximum_current_decomposition_relative_error=None,
        low_frequency_qss_relative_error=float(
            certificate.low_frequency_qss_relative_error
        ),
        high_frequency_frozen_relative_error=float(
            certificate.high_frequency_frozen_relative_error
        ),
        frequency_window=window,
        reasons=tuple(certificate.reasons),
    )
    return DynamicDefectProductionResult(
        frequencies_Hz=result.frequencies_Hz,
        impedance_ohm_m2=result.impedance_ohm_m2,
        admittance_S_m2=result.admittance_S_m2,
        admittance_faces_S_m2=result.admittance_faces_S_m2,
        electron_admittance_faces_S_m2=(
            result.electron_conduction_admittance_faces_S_m2
        ),
        hole_admittance_faces_S_m2=result.hole_conduction_admittance_faces_S_m2,
        displacement_admittance_faces_S_m2=result.displacement_admittance_faces_S_m2,
        electron_storage_response_F_m2=result.electron_storage_response_F_m2,
        hole_storage_response_F_m2=result.hole_storage_response_F_m2,
        positive_ion_admittance_faces_S_m2=None,
        negative_ion_admittance_faces_S_m2=None,
        positive_ion_storage_response_F_m2=None,
        negative_ion_storage_response_F_m2=None,
        bulk_trap_charge_storage_response_F_m2=(
            result.trap_charge_storage_response_F_m2
        ),
        interface_sheet_charge_storage_response_F_m2=None,
        bulk_trap_occupancy_response_per_V=result.trap_occupancy_response_per_V,
        interface_occupancy_response_per_V=None,
        protocol=protocol,
        evidence=evidence,
    )


def _project_interface(
    result: InterfaceDefectDeviceACResult,
    protocol: DynamicDefectImpedanceProtocol,
    contact: ContactThermodynamicCertificate,
) -> DynamicDefectProductionResult:
    certificate = result.certificate
    dc = result.dc_state
    window = _window_evidence(result.frequencies_Hz, result)
    evidence = DynamicDefectImpedanceEvidence(
        model=DYNAMIC_DEFECT_IMPEDANCE_EVIDENCE,
        protocol=protocol,
        protocol_sha256=protocol.protocol_hash,
        capability=protocol.capability,
        engine_scope=result.scope,
        engine_version=result.version,
        interface_current_observation=protocol.interface_current_observation,
        dc_state_sha256=_state_sha256("interface-dynamic-defect-dc-v1", dc.y, dc.phi),
        contact_thermodynamics=contact,
        dc_certificate={
            "certified": certificate.dc_operating_point_certified,
            "state_operator_match_error": certificate.dc_state_operator_match_error,
            "maximum_normalized_residual": certificate.dc_maximum_normalized_residual,
            "electron_continuity_bound_A_m2": certificate.dc_electron_continuity_bound_A_m2,
            "hole_continuity_bound_A_m2": certificate.dc_hole_continuity_bound_A_m2,
            "face_current_spread_A_m2": certificate.dc_face_current_spread_A_m2,
            "poisson_residual": certificate.dc_poisson_residual,
        },
        engine_certificate=certificate,
        dc_operating_point_certified=bool(certificate.dc_operating_point_certified),
        thermodynamically_certified=bool(contact.certified),
        frequency_window_certified=bool(window.certified),
        numerically_certified=bool(certificate.certified),
        certified=bool(certificate.certified and contact.certified),
        qss_embedding_error=float(certificate.qss_embedding_normalized_error),
        maximum_bulk_trap_balance_relative_error=None,
        maximum_interface_trap_balance_relative_error=float(
            certificate.maximum_local_trap_balance_relative_error
        ),
        maximum_all_face_admittance_spread=float(
            certificate.maximum_all_face_admittance_spread
        ),
        maximum_linear_solve_backward_error=float(
            certificate.maximum_linear_solve_backward_error
        ),
        minimum_reciprocal_condition=float(certificate.minimum_reciprocal_condition),
        maximum_refinement_relative_change=float(
            certificate.maximum_refinement_relative_change
        ),
        maximum_ion_inventory_response_relative=None,
        maximum_current_decomposition_relative_error=None,
        low_frequency_qss_relative_error=float(
            certificate.low_frequency_qss_relative_error
        ),
        high_frequency_frozen_relative_error=float(
            certificate.high_frequency_frozen_relative_error
        ),
        frequency_window=window,
        reasons=tuple(certificate.reasons),
    )
    return DynamicDefectProductionResult(
        frequencies_Hz=result.frequencies_Hz,
        impedance_ohm_m2=result.impedance_ohm_m2,
        admittance_S_m2=result.admittance_S_m2,
        admittance_faces_S_m2=result.admittance_faces_S_m2,
        electron_admittance_faces_S_m2=(
            result.electron_conduction_admittance_faces_S_m2
        ),
        hole_admittance_faces_S_m2=result.hole_conduction_admittance_faces_S_m2,
        displacement_admittance_faces_S_m2=result.displacement_admittance_faces_S_m2,
        electron_storage_response_F_m2=result.electron_storage_response_F_m2,
        hole_storage_response_F_m2=result.hole_storage_response_F_m2,
        positive_ion_admittance_faces_S_m2=None,
        negative_ion_admittance_faces_S_m2=None,
        positive_ion_storage_response_F_m2=None,
        negative_ion_storage_response_F_m2=None,
        bulk_trap_charge_storage_response_F_m2=None,
        interface_sheet_charge_storage_response_F_m2=(
            result.interface_sheet_charge_storage_response_F_m2
        ),
        bulk_trap_occupancy_response_per_V=None,
        interface_occupancy_response_per_V=result.interface_occupancy_response_per_V,
        protocol=protocol,
        evidence=evidence,
    )


def _project_combined(
    result: DefectIonCombinedResult,
    protocol: DynamicDefectImpedanceProtocol,
) -> DynamicDefectProductionResult:
    certificate = result.certificate
    contact = result.dc_state.certificate.contact_thermodynamics
    window = _window_evidence(result.frequencies_Hz, result)
    evidence = DynamicDefectImpedanceEvidence(
        model=DYNAMIC_DEFECT_IMPEDANCE_EVIDENCE,
        protocol=protocol,
        protocol_sha256=protocol.protocol_hash,
        capability=protocol.capability,
        engine_scope=result.scope,
        engine_version=result.version,
        interface_current_observation=result.interface_current_observation,
        dc_state_sha256=result.dc_state.state_sha256,
        contact_thermodynamics=contact,
        dc_certificate=result.dc_state.certificate,
        engine_certificate=certificate,
        dc_operating_point_certified=bool(certificate.dc_operating_point_certified),
        thermodynamically_certified=bool(contact.certified),
        frequency_window_certified=bool(window.certified),
        numerically_certified=bool(certificate.certified),
        certified=bool(certificate.certified and contact.certified),
        qss_embedding_error=float(certificate.qss_embedding_relative_error),
        maximum_bulk_trap_balance_relative_error=float(
            certificate.maximum_bulk_trap_balance_relative_error
        ),
        maximum_interface_trap_balance_relative_error=float(
            certificate.maximum_interface_trap_balance_relative_error
        ),
        maximum_all_face_admittance_spread=float(
            certificate.maximum_all_face_admittance_spread
        ),
        maximum_linear_solve_backward_error=float(
            certificate.maximum_linear_solve_backward_error
        ),
        minimum_reciprocal_condition=float(certificate.minimum_reciprocal_condition),
        maximum_refinement_relative_change=float(
            certificate.maximum_refinement_relative_change
        ),
        maximum_ion_inventory_response_relative=float(
            certificate.maximum_ion_inventory_response_relative
        ),
        maximum_current_decomposition_relative_error=float(
            certificate.maximum_current_decomposition_relative_error
        ),
        low_frequency_qss_relative_error=float(
            certificate.low_frequency_qss_relative_error
        ),
        high_frequency_frozen_relative_error=float(
            certificate.high_frequency_frozen_relative_error
        ),
        frequency_window=window,
        reasons=tuple(certificate.reasons),
    )
    return DynamicDefectProductionResult(
        frequencies_Hz=result.frequencies_Hz,
        impedance_ohm_m2=result.impedance_ohm_m2,
        admittance_S_m2=result.admittance_S_m2,
        admittance_faces_S_m2=result.admittance_faces_S_m2,
        electron_admittance_faces_S_m2=result.electron_admittance_faces_S_m2,
        hole_admittance_faces_S_m2=result.hole_admittance_faces_S_m2,
        displacement_admittance_faces_S_m2=result.displacement_admittance_faces_S_m2,
        electron_storage_response_F_m2=result.electron_storage_response_F_m2,
        hole_storage_response_F_m2=result.hole_storage_response_F_m2,
        positive_ion_admittance_faces_S_m2=(result.positive_ion_admittance_faces_S_m2),
        negative_ion_admittance_faces_S_m2=(result.negative_ion_admittance_faces_S_m2),
        positive_ion_storage_response_F_m2=(result.positive_ion_storage_response_F_m2),
        negative_ion_storage_response_F_m2=(result.negative_ion_storage_response_F_m2),
        bulk_trap_charge_storage_response_F_m2=(
            result.bulk_trap_charge_storage_response_F_m2
        ),
        interface_sheet_charge_storage_response_F_m2=(
            result.interface_sheet_charge_storage_response_F_m2
        ),
        bulk_trap_occupancy_response_per_V=(result.bulk_trap_occupancy_response_per_V),
        interface_occupancy_response_per_V=(result.interface_occupancy_response_per_V),
        protocol=protocol,
        evidence=evidence,
    )


def run_dynamic_defect_impedance(
    grid: np.ndarray,
    stack: DeviceStack,
    protocol: DynamicDefectImpedanceProtocol,
    *,
    progress: ProgressCallback | None = None,
) -> DynamicDefectProductionResult:
    """Execute the exact adapter and gates frozen in ``protocol``."""
    if not isinstance(protocol, DynamicDefectImpedanceProtocol):
        raise TypeError("protocol must be DynamicDefectImpedanceProtocol")
    expected = build_dynamic_defect_impedance_protocol(
        stack,
        grid,
        protocol.frequencies_Hz,
        requested_grid_intervals=protocol.requested_grid_intervals,
        V_dc=protocol.V_dc_V,
        delta_V=protocol.delta_V,
        illuminated=protocol.illuminated,
        defect_energy_quadrature_order=protocol.defect_energy_quadrature_order,
        state_step=protocol.state_step,
        voltage_step=protocol.voltage_step,
        refinement_factors=protocol.refinement_factors,
        gates=protocol.gates,
    )
    resolve_dynamic_defect_impedance_protocol(protocol, expected)
    frequencies = np.asarray(protocol.frequencies_Hz, dtype=float)
    gates = protocol.gates
    try:
        if protocol.capability == "bulk_dynamic_defect":
            material, contact = _contact_certificate(
                np.asarray(grid, dtype=float),
                stack,
                energy_order=protocol.defect_energy_quadrature_order,
            )
            result = run_bulk_defect_device_impedance(
                grid,
                stack,
                frequencies,
                V_dc=protocol.V_dc_V,
                delta_V=protocol.delta_V,
                illuminated=protocol.illuminated,
                mat=material,
                defect_energy_quadrature_order=(
                    protocol.defect_energy_quadrature_order
                ),
                state_step=protocol.state_step,
                voltage_step=protocol.voltage_step,
                refinement_factors=protocol.refinement_factors,
                frequency_branch_margin_decades=(gates.frequency_branch_margin_decades),
                maximum_frequency_sampling_gap_decades=(
                    gates.maximum_frequency_sampling_gap_decades
                ),
                maximum_dc_normalized_residual=(gates.maximum_dc_normalized_residual),
                maximum_dc_continuity_bound_A_m2=(
                    gates.maximum_dc_continuity_bound_A_m2
                ),
                maximum_dc_face_current_spread_A_m2=(
                    gates.maximum_dc_face_current_spread_A_m2
                ),
                maximum_dc_poisson_residual=gates.maximum_dc_poisson_residual,
                maximum_qss_embedding_normalized_error=(
                    gates.maximum_qss_embedding_error
                ),
                maximum_local_trap_balance_relative_error=(
                    gates.maximum_trap_balance_relative_error
                ),
                maximum_all_face_admittance_spread=(
                    gates.maximum_all_face_admittance_spread
                ),
                maximum_linear_solve_backward_error=(
                    gates.maximum_linear_solve_backward_error
                ),
                maximum_refinement_relative_change=(
                    gates.maximum_refinement_relative_change
                ),
                maximum_limit_relative_error=gates.maximum_limit_relative_error,
                require_certificate=True,
                progress=progress,
            )
            return _project_bulk(result, protocol, contact)
        if protocol.capability == "interface_dynamic_defect":
            _material, contact = _contact_certificate(
                np.asarray(grid, dtype=float),
                stack,
                energy_order=protocol.defect_energy_quadrature_order,
            )
            result = run_interface_defect_device_impedance(
                grid,
                stack,
                frequencies,
                V_dc=protocol.V_dc_V,
                delta_V=protocol.delta_V,
                illuminated=protocol.illuminated,
                state_step=protocol.state_step,
                voltage_step=protocol.voltage_step,
                refinement_factors=protocol.refinement_factors,
                frequency_branch_margin_decades=(gates.frequency_branch_margin_decades),
                maximum_frequency_sampling_gap_decades=(
                    gates.maximum_frequency_sampling_gap_decades
                ),
                maximum_dc_operator_match_error=(gates.maximum_dc_operator_match_error),
                maximum_dc_normalized_residual=(gates.maximum_dc_normalized_residual),
                maximum_dc_continuity_bound_A_m2=(
                    gates.maximum_dc_continuity_bound_A_m2
                ),
                maximum_dc_face_current_spread_A_m2=(
                    gates.maximum_dc_face_current_spread_A_m2
                ),
                maximum_dc_poisson_residual=gates.maximum_dc_poisson_residual,
                maximum_qss_embedding_normalized_error=(
                    gates.maximum_qss_embedding_error
                ),
                maximum_local_interface_residual=(
                    gates.maximum_local_interface_residual
                ),
                maximum_local_gauss_residual=(gates.maximum_local_gauss_residual),
                maximum_local_trap_balance_relative_error=(
                    gates.maximum_trap_balance_relative_error
                ),
                maximum_all_face_admittance_spread=(
                    gates.maximum_all_face_admittance_spread
                ),
                maximum_linear_solve_backward_error=(
                    gates.maximum_linear_solve_backward_error
                ),
                maximum_refinement_relative_change=(
                    gates.maximum_refinement_relative_change
                ),
                maximum_limit_relative_error=gates.maximum_limit_relative_error,
                require_certificate=True,
                progress=progress,
            )
            return _project_interface(result, protocol, contact)
        result = run_defect_ion_combined_impedance(
            grid,
            stack,
            frequencies,
            V_dc=protocol.V_dc_V,
            delta_V=protocol.delta_V,
            illuminated=protocol.illuminated,
            state_step=protocol.state_step,
            voltage_step=protocol.voltage_step,
            refinement_factors=protocol.refinement_factors,
            defect_energy_quadrature_order=protocol.defect_energy_quadrature_order,
            frequency_branch_margin_decades=gates.frequency_branch_margin_decades,
            maximum_frequency_sampling_gap_decades=(
                gates.maximum_frequency_sampling_gap_decades
            ),
            maximum_dc_normalized_residual=gates.maximum_dc_normalized_residual,
            maximum_dc_continuity_bound_A_m2=(gates.maximum_dc_continuity_bound_A_m2),
            maximum_dc_ionic_face_current_A_m2=(
                gates.maximum_dc_ionic_face_current_A_m2
            ),
            maximum_dc_inventory_error=gates.maximum_dc_inventory_error,
            maximum_dc_poisson_residual=gates.maximum_dc_poisson_residual,
            maximum_dc_face_current_spread_A_m2=(
                gates.maximum_dc_face_current_spread_A_m2
            ),
            maximum_qss_embedding_relative_error=(gates.maximum_qss_embedding_error),
            maximum_trap_balance_relative_error=(
                gates.maximum_trap_balance_relative_error
            ),
            maximum_all_face_admittance_spread=(
                gates.maximum_all_face_admittance_spread
            ),
            maximum_linear_solve_backward_error=(
                gates.maximum_linear_solve_backward_error
            ),
            maximum_refinement_relative_change=(
                gates.maximum_refinement_relative_change
            ),
            maximum_ion_inventory_response_relative=(
                gates.maximum_ion_inventory_response_relative
            ),
            maximum_current_decomposition_relative_error=(
                gates.maximum_current_decomposition_relative_error
            ),
            maximum_limit_relative_error=gates.maximum_limit_relative_error,
            dc_max_nfev=gates.dc_max_nfev,
            require_certificate=True,
            progress=progress,
        )
        return _project_combined(result, protocol)
    except (
        BulkDefectDeviceACError,
        InterfaceDefectDeviceACError,
        DefectIonCombinedError,
        QuasiFermiSteadyStateError,
    ) as exc:
        raise DynamicDefectImpedanceCertificationError(str(exc)) from exc


__all__ = [
    "DEFAULT_REFINEMENT_FACTORS",
    "DYNAMIC_DEFECT_IMPEDANCE_EVIDENCE",
    "DYNAMIC_DEFECT_IMPEDANCE_METHOD",
    "DYNAMIC_DEFECT_IMPEDANCE_SCHEMA",
    "DynamicDefectCapability",
    "DynamicDefectFrequencyWindowEvidence",
    "DynamicDefectImpedanceCapabilityError",
    "DynamicDefectImpedanceCertificationError",
    "DynamicDefectImpedanceError",
    "DynamicDefectImpedanceEvidence",
    "DynamicDefectImpedanceGates",
    "DynamicDefectImpedanceProtocol",
    "DynamicDefectImpedanceProtocolError",
    "DynamicDefectProductionResult",
    "build_dynamic_defect_impedance_protocol",
    "classify_dynamic_defect_capability",
    "default_dynamic_defect_impedance_gates",
    "resolve_dynamic_defect_impedance_protocol",
    "run_dynamic_defect_impedance",
]
