"""Protocol-bound J-V execution for equilibrium-referenced interface charge.

This module is intentionally narrower than the general J-V drivers. It runs an
illuminated, ion-free, ascending quasi-Fermi sweep from one certified charge-off
dark reference and retains the interface-charge evidence at every requested
voltage. No transient, dark-JV, AC, 2-D, or hysteresis capability is implied.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Self

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments.jv_sweep import (
    compute_metrics,
    thermodynamic_voc_ceiling,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    DEFAULT_ILLUMINATION_STEPS,
    EquilibriumReferencedInterfaceChargeDarkReference,
    QuasiFermiJVSweepResult,
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    build_equilibrium_referenced_interface_charge_dark_reference,
    solve_equilibrium_referenced_interface_charge_steady_state,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.contacts import CONTACT_THERMODYNAMIC_TOLERANCE_EV
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import TWO_SIDED_TRACE


INTERFACE_CHARGE_JV_CAPABILITY = "equilibrium_referenced_interface_charge_qf_dc_v1"
INTERFACE_CHARGE_JV_PROTOCOL_SCHEMA = "interface-charge-jv-protocol-v1"
INTERFACE_CHARGE_JV_EVIDENCE_MODEL = "interface-charge-jv-evidence-v1"
DEFAULT_INTERFACE_CHARGE_JV_LIMITATIONS = (
    "illuminated ion-free ascending quasi-Fermi J-V only",
    "no transient hysteresis, dark J-V, AC, 2-D, or external-circuit claim",
    "internal numerical evidence is not SCAPS, external-solver, or experimental validation",
)

ProgressCallback = Callable[[str, int, int, str], None]


class InterfaceChargeJVProtocolError(ValueError):
    """The charged J-V execution protocol is invalid or mismatched."""


class InterfaceChargeJVCertificationError(RuntimeError):
    """A requested charged J-V point failed a registered physical gate."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return 0.0 if number == 0.0 else number


def _positive(value: object, field: str) -> float:
    number = _finite(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _positive_integer(value: object, field: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise TypeError(f"{field} must be a positive integer")
    return int(value)


def _exact_mapping(payload: Mapping[str, Any], cls: type, label: str) -> dict[str, Any]:
    expected = {field.name for field in dataclasses.fields(cls)}
    actual = set(payload)
    if actual != expected:
        raise InterfaceChargeJVProtocolError(
            f"{label} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return dict(payload)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise InterfaceChargeJVProtocolError(
                f"duplicate interface-charge J-V protocol key: {key!r}"
            )
        payload[key] = value
    return payload


@dataclass(frozen=True, slots=True)
class InterfaceChargeJVSolverControls:
    """Frozen base controls; refinement tightens only residual tolerances."""

    illumination_steps: tuple[float, ...] = DEFAULT_ILLUMINATION_STEPS
    finite_difference_step: float = 7.0e-6
    newton_residual_tolerance: float = 4.0e-7
    max_newton_iterations: int = 60
    poisson_tolerance_V: float = 1.0e-12
    poisson_max_iterations: int = 100
    continuity_tolerance_A_m2: float = 1.0e-4
    current_spread_tolerance_A_m2: float = 1.0e-4
    poisson_residual_tolerance: float = 1.0e-8
    minimum_voltage_step_V: float = 1.0e-3
    max_voltage_bridge_points: int = 256

    def __post_init__(self) -> None:
        steps = tuple(
            _finite(value, f"illumination_steps[{index}]")
            for index, value in enumerate(self.illumination_steps)
        )
        if (
            not steps
            or steps[0] != 0.0
            or steps[-1] != 1.0
            or any(right <= left for left, right in zip(steps, steps[1:]))
        ):
            raise ValueError("illumination_steps must increase strictly from 0 to 1")
        object.__setattr__(self, "illumination_steps", steps)
        finite_difference_step = _positive(
            self.finite_difference_step, "finite_difference_step"
        )
        if finite_difference_step != 7.0e-6:
            raise ValueError(
                "interface-charge J-V v1 fixes finite_difference_step at 7e-6"
            )
        object.__setattr__(self, "finite_difference_step", finite_difference_step)
        for name in (
            "newton_residual_tolerance",
            "poisson_tolerance_V",
            "continuity_tolerance_A_m2",
            "current_spread_tolerance_A_m2",
            "poisson_residual_tolerance",
            "minimum_voltage_step_V",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_newton_iterations",
            _positive_integer(self.max_newton_iterations, "max_newton_iterations"),
        )
        object.__setattr__(
            self,
            "poisson_max_iterations",
            _positive_integer(self.poisson_max_iterations, "poisson_max_iterations"),
        )
        object.__setattr__(
            self,
            "max_voltage_bridge_points",
            _positive_integer(
                self.max_voltage_bridge_points, "max_voltage_bridge_points"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["illumination_steps"] = list(self.illumination_steps)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("interface-charge J-V solver controls must be a mapping")
        values = _exact_mapping(payload, cls, cls.__name__)
        values["illumination_steps"] = tuple(values["illumination_steps"])
        return cls(**values)

    def refined(self, tolerance_factor: float) -> dict[str, float | int | tuple[float, ...]]:
        factor = _positive(tolerance_factor, "tolerance_factor")
        if factor > 1.0:
            raise ValueError("tolerance_factor may tighten but not relax the v1 controls")
        return {
            "illumination_steps": self.illumination_steps,
            "finite_difference_step": self.finite_difference_step,
            "newton_residual_tolerance": self.newton_residual_tolerance * factor,
            "max_newton_iterations": self.max_newton_iterations,
            "poisson_tolerance_V": self.poisson_tolerance_V * factor,
            "poisson_max_iterations": self.poisson_max_iterations,
            "continuity_tolerance_A_m2": self.continuity_tolerance_A_m2,
            "current_spread_tolerance_A_m2": self.current_spread_tolerance_A_m2,
            "poisson_residual_tolerance": self.poisson_residual_tolerance,
        }


@dataclass(frozen=True, slots=True)
class InterfaceChargeJVAcceptance:
    max_normalized_cell_residual: float = 4.0e-7
    max_interface_local_residual: float = 1.0e-7
    max_normalized_gauss_residual: float = 1.0e-10
    max_scaled_local_jacobian_condition: float = 1.0e8
    max_continuity_bound_A_m2: float = 1.0e-4
    max_face_current_spread_A_m2: float = 1.0e-4
    max_poisson_residual: float = 1.0e-8
    max_contact_fermi_level_span_eV: float = (
        CONTACT_THERMODYNAMIC_TOLERANCE_EV
    )
    require_contact_thermodynamic_certificate: bool = True
    require_dark_charge_off_bit_identity: bool = True
    require_voc_bracket: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_normalized_cell_residual",
            "max_interface_local_residual",
            "max_normalized_gauss_residual",
            "max_scaled_local_jacobian_condition",
            "max_continuity_bound_A_m2",
            "max_face_current_spread_A_m2",
            "max_poisson_residual",
            "max_contact_fermi_level_span_eV",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if (
            self.max_contact_fermi_level_span_eV
            > CONTACT_THERMODYNAMIC_TOLERANCE_EV
        ):
            raise ValueError(
                "interface-charge J-V cannot relax the fixed contact "
                "thermodynamic gate"
            )
        for name in (
            "require_contact_thermodynamic_certificate",
            "require_dark_charge_off_bit_identity",
            "require_voc_bracket",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"interface-charge J-V v1 requires {name}=true")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("interface-charge J-V acceptance must be a mapping")
        return cls(**_exact_mapping(payload, cls, cls.__name__))


@dataclass(frozen=True, slots=True)
class InterfaceChargeJVProtocol:
    """Canonical physical and numerical contract for the D4-E3 J-V slice."""

    voltages_V: tuple[float, ...]
    temperature_K: float
    P_in_W_m2: float = 1000.0
    solver_controls: InterfaceChargeJVSolverControls = dataclasses.field(
        default_factory=InterfaceChargeJVSolverControls
    )
    acceptance: InterfaceChargeJVAcceptance = dataclasses.field(
        default_factory=InterfaceChargeJVAcceptance
    )
    capability: Literal[
        "equilibrium_referenced_interface_charge_qf_dc_v1"
    ] = INTERFACE_CHARGE_JV_CAPABILITY
    solver: Literal["quasi_fermi"] = "quasi_fermi"
    illumination: Literal["stack_baseline_one_sun"] = "stack_baseline_one_sun"
    branch_semantics: Literal[
        "ascending_zero_scan_rate"
    ] = "ascending_zero_scan_rate"
    initial_state_source: Literal[
        "certified_charge_off_dark_reference"
    ] = "certified_charge_off_dark_reference"
    interface_topology: Literal["two_sided_trace"] = TWO_SIDED_TRACE
    interface_transport_model: Literal[
        "fermi_dirac_richardson"
    ] = FERMI_DIRAC_RICHARDSON
    interface_transmission: float = 1.0
    charge_law: Literal["-q*N_t*(f-f_eq)"] = "-q*N_t*(f-f_eq)"
    stop_after_voc: Literal[True] = True
    mpp_interpolation: Literal["sampled"] = "sampled"
    schema_version: Literal[
        "interface-charge-jv-protocol-v1"
    ] = INTERFACE_CHARGE_JV_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        voltages = tuple(
            _finite(value, f"voltages_V[{index}]")
            for index, value in enumerate(self.voltages_V)
        )
        if (
            len(voltages) < 2
            or voltages[0] != 0.0
            or any(
                right <= left
                for left, right in zip(voltages, voltages[1:])
            )
        ):
            raise ValueError(
                "voltages_V must start at 0 V and be finite and strictly increasing"
            )
        object.__setattr__(self, "voltages_V", voltages)
        object.__setattr__(self, "temperature_K", _positive(self.temperature_K, "temperature_K"))
        object.__setattr__(self, "P_in_W_m2", _positive(self.P_in_W_m2, "P_in_W_m2"))
        if not isinstance(self.solver_controls, InterfaceChargeJVSolverControls):
            raise TypeError("solver_controls must be InterfaceChargeJVSolverControls")
        if not isinstance(self.acceptance, InterfaceChargeJVAcceptance):
            raise TypeError("acceptance must be InterfaceChargeJVAcceptance")
        constants = {
            "capability": INTERFACE_CHARGE_JV_CAPABILITY,
            "solver": "quasi_fermi",
            "illumination": "stack_baseline_one_sun",
            "branch_semantics": "ascending_zero_scan_rate",
            "initial_state_source": "certified_charge_off_dark_reference",
            "interface_topology": TWO_SIDED_TRACE,
            "interface_transport_model": FERMI_DIRAC_RICHARDSON,
            "charge_law": "-q*N_t*(f-f_eq)",
            "mpp_interpolation": "sampled",
            "schema_version": INTERFACE_CHARGE_JV_PROTOCOL_SCHEMA,
        }
        for name, expected in constants.items():
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported {name} for interface-charge J-V v1")
        if self.interface_transmission != 1.0:
            raise ValueError("interface-charge J-V v1 requires unit transmission")
        if self.stop_after_voc is not True:
            raise ValueError("interface-charge J-V v1 requires stop_after_voc=true")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["voltages_V"] = list(self.voltages_V)
        payload["solver_controls"] = self.solver_controls.to_dict()
        payload["acceptance"] = self.acceptance.to_dict()
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
    def protocol_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("interface-charge J-V protocol must be a mapping")
        values = _exact_mapping(payload, cls, cls.__name__)
        values["voltages_V"] = tuple(values["voltages_V"])
        values["solver_controls"] = InterfaceChargeJVSolverControls.from_dict(
            values["solver_controls"]
        )
        values["acceptance"] = InterfaceChargeJVAcceptance.from_dict(
            values["acceptance"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(parsed, Mapping):
            raise TypeError("interface-charge J-V protocol JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class InterfaceChargeJVPointEvidence:
    voltage_V: float
    current_A_m2: float
    occupancy: tuple[float, ...]
    incremental_sheet_charge_C_m2: tuple[float, ...]
    trace_potential_shift_V: tuple[tuple[float, float], ...]
    normalized_gauss_residual: tuple[float, ...]
    scaled_local_jacobian_condition: tuple[float, ...]
    interface_local_residual: float
    max_normalized_cell_residual: float
    electron_continuity_bound_A_m2: float
    hole_continuity_bound_A_m2: float
    face_current_spread_A_m2: float
    poisson_residual: float
    contact_thermodynamic_status: Literal["certified"]
    contact_fermi_level_span_eV: float
    certified: Literal[True] = True


@dataclass(frozen=True, slots=True)
class InterfaceChargeJVEvidence:
    model: Literal["interface-charge-jv-evidence-v1"]
    capability: Literal[
        "equilibrium_referenced_interface_charge_qf_dc_v1"
    ]
    protocol: InterfaceChargeJVProtocol
    protocol_sha256: str
    grid_sha256: str
    stack_sha256: str
    dark_state_sha256: str
    dark_contact_thermodynamic_status: Literal["certified"]
    dark_contact_fermi_level_span_eV: float
    interface_defect_document_sha256: tuple[str, ...]
    capture_velocities_m_s: tuple[tuple[float, float], ...]
    trap_density_m2: tuple[float, ...]
    equilibrium_occupancy: tuple[float, ...]
    dark_charge_off_bit_identity_verified: Literal[True]
    points: tuple[InterfaceChargeJVPointEvidence, ...]
    continuation_bridges: tuple[InterfaceChargeJVPointEvidence, ...]
    continuation_bridge_count: int
    tolerance_factor: float
    minimum_occupancy: float
    maximum_occupancy: float
    maximum_absolute_sheet_charge_C_m2: float
    maximum_absolute_trace_potential_shift_V: float
    maximum_normalized_gauss_residual: float
    maximum_scaled_local_jacobian_condition: float
    maximum_interface_local_residual: float
    maximum_normalized_cell_residual: float
    maximum_continuity_bound_A_m2: float
    maximum_face_current_spread_A_m2: float
    maximum_poisson_residual: float
    maximum_contact_fermi_level_span_eV: float
    limitations: tuple[str, ...] = DEFAULT_INTERFACE_CHARGE_JV_LIMITATIONS


@dataclass(frozen=True, slots=True)
class InterfaceChargeJVExecution:
    protocol: InterfaceChargeJVProtocol
    dark_reference: EquilibriumReferencedInterfaceChargeDarkReference
    charged_dark: QuasiFermiSteadyStateResult
    sweep: QuasiFermiJVSweepResult
    evidence: InterfaceChargeJVEvidence


def build_interface_charge_jv_protocol(
    stack: DeviceStack,
    voltages_V: np.ndarray | tuple[float, ...] | list[float],
    *,
    P_in_W_m2: float = 1000.0,
) -> InterfaceChargeJVProtocol:
    """Build the default v1 protocol without inventing a finite scan rate."""

    return InterfaceChargeJVProtocol(
        voltages_V=tuple(float(value) for value in voltages_V),
        temperature_K=float(stack.T),
        P_in_W_m2=P_in_W_m2,
    )


def _dark_identity_verified(
    reference: EquilibriumReferencedInterfaceChargeDarkReference,
    charged_dark: QuasiFermiSteadyStateResult,
) -> bool:
    fields = (
        "y",
        "phi",
        "electron_quasi_fermi_potential_V",
        "hole_quasi_fermi_potential_V",
        "electron_face_current_A_m2",
        "hole_face_current_A_m2",
        "total_face_current_A_m2",
        "electron_rate_per_s",
        "hole_rate_per_s",
    )
    return bool(
        all(
            np.array_equal(
                np.asarray(getattr(reference.dark_state, name)),
                np.asarray(getattr(charged_dark, name)),
            )
            for name in fields
        )
        and charged_dark.interface_equilibrium_occupancy
        == reference.equilibrium_occupancy
        and charged_dark.interface_occupancy == reference.equilibrium_occupancy
        and all(value == 0.0 for value in charged_dark.interface_incremental_sheet_charge_C_m2)
        and all(
            values == (0.0, 0.0)
            for values in charged_dark.interface_trace_potential_shift_V
        )
        and charged_dark.interface_charge_reference_grid_sha256
        == reference.grid_sha256
        and charged_dark.interface_charge_reference_stack_sha256
        == reference.stack_sha256
        and charged_dark.interface_charge_reference_dark_state_sha256
        == reference.dark_state_sha256
    )


def _point_evidence(
    point: QuasiFermiSteadyStateResult,
    voltage: float,
    reference: EquilibriumReferencedInterfaceChargeDarkReference,
    acceptance: InterfaceChargeJVAcceptance,
) -> InterfaceChargeJVPointEvidence:
    count = len(reference.equilibrium_occupancy)
    occupancy = np.asarray(point.interface_occupancy, dtype=float)
    charge = np.asarray(point.interface_incremental_sheet_charge_C_m2, dtype=float)
    density = np.asarray(reference.trap_density_m2, dtype=float)
    equilibrium = np.asarray(reference.equilibrium_occupancy, dtype=float)
    point_equilibrium = np.asarray(
        point.interface_equilibrium_occupancy,
        dtype=float,
    )
    gauss = np.asarray(point.interface_normalized_gauss_residual, dtype=float)
    condition = np.asarray(
        point.interface_scaled_local_jacobian_condition, dtype=float
    )
    trace = np.asarray(point.interface_trace_potential_shift_V, dtype=float)
    arrays = (
        occupancy,
        charge,
        density,
        equilibrium,
        point_equilibrium,
        gauss,
        condition,
    )
    if (
        not point.certified
        or not point.illuminated
        or point.interface_charge_closure != "equilibrium_referenced"
        or not point.interface_boundary
        or point.interface_topology != TWO_SIDED_TRACE
        or point.interface_transport_model != FERMI_DIRAC_RICHARDSON
        or not np.isclose(point.V_app, voltage, rtol=0.0, atol=1.0e-14)
        or not np.isclose(
            point.interface_transmission,
            reference.interface_transmission,
            rtol=0.0,
            atol=0.0,
        )
        or not np.isfinite(point.current_A_m2)
        or count == 0
        or len(point.interface_faces) != count
        or any(values.shape != (count,) for values in arrays)
        or trace.shape != (count, 2)
        or any(not np.all(np.isfinite(values)) for values in (*arrays, trace))
        or not np.array_equal(point_equilibrium, equilibrium)
        or np.any(occupancy < 0.0)
        or np.any(occupancy > 1.0)
        or point.interface_charge_reference_grid_sha256 != reference.grid_sha256
        or point.interface_charge_reference_stack_sha256 != reference.stack_sha256
        or point.interface_charge_reference_dark_state_sha256
        != reference.dark_state_sha256
    ):
        raise InterfaceChargeJVCertificationError(
            f"charged J-V point at {voltage:.9g} V lacks aligned physical evidence"
        )
    expected_charge = -Q * density * (occupancy - equilibrium)
    charge_scale = np.maximum(Q * density, np.finfo(float).tiny)
    if np.any(np.abs(charge - expected_charge) > 1.0e-11 * charge_scale):
        raise InterfaceChargeJVCertificationError(
            f"charged J-V point at {voltage:.9g} V violates -q*N_t*(f-f_eq)"
        )
    if np.any(np.abs(charge) > charge_scale * (1.0 + 1.0e-12)):
        raise InterfaceChargeJVCertificationError(
            f"charged J-V point at {voltage:.9g} V exceeds one electron per trap"
        )
    scalars = {
        "max_normalized_cell_residual": (
            abs(float(point.max_normalized_cell_residual)),
            acceptance.max_normalized_cell_residual,
        ),
        "interface_local_residual": (
            abs(float(point.interface_local_residual)),
            acceptance.max_interface_local_residual,
        ),
        "electron_continuity_bound_A_m2": (
            abs(float(point.electron_continuity_bound_A_m2)),
            acceptance.max_continuity_bound_A_m2,
        ),
        "hole_continuity_bound_A_m2": (
            abs(float(point.hole_continuity_bound_A_m2)),
            acceptance.max_continuity_bound_A_m2,
        ),
        "face_current_spread_A_m2": (
            abs(float(point.face_current_spread_A_m2)),
            acceptance.max_face_current_spread_A_m2,
        ),
        "poisson_residual": (
            abs(float(point.poisson_residual)),
            acceptance.max_poisson_residual,
        ),
    }
    for name, (value, limit) in scalars.items():
        if not np.isfinite(value) or value > limit:
            raise InterfaceChargeJVCertificationError(
                f"charged J-V point at {voltage:.9g} V fails {name}: "
                f"{value:.6g} > {limit:.6g}"
            )
    if np.max(np.abs(gauss)) > acceptance.max_normalized_gauss_residual:
        raise InterfaceChargeJVCertificationError(
            f"charged J-V point at {voltage:.9g} V fails the Gauss residual gate"
        )
    if np.max(condition) > acceptance.max_scaled_local_jacobian_condition:
        raise InterfaceChargeJVCertificationError(
            f"charged J-V point at {voltage:.9g} V fails the Jacobian condition gate"
        )
    contact_span = point.contact_fermi_level_span_eV
    if acceptance.require_contact_thermodynamic_certificate and (
        point.contact_thermodynamic_status != "certified"
        or contact_span is None
        or not np.isfinite(contact_span)
        or contact_span < 0.0
        or contact_span > acceptance.max_contact_fermi_level_span_eV
    ):
        raise InterfaceChargeJVCertificationError(
            f"charged J-V point at {voltage:.9g} V lacks contact certification"
        )
    return InterfaceChargeJVPointEvidence(
        voltage_V=float(voltage),
        current_A_m2=float(point.current_A_m2),
        occupancy=tuple(float(value) for value in occupancy),
        incremental_sheet_charge_C_m2=tuple(float(value) for value in charge),
        trace_potential_shift_V=tuple(
            (float(values[0]), float(values[1])) for values in trace
        ),
        normalized_gauss_residual=tuple(float(value) for value in gauss),
        scaled_local_jacobian_condition=tuple(float(value) for value in condition),
        interface_local_residual=scalars["interface_local_residual"][0],
        max_normalized_cell_residual=scalars["max_normalized_cell_residual"][0],
        electron_continuity_bound_A_m2=scalars[
            "electron_continuity_bound_A_m2"
        ][0],
        hole_continuity_bound_A_m2=scalars["hole_continuity_bound_A_m2"][0],
        face_current_spread_A_m2=scalars["face_current_spread_A_m2"][0],
        poisson_residual=scalars["poisson_residual"][0],
        contact_thermodynamic_status="certified",
        contact_fermi_level_span_eV=float(contact_span),
    )


def _build_evidence(
    protocol: InterfaceChargeJVProtocol,
    reference: EquilibriumReferencedInterfaceChargeDarkReference,
    charged_dark: QuasiFermiSteadyStateResult,
    sweep: QuasiFermiJVSweepResult,
    continuation_bridges: tuple[InterfaceChargeJVPointEvidence, ...],
    tolerance_factor: float,
) -> InterfaceChargeJVEvidence:
    if not _dark_identity_verified(reference, charged_dark):
        raise InterfaceChargeJVCertificationError(
            "charged dark state is not bit-identical to the charge-off reference"
        )
    dark_spans = (
        reference.dark_state.contact_fermi_level_span_eV,
        charged_dark.contact_fermi_level_span_eV,
    )
    if (
        reference.dark_state.contact_thermodynamic_status != "certified"
        or charged_dark.contact_thermodynamic_status != "certified"
        or any(value is None or not np.isfinite(value) for value in dark_spans)
        or any(float(value) < 0.0 for value in dark_spans)
        or any(
            float(value) > protocol.acceptance.max_contact_fermi_level_span_eV
            for value in dark_spans
        )
    ):
        raise InterfaceChargeJVCertificationError(
            "charged J-V dark anchor lacks contact certification"
        )
    points = tuple(
        _point_evidence(point, float(voltage), reference, protocol.acceptance)
        for voltage, point in zip(sweep.voltages_V, sweep.points, strict=True)
    )
    if not points or len(points) != len(sweep.voltages_V):
        raise InterfaceChargeJVCertificationError(
            "charged J-V point evidence is empty or misaligned"
        )
    if len(continuation_bridges) != sweep.continuation_bridge_count:
        raise InterfaceChargeJVCertificationError(
            "charged J-V continuation bridge evidence is misaligned"
        )
    audited_points = points + continuation_bridges
    occupancy = np.asarray(
        [value for point in audited_points for value in point.occupancy], dtype=float
    )
    sheet_charge = np.asarray(
        [
            value
            for point in audited_points
            for value in point.incremental_sheet_charge_C_m2
        ],
        dtype=float,
    )
    trace_shift = np.asarray(
        [
            value
            for point in audited_points
            for pair in point.trace_potential_shift_V
            for value in pair
        ],
        dtype=float,
    )
    gauss = np.asarray(
        [
            value
            for point in audited_points
            for value in point.normalized_gauss_residual
        ],
        dtype=float,
    )
    condition = np.asarray(
        [
            value
            for point in audited_points
            for value in point.scaled_local_jacobian_condition
        ],
        dtype=float,
    )
    return InterfaceChargeJVEvidence(
        model=INTERFACE_CHARGE_JV_EVIDENCE_MODEL,
        capability=INTERFACE_CHARGE_JV_CAPABILITY,
        protocol=protocol,
        protocol_sha256=protocol.protocol_sha256,
        grid_sha256=reference.grid_sha256,
        stack_sha256=reference.stack_sha256,
        dark_state_sha256=reference.dark_state_sha256,
        dark_contact_thermodynamic_status="certified",
        dark_contact_fermi_level_span_eV=float(dark_spans[0]),
        interface_defect_document_sha256=(
            reference.interface_defect_document_sha256
        ),
        capture_velocities_m_s=reference.capture_velocities_m_s,
        trap_density_m2=reference.trap_density_m2,
        equilibrium_occupancy=reference.equilibrium_occupancy,
        dark_charge_off_bit_identity_verified=True,
        points=points,
        continuation_bridges=continuation_bridges,
        continuation_bridge_count=int(sweep.continuation_bridge_count),
        tolerance_factor=float(tolerance_factor),
        minimum_occupancy=float(np.min(occupancy)),
        maximum_occupancy=float(np.max(occupancy)),
        maximum_absolute_sheet_charge_C_m2=float(np.max(np.abs(sheet_charge))),
        maximum_absolute_trace_potential_shift_V=float(
            np.max(np.abs(trace_shift))
        ),
        maximum_normalized_gauss_residual=float(np.max(np.abs(gauss))),
        maximum_scaled_local_jacobian_condition=float(np.max(condition)),
        maximum_interface_local_residual=max(
            point.interface_local_residual for point in audited_points
        ),
        maximum_normalized_cell_residual=max(
            point.max_normalized_cell_residual for point in audited_points
        ),
        maximum_continuity_bound_A_m2=max(
            max(
                point.electron_continuity_bound_A_m2,
                point.hole_continuity_bound_A_m2,
            )
            for point in audited_points
        ),
        maximum_face_current_spread_A_m2=max(
            point.face_current_spread_A_m2 for point in audited_points
        ),
        maximum_poisson_residual=max(
            point.poisson_residual for point in audited_points
        ),
        maximum_contact_fermi_level_span_eV=max(
            point.contact_fermi_level_span_eV for point in audited_points
        ),
    )


def solve_interface_charge_jv(
    x: np.ndarray,
    stack: DeviceStack,
    protocol: InterfaceChargeJVProtocol,
    *,
    tolerance_factor: float = 1.0,
    progress: ProgressCallback | None = None,
) -> InterfaceChargeJVExecution:
    """Execute and certify the protocol-bound charged QF J-V slice."""

    if not isinstance(protocol, InterfaceChargeJVProtocol):
        raise TypeError("protocol must be InterfaceChargeJVProtocol")
    if not np.isclose(protocol.temperature_K, stack.T, rtol=0.0, atol=1.0e-12):
        raise InterfaceChargeJVProtocolError(
            "protocol temperature_K does not match the device stack"
        )
    factor = _positive(tolerance_factor, "tolerance_factor")
    controls = protocol.solver_controls.refined(factor)
    illumination_steps = tuple(controls.pop("illumination_steps"))
    grid = np.asarray(x, dtype=float)
    reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid,
        stack,
        interface_transmission=protocol.interface_transmission,
        require_contact_certificate=(
            protocol.acceptance.require_contact_thermodynamic_certificate
        ),
        **controls,
    )
    charged_dark = solve_equilibrium_referenced_interface_charge_steady_state(
        grid,
        stack,
        0.0,
        dark_reference=reference,
        illuminated=False,
        require_contact_certificate=(
            protocol.acceptance.require_contact_thermodynamic_certificate
        ),
        **controls,
    )

    requested = np.asarray(protocol.voltages_V, dtype=float)
    retained_voltages: list[float] = []
    retained_points: list[QuasiFermiSteadyStateResult] = []
    previous: QuasiFermiSteadyStateResult | None = None
    previous_voltage: float | None = None
    bridge_count = 0
    bridge_evidence: list[InterfaceChargeJVPointEvidence] = []
    voc_crossed = False

    def solve_at_voltage(
        voltage: float,
        seed: QuasiFermiSteadyStateResult | None,
        stages: tuple[float, ...],
    ) -> QuasiFermiSteadyStateResult:
        return solve_equilibrium_referenced_interface_charge_steady_state(
            grid,
            stack,
            voltage,
            dark_reference=reference,
            illuminated=True,
            initial_state=seed,
            illumination_steps=stages,
            require_contact_certificate=(
                protocol.acceptance.require_contact_thermodynamic_certificate
            ),
            **controls,
        )

    def advance(
        left_voltage: float,
        left_state: QuasiFermiSteadyStateResult,
        right_voltage: float,
    ) -> QuasiFermiSteadyStateResult:
        nonlocal bridge_count
        try:
            return solve_at_voltage(right_voltage, left_state, (1.0,))
        except QuasiFermiSteadyStateError as exc:
            span = right_voltage - left_voltage
            minimum = protocol.solver_controls.minimum_voltage_step_V
            if span <= minimum * (1.0 + 1.0e-12):
                raise InterfaceChargeJVCertificationError(
                    "charged J-V continuation failed at the minimum interval "
                    f"[{left_voltage:.9g}, {right_voltage:.9g}] V: {exc}"
                ) from exc
            if bridge_count >= protocol.solver_controls.max_voltage_bridge_points:
                raise InterfaceChargeJVCertificationError(
                    "charged J-V continuation exceeded the bridge-point limit"
                ) from exc
            midpoint = 0.5 * (left_voltage + right_voltage)
            bridge_count += 1
            middle = advance(left_voltage, left_state, midpoint)
            bridge_evidence.append(
                _point_evidence(
                    middle,
                    midpoint,
                    reference,
                    protocol.acceptance,
                )
            )
            return advance(midpoint, middle, right_voltage)

    for index, voltage_value in enumerate(requested):
        voltage = float(voltage_value)
        if index == 0:
            point = solve_at_voltage(voltage, None, illumination_steps)
        else:
            assert previous is not None and previous_voltage is not None
            point = advance(previous_voltage, previous, voltage)
        retained_voltages.append(voltage)
        retained_points.append(point)
        if progress is not None:
            progress(
                "interface_charge_jv",
                index + 1,
                len(requested),
                f"Certified charged QF point {index + 1}/{len(requested)}",
            )
        if (
            len(retained_points) >= 2
            and retained_points[-2].current_A_m2 > 0.0
            and point.current_A_m2 <= 0.0
        ):
            voc_crossed = True
        previous = point
        previous_voltage = voltage
        if protocol.stop_after_voc and voc_crossed:
            break

    voltages = np.asarray(retained_voltages, dtype=float)
    currents = np.asarray(
        [point.current_A_m2 for point in retained_points], dtype=float
    )
    metrics = compute_metrics(
        voltages,
        currents,
        P_in=protocol.P_in_W_m2,
        V_oc_max=thermodynamic_voc_ceiling(stack),
        validity=[point.certified for point in retained_points],
        mpp_interpolation=protocol.mpp_interpolation,
    )
    if protocol.acceptance.require_voc_bracket and not metrics.voc_bracketed:
        raise InterfaceChargeJVCertificationError(
            "charged J-V voltage window did not bracket open circuit; "
            f"terminal point is V={voltages[-1]:.9g} V, "
            f"J={currents[-1]:.9g} A/m2"
        )
    sweep = QuasiFermiJVSweepResult(
        voltages_V=voltages,
        currents_A_m2=currents,
        points=tuple(retained_points),
        metrics=metrics,
        continuation_bridge_count=bridge_count,
        minimum_voltage_step_V=(
            protocol.solver_controls.minimum_voltage_step_V
        ),
        mpp_interpolation=protocol.mpp_interpolation,
        defect_energy_quadrature_order=(
            retained_points[0].defect_energy_quadrature_order
        ),
        defect_distribution_kinds=(
            retained_points[0].defect_distribution_kinds
        ),
    )
    evidence = _build_evidence(
        protocol,
        reference,
        charged_dark,
        sweep,
        tuple(bridge_evidence),
        factor,
    )
    return InterfaceChargeJVExecution(
        protocol=protocol,
        dark_reference=reference,
        charged_dark=charged_dark,
        sweep=sweep,
        evidence=evidence,
    )


__all__ = [
    "DEFAULT_INTERFACE_CHARGE_JV_LIMITATIONS",
    "INTERFACE_CHARGE_JV_CAPABILITY",
    "INTERFACE_CHARGE_JV_EVIDENCE_MODEL",
    "INTERFACE_CHARGE_JV_PROTOCOL_SCHEMA",
    "InterfaceChargeJVAcceptance",
    "InterfaceChargeJVCertificationError",
    "InterfaceChargeJVEvidence",
    "InterfaceChargeJVExecution",
    "InterfaceChargeJVPointEvidence",
    "InterfaceChargeJVProtocol",
    "InterfaceChargeJVProtocolError",
    "InterfaceChargeJVSolverControls",
    "build_interface_charge_jv_protocol",
    "solve_interface_charge_jv",
]
