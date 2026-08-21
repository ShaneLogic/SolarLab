"""Residual-certified one-dimensional DC preparation with mobile ions.

This module deliberately separates a *finite-time integration* from a DC
certificate.  A state is promoted only after independent full-MOL residual,
per-species ionic face-current, all-face current-spread, inventory, density,
and site-occupancy gates pass at consecutive endpoints of a declared time
ladder.  Contact thermodynamics is reported as a separate evidence axis so a
legacy deck cannot be mislabeled, while its numerical fixed point can still
be studied honestly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Literal, Mapping, Self

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    compute_ionic_current_components,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.contacts import (
    ContactThermodynamicCertificate,
    assess_contact_thermodynamics,
)
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver.mol import (
    MaterialArrays,
    StateVec,
    assemble_rhs,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsPolicy,
    NumericalDiagnosticsReport,
)
from perovskite_sim.solver.tolerances import AbsoluteTolerance, ComponentwiseAtol


ION_AWARE_DC_PROTOCOL_SCHEMA = "ion-aware-dc-protocol-v1"
ION_AWARE_DC_STATE_HASH_SCHEMA = "ion-aware-dc-state-f64le-v1"
DEFAULT_SETTLE_END_TIMES_S = (
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    32.0,
    64.0,
    128.0,
)
ProgressCallback = Callable[[str, int, int, str], None]


class IonAwareDCCapabilityError(ValueError):
    """The selected model topology cannot enter this certification lane."""


class IonAwareDCSolverError(RuntimeError):
    """A transient segment failed before producing assessable DC evidence."""

    def __init__(
        self,
        message: str,
        *,
        target_time_s: float | None = None,
        attempts: tuple["IonAwareDCAttempt", ...] = (),
    ) -> None:
        self.target_time_s = target_time_s
        self.attempts = attempts
        super().__init__(message)


class IonAwareDCCertificationError(RuntimeError):
    """The declared ladder ended without the requested certificate."""

    def __init__(self, message: str, result: "IonAwareDCResult") -> None:
        self.result = result
        super().__init__(message)


def ion_aware_dc_state_sha256(state: np.ndarray) -> str:
    """Return a platform-independent hash for one packed physical state."""
    array = np.asarray(state, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("ion-aware DC state hash requires a finite 1-D array")
    canonical = np.array(array, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(ION_AWARE_DC_STATE_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(int(canonical.size).to_bytes(8, byteorder="big", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


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


def _nonnegative(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


@dataclass(frozen=True, slots=True)
class IonAwareDCProtocol:
    """Physical history and fail-closed acceptance rule for one DC state."""

    V_dc: float
    illuminated: bool
    temperature_K: float
    initial_state_source: Literal[
        "dark_quasineutral_equilibrium", "user_supplied_state"
    ] = "dark_quasineutral_equilibrium"
    initial_state_sha256: str | None = None
    illumination_source: str | None = "stack_baseline_generation"
    settle_end_times_s: tuple[float, ...] = DEFAULT_SETTLE_END_TIMES_S
    required_consecutive_passes: int = 2
    max_carrier_area_rate_A_m2: float = 1.0e-1
    max_ion_area_rate_A_m2: float = 1.0e-6
    max_ionic_face_current_A_m2: float = 1.0e-6
    max_dc_face_current_spread_A_m2: float = 1.0e-1
    max_ion_inventory_relative_drift: float = 1.0e-10
    terminal_density_floor_m3: float = 0.0
    ion_boundary_condition: Literal["blocking"] = "blocking"
    schema_version: Literal["ion-aware-dc-protocol-v1"] = (
        ION_AWARE_DC_PROTOCOL_SCHEMA
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "V_dc", _finite(self.V_dc, "V_dc"))
        if not isinstance(self.illuminated, bool):
            raise TypeError("illuminated must be boolean")
        object.__setattr__(
            self, "temperature_K", _positive(self.temperature_K, "temperature_K")
        )
        if self.initial_state_source not in {
            "dark_quasineutral_equilibrium",
            "user_supplied_state",
        }:
            raise ValueError("unknown ion-aware DC initial_state_source")
        if self.initial_state_source == "dark_quasineutral_equilibrium":
            if self.initial_state_sha256 is not None:
                raise ValueError(
                    "dark_quasineutral_equilibrium cannot carry "
                    "initial_state_sha256"
                )
        elif (
            not isinstance(self.initial_state_sha256, str)
            or len(self.initial_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.initial_state_sha256
            )
        ):
            raise ValueError(
                "user_supplied_state requires a lowercase SHA-256 identity"
            )
        if self.illuminated:
            if not isinstance(self.illumination_source, str) or not (
                self.illumination_source.strip()
            ):
                raise ValueError(
                    "illuminated ion-aware DC requires illumination_source"
                )
        elif self.illumination_source is not None:
            raise ValueError("dark ion-aware DC cannot carry illumination_source")
        try:
            times = tuple(
                _positive(value, f"settle_end_times_s[{index}]")
                for index, value in enumerate(self.settle_end_times_s)
            )
        except TypeError as exc:
            raise TypeError("settle_end_times_s must be an iterable") from exc
        if not times or any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("settle_end_times_s must be non-empty and increasing")
        object.__setattr__(self, "settle_end_times_s", times)
        passes = self.required_consecutive_passes
        if (
            isinstance(passes, (bool, np.bool_))
            or not isinstance(passes, Integral)
            or passes < 1
            or passes > len(times)
        ):
            raise ValueError(
                "required_consecutive_passes must be an integer within the ladder"
            )
        object.__setattr__(self, "required_consecutive_passes", int(passes))
        for name in (
            "max_carrier_area_rate_A_m2",
            "max_ion_area_rate_A_m2",
            "max_ionic_face_current_A_m2",
            "max_dc_face_current_spread_A_m2",
            "max_ion_inventory_relative_drift",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "terminal_density_floor_m3",
            _nonnegative(self.terminal_density_floor_m3, "terminal_density_floor_m3"),
        )
        if self.ion_boundary_condition != "blocking":
            raise ValueError("ion-aware DC v1 supports only blocking ion boundaries")
        if self.schema_version != ION_AWARE_DC_PROTOCOL_SCHEMA:
            raise ValueError("unsupported ion-aware DC protocol schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["settle_end_times_s"] = list(self.settle_end_times_s)
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
            raise TypeError("ion-aware DC protocol must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "ion-aware DC protocol keys do not match schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        values = dict(payload)
        values["settle_end_times_s"] = tuple(values["settle_end_times_s"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("ion-aware DC protocol JSON must contain an object")
        return cls.from_dict(parsed)


def build_ion_aware_dc_protocol(
    stack: DeviceStack,
    *,
    V_dc: float,
    illuminated: bool,
    initial_state_source: Literal[
        "dark_quasineutral_equilibrium", "user_supplied_state"
    ] = "dark_quasineutral_equilibrium",
    initial_state_sha256: str | None = None,
    settle_end_times_s: tuple[float, ...] = DEFAULT_SETTLE_END_TIMES_S,
    required_consecutive_passes: int = 2,
    max_carrier_area_rate_A_m2: float = 1.0e-1,
    max_ion_area_rate_A_m2: float = 1.0e-6,
    max_ionic_face_current_A_m2: float = 1.0e-6,
    max_dc_face_current_spread_A_m2: float = 1.0e-1,
    max_ion_inventory_relative_drift: float = 1.0e-10,
) -> IonAwareDCProtocol:
    """Build the explicit protocol consumed by :func:`solve_ion_aware_dc`."""
    from perovskite_sim.models.mode import resolve_mode

    mode = resolve_mode(getattr(stack, "mode", "full"))
    effective_temperature = float(stack.T) if mode.use_temperature_scaling else 300.0
    return IonAwareDCProtocol(
        V_dc=V_dc,
        illuminated=illuminated,
        temperature_K=effective_temperature,
        initial_state_source=initial_state_source,
        initial_state_sha256=initial_state_sha256,
        illumination_source=("stack_baseline_generation" if illuminated else None),
        settle_end_times_s=settle_end_times_s,
        required_consecutive_passes=required_consecutive_passes,
        max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
        max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
        max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
        max_dc_face_current_spread_A_m2=max_dc_face_current_spread_A_m2,
        max_ion_inventory_relative_drift=max_ion_inventory_relative_drift,
    )


@dataclass(frozen=True, slots=True)
class IonInventoryEvidence:
    species: Literal["positive", "negative"]
    initial_inventory_m2: float
    terminal_inventory_m2: float
    relative_drift: float
    terminal_centroid_fraction: float


@dataclass(frozen=True, slots=True)
class IonAwareDCStateCertificate:
    """Independent evidence evaluated on one fixed-bias terminal state."""

    certified: bool
    numerically_certified: bool
    thermodynamically_certified: bool
    carrier_area_rate_A_m2: float
    electron_area_rate_A_m2: float
    hole_area_rate_A_m2: float
    ion_area_rate_A_m2: float
    positive_ion_area_rate_A_m2: float
    negative_ion_area_rate_A_m2: float | None
    max_ionic_face_current_A_m2: float
    max_positive_ionic_face_current_A_m2: float
    max_negative_ionic_face_current_A_m2: float | None
    dc_face_current_spread_A_m2: float
    dc_current_density_A_m2: float
    minimum_electron_density_m3: float
    minimum_hole_density_m3: float
    minimum_positive_ion_density_m3: float
    minimum_negative_ion_density_m3: float | None
    maximum_site_occupancy_fraction: float
    positive_ion_inventory: IonInventoryEvidence
    negative_ion_inventory: IonInventoryEvidence | None
    max_ion_inventory_relative_drift: float
    contact_thermodynamics: ContactThermodynamicCertificate
    numerical_reasons: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareDCAttempt:
    method: str
    success: bool
    message: str
    numerical_diagnostics: NumericalDiagnosticsReport | None
    nfev: int | None
    njev: int | None
    nlu: int | None


@dataclass(frozen=True, slots=True)
class IonAwareDCStep:
    target_time_s: float
    segment_duration_s: float
    state_certificate: IonAwareDCStateCertificate
    numerical_diagnostics: NumericalDiagnosticsReport
    independent_state_passed: bool
    diagnostics_passed: bool
    accepted_for_closure: bool
    accepted_method: str
    attempts: tuple[IonAwareDCAttempt, ...]
    nfev: int | None
    njev: int | None
    nlu: int | None


@dataclass(frozen=True, slots=True)
class IonAwareDCResult:
    x: np.ndarray
    y: np.ndarray
    protocol: IonAwareDCProtocol
    steps: tuple[IonAwareDCStep, ...]
    state_certificate: IonAwareDCStateCertificate
    consecutive_certified_steps: int
    numerically_certified: bool
    thermodynamically_certified: bool
    certified: bool
    total_settle_time_s: float

    @property
    def protocol_hash(self) -> str:
        return self.protocol.protocol_hash


def _optional_count(solution: object, field: str) -> int | None:
    value = getattr(solution, field, None)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return None


def _active_minimum(values: np.ndarray, active: np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)[np.asarray(active, dtype=bool)]
    return float(np.min(selected)) if selected.size else float("nan")


def _inventory_evidence(
    x: np.ndarray,
    initial: np.ndarray,
    terminal: np.ndarray,
    species: Literal["positive", "negative"],
) -> IonInventoryEvidence:
    initial_inventory = dual_cell_integral(x, initial)
    terminal_inventory = dual_cell_integral(x, terminal)
    scale = max(abs(initial_inventory), np.finfo(float).tiny)
    drift = abs(terminal_inventory - initial_inventory) / scale
    moment = dual_cell_integral(x, np.asarray(terminal, dtype=float) * x)
    length = float(x[-1] - x[0])
    centroid_fraction = (
        (moment / terminal_inventory - float(x[0])) / length
        if terminal_inventory > 0.0 and length > 0.0
        else float("nan")
    )
    return IonInventoryEvidence(
        species=species,
        initial_inventory_m2=float(initial_inventory),
        terminal_inventory_m2=float(terminal_inventory),
        relative_drift=float(drift),
        terminal_centroid_fraction=float(centroid_fraction),
    )


def _maximum_site_occupancy(state: StateVec, mat: MaterialArrays) -> float:
    if mat.has_dual_ions and state.P_neg is not None:
        if mat.ion_steric_shared_site:
            limit = np.minimum(mat.P_lim_node, mat.P_lim_neg_node)
            density = state.P + state.P_neg
            ratios = np.divide(
                density,
                limit,
                out=np.full_like(density, np.inf, dtype=float),
                where=np.asarray(limit) > 0.0,
            )
            return float(np.max(ratios))
        positive = np.divide(state.P, mat.P_lim_node)
        negative = np.divide(state.P_neg, mat.P_lim_neg_node)
        return float(max(np.max(positive), np.max(negative)))
    return float(np.max(np.divide(state.P, mat.P_lim_node)))


def _validate_initial_state(
    state_array: np.ndarray,
    n_nodes: int,
    material: MaterialArrays,
    protocol: IonAwareDCProtocol,
) -> None:
    expected = (4 if material.has_dual_ions else 3) * n_nodes
    if state_array.shape != (expected,):
        raise ValueError(
            f"ion-aware DC initial state must have shape ({expected},)"
        )
    if not np.all(np.isfinite(state_array)):
        raise ValueError("ion-aware DC initial state must be finite")
    state = StateVec.unpack(state_array, n_nodes)
    positive_active = np.asarray(material.P_ion0, dtype=float) > 0.0
    negative_active = (
        np.asarray(material.P_ion0_neg, dtype=float) > 0.0
        if material.has_dual_ions and material.P_ion0_neg is not None
        else np.zeros(n_nodes, dtype=bool)
    )
    floor = protocol.terminal_density_floor_m3
    if np.any(state.n <= floor) or np.any(state.p <= floor):
        raise ValueError(
            "ion-aware DC initial electron and hole densities must be above "
            "the protocol floor"
        )
    if np.any(state.P[positive_active] <= floor):
        raise ValueError(
            "active positive-ion initial densities must be above the protocol floor"
        )
    if np.any(state.P[~positive_active] < 0.0):
        raise ValueError("inactive positive-ion initial densities cannot be negative")
    if state.P_neg is not None:
        if np.any(state.P_neg[negative_active] <= floor):
            raise ValueError(
                "active negative-ion initial densities must be above the protocol floor"
            )
        if np.any(state.P_neg[~negative_active] < 0.0):
            raise ValueError(
                "inactive negative-ion initial densities cannot be negative"
            )
    occupancy = _maximum_site_occupancy(state, material)
    if not np.isfinite(occupancy) or occupancy > 1.0 + 1.0e-8:
        raise ValueError("ion-aware DC initial ion-site occupancy is inadmissible")


def assess_ion_aware_dc_state(
    x: np.ndarray,
    y: np.ndarray,
    initial_state: np.ndarray,
    stack: DeviceStack,
    protocol: IonAwareDCProtocol,
    *,
    mat: MaterialArrays | None = None,
) -> IonAwareDCStateCertificate:
    """Evaluate a full-MOL DC state without trusting elapsed time."""
    coordinates = np.asarray(x, dtype=float)
    state_array = np.asarray(y, dtype=float)
    initial_array = np.asarray(initial_state, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 3:
        raise ValueError("ion-aware DC requires a one-dimensional grid with >=3 nodes")
    if not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("ion-aware DC grid must be finite and strictly increasing")
    material = build_material_arrays(coordinates, stack) if mat is None else mat
    if material.N_iface_state:
        raise IonAwareDCCapabilityError(
            "ion-aware DC v1 excludes dynamic interface-state blocks until "
            "their electrostatic charge closure is certified"
        )
    expected = (4 if material.has_dual_ions else 3) * coordinates.size
    if state_array.shape != (expected,) or initial_array.shape != (expected,):
        raise ValueError(f"ion-aware DC state vectors must both have length {expected}")

    state = StateVec.unpack(state_array, coordinates.size)
    initial = StateVec.unpack(initial_array, coordinates.size)
    rate_array = assemble_rhs(
        0.0,
        state_array,
        coordinates,
        stack,
        material,
        illuminated=protocol.illuminated,
        V_app=protocol.V_dc,
    )
    rate = StateVec.unpack(rate_array, coordinates.size)
    widths = np.asarray(material.dx_cell, dtype=float)
    electron_rate = float(Q * np.sum(np.abs(rate.n) * widths))
    hole_rate = float(Q * np.sum(np.abs(rate.p) * widths))
    positive_ion_rate = float(Q * np.sum(np.abs(rate.P) * widths))
    negative_ion_rate = None
    if rate.P_neg is not None:
        negative_ion_rate = float(Q * np.sum(np.abs(rate.P_neg) * widths))
    ion_rate = positive_ion_rate + (negative_ion_rate or 0.0)

    current = compute_current_components(
        coordinates, state_array, stack, protocol.V_dc, mat=material
    )
    ionic = compute_ionic_current_components(
        coordinates, state_array, stack, protocol.V_dc, mat=material
    )
    positive_face_current = float(np.max(np.abs(ionic.J_positive)))
    negative_face_current = (
        None
        if ionic.J_negative is None
        else float(np.max(np.abs(ionic.J_negative)))
    )
    maximum_ionic_face_current = max(
        positive_face_current, negative_face_current or 0.0
    )
    face_spread = float(np.ptp(current.J_total))
    dc_current = float(np.mean(current.J_total))

    positive_active = np.asarray(material.P_ion0, dtype=float) > 0.0
    negative_active = (
        np.asarray(material.P_ion0_neg, dtype=float) > 0.0
        if material.has_dual_ions and material.P_ion0_neg is not None
        else np.zeros(coordinates.size, dtype=bool)
    )
    minimum_n = float(np.min(state.n))
    minimum_p = float(np.min(state.p))
    minimum_positive = _active_minimum(state.P, positive_active)
    minimum_negative = (
        _active_minimum(state.P_neg, negative_active)
        if state.P_neg is not None
        else None
    )
    occupancy = _maximum_site_occupancy(state, material)
    positive_inventory = _inventory_evidence(
        coordinates, initial.P, state.P, "positive"
    )
    negative_inventory = (
        _inventory_evidence(coordinates, initial.P_neg, state.P_neg, "negative")
        if initial.P_neg is not None and state.P_neg is not None
        else None
    )
    inventory_drift = max(
        positive_inventory.relative_drift,
        negative_inventory.relative_drift if negative_inventory is not None else 0.0,
    )
    contact = assess_contact_thermodynamics(stack, material)

    reasons: list[str] = []
    if not np.all(np.isfinite(state_array)):
        reasons.append("dc_state_nonfinite")
    if not np.all(np.isfinite(rate_array)):
        reasons.append("state_rate_nonfinite")
    finite_metrics = (
        electron_rate,
        hole_rate,
        ion_rate,
        maximum_ionic_face_current,
        face_spread,
        dc_current,
        inventory_drift,
        occupancy,
        minimum_n,
        minimum_p,
        minimum_positive,
    )
    if not all(np.isfinite(value) for value in finite_metrics):
        reasons.append("dc_evidence_nonfinite")
    if minimum_negative is not None and not np.isfinite(minimum_negative):
        reasons.append("negative_ion_evidence_nonfinite")
    floor = protocol.terminal_density_floor_m3
    for label, minimum in (
        ("electron", minimum_n),
        ("hole", minimum_p),
        ("positive_ion", minimum_positive),
        ("negative_ion", minimum_negative),
    ):
        if minimum is not None and np.isfinite(minimum) and minimum <= floor:
            reasons.append(f"{label}_density_not_above_floor")
    if np.any(state.P[~positive_active] < 0.0):
        reasons.append("inactive_positive_ion_density_negative")
    if state.P_neg is not None and np.any(state.P_neg[~negative_active] < 0.0):
        reasons.append("inactive_negative_ion_density_negative")
    if np.isfinite(occupancy) and occupancy > 1.0 + 1.0e-8:
        reasons.append("ion_site_occupancy_exceeds_limit")
    for label, value, limit in (
        (
            "carrier_area_rate",
            max(electron_rate, hole_rate),
            protocol.max_carrier_area_rate_A_m2,
        ),
        ("ion_area_rate", ion_rate, protocol.max_ion_area_rate_A_m2),
        (
            "ionic_face_current",
            maximum_ionic_face_current,
            protocol.max_ionic_face_current_A_m2,
        ),
        (
            "dc_face_current_spread",
            face_spread,
            protocol.max_dc_face_current_spread_A_m2,
        ),
        (
            "ion_inventory_relative_drift",
            inventory_drift,
            protocol.max_ion_inventory_relative_drift,
        ),
    ):
        if not np.isfinite(value):
            reasons.append(f"{label}_nonfinite")
        elif value > limit:
            reasons.append(f"{label}_exceeds_limit")
    numerical_reasons = tuple(dict.fromkeys(reasons))
    numerical = not numerical_reasons
    all_reasons = list(numerical_reasons)
    if not contact.certified:
        all_reasons.append(f"contact_thermodynamics_{contact.status}")
    return IonAwareDCStateCertificate(
        certified=numerical and contact.certified,
        numerically_certified=numerical,
        thermodynamically_certified=contact.certified,
        carrier_area_rate_A_m2=max(electron_rate, hole_rate),
        electron_area_rate_A_m2=electron_rate,
        hole_area_rate_A_m2=hole_rate,
        ion_area_rate_A_m2=ion_rate,
        positive_ion_area_rate_A_m2=positive_ion_rate,
        negative_ion_area_rate_A_m2=negative_ion_rate,
        max_ionic_face_current_A_m2=maximum_ionic_face_current,
        max_positive_ionic_face_current_A_m2=positive_face_current,
        max_negative_ionic_face_current_A_m2=negative_face_current,
        dc_face_current_spread_A_m2=face_spread,
        dc_current_density_A_m2=dc_current,
        minimum_electron_density_m3=minimum_n,
        minimum_hole_density_m3=minimum_p,
        minimum_positive_ion_density_m3=minimum_positive,
        minimum_negative_ion_density_m3=minimum_negative,
        maximum_site_occupancy_fraction=occupancy,
        positive_ion_inventory=positive_inventory,
        negative_ion_inventory=negative_inventory,
        max_ion_inventory_relative_drift=inventory_drift,
        contact_thermodynamics=contact,
        numerical_reasons=numerical_reasons,
        reasons=tuple(all_reasons),
    )


def solve_ion_aware_dc(
    x: np.ndarray,
    stack: DeviceStack,
    protocol: IonAwareDCProtocol,
    *,
    y0: np.ndarray | None = None,
    mat: MaterialArrays | None = None,
    rtol: float = 1.0e-4,
    atol: AbsoluteTolerance | None = None,
    method_ladder: tuple[str, ...] = ("Radau", "BDF"),
    max_nfev_per_attempt: int | None = 20_000,
    require_numerical_certificate: bool = True,
    require_contact_certificate: bool = False,
    progress: ProgressCallback | None = None,
) -> IonAwareDCResult:
    """Integrate a declared ladder and return only explicitly graded evidence."""
    if not isinstance(protocol, IonAwareDCProtocol):
        raise TypeError("protocol must be an IonAwareDCProtocol")
    rtol_value = _positive(rtol, "rtol")
    try:
        methods = tuple(method_ladder)
    except TypeError as exc:
        raise TypeError("method_ladder must be an iterable of solver names") from exc
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(not isinstance(item, str) or not item.strip() for item in methods)
    ):
        raise ValueError("method_ladder must contain unique non-empty solver names")
    if max_nfev_per_attempt is not None and (
        isinstance(max_nfev_per_attempt, (bool, np.bool_))
        or not isinstance(max_nfev_per_attempt, Integral)
        or max_nfev_per_attempt < 1
    ):
        raise ValueError("max_nfev_per_attempt must be a positive integer or None")
    material = build_material_arrays(x, stack) if mat is None else mat
    if not np.isclose(
        float(material.T_device), protocol.temperature_K, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "protocol temperature_K does not match the solver's effective temperature"
        )
    if material.N_iface_state:
        raise IonAwareDCCapabilityError(
            "ion-aware DC v1 excludes dynamic interface-state blocks"
        )
    has_positive_mobile_ions = bool(
        np.any(
            (np.asarray(material.P_ion0) > 0.0)
            & (np.asarray(material.D_ion_node) > 0.0)
        )
    )
    has_negative_mobile_ions = bool(
        material.has_dual_ions
        and material.P_ion0_neg is not None
        and material.D_ion_neg_node is not None
        and np.any(
            (np.asarray(material.P_ion0_neg) > 0.0)
            & (np.asarray(material.D_ion_neg_node) > 0.0)
        )
    )
    if not (has_positive_mobile_ions or has_negative_mobile_ions):
        raise IonAwareDCCapabilityError(
            "ion-aware DC requires at least one active mobile-ion species"
        )
    if y0 is None:
        if protocol.initial_state_source != "dark_quasineutral_equilibrium":
            raise ValueError("user_supplied_state protocol requires y0")
        initial_state = solve_equilibrium(np.asarray(x, dtype=float), stack)
    else:
        if protocol.initial_state_source != "user_supplied_state":
            raise ValueError(
                "supplying y0 requires initial_state_source='user_supplied_state'"
            )
        initial_state = np.asarray(y0, dtype=float).copy()
        actual_initial_hash = ion_aware_dc_state_sha256(initial_state)
        if actual_initial_hash != protocol.initial_state_sha256:
            raise ValueError(
                "user-supplied initial state does not match protocol "
                "initial_state_sha256"
            )
    _validate_initial_state(initial_state, len(x), material, protocol)

    tolerance = ComponentwiseAtol() if atol is None else atol
    diagnostic_policy = NumericalDiagnosticsPolicy(
        terminal_density_floor_m3=protocol.terminal_density_floor_m3,
        bulk_srh_denominator_floor_s_m3=0.0,
        interface_srh_denominator_floor_s_m4=0.0,
    )
    state = initial_state.copy()
    steps: list[IonAwareDCStep] = []
    consecutive = 0
    previous_target = 0.0
    total_steps = len(protocol.settle_end_times_s)
    for index, target in enumerate(protocol.settle_end_times_s, start=1):
        duration = target - previous_target
        if progress is not None:
            progress(
                "ion_aware_dc",
                index - 1,
                total_steps,
                f"settling fixed-bias DC state to t={target:.6g} s",
            )
        attempts: list[IonAwareDCAttempt] = []
        solution = None
        for method in methods:
            candidate = run_transient(
                np.asarray(x, dtype=float),
                state,
                (0.0, duration),
                np.asarray([duration]),
                stack,
                illuminated=protocol.illuminated,
                V_app=protocol.V_dc,
                rtol=rtol_value,
                atol=tolerance,
                mat=material,
                max_nfev=max_nfev_per_attempt,
                method=method,
                numerical_diagnostics=diagnostic_policy,
            )
            candidate_report = getattr(candidate, "numerical_diagnostics", None)
            attempts.append(
                IonAwareDCAttempt(
                    method=method,
                    success=bool(getattr(candidate, "success", False)),
                    message=str(getattr(candidate, "message", "")),
                    numerical_diagnostics=(
                        candidate_report
                        if isinstance(candidate_report, NumericalDiagnosticsReport)
                        else None
                    ),
                    nfev=_optional_count(candidate, "nfev"),
                    njev=_optional_count(candidate, "njev"),
                    nlu=_optional_count(candidate, "nlu"),
                )
            )
            if bool(getattr(candidate, "success", False)):
                solution = candidate
                break
        if solution is None:
            detail = "; ".join(
                f"{attempt.method}: {attempt.message or 'failed'}"
                for attempt in attempts
            )
            raise IonAwareDCSolverError(
                f"ion-aware DC segment ending at {target:.6g} s exhausted "
                f"the method ladder: {detail}",
                target_time_s=target,
                attempts=tuple(attempts),
            )
        values = np.asarray(getattr(solution, "y", np.empty((0, 0))), dtype=float)
        if values.ndim != 2 or values.shape[1] != 1:
            raise IonAwareDCSolverError(
                "ion-aware DC solver returned a malformed terminal state",
                target_time_s=target,
                attempts=tuple(attempts),
            )
        state = values[:, -1]
        report = getattr(solution, "numerical_diagnostics", None)
        if not isinstance(report, NumericalDiagnosticsReport):
            raise IonAwareDCSolverError(
                "ion-aware DC solver omitted numerical diagnostics",
                target_time_s=target,
                attempts=tuple(attempts),
            )
        certificate = assess_ion_aware_dc_state(
            np.asarray(x, dtype=float),
            state,
            initial_state,
            stack,
            protocol,
            mat=material,
        )
        diagnostics_passed = bool(report.would_pass_strict)
        accepted = bool(certificate.numerically_certified and diagnostics_passed)
        consecutive = consecutive + 1 if accepted else 0
        steps.append(
            IonAwareDCStep(
                target_time_s=target,
                segment_duration_s=duration,
                state_certificate=certificate,
                numerical_diagnostics=report,
                independent_state_passed=certificate.numerically_certified,
                diagnostics_passed=diagnostics_passed,
                accepted_for_closure=accepted,
                accepted_method=attempts[-1].method,
                attempts=tuple(attempts),
                nfev=(
                    sum(item.nfev for item in attempts if item.nfev is not None)
                    or None
                ),
                njev=(
                    sum(item.njev for item in attempts if item.njev is not None)
                    or None
                ),
                nlu=(
                    sum(item.nlu for item in attempts if item.nlu is not None)
                    or None
                ),
            )
        )
        previous_target = target
        if consecutive >= protocol.required_consecutive_passes:
            break

    final_certificate = steps[-1].state_certificate
    numerical = consecutive >= protocol.required_consecutive_passes
    result = IonAwareDCResult(
        x=np.asarray(x, dtype=float).copy(),
        y=state.copy(),
        protocol=protocol,
        steps=tuple(steps),
        state_certificate=final_certificate,
        consecutive_certified_steps=consecutive,
        numerically_certified=numerical,
        thermodynamically_certified=final_certificate.thermodynamically_certified,
        certified=numerical and final_certificate.thermodynamically_certified,
        total_settle_time_s=steps[-1].target_time_s,
    )
    if progress is not None:
        progress(
            "ion_aware_dc",
            len(steps),
            total_steps,
            (
                "DC numerical certificate passed"
                if numerical
                else "DC ladder exhausted without a numerical certificate"
            ),
        )
    if require_numerical_certificate and not result.numerically_certified:
        raise IonAwareDCCertificationError(
            "ion-aware DC ladder exhausted without consecutive numerical passes",
            result,
        )
    if require_contact_certificate and not result.thermodynamically_certified:
        raise IonAwareDCCertificationError(
            "ion-aware DC contact thermodynamics is not certified; "
            f"status={final_certificate.contact_thermodynamics.status}",
            result,
        )
    return result


__all__ = [
    "DEFAULT_SETTLE_END_TIMES_S",
    "ION_AWARE_DC_PROTOCOL_SCHEMA",
    "ION_AWARE_DC_STATE_HASH_SCHEMA",
    "IonAwareDCCapabilityError",
    "IonAwareDCCertificationError",
    "IonAwareDCAttempt",
    "IonAwareDCProtocol",
    "IonAwareDCResult",
    "IonAwareDCSolverError",
    "IonAwareDCStateCertificate",
    "IonAwareDCStep",
    "IonInventoryEvidence",
    "assess_ion_aware_dc_state",
    "build_ion_aware_dc_protocol",
    "ion_aware_dc_state_sha256",
    "solve_ion_aware_dc",
]
