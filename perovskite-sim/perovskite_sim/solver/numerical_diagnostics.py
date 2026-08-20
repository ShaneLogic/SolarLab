"""Observational numerical-health diagnostics for method-of-lines solves.

The default policy records evidence without modifying states or solver success.
The research-strict policy is an explicit opt-in gate: it rejects non-finite
RHS values, caller-defined near-zero SRH denominators, and non-positive or
non-finite terminal densities. Negative intermediate trial states are counted
but are not clipped and do not fail a solve by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from perovskite_sim.physics.generation import dual_cell_integral


DiagnosticMode = Literal["observe", "research_strict"]
DensityBlock = Literal["n", "p", "P", "P_neg", "interface_state"]
StateCoordinateMode = Literal["density", "research_log_density"]


@dataclass(frozen=True)
class NumericalDiagnosticsPolicy:
    """Policy controlling observational versus fail-closed diagnostics.

    SRH denominator floors are deliberately explicit and carry units because a
    universal dimensionless threshold would not be physically meaningful.
    Bulk and surface SRH denominators have different units and therefore have
    separate thresholds.
    """

    mode: DiagnosticMode = "observe"
    terminal_density_floor_m3: float = 0.0
    bulk_srh_denominator_floor_s_m3: float = 0.0
    interface_srh_denominator_floor_s_m4: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("observe", "research_strict"):
            raise ValueError("mode must be 'observe' or 'research_strict'")
        for name in (
            "terminal_density_floor_m3",
            "bulk_srh_denominator_floor_s_m3",
            "interface_srh_denominator_floor_s_m4",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be finite and non-negative")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                ) from exc
            if not np.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, normalized)

    @property
    def strict(self) -> bool:
        return self.mode == "research_strict"

    @classmethod
    def research_strict(
        cls,
        *,
        terminal_density_floor_m3: float = 0.0,
        bulk_srh_denominator_floor_s_m3: float,
        interface_srh_denominator_floor_s_m4: float = 0.0,
    ) -> "NumericalDiagnosticsPolicy":
        """Build an explicit fail-closed policy for research/certification runs."""

        return cls(
            mode="research_strict",
            terminal_density_floor_m3=terminal_density_floor_m3,
            bulk_srh_denominator_floor_s_m3=(
                bulk_srh_denominator_floor_s_m3
            ),
            interface_srh_denominator_floor_s_m4=(
                interface_srh_denominator_floor_s_m4
            ),
        )


@dataclass(frozen=True)
class StateLayout:
    """Packed 1D state layout, including structurally active ion nodes."""

    n_nodes: int
    has_dual_ions: bool = False
    n_interface_states: int = 0
    positive_ion_active: tuple[bool, ...] = ()
    negative_ion_active: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.n_nodes, bool) or int(self.n_nodes) != self.n_nodes:
            raise ValueError("n_nodes must be a positive integer")
        if int(self.n_nodes) <= 0:
            raise ValueError("n_nodes must be a positive integer")
        if (
            isinstance(self.n_interface_states, bool)
            or int(self.n_interface_states) != self.n_interface_states
            or int(self.n_interface_states) < 0
        ):
            raise ValueError("n_interface_states must be a non-negative integer")
        object.__setattr__(self, "n_nodes", int(self.n_nodes))
        object.__setattr__(
            self, "n_interface_states", int(self.n_interface_states)
        )

        positive = self._normalize_mask(
            self.positive_ion_active, "positive_ion_active", default=True
        )
        negative = self._normalize_mask(
            self.negative_ion_active,
            "negative_ion_active",
            default=bool(self.has_dual_ions),
        )
        if not self.has_dual_ions and any(negative):
            raise ValueError(
                "negative_ion_active cannot select nodes in single-ion mode"
            )
        object.__setattr__(self, "positive_ion_active", positive)
        object.__setattr__(self, "negative_ion_active", negative)

    def _normalize_mask(
        self, values: tuple[bool, ...], name: str, *, default: bool
    ) -> tuple[bool, ...]:
        if not values:
            return (default,) * self.n_nodes
        if len(values) != self.n_nodes:
            raise ValueError(f"{name} must have n_nodes entries")
        return tuple(bool(value) for value in values)

    @property
    def expected_size(self) -> int:
        ion_blocks = 2 if self.has_dual_ions else 1
        return (
            (2 + ion_blocks) * self.n_nodes
            + 4 * self.n_interface_states
        )

    def split(self, state: np.ndarray) -> dict[DensityBlock, np.ndarray]:
        array = np.asarray(state)
        if array.ndim != 1 or array.size != self.expected_size:
            raise ValueError(
                f"state has shape {array.shape}; expected one-dimensional "
                f"layout of length {self.expected_size}"
            )
        n = self.n_nodes
        blocks: dict[DensityBlock, np.ndarray] = {
            "n": array[:n],
            "p": array[n : 2 * n],
            "P": array[2 * n : 3 * n],
        }
        cursor = 3 * n
        if self.has_dual_ions:
            blocks["P_neg"] = array[cursor : cursor + n]
            cursor += n
        if self.n_interface_states:
            blocks["interface_state"] = array[cursor:]
        return blocks

    @property
    def active_density_mask(self) -> np.ndarray:
        """Mask of physical densities represented in logarithmic coordinates."""

        blocks = [
            np.ones(self.n_nodes, dtype=bool),
            np.ones(self.n_nodes, dtype=bool),
            np.asarray(self.positive_ion_active, dtype=bool),
        ]
        if self.has_dual_ions:
            blocks.append(np.asarray(self.negative_ion_active, dtype=bool))
        if self.n_interface_states:
            blocks.append(
                np.ones(4 * self.n_interface_states, dtype=bool)
            )
        return np.concatenate(blocks)


@dataclass(frozen=True)
class LogDensityCoordinateReport:
    """Immutable description of one research log-coordinate mapping."""

    mode: Literal["research_log_density"]
    active_density_components: int
    inactive_structural_ion_components: int
    reference_density_min_m3: float
    reference_density_max_m3: float
    physical_rtol: float
    coordinate_rtol: float
    physical_atol_min_m3: float
    physical_atol_max_m3: float
    active_coordinate_atol_min: float
    active_coordinate_atol_max: float
    inactive_coordinate_atol_min_m3: float | None
    inactive_coordinate_atol_max_m3: float | None
    atol_mapping: str


class LogDensityCoordinateError(ValueError):
    """The opt-in log-coordinate contract cannot represent a physical state."""


class LogDensityCoordinateTransform:
    """Hybrid log/linear coordinates for one packed transient state.

    Every physically active density uses ``y_i = s_i * exp(z_i)`` with
    ``s_i`` equal to its positive initial density. Structurally inactive ion
    nodes remain linear, so their exact physical zero is representable.
    """

    _COORDINATE_RTOL = 100.0 * np.finfo(float).eps
    _LOG_MIN_POSITIVE = float(np.log(np.nextafter(0.0, 1.0)))
    _LOG_MAX_FINITE = float(np.log(np.finfo(float).max))

    def __init__(
        self,
        layout: StateLayout,
        initial_physical_state: np.ndarray,
        physical_atol: float | np.ndarray,
        physical_rtol: float,
    ) -> None:
        self.layout = layout
        initial = self._state_vector(
            initial_physical_state, "initial physical state"
        )
        active = layout.active_density_mask
        if active.size != initial.size:
            raise LogDensityCoordinateError(
                "active density mask does not match the packed state"
            )
        if not np.all(np.isfinite(initial)):
            raise LogDensityCoordinateError(
                "research_log_density requires a finite initial state"
            )
        if np.any(initial[active] <= 0.0):
            raise LogDensityCoordinateError(
                "research_log_density requires every active initial density "
                "to be strictly positive"
            )
        if np.any(initial[~active] < 0.0):
            raise LogDensityCoordinateError(
                "inactive structural ion densities must be non-negative; "
                "exact zeros are allowed"
            )

        if isinstance(physical_rtol, (bool, np.bool_)):
            raise LogDensityCoordinateError(
                "physical rtol must be finite and positive"
            )
        try:
            rtol = float(physical_rtol)
        except (TypeError, ValueError) as exc:
            raise LogDensityCoordinateError(
                "physical rtol must be finite and positive"
            ) from exc
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise LogDensityCoordinateError(
                "physical rtol must be finite and positive"
            )

        atol = np.asarray(physical_atol, dtype=float)
        if atol.ndim == 0:
            atol = np.full(initial.size, float(atol), dtype=float)
        if atol.ndim != 1 or atol.size != initial.size:
            raise LogDensityCoordinateError(
                "physical atol must be scalar or match the packed state"
            )
        if not np.all(np.isfinite(atol)) or np.any(atol <= 0.0):
            raise LogDensityCoordinateError(
                "physical atol entries must be finite and positive"
            )

        self.active_mask = active
        self.initial_physical_state = initial.copy()
        self.reference_scale = np.ones_like(initial)
        self.reference_scale[active] = initial[active]
        self.physical_atol = atol.copy()
        self.physical_rtol = rtol
        self.coordinate_rtol = self._COORDINATE_RTOL
        self.coordinate_atol = np.empty_like(initial)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            relative_budget = rtol + atol[active] / initial[active]
            self.coordinate_atol[active] = np.log1p(relative_budget)
        self.coordinate_atol[~active] = (
            atol[~active] + rtol * np.abs(initial[~active])
        )
        if (
            not np.all(np.isfinite(self.coordinate_atol))
            or np.any(self.coordinate_atol <= 0.0)
        ):
            raise LogDensityCoordinateError(
                "physical tolerances cannot be represented in log coordinates"
            )

    def initial_coordinates(self) -> np.ndarray:
        coordinates = self.initial_physical_state.copy()
        coordinates[self.active_mask] = 0.0
        return coordinates

    def to_physical(self, coordinates: np.ndarray) -> np.ndarray:
        values = self._state_vector(coordinates, "log-coordinate trial state")
        if not np.all(np.isfinite(values)):
            raise LogDensityCoordinateError(
                "log-coordinate trial state contains non-finite values"
            )
        physical = values.copy()
        active_values = values[self.active_mask]
        scales = self.reference_scale[self.active_mask]
        log_physical = np.log(scales) + active_values
        if np.any(log_physical < self._LOG_MIN_POSITIVE):
            raise LogDensityCoordinateError(
                "log-coordinate trial underflows a strictly positive density"
            )
        if np.any(log_physical > self._LOG_MAX_FINITE):
            raise LogDensityCoordinateError(
                "log-coordinate trial overflows a finite density"
            )

        direct = (
            (active_values >= self._LOG_MIN_POSITIVE)
            & (active_values <= self._LOG_MAX_FINITE)
        )
        mapped = np.empty_like(active_values)
        mapped[direct] = scales[direct] * np.exp(active_values[direct])
        mapped[~direct] = np.exp(log_physical[~direct])
        if not np.all(np.isfinite(mapped)) or np.any(mapped <= 0.0):
            raise LogDensityCoordinateError(
                "log-coordinate trial did not map to finite positive densities"
            )
        physical[self.active_mask] = mapped
        return physical

    def rhs_to_coordinates(
        self, physical_rhs: np.ndarray, physical_state: np.ndarray
    ) -> np.ndarray:
        rhs = self._state_vector(physical_rhs, "physical RHS")
        state = self._state_vector(physical_state, "physical state")
        if not np.all(np.isfinite(rhs)):
            raise LogDensityCoordinateError(
                "research_log_density cannot transform a non-finite physical RHS"
            )
        transformed = rhs.copy()
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            transformed[self.active_mask] = (
                rhs[self.active_mask] / state[self.active_mask]
            )
        if not np.all(np.isfinite(transformed)):
            raise LogDensityCoordinateError(
                "physical RHS overflowed in log-coordinate form"
            )
        return transformed

    def solution_to_physical(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.layout.expected_size:
            raise LogDensityCoordinateError(
                "solver output does not match the packed log-coordinate state"
            )
        if values.shape[1] == 0:
            return values.copy()
        return np.column_stack(
            [self.to_physical(values[:, index]) for index in range(values.shape[1])]
        )

    def report(self) -> LogDensityCoordinateReport:
        active_reference = self.reference_scale[self.active_mask]
        active_coordinate_atol = self.coordinate_atol[self.active_mask]
        inactive_coordinate_atol = self.coordinate_atol[~self.active_mask]
        return LogDensityCoordinateReport(
            mode="research_log_density",
            active_density_components=int(np.count_nonzero(self.active_mask)),
            inactive_structural_ion_components=int(
                np.count_nonzero(~self.active_mask)
            ),
            reference_density_min_m3=float(np.min(active_reference)),
            reference_density_max_m3=float(np.max(active_reference)),
            physical_rtol=self.physical_rtol,
            coordinate_rtol=self.coordinate_rtol,
            physical_atol_min_m3=float(np.min(self.physical_atol)),
            physical_atol_max_m3=float(np.max(self.physical_atol)),
            active_coordinate_atol_min=float(
                np.min(active_coordinate_atol)
            ),
            active_coordinate_atol_max=float(
                np.max(active_coordinate_atol)
            ),
            inactive_coordinate_atol_min_m3=(
                None
                if inactive_coordinate_atol.size == 0
                else float(np.min(inactive_coordinate_atol))
            ),
            inactive_coordinate_atol_max_m3=(
                None
                if inactive_coordinate_atol.size == 0
                else float(np.max(inactive_coordinate_atol))
            ),
            atol_mapping=(
                "active dz=log1p(physical_rtol + physical_atol/reference); "
                "inactive dy=physical_atol + physical_rtol*abs(initial)"
            ),
        )

    def _state_vector(self, values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size != self.layout.expected_size:
            raise LogDensityCoordinateError(
                f"{name} must be one-dimensional with length "
                f"{self.layout.expected_size}"
            )
        return array


@dataclass(frozen=True)
class DensityMinima:
    """Minimum finite density in each physical state block [m^-3]."""

    n: float | None
    p: float | None
    positive_ion_active: float | None
    negative_ion_active: float | None
    interface_state: float | None


@dataclass(frozen=True)
class NegativeEntryCounts:
    n: int = 0
    p: int = 0
    positive_ion: int = 0
    negative_ion: int = 0
    interface_state: int = 0


@dataclass(frozen=True)
class NumericalDiagnosticsReport:
    """Immutable, serializable summary attached to each transient result."""

    mode: DiagnosticMode
    solver_success: bool | None
    trial_evaluations: int
    negative_trial_evaluations: int
    negative_trial_entries: NegativeEntryCounts
    nonfinite_trial_evaluations: int
    nonfinite_rhs_evaluations: int
    minimum_trial_density_m3: DensityMinima
    final_minimum_density_m3: DensityMinima | None
    minimum_bulk_srh_denominator_s_m3: float | None
    minimum_interface_srh_denominator_s_m4: float | None
    terminal_density_floor_m3: float
    bulk_srh_denominator_floor_s_m3: float
    interface_srh_denominator_floor_s_m4: float
    violations: tuple[str, ...]
    would_pass_strict: bool


class NumericalDiagnosticsError(RuntimeError):
    """A research-strict numerical-health gate rejected a solve."""

    def __init__(
        self, message: str, report: NumericalDiagnosticsReport
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SplitStepDiagnosticsPolicy:
    """Opt-in evidence and fail-closed policy for the ion split step.

    The legacy split step projects negative ion values inside its implicit
    RHS and clips the accepted ion terminal state to the finite-site bounds.
    This policy never changes those equations.  Instead it makes the raw
    states and the resulting inventory change observable.  Research-strict
    mode always rejects non-finite, clipped initial/terminal, failed-solver,
    and excessive-inventory-drift outcomes before a projected state can be
    accepted.

    Negative or over-limit *implicit trial* states are evidence of nonlinear
    solver exploration rather than accepted physical states.  They are always
    counted.  Rejection is separately configurable and defaults to ``False``
    so the strict contract matches the transient diagnostics policy.
    """

    mode: DiagnosticMode = "observe"
    maximum_relative_inventory_drift: float = 1.0e-8
    reject_negative_trial_states: bool = False
    reject_overlimit_trial_states: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("observe", "research_strict"):
            raise ValueError("mode must be 'observe' or 'research_strict'")
        value = self.maximum_relative_inventory_drift
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "maximum_relative_inventory_drift must be finite and non-negative"
            )
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "maximum_relative_inventory_drift must be finite and non-negative"
            ) from exc
        if not np.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                "maximum_relative_inventory_drift must be finite and non-negative"
            )
        object.__setattr__(
            self, "maximum_relative_inventory_drift", normalized
        )
        for name in (
            "reject_negative_trial_states",
            "reject_overlimit_trial_states",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))

    @property
    def strict(self) -> bool:
        return self.mode == "research_strict"

    @classmethod
    def research_strict(
        cls,
        *,
        maximum_relative_inventory_drift: float = 1.0e-8,
        reject_negative_trial_states: bool = False,
        reject_overlimit_trial_states: bool = False,
    ) -> "SplitStepDiagnosticsPolicy":
        """Build an explicit fail-closed split-step certification policy."""

        return cls(
            mode="research_strict",
            maximum_relative_inventory_drift=(
                maximum_relative_inventory_drift
            ),
            reject_negative_trial_states=reject_negative_trial_states,
            reject_overlimit_trial_states=reject_overlimit_trial_states,
        )


@dataclass(frozen=True)
class IonInventoryDiagnostics:
    """Ion inventory evidence for one species [m^-2]."""

    initial_m2: float | None
    raw_terminal_m2: float | None
    projected_terminal_m2: float | None
    final_m2: float | None
    raw_terminal_relative_drift: float | None
    projected_terminal_relative_drift: float | None
    final_relative_drift: float | None


@dataclass(frozen=True)
class SplitStepIonDiagnostics:
    """Raw bound, projection, and inventory evidence for one ion species."""

    initial_negative_entries: int
    initial_overlimit_entries: int
    initial_nonfinite_entries: int
    initial_projection_entries: int
    negative_trial_entries: int
    overlimit_trial_entries: int
    nonfinite_trial_entries: int
    raw_terminal_negative_entries: int
    raw_terminal_overlimit_entries: int
    raw_terminal_nonfinite_entries: int
    terminal_projection_entries: int
    final_negative_entries: int
    final_overlimit_entries: int
    final_nonfinite_entries: int
    minimum_raw_trial_density_m3: float | None
    maximum_raw_trial_density_m3: float | None
    raw_terminal_minimum_density_m3: float | None
    raw_terminal_maximum_density_m3: float | None
    inventory: IonInventoryDiagnostics


@dataclass(frozen=True)
class SplitStepDiagnosticsReport:
    """Immutable report for one explicitly monitored split step."""

    mode: DiagnosticMode
    dual_ion: bool
    ion_solver_success: bool | None
    carrier_reequilibration_success: bool | None
    trial_evaluations: int
    negative_trial_evaluations: int
    overlimit_trial_evaluations: int
    nonfinite_trial_evaluations: int
    projection_events: int
    initial_state_nonfinite: bool
    final_state_nonfinite: bool | None
    final_electron_minimum_density_m3: float | None
    final_hole_minimum_density_m3: float | None
    final_interface_state_minimum_density_m3: float | None
    final_electron_nonpositive_entries: int
    final_hole_nonpositive_entries: int
    final_interface_state_nonpositive_entries: int
    maximum_relative_inventory_drift: float
    reject_negative_trial_states: bool
    reject_overlimit_trial_states: bool
    positive_ion: SplitStepIonDiagnostics
    negative_ion: SplitStepIonDiagnostics | None
    violations: tuple[str, ...]
    would_pass_strict: bool


class SplitStepDiagnosticsError(RuntimeError):
    """A research-strict split-step gate rejected raw solver evidence."""

    def __init__(
        self, message: str, report: SplitStepDiagnosticsReport
    ) -> None:
        super().__init__(message)
        self.report = report


class SplitStepDiagnosticsMonitor:
    """Collect split-step evidence without modifying the numerical path."""

    _SPECIES = ("positive_ion", "negative_ion")

    def __init__(
        self,
        x: np.ndarray,
        positive_initial: np.ndarray,
        positive_limit: np.ndarray,
        *,
        full_initial_state: np.ndarray,
        negative_initial: np.ndarray | None = None,
        negative_limit: np.ndarray | None = None,
        policy: SplitStepDiagnosticsPolicy | None = None,
    ) -> None:
        self.policy = policy or SplitStepDiagnosticsPolicy()
        self.x = self._one_dimensional("x", x)
        if self.x.size < 2 or not np.all(np.isfinite(self.x)):
            raise ValueError("x must contain at least two finite points")
        if np.any(np.diff(self.x) <= 0.0):
            raise ValueError("x must be strictly increasing")

        positive = self._one_dimensional(
            "positive_initial", positive_initial, size=self.x.size
        )
        positive_cap = self._validated_limit(
            "positive_limit", positive_limit
        )
        if (negative_initial is None) != (negative_limit is None):
            raise ValueError(
                "negative_initial and negative_limit must be provided together"
            )
        negative = (
            None
            if negative_initial is None
            else self._one_dimensional(
                "negative_initial", negative_initial, size=self.x.size
            )
        )
        negative_cap = (
            None
            if negative_limit is None
            else self._validated_limit("negative_limit", negative_limit)
        )

        self.dual_ion = negative is not None
        self._limits = {
            "positive_ion": positive_cap,
            "negative_ion": negative_cap,
        }
        self._initial = {
            "positive_ion": positive.copy(),
            "negative_ion": None if negative is None else negative.copy(),
        }
        self.initial_state_nonfinite = not bool(
            np.all(np.isfinite(np.asarray(full_initial_state)))
        )
        self.final_state_nonfinite: bool | None = None
        self.final_electron_minimum_density_m3: float | None = None
        self.final_hole_minimum_density_m3: float | None = None
        self.final_interface_state_minimum_density_m3: float | None = None
        self.final_electron_nonpositive_entries = 0
        self.final_hole_nonpositive_entries = 0
        self.final_interface_state_nonpositive_entries = 0
        self.ion_solver_success: bool | None = None
        self.carrier_reequilibration_success: bool | None = None
        self.trial_evaluations = 0
        self.negative_trial_evaluations = 0
        self.overlimit_trial_evaluations = 0
        self.nonfinite_trial_evaluations = 0
        self.projection_events = 0
        self._raw_terminal = {name: None for name in self._SPECIES}
        self._projected_terminal = {name: None for name in self._SPECIES}
        self._final = {name: None for name in self._SPECIES}
        self._stats = {
            name: {
                "negative_trial_entries": 0,
                "overlimit_trial_entries": 0,
                "nonfinite_trial_entries": 0,
                "minimum_raw_trial_density_m3": None,
                "maximum_raw_trial_density_m3": None,
            }
            for name in self._SPECIES
        }

        self._record_initial_projection_events()
        self._raise_if_strict_rejected("initial ion state")

    def observe_trial(
        self,
        positive: np.ndarray,
        negative: np.ndarray | None = None,
    ) -> None:
        values = self._species_values(positive, negative)
        self.trial_evaluations += 1
        has_negative = False
        has_overlimit = False
        has_nonfinite = False
        for name, array in values.items():
            if array is None:
                continue
            cap = self._limits[name]
            assert cap is not None
            finite = np.isfinite(array)
            negative_count = int(np.count_nonzero(finite & (array < 0.0)))
            overlimit_count = int(np.count_nonzero(finite & (array > cap)))
            nonfinite_count = int(array.size - np.count_nonzero(finite))
            stats = self._stats[name]
            stats["negative_trial_entries"] += negative_count
            stats["overlimit_trial_entries"] += overlimit_count
            stats["nonfinite_trial_entries"] += nonfinite_count
            has_negative = has_negative or negative_count > 0
            has_overlimit = has_overlimit or overlimit_count > 0
            has_nonfinite = has_nonfinite or nonfinite_count > 0
            if np.any(finite):
                minimum = float(np.min(array[finite]))
                maximum = float(np.max(array[finite]))
                stats["minimum_raw_trial_density_m3"] = self._minimum(
                    stats["minimum_raw_trial_density_m3"], minimum
                )
                stats["maximum_raw_trial_density_m3"] = self._maximum(
                    stats["maximum_raw_trial_density_m3"], maximum
                )
        self.negative_trial_evaluations += int(has_negative)
        self.overlimit_trial_evaluations += int(has_overlimit)
        self.nonfinite_trial_evaluations += int(has_nonfinite)
        self._raise_if_strict_rejected("implicit ion trial state")

    def observe_raw_terminal(
        self,
        positive: np.ndarray | None,
        negative: np.ndarray | None = None,
        *,
        solver_success: bool,
    ) -> None:
        self.ion_solver_success = bool(solver_success)
        if positive is not None:
            values = self._species_values(positive, negative)
            for name, array in values.items():
                self._raw_terminal[name] = (
                    None if array is None else array.copy()
                )
            projection_entries = sum(
                self._bound_counts(name, array)[0]
                + self._bound_counts(name, array)[1]
                for name, array in values.items()
                if array is not None
            )
            if projection_entries:
                self.projection_events += 1
        self._raise_if_strict_rejected("raw ion terminal state")

    def observe_projected_terminal(
        self,
        positive: np.ndarray,
        negative: np.ndarray | None = None,
    ) -> None:
        values = self._species_values(positive, negative)
        for name, array in values.items():
            self._projected_terminal[name] = (
                None if array is None else array.copy()
            )

    def finalize(
        self,
        positive: np.ndarray,
        negative: np.ndarray | None,
        *,
        full_final_state: np.ndarray,
        carrier_reequilibration_success: bool | None,
    ) -> SplitStepDiagnosticsReport:
        values = self._species_values(positive, negative)
        for name, array in values.items():
            self._final[name] = None if array is None else array.copy()
        state = self._one_dimensional("full_final_state", full_final_state)
        base_size = (4 if self.dual_ion else 3) * self.x.size
        if state.size < base_size or (state.size - base_size) % 4:
            raise ValueError(
                "full_final_state has an invalid carrier/ion/interface layout"
            )
        self.final_state_nonfinite = not bool(np.all(np.isfinite(state)))
        electrons = state[: self.x.size]
        holes = state[self.x.size : 2 * self.x.size]
        interface_state = state[base_size:]
        (
            self.final_electron_minimum_density_m3,
            self.final_electron_nonpositive_entries,
        ) = self._positive_state_health(electrons)
        (
            self.final_hole_minimum_density_m3,
            self.final_hole_nonpositive_entries,
        ) = self._positive_state_health(holes)
        (
            self.final_interface_state_minimum_density_m3,
            self.final_interface_state_nonpositive_entries,
        ) = self._positive_state_health(interface_state)
        self.carrier_reequilibration_success = (
            None
            if carrier_reequilibration_success is None
            else bool(carrier_reequilibration_success)
        )
        report = self.report()
        if self.policy.strict and report.violations:
            self._raise(report, "final split-step state")
        return report

    def report(self) -> SplitStepDiagnosticsReport:
        violations = self._violations()
        return SplitStepDiagnosticsReport(
            mode=self.policy.mode,
            dual_ion=self.dual_ion,
            ion_solver_success=self.ion_solver_success,
            carrier_reequilibration_success=(
                self.carrier_reequilibration_success
            ),
            trial_evaluations=self.trial_evaluations,
            negative_trial_evaluations=self.negative_trial_evaluations,
            overlimit_trial_evaluations=self.overlimit_trial_evaluations,
            nonfinite_trial_evaluations=self.nonfinite_trial_evaluations,
            projection_events=self.projection_events,
            initial_state_nonfinite=self.initial_state_nonfinite,
            final_state_nonfinite=self.final_state_nonfinite,
            final_electron_minimum_density_m3=(
                self.final_electron_minimum_density_m3
            ),
            final_hole_minimum_density_m3=self.final_hole_minimum_density_m3,
            final_interface_state_minimum_density_m3=(
                self.final_interface_state_minimum_density_m3
            ),
            final_electron_nonpositive_entries=(
                self.final_electron_nonpositive_entries
            ),
            final_hole_nonpositive_entries=self.final_hole_nonpositive_entries,
            final_interface_state_nonpositive_entries=(
                self.final_interface_state_nonpositive_entries
            ),
            maximum_relative_inventory_drift=(
                self.policy.maximum_relative_inventory_drift
            ),
            reject_negative_trial_states=(
                self.policy.reject_negative_trial_states
            ),
            reject_overlimit_trial_states=(
                self.policy.reject_overlimit_trial_states
            ),
            positive_ion=self._species_report("positive_ion"),
            negative_ion=(
                self._species_report("negative_ion")
                if self.dual_ion
                else None
            ),
            violations=violations,
            would_pass_strict=not violations,
        )

    def _species_report(self, name: str) -> SplitStepIonDiagnostics:
        initial = self._initial[name]
        limit = self._limits[name]
        assert initial is not None and limit is not None
        raw = self._raw_terminal[name]
        projected = self._projected_terminal[name]
        final = self._final[name]
        initial_negative, initial_over, initial_nonfinite = self._bound_counts(
            name, initial
        )
        raw_negative, raw_over, raw_nonfinite = self._bound_counts(name, raw)
        final_negative, final_over, final_nonfinite = self._bound_counts(
            name, final
        )
        stats = self._stats[name]
        initial_inventory = self._inventory(initial)
        raw_inventory = self._inventory(raw)
        projected_inventory = self._inventory(projected)
        final_inventory = self._inventory(final)
        return SplitStepIonDiagnostics(
            initial_negative_entries=initial_negative,
            initial_overlimit_entries=initial_over,
            initial_nonfinite_entries=initial_nonfinite,
            initial_projection_entries=initial_negative,
            negative_trial_entries=stats["negative_trial_entries"],
            overlimit_trial_entries=stats["overlimit_trial_entries"],
            nonfinite_trial_entries=stats["nonfinite_trial_entries"],
            raw_terminal_negative_entries=raw_negative,
            raw_terminal_overlimit_entries=raw_over,
            raw_terminal_nonfinite_entries=raw_nonfinite,
            terminal_projection_entries=raw_negative + raw_over,
            final_negative_entries=final_negative,
            final_overlimit_entries=final_over,
            final_nonfinite_entries=final_nonfinite,
            minimum_raw_trial_density_m3=(
                stats["minimum_raw_trial_density_m3"]
            ),
            maximum_raw_trial_density_m3=(
                stats["maximum_raw_trial_density_m3"]
            ),
            raw_terminal_minimum_density_m3=self._finite_extreme(raw, np.min),
            raw_terminal_maximum_density_m3=self._finite_extreme(raw, np.max),
            inventory=IonInventoryDiagnostics(
                initial_m2=initial_inventory,
                raw_terminal_m2=raw_inventory,
                projected_terminal_m2=projected_inventory,
                final_m2=final_inventory,
                raw_terminal_relative_drift=self._relative_drift(
                    initial_inventory, raw_inventory
                ),
                projected_terminal_relative_drift=self._relative_drift(
                    initial_inventory, projected_inventory
                ),
                final_relative_drift=self._relative_drift(
                    initial_inventory, final_inventory
                ),
            ),
        )

    def _violations(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.initial_state_nonfinite:
            reasons.append("initial_state_nonfinite")
        if self.ion_solver_success is False:
            reasons.append("ion_solver_not_successful")
        if self.nonfinite_trial_evaluations:
            reasons.append("nonfinite_ion_trial_state")
        if (
            self.policy.reject_negative_trial_states
            and self.negative_trial_evaluations
        ):
            reasons.append("negative_ion_trial_state")
        if (
            self.policy.reject_overlimit_trial_states
            and self.overlimit_trial_evaluations
        ):
            reasons.append("overlimit_ion_trial_state")
        if self.carrier_reequilibration_success is False:
            reasons.append("carrier_reequilibration_not_successful")
        if self.final_state_nonfinite:
            reasons.append("final_state_nonfinite")
        if self.final_electron_nonpositive_entries:
            reasons.append("final_electron_density_nonpositive")
        if self.final_hole_nonpositive_entries:
            reasons.append("final_hole_density_nonpositive")
        if self.final_interface_state_nonpositive_entries:
            reasons.append("final_interface_state_density_nonpositive")

        for name in self._SPECIES:
            if name == "negative_ion" and not self.dual_ion:
                continue
            report = self._species_report_without_violations(name)
            prefix = "positive_ion" if name == "positive_ion" else "negative_ion"
            if report.initial_nonfinite_entries:
                reasons.append(f"{prefix}_initial_nonfinite")
            if report.initial_negative_entries:
                reasons.append(f"{prefix}_initial_negative")
            if report.initial_overlimit_entries:
                reasons.append(f"{prefix}_initial_overlimit")
            if self.ion_solver_success and self._raw_terminal[name] is None:
                reasons.append(f"{prefix}_raw_terminal_missing")
            if report.raw_terminal_nonfinite_entries:
                reasons.append(f"{prefix}_raw_terminal_nonfinite")
            if report.raw_terminal_negative_entries:
                reasons.append(f"{prefix}_raw_terminal_negative")
            if report.raw_terminal_overlimit_entries:
                reasons.append(f"{prefix}_raw_terminal_overlimit")
            if report.final_nonfinite_entries:
                reasons.append(f"{prefix}_final_nonfinite")
            if report.final_negative_entries:
                reasons.append(f"{prefix}_final_negative")
            if report.final_overlimit_entries:
                reasons.append(f"{prefix}_final_overlimit")
            for stage, drift in (
                (
                    "raw_terminal",
                    report.inventory.raw_terminal_relative_drift,
                ),
                ("final", report.inventory.final_relative_drift),
            ):
                if (
                    drift is not None
                    and drift > self.policy.maximum_relative_inventory_drift
                ):
                    reasons.append(f"{prefix}_{stage}_inventory_drift")
        return tuple(dict.fromkeys(reasons))

    def _species_report_without_violations(
        self, name: str
    ) -> SplitStepIonDiagnostics:
        """Build a species report while avoiding report/violation recursion."""

        return self._species_report(name)

    def _record_initial_projection_events(self) -> None:
        entries = 0
        for name in self._SPECIES:
            values = self._initial[name]
            if values is None:
                continue
            finite = np.isfinite(values)
            entries += int(np.count_nonzero(finite & (values < 0.0)))
        if entries:
            self.projection_events += 1

    def _species_values(
        self,
        positive: np.ndarray,
        negative: np.ndarray | None,
    ) -> dict[str, np.ndarray | None]:
        positive_array = self._one_dimensional(
            "positive ion state", positive, size=self.x.size
        )
        if self.dual_ion:
            if negative is None:
                raise ValueError("negative ion state is required in dual-ion mode")
            negative_array = self._one_dimensional(
                "negative ion state", negative, size=self.x.size
            )
        else:
            if negative is not None:
                raise ValueError("negative ion state is invalid in single-ion mode")
            negative_array = None
        return {
            "positive_ion": positive_array,
            "negative_ion": negative_array,
        }

    def _bound_counts(
        self, name: str, values: np.ndarray | None
    ) -> tuple[int, int, int]:
        if values is None:
            return 0, 0, 0
        cap = self._limits[name]
        assert cap is not None
        finite = np.isfinite(values)
        return (
            int(np.count_nonzero(finite & (values < 0.0))),
            int(np.count_nonzero(finite & (values > cap))),
            int(values.size - np.count_nonzero(finite)),
        )

    def _inventory(self, values: np.ndarray | None) -> float | None:
        if values is None or not np.all(np.isfinite(values)):
            return None
        return dual_cell_integral(self.x, values)

    @staticmethod
    def _positive_state_health(values: np.ndarray) -> tuple[float | None, int]:
        if values.size == 0:
            return None, 0
        finite = np.isfinite(values)
        minimum = float(np.min(values[finite])) if np.any(finite) else None
        return minimum, int(np.count_nonzero(finite & (values <= 0.0)))

    @staticmethod
    def _relative_drift(
        initial: float | None, terminal: float | None
    ) -> float | None:
        if initial is None or terminal is None:
            return None
        if initial == 0.0:
            return 0.0 if terminal == 0.0 else float("inf")
        return abs(terminal - initial) / abs(initial)

    @staticmethod
    def _finite_extreme(values, reducer) -> float | None:
        if values is None:
            return None
        finite = np.asarray(values)[np.isfinite(values)]
        return float(reducer(finite)) if finite.size else None

    @staticmethod
    def _minimum(current: float | None, observed: float) -> float:
        return observed if current is None else min(current, observed)

    @staticmethod
    def _maximum(current: float | None, observed: float) -> float:
        return observed if current is None else max(current, observed)

    def _validated_limit(self, name: str, values: np.ndarray) -> np.ndarray:
        array = self._one_dimensional(name, values, size=self.x.size)
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        return array.copy()

    @staticmethod
    def _one_dimensional(
        name: str, values: np.ndarray, *, size: int | None = None
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or (size is not None and array.size != size):
            suffix = "" if size is None else f" with length {size}"
            raise ValueError(f"{name} must be one-dimensional{suffix}")
        return array

    def _raise_if_strict_rejected(self, stage: str) -> None:
        if not self.policy.strict:
            return
        report = self.report()
        if report.violations:
            self._raise(report, stage)

    @staticmethod
    def _raise(report: SplitStepDiagnosticsReport, stage: str) -> None:
        raise SplitStepDiagnosticsError(
            f"research-strict split-step diagnostics rejected {stage}: "
            + "; ".join(report.violations),
            report,
        )


class NumericalDiagnosticsMonitor:
    """Mutable per-solve collector; not shared between solver invocations."""

    _BLOCKS: tuple[DensityBlock, ...] = (
        "n",
        "p",
        "P",
        "P_neg",
        "interface_state",
    )

    def __init__(
        self,
        layout: StateLayout,
        policy: NumericalDiagnosticsPolicy | None = None,
    ) -> None:
        self.layout = layout
        self.policy = policy or NumericalDiagnosticsPolicy()
        self._positive_ion_active = np.asarray(
            layout.positive_ion_active, dtype=bool
        )
        self._negative_ion_active = np.asarray(
            layout.negative_ion_active, dtype=bool
        )
        self.trial_evaluations = 0
        self.negative_trial_evaluations = 0
        self.nonfinite_trial_evaluations = 0
        self.nonfinite_rhs_evaluations = 0
        self._negative_entries = {name: 0 for name in self._BLOCKS}
        self._trial_minima: dict[DensityBlock, float | None] = {
            name: None for name in self._BLOCKS
        }
        self._final_minima: DensityMinima | None = None
        self._final_blocks: dict[DensityBlock, np.ndarray] | None = None
        self._minimum_bulk_srh: float | None = None
        self._minimum_interface_srh: float | None = None
        self._nonfinite_srh_evaluations = 0
        self._solver_success: bool | None = None

    def observe_trial_state(self, state: np.ndarray) -> None:
        blocks = self.layout.split(state)
        self.trial_evaluations += 1
        has_negative = False
        has_nonfinite = False
        for name, values in blocks.items():
            array = np.asarray(values)
            negative = int(np.count_nonzero(array < 0.0))
            self._negative_entries[name] += negative
            has_negative = has_negative or negative > 0
            finite_mask = np.isfinite(array)
            block_is_finite = bool(np.all(finite_mask))
            has_nonfinite = has_nonfinite or not block_is_finite
            self._update_minimum(
                name, array, finite_mask, block_is_finite=block_is_finite
            )
        if has_negative:
            self.negative_trial_evaluations += 1
        if has_nonfinite:
            self.nonfinite_trial_evaluations += 1

    def observe_rhs(self, rhs: np.ndarray) -> None:
        array = np.asarray(rhs)
        if array.ndim != 1 or array.size != self.layout.expected_size:
            raise ValueError(
                f"RHS has shape {array.shape}; expected one-dimensional "
                f"layout of length {self.layout.expected_size}"
            )
        if np.all(np.isfinite(array)):
            return
        self.nonfinite_rhs_evaluations += 1
        if self.policy.strict:
            self._raise_strict("non-finite RHS evaluation")

    def observe_nonfinite_rhs_exception(self) -> None:
        """Record the legacy environment guard raising before RHS returns."""

        self.nonfinite_rhs_evaluations += 1
        if self.policy.strict:
            self._raise_strict("non-finite RHS evaluation")

    def observe_srh_denominator(
        self, kind: Literal["bulk", "interface"], denominator: np.ndarray
    ) -> None:
        if kind not in ("bulk", "interface"):
            raise ValueError(f"unknown SRH denominator kind: {kind!r}")
        array = np.asarray(denominator, dtype=float)
        finite_mask = np.isfinite(array)
        all_finite = bool(np.all(finite_mask))
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count:
            observed = float(
                np.min(array) if all_finite else np.min(array[finite_mask])
            )
            if kind == "bulk":
                self._minimum_bulk_srh = self._minimum(
                    self._minimum_bulk_srh, observed
                )
                floor = self.policy.bulk_srh_denominator_floor_s_m3
            elif kind == "interface":
                self._minimum_interface_srh = self._minimum(
                    self._minimum_interface_srh, observed
                )
                floor = self.policy.interface_srh_denominator_floor_s_m4
        else:
            floor = 0.0
        if not all_finite:
            self._nonfinite_srh_evaluations += 1
            if self.policy.strict:
                self._raise_strict(f"non-finite {kind} SRH denominator")
        if self.policy.strict and finite_count and observed <= floor:
            self._raise_strict(
                f"{kind} SRH denominator {observed:.6e} is not above "
                f"the declared floor {floor:.6e}"
            )

    def finalize(
        self, terminal_state: np.ndarray | None, *, solver_success: bool
    ) -> NumericalDiagnosticsReport:
        self._solver_success = bool(solver_success)
        if terminal_state is not None:
            blocks = self.layout.split(terminal_state)
            self._final_blocks = blocks
            self._final_minima = self._density_minima(blocks)
        report = self.report()
        if self.policy.strict and report.violations:
            raise NumericalDiagnosticsError(
                "research-strict numerical diagnostics rejected the solve: "
                + "; ".join(report.violations),
                report,
            )
        return report

    def report(self) -> NumericalDiagnosticsReport:
        violations = self._violations()
        return NumericalDiagnosticsReport(
            mode=self.policy.mode,
            solver_success=self._solver_success,
            trial_evaluations=self.trial_evaluations,
            negative_trial_evaluations=self.negative_trial_evaluations,
            negative_trial_entries=NegativeEntryCounts(
                n=self._negative_entries["n"],
                p=self._negative_entries["p"],
                positive_ion=self._negative_entries["P"],
                negative_ion=self._negative_entries["P_neg"],
                interface_state=self._negative_entries["interface_state"],
            ),
            nonfinite_trial_evaluations=self.nonfinite_trial_evaluations,
            nonfinite_rhs_evaluations=self.nonfinite_rhs_evaluations,
            minimum_trial_density_m3=self._density_minima_from_values(
                self._trial_minima
            ),
            final_minimum_density_m3=self._final_minima,
            minimum_bulk_srh_denominator_s_m3=self._minimum_bulk_srh,
            minimum_interface_srh_denominator_s_m4=(
                self._minimum_interface_srh
            ),
            terminal_density_floor_m3=(
                self.policy.terminal_density_floor_m3
            ),
            bulk_srh_denominator_floor_s_m3=(
                self.policy.bulk_srh_denominator_floor_s_m3
            ),
            interface_srh_denominator_floor_s_m4=(
                self.policy.interface_srh_denominator_floor_s_m4
            ),
            violations=violations,
            would_pass_strict=not violations,
        )

    def _violations(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self._solver_success is False:
            reasons.append("solver_not_successful")
        if self.nonfinite_rhs_evaluations:
            reasons.append("nonfinite_rhs")
        if self._nonfinite_srh_evaluations:
            reasons.append("nonfinite_srh_denominator")
        if self._minimum_bulk_srh is None:
            reasons.append("bulk_srh_denominator_not_observed")
        elif (
            self._minimum_bulk_srh
            <= self.policy.bulk_srh_denominator_floor_s_m3
        ):
            reasons.append("bulk_srh_denominator_below_floor")
        if (
            self._minimum_interface_srh is not None
            and self._minimum_interface_srh
            <= self.policy.interface_srh_denominator_floor_s_m4
        ):
            reasons.append("interface_srh_denominator_below_floor")
        if self._solver_success and self._final_blocks is None:
            reasons.append("terminal_state_missing")
        if self._final_blocks is not None:
            reasons.extend(self._terminal_violations(self._final_blocks))
        return tuple(reasons)

    def _terminal_violations(
        self, blocks: dict[DensityBlock, np.ndarray]
    ) -> list[str]:
        reasons: list[str] = []
        floor = self.policy.terminal_density_floor_m3
        for name in ("n", "p", "interface_state"):
            values = blocks.get(name)
            if values is None:
                continue
            if not np.all(np.isfinite(values)):
                reasons.append(f"terminal_{name}_nonfinite")
            elif np.any(values <= floor):
                reasons.append(f"terminal_{name}_not_above_floor")

        for name, mask_tuple in (
            ("P", self.layout.positive_ion_active),
            ("P_neg", self.layout.negative_ion_active),
        ):
            values = blocks.get(name)
            if values is None:
                continue
            if not np.all(np.isfinite(values)):
                reasons.append(f"terminal_{name}_nonfinite")
                continue
            mask = np.asarray(mask_tuple, dtype=bool)
            if np.any(values[mask] <= floor):
                reasons.append(f"terminal_{name}_active_not_above_floor")
            if np.any(values[~mask] < 0.0):
                reasons.append(f"terminal_{name}_inactive_negative")
        return reasons

    def _density_minima(
        self, blocks: dict[DensityBlock, np.ndarray]
    ) -> DensityMinima:
        values: dict[DensityBlock, float | None] = {
            name: self._finite_minimum(array)
            for name, array in blocks.items()
        }
        values["P"] = self._active_minimum(
            blocks.get("P"), self.layout.positive_ion_active
        )
        values["P_neg"] = self._active_minimum(
            blocks.get("P_neg"), self.layout.negative_ion_active
        )
        return self._density_minima_from_values(values)

    @staticmethod
    def _density_minima_from_values(
        values: dict[DensityBlock, float | None]
    ) -> DensityMinima:
        return DensityMinima(
            n=values.get("n"),
            p=values.get("p"),
            positive_ion_active=values.get("P"),
            negative_ion_active=values.get("P_neg"),
            interface_state=values.get("interface_state"),
        )

    def _update_minimum(
        self,
        name: DensityBlock,
        values: np.ndarray,
        finite_mask: np.ndarray,
        *,
        block_is_finite: bool,
    ) -> None:
        if name == "P":
            selected = self._positive_ion_active
        elif name == "P_neg":
            selected = self._negative_ion_active
        else:
            if block_is_finite:
                observed = float(np.min(values))
            elif np.any(finite_mask):
                observed = float(np.min(values[finite_mask]))
            else:
                return
            self._trial_minima[name] = self._minimum(
                self._trial_minima[name], observed
            )
            return
        if not block_is_finite:
            selected = selected & finite_mask
        if not np.any(selected):
            return
        observed = float(np.min(values[selected]))
        self._trial_minima[name] = self._minimum(
            self._trial_minima[name], observed
        )

    @staticmethod
    def _active_minimum(
        values: np.ndarray | None, mask_tuple: tuple[bool, ...]
    ) -> float | None:
        if values is None:
            return None
        mask = np.asarray(mask_tuple, dtype=bool)
        return NumericalDiagnosticsMonitor._finite_minimum(values[mask])

    @staticmethod
    def _finite_minimum(values: np.ndarray) -> float | None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.min(finite)) if finite.size else None

    @staticmethod
    def _minimum(current: float | None, observed: float) -> float:
        return observed if current is None else min(current, observed)

    def _raise_strict(self, reason: str) -> None:
        report = self.report()
        raise NumericalDiagnosticsError(
            f"research-strict numerical diagnostics rejected the solve: {reason}",
            report,
        )


__all__ = [
    "DensityMinima",
    "NegativeEntryCounts",
    "NumericalDiagnosticsError",
    "NumericalDiagnosticsMonitor",
    "NumericalDiagnosticsPolicy",
    "NumericalDiagnosticsReport",
    "StateLayout",
]
