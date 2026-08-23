"""Transient lock-in cross-check for certified ion-aware impedance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Literal, Mapping, Self

import numpy as np

from perovskite_sim.experiments.impedance import _lockin_extract
from perovskite_sim.experiments.ion_aware_dc import (
    IonAwareDCResult,
    ion_aware_dc_state_sha256,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    IonAwareImpedanceResult,
    MAX_LINEAR_PERTURBATION_V,
)
from perovskite_sim.experiments.ion_aware_impedance_grid import (
    ion_aware_impedance_grid_sha256,
)
from perovskite_sim.experiments.jv_sweep import (
    _total_current_faces,
    compute_current_components,
)
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver.mol import (
    StateVec,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsError,
    NumericalDiagnosticsPolicy,
    NumericalDiagnosticsReport,
)
from perovskite_sim.solver.tolerances import ComponentwiseAtol


ION_AWARE_TRANSIENT_LOCKIN_PROTOCOL_SCHEMA = (
    "ion-aware-transient-lockin-protocol-v1"
)
ProgressCallback = Callable[[str, int, int, str], None]


class IonAwareTransientLockInCapabilityError(RuntimeError):
    """The supplied DC/frequency-domain evidence cannot be cross-checked."""


class IonAwareTransientLockInError(RuntimeError):
    """A transient lock-in solve failed before producing graded evidence."""


class IonAwareTransientLockInCertificationError(RuntimeError):
    """A completed transient cross-check failed a requested evidence axis."""

    def __init__(
        self,
        message: str,
        result: IonAwareTransientLockInResult,
    ) -> None:
        self.result = result
        super().__init__(message)


def _positive(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _positive_integer(value: object, field: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class IonAwareTransientLockInProtocol:
    """Exact DC identity, AC waveform, time ladder, and acceptance contract."""

    grid_sha256: str
    dc_protocol_sha256: str
    dc_state_sha256: str
    frequency_domain_protocol_sha256: str
    frequencies_Hz: tuple[float, ...]
    delta_V: float
    cycles: int = 6
    extraction_cycles: int = 2
    points_per_cycle_levels: tuple[int, ...] = (40, 80, 160)
    solver_steps_per_sample_interval: int = 5
    rtol: float = 1.0e-6
    carrier_atol_fraction: float = 1.0e-12
    ion_atol_fraction: float = 1.0e-12
    interface_atol_fraction: float = 1.0e-12
    minimum_atol: float = 1.0e-6
    max_nfev_per_solve: int = 500_000
    max_frequency_domain_magnitude_relative_difference: float = 2.0e-2
    max_frequency_domain_phase_difference_deg: float = 1.0
    max_time_resolution_magnitude_relative_change: float = 1.0e-2
    max_time_resolution_phase_change_deg: float = 0.5
    max_cycle_current_relative_change: float = 1.0e-2
    max_state_periodicity_relative_change: float = 1.0e-3
    max_ion_inventory_relative_drift: float = 1.0e-10
    current_convention: Literal["passive"] = "passive"
    schema_version: Literal["ion-aware-transient-lockin-protocol-v1"] = (
        ION_AWARE_TRANSIENT_LOCKIN_PROTOCOL_SCHEMA
    )

    def __post_init__(self) -> None:
        for field in (
            "grid_sha256",
            "dc_protocol_sha256",
            "dc_state_sha256",
            "frequency_domain_protocol_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
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
        delta_voltage = _positive(self.delta_V, "delta_V")
        if delta_voltage >= MAX_LINEAR_PERTURBATION_V:
            raise ValueError("delta_V must be below the 20 mV small-signal limit")
        object.__setattr__(self, "delta_V", delta_voltage)
        cycles = _positive_integer(self.cycles, "cycles", minimum=3)
        extraction = _positive_integer(
            self.extraction_cycles,
            "extraction_cycles",
            minimum=2,
        )
        if extraction > cycles:
            raise ValueError("extraction_cycles cannot exceed cycles")
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "extraction_cycles", extraction)
        try:
            levels = tuple(
                _positive_integer(
                    value,
                    f"points_per_cycle_levels[{index}]",
                    minimum=8,
                )
                for index, value in enumerate(self.points_per_cycle_levels)
            )
        except TypeError as exc:
            raise TypeError("points_per_cycle_levels must be an iterable") from exc
        if len(levels) < 3 or any(
            right <= left for left, right in zip(levels, levels[1:])
        ):
            raise ValueError(
                "points_per_cycle_levels must contain at least three strictly "
                "increasing levels"
            )
        object.__setattr__(self, "points_per_cycle_levels", levels)
        object.__setattr__(
            self,
            "solver_steps_per_sample_interval",
            _positive_integer(
                self.solver_steps_per_sample_interval,
                "solver_steps_per_sample_interval",
            ),
        )
        object.__setattr__(
            self,
            "max_nfev_per_solve",
            _positive_integer(self.max_nfev_per_solve, "max_nfev_per_solve"),
        )
        for field in (
            "rtol",
            "carrier_atol_fraction",
            "ion_atol_fraction",
            "interface_atol_fraction",
            "minimum_atol",
            "max_frequency_domain_magnitude_relative_difference",
            "max_frequency_domain_phase_difference_deg",
            "max_time_resolution_magnitude_relative_change",
            "max_time_resolution_phase_change_deg",
            "max_cycle_current_relative_change",
            "max_state_periodicity_relative_change",
            "max_ion_inventory_relative_drift",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.current_convention != "passive":
            raise ValueError("ion-aware lock-in uses the passive current convention")
        if self.schema_version != ION_AWARE_TRANSIENT_LOCKIN_PROTOCOL_SCHEMA:
            raise ValueError("unsupported ion-aware transient lock-in schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["frequencies_Hz"] = list(self.frequencies_Hz)
        payload["points_per_cycle_levels"] = list(self.points_per_cycle_levels)
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
            raise TypeError("ion-aware transient lock-in protocol must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "ion-aware transient lock-in protocol keys do not match schema; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        values = dict(payload)
        values["frequencies_Hz"] = tuple(values["frequencies_Hz"])
        values["points_per_cycle_levels"] = tuple(
            values["points_per_cycle_levels"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("lock-in protocol JSON must contain an object")
        return cls.from_dict(parsed)


def build_ion_aware_transient_lockin_protocol(
    frequency_domain: IonAwareImpedanceResult,
    frequencies_Hz: np.ndarray | tuple[float, ...],
    **overrides: Any,
) -> IonAwareTransientLockInProtocol:
    """Bind a transient comparison to one exact frequency-domain result."""
    if not isinstance(frequency_domain, IonAwareImpedanceResult):
        raise TypeError("frequency_domain must be an IonAwareImpedanceResult")
    selected = np.asarray(frequencies_Hz, dtype=float)
    available = np.asarray(frequency_domain.frequencies, dtype=float)
    if (
        selected.ndim != 1
        or selected.size == 0
        or not np.all(np.isfinite(selected))
        or np.any(selected <= 0.0)
        or np.any(np.diff(selected) <= 0.0)
    ):
        raise ValueError("selected lock-in frequencies must be positive and increasing")
    if any(not np.any(available == value) for value in selected):
        raise IonAwareTransientLockInCapabilityError(
            "every lock-in frequency must exactly match the frequency-domain request"
        )
    dc_state = frequency_domain.dc_state
    return IonAwareTransientLockInProtocol(
        grid_sha256=ion_aware_impedance_grid_sha256(dc_state.x),
        dc_protocol_sha256=dc_state.protocol_hash,
        dc_state_sha256=ion_aware_dc_state_sha256(dc_state.y),
        frequency_domain_protocol_sha256=frequency_domain.protocol_hash,
        frequencies_Hz=tuple(selected.tolist()),
        delta_V=frequency_domain.protocol.delta_V,
        **overrides,
    )


@dataclass(frozen=True, slots=True)
class TransientLockInFrequencyEvidence:
    frequency_Hz: float
    points_per_cycle: int
    impedance_ohm_m2: complex
    cycle_current_relative_change: float
    state_periodicity_relative_change: float
    max_ion_inventory_relative_drift: float
    accepted_method: str
    numerical_diagnostics: NumericalDiagnosticsReport


@dataclass(frozen=True, slots=True)
class TransientLockInResolutionEvidence:
    points_per_cycle: int
    frequency_evidence: tuple[TransientLockInFrequencyEvidence, ...]


@dataclass(frozen=True, slots=True)
class TransientLockInFrequencyCertificate:
    frequency_Hz: float
    numerically_certified: bool
    frequency_domain_impedance_ohm_m2: complex
    transient_impedance_ohm_m2: complex
    frequency_domain_magnitude_relative_difference: float
    frequency_domain_phase_difference_deg: float
    time_resolution_magnitude_relative_change: float
    time_resolution_phase_change_deg: float
    cycle_current_relative_change: float
    state_periodicity_relative_change: float
    max_ion_inventory_relative_drift: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareTransientLockInCertificate:
    numerically_certified: bool
    frequency_domain_agreement_certified: bool
    time_resolution_certified: bool
    periodicity_certified: bool
    inventory_certified: bool
    thermodynamically_certified: bool
    frequency_window_certified: bool
    certified: bool
    max_frequency_domain_magnitude_relative_difference: float
    max_frequency_domain_phase_difference_deg: float
    max_time_resolution_magnitude_relative_change: float
    max_time_resolution_phase_change_deg: float
    max_cycle_current_relative_change: float
    max_state_periodicity_relative_change: float
    max_ion_inventory_relative_drift: float
    frequency_certificates: tuple[TransientLockInFrequencyCertificate, ...]
    numerical_reasons: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareTransientLockInResult:
    protocol: IonAwareTransientLockInProtocol
    frequency_domain: IonAwareImpedanceResult
    dc_state: IonAwareDCResult
    resolution_evidence: tuple[TransientLockInResolutionEvidence, ...]
    certificate: IonAwareTransientLockInCertificate

    @property
    def protocol_hash(self) -> str:
        return self.protocol.protocol_hash


def _block_state_change(
    left: np.ndarray,
    right: np.ndarray,
    n_nodes: int,
    material,
) -> float:
    first = StateVec.unpack(left, n_nodes, N_iface_state=material.N_iface_state)
    second = StateVec.unpack(right, n_nodes, N_iface_state=material.N_iface_state)
    comparisons: list[float] = []
    for name, mask in (
        ("n", np.ones(n_nodes, dtype=bool)),
        ("p", np.ones(n_nodes, dtype=bool)),
        (
            "P",
            (np.asarray(material.P_ion0) > 0.0)
            & (np.asarray(material.D_ion_node) > 0.0),
        ),
    ):
        a = np.asarray(getattr(first, name), dtype=float)[mask]
        b = np.asarray(getattr(second, name), dtype=float)[mask]
        if a.size:
            scale = max(
                float(np.max(np.abs(a))),
                float(np.max(np.abs(b))),
                np.finfo(float).tiny,
            )
            comparisons.append(float(np.max(np.abs(b - a))) / scale)
    if material.has_dual_ions:
        if (
            first.P_neg is None
            or second.P_neg is None
            or material.P_ion0_neg is None
            or material.D_ion_neg_node is None
        ):
            raise IonAwareTransientLockInCapabilityError(
                "dual-ion lock-in state is missing the negative-ion block"
            )
        mask = (np.asarray(material.P_ion0_neg) > 0.0) & (
            np.asarray(material.D_ion_neg_node) > 0.0
        )
        a = np.asarray(first.P_neg, dtype=float)[mask]
        b = np.asarray(second.P_neg, dtype=float)[mask]
        if a.size:
            scale = max(
                float(np.max(np.abs(a))),
                float(np.max(np.abs(b))),
                np.finfo(float).tiny,
            )
            comparisons.append(float(np.max(np.abs(b - a))) / scale)
    return max(comparisons, default=0.0)


def _ion_inventory_drift(
    x: np.ndarray,
    initial: np.ndarray,
    terminal: np.ndarray,
    material,
) -> float:
    first = StateVec.unpack(initial, len(x), N_iface_state=material.N_iface_state)
    last = StateVec.unpack(terminal, len(x), N_iface_state=material.N_iface_state)
    drifts: list[float] = []
    for a, b in ((first.P, last.P), (first.P_neg, last.P_neg)):
        if a is None or b is None:
            continue
        initial_inventory = dual_cell_integral(x, a)
        terminal_inventory = dual_cell_integral(x, b)
        scale = max(abs(initial_inventory), np.finfo(float).tiny)
        drifts.append(abs(terminal_inventory - initial_inventory) / scale)
    return max(drifts, default=0.0)


def _relative_magnitude_change(left: complex, right: complex) -> float:
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    return float(abs(abs(right) - abs(left)) / scale)


def _phase_change_deg(left: complex, right: complex) -> float:
    if left == 0.0 or right == 0.0:
        return float("inf")
    return float(abs(np.angle(right / left, deg=True)))


def _run_frequency(
    dc_state: IonAwareDCResult,
    protocol: IonAwareTransientLockInProtocol,
    frequency: float,
    points_per_cycle: int,
    material,
) -> TransientLockInFrequencyEvidence:
    x = np.asarray(dc_state.x, dtype=float)
    stack = dc_state.stack
    period = 1.0 / frequency
    sample_interval = period / points_per_cycle
    intervals = protocol.cycles * points_per_cycle
    edges = np.arange(intervals + 1, dtype=float) * sample_interval
    midpoints = edges[:-1] + 0.5 * sample_interval
    samples = np.empty(2 * intervals + 1, dtype=float)
    samples[0::2] = edges
    samples[1::2] = midpoints

    def applied_voltage(time_s: float) -> float:
        return dc_state.protocol.V_dc + protocol.delta_V * np.sin(
            2.0 * np.pi * frequency * time_s
        )

    diagnostics_policy = NumericalDiagnosticsPolicy.research_strict(
        terminal_density_floor_m3=dc_state.protocol.terminal_density_floor_m3,
        bulk_srh_denominator_floor_s_m3=0.0,
        interface_srh_denominator_floor_s_m4=0.0,
    )
    atol = ComponentwiseAtol(
        carrier_fraction=protocol.carrier_atol_fraction,
        ion_fraction=protocol.ion_atol_fraction,
        interface_fraction=protocol.interface_atol_fraction,
        minimum_atol=protocol.minimum_atol,
    )
    try:
        solution = run_transient(
            x,
            np.asarray(dc_state.y, dtype=float).copy(),
            (0.0, float(edges[-1])),
            samples,
            stack,
            illuminated=dc_state.protocol.illuminated,
            V_app=applied_voltage,
            rtol=protocol.rtol,
            atol=atol,
            max_step=(
                sample_interval / protocol.solver_steps_per_sample_interval
            ),
            mat=material,
            max_nfev=protocol.max_nfev_per_solve,
            method="Radau",
            numerical_diagnostics=diagnostics_policy,
        )
    except NumericalDiagnosticsError as exc:
        raise IonAwareTransientLockInError(
            "strict transient diagnostics failed at "
            f"f={frequency:.6g} Hz, points_per_cycle={points_per_cycle}: {exc}"
        ) from exc
    report = getattr(solution, "numerical_diagnostics", None)
    if not bool(getattr(solution, "success", False)) or not isinstance(
        report, NumericalDiagnosticsReport
    ):
        detail = getattr(solution, "message", "no solver diagnostic")
        raise IonAwareTransientLockInError(
            "transient lock-in solve failed at "
            f"f={frequency:.6g} Hz, points_per_cycle={points_per_cycle}: {detail}"
        )
    values = np.asarray(solution.y, dtype=float)
    expected_shape = (np.asarray(dc_state.y).size, samples.size)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise IonAwareTransientLockInError(
            "transient lock-in solver returned a malformed state history"
        )

    edge_states = values[:, 0::2]
    midpoint_states = values[:, 1::2]
    dx_faces = np.diff(x)
    device_length = float(x[-1] - x[0])
    solar_current = np.empty(intervals, dtype=float)
    for index in range(intervals):
        voltage_left = applied_voltage(edges[index])
        voltage_right = applied_voltage(edges[index + 1])
        voltage_mid = applied_voltage(midpoints[index])
        total_at_edge = _total_current_faces(
            x,
            edge_states[:, index + 1],
            stack,
            voltage_right,
            y_prev=edge_states[:, index],
            dt=sample_interval,
            mat=material,
            V_app_prev=voltage_left,
        )
        conduction_at_edge = compute_current_components(
            x,
            edge_states[:, index + 1],
            stack,
            voltage_right,
            mat=material,
        ).J_total
        conduction_at_midpoint = compute_current_components(
            x,
            midpoint_states[:, index],
            stack,
            voltage_mid,
            mat=material,
        ).J_total
        current_faces = (
            conduction_at_midpoint + total_at_edge - conduction_at_edge
        )
        solar_current[index] = float(
            np.sum(current_faces * dx_faces) / device_length
        )
    passive_current = -solar_current
    extraction_points = protocol.extraction_cycles * points_per_cycle
    impedance = _lockin_extract(
        passive_current[-extraction_points:],
        midpoints[-extraction_points:],
        frequency,
        protocol.delta_V,
    )
    if not np.isfinite(impedance) or impedance == 0.0:
        raise IonAwareTransientLockInError(
            f"lock-in extraction was non-finite at f={frequency:.6g} Hz"
        )
    previous_cycle = passive_current[-2 * points_per_cycle : -points_per_cycle]
    final_cycle = passive_current[-points_per_cycle:]
    previous_ac = previous_cycle - np.mean(previous_cycle)
    final_ac = final_cycle - np.mean(final_cycle)
    current_scale = max(
        float(np.max(np.abs(previous_ac))),
        float(np.max(np.abs(final_ac))),
        np.finfo(float).tiny,
    )
    cycle_change = float(np.max(np.abs(final_cycle - previous_cycle))) / current_scale
    state_change = _block_state_change(
        edge_states[:, -(points_per_cycle + 1)],
        edge_states[:, -1],
        len(x),
        material,
    )
    inventory_drift = _ion_inventory_drift(
        x,
        np.asarray(dc_state.y, dtype=float),
        edge_states[:, -1],
        material,
    )
    return TransientLockInFrequencyEvidence(
        frequency_Hz=frequency,
        points_per_cycle=points_per_cycle,
        impedance_ohm_m2=complex(impedance),
        cycle_current_relative_change=cycle_change,
        state_periodicity_relative_change=state_change,
        max_ion_inventory_relative_drift=inventory_drift,
        accepted_method="Radau",
        numerical_diagnostics=report,
    )


def _certificate(
    protocol: IonAwareTransientLockInProtocol,
    frequency_domain: IonAwareImpedanceResult,
    levels: tuple[TransientLockInResolutionEvidence, ...],
) -> IonAwareTransientLockInCertificate:
    reference_index = {
        float(frequency): index
        for index, frequency in enumerate(frequency_domain.frequencies)
    }
    previous = levels[-2].frequency_evidence
    final = levels[-1].frequency_evidence
    point_certificates: list[TransientLockInFrequencyCertificate] = []
    for coarse, fine in zip(previous, final, strict=True):
        reference = complex(
            frequency_domain.Z[reference_index[fine.frequency_Hz]]
        )
        frequency_magnitude = _relative_magnitude_change(reference, fine.impedance_ohm_m2)
        frequency_phase = _phase_change_deg(reference, fine.impedance_ohm_m2)
        time_magnitude = _relative_magnitude_change(
            coarse.impedance_ohm_m2,
            fine.impedance_ohm_m2,
        )
        time_phase = _phase_change_deg(
            coarse.impedance_ohm_m2,
            fine.impedance_ohm_m2,
        )
        reasons: list[str] = []
        if not fine.numerical_diagnostics.would_pass_strict:
            reasons.append("strict_numerical_diagnostics_failed")
        if (
            frequency_magnitude
            > protocol.max_frequency_domain_magnitude_relative_difference
        ):
            reasons.append("frequency_domain_magnitude_disagreement")
        if frequency_phase > protocol.max_frequency_domain_phase_difference_deg:
            reasons.append("frequency_domain_phase_disagreement")
        if (
            time_magnitude
            > protocol.max_time_resolution_magnitude_relative_change
        ):
            reasons.append("time_resolution_magnitude_not_converged")
        if time_phase > protocol.max_time_resolution_phase_change_deg:
            reasons.append("time_resolution_phase_not_converged")
        if fine.cycle_current_relative_change > (
            protocol.max_cycle_current_relative_change
        ):
            reasons.append("periodic_current_waveform_not_closed")
        if fine.state_periodicity_relative_change > (
            protocol.max_state_periodicity_relative_change
        ):
            reasons.append("periodic_state_not_closed")
        if fine.max_ion_inventory_relative_drift > (
            protocol.max_ion_inventory_relative_drift
        ):
            reasons.append("blocking_ion_inventory_drift_exceeds_limit")
        point_certificates.append(
            TransientLockInFrequencyCertificate(
                frequency_Hz=fine.frequency_Hz,
                numerically_certified=not reasons,
                frequency_domain_impedance_ohm_m2=reference,
                transient_impedance_ohm_m2=fine.impedance_ohm_m2,
                frequency_domain_magnitude_relative_difference=frequency_magnitude,
                frequency_domain_phase_difference_deg=frequency_phase,
                time_resolution_magnitude_relative_change=time_magnitude,
                time_resolution_phase_change_deg=time_phase,
                cycle_current_relative_change=fine.cycle_current_relative_change,
                state_periodicity_relative_change=(
                    fine.state_periodicity_relative_change
                ),
                max_ion_inventory_relative_drift=(
                    fine.max_ion_inventory_relative_drift
                ),
                reasons=tuple(reasons),
            )
        )
    all_evidence = tuple(
        evidence
        for level in levels
        for evidence in level.frequency_evidence
    )
    diagnostics = all(
        item.numerical_diagnostics.would_pass_strict for item in all_evidence
    )
    frequency_agreement = all(
        item.frequency_domain_magnitude_relative_difference
        <= protocol.max_frequency_domain_magnitude_relative_difference
        and item.frequency_domain_phase_difference_deg
        <= protocol.max_frequency_domain_phase_difference_deg
        for item in point_certificates
    )
    time_resolution = all(
        item.time_resolution_magnitude_relative_change
        <= protocol.max_time_resolution_magnitude_relative_change
        and item.time_resolution_phase_change_deg
        <= protocol.max_time_resolution_phase_change_deg
        for item in point_certificates
    )
    periodicity = all(
        item.cycle_current_relative_change
        <= protocol.max_cycle_current_relative_change
        and item.state_periodicity_relative_change
        <= protocol.max_state_periodicity_relative_change
        for item in point_certificates
    )
    inventory = all(
        item.max_ion_inventory_relative_drift
        <= protocol.max_ion_inventory_relative_drift
        for item in all_evidence
    )
    numerical_reasons: list[str] = []
    if not diagnostics:
        numerical_reasons.append("strict_numerical_diagnostics_failed")
    if not frequency_agreement:
        numerical_reasons.append("frequency_domain_agreement_failed")
    if not time_resolution:
        numerical_reasons.append("time_resolution_not_converged")
    if not periodicity:
        numerical_reasons.append("periodicity_not_converged")
    if not inventory:
        numerical_reasons.append("blocking_ion_inventory_not_conserved")
    numerical = not numerical_reasons
    thermodynamic = frequency_domain.certificate.thermodynamically_certified
    frequency_window = frequency_domain.certificate.frequency_window_certified
    reasons = list(numerical_reasons)
    if not thermodynamic:
        reasons.append("contact_thermodynamics_not_certified")
    if not frequency_window:
        reasons.append("frequency_window_not_certified")
    return IonAwareTransientLockInCertificate(
        numerically_certified=numerical,
        frequency_domain_agreement_certified=frequency_agreement,
        time_resolution_certified=time_resolution,
        periodicity_certified=periodicity,
        inventory_certified=inventory,
        thermodynamically_certified=thermodynamic,
        frequency_window_certified=frequency_window,
        certified=numerical and thermodynamic and frequency_window,
        max_frequency_domain_magnitude_relative_difference=max(
            item.frequency_domain_magnitude_relative_difference
            for item in point_certificates
        ),
        max_frequency_domain_phase_difference_deg=max(
            item.frequency_domain_phase_difference_deg
            for item in point_certificates
        ),
        max_time_resolution_magnitude_relative_change=max(
            item.time_resolution_magnitude_relative_change
            for item in point_certificates
        ),
        max_time_resolution_phase_change_deg=max(
            item.time_resolution_phase_change_deg
            for item in point_certificates
        ),
        max_cycle_current_relative_change=max(
            item.cycle_current_relative_change for item in point_certificates
        ),
        max_state_periodicity_relative_change=max(
            item.state_periodicity_relative_change for item in point_certificates
        ),
        max_ion_inventory_relative_drift=max(
            item.max_ion_inventory_relative_drift for item in all_evidence
        ),
        frequency_certificates=tuple(point_certificates),
        numerical_reasons=tuple(numerical_reasons),
        reasons=tuple(reasons),
    )


def run_ion_aware_transient_lockin_crosscheck(
    frequency_domain: IonAwareImpedanceResult,
    protocol: IonAwareTransientLockInProtocol,
    *,
    dc_state: IonAwareDCResult,
    require_numerical_certificate: bool = True,
    require_contact_certificate: bool = False,
    require_frequency_window_certificate: bool = False,
    progress: ProgressCallback | None = None,
) -> IonAwareTransientLockInResult:
    """Compare selected frequency-domain points with exact-DC transients."""
    if not isinstance(frequency_domain, IonAwareImpedanceResult):
        raise TypeError("frequency_domain must be an IonAwareImpedanceResult")
    if not isinstance(protocol, IonAwareTransientLockInProtocol):
        raise TypeError("protocol must be an IonAwareTransientLockInProtocol")
    if not isinstance(dc_state, IonAwareDCResult):
        raise TypeError("dc_state must be an IonAwareDCResult")
    grid = np.asarray(dc_state.x, dtype=float)
    if protocol.grid_sha256 != ion_aware_impedance_grid_sha256(grid):
        raise IonAwareTransientLockInCapabilityError(
            "transient lock-in grid does not match the protocol hash"
        )
    if protocol.dc_protocol_sha256 != dc_state.protocol_hash:
        raise IonAwareTransientLockInCapabilityError(
            "transient lock-in DC history does not match the protocol hash"
        )
    if protocol.dc_state_sha256 != ion_aware_dc_state_sha256(dc_state.y):
        raise IonAwareTransientLockInCapabilityError(
            "transient lock-in DC state does not match the protocol hash"
        )
    if frequency_domain.protocol_hash != (
        protocol.frequency_domain_protocol_sha256
    ):
        raise IonAwareTransientLockInCapabilityError(
            "frequency-domain result does not match the lock-in protocol"
        )
    if frequency_domain.dc_state.stack != dc_state.stack or not np.array_equal(
        frequency_domain.dc_state.x,
        grid,
    ):
        raise IonAwareTransientLockInCapabilityError(
            "frequency-domain and transient DC stack/grid differ"
        )
    if ion_aware_dc_state_sha256(frequency_domain.dc_state.y) != (
        protocol.dc_state_sha256
    ):
        raise IonAwareTransientLockInCapabilityError(
            "frequency-domain result was built from a different DC state"
        )
    if protocol.delta_V != frequency_domain.protocol.delta_V:
        raise IonAwareTransientLockInCapabilityError(
            "lock-in amplitude differs from the frequency-domain protocol"
        )
    if not dc_state.numerically_certified:
        raise IonAwareTransientLockInCapabilityError(
            "transient lock-in requires a numerically certified DC state"
        )
    if not frequency_domain.certificate.numerically_certified:
        raise IonAwareTransientLockInCapabilityError(
            "transient lock-in requires certified frequency-domain points"
        )
    available = np.asarray(frequency_domain.frequencies, dtype=float)
    selected = np.asarray(protocol.frequencies_Hz, dtype=float)
    if any(not np.any(available == value) for value in selected):
        raise IonAwareTransientLockInCapabilityError(
            "lock-in frequencies are absent from the bound frequency-domain result"
        )
    material = build_material_arrays(grid, dc_state.stack)
    if material.N_iface_state:
        raise IonAwareTransientLockInCapabilityError(
            "dynamic interface-state blocks remain outside ion-aware lock-in"
        )

    total = len(protocol.points_per_cycle_levels) * len(selected)
    completed = 0
    levels: list[TransientLockInResolutionEvidence] = []
    for points_per_cycle in protocol.points_per_cycle_levels:
        frequency_evidence: list[TransientLockInFrequencyEvidence] = []
        for frequency in selected:
            if progress is not None:
                progress(
                    "ion_aware_transient_lockin",
                    completed,
                    total,
                    f"f={frequency:.6g} Hz, {points_per_cycle} points/cycle",
                )
            frequency_evidence.append(
                _run_frequency(
                    dc_state,
                    protocol,
                    float(frequency),
                    points_per_cycle,
                    material,
                )
            )
            completed += 1
        levels.append(
            TransientLockInResolutionEvidence(
                points_per_cycle=points_per_cycle,
                frequency_evidence=tuple(frequency_evidence),
            )
        )
    if progress is not None:
        progress(
            "ion_aware_transient_lockin",
            total,
            total,
            "transient lock-in ladder complete",
        )
    resolution_evidence = tuple(levels)
    certificate = _certificate(protocol, frequency_domain, resolution_evidence)
    result = IonAwareTransientLockInResult(
        protocol=protocol,
        frequency_domain=frequency_domain,
        dc_state=dc_state,
        resolution_evidence=resolution_evidence,
        certificate=certificate,
    )
    if require_numerical_certificate and not certificate.numerically_certified:
        raise IonAwareTransientLockInCertificationError(
            "ion-aware transient lock-in numerical certificate failed: "
            + ", ".join(certificate.numerical_reasons),
            result,
        )
    if require_contact_certificate and not certificate.thermodynamically_certified:
        raise IonAwareTransientLockInCertificationError(
            "ion-aware transient lock-in contact certificate failed",
            result,
        )
    if (
        require_frequency_window_certificate
        and not certificate.frequency_window_certified
    ):
        raise IonAwareTransientLockInCertificationError(
            "ion-aware transient lock-in frequency-window certificate failed",
            result,
        )
    return result


__all__ = [
    "ION_AWARE_TRANSIENT_LOCKIN_PROTOCOL_SCHEMA",
    "IonAwareTransientLockInCapabilityError",
    "IonAwareTransientLockInCertificate",
    "IonAwareTransientLockInCertificationError",
    "IonAwareTransientLockInError",
    "IonAwareTransientLockInProtocol",
    "IonAwareTransientLockInResult",
    "TransientLockInFrequencyCertificate",
    "TransientLockInFrequencyEvidence",
    "TransientLockInResolutionEvidence",
    "build_ion_aware_transient_lockin_protocol",
    "run_ion_aware_transient_lockin_crosscheck",
]
