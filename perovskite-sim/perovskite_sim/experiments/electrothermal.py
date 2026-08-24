"""Protocol-conditioned steady electrothermal operating-point coupling.

Every temperature evaluation starts a fresh, explicitly described J-V
experiment from the same frozen device stack. The certified intrinsic curve is
mapped through the area-normalized series/shunt circuit, its sampled terminal
maximum-power point is selected, and that exported power closes the lumped
first-law balance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import brentq

from perovskite_sim.experiments.external_circuit import (
    ExternalCircuitBranch,
    ExternalCircuitJVResult,
    ExternalCircuitProtocol,
    apply_external_circuit,
)
from perovskite_sim.experiments.jv_sweep import (
    build_jv_experiment_protocol,
    run_jv_sweep,
)
from perovskite_sim.experiments.protocol import ExperimentProtocol
from perovskite_sim.experiments.thermal_balance import (
    LumpedThermalProtocol,
    ThermalPowerLedger,
    thermal_power_ledger,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.solver.tolerances import ComponentwiseAtol


class ElectrothermalError(ValueError):
    """Base class for invalid electrothermal contracts or results."""


class ElectrothermalCapabilityError(ElectrothermalError):
    """Raised when the selected device mode cannot respond to temperature."""


class ElectrothermalSourceError(ElectrothermalError):
    """Raised when an electrical evaluation is incomplete or inconsistent."""


class ElectrothermalConvergenceError(ElectrothermalError):
    """Raised when the bounded electrothermal root cannot be certified."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return 0.0 if number == 0.0 else number


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


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer >= {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be an integer >= {minimum}") from exc
    if integer != value or integer < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return integer


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("electrothermal document contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "electrothermal document contains a non-JSON value of type "
        f"{type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _exact_keys(payload: Mapping[str, Any], cls: type, label: str) -> None:
    expected = {field.name for field in dataclasses.fields(cls)}
    actual = set(payload)
    if actual != expected:
        raise ElectrothermalError(
            f"{label} keys do not match schema; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True, slots=True)
class ElectrothermalJVProtocol:
    """Frozen electrical execution contract for every trial temperature."""

    grid_points_per_electrical_layer: int
    voltage_points_per_branch: int
    scan_rate_V_s: float
    voltage_max_V: float
    relative_tolerance: float = 1.0e-4
    carrier_atol_fraction: float = 1.0e-12
    ion_atol_fraction: float = 1.0e-12
    interface_atol_fraction: float = 1.0e-12
    minimum_atol: float = 1.0e-6
    atol_refinement_factor: float = 1.0
    incident_power_W_m2: float = 1000.0
    schema_version: Literal[1] = 1
    driver: Literal["transient_jv_strict"] = "transient_jv_strict"
    illumination: Literal["stack_baseline_generation"] = (
        "stack_baseline_generation"
    )
    temperature_initialization: Literal["fresh_state_per_temperature"] = (
        "fresh_state_per_temperature"
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("electrothermal J-V schema_version must be 1")
        if self.driver != "transient_jv_strict":
            raise ValueError("electrothermal driver must be transient_jv_strict")
        if self.illumination != "stack_baseline_generation":
            raise ValueError("electrothermal illumination must use the stack baseline")
        if self.temperature_initialization != "fresh_state_per_temperature":
            raise ValueError(
                "electrothermal temperature evaluations require fresh states"
            )
        object.__setattr__(
            self,
            "grid_points_per_electrical_layer",
            _integer(
                self.grid_points_per_electrical_layer,
                "grid_points_per_electrical_layer",
                minimum=3,
            ),
        )
        object.__setattr__(
            self,
            "voltage_points_per_branch",
            _integer(
                self.voltage_points_per_branch,
                "voltage_points_per_branch",
                minimum=3,
            ),
        )
        for name in (
            "scan_rate_V_s",
            "voltage_max_V",
            "relative_tolerance",
            "carrier_atol_fraction",
            "ion_atol_fraction",
            "interface_atol_fraction",
            "minimum_atol",
            "atol_refinement_factor",
            "incident_power_W_m2",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.relative_tolerance >= 1.0:
            raise ValueError("relative_tolerance must be less than 1")

    @property
    def absolute_tolerance(self) -> ComponentwiseAtol:
        return ComponentwiseAtol(
            carrier_fraction=self.carrier_atol_fraction,
            ion_fraction=self.ion_atol_fraction,
            interface_fraction=self.interface_atol_fraction,
            minimum_atol=self.minimum_atol,
            refinement_factor=self.atol_refinement_factor,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    def canonical_json(self) -> str:
        return _canonical_json(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("electrothermal J-V protocol must be a mapping")
        _exact_keys(payload, cls, "electrothermal J-V protocol")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ElectrothermalOperatingPointProtocol:
    """Frozen root and operating-point selection contract."""

    temperature_absolute_tolerance_K: float = 1.0e-8
    maximum_root_iterations: int = 50
    operating_branch: Literal["forward", "reverse"] = "forward"
    schema_version: Literal[1] = 1
    temperature_solver: Literal["brentq"] = "brentq"
    operating_point: Literal["sampled_terminal_maximum_power"] = (
        "sampled_terminal_maximum_power"
    )
    circuit_control_volume: Literal["device_plus_lumped_parasitics"] = (
        "device_plus_lumped_parasitics"
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("electrothermal operating schema_version must be 1")
        if self.temperature_solver != "brentq":
            raise ValueError("electrothermal temperature_solver must be brentq")
        if self.operating_point != "sampled_terminal_maximum_power":
            raise ValueError("unsupported electrothermal operating point")
        if self.circuit_control_volume != "device_plus_lumped_parasitics":
            raise ValueError("electrothermal control volume must include parasitics")
        if self.operating_branch not in ("forward", "reverse"):
            raise ValueError("operating_branch must be forward or reverse")
        object.__setattr__(
            self,
            "temperature_absolute_tolerance_K",
            _positive(
                self.temperature_absolute_tolerance_K,
                "temperature_absolute_tolerance_K",
            ),
        )
        object.__setattr__(
            self,
            "maximum_root_iterations",
            _integer(
                self.maximum_root_iterations,
                "maximum_root_iterations",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    def canonical_json(self) -> str:
        return _canonical_json(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("electrothermal operating protocol must be a mapping")
        _exact_keys(payload, cls, "electrothermal operating protocol")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ElectrothermalTemperatureEvaluation:
    """One complete electrical and thermal residual evaluation."""

    temperature_K: float
    terminal_voltage_V: float
    terminal_current_A_m2: float
    terminal_power_W_m2: float
    power_balance_residual_W_m2: float
    source_experiment_protocol: ExperimentProtocol
    source_result_sha256: str
    source_experiment_protocol_sha256: str
    external_mapping_sha256: str
    source_certified: bool
    external_circuit_certified: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temperature_K",
            _positive(self.temperature_K, "temperature_K"),
        )
        for name in (
            "terminal_voltage_V",
            "terminal_current_A_m2",
            "terminal_power_W_m2",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        residual = _finite(
            self.power_balance_residual_W_m2,
            "power_balance_residual_W_m2",
        )
        object.__setattr__(self, "power_balance_residual_W_m2", residual)
        if self.terminal_power_W_m2 != (
            self.terminal_voltage_V * self.terminal_current_A_m2
        ):
            raise ValueError("terminal power must equal terminal voltage times current")
        if not isinstance(self.source_experiment_protocol, ExperimentProtocol):
            raise TypeError("source_experiment_protocol must be an ExperimentProtocol")
        for name in (
            "source_result_sha256",
            "source_experiment_protocol_sha256",
            "external_mapping_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.source_experiment_protocol_sha256 != (
            self.source_experiment_protocol.sha256
        ):
            raise ValueError("source experiment protocol hash does not match protocol")
        for name in ("source_certified", "external_circuit_certified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


def _sampled_terminal_mpp(
    result: ExternalCircuitJVResult,
    branch_name: Literal["forward", "reverse"],
) -> tuple[float, float, float]:
    if not result.certified:
        raise ElectrothermalSourceError(
            "electrothermal coupling requires a certified external J-V result"
        )
    branch: ExternalCircuitBranch = getattr(result, branch_name)
    metrics = getattr(result, f"metrics_{'fwd' if branch_name == 'forward' else 'rev'}")
    if not metrics.voc_bracketed:
        raise ElectrothermalSourceError(
            f"{branch_name} terminal branch does not bracket open circuit"
        )
    voltage = branch.terminal_voltage_V
    current = branch.terminal_current_A_m2
    power = branch.terminal_power_W_m2
    mask = (voltage >= 0.0) & (voltage <= metrics.V_oc)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ElectrothermalSourceError(
            f"{branch_name} terminal branch has no power-quadrant sample"
        )
    index = int(indices[int(np.argmax(power[indices]))])
    selected_power = float(power[index])
    if not math.isfinite(selected_power) or selected_power <= 0.0:
        raise ElectrothermalSourceError("terminal maximum power must be positive")
    metric_power = float(metrics.PCE) * result.incident_power_W_m2
    if not math.isclose(
        selected_power,
        metric_power,
        rel_tol=4.0 * np.finfo(float).eps,
        abs_tol=4.0 * np.finfo(float).eps,
    ):
        raise ElectrothermalSourceError(
            "terminal maximum-power sample does not match branch metrics"
        )
    return float(voltage[index]), float(current[index]), selected_power


def _result_mapping_sha256(
    *,
    base_stack_sha256: str,
    thermal_protocol_sha256: str,
    circuit_protocol_sha256: str,
    electrical_protocol_sha256: str,
    operating_protocol_sha256: str,
    evaluations: tuple[ElectrothermalTemperatureEvaluation, ...],
    final_external_mapping_sha256: str,
) -> str:
    document = {
        "base_stack_sha256": base_stack_sha256,
        "circuit_protocol_sha256": circuit_protocol_sha256,
        "electrical_protocol_sha256": electrical_protocol_sha256,
        "evaluations": _json_ready(evaluations),
        "final_external_mapping_sha256": final_external_mapping_sha256,
        "operating_protocol_sha256": operating_protocol_sha256,
        "schema": "electrothermal-operating-point-mapping-v1",
        "thermal_protocol_sha256": thermal_protocol_sha256,
    }
    return hashlib.sha256(_canonical_json(document).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ElectrothermalOperatingPointResult:
    """Certified coupled root with complete temperature-evaluation provenance."""

    operating_temperature_K: float
    temperature_rise_K: float
    terminal_voltage_V: float
    terminal_current_A_m2: float
    terminal_power_W_m2: float
    power_balance_residual_W_m2: float
    thermal_ledger: ThermalPowerLedger
    final_external_result: ExternalCircuitJVResult
    temperature_evaluations: tuple[ElectrothermalTemperatureEvaluation, ...]
    electrical_evaluations: int
    root_iterations: int
    base_stack_sha256: str
    thermal_protocol: LumpedThermalProtocol
    circuit_protocol: ExternalCircuitProtocol
    electrical_protocol: ElectrothermalJVProtocol
    operating_protocol: ElectrothermalOperatingPointProtocol
    thermal_protocol_sha256: str
    circuit_protocol_sha256: str
    electrical_protocol_sha256: str
    operating_protocol_sha256: str
    mapping_sha256: str
    certified: bool

    def __post_init__(self) -> None:
        temperature = _positive(
            self.operating_temperature_K,
            "operating_temperature_K",
        )
        rise = _nonnegative(self.temperature_rise_K, "temperature_rise_K")
        for name in (
            "terminal_voltage_V",
            "terminal_current_A_m2",
            "terminal_power_W_m2",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        residual = _finite(
            self.power_balance_residual_W_m2,
            "power_balance_residual_W_m2",
        )
        object.__setattr__(self, "power_balance_residual_W_m2", residual)
        if not isinstance(self.thermal_protocol, LumpedThermalProtocol):
            raise TypeError("thermal_protocol must be a LumpedThermalProtocol")
        if not isinstance(self.circuit_protocol, ExternalCircuitProtocol):
            raise TypeError("circuit_protocol must be an ExternalCircuitProtocol")
        if not isinstance(self.electrical_protocol, ElectrothermalJVProtocol):
            raise TypeError("electrical_protocol must be ElectrothermalJVProtocol")
        if not isinstance(
            self.operating_protocol,
            ElectrothermalOperatingPointProtocol,
        ):
            raise TypeError(
                "operating_protocol must be ElectrothermalOperatingPointProtocol"
            )
        if not isinstance(self.thermal_ledger, ThermalPowerLedger):
            raise TypeError("thermal_ledger must be a ThermalPowerLedger")
        if not isinstance(self.final_external_result, ExternalCircuitJVResult):
            raise TypeError("final_external_result must be ExternalCircuitJVResult")
        evaluations = tuple(self.temperature_evaluations)
        if not evaluations or not all(
            isinstance(item, ElectrothermalTemperatureEvaluation)
            for item in evaluations
        ):
            raise TypeError("temperature_evaluations must contain evaluations")
        if len({item.temperature_K.hex() for item in evaluations}) != len(evaluations):
            raise ValueError("temperature evaluations must have unique temperatures")
        object.__setattr__(self, "temperature_evaluations", evaluations)
        evaluation_count = _integer(
            self.electrical_evaluations,
            "electrical_evaluations",
            minimum=1,
        )
        if evaluation_count != len(evaluations):
            raise ValueError("electrical_evaluations does not match evaluation trace")
        object.__setattr__(self, "electrical_evaluations", evaluation_count)
        root_iterations = _integer(
            self.root_iterations,
            "root_iterations",
            minimum=0,
        )
        if root_iterations > self.operating_protocol.maximum_root_iterations:
            raise ValueError("root_iterations exceeds the operating protocol")
        object.__setattr__(self, "root_iterations", root_iterations)

        hashes = {
            "base_stack_sha256": self.base_stack_sha256,
            "thermal_protocol_sha256": self.thermal_protocol_sha256,
            "circuit_protocol_sha256": self.circuit_protocol_sha256,
            "electrical_protocol_sha256": self.electrical_protocol_sha256,
            "operating_protocol_sha256": self.operating_protocol_sha256,
        }
        for name, value in hashes.items():
            object.__setattr__(self, name, _sha256(value, name))
        expected_hashes = {
            "thermal_protocol_sha256": self.thermal_protocol.sha256,
            "circuit_protocol_sha256": self.circuit_protocol.sha256,
            "electrical_protocol_sha256": self.electrical_protocol.sha256,
            "operating_protocol_sha256": self.operating_protocol.sha256,
        }
        for name, expected in expected_hashes.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} does not match its protocol")
        if self.final_external_result.circuit_protocol_sha256 != (
            self.circuit_protocol_sha256
        ):
            raise ValueError("final external result uses a different circuit protocol")
        if self.final_external_result.incident_power_W_m2 != (
            self.electrical_protocol.incident_power_W_m2
        ):
            raise ValueError("final external result uses a different incident power")

        for item in evaluations:
            if not (
                self.thermal_protocol.ambient_temperature_K
                <= item.temperature_K
                <= self.thermal_protocol.maximum_temperature_K
            ):
                raise ValueError("temperature evaluation lies outside the envelope")
            expected_ledger = thermal_power_ledger(
                self.thermal_protocol,
                temperature_K=item.temperature_K,
                terminal_electrical_export_W_m2=item.terminal_power_W_m2,
            )
            if item.power_balance_residual_W_m2 != expected_ledger.net_heating_W_m2:
                raise ValueError("evaluation residual does not match thermal protocol")
        final_matches = [item for item in evaluations if item.temperature_K == temperature]
        if len(final_matches) != 1:
            raise ValueError("exactly one evaluation must match the operating temperature")
        final = final_matches[0]
        for name in (
            "terminal_voltage_V",
            "terminal_current_A_m2",
            "terminal_power_W_m2",
            "power_balance_residual_W_m2",
        ):
            if getattr(self, name) != getattr(final, name):
                raise ValueError(f"{name} does not match the final evaluation")
        if final.external_mapping_sha256 != self.final_external_result.mapping_sha256:
            raise ValueError("final evaluation does not match external result mapping")
        if final.source_result_sha256 != self.final_external_result.source_result_sha256:
            raise ValueError("final evaluation does not match external source result")
        if final.source_experiment_protocol_sha256 != (
            self.final_external_result.source_experiment_protocol_sha256
        ):
            raise ValueError("final evaluation does not match source protocol")
        expected_mpp = _sampled_terminal_mpp(
            self.final_external_result,
            self.operating_protocol.operating_branch,
        )
        if expected_mpp != (
            self.terminal_voltage_V,
            self.terminal_current_A_m2,
            self.terminal_power_W_m2,
        ):
            raise ValueError("operating point does not match final terminal MPP")
        expected_ledger = thermal_power_ledger(
            self.thermal_protocol,
            temperature_K=temperature,
            terminal_electrical_export_W_m2=self.terminal_power_W_m2,
        )
        if self.thermal_ledger != expected_ledger:
            raise ValueError("thermal ledger does not match the operating point")
        if rise != temperature - self.thermal_protocol.ambient_temperature_K:
            raise ValueError("temperature rise does not match ambient temperature")
        expected_mapping = _result_mapping_sha256(
            base_stack_sha256=self.base_stack_sha256,
            thermal_protocol_sha256=self.thermal_protocol_sha256,
            circuit_protocol_sha256=self.circuit_protocol_sha256,
            electrical_protocol_sha256=self.electrical_protocol_sha256,
            operating_protocol_sha256=self.operating_protocol_sha256,
            evaluations=evaluations,
            final_external_mapping_sha256=self.final_external_result.mapping_sha256,
        )
        if _sha256(self.mapping_sha256, "mapping_sha256") != expected_mapping:
            raise ValueError("mapping_sha256 does not match electrothermal evidence")
        expected_certified = (
            self.final_external_result.certified
            and all(
                item.source_certified and item.external_circuit_certified
                for item in evaluations
            )
            and temperature <= self.thermal_protocol.maximum_temperature_K
            and abs(residual)
            <= self.thermal_protocol.steady_power_residual_tolerance_W_m2
        )
        if not isinstance(self.certified, bool) or self.certified != expected_certified:
            raise ValueError("certified flag does not match electrothermal evidence")
        object.__setattr__(self, "operating_temperature_K", temperature)
        object.__setattr__(self, "temperature_rise_K", rise)


def _stack_sha256(stack: DeviceStack) -> str:
    return hashlib.sha256(repr(stack).encode("utf-8")).hexdigest()


def solve_electrothermal_operating_point(
    stack: DeviceStack,
    thermal_protocol: LumpedThermalProtocol,
    circuit_protocol: ExternalCircuitProtocol,
    electrical_protocol: ElectrothermalJVProtocol,
    operating_protocol: ElectrothermalOperatingPointProtocol,
) -> ElectrothermalOperatingPointResult:
    """Solve a bounded protocol-conditioned terminal-MPP thermal root."""

    if not isinstance(stack, DeviceStack):
        raise TypeError("stack must be a DeviceStack")
    if not isinstance(thermal_protocol, LumpedThermalProtocol):
        raise TypeError("thermal_protocol must be a LumpedThermalProtocol")
    if not isinstance(circuit_protocol, ExternalCircuitProtocol):
        raise TypeError("circuit_protocol must be an ExternalCircuitProtocol")
    if not isinstance(electrical_protocol, ElectrothermalJVProtocol):
        raise TypeError("electrical_protocol must be an ElectrothermalJVProtocol")
    if not isinstance(operating_protocol, ElectrothermalOperatingPointProtocol):
        raise TypeError(
            "operating_protocol must be ElectrothermalOperatingPointProtocol"
        )
    if not resolve_mode(stack.mode).use_temperature_scaling:
        raise ElectrothermalCapabilityError(
            "electrothermal coupling requires a temperature-scaling simulation mode"
        )

    evaluations: list[ElectrothermalTemperatureEvaluation] = []
    external_results: dict[float, ExternalCircuitJVResult] = {}

    def evaluate(temperature_K: float) -> ElectrothermalTemperatureEvaluation:
        temperature = float(temperature_K)
        if temperature in external_results:
            return next(
                item for item in evaluations if item.temperature_K == temperature
            )
        temperature_stack = replace(stack, T=temperature)
        experiment_protocol = build_jv_experiment_protocol(
            temperature_stack,
            v_rate=electrical_protocol.scan_rate_V_s,
            n_points=electrical_protocol.voltage_points_per_branch,
            V_max=electrical_protocol.voltage_max_V,
            illuminated=True,
            implicit_legacy_protocol=False,
        )
        source = run_jv_sweep(
            temperature_stack,
            N_grid=electrical_protocol.grid_points_per_electrical_layer,
            v_rate=electrical_protocol.scan_rate_V_s,
            n_points=electrical_protocol.voltage_points_per_branch,
            rtol=electrical_protocol.relative_tolerance,
            atol=electrical_protocol.absolute_tolerance,
            V_max=electrical_protocol.voltage_max_V,
            illuminated=True,
            certification_mode="strict",
            experiment_protocol=experiment_protocol,
            protocol_mode="research_strict",
            collect_numerical_diagnostics=True,
        )
        if not source.certified:
            raise ElectrothermalSourceError(
                "temperature evaluation returned an uncertified intrinsic J-V curve"
            )
        if source.protocol is None or source.protocol.sha256 != experiment_protocol.sha256:
            raise ElectrothermalSourceError(
                "temperature evaluation returned a different experiment protocol"
            )
        external = apply_external_circuit(
            source,
            circuit_protocol,
            incident_power_W_m2=electrical_protocol.incident_power_W_m2,
        )
        voltage, current, power = _sampled_terminal_mpp(
            external,
            operating_protocol.operating_branch,
        )
        ledger = thermal_power_ledger(
            thermal_protocol,
            temperature_K=temperature,
            terminal_electrical_export_W_m2=power,
        )
        if external.source_experiment_protocol_sha256 is None:
            raise ElectrothermalSourceError(
                "external mapping lost the source experiment protocol"
            )
        evaluation = ElectrothermalTemperatureEvaluation(
            temperature_K=temperature,
            terminal_voltage_V=voltage,
            terminal_current_A_m2=current,
            terminal_power_W_m2=power,
            power_balance_residual_W_m2=ledger.net_heating_W_m2,
            source_experiment_protocol=experiment_protocol,
            source_result_sha256=external.source_result_sha256,
            source_experiment_protocol_sha256=(
                external.source_experiment_protocol_sha256
            ),
            external_mapping_sha256=external.mapping_sha256,
            source_certified=source.certified,
            external_circuit_certified=external.certified,
        )
        external_results[temperature] = external
        evaluations.append(evaluation)
        return evaluation

    ambient = thermal_protocol.ambient_temperature_K
    maximum = thermal_protocol.maximum_temperature_K
    power_tolerance = thermal_protocol.steady_power_residual_tolerance_W_m2
    lower = evaluate(ambient)
    root_iterations = 0
    if abs(lower.power_balance_residual_W_m2) <= power_tolerance:
        root = ambient
    else:
        if lower.power_balance_residual_W_m2 < 0.0:
            raise ElectrothermalConvergenceError(
                "electrothermal residual is negative at ambient temperature"
            )
        upper = evaluate(maximum)
        if abs(upper.power_balance_residual_W_m2) <= power_tolerance:
            root = maximum
        else:
            if upper.power_balance_residual_W_m2 > 0.0:
                raise ElectrothermalConvergenceError(
                    "no electrothermal root exists below maximum_temperature_K"
                )
            root, root_info = brentq(
                lambda value: evaluate(value).power_balance_residual_W_m2,
                ambient,
                maximum,
                xtol=operating_protocol.temperature_absolute_tolerance_K,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=operating_protocol.maximum_root_iterations,
                full_output=True,
                disp=False,
            )
            if not root_info.converged:
                raise ElectrothermalConvergenceError(
                    "electrothermal temperature root did not converge"
                )
            root_iterations = int(root_info.iterations)

    final_evaluation = evaluate(float(root))
    if abs(final_evaluation.power_balance_residual_W_m2) > power_tolerance:
        raise ElectrothermalConvergenceError(
            "electrothermal root exceeds the registered power residual tolerance"
        )
    final_external = external_results[final_evaluation.temperature_K]
    final_ledger = thermal_power_ledger(
        thermal_protocol,
        temperature_K=final_evaluation.temperature_K,
        terminal_electrical_export_W_m2=final_evaluation.terminal_power_W_m2,
    )
    frozen_evaluations = tuple(evaluations)
    base_hash = _stack_sha256(stack)
    mapping_hash = _result_mapping_sha256(
        base_stack_sha256=base_hash,
        thermal_protocol_sha256=thermal_protocol.sha256,
        circuit_protocol_sha256=circuit_protocol.sha256,
        electrical_protocol_sha256=electrical_protocol.sha256,
        operating_protocol_sha256=operating_protocol.sha256,
        evaluations=frozen_evaluations,
        final_external_mapping_sha256=final_external.mapping_sha256,
    )
    return ElectrothermalOperatingPointResult(
        operating_temperature_K=final_evaluation.temperature_K,
        temperature_rise_K=(final_evaluation.temperature_K - ambient),
        terminal_voltage_V=final_evaluation.terminal_voltage_V,
        terminal_current_A_m2=final_evaluation.terminal_current_A_m2,
        terminal_power_W_m2=final_evaluation.terminal_power_W_m2,
        power_balance_residual_W_m2=(
            final_evaluation.power_balance_residual_W_m2
        ),
        thermal_ledger=final_ledger,
        final_external_result=final_external,
        temperature_evaluations=frozen_evaluations,
        electrical_evaluations=len(frozen_evaluations),
        root_iterations=root_iterations,
        base_stack_sha256=base_hash,
        thermal_protocol=thermal_protocol,
        circuit_protocol=circuit_protocol,
        electrical_protocol=electrical_protocol,
        operating_protocol=operating_protocol,
        thermal_protocol_sha256=thermal_protocol.sha256,
        circuit_protocol_sha256=circuit_protocol.sha256,
        electrical_protocol_sha256=electrical_protocol.sha256,
        operating_protocol_sha256=operating_protocol.sha256,
        mapping_sha256=mapping_hash,
        certified=True,
    )


__all__ = [
    "ElectrothermalCapabilityError",
    "ElectrothermalConvergenceError",
    "ElectrothermalError",
    "ElectrothermalJVProtocol",
    "ElectrothermalOperatingPointProtocol",
    "ElectrothermalOperatingPointResult",
    "ElectrothermalSourceError",
    "ElectrothermalTemperatureEvaluation",
    "solve_electrothermal_operating_point",
]
