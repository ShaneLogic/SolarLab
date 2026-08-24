"""Area-normalized DC series/shunt layer for certified J-V curves.

The drift-diffusion solve produces an intrinsic device current at the junction
voltage.  This module maps that curve to external terminal coordinates without
changing contacts, material parameters, or the state-advancing experiment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Self

import numpy as np

from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    JVResult,
    compute_metrics,
)


class ExternalCircuitError(ValueError):
    """Base class for invalid circuit parameters or curve mappings."""


class ExternalCircuitSourceError(ExternalCircuitError):
    """Raised when the intrinsic J-V source is incomplete or uncertified."""


class ExternalCircuitTopologyError(ExternalCircuitError):
    """Raised when the mapped terminal curve folds or changes orientation."""


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _readonly_vector(value: object, field: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or len(array) < 2:
        raise ExternalCircuitSourceError(
            f"{field} must be a one-dimensional vector with at least two points"
        )
    if not np.all(np.isfinite(array)):
        raise ExternalCircuitSourceError(f"{field} must contain only finite values")
    array.setflags(write=False)
    return array


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("external-circuit protocol contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "external-circuit protocol contains a non-JSON value of type "
        f"{type(value).__name__}"
    )


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ExternalCircuitProtocol:
    """Canonical topology and sign contract for a DC parasitic mapping."""

    schema_version: Literal[1] = 1
    application: Literal["dc_postprocess"] = "dc_postprocess"
    current_convention: Literal["photovoltaic_output_positive"] = (
        "photovoltaic_output_positive"
    )
    series_resistance_ohm_m2: float = 0.0
    shunt_resistance_ohm_m2: float | None = None
    shunt_voltage_reference: Literal["junction"] = "junction"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("external-circuit schema_version must be 1")
        if self.application != "dc_postprocess":
            raise ValueError("external circuit supports only dc_postprocess")
        if self.current_convention != "photovoltaic_output_positive":
            raise ValueError(
                "external circuit requires photovoltaic_output_positive current"
            )
        if self.shunt_voltage_reference != "junction":
            raise ValueError("shunt voltage reference must be junction")
        series = _finite_float(
            self.series_resistance_ohm_m2,
            "series_resistance_ohm_m2",
        )
        if series < 0.0:
            raise ValueError("series_resistance_ohm_m2 must be non-negative")
        object.__setattr__(self, "series_resistance_ohm_m2", series)
        shunt = self.shunt_resistance_ohm_m2
        if shunt is not None:
            shunt = _finite_float(shunt, "shunt_resistance_ohm_m2")
            if shunt <= 0.0:
                raise ValueError("shunt_resistance_ohm_m2 must be positive")
            object.__setattr__(self, "shunt_resistance_ohm_m2", shunt)

    @property
    def zero_coupling(self) -> bool:
        return (
            self.series_resistance_ohm_m2 == 0.0
            and self.shunt_resistance_ohm_m2 is None
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

    @property
    def sha256(self) -> str:
        return self.protocol_hash

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("external-circuit protocol must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(payload)
        if actual != expected:
            raise ExternalCircuitError(
                "external-circuit protocol keys do not match schema; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ExternalCircuitBranch:
    """One intrinsic branch mapped into terminal voltage/current coordinates."""

    junction_voltage_V: np.ndarray
    device_current_A_m2: np.ndarray
    shunt_current_A_m2: np.ndarray
    terminal_current_A_m2: np.ndarray
    series_voltage_drop_V: np.ndarray
    terminal_voltage_V: np.ndarray
    terminal_power_W_m2: np.ndarray
    orientation: Literal["ascending", "descending"]
    max_current_balance_error_A_m2: float
    max_voltage_balance_error_V: float

    def __post_init__(self) -> None:
        names = (
            "junction_voltage_V",
            "device_current_A_m2",
            "shunt_current_A_m2",
            "terminal_current_A_m2",
            "series_voltage_drop_V",
            "terminal_voltage_V",
            "terminal_power_W_m2",
        )
        arrays = []
        for name in names:
            array = _readonly_vector(getattr(self, name), name)
            object.__setattr__(self, name, array)
            arrays.append(array)
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays[1:]):
            raise ExternalCircuitSourceError(
                "all external-circuit branch arrays must have the same shape"
            )
        expected_power = self.terminal_voltage_V * self.terminal_current_A_m2
        if not np.array_equal(self.terminal_power_W_m2, expected_power):
            raise ValueError("terminal_power_W_m2 violates V_terminal * J_terminal")
        if self.orientation not in ("ascending", "descending"):
            raise ValueError(f"unknown branch orientation {self.orientation!r}")
        expected_errors = {
            "max_current_balance_error_A_m2": float(
                np.max(
                    np.abs(
                        self.device_current_A_m2
                        - self.shunt_current_A_m2
                        - self.terminal_current_A_m2
                    )
                )
            ),
            "max_voltage_balance_error_V": float(
                np.max(
                    np.abs(
                        self.junction_voltage_V
                        - self.series_voltage_drop_V
                        - self.terminal_voltage_V
                    )
                )
            ),
        }
        for name, expected in expected_errors.items():
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if value != expected:
                raise ValueError(f"{name} does not match branch arrays")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ExternalCircuitJVResult:
    """Circuit-adjusted branches with their intrinsic source provenance."""

    forward: ExternalCircuitBranch
    reverse: ExternalCircuitBranch
    metrics_fwd: JVMetrics
    metrics_rev: JVMetrics
    hysteresis_index: float
    circuit_protocol: ExternalCircuitProtocol
    circuit_protocol_sha256: str
    source_result_sha256: str
    source_experiment_protocol_sha256: str | None
    incident_power_W_m2: float
    mapping_sha256: str
    source_certified: bool
    mapping_certified: bool
    certified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.forward, ExternalCircuitBranch) or not isinstance(
            self.reverse, ExternalCircuitBranch
        ):
            raise TypeError("forward and reverse must be ExternalCircuitBranch values")
        if not isinstance(self.metrics_fwd, JVMetrics) or not isinstance(
            self.metrics_rev, JVMetrics
        ):
            raise TypeError("metrics_fwd and metrics_rev must be JVMetrics values")
        if not isinstance(self.circuit_protocol, ExternalCircuitProtocol):
            raise TypeError("circuit_protocol must be an ExternalCircuitProtocol")
        object.__setattr__(
            self,
            "hysteresis_index",
            _finite_float(self.hysteresis_index, "hysteresis_index"),
        )
        circuit_hash = _require_sha256(
            self.circuit_protocol_sha256,
            "circuit_protocol_sha256",
        )
        if circuit_hash != self.circuit_protocol.sha256:
            raise ValueError("circuit_protocol_sha256 does not match circuit_protocol")
        source_hash = _require_sha256(
            self.source_result_sha256,
            "source_result_sha256",
        )
        if self.source_experiment_protocol_sha256 is not None:
            _require_sha256(
                self.source_experiment_protocol_sha256,
                "source_experiment_protocol_sha256",
            )
        power = _finite_float(self.incident_power_W_m2, "incident_power_W_m2")
        if power <= 0.0:
            raise ValueError("incident_power_W_m2 must be positive")
        object.__setattr__(self, "incident_power_W_m2", power)
        mapping_hash = _require_sha256(self.mapping_sha256, "mapping_sha256")
        if mapping_hash != _mapping_sha256(circuit_hash, source_hash, power):
            raise ValueError("mapping_sha256 does not match source/circuit/power")
        for name in ("source_certified", "mapping_certified", "certified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        expected_mapping = all(
            branch.max_current_balance_error_A_m2 == 0.0
            and branch.max_voltage_balance_error_V == 0.0
            for branch in (self.forward, self.reverse)
        )
        if self.mapping_certified != expected_mapping:
            raise ValueError(
                "mapping_certified must match the branch balance evidence"
            )
        expected = self.source_certified and self.mapping_certified
        if self.certified != expected:
            raise ValueError(
                "external-circuit certified flag must equal source AND mapping"
            )
        for branch in (self.forward, self.reverse):
            if self.circuit_protocol.shunt_resistance_ohm_m2 is None:
                expected_shunt = np.zeros_like(branch.junction_voltage_V)
            else:
                expected_shunt = (
                    branch.junction_voltage_V
                    / self.circuit_protocol.shunt_resistance_ohm_m2
                )
            expected_drop = (
                branch.terminal_current_A_m2
                * self.circuit_protocol.series_resistance_ohm_m2
            )
            if not np.array_equal(branch.shunt_current_A_m2, expected_shunt):
                raise ValueError("branch shunt current violates circuit protocol")
            if not np.array_equal(branch.series_voltage_drop_V, expected_drop):
                raise ValueError("branch series drop violates circuit protocol")


def _orientation(voltage: np.ndarray, field: str) -> Literal["ascending", "descending"]:
    difference = np.diff(voltage)
    if np.all(difference > 0.0):
        return "ascending"
    if np.all(difference < 0.0):
        return "descending"
    raise ExternalCircuitTopologyError(f"{field} must be strictly monotonic")


def map_external_circuit_branch(
    junction_voltage_V: object,
    device_current_A_m2: object,
    protocol: ExternalCircuitProtocol,
) -> ExternalCircuitBranch:
    """Map one branch using a junction shunt followed by a series resistor.

    With photovoltaic output current positive,

    ``J_terminal = J_device - V_junction / R_shunt`` and
    ``V_terminal = V_junction - J_terminal * R_series``.
    """

    if not isinstance(protocol, ExternalCircuitProtocol):
        raise TypeError("protocol must be an ExternalCircuitProtocol")
    voltage = _readonly_vector(junction_voltage_V, "junction_voltage_V")
    current = _readonly_vector(device_current_A_m2, "device_current_A_m2")
    if voltage.shape != current.shape:
        raise ExternalCircuitSourceError(
            "junction voltage and device current must have the same shape"
        )
    orientation = _orientation(voltage, "junction_voltage_V")

    if protocol.shunt_resistance_ohm_m2 is None:
        shunt_current = np.zeros_like(voltage)
    else:
        shunt_current = voltage / protocol.shunt_resistance_ohm_m2
    terminal_current = current - shunt_current
    series_drop = terminal_current * protocol.series_resistance_ohm_m2
    terminal_voltage = voltage - series_drop
    terminal_orientation = _orientation(terminal_voltage, "terminal_voltage_V")
    if terminal_orientation != orientation:
        raise ExternalCircuitTopologyError(
            "external-circuit mapping changed branch orientation"
        )
    terminal_power = terminal_voltage * terminal_current

    current_balance = current - shunt_current - terminal_current
    voltage_balance = voltage - series_drop - terminal_voltage
    return ExternalCircuitBranch(
        junction_voltage_V=voltage,
        device_current_A_m2=current,
        shunt_current_A_m2=shunt_current,
        terminal_current_A_m2=terminal_current,
        series_voltage_drop_V=series_drop,
        terminal_voltage_V=terminal_voltage,
        terminal_power_W_m2=terminal_power,
        orientation=orientation,
        max_current_balance_error_A_m2=float(np.max(np.abs(current_balance))),
        max_voltage_balance_error_V=float(np.max(np.abs(voltage_balance))),
    )


def _source_result_sha256(result: JVResult) -> str:
    digest = hashlib.sha256(b"external-circuit-jv-source-v1")
    for name in ("V_fwd", "J_fwd", "V_rev", "J_rev"):
        array = np.ascontiguousarray(np.asarray(getattr(result, name), dtype=np.float64))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    source_protocol = result.protocol
    if source_protocol is not None:
        digest.update(source_protocol.sha256.encode("ascii"))
    result_metadata = {
        "metrics_fwd": _json_ready(result.metrics_fwd),
        "metrics_rev": _json_ready(result.metrics_rev),
        "hysteresis_index": _finite_float(
            result.hysteresis_index,
            "source hysteresis_index",
        ),
        "certified": result.certified,
    }
    digest.update(
        json.dumps(
            result_metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    return digest.hexdigest()


def _mapping_sha256(
    circuit_protocol_sha256: str,
    source_result_sha256: str,
    incident_power_W_m2: float,
) -> str:
    digest = hashlib.sha256(b"external-circuit-dc-mapping-v1")
    digest.update(circuit_protocol_sha256.encode("ascii"))
    digest.update(source_result_sha256.encode("ascii"))
    digest.update(float(incident_power_W_m2).hex().encode("ascii"))
    digest.update(b"sampled-jv-metrics-v1")
    return digest.hexdigest()


def apply_external_circuit(
    result: JVResult,
    protocol: ExternalCircuitProtocol,
    *,
    incident_power_W_m2: float = 1000.0,
) -> ExternalCircuitJVResult:
    """Apply a DC circuit map to a complete intrinsic forward/reverse result."""

    if not isinstance(result, JVResult):
        raise TypeError("result must be a JVResult")
    source_certified = result.certified
    if not source_certified:
        raise ExternalCircuitSourceError(
            "external-circuit mapping requires a certified intrinsic JVResult"
        )
    power = _finite_float(incident_power_W_m2, "incident_power_W_m2")
    if power <= 0.0:
        raise ValueError("incident_power_W_m2 must be positive")

    forward = map_external_circuit_branch(result.V_fwd, result.J_fwd, protocol)
    reverse = map_external_circuit_branch(result.V_rev, result.J_rev, protocol)
    preserve_source_metrics = protocol.zero_coupling and power == 1000.0
    if preserve_source_metrics:
        metrics_fwd = result.metrics_fwd
        metrics_rev = result.metrics_rev
        hysteresis = result.hysteresis_index
    else:
        metrics_fwd = compute_metrics(
            forward.terminal_voltage_V,
            forward.terminal_current_A_m2,
            P_in=power,
        )
        metrics_rev = compute_metrics(
            reverse.terminal_voltage_V,
            reverse.terminal_current_A_m2,
            P_in=power,
        )
        hysteresis = (
            0.0
            if metrics_rev.PCE == 0.0
            else (metrics_rev.PCE - metrics_fwd.PCE) / metrics_rev.PCE
        )
    source_result_sha256 = _source_result_sha256(result)
    circuit_protocol_sha256 = protocol.sha256
    source_protocol = result.protocol
    source_protocol_sha256 = (
        None if source_protocol is None else source_protocol.sha256
    )
    mapping_certified = (
        forward.max_current_balance_error_A_m2 == 0.0
        and forward.max_voltage_balance_error_V == 0.0
        and reverse.max_current_balance_error_A_m2 == 0.0
        and reverse.max_voltage_balance_error_V == 0.0
    )
    return ExternalCircuitJVResult(
        forward=forward,
        reverse=reverse,
        metrics_fwd=metrics_fwd,
        metrics_rev=metrics_rev,
        hysteresis_index=float(hysteresis),
        circuit_protocol=protocol,
        circuit_protocol_sha256=circuit_protocol_sha256,
        source_result_sha256=source_result_sha256,
        source_experiment_protocol_sha256=source_protocol_sha256,
        incident_power_W_m2=power,
        mapping_sha256=_mapping_sha256(
            circuit_protocol_sha256,
            source_result_sha256,
            power,
        ),
        source_certified=source_certified,
        mapping_certified=mapping_certified,
        certified=source_certified and mapping_certified,
    )


__all__ = [
    "ExternalCircuitBranch",
    "ExternalCircuitError",
    "ExternalCircuitJVResult",
    "ExternalCircuitProtocol",
    "ExternalCircuitSourceError",
    "ExternalCircuitTopologyError",
    "apply_external_circuit",
    "map_external_circuit_branch",
]
