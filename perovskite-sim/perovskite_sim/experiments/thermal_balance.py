"""Area-normalized lumped thermal energy balance.

The thermal control volume contains the photovoltaic device and any lumped
series/shunt elements included in its terminal curve. Absorbed optical power is
an explicit input. Energy leaving as terminal electrical power is subtracted
once; the remaining heat is rejected through a linear boundary and optional
far-field radiation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Self

import numpy as np

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover - exercised only by the minimal shim
    from perovskite_sim._compat.scipy_shim import solve_ivp

    brentq = None


STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


class ThermalBalanceError(ValueError):
    """Base class for invalid thermal contracts or failed energy balances."""


class ThermalEnergySourceError(ThermalBalanceError):
    """Raised when the declared control-volume power ledger is impossible."""


class ThermalSteadyStateError(ThermalBalanceError):
    """Raised when no bounded lumped steady temperature exists."""


class ThermalTransientError(ThermalBalanceError):
    """Raised when the lumped thermal transient cannot be certified."""


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return 0.0 if number == 0.0 else number


def _positive(value: object, field: str) -> float:
    number = _finite_real(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _nonnegative(value: object, field: str) -> float:
    number = _finite_real(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _positive_integer(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer >= {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be an integer >= {minimum}") from exc
    if integer != value or integer < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return integer


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("thermal protocol contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        f"thermal protocol contains a non-JSON value of type {type(value).__name__}"
    )


def _require_exact_keys(payload: Mapping[str, Any], cls: type, label: str) -> None:
    expected = {field.name for field in dataclasses.fields(cls)}
    actual = set(payload)
    if actual != expected:
        raise ThermalBalanceError(
            f"{label} keys do not match schema; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class LumpedThermalProtocol:
    """Physical control-volume and heat-rejection contract."""

    absorbed_optical_power_W_m2: float
    ambient_temperature_K: float
    areal_heat_capacity_J_m2_K: float
    heat_transfer_coefficient_W_m2_K: float
    emissivity: float
    maximum_temperature_K: float
    constant_internal_heat_W_m2: float = 0.0
    steady_temperature_tolerance_K: float = 1.0e-10
    steady_power_residual_tolerance_W_m2: float = 1.0e-8
    schema_version: Literal[1] = 1
    system_boundary: Literal["device_plus_lumped_parasitics"] = (
        "device_plus_lumped_parasitics"
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("lumped thermal schema_version must be 1")
        if self.system_boundary != "device_plus_lumped_parasitics":
            raise ValueError(
                "lumped thermal system_boundary must include device and parasitics"
            )
        fields = {
            "absorbed_optical_power_W_m2": _nonnegative,
            "ambient_temperature_K": _positive,
            "areal_heat_capacity_J_m2_K": _positive,
            "heat_transfer_coefficient_W_m2_K": _nonnegative,
            "constant_internal_heat_W_m2": _nonnegative,
            "maximum_temperature_K": _positive,
            "steady_temperature_tolerance_K": _positive,
            "steady_power_residual_tolerance_W_m2": _positive,
        }
        for name, validator in fields.items():
            object.__setattr__(self, name, validator(getattr(self, name), name))
        emissivity = _finite_real(self.emissivity, "emissivity")
        if emissivity < 0.0 or emissivity > 1.0:
            raise ValueError("emissivity must lie in [0, 1]")
        object.__setattr__(self, "emissivity", emissivity)
        if self.maximum_temperature_K <= self.ambient_temperature_K:
            raise ValueError("maximum_temperature_K must exceed ambient_temperature_K")

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
            raise TypeError("lumped thermal protocol must be a mapping")
        _require_exact_keys(payload, cls, "lumped thermal protocol")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ThermalIntegrationProtocol:
    """Canonical numerical contract for a constant-power thermal transient."""

    duration_s: float
    initial_temperature_K: float
    sample_count: int
    relative_tolerance: float = 1.0e-9
    absolute_temperature_tolerance_K: float = 1.0e-10
    absolute_energy_tolerance_J_m2: float = 1.0e-9
    max_step_divisor: int = 200
    energy_balance_tolerance_J_m2: float = 1.0e-7
    method: Literal["DOP853"] = "DOP853"
    schema_version: Literal[1] = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("thermal integration schema_version must be 1")
        if self.method != "DOP853":
            raise ValueError("thermal integration method must be DOP853")
        for name in (
            "duration_s",
            "initial_temperature_K",
            "relative_tolerance",
            "absolute_temperature_tolerance_K",
            "absolute_energy_tolerance_J_m2",
            "energy_balance_tolerance_J_m2",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.relative_tolerance >= 1.0:
            raise ValueError("relative_tolerance must be less than 1")
        object.__setattr__(
            self,
            "sample_count",
            _positive_integer(self.sample_count, "sample_count", minimum=2),
        )
        object.__setattr__(
            self,
            "max_step_divisor",
            _positive_integer(self.max_step_divisor, "max_step_divisor"),
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
            raise TypeError("thermal integration protocol must be a mapping")
        _require_exact_keys(payload, cls, "thermal integration protocol")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ThermalPowerLedger:
    """Instantaneous area-normalized first-law terms at one temperature."""

    temperature_K: float
    absorbed_optical_power_W_m2: float
    constant_internal_heat_W_m2: float
    terminal_electrical_export_W_m2: float
    linear_heat_rejection_W_m2: float
    radiative_heat_rejection_W_m2: float
    net_heating_W_m2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temperature_K",
            _positive(self.temperature_K, "temperature_K"),
        )
        for name in (
            "absorbed_optical_power_W_m2",
            "constant_internal_heat_W_m2",
            "terminal_electrical_export_W_m2",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "linear_heat_rejection_W_m2",
            "radiative_heat_rejection_W_m2",
            "net_heating_W_m2",
        ):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if (
            self.terminal_electrical_export_W_m2
            > self.absorbed_optical_power_W_m2
        ):
            raise ThermalEnergySourceError(
                "terminal electrical export exceeds declared absorbed optical power"
            )
        expected = (
            self.absorbed_optical_power_W_m2
            + self.constant_internal_heat_W_m2
            - self.terminal_electrical_export_W_m2
            - self.linear_heat_rejection_W_m2
            - self.radiative_heat_rejection_W_m2
        )
        if self.net_heating_W_m2 != expected:
            raise ValueError("net_heating_W_m2 violates the thermal power ledger")


def _mapping_sha256(
    protocol_sha256: str,
    electrical_output_W_m2: float,
    integration_sha256: str | None,
) -> str:
    digest = hashlib.sha256(b"lumped-thermal-energy-balance-v1")
    digest.update(protocol_sha256.encode("ascii"))
    digest.update(float(electrical_output_W_m2).hex().encode("ascii"))
    digest.update((integration_sha256 or "steady-state").encode("ascii"))
    return digest.hexdigest()


def _electrical_output(
    protocol: LumpedThermalProtocol,
    electrical_output_W_m2: object,
) -> float:
    electrical = _nonnegative(
        electrical_output_W_m2,
        "terminal_electrical_export_W_m2",
    )
    if electrical > protocol.absorbed_optical_power_W_m2:
        raise ThermalEnergySourceError(
            "terminal electrical export exceeds declared absorbed optical power"
        )
    return electrical


def thermal_power_ledger(
    protocol: LumpedThermalProtocol,
    *,
    temperature_K: object,
    terminal_electrical_export_W_m2: object,
) -> ThermalPowerLedger:
    """Evaluate every first-law term at one lumped temperature."""

    if not isinstance(protocol, LumpedThermalProtocol):
        raise TypeError("protocol must be a LumpedThermalProtocol")
    temperature = _positive(temperature_K, "temperature_K")
    electrical = _electrical_output(protocol, terminal_electrical_export_W_m2)
    delta = temperature - protocol.ambient_temperature_K
    linear = protocol.heat_transfer_coefficient_W_m2_K * delta
    radiative = (
        protocol.emissivity
        * STEFAN_BOLTZMANN_W_M2_K4
        * (temperature**4 - protocol.ambient_temperature_K**4)
    )
    net = (
        protocol.absorbed_optical_power_W_m2
        + protocol.constant_internal_heat_W_m2
        - electrical
        - linear
        - radiative
    )
    return ThermalPowerLedger(
        temperature_K=temperature,
        absorbed_optical_power_W_m2=protocol.absorbed_optical_power_W_m2,
        constant_internal_heat_W_m2=protocol.constant_internal_heat_W_m2,
        terminal_electrical_export_W_m2=electrical,
        linear_heat_rejection_W_m2=linear,
        radiative_heat_rejection_W_m2=radiative,
        net_heating_W_m2=net,
    )


@dataclass(frozen=True, slots=True)
class LumpedThermalSteadyStateResult:
    temperature_K: float
    temperature_rise_K: float
    ledger: ThermalPowerLedger
    thermal_protocol: LumpedThermalProtocol
    thermal_protocol_sha256: str
    mapping_sha256: str
    certified: bool

    def __post_init__(self) -> None:
        temperature = _positive(self.temperature_K, "temperature_K")
        rise = _nonnegative(self.temperature_rise_K, "temperature_rise_K")
        if not isinstance(self.ledger, ThermalPowerLedger):
            raise TypeError("ledger must be a ThermalPowerLedger")
        if not isinstance(self.thermal_protocol, LumpedThermalProtocol):
            raise TypeError("thermal_protocol must be a LumpedThermalProtocol")
        protocol_hash = _require_sha256(
            self.thermal_protocol_sha256,
            "thermal_protocol_sha256",
        )
        if protocol_hash != self.thermal_protocol.sha256:
            raise ValueError("thermal_protocol_sha256 does not match protocol")
        expected_mapping = _mapping_sha256(
            protocol_hash,
            self.ledger.terminal_electrical_export_W_m2,
            None,
        )
        if _require_sha256(self.mapping_sha256, "mapping_sha256") != expected_mapping:
            raise ValueError("mapping_sha256 does not match the steady-state inputs")
        expected_rise = temperature - self.thermal_protocol.ambient_temperature_K
        if rise != expected_rise:
            raise ValueError("temperature_rise_K does not match temperature")
        if self.ledger.temperature_K != temperature:
            raise ValueError("ledger temperature does not match result temperature")
        expected_ledger = thermal_power_ledger(
            self.thermal_protocol,
            temperature_K=temperature,
            terminal_electrical_export_W_m2=(
                self.ledger.terminal_electrical_export_W_m2
            ),
        )
        if self.ledger != expected_ledger:
            raise ValueError("steady-state ledger does not match protocol inputs")
        expected_certified = (
            temperature <= self.thermal_protocol.maximum_temperature_K
            and abs(self.ledger.net_heating_W_m2)
            <= self.thermal_protocol.steady_power_residual_tolerance_W_m2
        )
        if not isinstance(self.certified, bool) or self.certified != expected_certified:
            raise ValueError("certified flag does not match steady-state evidence")
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "temperature_rise_K", rise)


def solve_lumped_thermal_steady_state(
    protocol: LumpedThermalProtocol,
    *,
    terminal_electrical_export_W_m2: object,
) -> LumpedThermalSteadyStateResult:
    """Solve the unique bounded constant-power lumped thermal steady state."""

    if not isinstance(protocol, LumpedThermalProtocol):
        raise TypeError("protocol must be a LumpedThermalProtocol")
    electrical = _electrical_output(protocol, terminal_electrical_export_W_m2)
    source_heat = (
        protocol.absorbed_optical_power_W_m2
        + protocol.constant_internal_heat_W_m2
        - electrical
    )
    ambient = protocol.ambient_temperature_K
    if source_heat == 0.0:
        temperature = ambient
    else:
        if (
            protocol.heat_transfer_coefficient_W_m2_K == 0.0
            and protocol.emissivity == 0.0
        ):
            raise ThermalSteadyStateError(
                "positive heat has no declared rejection mechanism"
            )

        def residual(value: float) -> float:
            return thermal_power_ledger(
                protocol,
                temperature_K=value,
                terminal_electrical_export_W_m2=electrical,
            ).net_heating_W_m2

        upper_residual = residual(protocol.maximum_temperature_K)
        if upper_residual > 0.0:
            raise ThermalSteadyStateError(
                "no steady temperature exists below maximum_temperature_K"
            )
        if brentq is None:  # pragma: no cover - SciPy is a project dependency
            raise ThermalSteadyStateError(
                "SciPy is required for a radiative thermal steady state"
            )
        temperature = float(
            brentq(
                residual,
                ambient,
                protocol.maximum_temperature_K,
                xtol=protocol.steady_temperature_tolerance_K,
                rtol=4.0 * np.finfo(float).eps,
            )
        )
    ledger = thermal_power_ledger(
        protocol,
        temperature_K=temperature,
        terminal_electrical_export_W_m2=electrical,
    )
    certified = (
        temperature <= protocol.maximum_temperature_K
        and abs(ledger.net_heating_W_m2)
        <= protocol.steady_power_residual_tolerance_W_m2
    )
    if not certified:
        raise ThermalSteadyStateError(
            "thermal steady state does not satisfy the registered power residual"
        )
    return LumpedThermalSteadyStateResult(
        temperature_K=temperature,
        temperature_rise_K=temperature - ambient,
        ledger=ledger,
        thermal_protocol=protocol,
        thermal_protocol_sha256=protocol.sha256,
        mapping_sha256=_mapping_sha256(protocol.sha256, electrical, None),
        certified=True,
    )


def _readonly_vector(value: object, field: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite vector with at least two entries")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LumpedThermalTransientResult:
    time_s: np.ndarray
    temperature_K: np.ndarray
    net_heating_W_m2: np.ndarray
    cumulative_absorbed_energy_J_m2: np.ndarray
    cumulative_internal_energy_J_m2: np.ndarray
    cumulative_electrical_export_J_m2: np.ndarray
    cumulative_linear_rejection_J_m2: np.ndarray
    cumulative_radiative_rejection_J_m2: np.ndarray
    energy_balance_residual_J_m2: np.ndarray
    max_abs_energy_balance_residual_J_m2: float
    terminal_electrical_export_W_m2: float
    solver_nfev: int
    thermal_protocol: LumpedThermalProtocol
    integration_protocol: ThermalIntegrationProtocol
    thermal_protocol_sha256: str
    integration_protocol_sha256: str
    mapping_sha256: str
    certified: bool

    def __post_init__(self) -> None:
        vector_names = (
            "time_s",
            "temperature_K",
            "net_heating_W_m2",
            "cumulative_absorbed_energy_J_m2",
            "cumulative_internal_energy_J_m2",
            "cumulative_electrical_export_J_m2",
            "cumulative_linear_rejection_J_m2",
            "cumulative_radiative_rejection_J_m2",
            "energy_balance_residual_J_m2",
        )
        vectors = []
        for name in vector_names:
            vector = _readonly_vector(getattr(self, name), name)
            object.__setattr__(self, name, vector)
            vectors.append(vector)
        if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
            raise ValueError("thermal transient vectors must have identical shapes")
        if not np.all(np.diff(self.time_s) > 0.0) or self.time_s[0] != 0.0:
            raise ValueError("thermal transient time must start at zero and increase")
        if not isinstance(self.thermal_protocol, LumpedThermalProtocol):
            raise TypeError("thermal_protocol must be a LumpedThermalProtocol")
        if not isinstance(self.integration_protocol, ThermalIntegrationProtocol):
            raise TypeError("integration_protocol must be ThermalIntegrationProtocol")
        thermal_hash = _require_sha256(
            self.thermal_protocol_sha256,
            "thermal_protocol_sha256",
        )
        integration_hash = _require_sha256(
            self.integration_protocol_sha256,
            "integration_protocol_sha256",
        )
        if thermal_hash != self.thermal_protocol.sha256:
            raise ValueError("thermal protocol hash mismatch")
        if integration_hash != self.integration_protocol.sha256:
            raise ValueError("integration protocol hash mismatch")
        electrical = _nonnegative(
            self.terminal_electrical_export_W_m2,
            "terminal_electrical_export_W_m2",
        )
        _electrical_output(self.thermal_protocol, electrical)
        if _require_sha256(self.mapping_sha256, "mapping_sha256") != (
            _mapping_sha256(thermal_hash, electrical, integration_hash)
        ):
            raise ValueError("mapping_sha256 does not match transient inputs")
        nfev = _positive_integer(self.solver_nfev, "solver_nfev")
        object.__setattr__(self, "solver_nfev", nfev)
        if self.time_s.size != self.integration_protocol.sample_count:
            raise ValueError("thermal transient sample count does not match protocol")
        if self.time_s[-1] != self.integration_protocol.duration_s:
            raise ValueError("thermal transient duration does not match protocol")
        if self.temperature_K[0] != self.integration_protocol.initial_temperature_K:
            raise ValueError("initial temperature does not match integration protocol")
        expected_net_heating = np.asarray(
            [
                thermal_power_ledger(
                    self.thermal_protocol,
                    temperature_K=temperature,
                    terminal_electrical_export_W_m2=electrical,
                ).net_heating_W_m2
                for temperature in self.temperature_K
            ]
        )
        if not np.array_equal(self.net_heating_W_m2, expected_net_heating):
            raise ValueError("net heating trace does not match protocol inputs")
        expected_energy_residual = self.thermal_protocol.areal_heat_capacity_J_m2_K * (
            self.temperature_K - self.integration_protocol.initial_temperature_K
        ) - (
            self.cumulative_absorbed_energy_J_m2
            + self.cumulative_internal_energy_J_m2
            - self.cumulative_electrical_export_J_m2
            - self.cumulative_linear_rejection_J_m2
            - self.cumulative_radiative_rejection_J_m2
        )
        if not np.array_equal(
            self.energy_balance_residual_J_m2,
            expected_energy_residual,
        ):
            raise ValueError("energy residual trace does not match cumulative ledger")
        maximum_residual = _nonnegative(
            self.max_abs_energy_balance_residual_J_m2,
            "max_abs_energy_balance_residual_J_m2",
        )
        if maximum_residual != float(np.max(np.abs(self.energy_balance_residual_J_m2))):
            raise ValueError("maximum energy residual does not match residual trace")
        expected_certified = (
            np.max(self.temperature_K) <= self.thermal_protocol.maximum_temperature_K
            and maximum_residual
            <= self.integration_protocol.energy_balance_tolerance_J_m2
        )
        if not isinstance(self.certified, bool) or self.certified != expected_certified:
            raise ValueError("certified flag does not match transient evidence")
        object.__setattr__(
            self,
            "max_abs_energy_balance_residual_J_m2",
            maximum_residual,
        )


def run_lumped_thermal_transient(
    thermal_protocol: LumpedThermalProtocol,
    integration_protocol: ThermalIntegrationProtocol,
    *,
    terminal_electrical_export_W_m2: object,
) -> LumpedThermalTransientResult:
    """Integrate temperature and a complete cumulative first-law ledger."""

    if not isinstance(thermal_protocol, LumpedThermalProtocol):
        raise TypeError("thermal_protocol must be a LumpedThermalProtocol")
    if not isinstance(integration_protocol, ThermalIntegrationProtocol):
        raise TypeError("integration_protocol must be a ThermalIntegrationProtocol")
    initial_temperature = integration_protocol.initial_temperature_K
    if initial_temperature > thermal_protocol.maximum_temperature_K:
        raise ThermalTransientError("initial temperature exceeds maximum_temperature_K")
    electrical = _electrical_output(
        thermal_protocol,
        terminal_electrical_export_W_m2,
    )
    heat_capacity = thermal_protocol.areal_heat_capacity_J_m2_K

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        temperature = float(state[0])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermalTransientError(
                "thermal solver evaluated a non-positive temperature"
            )
        ledger = thermal_power_ledger(
            thermal_protocol,
            temperature_K=temperature,
            terminal_electrical_export_W_m2=electrical,
        )
        return np.asarray(
            [
                ledger.net_heating_W_m2 / heat_capacity,
                ledger.absorbed_optical_power_W_m2,
                ledger.constant_internal_heat_W_m2,
                ledger.terminal_electrical_export_W_m2,
                ledger.linear_heat_rejection_W_m2,
                ledger.radiative_heat_rejection_W_m2,
            ],
            dtype=float,
        )

    times = np.linspace(
        0.0,
        integration_protocol.duration_s,
        integration_protocol.sample_count,
    )
    solution = solve_ivp(
        rhs,
        (0.0, integration_protocol.duration_s),
        np.asarray([initial_temperature, 0.0, 0.0, 0.0, 0.0, 0.0]),
        method=integration_protocol.method,
        t_eval=times,
        rtol=integration_protocol.relative_tolerance,
        atol=np.asarray(
            [
                integration_protocol.absolute_temperature_tolerance_K,
                *([integration_protocol.absolute_energy_tolerance_J_m2] * 5),
            ]
        ),
        max_step=(
            integration_protocol.duration_s / integration_protocol.max_step_divisor
        ),
    )
    if not solution.success:
        raise ThermalTransientError(f"thermal integration failed: {solution.message}")
    state = np.asarray(solution.y, dtype=float)
    if state.shape != (6, integration_protocol.sample_count):
        raise ThermalTransientError("thermal integration returned an invalid shape")
    temperatures = state[0]
    if (
        not np.all(np.isfinite(state))
        or np.any(temperatures <= 0.0)
        or np.max(temperatures) > thermal_protocol.maximum_temperature_K
    ):
        raise ThermalTransientError(
            "thermal transient exceeded its finite positive temperature envelope"
        )
    net_heating = np.asarray(
        [
            thermal_power_ledger(
                thermal_protocol,
                temperature_K=temperature,
                terminal_electrical_export_W_m2=electrical,
            ).net_heating_W_m2
            for temperature in temperatures
        ]
    )
    energy_residual = heat_capacity * (temperatures - initial_temperature) - (
        state[1] + state[2] - state[3] - state[4] - state[5]
    )
    max_residual = float(np.max(np.abs(energy_residual)))
    if max_residual > integration_protocol.energy_balance_tolerance_J_m2:
        raise ThermalTransientError(
            "thermal transient does not satisfy its energy-balance tolerance"
        )
    return LumpedThermalTransientResult(
        time_s=times,
        temperature_K=temperatures,
        net_heating_W_m2=net_heating,
        cumulative_absorbed_energy_J_m2=state[1],
        cumulative_internal_energy_J_m2=state[2],
        cumulative_electrical_export_J_m2=state[3],
        cumulative_linear_rejection_J_m2=state[4],
        cumulative_radiative_rejection_J_m2=state[5],
        energy_balance_residual_J_m2=energy_residual,
        max_abs_energy_balance_residual_J_m2=max_residual,
        terminal_electrical_export_W_m2=electrical,
        solver_nfev=int(getattr(solution, "nfev", 1)),
        thermal_protocol=thermal_protocol,
        integration_protocol=integration_protocol,
        thermal_protocol_sha256=thermal_protocol.sha256,
        integration_protocol_sha256=integration_protocol.sha256,
        mapping_sha256=_mapping_sha256(
            thermal_protocol.sha256,
            electrical,
            integration_protocol.sha256,
        ),
        certified=True,
    )


__all__ = [
    "LumpedThermalProtocol",
    "LumpedThermalSteadyStateResult",
    "LumpedThermalTransientResult",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "ThermalBalanceError",
    "ThermalEnergySourceError",
    "ThermalIntegrationProtocol",
    "ThermalPowerLedger",
    "ThermalSteadyStateError",
    "ThermalTransientError",
    "run_lumped_thermal_transient",
    "solve_lumped_thermal_steady_state",
    "thermal_power_ledger",
]
