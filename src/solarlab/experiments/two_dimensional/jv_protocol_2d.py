"""Canonical execution protocols for 2D J-V dwell sweeps and dynamic forward/reverse pairs.

Provides legacy-free protocol declarations, SHA-256 canonical JSON hashing, strict
key/type validation, ``compatibility`` vs ``research_strict`` resolution, and
explicit forward/reverse dynamic scan state-chain protocols.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from typing import Any, Literal, Self, TypeAlias

import numpy as np

from solarlab.experiments.two_dimensional.inputs import (
    ComponentwiseAtolInput,
    JV2DAtolInput,
    JV2DProtocolInput,
)

__all__ = [
    "AbsoluteToleranceSpec",
    "ComponentwiseAtolSpec",
    "DYNAMIC_JV_PAIR_2D_SCHEMA",
    "DynamicJVPairProtocol2D",
    "ImplicitProtocolError",
    "JV2DAtolProtocol",
    "JV2DGrainBoundaryProtocol",
    "JV2DProtocol",
    "JV2D_PROTOCOL_SCHEMA",
    "ProtocolError",
    "ProtocolMismatchError",
    "ProtocolMode",
    "build_jv_2d_protocol",
    "resolve_jv_2d_protocol",
]

JV2D_PROTOCOL_SCHEMA: Literal["jv-2d-execution-protocol-v1"] = "jv-2d-execution-protocol-v1"
DYNAMIC_JV_PAIR_2D_SCHEMA: Literal["jv-2d-dynamic-pair-protocol-v1"] = "jv-2d-dynamic-pair-protocol-v1"
ProtocolMode: TypeAlias = Literal["compatibility", "research_strict"]


class ProtocolError(ValueError):
    """Base error for 2D execution protocol validation or binding failures."""


class ImplicitProtocolError(ProtocolError):
    """Raised when ``research_strict`` execution lacks an explicit protocol."""


class ProtocolMismatchError(ProtocolError):
    """Raised when a supplied protocol disagrees with effective execution controls."""


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


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{field} must be positive")
    return integer


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{field} must be non-negative")
    return integer


def _strict_keys(payload: Mapping[str, Any], cls: type[Any]) -> None:
    expected = {f.name for f in dataclasses.fields(cls)}
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{cls.__name__} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _finite_tuple(values: object, field: str, *, minimum_size: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{field} must be an iterable of real numbers")
    result = tuple(_finite(v, f"{field}[{idx}]") for idx, v in enumerate(values))
    if len(result) < minimum_size:
        raise ValueError(f"{field} must contain at least {minimum_size} values")
    return result


@dataclass(frozen=True, slots=True)
class ComponentwiseAtolSpec:
    """Componentwise absolute tolerance ratios for 2D transient states."""

    carrier_fraction: float = 1.0e-4
    ion_fraction: float = 1.0e-4
    interface_fraction: float = 1.0e-4
    minimum_atol: float = 1.0e6
    refinement_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "carrier_fraction",
            "ion_fraction",
            "interface_fraction",
            "minimum_atol",
            "refinement_factor",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


AbsoluteToleranceSpec: TypeAlias = float | ComponentwiseAtolSpec


@dataclass(frozen=True, slots=True)
class JV2DAtolProtocol:
    """JSON-stable scalar or componentwise absolute-tolerance declaration."""

    mode: Literal["scalar", "componentwise"]
    scalar_atol: float | None
    carrier_fraction: float | None
    ion_fraction: float | None
    interface_fraction: float | None
    minimum_atol: float | None
    refinement_factor: float | None

    def __post_init__(self) -> None:
        component_fields = (
            "carrier_fraction",
            "ion_fraction",
            "interface_fraction",
            "minimum_atol",
            "refinement_factor",
        )
        if self.mode == "scalar":
            if self.scalar_atol is None:
                raise ValueError("scalar atol mode requires scalar_atol")
            object.__setattr__(self, "scalar_atol", _positive(self.scalar_atol, "scalar_atol"))
            if any(getattr(self, name) is not None for name in component_fields):
                raise ValueError("scalar atol mode cannot carry componentwise fields")
            return
        if self.mode != "componentwise":
            raise ValueError(f"unsupported 2D atol mode {self.mode!r}")
        if self.scalar_atol is not None:
            raise ValueError("componentwise atol mode cannot carry scalar_atol")
        for name in component_fields:
            val = getattr(self, name)
            if val is None:
                raise ValueError(f"componentwise atol mode requires {name}")
            object.__setattr__(self, name, _positive(val, name))

    @classmethod
    def from_absolute_tolerance(cls, value: object) -> Self:
        if isinstance(value, cls):
            return value
        if isinstance(value, JV2DAtolInput):
            norm = value.normalized_data()
            return cls(
                mode=norm["mode"],
                scalar_atol=norm.get("scalar_atol"),
                carrier_fraction=norm.get("carrier_fraction"),
                ion_fraction=norm.get("ion_fraction"),
                interface_fraction=norm.get("interface_fraction"),
                minimum_atol=norm.get("minimum_atol"),
                refinement_factor=norm.get("refinement_factor"),
            )
        if isinstance(value, ComponentwiseAtolInput):
            norm = value.normalized_data()
            return cls(
                mode="componentwise",
                scalar_atol=None,
                carrier_fraction=norm["carrier_fraction"],
                ion_fraction=norm["ion_fraction"],
                interface_fraction=norm["interface_fraction"],
                minimum_atol=norm["minimum_atol"],
                refinement_factor=norm["refinement_factor"],
            )
        if isinstance(value, ComponentwiseAtolSpec) or all(
            hasattr(value, attr)
            for attr in (
                "carrier_fraction",
                "ion_fraction",
                "interface_fraction",
                "minimum_atol",
                "refinement_factor",
            )
        ):
            return cls(
                mode="componentwise",
                scalar_atol=None,
                carrier_fraction=getattr(value, "carrier_fraction"),
                ion_fraction=getattr(value, "ion_fraction"),
                interface_fraction=getattr(value, "interface_fraction"),
                minimum_atol=getattr(value, "minimum_atol"),
                refinement_factor=getattr(value, "refinement_factor"),
            )
        return cls(
            mode="scalar",
            scalar_atol=_positive(value, "atol"),
            carrier_fraction=None,
            ion_fraction=None,
            interface_fraction=None,
            minimum_atol=None,
            refinement_factor=None,
        )

    def to_absolute_tolerance(self) -> AbsoluteToleranceSpec:
        if self.mode == "scalar":
            assert self.scalar_atol is not None
            return self.scalar_atol
        assert self.carrier_fraction is not None
        assert self.ion_fraction is not None
        assert self.interface_fraction is not None
        assert self.minimum_atol is not None
        assert self.refinement_factor is not None
        return ComponentwiseAtolSpec(
            carrier_fraction=self.carrier_fraction,
            ion_fraction=self.ion_fraction,
            interface_fraction=self.interface_fraction,
            minimum_atol=self.minimum_atol,
            refinement_factor=self.refinement_factor,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("JV2DAtolProtocol must be a mapping")
        _strict_keys(payload, cls)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class JV2DGrainBoundaryProtocol:
    """Physical grain-boundary geometry and SRH lifetime declaration."""

    x_position_m: float
    width_m: float
    tau_n_s: float
    tau_p_s: float
    layer_role: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_position_m", _finite(self.x_position_m, "grain boundary x_position_m")
        )
        for name in ("width_m", "tau_n_s", "tau_p_s"):
            object.__setattr__(
                self, name, _positive(getattr(self, name), f"grain boundary {name}")
            )
        if not isinstance(self.layer_role, str) or not self.layer_role.strip():
            raise ValueError("grain boundary layer_role must be non-empty")

    @classmethod
    def from_declaration(cls, value: object, *, default_layer_role: str = "absorber") -> Self:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "x_position_m" in value:
                return cls.from_dict(value)
            return cls(
                x_position_m=float(value["x_position"]),
                width_m=float(value["width"]),
                tau_n_s=float(value["tau_n"]),
                tau_p_s=float(value["tau_p"]),
                layer_role=str(value.get("layer_role", default_layer_role)),
            )
        if all(hasattr(value, k) for k in ("x_position", "width", "tau_n", "tau_p")):
            return cls(
                x_position_m=float(getattr(value, "x_position")),
                width_m=float(getattr(value, "width")),
                tau_n_s=float(getattr(value, "tau_n")),
                tau_p_s=float(getattr(value, "tau_p")),
                layer_role=str(getattr(value, "layer_role", default_layer_role)),
            )
        raise TypeError(f"unsupported grain boundary declaration {type(value)!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("JV2DGrainBoundaryProtocol must be a mapping")
        _strict_keys(payload, cls)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class JV2DProtocol:
    """Physical history, topology, mesh, and numerical policy for one 2D J-V sweep."""

    temperature_K: float
    illuminated: bool
    illumination_source: str | None
    initial_state_source: Literal[
        "one_dimensional_illuminated_finite_time",
        "one_dimensional_dark_equilibrium",
    ]
    initial_state_voltage_V: float
    initial_state_settle_s: float | None
    voltage_values_V: tuple[float, ...]
    dwell_time_per_voltage_s: float
    state_topology: Literal["frozen_ion_background", "single_positive_mobile_ion"]
    ion_boundary_condition: Literal["frozen", "blocking"]
    carrier_boundary_condition: Literal["ohmic", "selective_robin"]
    interface_srh: Literal["off", "two_sided_cross_node"]
    lateral_bc: Literal["periodic", "neumann"]
    x_coordinates_m: tuple[float, ...]
    y_coordinates_m: tuple[float, ...]
    grain_boundaries: tuple[JV2DGrainBoundaryProtocol, ...]
    current_composition: Literal[
        "electron_hole_conduction",
        "electron_hole_positive_ion_displacement",
    ]
    current_sampling: Literal["instantaneous_dwell_endpoint"]
    applied_voltage_rate_at_sampling_V_s: float
    solver_method: Literal["Radau", "backward_euler"]
    solver_rtol: float
    solver_atol: JV2DAtolProtocol
    solver_max_step_divisor: int
    max_nfev_per_solve: int
    max_bisect: int
    ion_inventory_rtol: float
    save_snapshots: bool
    implicit_legacy_protocol: bool = False
    schema_version: Literal["jv-2d-execution-protocol-v1"] = JV2D_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_K", _positive(self.temperature_K, "temperature_K"))
        if not isinstance(self.illuminated, bool):
            raise TypeError("illuminated must be boolean")
        if self.illuminated:
            if not isinstance(self.illumination_source, str) or not self.illumination_source.strip():
                raise ValueError("illuminated 2D J-V requires illumination_source")
            if self.initial_state_source != "one_dimensional_illuminated_finite_time":
                raise ValueError("illuminated 2D J-V requires illuminated 1D initial state")
            if self.initial_state_settle_s is None:
                raise ValueError("illuminated 1D initial state requires settle time")
            object.__setattr__(
                self,
                "initial_state_settle_s",
                _positive(self.initial_state_settle_s, "initial_state_settle_s"),
            )
        else:
            if self.illumination_source is not None:
                raise ValueError("dark 2D J-V cannot carry illumination_source")
            if self.initial_state_source != "one_dimensional_dark_equilibrium":
                raise ValueError("dark 2D J-V requires dark-equilibrium initial state")
            if self.initial_state_settle_s is not None:
                raise ValueError("dark-equilibrium initial state has no settle time")

        object.__setattr__(
            self,
            "initial_state_voltage_V",
            _finite(self.initial_state_voltage_V, "initial_state_voltage_V"),
        )
        if self.initial_state_voltage_V != 0.0:
            raise ValueError("2D J-V v1 initial state voltage must be 0 V")

        voltages = _finite_tuple(self.voltage_values_V, "voltage_values_V", minimum_size=1)
        if voltages[0] != 0.0 or any(right <= left for left, right in zip(voltages, voltages[1:])):
            raise ValueError("voltage_values_V must start at 0 and increase")
        object.__setattr__(self, "voltage_values_V", voltages)
        object.__setattr__(
            self,
            "dwell_time_per_voltage_s",
            _positive(self.dwell_time_per_voltage_s, "dwell_time_per_voltage_s"),
        )

        x = _finite_tuple(self.x_coordinates_m, "x_coordinates_m", minimum_size=2)
        y = _finite_tuple(self.y_coordinates_m, "y_coordinates_m", minimum_size=3)
        if any(right <= left for left, right in zip(x, x[1:])):
            raise ValueError("x_coordinates_m must be strictly increasing")
        if any(right <= left for left, right in zip(y, y[1:])):
            raise ValueError("y_coordinates_m must be strictly increasing")
        object.__setattr__(self, "x_coordinates_m", x)
        object.__setattr__(self, "y_coordinates_m", y)

        boundaries = tuple(self.grain_boundaries)
        if not all(isinstance(item, JV2DGrainBoundaryProtocol) for item in boundaries):
            raise TypeError("grain_boundaries must contain JV2DGrainBoundaryProtocol values")
        object.__setattr__(self, "grain_boundaries", boundaries)

        if self.lateral_bc not in {"periodic", "neumann"}:
            raise ValueError(f"unsupported lateral_bc {self.lateral_bc!r}")
        if boundaries and self.lateral_bc != "neumann":
            raise ValueError("grain boundaries require Neumann-x topology")
        if self.interface_srh not in {"off", "two_sided_cross_node"}:
            raise ValueError(f"unsupported interface_srh {self.interface_srh!r}")
        if self.interface_srh != "off" and self.lateral_bc != "neumann":
            raise ValueError("two-sided interface SRH requires Neumann-x topology")
        if self.carrier_boundary_condition not in {"ohmic", "selective_robin"}:
            raise ValueError("carrier_boundary_condition must be 'ohmic' or 'selective_robin'")
        if self.interface_srh != "off" and self.carrier_boundary_condition != "ohmic":
            raise ValueError("two-sided interface-SRH J-V requires ohmic contacts")

        if self.state_topology == "single_positive_mobile_ion":
            if self.ion_boundary_condition != "blocking":
                raise ValueError("mobile-ion 2D J-V requires blocking ion boundaries")
            if self.lateral_bc != "neumann":
                raise ValueError("mobile-ion 2D J-V requires Neumann-x topology")
            if self.carrier_boundary_condition != "ohmic":
                raise ValueError("mobile-ion 2D J-V requires ohmic contacts")
            expected_current = "electron_hole_positive_ion_displacement"
        elif self.state_topology == "frozen_ion_background":
            if self.ion_boundary_condition != "frozen":
                raise ValueError("frozen-ion topology requires frozen ion boundary label")
            expected_current = "electron_hole_conduction"
        else:
            raise ValueError(f"unsupported state_topology {self.state_topology!r}")
        if self.current_composition != expected_current:
            raise ValueError("current_composition does not match the declared state topology")
        if self.current_sampling != "instantaneous_dwell_endpoint":
            raise ValueError("unsupported 2D current sampling mode")
        object.__setattr__(
            self,
            "applied_voltage_rate_at_sampling_V_s",
            _finite(self.applied_voltage_rate_at_sampling_V_s, "applied_voltage_rate_at_sampling_V_s"),
        )
        if self.applied_voltage_rate_at_sampling_V_s != 0.0:
            raise ValueError("fixed-voltage dwell endpoint must have dV/dt = 0")

        if self.solver_method not in {"Radau", "backward_euler"}:
            raise ValueError("2D J-V v1 supports only 'Radau' or 'backward_euler'")
        object.__setattr__(self, "solver_rtol", _positive(self.solver_rtol, "solver_rtol"))
        if not isinstance(self.solver_atol, JV2DAtolProtocol):
            raise TypeError("solver_atol must be JV2DAtolProtocol")
        object.__setattr__(
            self,
            "solver_max_step_divisor",
            _positive_integer(self.solver_max_step_divisor, "solver_max_step_divisor"),
        )
        object.__setattr__(
            self,
            "max_nfev_per_solve",
            _positive_integer(self.max_nfev_per_solve, "max_nfev_per_solve"),
        )
        object.__setattr__(
            self,
            "max_bisect",
            _nonnegative_integer(self.max_bisect, "max_bisect"),
        )
        object.__setattr__(
            self,
            "ion_inventory_rtol",
            _nonnegative(self.ion_inventory_rtol, "ion_inventory_rtol"),
        )
        for name in ("save_snapshots", "implicit_legacy_protocol"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.schema_version != JV2D_PROTOCOL_SCHEMA:
            raise ValueError("unsupported 2D J-V protocol schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["voltage_values_V"] = list(self.voltage_values_V)
        payload["x_coordinates_m"] = list(self.x_coordinates_m)
        payload["y_coordinates_m"] = list(self.y_coordinates_m)
        payload["grain_boundaries"] = [dataclasses.asdict(item) for item in self.grain_boundaries]
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

    @property
    def sha256(self) -> str:
        return self.protocol_hash

    def as_explicit(self) -> Self:
        return dataclasses.replace(self, implicit_legacy_protocol=False)

    @classmethod
    def from_protocol_input(cls, inp: JV2DProtocolInput) -> Self:
        if not isinstance(inp, JV2DProtocolInput):
            raise TypeError("JV2DProtocolInput is required")
        return cls.from_dict(inp.normalized_data())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("2D J-V protocol must be a mapping")
        _strict_keys(payload, cls)
        values = dict(payload)
        for name in ("voltage_values_V", "x_coordinates_m", "y_coordinates_m"):
            values[name] = tuple(values[name])
        boundaries = values["grain_boundaries"]
        if not isinstance(boundaries, (list, tuple)):
            raise TypeError("grain_boundaries must be a JSON array")
        values["grain_boundaries"] = tuple(
            JV2DGrainBoundaryProtocol.from_dict(item) for item in boundaries
        )
        values["solver_atol"] = JV2DAtolProtocol.from_dict(values["solver_atol"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("2D J-V protocol JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class DynamicJVPairProtocol2D:
    """Explicit 2D preconditioning/hold + forward/reverse dynamic scan state-chain protocol."""

    temperature_K: float
    illuminated: bool
    precondition_voltage_V: float
    precondition_hold_s: float
    forward_voltages_V: tuple[float, ...]
    turnaround_hold_s: float
    reverse_voltages_V: tuple[float, ...]
    step_mode: Literal["dwell", "linear_ramp"]
    dwell_or_step_time_s: float
    state_topology: Literal["frozen_ion_background", "single_positive_mobile_ion"]
    lateral_bc: Literal["periodic", "neumann"]
    carrier_boundary_condition: Literal["ohmic", "selective_robin"]
    interface_srh: Literal["off", "two_sided_cross_node"]
    x_coordinates_m: tuple[float, ...]
    y_coordinates_m: tuple[float, ...]
    grain_boundaries: tuple[JV2DGrainBoundaryProtocol, ...]
    substeps_per_point: int = 12
    ion_inventory_rtol: float = 1.0e-9
    save_snapshots: bool = False
    schema_version: Literal["jv-2d-dynamic-pair-protocol-v1"] = DYNAMIC_JV_PAIR_2D_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_K", _positive(self.temperature_K, "temperature_K"))
        if not isinstance(self.illuminated, bool):
            raise TypeError("illuminated must be boolean")
        object.__setattr__(
            self, "precondition_voltage_V", _finite(self.precondition_voltage_V, "precondition_voltage_V")
        )
        object.__setattr__(
            self, "precondition_hold_s", _nonnegative(self.precondition_hold_s, "precondition_hold_s")
        )
        fwd = _finite_tuple(self.forward_voltages_V, "forward_voltages_V", minimum_size=2)
        rev = _finite_tuple(self.reverse_voltages_V, "reverse_voltages_V", minimum_size=2)
        fwd_diffs: np.ndarray = np.diff(np.asarray(fwd, dtype=np.float64))
        rev_diffs: np.ndarray = np.diff(np.asarray(rev, dtype=np.float64))
        if not (np.all(fwd_diffs > 0.0) or np.all(fwd_diffs < 0.0)):
            raise ValueError("forward_voltages_V must be strictly monotonic")
        if not (np.all(rev_diffs > 0.0) or np.all(rev_diffs < 0.0)):
            raise ValueError("reverse_voltages_V must be strictly monotonic")
        if (fwd_diffs[0] > 0.0) == (rev_diffs[0] > 0.0):
            raise ValueError("reverse_voltages_V must reverse the forward scan direction")
        if abs(rev[0] - fwd[-1]) > 1.0e-12:
            raise ValueError("reverse_voltages_V must start at forward_voltages_V[-1] to preserve state continuity")
        object.__setattr__(self, "forward_voltages_V", fwd)
        object.__setattr__(self, "reverse_voltages_V", rev)
        object.__setattr__(
            self, "turnaround_hold_s", _nonnegative(self.turnaround_hold_s, "turnaround_hold_s")
        )
        if self.step_mode not in {"dwell", "linear_ramp"}:
            raise ValueError("step_mode must be 'dwell' or 'linear_ramp'")
        object.__setattr__(
            self, "dwell_or_step_time_s", _positive(self.dwell_or_step_time_s, "dwell_or_step_time_s")
        )

        x = _finite_tuple(self.x_coordinates_m, "x_coordinates_m", minimum_size=2)
        y = _finite_tuple(self.y_coordinates_m, "y_coordinates_m", minimum_size=3)
        if any(right <= left for left, right in zip(x, x[1:])) or any(
            right <= left for left, right in zip(y, y[1:])
        ):
            raise ValueError("x_coordinates_m and y_coordinates_m must be strictly increasing")
        object.__setattr__(self, "x_coordinates_m", x)
        object.__setattr__(self, "y_coordinates_m", y)

        boundaries = tuple(self.grain_boundaries)
        if not all(isinstance(item, JV2DGrainBoundaryProtocol) for item in boundaries):
            raise TypeError("grain_boundaries must contain JV2DGrainBoundaryProtocol values")
        object.__setattr__(self, "grain_boundaries", boundaries)

        mobile = self.state_topology == "single_positive_mobile_ion"
        if (boundaries or mobile or self.interface_srh != "off") and self.lateral_bc != "neumann":
            raise ValueError("grain, mobile-ion, or two-sided SRH dynamic scan requires Neumann-x")
        if (mobile or self.interface_srh != "off") and self.carrier_boundary_condition != "ohmic":
            raise ValueError("mobile-ion or two-sided SRH dynamic scan requires ohmic contacts")
        object.__setattr__(
            self, "substeps_per_point", _positive_integer(self.substeps_per_point, "substeps_per_point")
        )
        object.__setattr__(
            self, "ion_inventory_rtol", _nonnegative(self.ion_inventory_rtol, "ion_inventory_rtol")
        )
        if not isinstance(self.save_snapshots, bool):
            raise TypeError("save_snapshots must be boolean")
        if self.schema_version != DYNAMIC_JV_PAIR_2D_SCHEMA:
            raise ValueError("unsupported 2D dynamic pair protocol schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["forward_voltages_V"] = list(self.forward_voltages_V)
        payload["reverse_voltages_V"] = list(self.reverse_voltages_V)
        payload["x_coordinates_m"] = list(self.x_coordinates_m)
        payload["y_coordinates_m"] = list(self.y_coordinates_m)
        payload["grain_boundaries"] = [dataclasses.asdict(item) for item in self.grain_boundaries]
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

    @property
    def sha256(self) -> str:
        return self.protocol_hash

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("DynamicJVPairProtocol2D must be a mapping")
        _strict_keys(payload, cls)
        values = dict(payload)
        for name in ("forward_voltages_V", "reverse_voltages_V", "x_coordinates_m", "y_coordinates_m"):
            values[name] = tuple(values[name])
        values["grain_boundaries"] = tuple(
            JV2DGrainBoundaryProtocol.from_dict(item) for item in values["grain_boundaries"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("DynamicJVPairProtocol2D JSON must contain an object")
        return cls.from_dict(parsed)


def build_jv_2d_protocol(
    *,
    temperature_K: float,
    illuminated: bool,
    x_coordinates_m: Iterable[float],
    y_coordinates_m: Iterable[float],
    voltages_V: Iterable[float],
    dwell_time_per_voltage_s: float,
    ion_dynamics: Literal["frozen", "single_mobile"],
    carrier_boundary_condition: Literal["ohmic", "selective_robin"],
    interface_srh: Literal["off", "two_sided_cross_node"],
    lateral_bc: Literal["periodic", "neumann"],
    solver_rtol: float,
    solver_atol: object,
    max_nfev_per_solve: int,
    max_bisect: int,
    ion_inventory_rtol: float,
    save_snapshots: bool,
    grain_boundaries: Iterable[object] = (),
    initial_state_settle_s: float = 1.0e-3,
    solver_method: Literal["Radau", "backward_euler"] = "Radau",
    solver_max_step_divisor: int = 50,
    implicit_legacy_protocol: bool = False,
) -> JV2DProtocol:
    """Build the canonical 2D J-V protocol declaration."""
    if ion_dynamics not in {"frozen", "single_mobile"}:
        raise ValueError("ion_dynamics must be 'frozen' or 'single_mobile'")
    state_topology: Literal["frozen_ion_background", "single_positive_mobile_ion"] = (
        "single_positive_mobile_ion" if ion_dynamics == "single_mobile" else "frozen_ion_background"
    )
    return JV2DProtocol(
        temperature_K=temperature_K,
        illuminated=illuminated,
        illumination_source=("stack_baseline_generation" if illuminated else None),
        initial_state_source=(
            "one_dimensional_illuminated_finite_time"
            if illuminated
            else "one_dimensional_dark_equilibrium"
        ),
        initial_state_voltage_V=0.0,
        initial_state_settle_s=(initial_state_settle_s if illuminated else None),
        voltage_values_V=tuple(float(v) for v in voltages_V),
        dwell_time_per_voltage_s=dwell_time_per_voltage_s,
        state_topology=state_topology,
        ion_boundary_condition=("blocking" if ion_dynamics == "single_mobile" else "frozen"),
        carrier_boundary_condition=carrier_boundary_condition,
        interface_srh=interface_srh,
        lateral_bc=lateral_bc,
        x_coordinates_m=tuple(float(v) for v in x_coordinates_m),
        y_coordinates_m=tuple(float(v) for v in y_coordinates_m),
        grain_boundaries=tuple(
            JV2DGrainBoundaryProtocol.from_declaration(item) for item in grain_boundaries
        ),
        current_composition=(
            "electron_hole_positive_ion_displacement"
            if ion_dynamics == "single_mobile"
            else "electron_hole_conduction"
        ),
        current_sampling="instantaneous_dwell_endpoint",
        applied_voltage_rate_at_sampling_V_s=0.0,
        solver_method=solver_method,
        solver_rtol=solver_rtol,
        solver_atol=JV2DAtolProtocol.from_absolute_tolerance(solver_atol),
        solver_max_step_divisor=solver_max_step_divisor,
        max_nfev_per_solve=max_nfev_per_solve,
        max_bisect=max_bisect,
        ion_inventory_rtol=ion_inventory_rtol,
        save_snapshots=save_snapshots,
        implicit_legacy_protocol=implicit_legacy_protocol,
    )


def resolve_jv_2d_protocol(
    supplied: JV2DProtocol | JV2DProtocolInput | None,
    expected_legacy: JV2DProtocol,
    *,
    mode: ProtocolMode,
) -> JV2DProtocol:
    """Bind a supplied 2D J-V protocol declaration to exact execution controls."""
    if mode not in {"compatibility", "research_strict"}:
        raise ValueError(f"unsupported 2D J-V protocol mode {mode!r}")
    if not isinstance(expected_legacy, JV2DProtocol):
        raise TypeError("expected_legacy must be a JV2DProtocol")
    if not expected_legacy.implicit_legacy_protocol:
        raise ValueError("expected_legacy must be marked implicit")

    converted: JV2DProtocol | None
    if isinstance(supplied, JV2DProtocolInput):
        converted = JV2DProtocol.from_protocol_input(supplied)
    else:
        converted = supplied

    resolved = expected_legacy if converted is None else converted
    if not isinstance(resolved, JV2DProtocol):
        raise TypeError("jv_2d_protocol must be a JV2DProtocol")
    if converted is not None:
        expected = expected_legacy.to_dict()
        actual = converted.to_dict()
        expected.pop("implicit_legacy_protocol")
        actual.pop("implicit_legacy_protocol")
        # Allow explicit solver_method="backward_euler" when matching native BE execution
        if actual.get("solver_method") == "backward_euler" and expected.get("solver_method") == "Radau":
            expected["solver_method"] = "backward_euler"
        mismatches = tuple(name for name in sorted(expected) if actual[name] != expected[name])
        if mismatches:
            raise ProtocolMismatchError(
                "jv_2d_protocol does not match the requested execution; "
                f"mismatched fields: {', '.join(mismatches)}"
            )
    if mode == "research_strict" and resolved.implicit_legacy_protocol:
        raise ImplicitProtocolError(
            "research_strict 2D J-V requires an explicit execution protocol"
        )
    return resolved
