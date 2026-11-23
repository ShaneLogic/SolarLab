"""Compensated evaluation of the local two-sided carrier constitutive law.

Built strictly on ``solarlab_research.precision.double_double`` and the public
``solarlab.physics.statistics.fermi_dirac`` table, with zero imports from
``perovskite_sim``. Missing constitutive branches raise
``UnsupportedCompensatedInterfaceDomain`` rather than falling back to binary64.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import ELEMENTARY_CHARGE_C, frozen_array
from solarlab.physics.statistics.fermi_dirac import fermi_dirac_half_table
from solarlab_research.precision.double_double import DD

__all__ = [
    "CompensatedCarrierBalance",
    "CompensatedCarrierTangent",
    "FrozenLocalCoefficients",
    "LocalCarrierEvaluation",
    "SCHEMA",
    "SCOPE",
    "UnsupportedCompensatedInterfaceDomain",
    "default_fd_table",
    "evaluate_local_carrier_pair",
    "freeze_coefficients",
    "to_float_tangent",
]

SCHEMA = "TwoSidedInterfaceCompensatedV1"
SCOPE = "bounded_local_constitutive_evaluation_not_a_nonlinear_solve"
_ZERO = DD(0)
_ONE = DD(1)
_HALF = DD(0.5)
_SMALL_B = DD(1e-8)
_DERIVATIVE_SERIES_LIMIT = DD(1e-3)
_MAX_EXP_ARGUMENT = DD(672)
_RECIPROCAL = tuple(_ONE / order for order in range(1, 14))
_RECIPROCAL_SQUARE = tuple(_ONE / (order * order) for order in range(1, 13))
_B_DERIVATIVE_COEFFICIENTS = (
    _ONE / 6,
    -_ONE / 180,
    _ONE / 5040,
    -_ONE / 151200,
    _ONE / 4790016,
)


class UnsupportedCompensatedInterfaceDomain(ArithmeticError):
    """Raised when an interface state lies outside the compensated constitutive domain."""


def _freeze_value(value: Any) -> Any:
    if isinstance(value, DD):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class LocalCarrierEvaluation(Mapping[str, object]):
    """Immutable compensated local carrier balance, tangent, and audit metadata."""

    balance: Mapping[str, DD]
    tangent: Mapping[str, DD]
    metadata: Mapping[str, object]
    schema: str = SCHEMA
    scope: str = SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "balance", _freeze_value(self.balance))
        object.__setattr__(self, "tangent", _freeze_value(self.tangent))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    def __getitem__(self, key: str) -> object:
        if key not in ("schema", "scope", "balance", "tangent", "metadata"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("schema", "scope", "balance", "tangent", "metadata"))

    def __len__(self) -> int:
        return 5

    def __deepcopy__(self, memo: object) -> LocalCarrierEvaluation:
        return self


@dataclass(frozen=True, slots=True)
class CompensatedCarrierBalance:
    """Explicit binary64 projection of compensated local carrier balance arrays."""

    state_m3: NDArray[np.float64]
    bulk_flux_m2_s: NDArray[np.float64]
    cross_flux_m2_s: NDArray[np.float64]
    capture_flux_m2_s: NDArray[np.float64]
    residual_m2_s: NDArray[np.float64]
    jacobian_log_state_m2_s: NDArray[np.float64]
    jacobian_trace_potential_m2_s_V: NDArray[np.float64]
    jacobian_bulk_coordinates: NDArray[np.float64]
    one_way_cross_scale_m2_s: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CompensatedCarrierTangent:
    """Explicit binary64 projection of compensated local carrier balance and tangents."""

    balance: CompensatedCarrierBalance
    bulk_flux_jacobian_log_state_m2_s: NDArray[np.float64]
    bulk_flux_jacobian_trace_potential_m2_s_V: NDArray[np.float64]
    bulk_flux_jacobian_bulk_coordinates: NDArray[np.float64]
    capture_flux_jacobian_log_state_m2_s: NDArray[np.float64]
    capture_flux_occupancy_derivative_m2_s: NDArray[np.float64]
    residual_occupancy_derivative_m2_s: NDArray[np.float64]


@lru_cache(maxsize=1)
def default_fd_table() -> tuple[Mapping[str, tuple[float, ...]], str]:
    """Return the immutable Fermi-Dirac order-1/2 table and its SHA-256 digest."""
    eta, _values, log_half = fermi_dirac_half_table()
    payload: dict[str, tuple[float, ...]] = {
        "eta": tuple(float(v) for v in eta),
        "log_half": tuple(float(v) for v in log_half),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return MappingProxyType(payload), hashlib.sha256(encoded).hexdigest()


def _cat(*values: DD) -> DD:
    return DD.from_normalized_parts(
        np.concatenate([np.atleast_1d(v.hi) for v in values]),
        np.concatenate([np.atleast_1d(v.lo) for v in values]),
    )


def _put(value: DD, index: Any, part: DD) -> DD:
    high, low = value.hi.copy(), value.lo.copy()
    high[index], low[index] = part.hi, part.lo
    return DD.from_normalized_parts(high, low)


def _choose(mask: NDArray[np.bool_], first: DD, second: DD) -> DD:
    return DD.from_normalized_parts(
        np.where(mask, first.hi, second.hi),
        np.where(mask, first.lo, second.lo),
    )


def _diagonal(values: DD) -> DD:
    return DD.from_normalized_parts(np.diag(values.hi), np.diag(values.lo))


def _require_dd(name: str, value: object, shape: tuple[int, ...]) -> DD:
    if not isinstance(value, DD) or value.shape != shape:
        raise TypeError(f"{name} must be DD with shape {shape!r}")
    return value


def _frozen_vector(values: ArrayLike) -> NDArray[np.float64]:
    arr = frozen_array(values)
    if arr.ndim != 1:
        raise ValueError("frozen interpolation nodes must be finite 1D vectors")
    return arr


@dataclass(frozen=True, slots=True)
class FrozenLocalCoefficients:
    """State-independent compensated coefficients for one two-sided interface."""

    physics: Mapping[str, DD]
    carrier_diffusion_over_distance: DD
    density_of_states: DD
    emission_densities: DD
    cross_prefactors: DD
    eta_nodes: DD
    log_half_nodes: DD
    inverse_log_slopes: DD
    _log_half_search_bytes: bytes = field(repr=False)
    fd_table_sha256: str

    @property
    def log_half_search(self) -> NDArray[np.float64]:
        return np.frombuffer(self._log_half_search_bytes, dtype=np.float64)

    def __deepcopy__(self, memo: object) -> FrozenLocalCoefficients:
        return self


def freeze_coefficients(
    geometry: Mapping[str, float],
    physics: Mapping[str, float],
    *,
    fd_table: Mapping[str, ArrayLike] | None = None,
    fd_table_sha256: str | None = None,
    q_C: float = ELEMENTARY_CHARGE_C,
) -> FrozenLocalCoefficients:
    """Bind exact binary64 geometry, physics, and Fermi-table coefficients."""
    if fd_table is None and fd_table_sha256 is None:
        fd_table, observed = default_fd_table()
    else:
        if (
            not isinstance(fd_table, Mapping)
            or set(fd_table) != {"eta", "log_half"}
            or not isinstance(fd_table_sha256, str)
        ):
            raise ValueError("frozen Fermi table fields differ from the declared contract")
        table_payload = {name: [float(x) for x in np.asarray(vals)] for name, vals in fd_table.items()}
        encoded = json.dumps(table_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        observed = hashlib.sha256(encoded).hexdigest()
        if observed != fd_table_sha256:
            raise ValueError("frozen Fermi table identity mismatch")
    eta = _frozen_vector(fd_table["eta"])
    log_half = _frozen_vector(fd_table["log_half"])
    if (
        eta.size != log_half.size
        or eta.size < 2
        or not np.all(np.diff(eta) > 0.0)
        or not np.all(np.diff(log_half) > 0.0)
    ):
        raise ValueError("frozen Fermi nodes must be strictly increasing and paired")
    values = {name: DD(val) for name, val in physics.items()}
    if any(val.shape != () for val in values.values()):
        raise ValueError("material coefficients must be scalar binary64 inputs")
    vt, temperature = values["thermal_voltage_V"], values["temperature_K"]
    charge = DD(q_C)
    left, right = DD(geometry["left_distance_m"]), DD(geometry["right_distance_m"])
    if (
        not bool(vt > 0)
        or not bool(temperature > 0)
        or not bool(charge > 0)
        or not bool(left > 0)
        or not bool(right > 0)
        or not bool((values["transmission"] >= 0) & (values["transmission"] <= 1))
    ):
        raise ValueError("invalid frozen local material or geometry")
    diffusion = _cat(
        values["D_n_left_m2_s"],
        values["D_p_left_m2_s"],
        values["D_n_right_m2_s"],
        values["D_p_right_m2_s"],
    )
    dos = _cat(
        values["N_C_left_m3"],
        values["N_V_left_m3"],
        values["N_C_right_m3"],
        values["N_V_right_m3"],
    )
    emission = _cat(
        values["n1_left_m3"],
        values["p1_left_m3"],
        values["n1_right_m3"],
        values["p1_right_m3"],
    )
    richardson = _cat(values["richardson_n_A_m2_K2"], values["richardson_p_A_m2_K2"])
    velocity = _cat(
        values["surface_recombination_velocity_n_m_s"],
        values["surface_recombination_velocity_p_m_s"],
    )
    if (
        np.any(diffusion < 0)
        or np.any(dos <= 0)
        or np.any(emission < 0)
        or np.any(richardson < 0)
        or np.any(velocity < 0)
    ):
        raise ValueError("invalid frozen local carrier coefficients")
    eta_dd, log_dd = DD(eta), DD(log_half)
    return FrozenLocalCoefficients(
        MappingProxyType(values),
        diffusion / _cat(left, left, right, right),
        dos,
        emission,
        values["transmission"] * richardson * temperature * temperature / charge,
        eta_dd,
        log_dd,
        (eta_dd[1:] - eta_dd[:-1]) / (log_dd[1:] - log_dd[:-1]),
        log_half.tobytes(),
        observed,
    )


def _bernoulli_values_and_derivatives(x: DD) -> tuple[DD, DD, DD, DD]:
    magnitude = abs(x)
    if np.any(magnitude > _MAX_EXP_ARGUMENT):
        raise UnsupportedCompensatedInterfaceDomain("Bernoulli argument exceeds compensated domain")
    b = DD(np.ones(x.shape, dtype=np.float64))
    derivative = DD(np.full(x.shape, -0.5, dtype=np.float64))
    quadratic = magnitude < _SMALL_B
    if np.any(quadratic):
        a = magnitude[quadratic]
        b = _put(b, quadratic, _ONE - a * _HALF + a * a * _RECIPROCAL[11])
        derivative = _put(derivative, quadratic, -_HALF + a * _RECIPROCAL[5])
    moderate = (~quadratic) & (magnitude <= 50)
    if np.any(moderate):
        a = magnitude[moderate]
        denominator = a.expm1()
        b = _put(b, moderate, a / denominator)
        stable = a < _DERIVATIVE_SERIES_LIMIT
        db = DD(np.zeros(a.shape, dtype=np.float64))
        if np.any(stable):
            z = a[stable]
            z2 = z * z
            c1, c3, c5, c7, c9 = _B_DERIVATIVE_COEFFICIENTS
            db = _put(
                db,
                stable,
                -_HALF + z * (c1 + z2 * (c3 + z2 * (c5 + z2 * (c7 + z2 * c9)))),
            )
        if np.any(~stable):
            z, den = a[~stable], denominator[~stable]
            db = _put(db, ~stable, (den - z * (den + _ONE)) / (den * den))
        derivative = _put(derivative, moderate, db)
    large = magnitude > 50
    if np.any(large):
        a = magnitude[large]
        decay = (-a).exp()
        denominator = _ONE - decay
        b = _put(b, large, a * decay / denominator)
        derivative = _put(
            derivative,
            large,
            decay * (_ONE - a - decay) / (denominator * denominator),
        )
    backward = b + magnitude
    backward_derivative = -_ONE - derivative
    positive = x >= 0
    return (
        _choose(positive, b, backward),
        _choose(positive, backward, b),
        _choose(positive, derivative, backward_derivative),
        _choose(positive, backward_derivative, derivative),
    )


def _inverse_half(
    log_ratio: DD, coefficients: FrozenLocalCoefficients
) -> tuple[DD, DD, NDArray[np.int64]]:
    if np.any(log_ratio > coefficients.log_half_nodes[-1]):
        raise UnsupportedCompensatedInterfaceDomain(
            "compensated local carrier law does not implement the Sommerfeld inverse branch"
        )
    below = log_ratio <= coefficients.log_half_nodes[0]
    eta = log_ratio.copy()
    slope = DD(np.ones(log_ratio.shape, dtype=np.float64))
    intervals: NDArray[np.int64] = np.full(log_ratio.shape, -1, dtype=np.int64)
    if np.any(~below):
        target = log_ratio[~below]
        index = np.searchsorted(coefficients.log_half_search, target.hi, side="right") - 1
        index = np.clip(index, 0, coefficients.log_half_search.size - 2)
        lower = target < coefficients.log_half_nodes[index]
        index = index - lower.astype(np.int64)
        upper = (index < coefficients.log_half_search.size - 2) & (
            target >= coefficients.log_half_nodes[index + 1]
        )
        index = index + upper.astype(np.int64)
        if (
            np.any(index < 0)
            or np.any(index >= coefficients.log_half_search.size - 1)
            or np.any(target < coefficients.log_half_nodes[index])
            or np.any(target > coefficients.log_half_nodes[index + 1])
        ):
            raise ArithmeticError("full-pair inverse-table interval selection failed")
        derivative = coefficients.inverse_log_slopes[index]
        eta = _put(
            eta,
            ~below,
            coefficients.eta_nodes[index] + (target - coefficients.log_half_nodes[index]) * derivative,
        )
        slope = _put(slope, ~below, derivative)
        intervals[~below] = index
    return eta, slope, intervals


def _series_power_mul(power: DD, activity: DD) -> DD:
    """Multiply positive Fermi-series power ``a^k`` by ``a`` without subnormal DD underflow.

    When ``|a^k * a| < 1e-290`` (with ``0 < a < exp(-4)``), ``a^k / a <= 1e-96 << 2^-106``,
    so higher series powers are strictly zero at 106-bit double-double precision.
    """
    with np.errstate(under="ignore"):
        approx = np.abs(power.hi) * np.abs(activity.hi)
    active = approx >= 1.0e-290
    if bool(np.all(active)):
        return power * activity
    if not bool(np.any(active)):
        return DD(np.zeros(power.shape, dtype=np.float64))
    safe_power = _choose(active, power, DD(np.zeros(power.shape, dtype=np.float64)))
    return safe_power * activity


def _supplies(
    state: DD, barriers: DD, coefficients: FrozenLocalCoefficients
) -> tuple[DD, DD, DD, dict[str, object]]:
    ratio = state / coefficients.density_of_states
    log_ratio = ratio.log()
    eta, inverse_slope, intervals = _inverse_half(log_ratio, coefficients)
    argument = eta - barriers / coefficients.physics["thermal_voltage_V"]
    if np.any(argument >= -4):
        raise UnsupportedCompensatedInterfaceDomain(
            "compensated local carrier law only covers the frozen twelve-term Fermi-one branch"
        )
    activity = argument.exp()
    total = DD(np.zeros(state.shape, dtype=np.float64))
    derivative = DD(np.zeros(state.shape, dtype=np.float64))
    power = activity
    for order in range(1, 13):
        sign = 1 if order % 2 else -1
        total = total + sign * power * _RECIPROCAL_SQUARE[order - 1]
        derivative = derivative + sign * power * _RECIPROCAL[order - 1]
        if order != 12:
            power = _series_power_mul(power, activity)
    old_fermi_zero = activity.log1p()
    tail_bound = _series_power_mul(power, activity) * _RECIPROCAL[12] / (_ONE - activity)
    return (
        total,
        derivative * inverse_slope,
        -derivative / coefficients.physics["thermal_voltage_V"],
        {
            "inverse_half_arguments": eta,
            "inverse_log_ratio": log_ratio,
            "inverse_intervals": intervals,
            "inverse_log_derivative": inverse_slope,
            "supply_arguments": argument,
            "polynomial_argument_derivative": derivative,
            "old_complete_F0_argument_derivative": old_fermi_zero,
            "computed_DD_old_F0_minus_polynomial_derivative": old_fermi_zero - derivative,
            "analytic_old_F0_tail_bound": tail_bound,
        },
    )


def evaluate_local_carrier_pair(
    *,
    state_density: DD,
    trace_potential: DD,
    bulk_density: DD,
    bulk_potential: DD,
    occupancy: DD,
    capture_multiplier: float | int,
    coefficients: FrozenLocalCoefficients,
) -> LocalCarrierEvaluation:
    """Evaluate four compensated local interface carrier rows and exact partials."""
    _require_dd("state_density", state_density, (4,))
    _require_dd("trace_potential", trace_potential, (2,))
    _require_dd("bulk_density", bulk_density, (4,))
    _require_dd("bulk_potential", bulk_potential, (2,))
    _require_dd("occupancy", occupancy, ())
    multiplier = DD(capture_multiplier)
    if (
        np.any(state_density <= 0)
        or np.any(bulk_density <= 0)
        or not bool((occupancy >= 0) & (occupancy <= 1))
        or not bool((multiplier >= 0) & (multiplier <= 1))
    ):
        raise ValueError("compensated local carriers require positive densities and physical occupancies")
    if not isinstance(coefficients, FrozenLocalCoefficients):
        raise TypeError("explicit FrozenLocalCoefficients are required")

    physics = coefficients.physics
    vt = physics["thermal_voltage_V"]
    xl = (trace_potential[0] - bulk_potential[0]) / vt
    xr = (bulk_potential[1] - trace_potential[1]) / vt
    b, bm, db, dbm = _bernoulli_values_and_derivatives(_cat(xl, xr))
    incoming = _cat(bm[0], b[0], b[1], bm[1])
    outgoing = _cat(b[0], bm[0], bm[1], b[1])
    k = coefficients.carrier_diffusion_over_distance
    bulk_flux = k * (incoming * bulk_density - outgoing * state_density)
    bulk_log_jacobian = _diagonal(-k * outgoing * state_density)
    potential_derivative = (k / vt) * _cat(
        -dbm[0] * bulk_density[0] - db[0] * state_density[0],
        db[0] * bulk_density[1] + dbm[0] * state_density[1],
        -db[1] * bulk_density[2] - dbm[1] * state_density[2],
        dbm[1] * bulk_density[3] + db[1] * state_density[3],
    )
    bulk_trace_jacobian = DD(np.zeros((4, 2), dtype=np.float64))
    bulk_trace_jacobian = _put(bulk_trace_jacobian, (slice(0, 2), 0), potential_derivative[:2])
    bulk_trace_jacobian = _put(bulk_trace_jacobian, (slice(2, 4), 1), potential_derivative[2:])
    bulk_coordinate_jacobian = DD(np.zeros((4, 6), dtype=np.float64))
    bulk_coordinate_jacobian = _put(
        bulk_coordinate_jacobian, (slice(None), slice(0, 2)), -bulk_trace_jacobian
    )
    bulk_coordinate_jacobian = _put(
        bulk_coordinate_jacobian,
        (slice(None), slice(2, 6)),
        _diagonal(k * incoming * bulk_density),
    )

    jump = trace_potential[1] - trace_potential[0]
    electron_step = physics["conduction_band_step_eV"] - jump
    hole_step = physics["hole_transport_step_eV"] + jump
    barriers = _cat(
        electron_step if bool(electron_step > 0) else _ZERO,
        hole_step if bool(hole_step > 0) else _ZERO,
        -electron_step if bool(electron_step < 0) else _ZERO,
        -hole_step if bool(hole_step < 0) else _ZERO,
    )
    supplies, supply_log_derivative, supply_barrier_derivative, metadata = _supplies(
        state_density, barriers, coefficients
    )
    pn, pp = coefficients.cross_prefactors[0], coefficients.cross_prefactors[1]
    xn = pn * (supplies[0] - supplies[2])
    xp = pp * (supplies[1] - supplies[3])
    cross_flux = _cat(-xn, -xp, xn, xp)
    cross_pair_log = DD(np.zeros((2, 4), dtype=np.float64))
    cross_pair_log = _put(cross_pair_log, (0, 0), pn * supply_log_derivative[0])
    cross_pair_log = _put(cross_pair_log, (0, 2), -pn * supply_log_derivative[2])
    cross_pair_log = _put(cross_pair_log, (1, 1), pp * supply_log_derivative[1])
    cross_pair_log = _put(cross_pair_log, (1, 3), -pp * supply_log_derivative[3])

    electron_jump_derivative = _ZERO
    if bool(electron_step > 0):
        electron_jump_derivative = -pn * supply_barrier_derivative[0]
    elif bool(electron_step < 0):
        electron_jump_derivative = -pn * supply_barrier_derivative[2]
    hole_jump_derivative = _ZERO
    if bool(hole_step > 0):
        hole_jump_derivative = pp * supply_barrier_derivative[1]
    elif bool(hole_step < 0):
        hole_jump_derivative = pp * supply_barrier_derivative[3]

    cross_pair_trace = _cat(
        -electron_jump_derivative,
        electron_jump_derivative,
        -hole_jump_derivative,
        hole_jump_derivative,
    ).reshape(2, 2)
    cross_log_jacobian = _cat(
        -cross_pair_log[0],
        -cross_pair_log[1],
        cross_pair_log[0],
        cross_pair_log[1],
    ).reshape(4, 4)
    cross_trace_jacobian = _cat(
        -cross_pair_trace[0],
        -cross_pair_trace[1],
        cross_pair_trace[0],
        cross_pair_trace[1],
    ).reshape(4, 2)

    vn = multiplier * physics["surface_recombination_velocity_n_m_s"]
    vp = multiplier * physics["surface_recombination_velocity_p_m_s"]
    velocity = _cat(vn, vp, vn, vp)
    live_factor = _cat(_ONE - occupancy, occupancy, _ONE - occupancy, occupancy)
    emission_factor = _ONE - live_factor
    capture_flux = velocity * (
        state_density * live_factor - coefficients.emission_densities * emission_factor
    )
    capture_log_jacobian = _diagonal(velocity * live_factor * state_density)
    capture_occupancy_derivative = _cat(-vn, vp, -vn, vp) * (
        state_density + coefficients.emission_densities
    )
    residual = bulk_flux + cross_flux - capture_flux
    one_way_scale = _cat(
        pn * _choose(supplies[0] >= supplies[2], supplies[0], supplies[2]),
        pp * _choose(supplies[1] >= supplies[3], supplies[1], supplies[3]),
    )

    balance: dict[str, DD] = {
        "state_m3": state_density,
        "bulk_flux_m2_s": bulk_flux,
        "cross_flux_m2_s": cross_flux,
        "capture_flux_m2_s": capture_flux,
        "residual_m2_s": residual,
        "jacobian_log_state_m2_s": bulk_log_jacobian + cross_log_jacobian - capture_log_jacobian,
        "jacobian_trace_potential_m2_s_V": bulk_trace_jacobian + cross_trace_jacobian,
        "jacobian_bulk_coordinates": bulk_coordinate_jacobian,
        "one_way_cross_scale_m2_s": one_way_scale,
    }
    tangent: dict[str, DD] = {
        "bulk_flux_jacobian_log_state_m2_s": bulk_log_jacobian,
        "bulk_flux_jacobian_trace_potential_m2_s_V": bulk_trace_jacobian,
        "bulk_flux_jacobian_bulk_coordinates": bulk_coordinate_jacobian,
        "capture_flux_jacobian_log_state_m2_s": capture_log_jacobian,
        "capture_flux_occupancy_derivative_m2_s": capture_occupancy_derivative,
        "residual_occupancy_derivative_m2_s": -capture_occupancy_derivative,
    }
    return LocalCarrierEvaluation(
        balance=balance,
        tangent=tangent,
        metadata={
            **metadata,
            "fd_table_sha256": coefficients.fd_table_sha256,
            "bulk_bernoulli_arguments": _cat(xl, xr),
            "band_steps": _cat(electron_step, hole_step),
            "barriers": barriers,
            "fermi_one_polynomial_degree": 12,
            "supply_derivative_policy": "derivative_of_the_same_frozen_twelve_term_polynomial",
            "inverse_derivative_policy": "selected_linear_interval_not_legacy_knot_average",
            "zero_band_step_derivative_policy": "original_zero_derivative_at_exact_zero",
            "small_B_derivative_policy": "derivative_of_original_quadratic_B_below_1e_minus_8",
            "old_F0_difference_note": "computed_DD_difference_may_not_resolve_the_smaller_analytic_tail",
            "constitutive_domain_complete": False,
            "fallback_used": False,
        },
    )


def _dd_array_float(value: DD) -> NDArray[np.float64]:
    out = value.to_float()
    return np.asarray(out, dtype=np.float64)


def to_float_tangent(evaluation: LocalCarrierEvaluation) -> CompensatedCarrierTangent:
    """Explicit binary64 projection boundary from a ``LocalCarrierEvaluation``."""
    if evaluation.schema != SCHEMA:
        raise ValueError("unexpected compensated local carrier evaluation schema")
    bal = evaluation.balance
    tan = evaluation.tangent
    balance = CompensatedCarrierBalance(
        state_m3=_dd_array_float(bal["state_m3"]),
        bulk_flux_m2_s=_dd_array_float(bal["bulk_flux_m2_s"]),
        cross_flux_m2_s=_dd_array_float(bal["cross_flux_m2_s"]),
        capture_flux_m2_s=_dd_array_float(bal["capture_flux_m2_s"]),
        residual_m2_s=_dd_array_float(bal["residual_m2_s"]),
        jacobian_log_state_m2_s=_dd_array_float(bal["jacobian_log_state_m2_s"]),
        jacobian_trace_potential_m2_s_V=_dd_array_float(bal["jacobian_trace_potential_m2_s_V"]),
        jacobian_bulk_coordinates=_dd_array_float(bal["jacobian_bulk_coordinates"]),
        one_way_cross_scale_m2_s=_dd_array_float(bal["one_way_cross_scale_m2_s"]),
    )
    return CompensatedCarrierTangent(
        balance=balance,
        bulk_flux_jacobian_log_state_m2_s=_dd_array_float(tan["bulk_flux_jacobian_log_state_m2_s"]),
        bulk_flux_jacobian_trace_potential_m2_s_V=_dd_array_float(
            tan["bulk_flux_jacobian_trace_potential_m2_s_V"]
        ),
        bulk_flux_jacobian_bulk_coordinates=_dd_array_float(tan["bulk_flux_jacobian_bulk_coordinates"]),
        capture_flux_jacobian_log_state_m2_s=_dd_array_float(tan["capture_flux_jacobian_log_state_m2_s"]),
        capture_flux_occupancy_derivative_m2_s=_dd_array_float(
            tan["capture_flux_occupancy_derivative_m2_s"]
        ),
        residual_occupancy_derivative_m2_s=_dd_array_float(tan["residual_occupancy_derivative_m2_s"]),
    )
