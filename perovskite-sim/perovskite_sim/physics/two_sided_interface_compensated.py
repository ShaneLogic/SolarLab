"""Explicit compensated evaluation of the local two-sided carrier law.

All physical state inputs are explicit DD values. Callers supply immutable
Fermi interpolation coefficients and their expected canonical hash. The
default coefficient table is frozen from the existing runtime's table once.
There is no import or reuse of the Decimal diagnostic oracle.

Supported constitutive domain: inverse Fermi half below or inside the frozen
log table, and all four Fermi-one supply arguments strictly below -4. Missing
branches and unsupported DD ranges raise; no binary64 fallback is allowed.
The frozen twelve-term Fermi-one polynomial is evaluated with its own exact
polynomial derivative. This differs from the old complete Fermi-zero tangent
by a declared truncation term; the original inverse-table knot-average
tangent is also not silently substituted for a selected interval derivative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np

from perovskite_sim.physics.compensated import DD


__all__ = ["FrozenLocalCoefficients", "LocalCarrierEvaluation",
           "UnsupportedCompensatedInterfaceDomain", "default_fd_table",
           "freeze_coefficients", "evaluate_local_carrier_pair", "to_production_tangent"]

SCHEMA = "TwoSidedInterfaceCompensatedV1"
SCOPE = "bounded_local_constitutive_evaluation_not_a_nonlinear_solve"
_ZERO, _ONE, _HALF = DD(0), DD(1), DD(0.5)
_SMALL_B = DD(1e-8)  # Exact original binary64 branch threshold.
_DERIVATIVE_SERIES_LIMIT = DD(1e-3)
_MAX_EXP_ARGUMENT = DD(672)
_RECIPROCAL = tuple(_ONE / order for order in range(1, 14))
_RECIPROCAL_SQUARE = tuple(_ONE / (order * order) for order in range(1, 13))
_B_DERIVATIVE_COEFFICIENTS = (
    _ONE / 6, -_ONE / 180, _ONE / 5040, -_ONE / 151200, _ONE / 4790016,
)


class UnsupportedCompensatedInterfaceDomain(ArithmeticError):
    """An uncovered constitutive probe must terminate its scientific worker.

    This intentionally does not inherit ValueError or RuntimeError: existing
    Newton and initial-local line searches catch those for admissible-domain
    backtracking. Missing arithmetic coverage is not such a backtrack.
    """


def _freeze_value(value):
    if isinstance(value, DD):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class LocalCarrierEvaluation(Mapping[str, object]):
    """State-owned immutable result, compatible with the mapping adapter API."""

    balance: Mapping[str, DD]
    tangent: Mapping[str, DD]
    metadata: Mapping[str, object]
    schema: str = SCHEMA
    scope: str = SCOPE

    def __post_init__(self):
        object.__setattr__(self, "balance", _freeze_value(self.balance))
        object.__setattr__(self, "tangent", _freeze_value(self.tangent))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    def __getitem__(self, key):
        if key not in ("schema", "scope", "balance", "tangent", "metadata"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("schema", "scope", "balance", "tangent", "metadata"))

    def __len__(self):
        return 5

    def __deepcopy__(self, memo):
        # Retained states may be copied for independent tampering controls.
        # Every member is immutable, so sharing this result remains safe.
        return self


@lru_cache(maxsize=1)
def default_fd_table():
    """Freeze the current original table; return immutable nodes and identity."""
    from perovskite_sim.physics.fermi_dirac import _half_table

    eta, _values, log_half = _half_table()
    payload = {"eta": tuple(float(value) for value in eta),
               "log_half": tuple(float(value) for value in log_half)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return MappingProxyType(payload), hashlib.sha256(encoded).hexdigest()


def _cat(*values):
    """Copy/select already normalized DD words, without a float boundary."""
    return DD._trusted_parts(
        np.concatenate([np.atleast_1d(value.hi) for value in values]),
        np.concatenate([np.atleast_1d(value.lo) for value in values]),
    )


def _put(value, index, part):
    high, low = value.hi.copy(), value.lo.copy()
    high[index], low[index] = part.hi, part.lo
    return DD._trusted_parts(high, low)


def _choose(mask, first, second):
    return DD._trusted_parts(np.where(mask, first.hi, second.hi),
                             np.where(mask, first.lo, second.lo))


def _diagonal(values):
    return DD._trusted_parts(np.diag(values.hi), np.diag(values.lo))


def _require_dd(name, value, shape):
    if not isinstance(value, DD) or value.shape != shape:
        raise TypeError(name + " must be DD with shape " + repr(shape))


def _frozen_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("frozen interpolation nodes must be finite vectors")
    return np.frombuffer(values.tobytes(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FrozenLocalCoefficients:
    """Immutable, state-independent coefficients reusable across evaluations."""

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
    def log_half_search(self):
        # A fresh view also protects cached shape metadata, not just values.
        return np.frombuffer(self._log_half_search_bytes, dtype=np.float64)

    def __deepcopy__(self, memo):
        return self


def freeze_coefficients(geometry, physics, *, fd_table, fd_table_sha256, q_C):
    """Bind exact binary64 coefficients; never fetch or regenerate a table."""
    if not isinstance(fd_table, Mapping) or set(fd_table) != {"eta", "log_half"}:
        raise ValueError("frozen Fermi table fields differ from the declared contract")
    table_payload = {name: list(values) for name, values in fd_table.items()}
    encoded = json.dumps(table_payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    observed = hashlib.sha256(encoded).hexdigest()
    if observed != fd_table_sha256 or set(fd_table) != {"eta", "log_half"}:
        raise ValueError("frozen Linux Fermi table identity mismatch")
    eta = _frozen_array(fd_table["eta"])
    log_half = _frozen_array(fd_table["log_half"])
    if (eta.size != log_half.size or eta.size < 2
            or not np.all(np.diff(eta) > 0)
            or not np.all(np.diff(log_half) > 0)):
        raise ValueError("frozen Fermi nodes must be strictly increasing and paired")
    values = {name: DD(value) for name, value in physics.items()}
    if any(value.shape != () for value in values.values()):
        raise ValueError("material coefficients must be scalar binary64 inputs")
    vt, temperature = values["thermal_voltage_V"], values["temperature_K"]
    charge = DD(q_C)
    left, right = DD(geometry["left_distance_m"]), DD(geometry["right_distance_m"])
    if (not bool(vt > 0) or not bool(temperature > 0) or not bool(charge > 0)
            or not bool(left > 0) or not bool(right > 0)
            or not bool((values["transmission"] >= 0) & (values["transmission"] <= 1))):
        raise ValueError("invalid frozen local material or geometry")
    diffusion = _cat(values["D_n_left_m2_s"], values["D_p_left_m2_s"],
                     values["D_n_right_m2_s"], values["D_p_right_m2_s"])
    dos = _cat(values["N_C_left_m3"], values["N_V_left_m3"],
               values["N_C_right_m3"], values["N_V_right_m3"])
    emission = _cat(values["n1_left_m3"], values["p1_left_m3"],
                    values["n1_right_m3"], values["p1_right_m3"])
    richardson = _cat(values["richardson_n_A_m2_K2"], values["richardson_p_A_m2_K2"])
    velocity = _cat(values["surface_recombination_velocity_n_m_s"],
                    values["surface_recombination_velocity_p_m_s"])
    if (np.any(diffusion < 0) or np.any(dos <= 0) or np.any(emission < 0)
            or np.any(richardson < 0) or np.any(velocity < 0)):
        raise ValueError("invalid frozen local carrier coefficients")
    eta_dd, log_dd = DD(eta), DD(log_half)
    return FrozenLocalCoefficients(
        MappingProxyType(values), diffusion / _cat(left, left, right, right),
        dos, emission,
        values["transmission"] * richardson * temperature * temperature / charge,
        eta_dd, log_dd, (eta_dd[1:] - eta_dd[:-1]) / (log_dd[1:] - log_dd[:-1]),
        log_half.tobytes(), observed,
    )


def _bernoulli_values_and_derivatives(x):
    """B(x), B(-x), B'(x), B'(-x), retaining the original small-B law."""
    magnitude = abs(x)
    if np.any(magnitude > _MAX_EXP_ARGUMENT):
        raise UnsupportedCompensatedInterfaceDomain("Bernoulli argument exceeds the compensated domain")
    b = DD(np.ones(x.shape))
    derivative = DD(np.full(x.shape, -0.5))
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
        db = DD(np.zeros(a.shape))
        if np.any(stable):
            z = a[stable]
            z2 = z * z
            c1, c3, c5, c7, c9 = _B_DERIVATIVE_COEFFICIENTS
            db = _put(db, stable, -_HALF + z * (c1 + z2 * (
                c3 + z2 * (c5 + z2 * (c7 + z2 * c9)))))
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
        derivative = _put(derivative, large,
                          decay * (_ONE - a - decay) / (denominator * denominator))
    backward = b + magnitude
    backward_derivative = -_ONE - derivative
    positive = x >= 0
    return (_choose(positive, b, backward), _choose(positive, backward, b),
            _choose(positive, derivative, backward_derivative),
            _choose(positive, backward_derivative, derivative))


def _inverse_half(log_ratio, coefficients):
    """Frozen inverse interpolant and its selected-interval log derivative."""
    if np.any(log_ratio > coefficients.log_half_nodes[-1]):
        raise UnsupportedCompensatedInterfaceDomain("compensated local carrier law does not implement the Sommerfeld inverse branch")
    below = log_ratio <= coefficients.log_half_nodes[0]
    eta, slope = log_ratio.copy(), DD(np.ones(log_ratio.shape))
    intervals = np.full(log_ratio.shape, -1, dtype=np.int64)
    if np.any(~below):
        target = log_ratio[~below]
        index = np.searchsorted(coefficients.log_half_search, target.hi, side="right") - 1
        index = np.clip(index, 0, coefficients.log_half_search.size - 2)
        # A rounded high word can equal a knot while the full pair is below it.
        # Selection uses the full pair before the actual interpolant evaluation.
        lower = target < coefficients.log_half_nodes[index]
        index = index - lower.astype(np.int64)
        upper = ((index < coefficients.log_half_search.size - 2)
                 & (target >= coefficients.log_half_nodes[index + 1]))
        index = index + upper.astype(np.int64)
        if (np.any(index < 0) or np.any(index >= coefficients.log_half_search.size - 1)
                or np.any(target < coefficients.log_half_nodes[index])
                or np.any(target > coefficients.log_half_nodes[index + 1])):
            raise ArithmeticError("full-pair inverse-table interval selection failed")
        derivative = coefficients.inverse_log_slopes[index]
        eta = _put(eta, ~below, coefficients.eta_nodes[index]
                   + (target - coefficients.log_half_nodes[index]) * derivative)
        slope = _put(slope, ~below, derivative)
        intervals[~below] = index
    return eta, slope, intervals


def _supplies(state, barriers, coefficients):
    ratio = state / coefficients.density_of_states
    log_ratio = ratio.log()
    eta, inverse_slope, intervals = _inverse_half(log_ratio, coefficients)
    argument = eta - barriers / coefficients.physics["thermal_voltage_V"]
    if np.any(argument >= -4):
        raise UnsupportedCompensatedInterfaceDomain("compensated local carrier law only covers the frozen twelve-term Fermi-one branch")
    activity = argument.exp()
    total, derivative, power = DD(np.zeros(state.shape)), DD(np.zeros(state.shape)), activity
    for order in range(1, 13):
        sign = 1 if order % 2 else -1
        total = total + sign * power * _RECIPROCAL_SQUARE[order - 1]
        derivative = derivative + sign * power * _RECIPROCAL[order - 1]
        if order != 12:
            power = power * activity
    # This comparison is arithmetic metadata, not a substitution of the old
    # full-F0 tangent. A sub-DD truncation difference need not be resolved by
    # subtracting two DD evaluations; the analytic tail bound is separate.
    old_fermi_zero = activity.log1p()
    tail_bound = power * activity * _RECIPROCAL[12] / (_ONE - activity)
    return total, derivative * inverse_slope, -derivative / coefficients.physics["thermal_voltage_V"], {
        "inverse_half_arguments": eta, "inverse_log_ratio": log_ratio,
        "inverse_intervals": intervals, "inverse_log_derivative": inverse_slope,
        "supply_arguments": argument,
        "polynomial_argument_derivative": derivative,
        "old_complete_F0_argument_derivative": old_fermi_zero,
        "computed_DD_old_F0_minus_polynomial_derivative": old_fermi_zero - derivative,
        "analytic_old_F0_tail_bound": tail_bound,
    }


def evaluate_local_carrier_pair(*, state_density, trace_potential, bulk_density,
                                bulk_potential, occupancy, capture_multiplier,
                                coefficients):
    """Evaluate four frozen local rows and all direct partial derivatives.

    Carrier order is [nL,pL,nR,pR]. Bulk columns are
    [phiL,phiR,log(nL),log(pL),log(nR),log(pR)]. Trace-potential derivatives
    are per volt. Occupancy derivatives are per f, before a logit chain.
    No scale, residual threshold, state update, or line search is performed.
    """
    for name, value, shape in (("state_density", state_density, (4,)),
                               ("trace_potential", trace_potential, (2,)),
                               ("bulk_density", bulk_density, (4,)),
                               ("bulk_potential", bulk_potential, (2,)),
                               ("occupancy", occupancy, ())):
        _require_dd(name, value, shape)
    multiplier = DD(capture_multiplier)
    if (np.any(state_density <= 0) or np.any(bulk_density <= 0)
            or not bool((occupancy >= 0) & (occupancy <= 1))
            or not bool((multiplier >= 0) & (multiplier <= 1))):
        raise ValueError("compensated local carriers require positive densities and physical occupancies")
    if not isinstance(coefficients, FrozenLocalCoefficients):
        raise TypeError("explicit frozen Linux coefficients are required")
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
    potential_derivative = k / vt * _cat(
        -dbm[0] * bulk_density[0] - db[0] * state_density[0],
        db[0] * bulk_density[1] + dbm[0] * state_density[1],
        -db[1] * bulk_density[2] - dbm[1] * state_density[2],
        dbm[1] * bulk_density[3] + db[1] * state_density[3],
    )
    bulk_trace_jacobian = DD(np.zeros((4, 2)))
    bulk_trace_jacobian = _put(bulk_trace_jacobian, (slice(0, 2), 0), potential_derivative[:2])
    bulk_trace_jacobian = _put(bulk_trace_jacobian, (slice(2, 4), 1), potential_derivative[2:])
    bulk_coordinate_jacobian = DD(np.zeros((4, 6)))
    bulk_coordinate_jacobian = _put(bulk_coordinate_jacobian, (slice(None), slice(0, 2)),
                                     -bulk_trace_jacobian)
    bulk_coordinate_jacobian = _put(bulk_coordinate_jacobian, (slice(None), slice(2, 6)),
                                     _diagonal(k * incoming * bulk_density))

    jump = trace_potential[1] - trace_potential[0]
    electron_step = physics["conduction_band_step_eV"] - jump
    hole_step = physics["hole_transport_step_eV"] + jump
    barriers = _cat(electron_step if bool(electron_step > 0) else _ZERO,
                    hole_step if bool(hole_step > 0) else _ZERO,
                    -electron_step if bool(electron_step < 0) else _ZERO,
                    -hole_step if bool(hole_step < 0) else _ZERO)
    supplies, supply_log_derivative, supply_barrier_derivative, metadata = _supplies(
        state_density, barriers, coefficients)
    pn, pp = coefficients.cross_prefactors[0], coefficients.cross_prefactors[1]
    xn, xp = pn * (supplies[0] - supplies[2]), pp * (supplies[1] - supplies[3])
    cross_flux = _cat(-xn, -xp, xn, xp)
    cross_pair_log = DD(np.zeros((2, 4)))
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
    cross_pair_trace = _cat(-electron_jump_derivative, electron_jump_derivative,
                            -hole_jump_derivative, hole_jump_derivative).reshape(2, 2)
    cross_log_jacobian = _cat(-cross_pair_log[0], -cross_pair_log[1],
                              cross_pair_log[0], cross_pair_log[1]).reshape(4, 4)
    cross_trace_jacobian = _cat(-cross_pair_trace[0], -cross_pair_trace[1],
                                cross_pair_trace[0], cross_pair_trace[1]).reshape(4, 2)

    vn = multiplier * physics["surface_recombination_velocity_n_m_s"]
    vp = multiplier * physics["surface_recombination_velocity_p_m_s"]
    velocity = _cat(vn, vp, vn, vp)
    live_factor = _cat(_ONE - occupancy, occupancy, _ONE - occupancy, occupancy)
    emission_factor = _ONE - live_factor
    capture_flux = velocity * (state_density * live_factor
                               - coefficients.emission_densities * emission_factor)
    capture_log_jacobian = _diagonal(velocity * live_factor * state_density)
    capture_occupancy_derivative = _cat(-vn, vp, -vn, vp) * (
        state_density + coefficients.emission_densities)
    residual = bulk_flux + cross_flux - capture_flux
    one_way_scale = _cat(pn * _choose(supplies[0] >= supplies[2], supplies[0], supplies[2]),
                         pp * _choose(supplies[1] >= supplies[3], supplies[1], supplies[3]))
    balance = {
        "state_m3": state_density, "bulk_flux_m2_s": bulk_flux,
        "cross_flux_m2_s": cross_flux, "capture_flux_m2_s": capture_flux,
        "residual_m2_s": residual,
        "jacobian_log_state_m2_s": bulk_log_jacobian + cross_log_jacobian - capture_log_jacobian,
        "jacobian_trace_potential_m2_s_V": bulk_trace_jacobian + cross_trace_jacobian,
        "jacobian_bulk_coordinates": bulk_coordinate_jacobian,
        "one_way_cross_scale_m2_s": one_way_scale,
    }
    tangent = {
        "bulk_flux_jacobian_log_state_m2_s": bulk_log_jacobian,
        "bulk_flux_jacobian_trace_potential_m2_s_V": bulk_trace_jacobian,
        "bulk_flux_jacobian_bulk_coordinates": bulk_coordinate_jacobian,
        "capture_flux_jacobian_log_state_m2_s": capture_log_jacobian,
        "capture_flux_occupancy_derivative_m2_s": capture_occupancy_derivative,
        "residual_occupancy_derivative_m2_s": -capture_occupancy_derivative,
    }
    return LocalCarrierEvaluation(balance=balance, tangent=tangent, metadata={
                **metadata, "fd_table_sha256": coefficients.fd_table_sha256,
                "bulk_bernoulli_arguments": _cat(xl, xr),
                "band_steps": _cat(electron_step, hole_step),
                "barriers": barriers, "fermi_one_polynomial_degree": 12,
                "supply_derivative_policy": "derivative_of_the_same_frozen_twelve_term_polynomial",
                "inverse_derivative_policy": "selected_linear_interval_not_legacy_knot_average",
                "zero_band_step_derivative_policy": "original_zero_derivative_at_exact_zero",
                "small_B_derivative_policy": "derivative_of_original_quadratic_B_below_1e_minus_8",
                "old_F0_difference_note": "computed_DD_difference_may_not_resolve_the_smaller_analytic_tail",
                "constitutive_domain_complete": False, "fallback_used": False,
            })


def to_production_tangent(evaluation):
    """Explicit final float boundary into the unchanged data carrier types.

    Source, current and storage adapters must still consume the DD values in
    evaluation when their own sums cancel. This adapter alone is not complete
    production integration, and independently rebuilt checks must not read a
    direct evaluation cache.
    """
    if evaluation.get("schema") != SCHEMA:
        raise ValueError("unexpected compensated local carrier evaluation")
    from perovskite_sim.physics.two_sided_interface import (
        FixedOccupancyCarrierTangent, TwoSidedCarrierBalance,
    )
    balance = TwoSidedCarrierBalance(**{
        name: value.to_float() for name, value in evaluation["balance"].items()})
    return FixedOccupancyCarrierTangent(balance=balance, **{
        name: value.to_float() for name, value in evaluation["tangent"].items()})
