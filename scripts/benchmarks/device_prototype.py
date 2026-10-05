"""Physical precision consumers for P02; no device engine or time integrator.

The constitutive laws consume the public physical fields and endpoint-bound
increments. Only the admitted stateless precision provider is imported. Input
lifts below are kinematic evaluations, never a historical Newton replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from scripts.benchmarks.contract_prototype import (
    AREA, COULOMB, PARTICLE, SECOND, VOLUME, CellSource, ContractError,
    FaceFlux, Geometry, ImmutableArrays, Point, StateIncrement, StateView, TermSink,
    Vector, frozen_array, integer_indices,
)
from scripts.benchmarks.precision_prototype import (
    DD, DoubleArithmetic, DoubleArray, InputLiftCoordinates, bernoulli,
)


def physical(state: StateView, name: str) -> DoubleArray:
    value = state.field(name)
    if not isinstance(value, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    return value


def coefficient(value: ArrayLike, count: int, *, positive: bool = False) -> DD:
    raw = frozen_array(value)
    if raw.ndim == 0:
        raw = np.full(count, float(raw))
    if raw.shape != (count,):
        raise ContractError("physical_coefficient_shape_mismatch")
    if np.any(raw <= 0 if positive else raw < 0):
        raise ContractError("physical_coefficient_outside_domain")
    return DD(raw)


@dataclass(frozen=True)
class FaceGeometry(ImmutableArrays):
    """Actual node pairs and spacings; no assumption of uniform topology."""

    pairs: np.ndarray
    spacing_m: Vector

    def __post_init__(self) -> None:
        pairs = integer_indices(self.pairs)
        spacing = frozen_array(self.spacing_m)
        if pairs.ndim != 2 or pairs.shape != (spacing.size, 2) or spacing.ndim != 1:
            raise ContractError("face_geometry_shape_mismatch")
        if np.any(pairs < 0) or np.any(pairs[:, 0] == pairs[:, 1]) or np.any(spacing <= 0):
            raise ContractError("invalid_face_geometry")
        object.__setattr__(self, "pairs", np.frombuffer(pairs.tobytes(), dtype=pairs.dtype).reshape(pairs.shape))
        object.__setattr__(self, "spacing_m", spacing)

    def endpoints(self, value: DoubleArray) -> tuple[DD, DD]:
        if len(value.shape) != 1 or np.any(self.pairs >= value.shape[0]):
            raise ContractError("face_outside_physical_support")
        return value.take(self.pairs[:, 0]).as_dd(), value.take(self.pairs[:, 1]).as_dd()


def _bernoulli_pair(state: StateView, geometry: FaceGeometry,
                    thermal_voltage_V: float, potential_id: str) -> tuple[DD, DD, DD]:
    if not np.isfinite(thermal_voltage_V) or thermal_voltage_V <= 0:
        raise ContractError("invalid_thermal_voltage")
    potential = state.face_difference(potential_id, geometry.pairs)
    if not isinstance(potential, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    xi = potential.as_dd() / thermal_voltage_V
    return (bernoulli(DoubleArray.from_dd(xi)).as_dd(),
            bernoulli(DoubleArray.from_dd(-xi)).as_dd(), xi)


@lru_cache(maxsize=1)
def _bernoulli_even_coefficients() -> tuple[DD, ...]:
    """B_(2k)/(2k)! through degree 24, with exact integer inputs.

    The generating series has radius 2*pi (DLMF 24.2.1). On |x|<=1/4,
    the omitted derivative is below 3e-34; coefficient arithmetic stays DD.
    """
    numbers = ((1, 6), (-1, 30), (1, 42), (-1, 30), (5, 66),
               (-691, 2730), (7, 6), (-3617, 510), (43867, 798),
               (-174611, 330), (854513, 138), (-236364091, 2730))
    factorial = DD(1)
    values = []
    for k, (numerator, denominator) in enumerate(numbers, 1):
        factorial = factorial * (2*k-1) * (2*k)
        values.append(DD(numerator) / (factorial * denominator))
    return tuple(values)


def _bernoulli_log_change(x0: DD, x1: DD, dx: DD, b0: DD, b1: DD) -> DD:
    """Finite log(B(x1)/B(x0)) retaining an independently supplied tiny dx.

    Near zero use the divided difference of the convergent Bernoulli series.
    Away from zero use the exact ratio identity; large changes have a
    well-conditioned endpoint ratio. No difference of near-equal B values is
    used to recover a tiny increment.
    """
    high, low = np.zeros(dx.shape), np.zeros(dx.shape)
    changed = dx != 0
    near = changed & (abs(x0) <= 0.25) & (abs(x1) <= 0.25)
    if np.any(near):
        u0, u1 = x0[near]*x0[near], x1[near]*x1[near]
        coefficients = _bernoulli_even_coefficients()
        polynomial = coefficients[-1]
        divided = DD(np.zeros(u0.shape))
        for coefficient_value in (*reversed(coefficients[:-1]), DD(0)):
            divided = u1 * divided + polynomial
            polynomial = u0 * polynomial + coefficient_value
        db = dx[near] * (-0.5 + (x1[near] + x0[near]) * divided)
        value = (db / b0[near]).log1p()
        high[near], low[near] = value.hi, value.lo
    same_sign = ((x0 > 0) & (x1 > 0)) | ((x0 < 0) & (x1 < 0))
    close = changed & ~near & same_sign & (abs(dx) <= 0.125 * (1 + abs(x0)))
    if np.any(close):
        start, step = x0[close], dx[close]
        r = step / start
        positive = start > 0
        uh, ul = np.zeros(start.shape), np.zeros(start.shape)
        if np.any(positive):
            value = step[positive].expm1() / (-(-start[positive]).expm1())
            uh[positive], ul[positive] = value.hi, value.lo
        if np.any(~positive):
            value = start[~positive].exp() * step[~positive].expm1() / start[~positive].expm1()
            uh[~positive], ul[~positive] = value.hi, value.lo
        value = r.log1p() - DD(uh, ul).log1p()
        high[close], low[close] = value.hi, value.lo
    remaining = changed & ~near & ~close
    if np.any(remaining):
        value = DoubleArray.from_dd(b1[remaining]).log_ratio(DoubleArray.from_dd(b0[remaining])).as_dd()
        high[remaining], low[remaining] = value.hi, value.lo
    return DD(high, low)


def _exponential_change(start: DD, end: DD, log_ratio: DD) -> DD:
    """end-start from a retained logarithmic change of positive amplitudes."""
    positive = log_ratio >= 0
    high, low = np.zeros(log_ratio.shape), np.zeros(log_ratio.shape)
    if np.any(positive):
        value = end[positive] * (-(-log_ratio[positive]).expm1())
        high[positive], low[positive] = value.hi, value.lo
    if np.any(~positive):
        value = start[~positive] * log_ratio[~positive].expm1()
        high[~positive], low[~positive] = value.hi, value.lo
    return DD(high, low)


def _sg_net(forward: DD, backward: DD, affinity: DD) -> DD:
    positive = affinity >= 0
    high, low = np.zeros(affinity.shape), np.zeros(affinity.shape)
    if np.any(positive):
        value = forward[positive] * (-(-affinity[positive]).expm1())
        high[positive], low[positive] = value.hi, value.lo
    if np.any(~positive):
        value = backward[~positive] * affinity[~positive].expm1()
        high[~positive], low[~positive] = value.hi, value.lo
    return DD(high, low)


def sg_current(state: StateView, geometry: FaceGeometry, *, density_id: str,
               thermal_voltage_V: float, diffusion_m2_s: ArrayLike, charge_C: float,
               carrier: str = "electron", potential_id: str = "phi_V") -> DoubleArray:
    """Homogeneous-material SG conventional current, positive along each pair.

    Electrons: qD/dx*[B(xi)n_R-B(-xi)n_L]; holes reverse the densities.
    The cancellation-safe affinity form is used on positive populations.
    TE interfaces, DOS jumps and nonlinear ion sterics require their own laws.
    """
    if carrier not in {"electron", "hole"} or not np.isfinite(charge_C) or charge_C <= 0:
        raise ContractError("invalid_carrier_current_parameters")
    left, right = geometry.endpoints(physical(state, density_id))
    if np.any(left < 0) or np.any(right < 0):
        raise ContractError("negative_physical_population")
    b, bm, xi = _bernoulli_pair(state, geometry, thermal_voltage_V, potential_id)
    prefactor = coefficient(diffusion_m2_s, len(geometry.spacing_m)) * charge_C / DD(geometry.spacing_m)
    forward, backward = (b * right, bm * left) if carrier == "electron" else (b * left, bm * right)
    value = forward - backward
    active = (left > 0) & (right > 0) & (prefactor > 0)
    if np.any(active):
        driving = state.electrochemical_difference(
            density_id, potential_id, geometry.pairs[active], thermal_voltage_V,
            potential_sign=-1 if carrier == "electron" else 1,
            arithmetic=DoubleArithmetic(),
        ).as_dd()
        affinity = driving if carrier == "electron" else -driving
        net = _sg_net(forward[active], backward[active], affinity)
        value_high, value_low = value.hi.copy(), value.lo.copy()
        value_high[active], value_low[active] = net.hi, net.lo
        value = DD(value_high, value_low)
    return DoubleArray.from_dd(prefactor * value)


def sg_current_increment(left: Point, right: Point, increment: StateIncrement,
                         geometry: FaceGeometry, *, density_id: str,
                         thermal_voltage_V: float, diffusion_m2_s: ArrayLike,
                         charge_C: float, carrier: str = "electron",
                         potential_id: str = "phi_V") -> DoubleArray:
    """Finite current change with small physical increments retained throughout.

    Coefficients/geometry are fixed across these endpoints. Time-dependent
    mobility/temperature requires explicit coefficient increments, not this law.
    """
    increment.validate(left, right)
    if carrier not in {"electron", "hole"} or not np.isfinite(charge_C) or charge_C <= 0:
        raise ContractError("invalid_carrier_current_parameters")
    nleft, nright = geometry.endpoints(physical(left.state, density_id))
    endleft, endright = geometry.endpoints(physical(right.state, density_id))
    if any(np.any(value < 0) for value in (nleft, nright, endleft, endright)):
        raise ContractError("negative_physical_population")
    dn = increment.field(density_id)
    if not isinstance(dn, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    dleft, dright = geometry.endpoints(dn)
    b0, bm0, xi0 = _bernoulli_pair(left.state, geometry, thermal_voltage_V, potential_id)
    b1, bm1, xi1 = _bernoulli_pair(right.state, geometry, thermal_voltage_V, potential_id)
    potential_change = increment.face_delta(left, right, potential_id, geometry.pairs,
                                           arithmetic=DoubleArithmetic())
    if not isinstance(potential_change, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    dxi = potential_change.as_dd() / thermal_voltage_V
    log_b = _bernoulli_log_change(xi0, xi1, dxi, b0, b1)
    log_bm = _bernoulli_log_change(-xi0, -xi1, -dxi, bm0, bm1)
    db = _exponential_change(b0, b1, log_b)
    dbm = _exponential_change(bm0, bm1, log_bm)
    if carrier == "electron":
        forward0, backward0 = b0*nright, bm0*nleft
        forward1, backward1 = b1*endright, bm1*endleft
        value = b1*dright + db*nright - bm1*dleft - dbm*nleft
    else:
        forward0, backward0 = b0*nleft, bm0*nright
        forward1, backward1 = b1*endleft, bm1*endright
        value = b1*dleft + db*nleft - bm1*dright - dbm*nright
    prefactor = coefficient(diffusion_m2_s, len(geometry.spacing_m)) * charge_C / DD(geometry.spacing_m)
    affine_diffusion = np.zeros(len(geometry.pairs), dtype=bool)
    flat_field = (xi0 == 0) & (dxi == 0)
    if np.any(flat_field):
        # With a fixed zero field the SG law is exactly linear diffusion.
        # Use the public affine face increment when the provider supports it,
        # preserving constant gradients under a common density addition.
        try:
            gradient_delta = increment.face_delta(left, right, density_id, geometry.pairs[flat_field],
                                                   arithmetic=DoubleArithmetic()).as_dd()
        except ContractError as error:
            if error.reason != "face_delta_requires_affine_map":
                raise
        else:
            vh, vl = value.hi.copy(), value.lo.copy()
            part = gradient_delta if carrier == "electron" else -gradient_delta
            vh[flat_field], vl[flat_field] = part.hi, part.lo
            value = DD(vh, vl)
            affine_diffusion = flat_field
    active = ((nleft > 0) & (nright > 0) & (endleft > 0) & (endright > 0)
              & (prefactor > 0) & ~affine_diffusion)
    if np.any(active):
        pairs = geometry.pairs[active]
        sign = -1 if carrier == "electron" else 1
        affinities = [state.electrochemical_difference(
            density_id, potential_id, pairs, thermal_voltage_V,
            potential_sign=sign, arithmetic=DoubleArithmetic()).as_dd()
            for state in (left.state, right.state)]
        a0, a1 = affinities if carrier == "electron" else [-a for a in affinities]
        f0, f1 = forward0[active], forward1[active]
        r0, r1 = backward0[active], backward1[active]
        # Opposite signs (including zero) have no subtractive cancellation.
        change = _sg_net(f1, r1, a1) - _sg_net(f0, r0, a0)
        high, low = change.hi.copy(), change.lo.copy()
        for positive in (True, False):
            selected = ((a0 > 0) & (a1 > 0)) if positive else ((a0 < 0) & (a1 < 0))
            if not np.any(selected):
                continue
            delta_affinity = increment.electrochemical_delta(
                left, right, density_id, potential_id, pairs[selected], thermal_voltage_V,
                potential_sign=sign, arithmetic=DoubleArithmetic()).as_dd()
            da = delta_affinity if carrier == "electron" else -delta_affinity
            node = (1 if carrier == "electron" else 0) if positive else (0 if carrier == "electron" else 1)
            population_log = increment.log_ratio(left, right, density_id,
                                                indices=pairs[selected, node]).as_dd()
            coefficient_log = (log_b if positive else log_bm)[active][selected]
            g0, g1 = (f0[selected], f1[selected]) if positive else (r0[selected], r1[selected])
            dg = _exponential_change(g0, g1, population_log + coefficient_log)
            if positive:
                factor1 = -(-a1[selected]).expm1()
                df = -_exponential_change((-a0[selected]).exp(), (-a1[selected]).exp(), -da)
            else:
                factor1 = a1[selected].expm1()
                df = _exponential_change(a0[selected].exp(), a1[selected].exp(), da)
            # J=G*f(a): both temporal contributions retain their own weak
            # drivers before multiplication; large drift terms never cancel.
            part = dg*factor1 + g0*df
            high[selected], low[selected] = part.hi, part.lo
        vh, vl = value.hi.copy(), value.lo.copy()
        vh[active], vl[active] = high, low
        value = DD(vh, vl)
    return DoubleArray.from_dd(prefactor * value)


def assemble_charge_rate(state: StateView, equation_id: str, geometry: Geometry,
                         current: DoubleArray) -> DoubleArray:
    sink = TermSink(state.layout, equation_id, geometry, DoubleArithmetic())
    sink.add(FaceFlux("SG-conventional-current", geometry.face_support,
                      current, COULOMB / AREA / SECOND))
    return sink.value()


@dataclass(frozen=True)
class ElectronTrapCapture:
    capture_coefficient_m3_s: float
    trap_density_m3: float
    emission_density_m3: float

    def __post_init__(self) -> None:
        if (not np.isfinite([self.capture_coefficient_m3_s, self.trap_density_m3,
                             self.emission_density_m3]).all()
                or min(self.capture_coefficient_m3_s, self.trap_density_m3,
                       self.emission_density_m3) < 0):
            raise ContractError("invalid_capture_coefficient")

    def net_rate(self, state: StateView, n_id: str = "n_m3",
                  f_id: str = "occupancy") -> DoubleArray:
        n, f = physical(state, n_id).as_dd(), physical(state, f_id).as_dd()
        if n.shape != f.shape or np.any(n < 0) or np.any(f < 0) or np.any(f > 1):
            raise ContractError("invalid_capture_state")
        # Emission is cn*n1 by detailed balance in this single-level closure;
        # do not independently round that coefficient before cancellation.
        rate = (n * (1 - f) - self.emission_density_m3 * f)
        return DoubleArray.from_dd(rate * self.capture_coefficient_m3_s * self.trap_density_m3)

    def finite_change(self, left: Point, right: Point, increment: StateIncrement,
                       n_id: str = "n_m3", f_id: str = "occupancy") -> DoubleArray:
        increment.validate(left, right)
        n0, f1 = physical(left.state, n_id).as_dd(), physical(right.state, f_id).as_dd()
        dn, df = increment.field(n_id), increment.field(f_id)
        if not isinstance(dn, DoubleArray) or not isinstance(df, DoubleArray):
            raise ContractError("physical_operator_requires_explicit_precision_provider")
        if n0.shape != f1.shape or dn.shape != n0.shape or df.shape != n0.shape:
            raise ContractError("capture_increment_shape_mismatch")
        n1, f0 = physical(right.state, n_id).as_dd(), physical(left.state, f_id).as_dd()
        if (np.any(n0 < 0) or np.any(n1 < 0) or np.any(f0 < 0) or np.any(f0 > 1)
                or np.any(f1 < 0) or np.any(f1 > 1)):
            raise ContractError("invalid_capture_state")
        change = (1 - f1) * dn.as_dd() - (n0 + self.emission_density_m3) * df.as_dd()
        return DoubleArray.from_dd(change * self.capture_coefficient_m3_s * self.trap_density_m3)

    def assembled_inventory_sources(self, state: StateView, geometry: Geometry,
                                     carrier_equation: str, trap_equation: str
                                     ) -> tuple[DoubleArray, DoubleArray]:
        rate = self.net_rate(state).as_dd()
        result = []
        for equation, sign in ((carrier_equation, -1), (trap_equation, 1)):
            sink = TermSink(state.layout, equation, geometry, DoubleArithmetic())
            sink.add(CellSource("same-electron-trap-reaction", geometry.cell_support,
                                DoubleArray.from_dd(sign * rate), PARTICLE / VOLUME / SECOND))
            result.append(sink.value())
        return result[0], result[1]


def electric_displacement(state: StateView, geometry: FaceGeometry, epsilon_F_m: ArrayLike,
                          potential_id: str = "phi_V") -> DoubleArray:
    drop = state.face_difference(potential_id, geometry.pairs)
    if not isinstance(drop, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    eps = coefficient(epsilon_F_m, len(geometry.spacing_m), positive=True)
    return DoubleArray.from_dd(-eps * drop.as_dd() / DD(geometry.spacing_m))


def displacement_current(left: Point, right: Point, increment: StateIncrement,
                          displacement_id: str = "D_C_m2") -> DoubleArray:
    increment.validate(left, right)
    dt = right.time - left.time
    if dt <= 0:
        raise ContractError("nonpositive_physical_interval")
    value = increment.field(displacement_id)
    if not isinstance(value, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    return DoubleArray.from_dd(value.as_dd() / dt)


def displacement_current_from_potential(left: Point, right: Point, increment: StateIncrement,
                                        geometry: FaceGeometry, epsilon_F_m: ArrayLike,
                                        potential_id: str = "phi_V") -> DoubleArray:
    increment.validate(left, right)
    dt = right.time - left.time
    if dt <= 0:
        raise ContractError("nonpositive_physical_interval")
    change = increment.field(potential_id)
    if not isinstance(change, DoubleArray):
        raise ContractError("physical_operator_requires_explicit_precision_provider")
    dl, dr = geometry.endpoints(change)
    eps = coefficient(epsilon_F_m, len(geometry.spacing_m), positive=True)
    return DoubleArray.from_dd(-eps * (dr - dl) / DD(geometry.spacing_m) / dt)


def lift_physical_inputs(anchor: Point, thermal_voltage_V: float,
                          increments: Mapping[str, ArrayLike | DoubleArray], time: float,
                          voltage_lift_V: ArrayLike | DoubleArray | None = None,
                          trace_voltage_lift_V: ArrayLike | DoubleArray | None = None,
                          inputs: ArrayLike | None = None) -> tuple[Point, StateIncrement]:
    """Evaluate the public named input map and retain its complete authority."""
    def coordinate(value):
        return value if type(value) is DoubleArray else frozen_array(value)

    return InputLiftCoordinates(anchor.state.layout, thermal_voltage_V).advance(
        anchor, {name: coordinate(value) for name, value in increments.items()}, time,
        voltage_lift_V=None if voltage_lift_V is None else coordinate(voltage_lift_V),
        trace_voltage_lift_V=None if trace_voltage_lift_V is None else coordinate(trace_voltage_lift_V),
        inputs=inputs,
    )
