"""Bounded observations of a declared polynomial path, never a time solver.

The path, its endpoint defects, and its clocks have separate identities. Exact
rational arithmetic is used for the small affine actions; optional Arb balls
are used only for the nonlinear observations. Neither certifies the error of
the DAE time integrator or a continuum spatial model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Mapping

from scripts.benchmarks.contract_prototype import ContractError, PhysicalLinearForm


def rational(value) -> Fraction:
    """The exact value of a stored integer/binary64, with no decimal rounding."""
    if isinstance(value, bool):
        raise ContractError("observation_boolean_number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    value = float(value)
    if not math.isfinite(value):
        raise ContractError("observation_nonfinite_number")
    return Fraction.from_float(value)


def ratio_payload(value):
    value = rational(value)
    return [value.numerator, value.denominator]


def identity(payload) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Polynomial:
    """Ascending exact coefficients in a single declared interval coordinate."""

    coefficients: tuple[Fraction, ...]

    def __init__(self, coefficients=(0,)):
        values = tuple(rational(x) for x in coefficients)
        if not values:
            raise ContractError("observation_empty_polynomial")
        while len(values) > 1 and values[-1] == 0:
            values = values[:-1]
        object.__setattr__(self, "coefficients", values)

    def __add__(self, other):
        other = other if isinstance(other, Polynomial) else Polynomial((other,))
        a, b = self.coefficients, other.coefficients
        return Polynomial((a[i] if i < len(a) else 0)+(b[i] if i < len(b) else 0)
                          for i in range(max(len(a), len(b))))

    __radd__ = __add__

    def __neg__(self):
        return Polynomial(-x for x in self.coefficients)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Polynomial) else -rational(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Polynomial) else Polynomial((other,))
        result = [Fraction(0)]*(len(self.coefficients)+len(other.coefficients)-1)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result[i+j] += a*b
        return Polynomial(result)

    __rmul__ = __mul__

    def __truediv__(self, value):
        value = rational(value)
        if value == 0:
            raise ContractError("observation_zero_divisor")
        return self * (1/value)

    def at(self, coordinate):
        coordinate = rational(coordinate)
        result = Fraction(0)
        for coefficient in reversed(self.coefficients):
            result = result*coordinate+coefficient
        return result

    def derivative(self):
        return Polynomial((i*self.coefficients[i] for i in range(1, len(self.coefficients)))
                          if len(self.coefficients) > 1 else (0,))

    def integral(self, left, right):
        primitive = Polynomial((0, *(c/(i+1) for i, c in enumerate(self.coefficients))))
        return primitive.at(right)-primitive.at(left)

    def range(self, left, right):
        """Exact Bernstein convex-hull enclosure, not sampled extrema."""
        left, right = rational(left), rational(right)
        if right < left:
            raise ContractError("observation_reversed_range")
        n = len(self.coefficients)-1
        shifted = [sum((self.coefficients[k]*math.comb(k, j)*left**(k-j)
                        *(right-left)**j for k in range(j, n+1)), Fraction(0))
                   for j in range(n+1)]
        bernstein = [sum((shifted[j]*Fraction(math.comb(k, j), math.comb(n, j))
                          for j in range(k+1)), Fraction(0)) for k in range(n+1)]
        return min(bernstein), max(bernstein)

    def payload(self):
        return [ratio_payload(x) for x in self.coefficients]


def newton_polynomials(phi, psi, hused):
    """Exact algebra of the documented IDA divided-difference basis.

    This function does not acquire native data or certify its provenance.
    In particular, separate rounded Dky derivatives are not valid ``phi``.
    """
    phi = tuple(tuple(rational(v) for v in row) for row in phi)
    psi, hused = tuple(rational(v) for v in psi), rational(hused)
    if (not 2 <= len(phi) <= 6 or len(psi) != len(phi)-1 or not phi[0]
            or any(len(row) != len(phi[0]) for row in phi)
            or hused <= 0 or any(v <= 0 for v in psi)):
        raise ContractError("observation_native_basis_shape_or_order")
    basis, result = Polynomial((1,)), [Polynomial((v,)) for v in phi[0]]
    for j in range(1, len(phi)):
        basis = basis * Polynomial((0 if j == 1 else psi[j-2], hused)) / psi[j-1]
        result = [p+basis*v for p, v in zip(result, phi[j], strict=True)]
    return tuple(result)


@dataclass(frozen=True)
class AcceptedClock:
    predecessor: Fraction
    tn: Fraction
    hused: Fraction
    generation: int
    nsteps: int
    kused: int
    segment_start: Fraction
    segment_end: Fraction
    segment_id: str
    owner: str | None = None

    def __post_init__(self):
        for name in ("predecessor", "tn", "hused", "segment_start", "segment_end"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        if (any(type(x) is not int or x < 1 for x in (self.generation, self.nsteps, self.kused))
                or self.kused > 5 or not self.segment_id
                or self.owner is not None and (not isinstance(self.owner, str) or not self.owner)
                or not self.segment_start <= self.predecessor < self.tn <= self.segment_end
                or self.hused <= 0):
            raise ContractError("observation_invalid_accepted_clock")

    @property
    def native_left(self):
        return self.tn-self.hused

    @property
    def strip(self):
        return max(Fraction(0), self.native_left-self.predecessor)

    def coordinate(self, time):
        return (rational(time)-self.tn)/self.hused

    def require_covered(self):
        if self.strip:
            raise ContractError("observation_uncovered_predecessor_strip")

    def require_successor(self, other):
        if self.tn != other.predecessor:
            raise ContractError("observation_history_gap_or_overlap")
        if (self.owner is None) != (other.owner is None):
            raise ContractError("observation_clock_owner_authority_changed")
        same = self.segment_id == other.segment_id
        if same and (other.owner != self.owner or other.generation != self.generation
                     or other.nsteps != self.nsteps+1 or other.segment_start != self.segment_start
                     or other.segment_end != self.segment_end):
            raise ContractError("observation_skipped_step_or_generation")
        # Native generation belongs to one solver owner, not to the logical
        # protocol index. The controller creates a fresh owner per segment.
        next_generation = self.generation+1 if self.owner == other.owner else 1
        if not same and (self.tn != self.segment_end or other.predecessor != other.segment_start
                         or other.generation != next_generation or other.nsteps != 1):
            raise ContractError("observation_unbound_event_transition")

    def payload(self):
        return {**{key: ratio_payload(getattr(self, key)) for key in
                   ("predecessor", "tn", "hused", "segment_start", "segment_end")},
                "native_left": ratio_payload(self.native_left), "uncovered_strip": ratio_payload(self.strip),
                "generation": self.generation, "nsteps": self.nsteps, "kused": self.kused,
                "segment_id": self.segment_id, "owner": self.owner}


@dataclass(frozen=True)
class PolynomialPath:
    clock: AcceptedClock
    fields: Mapping[str, tuple[Polynomial, ...]]
    inputs: tuple[Polynomial, ...]
    source_identity: str
    layout_identity: str
    coefficient_identity: str
    origin: str
    clock_policy: str = "reject_uncovered"

    def __post_init__(self):
        if self.origin not in {"synthetic_component", "declared_reconstruction"}:
            # Native admission requires the Engineering export, its independent
            # source review and a concrete adapter. A caller label is not proof.
            raise ContractError("observation_native_basis_provenance_unavailable")
        if (self.clock_policy not in {"reject_uncovered", "declared_polynomial_extension"}
                or self.clock_policy == "declared_polynomial_extension" and self.origin != "declared_reconstruction"):
            raise ContractError("observation_clock_extension_identity")
        fields = {name: tuple(values) for name, values in self.fields.items()}
        if (not all((self.source_identity, self.layout_identity, self.coefficient_identity))
                or not fields or any(not values for values in fields.values())
                or any(not isinstance(p, Polynomial) or len(p.coefficients) > 6
                       for values in (*fields.values(), tuple(self.inputs)) for p in values)):
            raise ContractError("observation_polynomial_path_identity_or_degree")
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "inputs", tuple(self.inputs))

    @property
    def identity(self):
        return identity({"clock": self.clock.payload(), "origin": self.origin,
                         "source": self.source_identity, "layout": self.layout_identity,
                         "coefficients": self.coefficient_identity,
                         "clock_policy": self.clock_policy,
                         "fields": {k: [p.payload() for p in v] for k, v in self.fields.items()},
                         "inputs": [p.payload() for p in self.inputs]})

    def action(self, form: PhysicalLinearForm, sources=None):
        """Contract the actual public factored form, including its divisors."""
        if (form.source_identity != self.source_identity or form.layout.identity != self.layout_identity
                or set(self.fields) != {v.id for v in form.layout.variables}
                or any(len(self.fields[v.id]) != math.prod(v.shape) for v in form.layout.variables)):
            raise ContractError("observation_physical_form_binding")
        sources = {} if sources is None else dict(sources)
        if set(sources) != {s.id for s in form.sources}:
            raise ContractError("observation_missing_physical_source")
        values = dict(self.fields)
        if form.kind == "rate":
            values = {key: tuple(p.derivative()/self.clock.hused for p in row)
                      for key, row in values.items()}
        elif form.kind == "increment":
            raise ContractError("observation_use_bound_endpoint_increment")
        values.update(sources)
        rows = [Polynomial() for _ in form.row_ids]
        for term in form.terms:
            coefficient = Fraction(term.sign)
            for factor in term.factors:
                coefficient *= rational(factor.value)
            if term.divisor is not None:
                coefficient /= rational(term.divisor.value)
            operand = Polynomial((1,)) if term.field is None else values[term.field][term.index]
            rows[term.row] += coefficient*operand
        return tuple(rows)

    def integrate(self, polynomial, *, subinterval=None):
        self.require_covered()
        left, right = contained_subinterval(self.clock, subinterval)
        return self.clock.hused*polynomial.integral(self.clock.coordinate(left), self.clock.coordinate(right))

    def require_covered(self):
        if self.clock_policy == "reject_uncovered":
            self.clock.require_covered()
        # The alternative explicitly defines an observational extension. It
        # does not assert native getter legality or erase its required debit.


@dataclass(frozen=True)
class Enclosure:
    """A signed value and an absolute mathematical error in the same units."""

    center: Fraction
    radius: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, "center", rational(self.center))
        object.__setattr__(self, "radius", rational(self.radius))
        if self.radius < 0:
            raise ContractError("observation_negative_error_bound")

    def __add__(self, other):
        other = other if isinstance(other, Enclosure) else Enclosure(other)
        return Enclosure(self.center+other.center, self.radius+other.radius)

    __radd__ = __add__

    def __neg__(self):
        return Enclosure(-self.center, self.radius)

    def __sub__(self, other):
        return self+(-other if isinstance(other, Enclosure) else -rational(other))

    def scale(self, coefficient):
        coefficient = rational(coefficient)
        return Enclosure(self.center*coefficient, self.radius*abs(coefficient))

    @property
    def absolute_upper(self):
        return abs(self.center)+self.radius

    def payload(self):
        return {"center": ratio_payload(self.center), "absolute_error_bound": ratio_payload(self.radius)}


@dataclass(frozen=True)
class AbsoluteIntegralBound:
    """An upper bound, not an estimate of the absolute integral's value."""

    upper: Fraction
    squared_integral: Enclosure
    numerical_excess_bound: Fraction

    def __post_init__(self):
        object.__setattr__(self, "upper", rational(self.upper))
        object.__setattr__(self, "numerical_excess_bound", rational(self.numerical_excess_bound))
        if (self.upper < 0 or self.numerical_excess_bound < 0
                or self.squared_integral.center+self.squared_integral.radius < 0):
            raise ContractError("observation_invalid_absolute_integral_bound")

    def payload(self):
        return {"upper": ratio_payload(self.upper), "method": "Cauchy-Schwarz from a certified squared integral",
                "squared_integral": self.squared_integral.payload(),
                "numerical_excess_bound": ratio_payload(self.numerical_excess_bound),
                "is_integral_value_estimate": False}


def rational_sqrt_upper(value, rounding_error):
    """One outward dyadic square root using integer arithmetic only."""
    value, rounding_error = rational(value), rational(rounding_error)
    if value < 0 or rounding_error <= 0:
        raise ContractError("observation_sqrt_domain_or_error")
    if value == 0:
        return Fraction(0)
    bits = max(0, rounding_error.denominator.bit_length()-rounding_error.numerator.bit_length()+2)
    denominator = 1 << bits
    scaled_numerator = value.numerator*denominator**2
    root = math.isqrt(scaled_numerator//value.denominator)
    if root**2*value.denominator < scaled_numerator:
        root += 1
    return Fraction(root, denominator)


def certify_fixed4_partition(left, right, absolute_error, original, cells):
    """Check four same-function moment certificates using exact arithmetic.

    Function provenance belongs to the caller's path/port binding. This small
    reduction checks coverage and outward bounds, never fits the integrand or
    evaluates a physical model. Numerical excess is already inside each upper.
    """
    left, right, absolute_error = map(rational, (left, right, absolute_error))
    if left >= right or absolute_error <= 0 or type(cells) is not tuple or len(cells) != 4:
        raise ContractError("observation_partition_interval_or_count")

    def check(bound, length, allocation):
        if not isinstance(bound, AbsoluteIntegralBound):
            raise ContractError("observation_partition_missing_bound")
        moment = bound.squared_integral
        square = length*(moment.center+moment.radius)
        if (moment.radius > allocation**2/(8*length)
                or bound.upper**2 < square
                or max(Fraction(0), bound.upper-allocation/2)**2 > square
                or bound.numerical_excess_bound > allocation):
            raise ContractError("observation_partition_moment_or_outward_bound")

    check(original, right-left, absolute_error)
    for i, cell in enumerate(cells):
        a, b = left+(right-left)*Fraction(i, 4), left+(right-left)*Fraction(i+1, 4)
        if (not isinstance(cell, Mapping) or cell.get("left") != a or cell.get("right") != b
                or cell.get("allocation_C") != absolute_error/4):
            raise ContractError("observation_partition_coverage_or_allocation")
        check(cell["bound"], b-a, absolute_error/4)
    total = sum((cell["bound"].upper for cell in cells), Fraction(0))
    return {"cells": cells, "partition_upper_C": total,
            "retained_upper_C": min(original.upper, total)}


def public_action_enclosures(result):
    """Retain every returned word and the public action's own error bound."""
    words = result.value.words
    return tuple(Enclosure(sum((rational(word.flat[i]) for word in words), Fraction(0)),
                           rational(result.absolute_error_bound.values.flat[i]))
                 for i in range(result.value.shape[0]))


def contained_subinterval(clock, subinterval=None):
    """An observation range in an unchanged native clock; never a new step."""
    if subinterval is None:
        return clock.predecessor, clock.tn
    if not isinstance(subinterval, (tuple, list)) or len(subinterval) != 2:
        raise ContractError("observation_subinterval_shape")
    left, right = map(rational, subinterval)
    if not clock.predecessor <= left < right <= clock.tn:
        raise ContractError("observation_subinterval_outside_native_frame")
    return left, right


def affine_interval_actions(model, path: PolynomialPath, *, subinterval=None):
    """Exact affine endpoint/integral actions; no endpoint state is changed."""
    path.require_covered()
    first, last = contained_subinterval(path.clock, subinterval)
    left, right = path.clock.coordinate(first), path.clock.coordinate(last)
    names = ["storage", "body_charge", "metal_charge", "gauss_defect", "displacement"]
    if model.definition.dynamic:
        names.append("ion_inventory")
    result = {}
    for name in names:
        state = path.action(model.linear_forms["state"][name])
        rates = path.action(model.linear_forms["rate"][name])
        result[name] = {"left": tuple(p.at(left) for p in state), "right": tuple(p.at(right) for p in state),
                        "rate_integral": tuple(path.integrate(p) if subinterval is None
                                               else path.integrate(p, subinterval=subinterval) for p in rates),
                        "form_identity": model.linear_forms["state"][name].identity,
                        "rate_form_identity": model.linear_forms["rate"][name].identity}
    return result


def endpoint_charge_mismatch(model, path, left, right, *, subinterval=None):
    """Absolute endpoint mismatch to immutable full-word accepted Points."""
    first, last = contained_subinterval(path.clock, subinterval)
    if rational(left.time) != first or rational(right.time) != last:
        raise ContractError("observation_endpoint_time_mismatch")
    actions = (affine_interval_actions(model, path) if subinterval is None
               else affine_interval_actions(model, path, subinterval=subinterval))
    result = []
    for name in ("body_charge", "metal_charge"):
        actual_left = public_action_enclosures(model.linear_action(name, left))
        actual_right = public_action_enclosures(model.linear_action(name, right))
        result += [(a-Enclosure(x)).absolute_upper+(b-Enclosure(y)).absolute_upper
                   for a, b, x, y in zip(actual_left, actual_right,
                                         actions[name]["left"], actions[name]["right"], strict=True)]
    return tuple(result)


def segment_input_roundoff(segment, clock):
    """Outward bounds for the existing binary64 affine input evaluation.

    The comparison line uses the *stored rounded slope*, exactly, not a new
    idealized endpoint slope. Endpoint snapping defects are kept separately.
    This arithmetic bound assumes IEEE round-to-nearest with subnormals; the
    execution packet must still bind that assumption to its actual build.
    """
    if (segment.id != clock.segment_id or rational(segment.start) != clock.segment_start
            or rational(segment.end) != clock.segment_end):
        raise ContractError("observation_input_segment_binding")
    start_inputs, slopes = segment.inputs(segment.start)
    duration = rational(segment.end)-rational(segment.start)
    unit_roundoff, half_subnormal = Fraction(1, 2**53), Fraction(1, 2**1075)
    dt_bound = unit_roundoff*duration+half_subnormal
    result = []
    for start, slope, end in zip(start_inputs, slopes, (segment.voltage[1], segment.photons[1]), strict=True):
        start, slope, end = map(rational, (start, slope, end))
        product_error = abs(slope)*dt_bound+unit_roundoff*abs(slope)*(duration+dt_bound)+half_subnormal
        addition_error = unit_roundoff*(abs(start)+abs(slope)*duration+product_error)+half_subnormal
        result.append({"stored_slope": slope, "interior_arithmetic_bound": product_error+addition_error,
                       "end_snap_mismatch": abs(end-start-slope*duration),
                       "input_at_tn_exact_line": start+slope*(clock.tn-clock.segment_start)})
    return tuple(result)


@dataclass(frozen=True)
class ChargePrefix:
    """Absolute, non-resetting errors; a signed cancellation earns no budget."""

    budgets: tuple[Fraction, ...]
    absolute_defects: tuple[Fraction, ...]
    reference_errors: tuple[Fraction, ...]
    intervals: int = 0

    @classmethod
    def start(cls, budgets):
        budgets = tuple(rational(v) for v in budgets)
        if not budgets or any(v < 0 for v in budgets):
            raise ContractError("observation_charge_budget")
        return cls(budgets, (Fraction(0),)*len(budgets), (Fraction(0),)*len(budgets))

    def append(self, changes, integrals, additional_errors):
        if (any(len(v) != len(self.budgets) for v in (changes, integrals, additional_errors))
                or any(v is None for v in additional_errors)):
            raise ContractError("observation_missing_charge_error_term")
        defects, errors, passed = [], [], []
        for i, (change, integral, additional) in enumerate(zip(changes, integrals, additional_errors, strict=True)):
            additional = rational(additional)
            if additional < 0:
                raise ContractError("observation_negative_charge_error")
            defect = change-integral
            error = defect.radius+additional
            total = abs(defect.center)+error
            defects.append(self.absolute_defects[i]+total)
            errors.append(self.reference_errors[i]+error)
            passed.append(total <= self.budgets[i] and defects[-1] <= self.budgets[i]
                          and error*3 <= self.budgets[i] and errors[-1]*3 <= self.budgets[i])
        successor = ChargePrefix(self.budgets, tuple(defects), tuple(errors), self.intervals+1)
        return successor, tuple(passed)


class BallIntegrator:
    """Small optional Arb adapter with hard callback and observed-radius gates.

    python-flint 0.8.0 does not return acb_calc_integrate's convergence status.
    A requested tolerance or soft library work limit is therefore never the
    certificate: only the finite returned ball and its actual radius are used.
    """

    def __init__(self, *, bits=256, evaluations=2048, depth=16):
        try:
            import flint
        except ImportError as exc:
            raise ContractError("observation_optional_arb_unavailable") from exc
        if flint.__version__ != "0.8.0":
            raise ContractError("observation_arb_version_not_frozen")
        if (type(bits) is not int or not 128 <= bits <= 512
                or type(evaluations) is not int or not 1 <= evaluations <= 8192
                or type(depth) is not int or not 1 <= depth <= 32):
            raise ContractError("observation_arb_resource_parameters")
        self.flint, self.bits, self.evaluations, self.depth = flint, bits, evaluations, depth
        self.calls = []

    def number(self, value):
        value = rational(value)
        return self.flint.acb(self.flint.fmpq(value.numerator, value.denominator))

    def polynomial(self, polynomial, coordinate):
        result = self.number(0)
        for coefficient in reversed(polynomial.coefficients):
            result = result*coordinate+self.number(coefficient)
        return result

    @staticmethod
    def bernoulli(value):
        # 1F1(1;2;x)=(exp(x)-1)/x is entire, including x=0. Its
        # reciprocal has exactly the nonzero 2*pi*i*k Bernoulli poles.
        # Interval division rejects a denominator enclosure containing zero.
        return 1/value.hypgeom_1f1(1, 2)

    @staticmethod
    def _exact_arb(value):
        mantissa, exponent = value.man_exp()
        return Fraction(int(mantissa))*Fraction(2)**int(exponent)

    def enclosure(self, value):
        if not value.is_finite() or not value.imag.contains(0):
            raise ContractError("observation_nonfinite_or_nonreal_ball")
        return Enclosure(self._exact_arb(value.real.mid()), self._exact_arb(value.real.rad()))

    def point(self, function, coordinate):
        with self.flint.ctx.workprec(self.bits):
            return self.enclosure(function(self.number(coordinate), False))

    def integrate(self, function, left, right, absolute_error, *, absolute=False):
        left, right, absolute_error = map(rational, (left, right, absolute_error))
        if right < left or absolute_error <= 0:
            raise ContractError("observation_integral_interval_or_error")
        calls = 0

        def callback(value, analytic):
            nonlocal calls
            calls += 1
            if calls > self.evaluations:
                raise ContractError("observation_integral_evaluation_cap")
            value = function(value, analytic)
            return value.real_abs(analytic=analytic) if absolute else value

        with self.flint.ctx.workprec(self.bits):
            result = self.flint.acb.integral(
                callback, self.number(left), self.number(right),
                abs_tol=self.number(absolute_error).real, rel_tol=self.number(Fraction(1, 2**self.bits)).real,
                eval_limit=self.evaluations, depth_limit=self.depth, deg_limit=64)
            enclosure = self.enclosure(result)
        self.calls.append({"evaluations": calls, "actual_error": ratio_payload(enclosure.radius),
                           "requested_error": ratio_payload(absolute_error), "absolute_integrand": absolute})
        if enclosure.radius > absolute_error:
            raise ContractError("observation_integral_radius_exceeds_budget")
        return enclosure

    def absolute_bound(self, function, left, right, absolute_error):
        """Bound a real-valued function's L1 norm; zeros need no searches.

        The fixed 512-bit second-moment evaluation avoids trying to integrate
        a nonanalytic absolute-value kink. Cauchy-Schwarz slack is disclosed
        as a conservative bound, never called quadrature error or removed.
        The callback must supply valid interval enclosures: its finite range
        on the entire real input interval must have exactly zero imaginary
        part. An imaginary *integral* enclosure containing zero is not proof.
        """
        left, right, absolute_error = map(rational, (left, right, absolute_error))
        length = right-left
        if length <= 0 or absolute_error <= 0:
            raise ContractError("observation_absolute_integral_interval_or_error")
        with self.flint.ctx.workprec(512):
            midpoint, radius = (left+right)/2, length/2
            real_interval = self.flint.acb(self.flint.arb(
                self.flint.fmpq(midpoint.numerator, midpoint.denominator),
                self.flint.fmpq(radius.numerator, radius.denominator)))
            value_range = self.flint.acb(function(real_interval, False))
            if not value_range.is_finite():
                raise ContractError("observation_absolute_nonfinite_real_range")
            if not value_range.imag.is_zero():
                raise ContractError("observation_absolute_requires_real_enclosure")
            self.calls.append({"evaluations": 1, "purpose": "full-real-interval-precheck", "working_bits": 512})
            if value_range.real.is_zero():
                return AbsoluteIntegralBound(Fraction(0), Enclosure(0), Fraction(0))
        if self.evaluations < 2:
            raise ContractError("observation_integral_evaluation_cap")
        moment_goal = absolute_error**2/(8*length)
        moment_arithmetic = BallIntegrator(bits=512, evaluations=self.evaluations-1, depth=self.depth)

        def real_square(z, analytic):
            value = self.flint.acb(function(z, analytic))
            if not analytic and not value.imag.is_zero():
                raise ContractError("observation_absolute_requires_real_enclosure")
            return value**2

        moment = moment_arithmetic.integrate(real_square, left, right, moment_goal)
        self.calls.extend(dict(row, purpose="L2-moment", working_bits=512) for row in moment_arithmetic.calls)
        moment_upper = moment.center+moment.radius
        if moment_upper < 0:
            raise ContractError("observation_negative_squared_integral")
        upper = rational_sqrt_upper(length*moment_upper, absolute_error/2)
        # Since the true second moment is in the returned ball, its distance
        # from the upper endpoint is <=2*radius. The square-root inequality
        # bounds the numerical excess by sqrt(2*length*radius)+rounding_error.
        return AbsoluteIntegralBound(upper, moment, absolute_error)


class SlabPathObserver:
    """Two fixed slab models' physical observations on one declared path.

    The charge-current formulas eliminate *identical* optical/capture terms
    algebraically. They do not replace a finite physical residual with zero.
    The tangent is an explicitly named same-state affine-constraint estimate,
    not an IDA derivative, a changed state, or an additional trajectory.
    """

    def __init__(self, model, path: PolynomialPath):
        from perovskite_sim.constants import Q

        if (model.definition.id not in {"S0NeutralPublicDeviceV1", "DynamicAcceptorIonPublicDeviceV1"}
                or model.source_identity != path.source_identity
                or model.layout.identity != path.layout_identity or len(path.inputs) != 2):
            raise ContractError("observation_slab_scope_or_source")
        self.model, self.path, self.m = model, path, model.definition
        self.q, self.area = rational(Q), rational(self.m.area)
        self.volumes = tuple(rational(v) for v in model.geometry.volumes)
        self.dx = tuple(rational(v) for v in model.dx)
        if any(v <= 0 for v in (*self.volumes, *self.dx)):
            raise ContractError("observation_geometry_positive_required")
        self.length = sum(self.dx, Fraction(0))
        self.weights = (tuple(sum(self.dx[i:], Fraction(0))/self.length for i in range(model.count)),
                        tuple(sum(self.dx[:i], Fraction(0))/self.length for i in range(model.count)))
        self.affine = affine_interval_actions(model, path)
        self.domain_bounds = self._require_domain()
        fields, h = path.fields, path.clock.hused
        self.xi = tuple((b-a)/rational(self.m.vt)
                        for a, b in zip(fields["phi_V"][:-1], fields["phi_V"][1:], strict=True))
        self.dn = tuple(b-a for a, b in zip(fields["n_m3"][:-1], fields["n_m3"][1:], strict=True))
        self.dp = tuple(b-a for a, b in zip(fields["p_m3"][:-1], fields["p_m3"][1:], strict=True))
        if self.m.dynamic:
            self.dc = tuple(b-a for a, b in zip(fields["c_m3"][:-1], fields["c_m3"][1:], strict=True))
            cn, cp, n1, p1 = map(rational, (self.m.capture_n, self.m.capture_p, self.m.n1, self.m.p1))
            self.capture_difference = tuple(cn*(n*(1-f)-n1*f)-cp*(p*f-p1*(1-f))
                                            for n, p, f in zip(fields["n_m3"], fields["p_m3"], fields["f"], strict=True))
        else:
            self.capture_difference = (Polynomial(),)*model.count
        self.endpoint_carrier_rate = tuple(
            self.q*self.volumes[i]*(fields["p_m3"][i]-fields["n_m3"][i]).derivative()/h
            for i in (0, model.count-1))
        self.capture_charge_rate = tuple(-self.q*v*rational(self.m.trap_density)*r
                                         for v, r in zip(self.volumes, self.capture_difference, strict=True))
        self.raw_metal = path.action(model.linear_forms["rate"]["metal_charge"])
        voltage_rate = path.inputs[0].derivative()/h
        capacitance = self.area*rational(self.m.epsilon)/self.length
        # Only endpoint carrier storage is fixed by the ideal reservoirs.
        # Interior capture cancels against dynamic trap charge, not at contacts.
        self.tangent_metal_polynomial = tuple(
            sign*capacitance*voltage_rate
            +sum((-weights[i]*self.capture_charge_rate[i] for i in (0, model.count-1)), Polynomial())
            for sign, weights in zip((1, -1), self.weights, strict=True))

    def _require_domain(self):
        left = self.path.clock.coordinate(self.path.clock.predecessor)
        result = {}
        for spec in self.model.layout.variables:
            bounds = tuple(p.range(left, 0) for p in self.path.fields[spec.id])
            for lower, upper in bounds:
                if spec.id in {"n_m3", "p_m3"} and lower <= 0:
                    raise ContractError("observation_nonpositive_active_carrier:"+spec.id)
                if spec.lower is not None and lower < rational(spec.lower):
                    raise ContractError("observation_path_domain_lower:"+spec.id)
                if spec.upper is not None and upper > rational(spec.upper):
                    raise ContractError("observation_path_domain_upper:"+spec.id)
                if spec.id == "c_m3" and upper >= rational(self.m.ion_capacity):
                    raise ContractError("observation_vacancy_log_domain")
            result[spec.id] = bounds
        return MappingProxyType(result)

    def fluxes(self, arithmetic: BallIntegrator, coordinate, analytic=False):
        a, m, fields = arithmetic, self.m, self.path.fields
        evaluate = lambda polynomial: a.polynomial(polynomial, coordinate)
        number = a.number
        carrier, ion = [], []
        for i, xi_polynomial in enumerate(self.xi):
            xi = evaluate(xi_polynomial)
            bernoulli = a.bernoulli(xi)
            jn = number(self.q*rational(m.mu_n)*rational(m.vt)/self.dx[i])*(
                bernoulli*evaluate(self.dn[i])-xi*evaluate(fields["n_m3"][i]))
            jp = number(self.q*rational(m.mu_p)*rational(m.vt)/self.dx[i])*(
                -bernoulli*evaluate(self.dp[i])-xi*evaluate(fields["p_m3"][i+1]))
            carrier.append(jn+jp)
            if m.dynamic:
                # Exact rational polynomial differences preserve the vacancy
                # and weak ion changes before the interval nonlinear call.
                vacancy_left = evaluate(rational(m.ion_capacity)-fields["c_m3"][i])
                vacancy_right = evaluate(rational(m.ion_capacity)-fields["c_m3"][i+1])
                drive = xi+(vacancy_left/vacancy_right).log(analytic=analytic)
                ion.append(number(rational(m.diffusion_ion)/self.dx[i])*(
                    -a.bernoulli(drive)*evaluate(self.dc[i])-drive*evaluate(fields["c_m3"][i+1])))
            else:
                ion.append(number(0))
        return tuple(carrier), tuple(ion)

    def tangent_metal_flux(self, arithmetic, coordinate, analytic=False):
        """Closed form of the constant Dirichlet Poisson tangent action."""
        carrier, ion = self.fluxes(arithmetic, coordinate, analytic)
        number = arithmetic.number
        total = [j+number(self.q)*f for j, f in zip(carrier, ion, strict=True)]
        charge_rates = [number(-self.q*self.area)*ion[0]]
        charge_rates += [number(self.area)*(a-b) for a, b in zip(total[:-1], total[1:], strict=True)]
        charge_rates += [number(self.q*self.area)*ion[-1]]
        return tuple(sum((-number(w)*b for w, b in zip(weights, charge_rates, strict=True)), number(0))
                     for weights in self.weights)

    def currents(self, arithmetic, coordinate, analytic=False):
        a = arithmetic
        carrier, _ = self.fluxes(a, coordinate, analytic)
        tangent_flux = self.tangent_metal_flux(a, coordinate, analytic)
        raw, tangent = [], []
        for side, node, face, sign in ((0, 0, 0, 1), (1, self.model.count-1, self.model.count-2, -1)):
            reservoir = a.number(sign*self.area)*carrier[face]+a.polynomial(self.capture_charge_rate[node], coordinate)
            raw.append(reservoir+a.polynomial(self.endpoint_carrier_rate[side]+self.raw_metal[side], coordinate))
            tangent.append(reservoir+a.polynomial(self.tangent_metal_polynomial[side], coordinate)+tangent_flux[side])
        return {"raw_polynomial_current": tuple(raw), "same_state_affine_tangent_current": tuple(tangent)}

    def integrate(self, arithmetic: BallIntegrator, absolute_error, *, partition_cells=None, subinterval=None):
        """Signed charge integrals and conservative absolute raw/tangent gaps.

        ``absolute_error`` is an observation allocation supplied by the caller,
        not a new physical gate. All returned radii still enter the original
        interval and non-resetting prefix reference budget.
        """
        if partition_cells is not None and (type(partition_cells) is not int or partition_cells != 4):
            raise ContractError("observation_unsupported_partition")
        a, path = arithmetic, self.path
        path.require_covered()
        first, last = contained_subinterval(path.clock, subinterval)
        lower, upper, h = path.clock.coordinate(first), path.clock.coordinate(last), path.clock.hused
        integrate_polynomial = path.integrate if subinterval is None else lambda p: path.integrate(p, subinterval=subinterval)
        raw_conduction, tangent_conduction, tangent_metal, gaps = [], [], [], []
        partitions = []
        for side, node, face, sign in ((0, 0, 0, 1), (1, self.model.count-1, self.model.count-2, -1)):
            nonlinear = a.integrate(
                lambda u, analytic: a.number(sign*self.area*h)*self.fluxes(a, u, analytic)[0][face],
                lower, upper, absolute_error)
            tangent_conduction.append(nonlinear+integrate_polynomial(self.capture_charge_rate[node]))
            raw_conduction.append(tangent_conduction[-1]+integrate_polynomial(self.endpoint_carrier_rate[side]))
            flux_integral = a.integrate(
                lambda u, analytic: a.number(h)*self.tangent_metal_flux(a, u, analytic)[side],
                lower, upper, absolute_error)
            tangent_metal.append(flux_integral+integrate_polynomial(self.tangent_metal_polynomial[side]))
            difference = self.endpoint_carrier_rate[side]+self.raw_metal[side]-self.tangent_metal_polynomial[side]
            def departure(u, analytic):
                return a.number(h)*(a.polynomial(difference, u)
                                     -self.tangent_metal_flux(a, u, analytic)[side])

            first_call = len(a.calls)
            gaps.append(a.absolute_bound(departure, lower, upper, absolute_error))
            if partition_cells is not None:
                original_calls = (first_call, len(a.calls))
                cells = []
                for i in range(4):
                    left, right = lower+(upper-lower)*Fraction(i, 4), lower+(upper-lower)*Fraction(i+1, 4)
                    allocation = rational(absolute_error)/4
                    first_call = len(a.calls)
                    bound = a.absolute_bound(departure, left, right, allocation)
                    cells.append({"left": left, "right": right, "allocation_C": allocation,
                                  "bound": bound, "call_range": (first_call, len(a.calls))})
                port = ("left_metal", "right_metal")[side]
                partitions.append({"port": port,
                    "integrand_identity": identity({"schema": "solarlab.terminal-departure.v1",
                        "path_identity": path.identity, "port": port,
                        "formula": "hused*(endpoint_carrier_rate+raw_metal-tangent_metal_polynomial-tangent_metal_flux)"}),
                    "absolute_error_C": rational(absolute_error), "original": gaps[-1],
                    "original_call_range": original_calls,
                    **certify_fixed4_partition(lower, upper, absolute_error, gaps[-1], tuple(cells))})
        # Total-current and body/metal charge rows are different quantities.
        # A terminal L1 bound alone cannot bound its two charge contributions.
        # The endpoint-carrier terms are polynomials, so their second moments
        # are exact and need no further nonlinear quadrature.
        def polynomial_l1(polynomial):
            integrand = h*polynomial
            moment = (integrand*integrand).integral(lower, upper)
            return rational_sqrt_upper((upper-lower)*moment, rational(absolute_error)/2)

        endpoint_l1 = tuple(polynomial_l1(p) for p in self.endpoint_carrier_rate)
        body_l1 = polynomial_l1(sum(self.endpoint_carrier_rate, Polynomial()))
        terminal_upper = (tuple(row["retained_upper_C"] for row in partitions) if partition_cells is not None
                          else tuple(gap.upper for gap in gaps))
        charge_gap = (body_l1, *(gap+endpoint for gap, endpoint in zip(terminal_upper, endpoint_l1, strict=True)))
        result = {"raw_polynomial": (sum(raw_conduction, Enclosure(0)),
                                    *(Enclosure(integrate_polynomial(p)) for p in self.raw_metal)),
                "same_state_affine_tangent": (sum(tangent_conduction, Enclosure(0)), *tangent_metal),
                "raw_tangent_L1_upper_bounds": tuple(gaps),
                "raw_tangent_charge_L1_upper_C": charge_gap,
                "charge_gap_method": "exact polynomial second moment for body; total-current L1 plus endpoint-carrier L1 for each metal",
                "path_identity": path.identity,
                "certification_scope": "represented polynomial physical observations only",
                "DAE_time_accuracy_certified": False, "continuum_space_accuracy_certified": False}
        if partition_cells is not None:
            result["terminal_partition"] = {"schema": "solarlab.terminal-partition.v1",
                "path_identity": path.identity, "normalized_interval": (lower, upper),
                "cells_per_port": 4, "ports": tuple(partitions),
                "endpoint_carrier_L1_upper_C": endpoint_l1, "body_L1_upper_C": body_l1}
        return result

    def strip_current_debit(self, arithmetic: BallIntegrator, *, subinterval=None):
        """Conservative absolute current-charge debit on a declared extension.

        The output is separate from the signed integral and endpoint mismatch.
        A future native packet also needs actual same-snapshot getter success
        at the true predecessor. A synthetic path cannot supply that evidence.
        """
        first, last = contained_subinterval(self.path.clock, subinterval)
        strip_end = min(last, self.path.clock.native_left)
        width = max(Fraction(0), strip_end-first)
        if not width:
            return {"raw_polynomial": (Fraction(0),)*3,
                    "same_state_affine_tangent": (Fraction(0),)*3}
        if self.path.clock_policy != "declared_polynomial_extension":
            raise ContractError("observation_strip_not_declared")
        a, path = arithmetic, self.path
        left, right = path.clock.coordinate(first), path.clock.coordinate(strip_end)
        with a.flint.ctx.workprec(a.bits):
            midpoint, radius = (left+right)/2, (right-left)/2
            ball = a.flint.arb(a.flint.fmpq(midpoint.numerator, midpoint.denominator),
                               a.flint.fmpq(radius.numerator, radius.denominator))
            coordinate = a.flint.acb(ball)
            carrier, _ = self.fluxes(a, coordinate)
            flux_metal = self.tangent_metal_flux(a, coordinate)
            raw_cond, tangent_cond, raw_metal, tangent_metal = [], [], [], []
            for side, node, face, sign in ((0, 0, 0, 1), (1, self.model.count-1, self.model.count-2, -1)):
                tangent_cond.append(a.number(sign*self.area)*carrier[face]
                                    +a.polynomial(self.capture_charge_rate[node], coordinate))
                raw_cond.append(tangent_cond[-1]+a.polynomial(self.endpoint_carrier_rate[side], coordinate))
                raw_metal.append(a.polynomial(self.raw_metal[side], coordinate))
                tangent_metal.append(flux_metal[side]+a.polynomial(self.tangent_metal_polynomial[side], coordinate))
            values = {"raw_polynomial": (sum(raw_cond, a.number(0)), *raw_metal),
                      "same_state_affine_tangent": (sum(tangent_cond, a.number(0)), *tangent_metal)}
            return {key: tuple(a.enclosure(value).absolute_upper*width for value in row)
                    for key, row in values.items()}


def native_observation_readiness(*, native_basis, input_map_error, endpoint_error,
                                 clock_error, nonlinear_error, tangent_error, independent_review):
    """Expose missing terms, never convert component tests into admission."""
    terms = dict(native_basis=native_basis, input_map_error=input_map_error,
                 endpoint_error=endpoint_error, clock_error=clock_error,
                 nonlinear_error=nonlinear_error, tangent_error=tangent_error,
                 independent_review=independent_review)
    return {"missing": tuple(name for name, value in terms.items() if value is None),
            "native_admitted": False, "F06": False, "G2": False,
            "full_protocols_pending": {"S0_s": 1.2e-6, "dynamic_ion_trap_s": 9.2},
            "time_and_space_accuracy": "requires the original independent refinements"}
