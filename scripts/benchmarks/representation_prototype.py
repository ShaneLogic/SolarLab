"""Constant physical gauge, independent log references and solver unit charts.

The physical Point/reference/history is retained. Binary64 chart coordinates
have an explicit inverse enclosure; they are never asserted to be an exact
restoration of transcendental coordinates. This module changes no equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from types import MappingProxyType

import numpy as np

from scripts.benchmarks.contract_prototype import (
    ContractError, ImmutableArrays, Layout, Linearization, ONE, PARTICLE, Point, RateView,
    SparseLinearization, Unit, VOLT, VOLUME, frozen_array,
)
from scripts.benchmarks.interval_observation import BallIntegrator, Enclosure
from scripts.benchmarks.precision_prototype import (
    FrameInputExpansion, physical_state_words,
)


def rational(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ContractError("representation_nonfinite")
    return Fraction(float(value)) if isinstance(value, np.floating) else Fraction(value)


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_values(point):
    words = physical_state_words(point.state)
    return tuple(sum((rational(w.flat[i]) for w in words[v.id]), Fraction())
                 for v in point.state.layout.variables for i in range(math.prod(v.shape)))


def rate_values(rate):
    return tuple(sum((rational(w[i]) for w in rate.words), Fraction()) for i in range(rate.shape[0]))


def upper_float(value):
    value = rational(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ContractError("representation_weight_range") from error
    if not math.isfinite(result) or result <= 0:
        raise ContractError("representation_weight_range")
    if rational(result) < value:
        result = float(np.nextafter(result, math.inf))
    if not math.isfinite(result):
        raise ContractError("representation_weight_range")
    return result


@dataclass(frozen=True)
class UnitExpression:
    """Dimensions and exact SI amount per one internal solver unit."""
    dimension: Unit
    si_per_unit: Fraction
    name: str

    def __post_init__(self):
        scale = rational(self.si_per_unit)
        if not isinstance(self.dimension, Unit) or scale <= 0 or not self.name:
            raise ContractError("representation_invalid_unit_expression")
        object.__setattr__(self, "si_per_unit", scale)


@dataclass(frozen=True)
class PhysicalAllowance:
    """Unchanged SI component widths supplied by the original error promise."""
    point_identity: str
    reference_identity: str
    source_identity: str
    layout_identity: str
    widths_si: tuple[Fraction, ...]

    def __post_init__(self):
        widths = tuple(map(rational, self.widths_si))
        if not widths or any(v <= 0 for v in widths):
            raise ContractError("representation_nonpositive_physical_allowance")
        object.__setattr__(self, "widths_si", widths)


@dataclass(frozen=True)
class RepresentationImage(ImmutableArrays):
    point: Point
    rate: RateView
    representation_identity: str
    coordinates: np.ndarray
    coordinate_rate: np.ndarray
    contact_potentials_V: tuple[Fraction, Fraction]
    contact_rates_V_s: tuple[Fraction, Fraction]

    def __post_init__(self):
        object.__setattr__(self, "coordinates", frozen_array(self.coordinates))
        object.__setattr__(self, "coordinate_rate", frozen_array(self.coordinate_rate))
        object.__setattr__(self, "contact_potentials_V", tuple(map(rational,self.contact_potentials_V)))
        object.__setattr__(self, "contact_rates_V_s", tuple(map(rational,self.contact_rates_V_s)))

    @property
    def identity(self):
        return identity({"map": self.representation_identity, "Point": self.point.identity,
            "rate": self.rate.identity, "z": [float(x).hex() for x in self.coordinates],
            "zdot": [float(x).hex() for x in self.coordinate_rate],
            "contacts": list(map(str, self.contact_potentials_V)),
            "contact_rates": list(map(str, self.contact_rates_V_s))})


@dataclass(frozen=True)
class InverseImage:
    state_si: tuple[Enclosure, ...]
    rate_si: tuple[Enclosure, ...]
    inputs_si: tuple[Fraction, ...]
    input_rates_si: tuple[Fraction, ...]


@dataclass(frozen=True)
class WeightPromise(ImmutableArrays):
    weights: np.ndarray
    representation_debit_si: tuple[Fraction, ...]
    widths_si: tuple[Fraction, ...]
    eta: Fraction
    beta: Fraction
    component_norm_bound: int
    image_identity: str

    def __post_init__(self):
        object.__setattr__(self, "weights", frozen_array(self.weights))


@dataclass(frozen=True, init=False, eq=False)
class PhysicalRepresentation:
    """One constant chart bound to a physical source and its actual reference.

    Gauge: all electrostatic potentials and both contact potentials translate
    by the same constant, in volts. This is not a change of voltage difference
    or a solver origin. Nonuniform/time-dependent gauges are not supported.
    Log: n=Nref*exp(s*z), with positive independent Nref (SI density).
    Linear: physical field=s*z (potential=s*z-gauge). Unit scales, Nref and
    gauge are fixed; the original material/physical reference never changes.
    """
    def __init__(self, layout: Layout, source_identity: str, reference: Point, *,
                 potential_field="phi_V", voltage_input=0, right_contact_sign=-1,
                 gauge_offset_V=0, reference_densities=None, column_units=None):
        if not source_identity or reference.state.layout.identity != layout.identity:
            raise ContractError("representation_physical_binding")
        for name, value in (("layout", layout), ("source_identity", source_identity), ("reference", reference),
                            ("potential_field", potential_field), ("voltage_input", voltage_input)):
            object.__setattr__(self, name, value)
        if right_contact_sign not in (-1, 1) or not 0 <= voltage_input < reference.inputs.size:
            raise ContractError("representation_contact_convention")
        object.__setattr__(self, "right_contact_sign", right_contact_sign)
        object.__setattr__(self, "gauge", rational(gauge_offset_V))
        specs = {v.id: v for v in layout.variables}
        if potential_field not in specs or specs[potential_field].unit != VOLT:
            raise ContractError("representation_potential_dimension")
        references, units = dict(reference_densities or {}), dict(column_units or {})
        if not set(references).issubset(specs) or not set(units).issubset(specs):
            raise ContractError("representation_unknown_field")
        references = {k: rational(v) for k, v in references.items()}
        for name, value in references.items():
            if value <= 0 or specs[name].unit != PARTICLE/VOLUME:
                raise ContractError("representation_log_reference_dimension_or_domain")
        for name, spec in specs.items():
            dimension = ONE if name in references else spec.unit
            units.setdefault(name, UnitExpression(dimension, Fraction(1), "SI"))
            if not isinstance(units[name], UnitExpression) or units[name].dimension != dimension:
                raise ContractError("representation_column_dimension")
        object.__setattr__(self, "references", MappingProxyType(references))
        object.__setattr__(self, "units", MappingProxyType(units))
        object.__setattr__(self, "names", tuple(v.id for v in layout.variables for _ in range(math.prod(v.shape))))
        object.__setattr__(self, "scales", tuple(units[name].si_per_unit for name in self.names))
        object.__setattr__(self, "identity", identity({"schema": "solarlab.physical-representation.v1", "layout": layout.identity,
            "source": source_identity, "reference": reference.identity, "gauge_V": str(self.gauge),
            "potential_field": potential_field, "voltage_input": voltage_input,
            "right_contact_sign": right_contact_sign, "log_reference_density_SI": {k: str(v) for k,v in references.items()},
            "units": {k: {"dimensions": v.dimension.powers, "SI_per_unit": str(v.si_per_unit), "name": v.name}
                      for k,v in units.items()}, "arithmetic": "existing-Arb256-enclosures"}))
        object.__setattr__(self, "arithmetic", BallIntegrator(bits=256))

    def _physical(self, point, rate):
        if point.state.layout.identity != self.layout.identity or point.inputs.shape != self.reference.inputs.shape:
            raise ContractError("representation_layout_mismatch")
        if self.arithmetic.bits != 256:
            raise ContractError("representation_arithmetic_profile_changed")
        rate.validate(point, rate.input_rate, self.source_identity)
        for variable in self.layout.variables:
            self.reference.state.field(variable.id).difference(point.state.field(variable.id))
        y, yd = state_values(point), rate_values(rate)
        if any(y[i] <= 0 for i,name in enumerate(self.names) if name in self.references):
            raise ContractError("representation_log_nonpositive_density")
        return y, yd

    def physical_expression(self, point):
        """Full-word physical field values in the translated potential gauge.

        Applying this to ``reference`` gives the expressed reference without
        replacing its material, original authority or retained history.
        """
        if point.state.layout.identity != self.layout.identity:
            raise ContractError("representation_layout_mismatch")
        for variable in self.layout.variables:
            self.reference.state.field(variable.id).difference(point.state.field(variable.id))
        return tuple(v+(self.gauge if name==self.potential_field else 0)
                     for v,name in zip(state_values(point),self.names,strict=True))

    def forward(self, point: Point, rate: RateView) -> RepresentationImage:
        y, yd = self._physical(point, rate)
        expressed = self.physical_expression(point)
        z, zd = [], []
        for i, (name, scale) in enumerate(zip(self.names, self.scales, strict=True)):
            if name in self.references:
                value = self.arithmetic.point(lambda x, _: x.log(), y[i]/self.references[name])
                z.append(float(value.center/scale))
                zd.append(float(yd[i]/(scale*y[i])))
            else:
                z.append(float(expressed[i]/scale))
                zd.append(float(yd[i]/scale))
        voltage, slope = rational(point.inputs[self.voltage_input]), rational(rate.input_rate[self.voltage_input])
        return RepresentationImage(point, rate, self.identity, z, zd,
            (self.gauge, self.gauge+self.right_contact_sign*voltage),
            (Fraction(), self.right_contact_sign*slope))

    def _image(self, image):
        if image.representation_identity != self.identity:
            raise ContractError("representation_foreign_image")
        self._physical(image.point, image.rate)
        left, right = image.contact_potentials_V
        ld, rd = image.contact_rates_V_s
        if (left != self.gauge or ld != 0
                or (right-left)/self.right_contact_sign != rational(image.point.inputs[self.voltage_input])
                or (rd-ld)/self.right_contact_sign != rational(image.rate.input_rate[self.voltage_input])):
            raise ContractError("representation_contact_input_mismatch")

    def inverse(self, image, *, coordinates=None, coordinate_rate=None) -> InverseImage:
        self._image(image)
        z = image.coordinates if coordinates is None else frozen_array(coordinates)
        zd = image.coordinate_rate if coordinate_rate is None else frozen_array(coordinate_rate)
        if z.shape != (self.layout.size,) or zd.shape != z.shape:
            raise ContractError("representation_coordinate_shape")
        states, rates = [], []
        for i, (name, scale) in enumerate(zip(self.names, self.scales, strict=True)):
            if name in self.references:
                physical = self.arithmetic.point(lambda x, _: self.arithmetic.number(self.references[name])
                                                  *(x*self.arithmetic.number(scale)).exp(), rational(z[i]))
                scaled = physical.scale(scale*rational(zd[i]))
                rounded = self.arithmetic.point(lambda x, _: x, scaled.center)
                velocity = Enclosure(rounded.center, rounded.radius+scaled.radius)
            else:
                exact = scale*rational(z[i])-(self.gauge if name == self.potential_field else 0)
                physical = self.arithmetic.point(lambda x, _: x, exact)
                velocity = self.arithmetic.point(lambda x, _: x, scale*rational(zd[i]))
            states.append(physical); rates.append(velocity)
        inputs, slopes = list(map(rational, image.point.inputs)), list(map(rational, image.rate.input_rate))
        inputs[self.voltage_input] = (image.contact_potentials_V[1]-image.contact_potentials_V[0])/self.right_contact_sign
        slopes[self.voltage_input] = (image.contact_rates_V_s[1]-image.contact_rates_V_s[0])/self.right_contact_sign
        return InverseImage(tuple(states), tuple(rates), tuple(inputs), tuple(slopes))

    def physical_trial(self, model, image, *, coordinates=None, coordinate_rate=None):
        """Use the public mapped12 decoder; preserve the original Point as history.

        Enclosure centers become an explicitly new Point/rate. Every omitted
        mathematical remainder is retained in the returned inverse enclosure;
        a center that cannot fit the existing profile is rejected, never clipped.
        """
        if model.source_identity != self.source_identity or model.reference.identity != self.reference.identity:
            raise ContractError("representation_trial_model_mismatch")
        decoded = self.inverse(image, coordinates=coordinates, coordinate_rate=coordinate_rate)
        root = state_values(self.reference)

        def words(values):
            remaining, result = list(values), []
            for _ in range(12):
                high = frozen_array([float(v) for v in remaining]); result.append(high)
                remaining = [v-rational(h) for v,h in zip(remaining, high, strict=True)]
            if any(remaining):
                raise ContractError("representation_existing_profile_capacity")
            return FrameInputExpansion(tuple(result))

        state = words([v.center-r for v,r in zip(decoded.state_si, root, strict=True)])
        rate = words([v.center for v in decoded.rate_si])
        point, increment = model.trial(state, image.point.time, image.point.inputs,
            predecessor=image.point, transition_representation="paired-endpoints-v1")
        model.validate(point)
        physical_rate = RateView(point, rate, image.rate.input_rate, source_identity=self.source_identity,
            mapping_identity=self.identity, origin="mapped-coordinate-rate",
            raw_coordinates=image.coordinates if coordinates is None else coordinates,
            raw_rate=image.coordinate_rate if coordinate_rate is None else coordinate_rate)
        return point, physical_rate, increment, decoded

    def pullback(self, point, rate, physical):
        """Fz=Fy*A+Fydot*B, Fzdot=Fydot*A, at the supplied physical Point/rate.

        A=diag(s*n), B=diag(s*ndot) for log fields; A=s,B=0 for linear.
        Existing nonlinear storage, input and time partials are carried through.
        Every binary64 matrix output has an exact, separately returned rounding
        error. Sparse slot structure is unchanged, including explicit zero slots.
        """
        y, yd = self._physical(point, rate)
        a = [s*y[i] if name in self.references else s for i,(name,s) in enumerate(zip(self.names,self.scales))]
        b = [s*yd[i] if name in self.references else Fraction() for i,(name,s) in enumerate(zip(self.names,self.scales))]
        sparse = isinstance(physical, SparseLinearization)
        if sparse and physical.source_identity != self.source_identity:
            raise ContractError("representation_linearization_source_mismatch")
        fy, fd = physical.y, physical.ydot
        if sparse:
            columns = np.repeat(np.arange(self.layout.size), np.diff(physical.structure.indptr))
            yy, dd = fy.data, fd.data
        else:
            yy, dd = np.asarray(fy).ravel(), np.asarray(fd).ravel()
            if fy.shape != fd.shape or fy.shape[1] != self.layout.size:
                raise ContractError("representation_linearization_shape")
            columns = np.tile(np.arange(self.layout.size), fy.shape[0])
        exact_y = tuple(rational(v)*a[j]+rational(w)*b[j] for v,w,j in zip(yy,dd,columns,strict=True))
        exact_d = tuple(rational(w)*a[j] for w,j in zip(dd,columns,strict=True))
        out_y, out_d = frozen_array([float(v) for v in exact_y]), frozen_array([float(v) for v in exact_d])
        errors = {"y": tuple(abs(v-rational(w)) for v,w in zip(exact_y,out_y,strict=True)),
                  "ydot": tuple(abs(v-rational(w)) for v,w in zip(exact_d,out_d,strict=True)),
                  "A": tuple(a), "B": tuple(b), "Point_identity": point.identity, "rate_identity": rate.identity}
        if sparse:
            result = SparseLinearization(physical.structure.filled(out_y), physical.structure.filled(out_d),
                physical.inputs, physical.input_rate, physical.time, physical.structure, self.identity)
        else:
            result = Linearization(out_y.reshape(fy.shape), out_d.reshape(fd.shape),
                                   physical.inputs, physical.input_rate, physical.time)
        return result, errors

    def error_weights(self, image, allowance: PhysicalAllowance) -> WeightPromise:
        self._image(image)
        if (allowance.point_identity != image.point.identity or allowance.reference_identity != self.reference.identity
                or allowance.source_identity != self.source_identity or allowance.layout_identity != self.layout.identity
                or len(allowance.widths_si) != self.layout.size):
            raise ContractError("representation_allowance_binding")
        decoded, actual = self.inverse(image), state_values(image.point)
        # This single debit includes forward rounding AND inverse arithmetic;
        # adding a separate debit for either side would count that error twice.
        debit = tuple(abs(v.center-y)+v.radius for v,y in zip(decoded.state_si,actual,strict=True))
        eta = max(e/d for e,d in zip(debit,allowance.widths_si,strict=True))
        if 3*eta > 1:
            raise ContractError("representation_arithmetic_share_exceeded")
        beta, r = 1-eta, math.isqrt(self.layout.size-1)+1
        weights = []
        for i,(name,scale) in enumerate(zip(self.names,self.scales)):
            width = beta*allowance.widths_si[i]
            if name in self.references:
                nmax = decoded.state_si[i].center+decoded.state_si[i].radius
                log = self.arithmetic.point(lambda x, _: x.log(), 1+r*width/nmax)
                lower = (log.center-log.radius)/(scale*r)
                if lower <= 0:
                    raise ContractError("representation_log_weight_unresolved")
                weights.append(upper_float(1/lower))
            else:
                weights.append(upper_float(scale/width))
        # WRMS(w*dz)<=1 => each |w*dz|<=sqrt(N)<=r. Convex secants give
        # WRMS(exact_inverse_delta/D)<=beta*WRMS(w*dz)+eta<=1.
        # A newly decoded numerical Point additionally needs assess_error:
        # a ball radius from that distinct inverse evaluation is not assumed
        # zero or silently covered by the baseline inverse's radius.
        return WeightPromise(weights, debit, allowance.widths_si, eta, beta, r, image.identity)

    def assess_error(self, image, promise, *, coordinates, coordinate_rate=None):
        """Check actual chart error and one combined outward physical error.

        This guard is required when using decoded numerical Points. It uses
        the unchanged original SI widths, includes the new inverse's radius,
        and never adds the baseline debit again to the already combined
        physical difference. Solver weights alone certify the exact inverse,
        not an unexamined finite-precision implementation of that inverse.
        """
        if promise.image_identity != image.identity:
            raise ContractError("representation_weight_image_mismatch")
        z = frozen_array(coordinates)
        decoded = self.inverse(image, coordinates=z, coordinate_rate=coordinate_rate)
        original = state_values(image.point)
        raw = tuple(rational(x)-rational(y) for x,y in zip(z,image.coordinates,strict=True))
        errors = tuple(v.center-y for v,y in zip(decoded.state_si,original,strict=True))
        upper = tuple(abs(e)+v.radius for e,v in zip(errors,decoded.state_si,strict=True))
        raw_sq = sum((e*rational(w))**2 for e,w in zip(raw,promise.weights,strict=True))/self.layout.size
        physical_sq = sum((e/d)**2 for e,d in zip(upper,promise.widths_si,strict=True))/self.layout.size
        return {"accepted": raw_sq <= 1 and physical_sq <= 1,
                "solver_wrms_squared": raw_sq, "original_physical_wrms_upper_squared": physical_sq,
                "physical_error_center_si": errors, "physical_error_upper_si": upper,
                "inverse_arithmetic_bound_si": tuple(v.radius for v in decoded.state_si),
                "combined_physical_error_used_once": True, "original_widths_si": promise.widths_si}
