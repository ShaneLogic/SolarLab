"""Finite original qualification comparisons; no solver, model or native imports.

Values are exact represented words or outward enclosures.  This module measures
differences between two finite discretizations.  It does not certify a solution
of the DAE, a continuum limit, an oracle, or a scientific qualification gate.
The IO/admission adapter lives in the separately reviewed source packet.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import math

from scripts.benchmarks.hysteresis_current_bounds import (
    _Interval, _expm1_bounds, bernoulli_bounds,
)
from scripts.benchmarks.native_readback import _frame_word_values, _hex, _word_pair


class ComparisonError(ValueError):
    """A missing contract/evidence or an unsupported finite calculation."""


def require(condition, reason):
    if not condition:
        raise ComparisonError(reason)


def exact(value):
    require(type(value) in (int, float, Fraction, Decimal), "exact_scalar_type")
    if type(value) is float:
        require(math.isfinite(value), "nonfinite_scalar")
    result = Fraction(value)
    require(max(result.numerator.bit_length(), result.denominator.bit_length()) <= 131072,
            "rational_bit_limit")
    return result


@dataclass(frozen=True)
class Span:
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lo", exact(self.lo))
        object.__setattr__(self, "hi", exact(self.hi))
        require(self.lo <= self.hi, "reversed_enclosure")

    @classmethod
    def of(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, _Interval):
            return cls(Fraction(value.lo), Fraction(value.hi))
        return cls(exact(value), exact(value))

    def __add__(self, other):
        b = Span.of(other)
        return Span(self.lo+b.lo, self.hi+b.hi)

    __radd__ = __add__

    def __neg__(self):
        return Span(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-Span.of(other))

    def __rsub__(self, other):
        return Span.of(other) + (-self)

    def __mul__(self, other):
        b = Span.of(other)
        products = (self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi)
        return Span(min(products), max(products))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = Span.of(other)
        require(not b.lo <= 0 <= b.hi, "undefined_denominator")
        return self*Span(1/b.hi, 1/b.lo)

    @property
    def absolute_upper(self):
        return max(abs(self.lo), abs(self.hi))


def _log_reduced(value):
    """ln(value), 1<=value<=2: positive atanh series with geometric tail.

    Ninety-six fixed terms, outward decimal96 operations, no tolerance loop.
    The tail is <=2*t**193/(193*(1-t*t)), where 0<=t<=1/3.
    """
    t = _Interval.of((value-1)/(value+1))
    term, total = t, t
    square = t*t
    for k in range(1, 96):
        term = term*square
        total = total + term/(2*k+1)
    tail = 2*term*square/(193*(1-square))
    return Span.of(_Interval((2*total).lo, (2*total+tail).hi))


@lru_cache(maxsize=1)
def _log_two():
    return _log_reduced(Fraction(2))


def log_exact(value):
    value = exact(value)
    require(value > 0, "log_requires_positive_density")
    if value == 1:
        return Span.of(0)
    power = value.numerator.bit_length()-value.denominator.bit_length()
    require(abs(power) <= 16384, "log_range_limit")
    scale = Fraction(2)**power
    reduced = value/scale
    if reduced < 1:
        power -= 1
        reduced *= 2
    require(1 <= reduced < 2, "log_range_reduction")
    return _log_reduced(reduced)+_log_two()*power


def logarithm(value):
    value = Span.of(value)
    require(value.lo > 0, "log_requires_positive_density")
    return Span(log_exact(value.lo).lo, log_exact(value.hi).hi)


def bernoulli(value):
    value = Span.of(value)
    # B is positive and decreasing on the real line.  Each endpoint enclosure
    # comes from the existing bounded positive-series implementation.
    return Span(Fraction(bernoulli_bounds(value.hi).lo),
                Fraction(bernoulli_bounds(value.lo).hi))


def sg_interpolate(left, right, delta_phi_over_vt, fraction, *, carrier):
    """Constant-field SG edge solution, including its exact flat-field limit.

    n(s)=nL+(nR-nL)*expm1(xi*s)/expm1(xi); holes use -xi.
    Subtraction of densities precedes conversion to an enclosure, preserving
    arbitrarily small represented endpoint differences.  Contacts are exact.
    """
    a, b, xi, s = map(exact, (left, right, delta_phi_over_vt, fraction))
    require(carrier in ("electron", "hole") and a > 0 and b > 0 and 0 <= s <= 1,
            "sg_scope_or_domain")
    if s in (0, 1):
        return Span.of(a if s == 0 else b)
    if a == b or xi == 0:
        return Span.of(a+(b-a)*s)
    if carrier == "hole":
        xi = -xi
    weight = Span.of(_expm1_bounds(xi*s)/_expm1_bounds(xi))
    # The exact SG weight is in [0,1]; intersect with this proved domain.
    weight = Span(max(Fraction(0), weight.lo), min(Fraction(1), weight.hi))
    return a+(b-a)*weight


POSITIONS = tuple(Fraction(i, 18) for i in range(19))
STATE_UNITS = {"phi_V": "V", "n_m3": "m^-3", "p_m3": "m^-3",
               "c_m3": "m^-3", "f": "1"}
LEDGERS = ("raw_polynomial", "same_state_affine_tangent")
CHARGE_ROWS = ("body_conduction", "left_metal_displacement", "right_metal_displacement")


def check_mesh(mesh):
    mesh = tuple(map(exact, mesh))
    require(2 <= len(mesh) <= 65 and mesh[0] == 0 and mesh[-1] > 0 and
            all(a < b for a, b in zip(mesh, mesh[1:])), "mesh_domain")
    return mesh


def reconstruct(mesh, fields, thermal_voltage, *, trap_rule=None, positions=POSITIONS, support=None):
    mesh, vt = check_mesh(mesh), exact(thermal_voltage)
    require(vt > 0 and {"phi_V", "n_m3", "p_m3"} <= fields.keys() <= STATE_UNITS.keys(),
            "state_fields_or_thermal_voltage")
    values = {k: tuple(map(exact, v)) for k, v in fields.items()}
    require(all(len(v) == len(mesh) for v in values.values()), "field_mesh_shape")
    if support is not None:
        require(len(support) == len(mesh) and len(set(support)) == 1, "unsupported_material_or_defect_interface")
    if "f" in fields:
        require(trap_rule == "linear_nodal_occupancy_v1", "trap_reconstruction_unapproved_or_missing")
        require(all(0 <= v <= 1 for v in values["f"]), "invalid_trap_endpoint_no_clipping")
    if "c_m3" in fields:
        require(min(values["c_m3"]) >= 0, "negative_ion_endpoint")
    result = []
    for x in map(exact, positions):
        require(0 <= x <= 1, "position_outside_device")
        physical_x = x*mesh[-1]
        i = min(bisect_right(mesh, physical_x)-1, len(mesh)-2)
        s = (physical_x-mesh[i])/(mesh[i+1]-mesh[i])
        phi = values["phi_V"]
        row = {}
        for field, data in values.items():
            if field in ("n_m3", "p_m3"):
                row[field] = sg_interpolate(data[i], data[i+1], (phi[i+1]-phi[i])/vt, s,
                                            carrier="electron" if field == "n_m3" else "hole")
            else:
                row[field] = Span.of(data[i]+s*(data[i+1]-data[i]))
        result.append(row)
    return tuple(result)


def full_state(request, history):
    """Retained physical reference PLUS every mapped remainder/rate word."""
    packet = request["numeric_packet"]
    size = len(request["z0"])
    count = request["frame_input_policy"]["mapped_input_words"]
    remainder = _frame_word_values(history["physical_cumulative_words_hex"], size, count=count)
    rate = _frame_word_values(history["physical_rate_words_hex"], size, count=count)
    require(history["physical_model_identity"] == request["voltage_lift_map"]["physical_model"],
            "physical_source_mismatch")
    fields, rates = {}, {}
    for name, (start, stop) in packet["variable_offsets"].items():
        ref = packet["initial_reference"]["payload"]["fields"][name]
        roots = tuple(Fraction(_hex(h))+Fraction(_hex(l))
                      for h, l in zip(ref["high"], ref["low"], strict=True))
        require(len(roots) == stop-start, "physical_reference_shape")
        fields[name] = tuple(a+b for a, b in zip(roots, remainder[start:stop], strict=True))
        rates[name] = rate[start:stop]
    return fields, rates


def polynomial_state(coefficients, tn, h, when, offsets):
    tn, h, when = map(exact, (tn, h, when))
    require(h > 0, "polynomial_zero_step")
    u = (when-tn)/h
    state, rate = [], []
    for row in coefficients:
        require(1 <= len(row) <= 6, "polynomial_order")
        row = tuple(map(exact, row))
        state.append(sum((v*u**k for k, v in enumerate(row)), Fraction(0)))
        rate.append(sum((k*v*u**(k-1) for k, v in enumerate(row) if k), Fraction(0))/h)
    return ({k: tuple(state[a:b]) for k, (a, b) in offsets.items()},
            {k: tuple(rate[a:b]) for k, (a, b) in offsets.items()})


def observation_key(segment, time, event_side):
    require(type(segment) is str and segment and event_side in ("left", "right", "continuous"),
            "clock_or_event_side")
    return segment, exact(time), event_side


def compare_metric(left, right, *, metric, budget, input_error, reference_error, scale=None):
    """Worst enclosed finite difference + separately supplied uncertainty.

    Uncertainties are in the metric's units (log ratio for carriers), already
    combined across the two cases.  A missing contribution is never zero.
    """
    require(input_error is not None, "input_quantization_evidence_missing")
    require(reference_error is not None, "independent_reference_evidence_missing")
    budget, qe, re = map(exact, (budget, input_error, reference_error))
    require(budget >= 0 and qe >= 0 and re >= 0, "negative_budget_or_uncertainty")
    a, b = Span.of(left), Span.of(right)
    if metric == "absolute":
        difference = a-b
    elif metric == "log_ratio":
        difference = logarithm(a/b)
    elif metric == "declared_scale":
        require(scale is not None and exact(scale) > 0, "declared_positive_scale_required")
        difference = (a-b)/exact(scale)
    else:
        raise ComparisonError("unsupported_metric")
    error = difference.absolute_upper
    total = error+qe+re
    return {"difference_upper": error, "input_quantization_upper": qe,
            "independent_reference_upper": re, "total_upper": total, "budget": budget,
            "input_share_passed": 3*qe <= budget, "reference_share_passed": 3*re <= budget,
            "passed": total <= budget and 3*qe <= budget and 3*re <= budget,
            "continuum_or_global_error_certificate": False}


def accumulate_charge(previous, defect_centers, errors, budget):
    """Original three rows, absolute interval/prefix accumulation; no reset."""
    budget = exact(budget)
    previous = tuple((exact(a), exact(b)) for a, b in previous)
    require(len(previous) == len(defect_centers) == len(errors) == 3 and budget > 0,
            "charge_shape_or_budget")
    result = []
    for (pa, pe), defect, error in zip(previous, defect_centers, errors, strict=True):
        defect, error = exact(defect), exact(error)
        require(min(pa, pe, error) >= 0, "negative_charge_uncertainty")
        interval = abs(defect)+error
        prefix, uncertainty = pa+interval, pe+error
        result.append({"interval_bound_C": interval, "prefix_absolute_bound_C": prefix,
                       "interval_uncertainty_C": error, "prefix_uncertainty_C": uncertainty,
                       "passed": interval <= budget and prefix <= budget and
                                 3*error <= budget and 3*uncertainty <= budget})
    return tuple(result)


def physical_observables(packet, fields, rates, input_rate, constants):
    """Independent fixed homogeneous-slab scalar laws, no Point/F/J/Poisson.

    The formulas are those of the pinned independent SlabPathObserver and the
    original physical linear forms.  Exact algebra preserves finite Gauss
    defects and distinct native/polynomial versus same-state tangent rates.
    Only the original two zero-doping homogeneous definitions are supported.
    """
    m = packet["definition"]
    require(m["id"] in ("S0NeutralPublicDeviceV1", "DynamicAcceptorIonPublicDeviceV1"),
            "unsupported_physical_definition")
    x = check_mesh(packet["x_m"])
    count = len(x)
    require(set(fields) == set(rates) == ({"phi_V", "n_m3", "p_m3", "c_m3", "f"}
            if m["dynamic"] else {"phi_V", "n_m3", "p_m3"}), "physical_channel_set")
    f = {k: tuple(map(exact, v)) for k, v in fields.items()}
    r = {k: tuple(map(exact, v)) for k, v in rates.items()}
    require(all(len(v) == count for v in (*f.values(), *r.values())), "physical_channel_shape")
    q, kb = exact(constants["Q"]), exact(constants["K_B"])
    # Preserve the original SlabDefinition.vt binary64 operation sequence.
    vt = exact(float(kb)*m["temperature"]/float(q))
    area, eps, nt = map(exact, (m["area"], m["epsilon"], m["trap_density"]))
    volumes = tuple(map(exact, packet["volumes_m3"]))
    dx = tuple(b-a for a, b in zip(x, x[1:]))
    # dx in the frozen kernel is binary64 subtraction; bind those words too.
    dx = tuple(exact(float(b)-float(a)) for a, b in zip(x, x[1:]))
    require(len(volumes) == count and min(*volumes, *dx, vt, q, area) > 0,
            "positive_physical_geometry")
    n, p, phi = (f[k] for k in ("n_m3", "p_m3", "phi_V"))
    require(min(*n, *p) > 0, "nonpositive_active_carrier")
    zero = (Fraction(0),)*count
    c, trap = f.get("c_m3", zero), f.get("f", zero)
    cr, fr = r.get("c_m3", zero), r.get("f", zero)
    if m["dynamic"]:
        require(all(0 <= v < exact(m["ion_capacity"]) for v in c) and
                all(0 <= v <= 1 for v in trap), "ion_or_trap_domain")
    rho = tuple(q*(p[i]-n[i]+(c[i]-exact(m["ion_initial"])-nt*trap[i] if m["dynamic"] else 0))
                for i in range(count))
    rho_rate = tuple(q*(r["p_m3"][i]-r["n_m3"][i]+cr[i]-nt*fr[i]) for i in range(count))
    displacement = tuple(-eps*(b-a)/d for a, b, d in zip(phi, phi[1:], dx))
    displacement_rate = tuple(-eps*(b-a)/d for a, b, d in zip(r["phi_V"], r["phi_V"][1:], dx))
    metal = (area*displacement[0]-volumes[0]*rho[0],
             -area*displacement[-1]-volumes[-1]*rho[-1])
    metal_rate = (area*displacement_rate[0]-volumes[0]*rho_rate[0],
                  -area*displacement_rate[-1]-volumes[-1]*rho_rate[-1])
    body = sum((v*d for v, d in zip(volumes, rho)), Fraction(0))
    carrier, ion = [], []
    for i, d in enumerate(dx):
        xi = (phi[i+1]-phi[i])/vt
        b = bernoulli(xi)
        jn = q*exact(m["mu_n"])*vt/d*(b*(n[i+1]-n[i])-xi*n[i])
        jp = q*exact(m["mu_p"])*vt/d*(-b*(p[i+1]-p[i])-xi*p[i+1])
        carrier.append(jn+jp)
        if m["dynamic"]:
            cap = exact(m["ion_capacity"])
            drive = xi+logarithm(Span.of(cap-c[i])/Span.of(cap-c[i+1]))
            ion.append(exact(m["diffusion_ion"])/d*(-bernoulli(drive)*(c[i+1]-c[i])-drive*c[i+1]))
        else:
            ion.append(Span.of(0))
    capture = tuple(-q*volumes[i]*nt*(exact(m["capture_n"])*(n[i]*(1-trap[i])-exact(m["n1"])*trap[i])
                      -exact(m["capture_p"])*(p[i]*trap[i]-exact(m["p1"])*(1-trap[i])))
                    if m["dynamic"] else Fraction(0) for i in range(count))
    reservoir = (area*carrier[0]+capture[0], -area*carrier[-1]+capture[-1])
    endpoint_rate = tuple(q*volumes[i]*(r["p_m3"][i]-r["n_m3"][i]) for i in (0, count-1))
    raw = tuple(a+b+c for a, b, c in zip(reservoir, endpoint_rate, metal_rate))
    total_faces = tuple(a+q*b for a, b in zip(carrier, ion))
    charge_rates = [-q*area*ion[0]]
    charge_rates += [area*(a-b) for a, b in zip(total_faces, total_faces[1:])]
    charge_rates += [q*area*ion[-1]]
    length = sum(dx, Fraction(0))
    weights = (tuple(sum(dx[i:], Fraction(0))/length for i in range(count)),
               tuple(sum(dx[:i], Fraction(0))/length for i in range(count)))
    tangent_metal = tuple(sign*area*eps/length*exact(input_rate[0])
                         -sum((w[i]*capture[i] for i in (0, count-1)), Fraction(0))
                         -sum((wi*qi for wi, qi in zip(w, charge_rates)), Span.of(0))
                         for sign, w in zip((1, -1), weights))
    tangent = tuple(a+b for a, b in zip(reservoir, tangent_metal))
    nodal_tangent = list(charge_rates)
    nodal_tangent[0] += capture[0]
    nodal_tangent[-1] += capture[-1]
    tangent_faces = tuple(total_faces[i]+(tangent_metal[0]+sum(nodal_tangent[:i+1], Span.of(0)))/area
                          for i in range(count-1))
    return {"raw_total_inward_A": raw, "same_state_tangent_total_inward_A": tangent,
            "raw_conduction_inward_A": tuple(a+b for a, b in zip(reservoir, endpoint_rate)),
            "same_state_tangent_conduction_inward_A": reservoir,
            "raw_interior_total_A_m2": tuple(a+b for a, b in zip(total_faces, displacement_rate)),
            "same_state_tangent_interior_total_A_m2": tangent_faces,
            "metal_charge_C": metal, "body_charge_C": body,
            "body_charge_rate_A": sum((v*d for v, d in zip(volumes, rho_rate)), Fraction(0)),
            "gauss_defect_C": body+sum(metal),
            "raw_metal_rate_A": metal_rate, "same_state_tangent_metal_rate_A": tangent_metal}


def retained_currents(sample):
    return {"raw_total_inward_A": tuple(Span.of(x) for x in _word_pair(sample["raw"]["total_inward"], 2)),
            "same_state_tangent_total_inward_A": tuple(Span.of(x) for x in _word_pair(sample["physical_tangent"]["total_inward"], 2))}


def validate_pair_clock(left, right):
    require(left["key"] == right["key"], "clock_or_contact_event_mismatch")
    require(left["fields"].keys() == right["fields"].keys(), "state_channel_mismatch")


def score_clock(left, right, *, thresholds, uncertainty, c0, thermal_voltage, trap_rule=None):
    validate_pair_clock(left, right)
    require(uncertainty is not None, "independent_reference_evidence_missing")
    a = reconstruct(left["mesh"], left["fields"], thermal_voltage, trap_rule=trap_rule)
    b = reconstruct(right["mesh"], right["fields"], thermal_voltage, trap_rule=trap_rule)
    policies = {"phi_V": ("absolute", thresholds["potential_max_abs_V"], None),
                "n_m3": ("log_ratio", thresholds["carrier_max_abs_log_ratio"], None),
                "p_m3": ("log_ratio", thresholds["carrier_max_abs_log_ratio"], None),
                "c_m3": ("declared_scale", thresholds["ion_max_abs_change_over_declared_c0"], c0),
                "f": ("absolute", thresholds["trap_max_abs_occupancy_change"], None)}
    rows = []
    for position, ra, rb in zip(POSITIONS, a, b):
        for channel in ra:
            require(channel in uncertainty, "uncertainty_channel_missing:"+channel)
            u = uncertainty[channel]
            metric, budget, scale = policies[channel]
            metric_unit = "1" if metric in ("log_ratio", "declared_scale") else STATE_UNITS[channel]
            rows.append({"channel": channel, "unit": metric_unit, "field_unit": STATE_UNITS[channel],
                         "position_x_over_L": position,
                         **compare_metric(ra[channel], rb[channel], metric=metric, budget=budget, scale=scale,
                                          input_error=u.get("input"), reference_error=u.get("reference"))})
    for channel in ("raw_total_inward_A", "same_state_tangent_total_inward_A"):
        require(channel in uncertainty, "uncertainty_channel_missing:"+channel)
        for i, port in enumerate(("left", "right")):
            u = uncertainty[channel]
            rows.append({"channel": channel, "port": port, "unit": "A",
                         **compare_metric(left["observables"][channel][i], right["observables"][channel][i],
                            metric="absolute", budget=thresholds["current_A"],
                            input_error=u.get("input"), reference_error=u.get("reference"))})
    return {"key": left["key"], "rows": rows, "passed": all(r["passed"] for r in rows),
            "scientific_qualification": False}


@dataclass
class StepSummary:
    """Constant-storage observed step/order summary, not a trajectory certificate."""
    count: int = 0
    minimum: Fraction | None = None
    maximum: Fraction | None = None
    orders: dict = field(default_factory=dict)

    def add(self, value, order=None):
        value = exact(value)
        require(value > 0, "actual_nonpositive_step")
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if order is not None:
            require(type(order) is int and 1 <= order <= 5, "actual_order")
            self.orders[order] = self.orders.get(order, 0)+1

    def record(self):
        require(self.count > 0, "actual_grid_missing")
        return {"count": self.count, "minimum_h": self.minimum, "maximum_h": self.maximum,
                "orders": {str(k): v for k, v in sorted(self.orders.items())}}


def physical_control_tightening(coarse, fine, scales):
    """Pointwise strict r*abs(q)+atol contraction in the same physical map.

    The IO boundary separately proves original policy ancestry, identical maps
    and actual observed controls.  No nominal factor or passed flag is used.
    """
    require(set(coarse) == set(fine) and
            {k: v for k, v in coarse.items() if k not in ("rtol", "atol")} ==
            {k: v for k, v in fine.items() if k not in ("rtol", "atol")}, "tolerance_unrelated_controls")
    ca, fa = tuple(map(exact, coarse["atol"])), tuple(map(exact, fine["atol"]))
    scales = tuple(map(exact, scales))
    cr, fr = exact(coarse["rtol"]), exact(fine["rtol"])
    require(len(ca) == len(fa) == len(scales) > 0 and min(scales) > 0, "tolerance_physical_shape")
    require(0 < fr < cr and all(0 < s*b < s*a for s, a, b in zip(scales, ca, fa)),
            "strict_pairwise_physical_controls_required")
    return {"rtol_ratio": fr/cr, "largest_physical_atol_ratio": max(b/a for a, b in zip(ca, fa)),
            "components": len(ca), "denominator_comparison": "all parent q under identical physical map",
            "global_error_certificate": False}


def actual_refinement(axis, coarse_mesh, fine_mesh, coarse_steps, fine_steps, *, tolerance_controls=None):
    coarse_mesh, fine_mesh = check_mesh(coarse_mesh), check_mesh(fine_mesh)
    def summary(values):
        if isinstance(values, StepSummary):
            return values.record()
        result = StepSummary()
        for value in values:
            result.add(value)
        return result.record()
    coarse_steps, fine_steps = summary(coarse_steps), summary(fine_steps)
    controls = None
    if axis == "space":
        passed = (len(fine_mesh)-1 == 2*(len(coarse_mesh)-1) and
                  max(b-a for a, b in zip(fine_mesh, fine_mesh[1:])) <
                  max(b-a for a, b in zip(coarse_mesh, coarse_mesh[1:])))
    elif axis == "tolerance":
        require(coarse_mesh == fine_mesh and isinstance(tolerance_controls, dict),
                "bound_tolerance_controls_missing")
        controls = physical_control_tightening(**tolerance_controls)
        passed = True
    else:
        require(axis == "time", "unknown_original_axis")
        passed = fine_steps["maximum_h"] < coarse_steps["maximum_h"]
    return {"axis": axis, "passed": passed, "coarse_max_h": coarse_steps["maximum_h"],
            "fine_max_h": fine_steps["maximum_h"], "coarse_steps": coarse_steps, "fine_steps": fine_steps,
            "physical_control_tightening": controls, "cap_change_is_not_refinement": True}


def joint_confirmation(disposition):
    if disposition is None or disposition.get("interaction") == "unknown":
        return {"passed": False, "reason": "axis_interaction_disposition_or_joint_confirmation_missing"}
    require(disposition.get("independently_accepted") is True, "unreviewed_axis_interaction")
    if disposition.get("interaction") is True:
        return {"passed": disposition.get("joint_finest_accepted") is True,
                "reason": "joint_finest_confirmation_required"}
    require(disposition.get("interaction") is False, "axis_interaction_type")
    return {"passed": True, "reason": "independent_no_interaction_disposition"}
