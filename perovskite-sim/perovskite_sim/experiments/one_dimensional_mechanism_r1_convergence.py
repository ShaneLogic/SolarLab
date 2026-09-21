"""R1 sections 6/10 grids and comparisons of already aligned responses.

This module neither runs a solver nor interpolates responses. A comparison is
not a physical-conservation certificate or an R1-2 acceptance certificate.
Inputs are response changes; DC potential has its own explicitly named rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping

import numpy as np


BASE_INTERVALS = (16, 32, 64)
ALLOWED_INTERVALS = (*BASE_INTERVALS, 128, 256)
BASE_TIME_SUBSTEPS = ((1, 2, 4), (2, 4, 8), (4, 8, 16))
ALLOWED_TIME_SUBSTEPS = (*BASE_TIME_SUBSTEPS, (8, 16, 32), (16, 32, 64))
BASE_NONLINEAR_FACTORS = (1.0, 0.1, 0.01)
ALLOWED_NONLINEAR_FACTORS = (*BASE_NONLINEAR_FACTORS, 0.001)
# Section 6: the three initial levels and the four permitted extensions.
AMPLITUDES_V = (.01, .005, .0025, .00125, .000625, .0003125, .00015625)
_AMPLITUDE_RESPONSE_RELATIVE_TOLERANCE = .01
_AMPLITUDE_MAXIMUM_ERROR_SIGNAL_FRACTION = .01
POINTS_PER_DECADE = 12


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _real_scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def validate_time_substeps(values):
    """Return one declared independent time-axis setting without coercing types."""
    substeps = tuple(_integer(value, "time substeps") for value in values)
    if substeps not in ALLOWED_TIME_SUBSTEPS:
        raise ValueError("R1 time setting must be a declared three-level nested tuple")
    return substeps


@dataclass(frozen=True, slots=True)
class R1ConvergenceCase:
    """One numerical setting; the tuple is one complete nested time setting."""

    intervals: int
    time_substeps: tuple[int, int, int]
    nonlinear_factor: float
    control: str = "D"
    amplitude_V: float = 0.005

    def __post_init__(self):
        intervals = _integer(self.intervals, "intervals")
        substeps = validate_time_substeps(self.time_substeps)
        factor = _real_scalar(self.nonlinear_factor, "nonlinear factor")
        amplitude = _real_scalar(self.amplitude_V, "amplitude_V")
        if intervals not in ALLOWED_INTERVALS:
            raise ValueError("R1 spatial settings are 16, 32, 64, 128 and 256")
        if factor not in ALLOWED_NONLINEAR_FACTORS:
            raise ValueError("R1 nonlinear factor must be 1, .1, .01 or .001")
        if self.control not in ("A", "B", "C", "D"):
            raise ValueError("R1 control must be A, B, C or D")
        if amplitude not in AMPLITUDES_V:
            raise ValueError("R1 amplitude must be a declared positive halving level")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "time_substeps", substeps)
        object.__setattr__(self, "nonlinear_factor", factor)
        object.__setattr__(self, "amplitude_V", amplitude)


def convergence_cases(*, intervals=BASE_INTERVALS, time_substeps=BASE_TIME_SUBSTEPS,
                      nonlinear_factors=BASE_NONLINEAR_FACTORS, control="D", amplitude_V=.005):
    """Construct a declared Cartesian product, including explicitly chosen extensions."""
    axes = (tuple(intervals), tuple(tuple(x) for x in time_substeps), tuple(nonlinear_factors))
    if any(not axis or len(set(axis)) != len(axis) for axis in axes):
        raise ValueError("convergence axes must be nonempty and contain no duplicates")
    return tuple(R1ConvergenceCase(n, steps, factor, control, amplitude_V)
                 for n, steps, factor in product(*axes))


def base_convergence_cases(control="D"):
    """Return Base-D's 27 cases or one A-C control's seven-axis-cross cases."""
    cases = convergence_cases(control=control)
    if control == "D":
        return cases
    return tuple(case for case in cases if sum((
        case.intervals != BASE_INTERVALS[-1],
        case.time_substeps != BASE_TIME_SUBSTEPS[-1],
        case.nonlinear_factor != BASE_NONLINEAR_FACTORS[-1],
    )) <= 1)


def validate_single_axis_comparison(left, right, *, axis):
    """Require an adjacent refinement of exactly one declared numerical axis.

    Metadata must come from verified trajectories. This structural check is
    separate from the pure response comparator, so time/nonlinear comparisons
    correctly retain the same spatial grid. Equal settings are never a
    convergence experiment.
    """
    from dataclasses import asdict
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy

    axes = {"spatial": "intervals", "time": "time_substeps", "nonlinear": "nonlinear_factor",
            "intervals": "intervals", "time_substeps": "time_substeps", "nonlinear_factor": "nonlinear_factor"}
    if axis not in axes:
        raise ValueError("comparison requires one declared numerical axis")
    axis = axes[axis]
    for key in ("stack_sha256", "reference_sha256", "source_sha256", "control", "amplitude_V"):
        if key not in left or key not in right or left[key] is None or left[key] != right[key]:
            raise ValueError("convergence physical identity differs or is absent: " + key)
    if left.get("times_s") != right.get("times_s") or not left.get("times_s"):
        raise ValueError("convergence requires identical explicit output times")

    def setting(metadata):
        policy = dict(metadata["policy"])
        steps = validate_time_substeps(policy["refinement_substeps"])
        policy["refinement_substeps"] = steps
        factors = [factor for factor in ALLOWED_NONLINEAR_FACTORS
                   if policy == asdict(r1_policy(factor, time_substeps=steps))]
        if len(factors) != 1:
            raise ValueError("convergence policy is not a declared frozen numerical setting")
        return R1ConvergenceCase(metadata["intervals"], steps, factors[0], metadata["control"], metadata["amplitude_V"])

    a, b = setting(left), setting(right)
    changes = [name for name in ("intervals", "time_substeps", "nonlinear_factor") if getattr(a, name) != getattr(b, name)]
    if changes != [axis]:
        raise ValueError("convergence requires exactly the declared axis to differ; self-comparison is not convergence")
    ordered = {"intervals": ALLOWED_INTERVALS, "time_substeps": ALLOWED_TIME_SUBSTEPS,
               "nonlinear_factor": ALLOWED_NONLINEAR_FACTORS}[axis]
    if ordered.index(getattr(b, axis)) != ordered.index(getattr(a, axis)) + 1:
        raise ValueError("convergence requires adjacent coarse-to-fine numerical levels")
    return {"axis": axis, "single_axis_verified": True, "left": asdict(a), "right": asdict(b)}


def observation_times(*, first_time_s=1e-9, last_time_s=1e2):
    """Return 0+ and the section-6 logarithmic grid, with exact shared ticks.

    Extensions are in whole decades: the first positive time can be 1e-9
    through 1e-12 s, and the endpoint can be 1e2 through 1e5 s. There are
    twelve logarithmic intervals per decade and both endpoints are retained.
    """
    first = _real_scalar(first_time_s, "first_time_s")
    last = _real_scalar(last_time_s, "last_time_s")
    starts = {10.0**power: power for power in range(-12, -8)}
    ends = {10.0**power: power for power in range(2, 6)}
    if first not in starts or last not in ends:
        raise ValueError("time endpoints must use the declared section-6 decade extensions")
    ticks = np.arange(starts[first]*POINTS_PER_DECADE,
                      ends[last]*POINTS_PER_DECADE + 1, dtype=np.int64)
    values = np.r_[0.0, np.power(10.0, ticks / POINTS_PER_DECADE)]
    values[1], values[-1] = first, last
    return values


def _numeric_array(value, name, *, complex_allowed=False):
    array = np.asarray(value)
    allowed = "iufc" if complex_allowed else "iuf"
    if array.dtype.kind not in allowed:
        raise ValueError(f"{name} must contain numeric {'real or complex' if complex_allowed else 'real'} values")
    return np.array(array, dtype=complex if array.dtype.kind == "c" else float, copy=True)


@dataclass(frozen=True, slots=True)
class R1Response:
    """Sample values and their exact coordinate identity, without reconstruction.

    Coordinate insertion order defines the array axes. Optional named
    components occupy a final axis. Nonfinite sample values are retained so
    the comparator can identify every failed point; coordinates must be finite.
    Ion inputs use changes in P/P0 and carrier inputs use changes in log density.
    """

    values: np.ndarray
    coordinates: Mapping[str, np.ndarray]
    components: tuple[str, ...] = ()

    def __post_init__(self):
        values = _numeric_array(self.values, "response", complex_allowed=True)
        if not isinstance(self.coordinates, Mapping) or not self.coordinates:
            raise ValueError("response coordinates must be a nonempty ordered mapping")
        coordinates = {}
        for name, raw in self.coordinates.items():
            if not isinstance(name, str) or not name:
                raise ValueError("coordinate names must be nonempty strings")
            axis = _numeric_array(raw, f"coordinate {name}")
            if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
                raise ValueError(f"coordinate {name} must be a finite nonempty vector")
            axis.setflags(write=False)
            coordinates[name] = axis
        components = tuple(self.components)
        if any(not isinstance(name, str) or not name for name in components) or len(set(components)) != len(components):
            raise ValueError("component names must be nonempty and unique")
        shape = tuple(len(axis) for axis in coordinates.values())
        if components:
            shape += (len(components),)
        if values.shape != shape:
            raise ValueError(f"response shape {values.shape} differs from coordinate/component shape {shape}")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))
        object.__setattr__(self, "components", components)


def _aligned(left, right, *, real=False):
    if not isinstance(left, R1Response) or not isinstance(right, R1Response):
        raise TypeError("comparisons require R1Response inputs with explicit coordinates")
    if left.values.shape != right.values.shape:
        raise ValueError("response shapes differ; broadcasting is not permitted")
    if tuple(left.coordinates) != tuple(right.coordinates) or left.components != right.components:
        raise ValueError("coordinate axes or component identities differ")
    if any(not np.array_equal(left.coordinates[key], right.coordinates[key]) for key in left.coordinates):
        raise ValueError("coordinate values differ; implicit interpolation is not permitted")
    if real and (np.iscomplexobj(left.values) or np.iscomplexobj(right.values)):
        raise ValueError("this response quantity requires real values")


def _evidence_number(value):
    value = float(value)
    if np.isfinite(value):
        return value
    return {"nonfinite": "nan" if np.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"}


def _evidence_array(array):
    return [_evidence_array(value) for value in array] if np.ndim(array) else _evidence_number(array)


def _location(response, index, part):
    point = {"index": [int(i) for i in index],
             "coordinates": {name: float(axis[index[i]]) for i, (name, axis) in enumerate(response.coordinates.items())}}
    if response.components:
        point["component"] = response.components[index[-1]]
    if part is not None:
        point["part"] = part
    return point


def _pointwise(left, right, response, *, absolute, relative, part=None, extra_reasons=None):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = np.maximum(np.abs(left), np.abs(right))
        difference = np.abs(left-right)
        budget = absolute + relative*scale
        ratio = np.divide(difference, budget, out=np.full_like(difference, np.inf), where=budget > 0)
        ratio = np.where((difference == 0) & (budget == 0), 0.0, ratio)
    finite = np.isfinite(left) & np.isfinite(right)
    arithmetic_finite = np.isfinite(difference) & np.isfinite(budget) & np.isfinite(ratio)
    passed = finite & arithmetic_finite & (difference <= budget)
    failures = []
    for index in np.ndindex(left.shape):
        reasons = []
        if not finite[index]:
            reasons.append("nonfinite_response")
        elif not arithmetic_finite[index]:
            reasons.append("nonfinite_comparison_arithmetic")
        elif not passed[index]:
            reasons.append("response_difference_exceeds_budget")
        if extra_reasons is not None:
            reasons.extend(extra_reasons(index))
        if reasons:
            failures.append({**_location(response, index, part), "reasons": reasons,
                             "left": _evidence_number(left[index]), "right": _evidence_number(right[index]),
                             "absolute_difference": _evidence_number(difference[index]),
                             "response_scale": _evidence_number(scale[index]),
                             "allowed_difference": _evidence_number(budget[index])})
    valid = finite & arithmetic_finite
    return {"failures": failures, "scalar_comparison_count": int(left.size),
            "maximum_absolute_difference": float(np.max(difference[valid])) if np.any(valid) else None,
            "maximum_budget_ratio": float(np.max(ratio[valid])) if np.any(valid) else None}


# Every absolute term has the stated quantity's units. These are response
# comparison budgets, not the transient solver's 2%/5% refinement checks.
_RESPONSE_RULES = MappingProxyType({
    "dc_potential": (1e-3, 0.0, "V"),
    "response_potential": (5e-5, .01, "V"),
    "carrier_log_density": (.01, 0.0, "1"),
    "ion_density_over_p0": (.01, 0.0, "1"),
    "trap_occupancy_change": (1e-7, .01, "1"),
    "ion_centroid_change": (5e-11, .01, "m"),
    "regular_current_response": (1e-7, .005, "A/m2"),
    "integrated_charge_response": (1e-10, .005, "C/m2"),
    "admittance": (1e-8, .01, "S/m2"),
})


def compare_responses(quantity, left, right):
    """Apply the fixed section-10.2 rule at every aligned point/component.

    Complex admittance is checked separately in its real and imaginary parts.
    Nonfinite samples fail explicitly. Shape or coordinate mismatches raise
    before comparison, because no common physical sample can be identified.
    """
    if quantity not in _RESPONSE_RULES:
        raise ValueError(f"unknown R1 response quantity: {quantity}")
    _aligned(left, right, real=quantity != "admittance")
    absolute, relative, units = _RESPONSE_RULES[quantity]
    parts = (("real", left.values.real, right.values.real),
             ("imaginary", left.values.imag, right.values.imag)) if quantity == "admittance" else ((None, left.values, right.values),)
    reports = [_pointwise(a, b, left, absolute=absolute, relative=relative, part=part)
               for part, a, b in parts]
    failures = [failure for report in reports for failure in report["failures"]]
    return {
        "scope": "aligned_response_comparison_only", "quantity": quantity, "units": units,
        "passed": not failures, "absolute_tolerance": absolute, "relative_tolerance": relative,
        "scale_definition": "pointwise maximum of the two absolute response components",
        "scalar_comparison_count": sum(report["scalar_comparison_count"] for report in reports),
        "failure_count": len(failures), "failures": failures,
        "maximum_budget_ratio": max((report["maximum_budget_ratio"] for report in reports
                                      if report["maximum_budget_ratio"] is not None), default=None),
    }


def _step_current_charge_inputs(record, expected_times_s):
    """One shared full-axis and unique-finest-row contract for both backends."""
    times = _numeric_array(record["times_s"], "step times")
    if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("step times must include zero and increasing positive samples")
    if expected_times_s is not None and not np.array_equal(times, expected_times_s):
        raise ValueError("regular response does not cover the complete requested time axis")
    current = _numeric_array([item["report_contact_current_A_m2"]
                              for item in record["regular_currents"]], "regular contact current")
    if current.shape != (len(times), 2):
        raise ValueError("regular current and physical contacts must align with exact output times")
    finest = max(validate_time_substeps(record["policy"]["refinement_substeps"]))
    rows = [row for row in record["accepted_steps"] if row["substeps"] == finest]
    impulse = _real_scalar(record["initial_event"]["impulse_charge_C_m2"], "impulse charge")
    integrated = []
    for time in times:
        matching = [row for row in rows if row["time_s"] == time]
        if len(matching) != 1:
            raise ValueError("integrated charge requires exactly one finest row at each output time")
        integrated.append(matching[0]["regular_integrated_charge_C_m2"])
    return times, current, impulse, integrated


def step_current_charge_responses(record, baseline_current_A_m2, *, expected_times_s=None):
    """Construct all regular-current and impulse-inclusive charge samples.

    This adapter does not establish provenance. Its caller must verify the
    step and same-control zero-minus baseline. Every declared time and both
    contacts are retained; no endpoint selection or implicit interpolation
    participates in the construction. Pair records require the explicit pair
    adapter and a complete verified pair DC record.
    """
    if (record.get("schema") == "R1ControlledStepV2"
            or record.get("representation") == "float64-pair-v1"):
        raise ValueError("pair step current/charge requires the explicit pair DC baseline adapter")
    baseline = _numeric_array(baseline_current_A_m2, "zero-minus contact baseline")
    if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
        raise ValueError("regular current and physical contacts must align with exact output times")
    times, current, impulse, integrated = _step_current_charge_inputs(record, expected_times_s)
    charge = [impulse + value - baseline[0]*time for time, value in zip(times, integrated)]
    return (R1Response(current-baseline, {"time_s": times}, ("left_contact", "right_contact")),
            R1Response(np.asarray(charge), {"time_s": times}))


def _pair_contact_baseline(record, baseline, *, step_prepared=None, dc_prepared=None):
    from .one_dimensional_mechanism_r1_pair_response import validate_schema
    from .one_dimensional_mechanism_r1_pair_codec import validate_snapshot, decode_prepared, REPRESENTATION, STEP_SCHEMA
    from .one_dimensional_mechanism_r1_state import canonical, digest, physical_preparation_identity
    from .one_dimensional_mechanism_r1_precision import pair_words
    from ..physics.compensated import DD, sum as dd_sum
    if record.get("schema") != STEP_SCHEMA or record.get("representation") != REPRESENTATION:
        raise ValueError("pair current/charge requires a matching pair step representation")
    validate_schema(baseline, "R1ControlledDCResponseV2")
    if baseline.get("voltage_V") != 0. or baseline.get("certified") is not True:
        raise ValueError("pair response baseline must be certified same-control zero-bias DC")
    from .one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
    from dataclasses import asdict
    if (record.get("control_label") != baseline.get("control")
            or record.get("controls") != asdict(R1DynamicsControls.from_label(baseline["control"]))):
        raise ValueError("pair step/DC baseline control identity mismatch")
    for key in ("intervals", "reference_sha256", "source"):
        if key not in record or key not in baseline or canonical(record[key]) != canonical(baseline[key]):
            raise ValueError("pair step/DC baseline identity mismatch: " + key)
    if record.get("prepared_sha256") != baseline.get("prepared_sha256"):
        if step_prepared is None or dc_prepared is None:
            raise ValueError("pair step/DC baseline preparation hashes differ without both original preparations")
        prepared_a = decode_prepared(step_prepared, representation=REPRESENTATION).to_dict()
        prepared_b = decode_prepared(dc_prepared, representation=REPRESENTATION).to_dict()
        if (prepared_a["sha256"] != record.get("prepared_sha256")
                or prepared_b["sha256"] != baseline.get("prepared_sha256")
                or canonical(physical_preparation_identity(prepared_a)) != canonical(physical_preparation_identity(prepared_b))
                or canonical(prepared_a["source"]) != canonical(record["source"])
                or canonical(prepared_b["source"]) != canonical(baseline["source"])):
            raise ValueError("pair step/DC physical preparation or source identity mismatch")
    validate_snapshot(baseline["state"])
    if baseline.get("consumed_state_sha256") != digest(baseline["state"]):
        raise ValueError("pair DC baseline consumed state digest mismatch")
    def words(name):
        value = baseline[name]
        if not isinstance(value, dict) or set(value) != {"hi", "lo"}:
            raise ValueError("pair DC baseline requires both high and low words: " + name)
        high, low = _numeric_array(value["hi"], name), _numeric_array(value["lo"], name)
        if high.shape != low.shape or not np.all(np.isfinite(high)) or not np.all(np.isfinite(low)):
            raise ValueError("pair DC baseline words have invalid shape or finite values")
        result = DD(high, low)
        if not np.array_equal(result.hi, high) or not np.array_equal(result.lo, low):
            raise ValueError("pair DC baseline words are not normalized")
        return result
    current, observations = words("precision_current_A_m2"), words("precision_observations")
    if (current.ndim != 1 or current.size < 2 or observations.shape != (4, current.size)
            or current.size != len(baseline["state"]["n_m3"])+len(baseline["state"]["occupancy"])+1
            or len(baseline.get("physical_face_labels", ())) != current.size
            or baseline.get("physical_face_labels", [None])[0] != "left_contact"
            or baseline.get("physical_face_labels", [None])[-1] != "right_contact"):
        raise ValueError("pair DC baseline physical contact shape differs")
    if (canonical(pair_words(dd_sum(observations[:3], axis=0))) != canonical(pair_words(current))
            or not np.array_equal(current.to_float(), baseline["current_A_m2"])
            or float(current[0].to_float()) != baseline["terminal_current_A_m2"]):
        raise ValueError("pair DC baseline current differs from its saved observations")
    return current[[0, -1]]


def pair_step_current_charge_responses(record, dc_record, *, expected_times_s=None,
                                      step_prepared=None, dc_prepared=None):
    """Subtract a verified pair DC baseline before rounding reported responses.

    The transient regular currents, impulse and integrated charge remain their
    original binary64 observations. This consumer retains the DC low words in
    baseline subtraction and baseline-times-time; it does not confer new
    precision or source provenance on those original transient observations.
    Different timestamped preparation hashes require both original, validated
    preparation artifacts with identical physical content and execution source.
    """
    from ..physics.compensated import DD
    baseline = _pair_contact_baseline(record, dc_record, step_prepared=step_prepared, dc_prepared=dc_prepared)
    times, current, impulse, integrated = _step_current_charge_inputs(record, expected_times_s)
    charge = (DD(impulse) + DD(integrated) - baseline[0]*DD(times)).to_float()
    return (R1Response((DD(current)-baseline).to_float(), {"time_s": times}, ("left_contact", "right_contact")),
            R1Response(charge, {"time_s": times}))


def compare_step_current_charge(left, right, left_baseline, right_baseline, *, expected_times_s=None):
    """Apply the unchanged current/charge budgets to the complete time axes."""
    a = step_current_charge_responses(left, left_baseline, expected_times_s=expected_times_s)
    b = step_current_charge_responses(right, right_baseline, expected_times_s=expected_times_s)
    current = compare_responses("regular_current_response", a[0], b[0])
    charge = compare_responses("integrated_charge_response", a[1], b[1])
    return {"scope": "regular_current_and_impulse_inclusive_charge_response_comparison",
            "regular_current": current, "integrated_charge": charge,
            "baseline_contact_current_A_m2": [left_baseline, right_baseline],
            "within_compared_budgets": current["passed"] and charge["passed"]}


def compare_pair_step_current_charge(left, right, left_dc, right_dc, *, expected_times_s=None,
                                     left_prepared=None, right_prepared=None,
                                     left_dc_prepared=None, right_dc_prepared=None):
    """Apply the original budgets to full pair-baseline-subtracted responses."""
    left_inputs = dict(step_prepared=left_prepared, dc_prepared=left_dc_prepared)
    right_inputs = dict(step_prepared=right_prepared, dc_prepared=right_dc_prepared)
    a = pair_step_current_charge_responses(left, left_dc, expected_times_s=expected_times_s, **left_inputs)
    b = pair_step_current_charge_responses(right, right_dc, expected_times_s=expected_times_s, **right_inputs)
    current = compare_responses("regular_current_response", a[0], b[0])
    charge = compare_responses("integrated_charge_response", a[1], b[1])
    from .one_dimensional_mechanism_r1_precision import pair_words
    return {"scope": "regular_current_and_impulse_inclusive_charge_response_comparison",
            "representation": "float64-pair-v1", "regular_current": current, "integrated_charge": charge,
            "baseline_contact_current_pair_A_m2": [pair_words(_pair_contact_baseline(left, left_dc, **left_inputs)),
                                                  pair_words(_pair_contact_baseline(right, right_dc, **right_inputs))],
            "baseline_applications": [{"step_prepared_sha256": record["prepared_sha256"],
                                       "dc_prepared_sha256": dc["prepared_sha256"]}
                                      for record, dc in ((left, left_dc), (right, right_dc))],
            "reported_transient_observations": "original_binary64_regular_current_impulse_and_integrated_charge",
            "subtraction": "pair_dc_baseline_and_pair_baseline_times_time_before_response_rounding",
            "within_compared_budgets": current["passed"] and charge["passed"]}


def compare_amplitude_halving(coarse, fine, *, coarse_amplitude_V, fine_amplitude_V,
                              coarse_current_error, fine_current_error):
    """Compare delta-j/a using measured numerical-error budgets in S/m2.

    Current error inputs are nonnegative absolute A/m2 error bounds with the
    same coordinates and components. Their individually voltage-normalized
    sum must itself be at most 1% of each point's normalized signal. Unresolved
    or zero signals report an undetermined linear range, not a passing result.
    """
    _aligned(coarse, fine, real=True)
    _aligned(coarse, coarse_current_error, real=True)
    _aligned(coarse, fine_current_error, real=True)
    a = _real_scalar(coarse_amplitude_V, "coarse_amplitude_V")
    b = _real_scalar(fine_amplitude_V, "fine_amplitude_V")
    if a not in AMPLITUDES_V or b not in AMPLITUDES_V or a != 2*b:
        raise ValueError("amplitude comparison requires adjacent declared positive halving levels")
    error_a, error_b = coarse_current_error.values, fine_current_error.values
    if np.any(error_a < 0) or np.any(error_b < 0):
        raise ValueError("current numerical-error bounds must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_a, normalized_b = coarse.values/a, fine.values/b
        absolute = error_a/a + error_b/b
        scale = np.maximum(np.abs(normalized_a), np.abs(normalized_b))
        signal_budget = _AMPLITUDE_MAXIMUM_ERROR_SIGNAL_FRACTION*scale
    resolved = np.isfinite(absolute) & np.isfinite(scale) & (scale > 0) & (absolute <= signal_budget)

    def reasons(index):
        if not np.isfinite(error_a[index]) or not np.isfinite(error_b[index]):
            return ["nonfinite_numerical_error_bound"]
        if not resolved[index]:
            return ["linearity_signal_not_resolved"]
        return []

    report = _pointwise(normalized_a, normalized_b, coarse, absolute=absolute,
                        relative=_AMPLITUDE_RESPONSE_RELATIVE_TOLERANCE, extra_reasons=reasons)
    for failure in report["failures"]:
        index = tuple(failure["index"])
        failure["absolute_numerical_error_budget_S_m2"] = _evidence_number(absolute[index])
        failure["maximum_resolvable_error_budget_S_m2"] = _evidence_number(signal_budget[index])
    failed = bool(report["failures"])
    valid = all(np.all(np.isfinite(value)) for value in (
        normalized_a, normalized_b, error_a, error_b, absolute, scale,
    )) and not any("nonfinite_comparison_arithmetic" in failure["reasons"]
                   for failure in report["failures"])
    # A known discrepancy remains a discrepancy even when another point (or
    # that same point) is unresolved. Resolution only limits a passing claim.
    exceeds_budget = any("response_difference_exceeds_budget" in failure["reasons"]
                         for failure in report["failures"])
    if not valid:
        status = "invalid_comparison"
    elif exceeds_budget:
        status = "response_not_linear"
    elif not np.all(resolved):
        status = "linearity_undetermined"
    else:
        status = "within_linearity_budget"
    return {
        "scope": "amplitude_halving_comparison_only", "units": "S/m2",
        "passed": not failed,
        "status": status,
        "coarse_amplitude_V": a, "fine_amplitude_V": b,
        "absolute_budget_definition": "coarse_current_error/coarse_amplitude + fine_current_error/fine_amplitude",
        "relative_tolerance": _AMPLITUDE_RESPONSE_RELATIVE_TOLERANCE,
        "maximum_error_budget_signal_fraction": _AMPLITUDE_MAXIMUM_ERROR_SIGNAL_FRACTION,
        "absolute_numerical_error_budget_S_m2": _evidence_array(absolute),
        "normalized_response_scale_S_m2": _evidence_array(scale),
        "resolved": resolved.tolist(),
        "unresolved_scalar_count": int(np.count_nonzero(~resolved)),
        "failure_count": len(report["failures"]), **report,
    }


__all__ = [
    "R1ConvergenceCase", "R1Response", "convergence_cases", "base_convergence_cases",
    "observation_times", "validate_time_substeps", "compare_responses", "compare_amplitude_halving",
    "step_current_charge_responses", "compare_step_current_charge",
    "pair_step_current_charge_responses", "compare_pair_step_current_charge",
    "validate_single_axis_comparison",
    "BASE_INTERVALS", "ALLOWED_INTERVALS", "BASE_TIME_SUBSTEPS", "ALLOWED_TIME_SUBSTEPS",
    "BASE_NONLINEAR_FACTORS", "ALLOWED_NONLINEAR_FACTORS", "AMPLITUDES_V",
]
