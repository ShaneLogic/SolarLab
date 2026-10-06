"""Small, source-bound physical comparisons for prospective P00 gates.

These functions compare supplied observations, not solutions of a device.
They never choose a scale, floor, gauge, mask, tolerance or reference error.
R1 full-protocol acceptance remains with its original bound comparators.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import struct

from reference_comparison import (
    Rejected, _identity, _number, _require, _timestamp,
)


NORMS = {
    "potential": "gauge_cv_rms_linf",
    "density": "log_cv_rms_linf_component_inventory",
    "cells": "conservative_l1_component_inventory_traces",
    "complex": "complex_re_im_difference_circular_phase",
    "semantic": "exact_structure_and_words",
}
DEFINITION_FIELDS = ("scope", "comparison_target", "quantity", "unit", "norm",
                     "support", "mask", "zero_policy")


def definition_fingerprint(gate):
    """Bind the complete gate, including error evidence and review, before data."""
    return hashlib.sha256(json.dumps({k: v for k, v in gate.items() if k != "definition_sha256"},
                                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fraction(value, name):
    return Fraction(_number(value, name))


def _vector(value, name):
    _require(isinstance(value, (list, tuple)) and bool(value), "shape", name)
    return [_fraction(x, name) for x in value]


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_definition(gate):
    """Check a numeric rule without granting reference or run eligibility."""
    _require(isinstance(gate, dict), "incomplete_gate", "Gate must be an object")
    _require(gate.get("kind") in NORMS, "incomplete_gate", "Unknown physical kind")
    for key in DEFINITION_FIELDS:
        _require(key in gate and gate[key] is not None, "incomplete_gate", key)
    for key in ("scope", "quantity", "unit", "norm", "support", "zero_policy"):
        _require(isinstance(gate[key], str) and bool(gate[key]), "incomplete_gate", key)
    _require(gate["norm"] == NORMS[gate["kind"]], "wrong_norm", "Kind/norm mismatch")
    target = gate["comparison_target"]
    _require(isinstance(target, dict) and target.get("kind") in
             {"fixed_discrete", "continuum_projected", "analytic_fixture", "engineering_semantics"}
             and bool(target.get("id")), "wrong_target", "Explicit comparison target required")
    _require(target.get("family") != "R1_full", "R1_contract_required", "Use the original R1 contract")
    _require(isinstance(gate.get("thresholds"), dict) and bool(gate["thresholds"]),
             "missing_threshold", "Named thresholds are required")
    for name, rule in gate["thresholds"].items():
        _require(isinstance(rule, dict), "missing_threshold", name)
        for key in ("atol", "rtol", "scale"):
            _number(rule.get(key), name+"."+key, True)
        _require(bool(rule.get("unit")) and bool(rule.get("basis")),
                 "missing_threshold", "Each metric needs units and scale/threshold provenance")
    _require(isinstance(gate.get("source_bindings"), list) and bool(gate["source_bindings"]),
             "missing_source", "Frozen source bindings required")
    _require(all(_hash(x.get("sha256")) and x.get("path") for x in gate["source_bindings"]),
             "missing_source", "Invalid source binding")
    return True


def _eligible(reference, candidate, gate):
    validate_definition(gate)
    expected = definition_fingerprint(gate)
    _require(reference.get("definition_sha256") == candidate.get("definition_sha256")
             == gate.get("definition_sha256") == expected,
             "definition_mismatch", "Frozen numerical rules or geometry have changed")
    _require(gate.get("status") == "frozen", "unreviewed_gate", "Gate is not frozen")
    review = gate.get("independent_review", {})
    _require(review.get("status") == "approved" and bool(review.get("evidence")),
             "unreviewed_gate", "Independent review required")
    _require(gate.get("reference_qualified") is True, "reference_unqualified", "Reference is not qualified")
    _require(_timestamp(gate.get("frozen_at"), "freeze") <
             _timestamp(candidate.get("run_started_at"), "run"),
             "posthoc_gate", "Freeze must precede the candidate run")
    for side, data in (("reference", reference), ("candidate", candidate)):
        _require(_identity(data.get("identity"), side) == _identity(gate.get(side+"_identity"), side),
                 "identity_mismatch", side)
        for key in DEFINITION_FIELDS:
            _require(key in data and data[key] == gate[key], "measurement_mismatch", side+"."+key)
    errors = gate.get("reference_errors")
    _require(isinstance(errors, dict) and set(errors) == set(gate["thresholds"]),
             "reference_error_missing", "Every metric needs an error bound")
    for name, error in errors.items():
        _number(error.get("bound"), "reference error", True)
        _require(error.get("unit") == gate["thresholds"][name]["unit"]
                 and _hash(error.get("evidence_sha256")), "reference_error_missing", name)


def _profile(reference, candidate, gate):
    a, b = _vector(reference.get("values"), "reference"), _vector(candidate.get("values"), "candidate")
    w = _vector(gate.get("volumes"), "control volumes")
    _require(len(a) == len(b) == len(w) and all(v > 0 for v in w), "geometry", "Positive CV weights required")
    _require(gate["mask"] == [True]*len(a), "mask", "This comparator covers every declared CV; no hidden exclusions")
    _require(reference.get("volumes") == candidate.get("volumes") == gate["volumes"],
             "geometry", "CV geometry must match the frozen geometry")
    return a, b, w


def _potential(reference, candidate, gate):
    a, b, w = _profile(reference, candidate, gate)
    _require(gate["unit"] == "V" and gate["zero_policy"] == "fixed_physical_scale",
             "unit_or_zero_policy", "Potential uses volts and a fixed physical scale")
    _require(all(rule["unit"] == "V" for rule in gate["thresholds"].values()),
             "metric_unit", "Potential errors are in volts")
    gauge = gate.get("gauge", {})
    if gauge.get("kind") == "cv_zero_mean":
        _require(gauge.get("physical_gauge_freedom") is True, "gauge", "Floating gauge must be explicit")
        am, bm = sum(x*v for x, v in zip(a, w))/sum(w), sum(x*v for x, v in zip(b, w))/sum(w)
        a, b = [x-am for x in a], [x-bm for x in b]
    elif gauge.get("kind") == "fixed_contact":
        i = gauge.get("index")
        _require(type(i) is int and 0 <= i < len(a), "gauge", "Missing contact gauge")
        value = _fraction(gauge.get("value_V"), "gauge value")
        error = _fraction(gauge.get("absolute_limit_V"), "gauge tolerance")
        _require(error >= 0 and abs(a[i]-value) <= error and abs(b[i]-value) <= error,
                 "gauge", "A physical Dirichlet error cannot be removed as a gauge shift")
    else:
        raise Rejected("gauge", "A common physical gauge is required")
    delta = [abs(x-y) for x, y in zip(a, b)]
    return {"rms": ("squared", sum(v*x*x for v, x in zip(w, delta))/sum(w)), "max": max(delta)}


def _log_absolute_upper(a, b):
    if a == b:
        return Fraction(0)
    # Decimal.ln is correctly rounded; adjacent80-digit values enclose it.
    # Rational subtraction/squaring then preserves this arithmetic allowance.
    with localcontext() as ctx:
        ctx.prec = 80
        # Enclose rational-to-decimal rounding too; large integers must not
        # first pass through binary64 and erase a weak but represented change.
        left = Decimal(a.numerator)/Decimal(a.denominator)
        right = Decimal(b.numerator)/Decimal(b.denominator)
        lo = Fraction(right.next_minus().ln().next_minus())-Fraction(left.next_plus().ln().next_plus())
        hi = Fraction(right.next_plus().ln().next_plus())-Fraction(left.next_minus().ln().next_minus())
    return max(abs(lo), abs(hi))


def _density(reference, candidate, gate):
    a, b, w = _profile(reference, candidate, gate)
    _require((gate["unit"], gate.get("volume_unit"), gate.get("inventory_unit")) in
             {("m-3", "m3", "1"), ("m-3", "m", "m-2")},
             "metric_unit", "Density and integration measure units must be explicit")
    for name, rule in gate["thresholds"].items():
        unit = "1" if name.startswith("log_") else gate["unit"] if name == "zero_density_max" else gate["inventory_unit"]
        _require(rule["unit"] == unit, "metric_unit", name)
    _require(gate["zero_policy"] == "explicit_zero_cells_no_floor", "zero_policy", "No density floor is permitted")
    zero = gate.get("zero_cells")
    components = gate.get("components")
    _require(isinstance(zero, list) and len(set(zero)) == len(zero)
             and all(type(i) is int and 0 <= i < len(a) for i in zero), "zero_policy", "Explicit zero-cell set required")
    _require(isinstance(components, list) and len(components) == len(a)
             and all(isinstance(x, str) and x for x in components), "support", "Connected-component identities required")
    _require(reference.get("components") == candidate.get("components") == components,
             "support", "Connected components differ")
    positive = [i for i in range(len(a)) if i not in zero]
    _require(all(a[i] == 0 and b[i] >= 0 for i in zero)
             and all(a[i] > 0 and b[i] > 0 for i in positive), "density_domain", "Zero species and positive species must be explicit")
    logs = [_log_absolute_upper(a[i], b[i]) for i in positive]
    weight = sum(w[i] for i in positive)
    metrics = {"log_rms": ("squared", sum(w[i]*z*z for i, z in zip(positive, logs))/weight if weight else Fraction(0)),
               "log_max": max(logs, default=Fraction(0)),
               "zero_density_max": max((abs(b[i]) for i in zero), default=Fraction(0))}
    for component in sorted(set(components)):
        metrics["inventory:"+component] = abs(sum(v*(y-x) for x, y, v, c in zip(a, b, w, components) if c == component))
    return metrics


def _cells(reference, candidate, gate):
    _require(gate["zero_policy"] == "absolute_integrals_no_floor", "zero_policy", "Cell integrals need an explicit absolute branch")
    _require(gate["mask"] == "all_cells_and_both_interface_sides", "mask", "Every cell and trace is required")
    _require(reference.get("representation") == candidate.get("representation") == "cell_average",
             "projection", "Only declared piecewise-constant CV averages are supported")
    if "overlap" in gate:
        return _overlap_cells(reference, candidate, gate)
    area = _fraction(gate.get("area"), "cross-sectional area")
    _require(area > 0, "geometry", "Area is not a default")
    _require(gate.get("area_unit") == "m2" and gate.get("edge_unit") == "m"
             and (gate["unit"], gate.get("integral_unit")) in
             {("m-3", "1"), ("C m-3", "C"), ("1", "m3")},
             "metric_unit", "The conservative integral must have the declared physical units")
    for name, rule in gate["thresholds"].items():
        _require(rule["unit"] == (gate["unit"] if name.startswith("trace:") else gate["integral_unit"]),
                 "metric_unit", name)
    partitions = gate.get("partitions")
    _require(isinstance(partitions, list) and bool(partitions), "support", "Frozen connected regions required")
    _require(all(isinstance(p.get("id"), str) and p["id"] for p in partitions), "support", "Region identity is missing")
    intervals = [(str(p["id"]), _fraction(p["left"], "left"), _fraction(p["right"], "right")) for p in partitions]
    _require(len({x[0] for x in intervals}) == len(intervals) and all(a < b for _, a, b in intervals)
             and all(x[2] == y[1] for x, y in zip(intervals, intervals[1:])), "support", "Invalid region partition")
    edges, values, inventories = [], [], []
    for data in (reference, candidate):
        x, y = _vector(data.get("edges"), "CV edges"), _vector(data.get("values"), "CV averages")
        _require(len(x) == len(y)+1 and all(a < b for a, b in zip(x, x[1:]))
                 and x[0] == intervals[0][1] and x[-1] == intervals[-1][2], "geometry", "Grid coverage differs")
        _require(all(a in x and b in x for _, a, b in intervals), "interface", "Cells cannot cross a declared interface")
        inventories.append({name: sum(v*(b-a)*area for a, b, v in zip(x, x[1:], y) if lo <= a and b <= hi)
                            for name, lo, hi in intervals})
        edges.append(x); values.append(y)
    i = j = 0; l1 = Fraction(0)
    while i < len(values[0]) and j < len(values[1]):
        lo, hi = max(edges[0][i], edges[1][j]), min(edges[0][i+1], edges[1][j+1])
        if hi > lo:
            l1 += (hi-lo)*area*abs(values[0][i]-values[1][j])
        end_a, end_b = edges[0][i+1], edges[1][j+1]
        i += end_a <= end_b; j += end_b <= end_a
    metrics = {"l1": l1, **{"inventory:"+k: abs(inventories[1][k]-v) for k, v in inventories[0].items()}}
    trace_names = [str(p["id"])+":"+side for p in partitions[:-1] for side in ("left", "right")]
    for data in (reference, candidate):
        _require(isinstance(data.get("traces"), dict) and set(data["traces"]) == set(trace_names),
                 "interface", "Both interface traces must be independently supplied")
    for name in trace_names:
        metrics["trace:"+name] = abs(_fraction(candidate["traces"][name], name)-_fraction(reference["traces"][name], name))
    return metrics


def _overlap_cells(reference, candidate, gate):
    """Any dimension, given an independently verified frozen CV overlap table."""
    projection = gate.get("projection_evidence", {})
    _require(projection.get("status") == "approved" and _hash(projection.get("sha256"))
             and projection.get("geometry_verified") is True,
             "projection", "Mass sums alone do not establish geometric overlap")
    _require(gate.get("volume_unit") == "m3" and (gate["unit"], gate.get("integral_unit")) in
             {("m-3", "1"), ("C m-3", "C"), ("1", "m3")}, "metric_unit", "Overlap volumes need physical units")
    a, b = _vector(reference.get("values"), "reference averages"), _vector(candidate.get("values"), "candidate averages")
    va, vb = _vector(gate.get("reference_volumes"), "reference volumes"), _vector(gate.get("candidate_volumes"), "candidate volumes")
    ca, cb = gate.get("reference_components"), gate.get("candidate_components")
    _require(len(a) == len(va) == len(ca) and len(b) == len(vb) == len(cb)
             and all(v > 0 for v in va+vb), "geometry", "Overlap geometry shape")
    _require(reference.get("volumes") == gate["reference_volumes"] and candidate.get("volumes") == gate["candidate_volumes"]
             and reference.get("components") == ca and candidate.get("components") == cb,
             "geometry", "Observation geometry differs from the frozen overlap table")
    overlap = [_vector(row, "overlap row") for row in gate["overlap"]]
    _require(len(overlap) == len(a) and all(len(row) == len(b) for row in overlap), "projection", "Overlap shape")
    _require(all(v >= 0 for row in overlap for v in row) and [sum(row) for row in overlap] == va
             and [sum(row[j] for row in overlap) for j in range(len(b))] == vb,
             "projection", "Overlap must exactly conserve each source and destination CV")
    _require(all(v == 0 or ca[i] == cb[j] for i, row in enumerate(overlap) for j, v in enumerate(row)),
             "interface", "No projection across distinct regions")
    metrics = {"l1": sum(v*abs(a[i]-b[j]) for i, row in enumerate(overlap) for j, v in enumerate(row))}
    for name in sorted(set(ca+cb)):
        metrics["inventory:"+name] = abs(sum(v*x for v, x, c in zip(va, a, ca) if c == name)
                                         -sum(v*x for v, x, c in zip(vb, b, cb) if c == name))
    names = [interface+":"+side for interface in gate.get("interface_ids", []) for side in ("left", "right")]
    _require("interface_ids" in gate and all(set(x.get("traces", {})) == set(names) for x in (reference, candidate)),
             "interface", "Both sides of every declared interface are required")
    for name in names:
        metrics["trace:"+name] = abs(_fraction(candidate["traces"][name], name)-_fraction(reference["traces"][name], name))
    for name, rule in gate["thresholds"].items():
        _require(rule["unit"] == (gate["unit"] if name.startswith("trace:") else gate["integral_unit"]), "metric_unit", name)
    return metrics


def _complex(reference, candidate, gate):
    _require(gate["mask"] == "all_frequencies" and gate["zero_policy"] == "explicit_zero_else_resolved_phase",
             "mask_or_zero_policy", "All frequencies and a resolved phase policy are required")
    _require(reference.get("coordinates") == candidate.get("coordinates") == gate.get("coordinates")
             and isinstance(gate.get("coordinates"), list) and bool(gate["coordinates"]),
             "coordinates", "Exact frozen frequencies required")
    frequencies = _vector(gate["coordinates"], "frequencies")
    _require(len(set(frequencies)) == len(frequencies) and all(f > 0 for f in frequencies),
             "coordinates", "Frequencies must be positive and unique")
    floor = _fraction(gate.get("phase_floor"), "phase resolution floor")
    evidence = gate.get("phase_floor_evidence", {})
    _require(floor > 0 and evidence.get("value") == gate["phase_floor"] and _hash(evidence.get("sha256")),
             "phase_floor", "A numeric independently bound resolvability floor is required")
    zero = gate.get("zero_indices")
    _require(isinstance(zero, list) and len(set(zero)) == len(zero), "zero_policy", "Explicit zero frequencies required")
    a, b = reference.get("values"), candidate.get("values")
    _require(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == len(gate["coordinates"]), "shape", "Complex vector length")
    _require(all(type(i) is int and 0 <= i < len(a) for i in zero), "zero_policy", "Invalid zero frequency")
    re, im, squared, phase = [], [], [], []
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = _vector(x, "reference complex"), _vector(y, "candidate complex")
        _require(len(x) == len(y) == 2, "shape", "Use real/imaginary pairs")
        dr, di = y[0]-x[0], y[1]-x[1]
        re.append(abs(dr)); im.append(abs(di)); squared.append(dr*dr+di*di)
        if i in zero:
            _require(x == y == [0, 0], "zero_signal_changed", "A declared exact zero cannot acquire a response")
            phase.append(Fraction(0)); continue
        if x[0]**2+x[1]**2 <= floor**2 or y[0]**2+y[1]**2 <= floor**2:
            raise Rejected("unresolved_phase", "Amplitude is below the frozen resolvability floor")
        real, imaginary = y[0]*x[0]+y[1]*x[1], y[1]*x[0]-y[0]*x[1]
        scale = max(abs(real), abs(imaginary))
        phase.append(Fraction(abs(math.atan2(float(imaginary/scale), float(real/scale)))))
    _require(gate["thresholds"].get("phase", {}).get("unit") == "rad", "phase_unit", "Phase is an absolute angle in radians")
    _require(all(gate["thresholds"].get(name, {}).get("unit") == gate["unit"]
                 for name in ("real", "imag", "difference")), "metric_unit", "Complex component units differ")
    _require(gate["thresholds"]["phase"]["rtol"] == 0, "phase_relative_error", "Relative phase error is not meaningful")
    return {"real": max(re), "imag": max(im), "difference": ("squared", max(squared)), "phase": max(phase)}


def _semantic(reference, candidate, gate):
    _require(gate.get("change_kind") == "M", "word_equality_is_M_only", "Only an M migration can use this comparison")
    exceptions = gate.get("metadata_exceptions")
    _require(isinstance(exceptions, list) and set(exceptions) <= {"source_module", "import_path", "source_location"},
             "metadata_exceptions", "Only explicit nonphysical metadata paths may differ")
    for field in reference["identity"]:
        if field != "source_sha256":
            _require(reference["identity"][field] == candidate["identity"][field], "M_identity", "M requires unchanged effective inputs/numerics")
    def words(value, top=False):
        if isinstance(value, dict):
            return tuple((key, words({k: v for k, v in item.items() if k not in exceptions})
                          if top and key == "metadata" and isinstance(item, dict) else words(item))
                         for key, item in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(words(x) for x in value))
        if type(value) is float:
            _number(value, "numeric payload")
            return ("binary64", struct.pack("!d", value))
        _require(type(value) in (str, int, bool, type(None)), "payload_type", "Unsupported payload type")
        return (type(value).__name__, value)
    _require(gate["zero_policy"] == "exact_words" and all(gate["thresholds"]["mismatch"][k] == 0 for k in ("atol", "rtol")),
             "M_threshold", "An M comparison cannot have a tolerance")
    return {"mismatch": Fraction(words(reference.get("values"), True) != words(candidate.get("values"), True))}


def compare_physical(reference, candidate, gate):
    """Return rejected/unresolved metadata or all named physical metric checks."""
    try:
        _require(all(isinstance(x, dict) for x in (reference, candidate, gate)),
                 "malformed_input", "Observations and gate must be objects")
        _eligible(reference, candidate, gate)
        metrics = {"potential": _potential, "density": _density, "cells": _cells,
                   "complex": _complex, "semantic": _semantic}[gate["kind"]](reference, candidate, gate)
        _require(set(metrics) == set(gate["thresholds"]), "metric_set", "No required metric may be omitted")
        checks = {}
        for name, metric in metrics.items():
            rule, uncertainty = gate["thresholds"][name], Fraction(gate["reference_errors"][name]["bound"])
            limit = Fraction(rule["atol"])+Fraction(rule["rtol"])*Fraction(rule["scale"])
            _require(3*uncertainty <= limit, "reference_budget", name)
            if isinstance(metric, tuple):
                passed = metric[1] <= (limit-uncertainty)**2
                shown = math.sqrt(float(metric[1]))
            else:
                passed = metric+uncertainty <= limit; shown = float(metric)
            exact = metric[1] if isinstance(metric, tuple) else metric
            checks[name] = {"value": shown, "limit": float(limit), "reference_error": float(uncertainty),
                            "exact_measure": [str(exact.numerator), str(exact.denominator)],
                            "decision_uses_exact_squared_metric": isinstance(metric, tuple), "passed": passed}
        return {"status": "passed" if all(x["passed"] for x in checks.values()) else "not_passed",
                "accepted": all(x["passed"] for x in checks.values()), "metrics": checks,
                "scientific_qualification_granted": False}
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError, ZeroDivisionError) as exc:
        code = exc.code if isinstance(exc, Rejected) else "malformed_input"
        return {"status": "unresolved" if code == "unresolved_phase" else "rejected",
                "accepted": False, "code": code, "detail": str(exc), "scientific_qualification_granted": False}
