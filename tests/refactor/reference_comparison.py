"""Small, solver-independent comparisons for the P00 reference inventory.

Callers supply an approved gate and two explicitly bound tables.  Each table
contains one scalar, one curve, or a two-parameter grid; branches are separate
tables.  There is no interpolation, unit conversion, fitting, or solver import.
"""

from collections import Counter
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
import math
import struct


IDENTITY_FIELDS = (
    "source_sha256", "input_sha256", "protocol_sha256", "initial_state_sha256",
    "model_sha256", "numerics_sha256", "environment_sha256", "driver",
)
DIMENSIONS = {"scalar": 0, "curve": 1, "parameter_grid": 2}


class Rejected(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _require(condition, code, detail):
    if not condition:
        raise Rejected(code, detail)


def _number(value, label, nonnegative=False):
    valid = type(value) in (int, float)
    try:
        valid = valid and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, "invalid_number", f"{label} must be a finite number")
    _require(not nonnegative or value >= 0, "invalid_number", f"{label} must be nonnegative")
    return value


def _timestamp(value, label):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise Rejected("invalid_time", f"{label} must be an ISO timestamp") from None
    _require(result.tzinfo is not None, "invalid_time", f"{label} must include a timezone")
    return result


def _identity(value, label):
    _require(isinstance(value, dict), "missing_identity", f"{label} must be an identity object")
    for field in IDENTITY_FIELDS:
        item = value.get(field)
        _require(isinstance(item, str) and bool(item.strip()), "missing_identity", f"{label}.{field} is missing")
        if field.endswith("sha256"):
            _require(len(item) == 64 and all(c in "0123456789abcdef" for c in item),
                     "invalid_identity", f"{label}.{field} must be a lowercase SHA-256")
    return value


def _coordinate(raw, dimension):
    _require(isinstance(raw, (list, tuple)) and len(raw) == dimension,
             "coordinate_mismatch", f"Expected {dimension} coordinates, received {raw!r}")
    # Decimal preserves large integer keys; nearby floats are never rounded to
    # one another. Any source-declared coordinate mapping happens before here.
    return tuple(Decimal(str(_number(v, "coordinate"))) for v in raw)


def _points(table, expected, dimension, side):
    raw = table.get("points")
    _require(isinstance(raw, list), "missing_points", f"{side}.points must be a list")
    indexed = {}
    statuses = {"ok", "unknown", "reference_missing"} if side == "reference" else {"ok", "unknown", "solver_failed"}
    for point in raw:
        _require(isinstance(point, dict), "invalid_point", f"Invalid {side} point")
        key = _coordinate(point.get("coordinates"), dimension)
        _require(key not in indexed, "duplicate_coordinate", f"Duplicate {side} coordinate {point['coordinates']!r}")
        status = point.get("status")
        _require(status in statuses, "invalid_status", f"Invalid {side} status {status!r}")
        if status == "ok":
            _number(point.get("value"), f"{side}.value")
            origins = {"native", "extracted"} if side == "reference" else {"native"}
            _require(point.get("origin") in origins, "unqualified_point",
                     f"{side} point must be native data or an explicitly extracted reference; interpolated points do not count")
        else:
            _require(point.get("value") is None, "invalid_status", f"{side} {status} must have null value")
            _require(isinstance(point.get("reason"), str) and bool(point["reason"].strip()),
                     "missing_reason", f"{side} {status} needs an explicit reason")
        indexed[key] = point
    missing, extra = set(expected) - set(indexed), set(indexed) - set(expected)
    _require(not missing and not extra, "coordinate_mismatch",
             f"{side} coordinates differ: missing={sorted(missing)!r}, extra={sorted(extra)!r}")
    return indexed


def compare(reference, candidate, gate):
    """Return an auditable report, including rejected or unresolved inputs.

    Thresholds and each side's complete identity are supplied independently in
    gate, so a planned source migration is possible but an unexpected source is
    rejected. A frozen candidate identity is not inferred from its output.
    """
    try:
        return _compare(reference, candidate, gate)
    except Rejected as exc:
        return {"accepted": False, "status": "rejected", "code": exc.code,
                "detail": str(exc), "rows": []}


def _compare(reference, candidate, gate):
    _require(all(isinstance(v, dict) for v in (reference, candidate, gate)),
             "invalid_input", "Reference, candidate, and gate must be objects")
    for field in ("id", "version", "scope", "quantity", "unit", "evidence_level"):
        _require(isinstance(gate.get(field), str) and bool(gate[field].strip()),
                 "incomplete_gate", f"gate.{field} is required")
    _require(gate["evidence_level"] in {"historical", "numerical", "external"},
             "incomplete_gate", "Unknown evidence level")
    _require(gate.get("status") == "frozen", "unreviewed_gate", "Gate is not frozen")
    review = gate.get("independent_review")
    _require(isinstance(review, dict) and review.get("status") == "approved"
             and all(isinstance(review.get(k), str) and review[k].strip() for k in ("reviewer", "evidence")),
             "unreviewed_gate", "An independent review and its evidence are required")
    for field in ("atol", "rtol", "scale", "reference_error_abs", "reference_error_evidence", "rule", "zero_policy"):
        _require(field in gate and gate[field] is not None, "missing_threshold", f"gate.{field} is required")
    atol = _number(gate["atol"], "atol", True)
    rtol = _number(gate["rtol"], "rtol", True)
    ref_error = _number(gate["reference_error_abs"], "reference_error_abs", True)
    _require(isinstance(gate["reference_error_evidence"], str) and bool(gate["reference_error_evidence"].strip()),
             "missing_threshold", "Reference-error evidence is missing")
    _require(isinstance(gate["rule"], str) and gate["rule"] in {"sum", "max", "bitwise"},
             "incomplete_gate", "Unknown comparison rule")
    _require(gate["zero_policy"] == "absolute_limit", "incomplete_gate", "Zero signals require the explicit absolute limit")
    if gate["scale"] != "abs_reference":
        _number(gate["scale"], "scale", True)
    if gate["rule"] == "bitwise":
        _require(atol == rtol == ref_error == 0 and gate["evidence_level"] == "historical",
                 "invalid_bitwise_gate", "Bitwise migration has zero tolerances and only proves historical payload equivalence")

    frozen = _timestamp(gate.get("frozen_at"), "gate.frozen_at")
    started = _timestamp(candidate.get("run_started_at"), "candidate.run_started_at")
    created = _timestamp(candidate.get("created_at"), "candidate.created_at")
    _require(frozen < started <= created, "posthoc_gate",
             "Gate must be frozen before the actual run starts; artifact creation must not precede the run")
    kind = gate.get("kind")
    _require(isinstance(kind, str) and kind in DIMENSIONS, "incomplete_gate", "Expected scalar, curve, or parameter_grid")
    dimension = DIMENSIONS[kind]
    axes = gate.get("axes")
    _require(isinstance(axes, list) and len(axes) == dimension,
             "incomplete_gate", "Gate axes do not match its kind")
    _require(all(isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"]
                 and isinstance(a.get("unit"), str) and a["unit"] for a in axes),
             "incomplete_gate", "Every axis needs a name and unit")
    _require(len({a["name"] for a in axes}) == dimension, "incomplete_gate", "Axis names must be distinct")
    raw_coordinates = gate.get("expected_coordinates")
    _require(isinstance(raw_coordinates, list) and bool(raw_coordinates), "incomplete_gate", "Expected coordinates are required")
    expected = [_coordinate(c, dimension) for c in raw_coordinates]
    _require(len(set(expected)) == len(expected), "duplicate_coordinate", "Gate coordinates must be unique")
    _require(kind != "scalar" or len(expected) == 1, "incomplete_gate", "A scalar has exactly one value")

    for side, table in (("reference", reference), ("candidate", candidate)):
        identity = _identity(table.get("identity"), f"{side}.identity")
        target = _identity(gate.get(f"{side}_identity"), f"gate.{side}_identity")
        _require(identity == target, "identity_mismatch", f"{side} identity differs from its frozen binding")
        for field in ("scope", "kind", "quantity", "unit", "axes"):
            _require(table.get(field) == gate[field], "measurement_mismatch", f"{side}.{field} differs from the gate")
    if gate["rule"] == "bitwise":
        left, right = gate["reference_identity"], gate["candidate_identity"]
        _require(all(left[k] == right[k] for k in IDENTITY_FIELDS if k != "source_sha256"),
                 "invalid_bitwise_gate", "Bitwise migration requires the same effective inputs, protocol, initial state, models, driver, numerics, and environment")

    refs = _points(reference, expected, dimension, "reference")
    candidates = _points(candidate, expected, dimension, "candidate")
    rows = []
    for raw, key in zip(raw_coordinates, expected):
        ref, new = refs[key], candidates[key]
        row = {"coordinates": list(raw), "reference_status": ref["status"], "candidate_status": new["status"],
               "reference_value": ref.get("value"), "candidate_value": new.get("value")}
        if ref["status"] != "ok" or new["status"] != "ok":
            row.update(status="unresolved", reference_reason=ref.get("reason"), candidate_reason=new.get("reason"))
        else:
            scale = abs(ref["value"]) if gate["scale"] == "abs_reference" else gate["scale"]
            limit = max(atol, rtol * scale) if gate["rule"] == "max" else atol + rtol * scale
            _number(limit, "computed limit", True)
            _require(ref_error <= limit / 3, "reference_budget_exceeded",
                     f"Reference error exceeds one third of the allowed error at {raw!r}")
            error = abs(new["value"] - ref["value"])
            _number(error, "absolute error", True)
            equal = error <= limit
            if gate["rule"] == "bitwise":
                # Do not round arbitrary integers through float64. Preserve
                # integer values, float64 bits, and the declared scalar type.
                equal = type(new["value"]) is type(ref["value"])
                if equal and type(ref["value"]) is float:
                    equal = struct.pack("!d", new["value"]) == struct.pack("!d", ref["value"])
                elif equal:
                    equal = new["value"] == ref["value"]
            row.update(status="passed" if equal else "failed", absolute_error=error,
                       scale=scale, limit=limit, margin=limit - error)
            if not equal:
                row["reason"] = "numeric_payload_mismatch" if gate["rule"] == "bitwise" else "tolerance_exceeded"
        rows.append(row)
    accepted = all(r["status"] == "passed" for r in rows)
    return {"gate_id": gate["id"], "gate_version": gate["version"], "evidence_level": gate["evidence_level"],
            "accepted": accepted, "status": "passed" if accepted else "not_passed",
            "counts": dict(Counter(r["status"] for r in rows)),
            "reference_status_counts": dict(Counter(p["status"] for p in refs.values())),
            "candidate_status_counts": dict(Counter(p["status"] for p in candidates.values())), "rows": rows}


def hysteresis_from_power(forward, reverse, definition, *, forward_error_abs=None, reverse_error_abs=None):
    """Use supplied absolute power-error bounds, never a fitted denominator floor.

    Bounds use the same power units as the inputs. Missing bounds or a
    denominator interval containing zero leave HI unresolved. Exact synthetic
    numbers may explicitly supply zero bounds; physical runs need evidence.
    """
    _require(definition in {"HI_P", "HI_normalized"}, "unknown_definition", "An explicit HI definition is required")
    _number(forward, "forward power", True)
    _number(reverse, "reverse power", True)
    denominator = forward if definition == "HI_P" else reverse
    if denominator == 0:
        return {"definition": definition, "status": "unknown", "value": None, "reason": "zero_denominator"}
    if forward_error_abs is None or reverse_error_abs is None:
        return {"definition": definition, "status": "unknown", "value": None, "reason": "power_uncertainty_missing"}
    _number(forward_error_abs, "forward power error", True)
    _number(reverse_error_abs, "reverse power error", True)
    denominator_error = forward_error_abs if definition == "HI_P" else reverse_error_abs
    if denominator <= denominator_error:
        return {"definition": definition, "status": "unknown", "value": None, "reason": "denominator_unresolved"}

    def ratio(fwd, rev):
        return rev / fwd - 1 if definition == "HI_P" else (rev - fwd) / rev

    # Exact rational endpoints avoid erasing a supplied sub-ULP uncertainty.
    forward, reverse = Fraction(forward), Fraction(reverse)
    forward_error_abs, reverse_error_abs = Fraction(forward_error_abs), Fraction(reverse_error_abs)
    value = float(ratio(forward, reverse))
    _number(value, definition)
    corners = [ratio(fwd, rev) for fwd in (forward - forward_error_abs, forward + forward_error_abs)
               for rev in (reverse - reverse_error_abs, reverse + reverse_error_abs)]
    exact_bound = max(abs(v - Fraction(value)) for v in corners)
    bound = float(exact_bound)
    if Fraction(bound) < exact_bound:
        bound = math.nextafter(bound, math.inf)
    _number(bound, "propagated HI error", True)
    return {"definition": definition, "status": "ok", "value": value, "absolute_error_bound": bound}
