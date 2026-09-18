"""Caller-frozen R1 observation windows, separate from result completeness.

Building a specification describes requested times; it certifies no result.
Device windows use the unchanged section-6 decade extensions and twelve
intervals per decade. Short functional runs remain diagnostic and need not
pretend to be either full device windows or analytic fixtures.
"""

from collections.abc import Mapping
import hashlib
import json

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import observation_times


WINDOW_SCHEMA = "R1ObservationWindowV1"
_WINDOW_KEYS = frozenset(("schema", "domain", "times_s", "first_time_s", "last_time_s",
                          "zero_time_definition", "zero_minus_definition",
                          "impulse_charge_definition", "sampling"))


def _time_axis(values):
    if isinstance(values, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("window times cannot contain booleans")
    array = np.asarray(values)
    if (array.dtype.kind not in "fiu" or array.ndim != 1 or len(array) < 3
            or not np.all(np.isfinite(array)) or array[0] != 0 or np.any(np.diff(array) <= 0)):
        raise ValueError("window requires 0+ and at least two finite increasing positive times")
    return np.asarray(array, dtype=float)


def build_window_spec(times_s, domain="device"):
    """Describe one exact requested axis without deriving it from a result."""
    if domain not in ("device", "analytic_fixture"):
        raise ValueError("window domain must be device or analytic_fixture")
    times = _time_axis(times_s)
    if domain == "device":
        expected = observation_times(first_time_s=float(times[1]), last_time_s=float(times[-1]))
        if not np.array_equal(times, expected):
            raise ValueError("device window must contain every section-6 tick: twelve intervals per decade")
    return {"schema": WINDOW_SCHEMA, "domain": domain, "times_s": times.tolist(),
            "first_time_s": float(times[1]), "last_time_s": float(times[-1]),
            "zero_time_definition": "zero_plus_regular_current",
            "zero_minus_definition": "separate_prepared_state",
            "impulse_charge_definition": "separate_from_regular_current",
            "sampling": {"kind": "logarithmic" if domain == "device" else "explicit_fixture",
                         "intervals_per_decade": 12 if domain == "device" else None,
                         "includes_positive_endpoints": True}}


def validate_window_spec(spec):
    """Return a detached canonical record, rejecting unknown/changed rules."""
    if not isinstance(spec, Mapping) or set(spec) != _WINDOW_KEYS:
        raise ValueError("window specification requires exactly the declared fields")
    expected = build_window_spec(spec["times_s"], spec["domain"])
    if (spec.get("schema") != WINDOW_SCHEMA or not isinstance(spec.get("sampling"), Mapping)
            or type(spec["sampling"].get("includes_positive_endpoints")) is not bool
            or type(spec.get("first_time_s")) not in (int, float)
            or type(spec.get("last_time_s")) not in (int, float)):
        raise ValueError("invalid window specification metadata")
    if spec["domain"] == "device" and type(spec["sampling"].get("intervals_per_decade")) is not int:
        raise ValueError("window interval count must be an integer")
    actual = dict(spec)
    actual["times_s"] = _time_axis(spec["times_s"]).tolist()
    actual["sampling"] = dict(spec["sampling"])
    if actual != expected:
        raise ValueError("window metadata differs from its exact requested axis and rules")
    return expected


def window_spec_digest(spec):
    """Hash a validated window with the same canonical JSON as evidence_digest."""
    payload = json.dumps(validate_window_spec(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_window_relation(base, other, *, relation):
    """Check a predeclared T/10T or earlier-start comparison, never infer one."""
    base, other = validate_window_spec(base), validate_window_spec(other)
    if base["domain"] != other["domain"]:
        raise ValueError("window comparison cannot cross device/fixture domains")
    if relation == "tenfold_extension":
        valid = other["first_time_s"] == base["first_time_s"] and other["last_time_s"] == 10*base["last_time_s"]
    elif relation == "earlier_start":
        valid = other["last_time_s"] == base["last_time_s"] and other["first_time_s"] == base["first_time_s"]/10
    else:
        raise ValueError("unknown window comparison relation")
    if not valid or not np.all(np.isin(base["times_s"], other["times_s"])):
        raise ValueError("windows do not have the declared relation and shared exact ticks")
    return {"relation": relation, "base_window_spec_sha256": window_spec_digest(base),
            "other_window_spec_sha256": window_spec_digest(other), "verified": True}


__all__ = ["WINDOW_SCHEMA", "build_window_spec", "validate_window_spec", "window_spec_digest",
           "validate_window_relation"]
