"""Exact accepted-row schedules for explicitly supplied R1 validation cases.

The arithmetic follows ``_integrate_trace``: calculate one interval duration,
divide it by the refinement level, and calculate each time from the preceding
output point. No observation grid or physical trajectory is generated here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _case_axes(case):
    levels, times = case["time_substeps"], case["times_s"]
    if (not isinstance(levels, (list, tuple)) or not levels
            or any(type(level) is not int or level <= 0 for level in levels)
            or len(set(levels)) != len(levels)):
        raise ValueError("time_substeps must contain distinct positive integers")
    if (not isinstance(times, (list, tuple)) or len(times) < 2
            or any(not _finite(value) for value in times)
            or times[0] != 0.0 or times[1] < 1e-12
            or any(a >= b for a, b in zip(times, times[1:]))):
        raise ValueError("times_s must start at zero then increase finitely")
    return tuple(levels), tuple(float(value) for value in times)


def expected_schedule(case):
    """Return the exact ordered ``substeps/time_s/dt_s`` rows of ``case``."""
    levels, times = _case_axes(case)
    result = []
    for level in levels:
        result.append({"substeps": level, "time_s": times[0], "dt_s": 0.0})
        for point in range(1, len(times)):
            interval = float(times[point] - times[point - 1])
            dt = interval / level
            for step in range(level):
                time = float(times[point - 1] + (step + 1) * dt)
                result.append({"substeps": level, "time_s": time, "dt_s": dt})
    # An increasing output grid can still collapse after float64 subdivision.
    # Reject that invalid schedule instead of accepting duplicate observations.
    previous = {}
    for row in result:
        level, time, dt = (row[key] for key in ("substeps", "time_s", "dt_s"))
        if (not math.isfinite(time) or not math.isfinite(dt)
                or (level in previous and (dt <= 0.0 or time <= previous[level]))):
            raise ValueError("float64 subdivision does not produce finite increasing rows")
        previous[level] = time
    return result


def _row_key(row):
    """Use exact binary64 values, including the sign of zero, for identity."""
    return row["substeps"], float(row["time_s"]).hex(), float(row["dt_s"]).hex()


def schedule_index_map(short_case, long_case):
    """Map every short-case row to its exact identity in the long case.

    Tier-major storage means a short observation window is generally not a
    global contiguous prefix of a longer one. This mapping does not compare
    physical states and refuses any missing or changed time/duration identity.
    """
    short, long = expected_schedule(short_case), expected_schedule(long_case)
    index = {_row_key(row): ordinal for ordinal, row in enumerate(long)}
    try:
        return [index[_row_key(row)] for row in short]
    except KeyError as exc:
        raise ValueError("short schedule is not contained in long schedule") from exc


def extent(rows, case):
    """Report completeness only for the exact ordered schedule or a true prefix."""
    expected = expected_schedule(case)
    levels, times = _case_axes(case)
    rows = list(rows)
    result = []
    for level in levels:
        part = [row for row in rows if isinstance(row, Mapping)
                and type(row.get("substeps")) is int and row["substeps"] == level]
        count = 1 + (len(times) - 1) * level
        observed_times = [row.get("time_s") for row in part]
        result.append({
            "substeps": level, "accepted_rows": len(part), "expected_rows": count,
            "last_time_s": observed_times[-1] if part else None,
            "row_count_complete": len(part) == count,
            "zero_plus_present": bool(part and observed_times[0] == 0.0),
            "reached_declared_end": bool(part and observed_times[-1] == times[-1]),
            "strictly_increasing": all(_finite(value) for value in observed_times)
                and all(a < b for a, b in zip(observed_times, observed_times[1:])),
        })
    unassigned = sum(not isinstance(row, Mapping)
        or type(row.get("substeps")) is not int or row["substeps"] not in levels
        for row in rows)
    seen, mismatch = {}, None
    for index, row in enumerate(rows):
        target = expected[index] if index < len(expected) else None
        observed = ({key: row.get(key) for key in ("substeps", "time_s", "dt_s")}
                    if isinstance(row, Mapping) else row)
        reason = None
        if not isinstance(row, Mapping):
            reason = "invalid_row"
        elif type(row.get("substeps")) is not int or row["substeps"] not in levels:
            reason = "unassigned_substeps"
        elif any(not _finite(row.get(key)) for key in ("time_s", "dt_s")):
            reason = "invalid_time_or_duration"
        else:
            key = _row_key(row)
            if key in seen:
                reason = "duplicate_row"
            elif target is None:
                reason = "unexpected_row"
            elif key != _row_key(target):
                reason = "schedule_mismatch"
            seen[key] = index
        if reason is not None and mismatch is None:
            mismatch = {"row_index": index, "reason": reason,
                        "expected": target, "observed": observed}
    prefix_valid = mismatch is None
    complete = (bool(rows) and prefix_valid and len(rows) == len(expected)
                and unassigned == 0 and all(all(item[key] for key in (
                    "row_count_complete", "zero_plus_present", "reached_declared_end",
                    "strictly_increasing")) for item in result))
    return {"accepted_rows": len(rows), "expected_rows": len(expected),
            "levels": result, "unassigned_rows": unassigned, "complete": complete,
            "prefix_valid": prefix_valid, "first_mismatch": mismatch}
