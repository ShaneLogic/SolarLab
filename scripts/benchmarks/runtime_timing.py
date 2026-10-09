"""Aggregate controller timing and separate outer-invocation clock checks.

RuntimeTiming uses one monotonic nanosecond clock. Inclusive spans overlap;
only exclusive totals form a partition. Its timestamps never enter physical
records, solver options or scientific decisions. Import has no clock call.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import math
from time import perf_counter_ns


PHASES = ("recorded_run", "controller", "ida_calls", "observation",
          "history_io", "artifact_io")


class RuntimeTiming:
    """Six aggregate rows and a stack bounded by those six distinct phases."""

    def __init__(self, *, clock=None):
        self._clock = perf_counter_ns if clock is None else clock
        self._stack = []
        self._totals = {name: {"calls": 0, "exceptions": 0,
                              "inclusive_ns": 0, "exclusive_ns": 0}
                        for name in PHASES}

    @contextmanager
    def phase(self, name):
        if name not in self._totals or any(row[0] == name for row in self._stack):
            raise ValueError("unknown or recursively active timing phase")
        row = self._totals[name]
        frame = [name, self._clock(), 0]
        self._stack.append(frame)
        row["calls"] += 1
        try:
            yield
        except BaseException:
            row["exceptions"] += 1
            raise
        finally:
            elapsed = self._clock() - frame[1]
            self._stack.pop()
            row["inclusive_ns"] += elapsed
            row["exclusive_ns"] += elapsed - frame[2]
            if self._stack:
                self._stack[-1][2] += elapsed

    def snapshot(self, outcome):
        """Return closed aggregate spans; no new clock read or record history."""
        if self._stack:
            raise ValueError("timing snapshot requires closed spans")
        if outcome not in {"returned_completed", "returned_other_status", "raised"}:
            raise ValueError("unknown recorded-call outcome")
        return {
            "schema": "solarlab.native-aggregate-timing.v1",
            "clock": "time.perf_counter_ns", "units": "integer nanoseconds",
            "recorded_call_outcome": outcome,
            "phases": {name: {**row, "entered": row["calls"] != 0}
                       for name, row in self._totals.items()},
            "partition_exclusive_ns": sum(row["exclusive_ns"] for row in self._totals.values()),
            "semantics": {
                "inclusive_ns": "sum of complete spans, including nested measured phases; never add inclusive rows",
                "exclusive_ns": "inclusive time minus immediate nested spans; add only exclusive rows",
                "controller": "controller invocation including mapping construction, admission checks, physical setup and instrumentation overhead",
                "ida_calls": "native constructor/control, init/step with result snapshot, statistics helpers and last-step snapshot; includes callbacks except separately timed nested history I/O",
                "observation": "independent sample, point, domain, reconstruction and charge-evidence calls, including their local preparation",
                "history_io": "emit_record serialization, strict checks, compression, history/sidecar writes, writer creation/finalization/abort",
                "artifact_io": "recorded-run metadata/result and exception publication, including any failure sidecars",
                "zero_rows": "no instrumented span entered; does not prove the corresponding work never occurred",
            },
            "unmeasured": [
                "callbacks are not separately timed; native-call time is not pure solver-core time",
                "interpreter/bootstrap and main preparation before run_recorded, including model construction",
                "timing snapshot serialization/publication itself and later main/OS finalization",
                "independent readback or any work in other processes",
            ],
            "OS_whole_run_timing": "separate existing supervisor/RuntimeEnd receipts",
            "scientific_decisions_use_timing": False,
        }


def reconcile_outer_timing(*, expected_identity, actual_identity, clock_names,
                           outer_started_s, inner_started_s, inner_elapsed_s,
                           observed_interval_s, os_elapsed_s, whole_limit_s,
                           primary_failure=None, prior_failures=(),
                           inner_passed, cleanup_positive, returned_code):
    """Check a closed invocation without reading a clock or replacing failures.

    All three counter sources must be the source-bound, same-host
    time.perf_counter clock: the pre-invocation ToolClock, inner CloseRun, and
    observer. observed_interval_s encloses an actual observer counter reading;
    a new observer passes (now, now). Saved analysis may supply explicitly
    derived rational endpoints, never a fabricated historical timestamp.

    The ToolClock entry must precede RunOnce. The resulting upper bound covers
    RunOnce through this observer checkpoint; bootstrap before that entry and
    later tool-return/final-seal checks remain with the existing outer caller.
    OS 'real' is mandatory separate evidence, not a subtractable clock sample.
    This helper does not award scientific acceptance or alter RuntimeTiming.
    """
    def exact(value):
        if type(value) not in (int, float, Fraction):
            raise ValueError("invalid clock number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite clock number")
        return Fraction(value)

    checks = {
        "invocation_identity": type(expected_identity) is dict and
            type(actual_identity) is dict and bool(expected_identity) and
            all(type(key) is str and bool(key.strip()) and
                type(value) is str and bool(value.strip())
                for identity in (expected_identity, actual_identity)
                for key, value in identity.items()) and
            expected_identity == actual_identity,
        "same_counter_clock": type(clock_names) in (tuple, list) and
            tuple(clock_names) == ("time.perf_counter",) * 3,
        "counter_values": False, "event_order": False,
        "whole_clock_limit": False, "closed_OS_receipt": False,
        "separate_OS_limit": False,
        "inner_pipeline_passed": inner_passed is True,
        "positive_cleanup": cleanup_positive is True,
        "tool_returned_zero": type(returned_code) is int and returned_code == 0,
    }
    upper = None
    exact_upper = None
    checkpoint = None
    counter_error = None
    discrepancy = None
    os_exact = None
    try:
        if type(observed_interval_s) not in (tuple, list) or len(observed_interval_s) != 2:
            raise ValueError("observer reading needs two enclosing endpoints")
        origin, inner, elapsed, low, high, limit = map(exact, (
            outer_started_s, inner_started_s, inner_elapsed_s,
            *observed_interval_s, whole_limit_s))
        checks["counter_values"] = min(origin, inner, elapsed, low, high) >= 0 and low <= high and limit > 0
        # CloseRun stores fl(checkpoint - inner_epoch). Enclose that subtraction's
        # rounding cell; no arbitrary time tolerance or change to the old record.
        if type(inner_elapsed_s) is float:
            elapsed_low = (exact(math.nextafter(inner_elapsed_s, -math.inf)) + elapsed) / 2
            elapsed_high = (elapsed + exact(math.nextafter(inner_elapsed_s, math.inf))) / 2
        else:
            elapsed_low = elapsed_high = elapsed
        checkpoint = [str(inner + elapsed_low), str(inner + elapsed_high)]
        checks["event_order"] = checks["counter_values"] and origin <= inner and inner + elapsed_high <= low
        if checks["counter_values"] and checks["event_order"] and checks["same_counter_clock"]:
            bound = high - origin
            upper = float(bound)
            if exact(upper) < bound:
                upper = math.nextafter(upper, math.inf)
            exact_upper = str(bound)
            checks["whole_clock_limit"] = exact(upper) < limit
    except (ValueError, TypeError, OverflowError) as error:
        counter_error = str(error)
    try:
        os_duration = exact(os_elapsed_s)
        os_exact = str(os_duration)
        checks["closed_OS_receipt"] = os_duration >= 0
        checks["separate_OS_limit"] = checks["closed_OS_receipt"] and 0 < exact(whole_limit_s) and os_duration < exact(whole_limit_s)
        discrepancy = str(exact(inner_elapsed_s) - os_duration)
    except (ValueError, TypeError, OverflowError):
        pass
    failed_checks = [name for name, passed in checks.items() if not passed]
    primary = deepcopy(primary_failure)
    later = deepcopy(list(prior_failures)) + failed_checks
    failures = ([primary] if primary is not None else []) + later
    return {
        "schema": "solarlab.outer-timing-reconciliation.v1",
        "passed": not failures, "first_failure": failures[0] if failures else None,
        "primary_failure": primary, "later_failures": later, "failures": failures,
        "checks": checks, "counter_error": counter_error,
        "elapsed_from_tool_epoch_upper_s": upper,
        "elapsed_from_tool_epoch_upper_exact_s": exact_upper,
        "inner_checkpoint_counter_interval_exact_s": checkpoint,
        "OS_comparison": {
            "elapsed_exact_s": os_exact,
            "inner_elapsed_minus_OS_exact_s": discrepancy,
            "status": "separate_clock_and_scope; no duration nesting assertion",
            "OS_used_in_counter_bound": False,
            "cause_of_disagreement": "not established by these receipts",
        },
        "scope": "closed invocation timing/status only; outer bootstrap and final tool-return/seal checks remain mandatory",
        "scientific_qualification": False,
    }
