"""Fixed-size diagnostic timing for one serial recorded controller call.

One monotonic clock supplies integer nanoseconds. Inclusive spans overlap by
design; only exclusive totals form a partition. No timestamp enters a request,
physical record, solver option or acceptance decision. Import has no clock call.
"""
from __future__ import annotations

from contextlib import contextmanager
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
