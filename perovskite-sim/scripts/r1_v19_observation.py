"""Extend an existing native observer through case finalization without roots."""
from contextlib import contextmanager
import time


class TailObservation:
    """Count callbacks after run_trajectory's receipt, then close exactly once."""

    def __init__(self, observer, summary):
        self.observer = observer
        self.summary = summary
        self.calls = 0
        self.elapsed_s = 0.0
        self.error = None
        self.closed = False
        self._started = {}
        self.phase_elapsed_s = {}

    def event(self, phase, event):
        if self.closed or self.error is not None:
            return
        started = time.monotonic()
        self.calls += 1
        if event == "begin":
            self._started[phase] = started
        try:
            # Physical trajectory owners have left scope. Only the returned
            # report is live here; no earlier root dictionary is retained.
            self.observer(phase, event, {"prepared": None, "raw_result": None,
                "result": self.summary, "rows": None, "persisted": None, "replay": None})
        except Exception as exc:
            self.error = {"type": type(exc).__name__, "message": str(exc)}
        except KeyboardInterrupt as exc:
            self.error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            self.elapsed_s += time.monotonic() - started
            if event != "begin" and phase in self._started:
                self.phase_elapsed_s[phase] = time.monotonic() - self._started.pop(phase)

    @contextmanager
    def phase(self, name):
        self.event(name, "begin")
        try:
            yield
        except BaseException:
            try:
                self.event(name, "error")
            except BaseException:
                pass  # The scientific or I/O exception remains primary.
            raise
        else:
            self.event(name, "end")

    def finish(self):
        if self.closed:
            return
        base = dict(self.summary.get("memory_observation", {}))
        interrupted = None
        try:
            self.observer.close()
        except BaseException as exc:
            if self.error is None:
                self.error = {"type": type(exc).__name__, "message": str(exc)}
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            self.closed = True
            self.summary["memory_collector"] = self.observer.summary()
            error = base.get("error") or self.error
            self.summary["memory_observation"] = {
                **base, "enabled": True,
                "passed": base.get("passed") is True and error is None,
                "call_count": base.get("call_count", 0) + self.calls,
                "elapsed_s": base.get("elapsed_s", 0.0) + self.elapsed_s,
                "main_elapsed_s": base.get("main_elapsed_s", 0.0), "error": error,
                "cost_scope": "observer_overhead_is_included_in_measured_costs_never_subtracted",
                "extension_scope": "analysis_callbacks_and_tail; final_metadata_reseal_and_stdout_after_close_use_external_sampling",
                "tail_phase_elapsed_s": dict(self.phase_elapsed_s),
            }
        if interrupted is not None:
            raise interrupted
