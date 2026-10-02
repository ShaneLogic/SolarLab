"""Observe the existing numeric sidecar path without changing its format."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_codec as codec
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol


@contextmanager
def storage_phase(phase_observer, phase, raw_result):
    """Pass only a live local root; the runner fills its other declared roots."""
    def emit(event):
        if phase_observer is not None:
            phase_observer(phase, event, {"raw_result": raw_result})

    emit("begin")
    try:
        yield
    except BaseException:
        try:
            emit("error")
        except BaseException:
            pass
        raise
    else:
        emit("end")


def persist_numeric_observed(path, result, *, phase_observer=None):
    """Keep the original open/collect/save order and nonfinite failure route.

    Payload and final array mappings are released at their original consumer
    boundaries. Observers retain neither mapping and do not evaluate physics.
    This sidecar retains its original direct ``wb`` write behavior.
    """
    nonfinite = bool(protocol.nonfinite_numeric_paths(result))
    with Path(path).open("wb") as stream:
        with storage_phase(phase_observer, "numeric_payload", result):
            payload = result if nonfinite else codec.state_sidecar_payload(result)
        with storage_phase(phase_observer, "numeric_array_collection", result):
            try:
                arrays = codec.numeric_arrays(payload, allow_nonfinite=True) if nonfinite else codec.numeric_arrays(payload)
            finally:
                payload = None
        with storage_phase(phase_observer, "numeric_archive_write", result):
            try:
                np.savez_compressed(stream, **arrays)
            finally:
                arrays = None
