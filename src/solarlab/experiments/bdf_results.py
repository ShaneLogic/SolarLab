"""Captured BDF publication and historical readback.

Solver imports are confined to computation; both readers use recorded data only.
No method, gate, control or qualification status is inferred from current code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from solarlab.experiments.results import decode_array
from solarlab.io.artifacts import bytes_artifact
from solarlab.reproducibility.job_provenance import (
    JobProvenance, job_provenance, publication_binding, read_provenance, result_provenance,
)

if TYPE_CHECKING:
    from solarlab.io.run_store import AttemptToken, RunStore
    from solarlab.solvers.time.bdf import BDFOptions, BDFResult
    from solarlab.solvers.time.bdf_1d import BDFDrive

__all__ = ["CapturedBDFRun", "run_bdf_for_publication", "publish_bdf_result",
           "read_bdf_result", "read_bdf_packet"]

_SCHEMA = "solarlab.bdf-publication.v1"
_HISTORY_SCHEMA = "solarlab.physical_storage_bdf12.v1"
_METHOD = "classical_physical_storage_bdf12"
_LABEL = "experimental_unqualified"
_TAGS = {"$array", "$scalar", "$float", "$tuple"}


def _canonical(value: Any) -> bytes:
    # Artifact content, unlike RunStore metadata, may exceed the 1 MiB inline
    # document limit. Large histories remain one integrity-checked byte artifact.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _loads(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        _require(len(dict(items)) == len(items), "duplicate BDF record key")
        return dict(items)

    def constant(value: str) -> Any:
        raise ValueError("nonfinite BDF JSON number must retain explicit words: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _fingerprint(value: np.ndarray) -> dict[str, Any]:
    return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _sha(value.tobytes(order="C"))}


def _same(left: Any, right: Any) -> bool:
    return type(left) is np.ndarray and type(right) is np.ndarray and _fingerprint(left) == _fingerprint(right)


def _array(value: np.ndarray) -> dict[str, Any]:
    _require(not value.dtype.hasobject and value.dtype.fields is None and value.dtype.kind in "biufc",
             "BDF arrays require plain numerical words")
    return {**_fingerprint(value), "words": value.tobytes(order="C").hex()}


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if type(value) is np.ndarray:
        return {"$array": _array(value)}
    if isinstance(value, np.generic):
        return {"$scalar": _array(np.asarray(value))}
    if isinstance(value, Enum):
        return _encode(value.value)
    if type(value) is float:
        # Preserve every binary64 word, including signed zero and failed NaNs.
        return {"$float": _array(np.asarray(value))}
    if type(value) is tuple:
        return {"$tuple": [_encode(item) for item in value]}
    if type(value) is list:
        return [_encode(item) for item in value]
    if type(value) is dict:
        _require(all(type(key) is str for key in value) and not set(value) & _TAGS,
                 "BDF record keys conflict with the numerical codec")
        return {key: _encode(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError("unsupported BDF record value: " + type(value).__name__)


def _decode(value: Any) -> Any:
    if type(value) is list:
        return [_decode(item) for item in value]
    if type(value) is not dict:
        return value
    tags = set(value) & _TAGS
    if not tags:
        return {key: _decode(item) for key, item in value.items()}
    _require(len(value) == 1, "ambiguous BDF numerical tag")
    tag, = tags
    item = value[tag]
    if tag == "$tuple":
        _require(type(item) is list, "invalid BDF tuple")
        return tuple(_decode(part) for part in item)
    if item.get("schema") == "solarlab.numeric-array.v1":
        # Exact ReadinessV1/V2 codec, not construction of historical solver classes.
        array = decode_array(item)
    else:
        _require(set(item) == {"dtype", "shape", "sha256", "words"}, "invalid BDF array fields")
        dtype = np.dtype(item["dtype"])
        shape = item["shape"]
        _require(dtype.str == item["dtype"] and not dtype.hasobject and dtype.fields is None
                 and dtype.kind in "biufc" and type(shape) is list and len(shape) <= 32
                 and all(type(n) is int and n >= 0 for n in shape), "invalid BDF array dtype/shape")
        raw = bytes.fromhex(item["words"])
        _require(len(raw) == math.prod(shape) * dtype.itemsize and _sha(raw) == item["sha256"],
                 "BDF array words or hash differ")
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    array.setflags(write=False)
    if tag == "$array":
        return array
    _require(array.shape == (), "BDF scalar requires scalar shape")
    return array.item() if tag == "$float" else array[()]


def _digest(value: Any) -> str:
    return _sha(_canonical(_encode(value)))


def _vector(value: Any, width: int | None = None, *, finite: bool = True) -> np.ndarray:
    _require(type(value) is np.ndarray and value.ndim == 1 and value.dtype.kind == "f",
             "BDF history requires real floating vectors")
    _require(width is None or value.size == width, "BDF coordinate width differs")
    _require(not finite or bool(np.all(np.isfinite(value))), "nonfinite accepted BDF vector")
    return value


def _work(history: dict[str, Any]) -> dict[str, Any]:
    attempts, samples = history["attempts"], history["samples"]
    main = [step["newton"] for step in attempts if step["newton"] is not None]
    auxiliary = [result for step in attempts for result in step["auxiliary_newton"]]
    sample = [item["projection"] for item in samples if item.get("projection") is not None]
    failure = history.get("sample_failure")
    if failure is not None and failure.get("newton") is not None:
        sample.append(failure["newton"])
    return {
        "accepted_attempts": sum(step["accepted"] for step in attempts),
        "rejected_attempts": sum(not step["accepted"] for step in attempts),
        "main_newton_records": len(main), "auxiliary_newton_records": len(auxiliary),
        "sample_newton_records": len(sample), "optional_samples": len(samples),
        "main_newton_failures": sum(not item["accepted"] for item in main),
        "auxiliary_newton_failures": sum(not item["accepted"] for item in auxiliary),
        "sample_newton_failures": sum(not item["accepted"] for item in sample),
        "nonlinear_calls": history["nonlinear_calls"], "evaluation_counts": history["evaluation_counts"],
        "wall_s": history["wall_s"], "restart_records": len(history["restarts"]),
        "initial_admission_wall_s": history["controls"].get("initial_admission_wall_s"),
    }


def _checks_pass(checks: Any) -> bool:
    _require(isinstance(checks, (tuple, list))
             and all(isinstance(check["passed"], (bool, np.bool_)) for check in checks),
             "BDF gate receipts must have explicit Boolean outcomes")
    # Read the recorded decisions; never re-evaluate a physical gate or limit.
    return all(check["passed"] for check in checks)


def _validate_newton(newton: dict[str, Any], width: int) -> None:
    _require(type(newton["accepted"]) is bool
             and newton["accepted"] == (newton["termination"] == "converged"),
             "BDF Newton completion contradicts its termination")
    passed = _checks_pass(newton["gates"])
    _require(not newton["accepted"] or passed, "accepted BDF Newton record contains a rejected gate")
    _vector(newton["state"], width, finite=newton["accepted"])
    _require(all(type(newton[key]) is int and newton[key] >= 0
                 for key in ("residual_evaluations", "jacobian_evaluations")),
             "invalid BDF Newton work counters")


def _validate_work(history: dict[str, Any]) -> None:
    attempts, samples = history["attempts"], history["samples"]
    integration = [record for step in attempts for record in (step["newton"], *step["auxiliary_newton"])
                   if record is not None]
    projections = [sample["projection"] for sample in samples if sample.get("projection") is not None]
    failure = history.get("sample_failure")
    if failure is not None and failure.get("newton") is not None:
        projections.append(failure["newton"])
    counts, calls = history["evaluation_counts"], history["nonlinear_calls"]
    required = {"residual", "jacobian", "storage", "assessment", "sample_projection",
                "sample_residual", "sample_jacobian"}
    _require(set(counts) == required and type(calls) is int and calls >= 0
             and all(type(n) is int and n >= 0 for n in counts.values()), "invalid BDF work counters")
    # Calls increment before Newton returns. A terminal exception can therefore
    # leave one call without a result, but cannot invent an arbitrary work count.
    integration_gaps = sum(step["termination"] in {"evaluation_error", "wall_budget_exhausted"}
                           for step in attempts)
    projection_gaps = int(failure is not None and failure.get("newton") is None)
    sampled_calls = counts["sample_projection"]
    _require(len(projections) <= sampled_calls <= len(projections) + projection_gaps
             and len(integration) <= calls - sampled_calls <= len(integration) + integration_gaps,
             "BDF nonlinear call counters contradict the saved histories")
    for kind in ("residual", "jacobian"):
        for records, observed, missing in (
            (projections, counts["sample_" + kind], sampled_calls - len(projections)),
            (integration, counts[kind] - counts["sample_" + kind], calls - sampled_calls - len(integration)),
        ):
            recorded = sum(record[kind + "_evaluations"] for record in records)
            _require(observed >= recorded and (missing > 0 or observed == recorded),
                     "BDF evaluation counters contradict the saved Newton work")
    assessed = sum(step["assessment"] is not None for step in attempts)
    successful = sum(step["newton"] is not None and step["newton"]["accepted"] for step in attempts)
    minimum_storage = (1 + sum(step["accepted"] for step in attempts) + len(history["restarts"])
                       + sum(not sample["accepted_step_endpoint"] for sample in samples))
    _require(assessed <= counts["assessment"] <= successful and counts["storage"] >= minimum_storage,
             "BDF assessment/storage counters omit saved work")
    _require(math.isfinite(history["wall_s"]) and history["wall_s"] >= 0
             and all(math.isfinite(step["wall_s"]) and step["wall_s"] >= 0 for step in attempts),
             "invalid BDF elapsed work")


def _validate_history(history: dict[str, Any]) -> dict[str, Any]:
    """Check saved relationships, without evaluating an equation or a gate."""
    _require(history["schema"] == _HISTORY_SCHEMA
             and history["physical_certificate_status"] == "not_evaluated", "not an unqualified BDF history")
    _require(history["controls"].get("method", _METHOD) == _METHOD, "mixed BDF/effective method record")
    completed, termination = history["completed"], history["termination"]
    _require(type(completed) is bool and completed == (termination == "completed"),
             "BDF completion and termination differ")
    start, end = history["initial_time_s"], history["final_time_s"]
    _require(math.isfinite(start) and math.isfinite(end) and start < end, "invalid requested BDF interval")
    initial = _vector(history["initial_u"])
    times = _vector(history["accepted_times_s"])
    states = history["accepted_coordinates"]
    _require(bool(times.size > 0 and times[0] == start and np.all(np.diff(times) > 0)
             and times[-1] <= end and len(states) == times.size), "BDF accepted prefix has invalid clocks or length")
    for state in states:
        _vector(state, initial.size)
    restarts = {item["time_s"]: item for item in history["restarts"]}
    _require(len(restarts) == len(history["restarts"]), "duplicate BDF restart clocks")
    initial_restart = restarts.get(start)
    initial_matches = _same(initial, states[0]) or (initial_restart is not None
        and initial_restart["accepted"] is True and _same(initial, initial_restart["left_u"])
        and _same(states[0], initial_restart["right_u"]))
    _require(initial_matches, "BDF initial state differs from accepted prefix/restart sides")
    position = 0
    accepted_trials = {start: initial}
    attempts = history["attempts"]
    consecutive = 0
    for index, step in enumerate(attempts):
        _require(type(step["accepted"]) is bool and type(step["order"]) is int and step["order"] in (1, 2)
                 and step["accepted"] == (step["termination"] == "accepted"), "invalid BDF attempt/order/termination")
        _require(step["start_s"] == times[position] and step["start_s"] < step["end_s"] <= end,
                 "BDF trial clock differs from its last accepted state")
        _vector(step["coefficients"], step["order"] + 1)
        _vector(step["initial_coordinates"], initial.size, finite=False)
        _vector(step["candidate_coordinates"], initial.size, finite=False)
        if step["accepted"]:
            position += 1
            _require(position < len(times) and step["end_s"] == times[position],
                     "BDF accepted clock differs from its accepted trial")
            accepted_trials[step["end_s"]] = step["candidate_coordinates"]
            restart = restarts.get(step["end_s"])
            if restart is None:
                matches = _same(step["candidate_coordinates"], states[position])
            else:
                # V1 retains the left integration node; some saved V2 histories
                # retain the accepted restart's right node instead. Both words
                # must be present: never reconstruct or overwrite either side.
                matches = _same(step["candidate_coordinates"], restart["left_u"]) and (
                    _same(states[position], restart["left_u"]) or
                    restart["accepted"] is True and _same(states[position], restart["right_u"]))
            _require(matches, "BDF accepted state differs from its accepted trial/restart sides")
            _require(step["termination"] == "accepted" and step["newton"] is not None
                     and step["newton"]["accepted"] is True, "BDF accepted trial has no successful Newton record")
            _require(_same(step["candidate_coordinates"], step["newton"]["state"])
                     and step["assessment"] is not None and _checks_pass(step["assessment"]["checks"])
                     and all(item["accepted"] for item in step["auxiliary_newton"]),
                     "accepted BDF trial contradicts its Newton/assessment evidence")
            consecutive = 0
        else:
            consecutive += 1
            terminal = (step["termination"] in {"physical_gate_rejected", "evaluation_error", "wall_budget_exhausted"}
                        or not history["controls"]["adaptive"])
            if terminal:
                _require(index == len(attempts) - 1 and not completed and termination == step["termination"],
                         "BDF history continues after a terminal rejected trial")
            elif consecutive > history["controls"]["max_consecutive_rejections"]:
                _require(index == len(attempts) - 1 and termination == "rejection_budget_exhausted",
                         "BDF history continues after exhausting its rejection budget")
        for newton in (step["newton"], *step["auxiliary_newton"]):
            if newton is not None:
                _validate_newton(newton, initial.size)
    _require(position + 1 == len(times), "BDF prefix contains an invented accepted endpoint")
    restart_times = []
    for restart in history["restarts"]:
        clock = restart["time_s"]
        indices = np.flatnonzero(times == clock)
        _require(indices.size == 1 and clock not in restart_times, "BDF restart is outside accepted progress")
        restart_times.append(clock)
        _require(_same(restart["left_u"], accepted_trials[clock]), "BDF restart lost its left state")
        for name in ("left_u", "right_u", "left_storage", "right_storage"):
            _vector(restart[name], initial.size)
        _require(type(restart["accepted"]) is bool, "BDF restart acceptance must be explicit")
        _require(restart["accepted"] == _checks_pass(restart["assessment"]["checks"]),
                 "BDF restart acceptance contradicts its recorded checks")
        # In the captured electronic/single-ion adapters S is a function of
        # the represented state. Identical sides cannot name different S words.
        if _same(restart["left_u"], restart["right_u"]):
            _require(_same(restart["left_storage"], restart["right_storage"]),
                     "BDF identical restart states have inconsistent left/right storage")
        if not restart["accepted"]:
            # The initial-event path records evaluation_error; later rejections
            # record restart_physical_gate_rejected. Neither path starts a trial
            # from the rejected right side or publishes optional output samples.
            expected = "evaluation_error" if clock == start else "restart_physical_gate_rejected"
            _require(not completed and clock == times[-1] and termination == expected
                     and all(step["start_s"] < clock for step in attempts) and not history["samples"],
                     "rejected BDF restart cannot complete or continue accepted progress")
    _require(all(left < right for left, right in zip(restart_times, restart_times[1:])),
             "BDF restart history is out of order")
    rejected_restarts = [item for item in history["restarts"] if not item["accepted"]]
    _require(termination != "restart_physical_gate_rejected" or bool(rejected_restarts),
             "BDF restart termination lacks its rejected restart")
    requested = _vector(history.get("requested_output_times_s"))
    _require(bool(np.all(np.diff(requested) >= 0) and np.all(requested >= start) and np.all(requested <= end)),
             "invalid requested BDF sample clocks")
    observed_samples: dict[float, dict[str, Any]] = {}
    for index, sample in enumerate(history["samples"]):
        _require(requested is not None and index < requested.size and sample["time_s"] == requested[index],
                 "BDF optional sample differs from the requested sample prefix")
        _require(sample["time_s"] <= times[-1], "BDF sample lies after accepted progress")
        _vector(sample["u"], initial.size)
        _vector(sample["storage_value"], initial.size)
        _require(type(sample["accepted_step_endpoint"]) is bool, "missing BDF sample role")
        clock = sample["time_s"]
        _require(sample["accepted_step_endpoint"] == bool(np.any(times == clock)),
                 "BDF sample role contradicts the accepted clocks")
        if sample["accepted_step_endpoint"]:
            restart = restarts.get(clock)
            state = states[int(np.flatnonzero(times == clock)[0])] if restart is None else restart["right_u"]
            _require(_same(sample["u"], state) and sample["interpolant_order"] == 0
                     and sample.get("projection") is None, "BDF endpoint sample has invented state or work")
            if restart is not None:
                _require(restart["accepted"] and _same(sample["storage_value"], restart["right_storage"]),
                         "BDF restart sample lost its right-side state/storage")
        else:
            _require(sample.get("projection") is not None and sample["projection"]["accepted"] is True,
                     "BDF interpolated sample lacks its accepted projection")
            _validate_newton(sample["projection"], initial.size)
            _require(_same(sample["u"], sample["projection"]["state"])
                     and sample["interpolant_order"] in (1, 2), "BDF sample contradicts its saved projection")
        previous = observed_samples.get(clock)
        if previous is not None:
            _require(_same(sample["u"], previous["u"]) and _same(sample["storage_value"], previous["storage_value"]),
                     "duplicate BDF sample clocks contain inconsistent state/storage")
        observed_samples[clock] = sample
    if completed:
        _require(times[-1] == end and requested is not None and len(history["samples"]) == requested.size
                 and history.get("sample_failure") is None and not rejected_restarts,
                 "completed BDF record did not finish its request/restarts")
    failure = history.get("sample_failure")
    _require((failure is not None) == (termination == "sample_projection_failed"),
             "BDF sample termination lacks its failure record")
    _require(not history["samples"] or (times[-1] == end and (completed or failure is not None)),
             "BDF optional samples precede completed integration")
    if failure is not None:
        _require(not completed and times[-1] == end and not rejected_restarts
                 and type(failure["accepted_sample_prefix"]) is int
                 and failure["accepted_sample_prefix"] == len(history["samples"])
                 and requested is not None and len(history["samples"]) < requested.size
                 and failure["time_s"] == requested[len(history["samples"])], "BDF sample failure lost its requested clock")
        if failure.get("newton") is not None:
            _validate_newton(failure["newton"], initial.size)
    controls = history["controls"]
    _require(len(times) - 1 <= controls["max_steps"] and len(attempts) <= controls["max_attempts"],
             "BDF history exceeds its saved step/attempt caps")
    if termination == "step_budget_exhausted":
        _require(times[-1] < end and (len(times) - 1 == controls["max_steps"] or len(attempts) == controls["max_attempts"]),
                 "BDF step-budget termination contradicts its accepted progress/work")
    if termination == "rejection_budget_exhausted":
        _require(times[-1] < end and controls["adaptive"] and consecutive > controls["max_consecutive_rejections"],
                 "BDF rejection-budget termination contradicts its saved trials")
    _validate_work(history)
    return {"schema": _SCHEMA, "requested_method": "bdf", "effective_method": _METHOD,
            "qualification": False, "label": _LABEL, "completed": completed,
            "termination": history["termination"], "requested_end_s": end,
            "last_accepted_time_s": float(times[-1]), "requested_end_reached": completed,
            "accepted_count": len(times), "sample_count": len(history["samples"]), "work": _work(history)}


def _input_record(request: dict[str, Any]) -> dict[str, Any]:
    return {"requested_method": request["method"], "request_sha256": _digest(request),
            "device_sha256": request["device_sha256"], "initial_sha256": request["initial_sha256"]}


def _requested_end_state(history: dict[str, Any]) -> np.ndarray | None:
    """Return only a saved completed endpoint, including an accepted right side.

    The accepted integration node is retained separately. A failed sample or
    restart never supplies a requested-end result, even if integration reached
    that clock; samples and rejected candidates cannot manufacture an endpoint.
    """
    if not history["completed"]:
        return None
    for restart in history["restarts"]:
        if restart["accepted"] and restart["time_s"] == history["final_time_s"]:
            return restart["right_u"]
    return history["accepted_coordinates"][-1]


def _validate_body(body: dict[str, Any]) -> dict[str, Any]:
    _require(body["schema"] == _SCHEMA and body["qualification"] is False and body["label"] == _LABEL,
             "BDF publication cannot claim qualification or another method")
    request, history = body["request"], body["history"]
    _require(request["method"] == "bdf" and body["effective_method"] == _METHOD, "mixed requested/effective BDF methods")
    summary = _validate_history(history)
    _require(body["summary"] == summary, "BDF publication summary differs from its saved history")
    controls = history["controls"]
    effective = body["effective_bdf_options"]
    _require(_digest(effective) == _digest({key: controls[key] for key in request["bdf_options"]}),
             "BDF effective options differ from history")
    _require(_digest(controls["physical_controls"]) == _digest(request["physical_controls"])
             and controls["compiled_identity"] == request["device_sha256"], "BDF physical/device controls differ from request")
    _require(_digest(controls["drive"]) == _digest(request["drive"]), "BDF drive sides differ from the admitted request")
    policy = request["event_restart_policy"]
    events = {clock: policy["jump_restart"] for clock in policy["jump_times_s"]}
    events.update({clock: policy["slope_restart"] for clock in policy["slope_breakpoints_s"]})
    actual = {restart["time_s"]: restart["name"] for restart in history["restarts"]}
    _require(all(events.get(clock) == name for clock, name in actual.items())
             and all(clock in actual for clock in events
                     if summary["completed"] or clock < summary["last_accepted_time_s"]),
             "BDF restart history contradicts the saved event policy")
    _require(_same(history["requested_output_times_s"], request["protocol"]["times_s"])
             and history["initial_time_s"] == request["protocol"]["times_s"][0]
             and history["final_time_s"] == request["protocol"]["times_s"][-1], "BDF requested/sample clocks differ")
    _require(_digest(request["initial"]) == request["initial_sha256"]
             and _same(request["initial"]["state"]["u"], history["initial_u"]), "BDF initial object differs from history")
    if request["initial_source_sha256"] is None:
        _require(_digest(controls["initial_controls"]) == _digest(request["initial_controls"]),
                 "BDF initial controls differ from admission")
    else:
        _require(controls["initial_source_sha256"] == request["initial_source_sha256"]
                 == request["initial"]["source_sha256"]
                 == _sha(request["initial"]["evidence_json"].encode()), "BDF initial evidence hash differs")
        _require(_digest(request["initial_controls"]) == _digest(request["initial"]["steady_controls"]),
                 "BDF ionic initial controls differ from admission")
    _require(body["remaining_wall_budget_s"] == controls["wall_s"]
             and body["initial_admission_wall_s"] == controls["initial_admission_wall_s"], "BDF wall budget accounting differs")
    _require(body["caps"] == {"requested_max_steps": request["bdf_options"]["max_steps"],
             "physical_max_steps": request["physical_controls"].get("electronic", request["physical_controls"])["max_steps"],
             "effective_max_steps": controls["max_steps"], "effective_max_attempts": controls["max_attempts"]},
             "BDF clipped caps differ from the saved controls")
    return summary


def _capture_identity(body: dict[str, Any], provenance: dict[str, Any]) -> dict[str, str]:
    description, request = provenance["description"], body["request"]
    _require(provenance["status"] == "captured_at_computation" and description["scope"] == "bdf_computation_call"
             and description["equation_implementation"] == request["equation_implementation"],
             "BDF equation implementation differs from its capture")
    admitted = read_provenance(description["job_provenance"])
    value = admitted.export()
    _require(value["assembler"]["operator"] == "BDFStep:" + request["equation_implementation"]["problem"].rsplit(".", 1)[1]
             and value["assembler"]["unified_assembler"] == request["equation_implementation"]["unified_assembler"],
             "BDF declared equation binding differs from admission")
    implementation = request["equation_implementation"]
    _require(implementation["kind"] == "original" and implementation["unified_assembler"] is False,
             "BDF currently supports original equations only")
    if "provenance_binding" in implementation:
        binding = implementation["provenance_binding"]
        _require(type(binding) is dict and set(binding) == {"operator", "assembler_identity", "analysis_identity"},
                 "BDF saved equation binding is incomplete")
        expected_binding = {"operator": binding["operator"], "identity": binding["assembler_identity"],
                            "analysis_identity": binding["analysis_identity"], "unified_assembler": False}
    else:
        # Pre-wiring captures used Q6's original operator-only analysis identity.
        # Verify that historical form from saved fields; never import today's helper.
        operator = "BDFStep:" + implementation["problem"].rsplit(".", 1)[1]
        expected_binding = {"operator": operator, "identity": None, "unified_assembler": False,
                            "analysis_identity": _sha(_canonical({"operator": operator}))}
    _require(value["assembler"] == expected_binding and expected_binding["identity"] is None,
             "BDF full saved equation binding differs from admission")
    hashes = {"source_tree_sha256": value["source_tree_sha256"],
              "runtime_sha256": _sha(_canonical(value["runtime_observed"])),
              "device_sha256": request["device_sha256"], "initial_sha256": request["initial_sha256"]}
    _require(body["hashes"] == hashes and description["runtime"] == value["runtime_observed"]
             and description["input_record"] == _input_record(request), "BDF source/runtime/input binding differs")
    expected = {"H_input": _sha(_canonical(_input_record(request))), "H_physics": request["device_sha256"],
                "H_execution": _sha(_canonical(description))}
    _require(provenance["identity"] == expected, "BDF computation capture differs from inputs/results")
    result_provenance(description, expected_inputs=_input_record(request))
    return expected


def read_bdf_result(store: RunStore, run_id: str, *, attempt_id: str | None = None,
                    current_identity: JobProvenance | dict[str, Any] | None = None,
                    strict_same_source: bool = False) -> dict[str, Any]:
    """Read a committed BDF publication without defaults, solvers or source recapture."""
    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    _require(attempt is not None and attempt["terminal"] is not None
             and attempt["state"] in {"succeeded", "failed", "cancelled"}, "a committed terminal BDF attempt is required")
    assert attempt is not None
    document = attempt["result"]
    _require(type(document) is dict and document.get("schema") == _SCHEMA, "attempt does not contain a BDF publication")
    owner = {"store_id": store.store_id, "run_id": run_id, "attempt_id": selected,
             "owner_id": attempt["owner_id"], "generation": attempt["generation"]}
    _require(document["owner"] == owner, "BDF publication belongs to another attempt")
    references = {r["artifact_id"]: r for r in attempt["terminal"]["artifacts"]}
    _require(len(references) == 2 and len(attempt["terminal"]["artifacts"]) == 2
             and set(document["artifact_ids"]) == set(references) and len(document["artifact_ids"]) == 2,
             "BDF terminal lacks its exact artifact set")

    def content(identifier: str, name: str) -> bytes:
        item = store.artifact(identifier)
        _require(identifier in references and item["name"] == name and item["phase"] == "committed"
                 and item["format"] == "bytes" and item["metadata"] == {"schema": _SCHEMA, "role": name}
                 and all(item[key] == owner[key] for key in ("run_id", "attempt_id", "generation"))
                 and item["manifest_sha256"] == references[identifier]["manifest_sha256"], "BDF artifact role/owner/manifest differs")
        return store.read_artifact(identifier)

    raw_manifest = content(document["manifest_artifact_id"], "bdf_manifest")
    _require(_sha(raw_manifest) == document["manifest_sha256"], "BDF manifest hash differs")
    manifest = _loads(raw_manifest)
    _require(manifest["schema"] == _SCHEMA and manifest["owner"] == owner
             and manifest["identity"] == run["request"]["identity"], "BDF manifest identity differs")
    raw = content(manifest["history_artifact_id"], "bdf_history")
    _require(_sha(raw) == manifest["history_sha256"], "BDF history content hash differs")
    body = _decode(_loads(raw))
    summary = _validate_body(body)
    _require(document["summary"] == summary and manifest["summary"] == summary, "BDF result and terminal summary differ")
    _require(summary["completed"] == (attempt["state"] == "succeeded"),
             "BDF completion and terminal outcome differ")
    provenance = manifest["provenance"]
    description = provenance["description"]
    request = body["request"]
    expected_identity = _capture_identity(body, provenance)
    _require(expected_identity == manifest["identity"]
             and provenance["history_sha256"] == _sha(raw), "BDF computation capture differs from inputs/results")
    _require(manifest["publication_binding"] == publication_binding(description["job_provenance"], owner=owner,
             request_identity=expected_identity, result_sha256=_sha(raw)), "BDF producer/publication binding differs")
    comparison = result_provenance(description, current_identity, strict_same_source=strict_same_source,
                                   expected_inputs=_input_record(request))
    return {"document": document, "manifest": manifest, "record": body, "history": body["history"],
            "summary": summary, "run_state": attempt["state"], "provenance_comparison": comparison,
            "last_accepted_state": body["history"]["accepted_coordinates"][-1],
            "requested_end_state": _requested_end_state(body["history"])}


def read_bdf_packet(path: str | Path, sha256: str, *, strict_same_source: bool = False) -> dict[str, Any]:
    """Read exact ReadinessV1/V2 packets, retaining unavailable historical provenance.

    The supplied digest binds the file bytes (including gzip when used). No old
    packet is rewritten or promoted into a contemporaneously captured publication.
    """
    raw = Path(path).read_bytes()
    _require(_sha(raw) == sha256, "historical BDF packet hash differs")
    content = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    encoded = _loads(content)
    _require(encoded.get("method") == "bdf" and encoded.get("qualification") is False
             and type(encoded.get("result")) is dict, "packet is not an unqualified BDF history")
    packet = _decode(encoded)
    summary = _validate_history(packet["result"])
    _require(packet["completed"] == summary["completed"] and packet["termination"] == summary["termination"],
             "historical BDF packet contradicts its raw history")
    _require(packet["compiled_identity"] == packet["result"]["controls"]["compiled_identity"]
             and _same(packet["requested"]["times_s"], packet["result"]["requested_output_times_s"]),
             "historical BDF packet input/clock binding differs")
    _require(not strict_same_source, "strict_same_source requires complete historical source/runtime admission evidence")
    return {"packet": packet, "history": packet["result"], "summary": summary,
            "file_sha256": sha256, "original_packet_sha256": _sha(content),
            "requested_bdf_options": packet["BDF_options"], "effective_controls": packet["result"]["controls"],
            "source_pins": packet["source_pins"], "runtime_identity": None,
            "provenance_status": "historical_packet; no admission/runtime identity inferred",
            "last_accepted_state": packet["result"]["accepted_coordinates"][-1],
            "requested_end_state": _requested_end_state(packet["result"])}


@dataclass(frozen=True, slots=True)
class CapturedBDFRun:
    result: BDFResult
    nodes_m: np.ndarray
    history_bytes: bytes
    capture_json: str

    @property
    def identity(self) -> dict[str, str]:
        return dict(_loads(self.capture_json.encode())["identity"])


def run_bdf_for_publication(
    compiled: Any, initial: Any, protocol: Any, *, options: Any,
    bdf_options: BDFOptions, initial_options: Any = None, bdf_drive: BDFDrive | None = None,
) -> CapturedBDFRun:
    """Capture an actual electronic or admitted single-ion BDF call.

    No default is flipped and no fallback is attempted. Admission refusals that
    produce no BDF history still raise; returned failed histories are retained.
    Initial/DC history is represented by its supplied bytes, not re-attributed.
    """
    from solarlab.equations.dc_1d import CompiledDevice1D
    from solarlab.equations.implementation import equation_implementation, provenance_binding, selected_implementation
    from solarlab.equations.ionic_1d import CompiledIonicDevice1D
    from solarlab.experiments.transient import solve_transient
    from solarlab.experiments.ionic_transient import solve_ionic_transient
    from solarlab.solvers.time.bdf import BDFOptions, BDFResult
    from solarlab.solvers.time.bdf_1d import BDFDrive

    _require(type(bdf_options) is BDFOptions, "BDF publication requires explicit BDFOptions")
    ionic = isinstance(compiled, CompiledIonicDevice1D)
    _require(ionic or isinstance(compiled, CompiledDevice1D), "BDF publication requires a supported compiled device")
    _require(ionic or initial_options is not None, "electronic BDF publication requires initial controls")
    operator = "Ionic1DProblem" if ionic else "DC1DProblem"
    selection = selected_implementation(operator="BDFStep:" + operator)
    binding = provenance_binding("BDFStep:" + operator, compiled, assembled=False)
    drive = BDFDrive.from_protocol(protocol) if bdf_drive is None else bdf_drive
    _require(type(drive) is BDFDrive, "BDF publication requires the actual drive")
    nodes = np.frombuffer(compiled.nodes_m.tobytes(), dtype=compiled.nodes_m.dtype).reshape(compiled.nodes_m.shape)
    implementation = {"problem": "solarlab.equations." + ("ionic_1d." if ionic else "dc_1d.") + operator,
        "time_adapter": "solarlab.solvers.time.bdf_1d." + ("solve_ionic_bdf" if ionic else "solve_electronic_bdf"),
        "kind": "original", "unified_assembler": False, "provenance_binding": binding}
    request = _decode(_encode({
        "method": "bdf", "bdf_options": asdict(bdf_options), "physical_controls": asdict(options),
        "initial_controls": initial.steady_controls if ionic else asdict(initial_options),
        "protocol": asdict(protocol), "drive": asdict(drive), "drive_was_explicit": bdf_drive is not None,
        "nodes_m": nodes, "initial": initial, "initial_sha256": _digest(initial),
        "initial_source_sha256": getattr(initial, "source_sha256", None), "device_sha256": compiled.identity,
        "equation_implementation": implementation,
        "event_restart_policy": {
            "interpolation": "right[i] to left[i+1] between drive knots",
            "breakpoints_s": tuple(float(t) for t in drive.times_s[1:-1]),
            "jump_times_s": tuple(float(t) for t, left, right in zip(drive.times_s, drive.left_values, drive.right_values)
                                  if not np.array_equal(left, right)),
            "slope_breakpoints_s": drive.slope_breakpoints_s,
            "jump_restart": "voltage_generation_protocol_jump",
            "slope_restart": "drive_slope_discontinuity", "restart_order": 1,
            "inventory": "retain physical inventory; check algebraic rows",
            "output_samples": "optional post-integration projections; never accepted trajectory nodes",
        },
    }))
    inputs = _input_record(request)
    started = datetime.now(timezone.utc).isoformat()
    # BDF's declared original-only capability includes its initial admission;
    # an explicit assembled request was refused above, before entering this scope.
    with equation_implementation(selection), job_provenance(inputs, **binding) as admitted:
        # A later RunStore publication is supported here, as in the BE capture
        # API. Existing spawned routes must retain their separate verified
        # transport/manifest admission rather than laundering an attempt claim.
        _require(admitted.export()["attempt_ref"] is None
                 and admitted.export()["execution_manifest_sha256"] is None,
                 "process-bound BDF publication requires its verified worker/manifest route")
        if ionic:
            result = solve_ionic_transient(compiled, initial, protocol, options=options,
                                          method="bdf", bdf_options=bdf_options, bdf_drive=drive)
        else:
            result = solve_transient(compiled, initial, protocol, options=options, initial_options=initial_options,
                                     method="bdf", bdf_options=bdf_options, bdf_drive=drive)
        _require(type(result) is BDFResult, "the requested BDF call returned another method")
        history = _decode(_encode(result))
        summary = _validate_history(history)
        runtime = admitted.export()["runtime_observed"]
        body = {"schema": _SCHEMA, "qualification": False, "label": _LABEL,
            "request": request, "history": history, "summary": summary,
            "effective_method": _METHOD,
            "effective_bdf_options": {key: history["controls"][key] for key in request["bdf_options"]},
            "remaining_wall_budget_s": history["controls"]["wall_s"],
            "initial_admission_wall_s": history["controls"]["initial_admission_wall_s"],
            "caps": {"requested_max_steps": bdf_options.max_steps,
                "physical_max_steps": request["physical_controls"].get("electronic", request["physical_controls"])["max_steps"],
                "effective_max_steps": history["controls"]["max_steps"],
                "effective_max_attempts": history["controls"]["max_attempts"]},
            "hashes": {"source_tree_sha256": admitted.export()["source_tree_sha256"],
                "runtime_sha256": _sha(_canonical(runtime)), "device_sha256": compiled.identity,
                "initial_sha256": request["initial_sha256"]}}
        _validate_body(body)
        raw = _canonical(_encode(body))
        description = {"scope": "bdf_computation_call", "input_record": inputs,
            "equation_implementation": implementation, "job_provenance": admitted.envelope(),
            "runtime": runtime, "initial_history": "supplied object; no source/runtime provenance inferred"}
        identity = {"H_input": _sha(_canonical(inputs)), "H_physics": compiled.identity,
                    "H_execution": _sha(_canonical(description))}
        capture = {"status": "captured_at_computation", "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "description": description,
            "identity": identity, "history_sha256": _sha(raw)}
        return CapturedBDFRun(result, nodes, raw, _canonical(capture).decode())


def publish_bdf_result(
    store: RunStore, token: AttemptToken, result: BDFResult, *, nodes_m: np.ndarray,
    capture: CapturedBDFRun,
) -> dict[str, Any]:
    """Publish captured completed/failed history; the attempt owner commits it."""
    _require(type(capture) is CapturedBDFRun, "BDF publication needs its contemporaneous capture, not inferred defaults")
    body = _decode(_loads(capture.history_bytes))
    summary = _validate_body(body)
    _require(_same(nodes_m, capture.nodes_m) and _same(nodes_m, body["request"]["nodes_m"])
             and _digest(result) == _digest(body["history"]), "BDF capture differs from result or grid words")
    provenance = _loads(capture.capture_json.encode())
    _capture_identity(body, provenance)
    _require(provenance["history_sha256"] == _sha(capture.history_bytes), "BDF capture history hash differs")
    identity = store.get_run(token.run_id)["request"]["identity"]
    _require(identity == capture.identity, "RunStore identity differs from the captured BDF computation")
    owner = asdict(token)
    binding = publication_binding(provenance["description"]["job_provenance"], owner=owner,
                                  request_identity=identity, result_sha256=_sha(capture.history_bytes))
    history = store.publish(token, "bdf_history", bytes_artifact(capture.history_bytes,
                            metadata={"schema": _SCHEMA, "role": "bdf_history"}))
    manifest = {"schema": _SCHEMA, "summary": summary, "owner": owner, "identity": identity,
        "history_artifact_id": history["artifact_id"], "history_sha256": history["sha256"],
        "provenance": provenance, "publication_binding": binding}
    raw = _canonical(manifest)
    record = store.publish(token, "bdf_manifest", bytes_artifact(raw, metadata={"schema": _SCHEMA, "role": "bdf_manifest"}))
    return {"schema": _SCHEMA, "summary": summary, "owner": owner,
            "manifest_artifact_id": record["artifact_id"], "manifest_sha256": _sha(raw),
            "artifact_ids": [history["artifact_id"], record["artifact_id"]]}
