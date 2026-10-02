"""Current termination bindings for explicit saved-record mutation fixtures.

The fixtures retain computed states and explicitly injected interruptions.
Refreshing their binding after a coordinated mutation lets a negative test
reach its intended equation/sidecar guard. It does not invent a terminal
iterate or claim formal execution for the in-process content fixtures.
"""
from pathlib import Path
import json

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json
from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import build_failure_witness


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def record_development_source(output, project):
    """Record the actual checkout as development provenance, not a formal run."""
    context = require_r1_checkout(project=project)
    assert context.run_class == "development"
    _write(Path(output) / "ExecutionSourceV1.json", context.to_dict())


def refresh_stage_one_failure_witness(output):
    output = Path(output)
    payload = output / "FailedResultV1.json"
    rows = output / "AcceptedStepsV1.json"
    witness = build_failure_witness(
        source_commit=read_json(output / "ExecutionSourceV1.json")["source_commit"],
        protocol=read_json(output / "ProtocolV1.json"), failure=read_json(output / "FailureV1.json"),
        result=read_json(payload) if payload.exists() else None,
        persisted_rows=read_json(rows) if rows.exists() else [],
    )
    _write(output / "FailureWitnessV1.json", witness)
    return witness


def refresh_study_failure_witness(output, case):
    from scripts.run_one_dimensional_mechanism_r1_physics_study import Study

    output, case = Path(output), Path(case)
    failure = read_json(case / "FailureV1.json")
    persisted = case / "AcceptedStepsV1.jsonl"
    witness = build_failure_witness(
        source_commit=read_json(output / "ExecutionSourceV1.json")["source_commit"],
        protocol=Study.failure_protocol(read_json(case / "RequestV1.json")),
        failure=failure, result=failure.get("partial_result"),
        persisted_rows=([json.loads(line) for line in persisted.read_text().splitlines()]
                        if persisted.exists() else []),
    )
    _write(case / "FailureWitnessV1.json", witness)
    return witness
