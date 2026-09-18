"""Bind a failed run's saved extent and terminal evidence without inventing it."""
from __future__ import annotations

import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _row_identity(rows):
    if not rows:
        return None
    row = rows[-1]
    return {"record_index": len(rows) - 1, "row_sha256": _digest(row),
            **{name: row.get(name) for name in ("substeps", "phase", "time_s", "dt_s", "solver_accepted")}}


def build_failure_witness(*, source_commit, protocol, failure, result=None, persisted_rows=()):
    """Use serialized raw observations; callers must not substitute inferred state."""
    result = result or {}
    rows = result.get("accepted_steps", [])
    payload_failure = result.get("failure", {})
    numerical = payload_failure.get("numerical_evidence")
    physical_row = bool(rows and payload_failure.get("type") == "PhysicalCheckFailure"
                        and rows[-1].get("physical_checks_passed") is False)
    if physical_row:
        terminal = {"kind": "saved_physical_failure_row", "available": True,
                    "record_index": len(rows) - 1, "missing_reason": None}
    elif isinstance(numerical, dict) and numerical.get("schema") == "R1NewtonFailureWitnessV1":
        available = numerical.get("terminal_state_available") is True
        terminal = {"kind": "saved_newton_failure", "available": available,
                    "numerical_evidence_sha256": _digest(numerical),
                    "time_s": numerical.get("time_s"), "dt_s": numerical.get("dt_s"),
                    "substeps": numerical.get("substeps"),
                    "missing_reason": None if available else numerical.get(
                        "witness_collection_error", "terminal_iterate_unavailable")}
    else:
        terminal = {"kind": "unavailable", "available": False,
                    "missing_reason": "exception_did_not_supply_a_supported_terminal_state"}
    witness = {"schema": "R1FailureWitnessV1", "source_commit": source_commit,
            "request": {name: protocol.get(name) for name in (
                "stage", "intervals", "control", "amplitude_V", "times_s", "policy", "preparation")},
            "recorded_failure_sha256": _digest(failure),
            "payload_failure_sha256": _digest(payload_failure),
            "last_observed_row": _row_identity(rows),
            "last_persisted_row": _row_identity(persisted_rows),
            "terminal_evidence": terminal,
            "scope": "saved_termination_and_extent_binding; no_inferred_unobserved_execution"}
    return json.loads(json.dumps(witness, allow_nan=False))


def verify_failure_witness(witness, *, source_commit, protocol, failure, result=None, persisted_rows=()):
    expected = build_failure_witness(source_commit=source_commit, protocol=protocol, failure=failure,
                                     result=result, persisted_rows=persisted_rows)
    if witness != expected:
        raise ValueError("failure witness differs from the saved termination, extent or numerical settings")
    return {"termination_binding_verified": True,
            "terminal_state_available": witness["terminal_evidence"]["available"],
            "terminal_evidence_kind": witness["terminal_evidence"]["kind"],
            "terminal_state_physics_verified": None,
            "scope": "binding_only; terminal_equations_require_reconstruction"}
