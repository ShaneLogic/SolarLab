"""Versioned preparation records for the opt-in float64-pair prototype.

The legacy DC preparation is a fully verified seed, never the claimed fine
state. The fine state is rebuilt deterministically and checked under the
unchanged equilibrium gates. Importing this module installs no hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from . import one_dimensional_mechanism_r1_state as states
from .one_dimensional_mechanism_r1_state import R1PreparedState as LegacyPreparedState
from .one_dimensional_mechanism_r1_state import R1StateError, canonical, digest, json_data


SCHEMA = "R1CommonStatePairV1"
REPRESENTATION = "float64-pair-v1"
SEED_SCHEMA = "R1CommonStateV1"
DC_STATE_ROLE = "verified_legacy_dc_seed_not_the_fine_physical_state"
EXTRA_FIELDS = {"seed_preparation", "seed_preparation_sha256", "representation", "dc_state_role"}
REPLACED_FIELDS = {"schema", "state", "preparation_checks", "qf_references_V", "verification", "sha256"}
REQUIRED_PAIRS = {"phi_V", "n_m3", "p_m3", "positive_m3", "occupancy",
                  "sheet_charge_C_m2", "trace_potential_V", "trace_state_m3", "dqfn_V", "dqfp_V"}


def verification_contract():
    return {
        "schema": "R1PreparationPairVerificationV1",
        "seed_route": "original_verify_prepared_physics_under_legacy_constructor_and_snapshot",
        "fine_route": "deterministic_fine_factory_from_verified_seed_system_and_state",
        "recomputed": ["seed_preparation", "state", "qf_references_V", "preparation_checks"],
        "precision": "flat_numeric_hi_lo_fields_are_part_of_exact_state_identity",
        "gates": "original_equilibrium_checks_with_declared_preparation_and_consumer_policy",
        "provenance_only": ["environment", "created_utc", "dc_state.optimizer_history"],
        "scientific_scope": "prototype_preparation_only_not_formal_R1_2_acceptance",
    }


def _snapshot_fields(record):
    if not isinstance(record, dict):
        raise R1StateError("pair common-state snapshot must be an object")
    names = {name.removeprefix("precision_").removesuffix("_hi")
             for name in record if name.startswith("precision_") and name.endswith("_hi")}
    lows = {name.removeprefix("precision_").removesuffix("_lo")
            for name in record if name.startswith("precision_") and name.endswith("_lo")}
    if names != lows or not REQUIRED_PAIRS <= names:
        raise R1StateError("pair common-state lacks required matching high/low fields")
    for name in names:
        high = np.asarray(record["precision_" + name + "_hi"])
        low = np.asarray(record["precision_" + name + "_lo"])
        if (high.dtype.kind not in "fiu" or low.dtype.kind not in "fiu"
                or high.shape != low.shape or not np.all(np.isfinite(high)) or not np.all(np.isfinite(low))):
            raise R1StateError("pair common-state has invalid high/low arrays: " + name)
        if name in record and np.asarray(record[name]).shape != high.shape:
            raise R1StateError("pair common-state high/low shape differs from the main state: " + name)


def _structural_record(record):
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise R1StateError("unrecognized R1 pair common-state schema; legacy state cannot be relabeled")
    if record.get("representation") != REPRESENTATION or record.get("dc_state_role") != DC_STATE_ROLE:
        raise R1StateError("pair common-state representation or seed role mismatch")
    seed_record = record.get("seed_preparation")
    if not isinstance(seed_record, dict) or seed_record.get("schema") != SEED_SCHEMA:
        raise R1StateError("pair common-state must contain its complete legacy seed preparation")
    seed = LegacyPreparedState.from_dict(seed_record)
    if record.get("seed_preparation_sha256") != seed.sha256:
        raise R1StateError("pair common-state seed identity mismatch")
    if set(record) != set(seed_record) | EXTRA_FIELDS:
        raise R1StateError("pair common-state top-level field coverage mismatch")
    for name in set(seed_record) - REPLACED_FIELDS:
        if canonical(record.get(name)) != canonical(seed_record[name]):
            raise R1StateError("pair common-state changed its seed identity field: " + name)
    if record.get("verification") != verification_contract():
        raise R1StateError("pair common-state verification contract mismatch")
    _snapshot_fields(record.get("state"))
    states._require_finite_preparation(record)
    body = {key: value for key, value in record.items() if key != "sha256"}
    if record.get("sha256") != digest(body):
        raise R1StateError("pair common-state content hash mismatch")
    return seed


@dataclass(frozen=True, slots=True)
class R1CommonStatePair(LegacyPreparedState):
    """A new schema that remains an immutable prepared-object API instance.

The original R1PreparedState.from_dict deliberately continues rejecting this
schema. Saved pair dictionaries must be decoded through this class explicitly.
"""

    def __post_init__(self):
        record = json.loads(self.canonical_json)
        _structural_record(record)
        object.__setattr__(self, "canonical_json", canonical(record))


def _fine_snapshot(system, initial):
    record = json_data(states.snapshot(system, initial))
    _snapshot_fields(record)
    return record


def _qf_references(system):
    return json_data({"electron": system.qfn_reference, "hole": system.qfp_reference})


def _checks(system, initial, policy):
    # This is intentionally the existing gate function, not a weaker new test.
    return json_data(states.equilibrium_checks(system, initial, policy))


def _require_checks(checks, label, raw=None):
    metrics, limits = checks.get("metrics", {}), checks.get("limits", {})
    passed = (checks.get("certified") is True and checks.get("reasons") == []
              and bool(metrics) and set(metrics) == set(limits)
              and all(type(value) in (float, int) and np.isfinite(value)
                      and value <= limits[name] for name, value in metrics.items()))
    if not passed:
        error = R1StateError(label + " fails original equilibrium gates: " + str(checks.get("reasons")))
        error.result = {"schema": "R1FailedPreparationPairV1", "certified": False,
                        "representation": REPRESENTATION, "checks": checks,
                        "raw_preparation": raw}
        raise error


def build_hooks(original_prepare, original_verify, legacy_context, fine_factory):
    """Return opt-in preparation and physics-verification hooks.

``legacy_context()`` must restore the original constructor, snapshot and seed
codec while active. ``fine_factory(baseline_system, baseline_state)`` returns a
new actual fine system/state without performing another legacy DC optimization.
Source identity remains strict; caller policy can add gates but cannot change
the recorded fine initial state. The hooks never accept a legacy schema as a
fine preparation.
"""

    def seed_system(seed, stack, binding, policy=None):
        with legacy_context():
            return original_verify(seed, stack, binding, policy=policy)

    def actual_fine(baseline, before):
        expected = {"nu_I": 1, "nu_t": 1}
        if json_data(getattr(baseline, "controls", None)) != expected:
            raise R1StateError("verified seed is not a common D preparation")
        expected_grid = np.array(baseline.grid, copy=True)
        system, initial = fine_factory(baseline, before)
        if json_data(getattr(system, "controls", None)) != expected:
            raise R1StateError("fine common preparation must preserve both D controls")
        if not np.array_equal(system.grid, expected_grid):
            raise R1StateError("fine common preparation changed the verified seed grid")
        return system, initial

    def prepare(stack, intervals, binding, *, policy=None):
        with legacy_context():
            seed = original_prepare(stack, intervals, binding, policy=policy)
            baseline, before = original_verify(seed, stack, binding, policy=None)
        if not isinstance(seed, LegacyPreparedState) or seed.to_dict().get("schema") != SEED_SCHEMA:
            raise R1StateError("legacy preparation hook did not return the original seed schema")
        seed_record = seed.to_dict()
        preparation_policy = states._preparation_policy(seed_record["preparation_policy"])
        system, initial = actual_fine(baseline, before)
        payload = {**seed_record, "schema": SCHEMA, "representation": REPRESENTATION,
                   "dc_state_role": DC_STATE_ROLE, "seed_preparation": seed_record,
                   "seed_preparation_sha256": seed.sha256,
                   "state": _fine_snapshot(system, initial),
                   "qf_references_V": _qf_references(system),
                   "preparation_checks": _checks(system, initial, preparation_policy),
                   "verification": verification_contract()}
        payload.pop("sha256", None)
        _require_checks(payload["preparation_checks"], "fine common D equilibrium", payload)
        if payload["source"] != states.execution_source():
            raise R1StateError("pair preparation source changed between legacy seed and fine state")
        payload["sha256"] = digest(payload)
        return R1CommonStatePair.from_dict(payload)

    def verify(prepared, stack, binding, *, policy=None):
        record = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
        pair = R1CommonStatePair.from_dict(record)
        record = pair.to_dict()
        if record["source"] != states.execution_source():
            raise R1StateError("pair common-state identity mismatch: source")
        seed = _structural_record(record)
        # All original seed checks run, including physical arrays, reference,
        # controls, declared policy, QSS certificate, grid and contact identity.
        baseline, before = seed_system(seed, stack, binding)
        preparation_policy = states._preparation_policy(record["preparation_policy"])
        system, initial = actual_fine(baseline, before)
        actual = _fine_snapshot(system, initial)
        if canonical(record["state"]) != canonical(actual):
            raise R1StateError("pair common-state physical fields differ from deterministic reconstruction")
        if canonical(record["qf_references_V"]) != canonical(_qf_references(system)):
            raise R1StateError("pair common-state fine QF references mismatch")
        checks = _checks(system, initial, preparation_policy)
        _require_checks(checks, "reconstructed fine common D equilibrium", record)
        if canonical(record["preparation_checks"]) != canonical(checks):
            raise R1StateError("pair common-state preparation_checks differ from recomputed physics")
        if policy is not None:
            states._preparation_policy(policy)
            # Reuse the original consumer-policy semantics, then require the
            # resulting actual fine state to retain the same stored identity.
            consumer_base, consumer_before = seed_system(seed, stack, binding, policy=policy)
            consumer, consumer_initial = actual_fine(consumer_base, consumer_before)
            if (canonical(_fine_snapshot(consumer, consumer_initial)) != canonical(actual)
                    or canonical(_qf_references(consumer)) != canonical(_qf_references(system))):
                raise R1StateError("consumer policy changed the fine common-state physical identity")
            _require_checks(_checks(consumer, consumer_initial, policy),
                            "fine common state under consumer policy", record)
            system, initial = consumer, consumer_initial
        return system, initial

    return prepare, verify


__all__ = ["SCHEMA", "REPRESENTATION", "R1CommonStatePair", "build_hooks"]
