"""Explicit production pair schemas, numeric sidecars and preparation codec.

Historical PairV1 is never relabeled. Numerical methods are supplied by the
selected run backend; no module aliases or process-global state are changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import one_dimensional_mechanism_r1_state as states
from .one_dimensional_mechanism_r1_state import R1PreparedState, R1StateError, canonical, digest, json_data
from ..physics.compensated import DD


SCHEMA = "R1CommonStatePairV2"
STEP_SCHEMA = "R1ControlledStepV2"
ZERO_SCHEMA = "R1ZeroExcitationV2"
FAILED_PREPARATION_SCHEMA = "R1FailedPreparationPairV2"
REPRESENTATION = "float64-pair-v1"
LEGACY_REPRESENTATION = "float64-baseline"
EVIDENCE_REVISION = 7
FINE_FIELDS = frozenset(("phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3",
    "occupancy", "sheet_charge_C_m2", "trace_potential_V", "trace_state_m3", "storage",
    "poisson_residual_C_m2", "electron_current_A_m2", "hole_current_A_m2", "positive_flux_m2_s",
    "positive_rate_m3_s", "boundary_flux_m2_s"))
LEGACY_SNAPSHOT_FIELDS = frozenset(("n_m3", "p_m3", "positive_m3", "occupancy", "phi_V",
    "sheet_charge_C_m2", "trace_potential_V", "trace_state_m3", "capture_m2_s",
    "positive_inventory_m2", "poisson_residual_C_m2", "local_residual"))
SNAPSHOT_FIELDS = LEGACY_SNAPSHOT_FIELDS | {"precision_" + name + suffix
                                           for name in FINE_FIELDS for suffix in ("_hi", "_lo")}
OUTPUT_FIELDS = {"n_m3", "p_m3", "positive_m3", "occupancy", "phi_V", "sheet_charge_C_m2"} | {
    "precision_" + name + suffix for name in FINE_FIELDS for suffix in ("_hi", "_lo")}
LEGACY_OUTPUT_FIELDS = frozenset(("n_m3", "p_m3", "positive_m3", "occupancy", "phi_V", "sheet_charge_C_m2"))
EQUATION_ARRAY_FIELDS = frozenset(("coordinate", "electron_qf_increment_V", "hole_qf_increment_V",
    "storage", "rate", "poisson_residual_C_m2", "direct_poisson_residual_C_m2", "local_residual",
    "electron_current_A_m2", "hole_current_A_m2", "positive_ion_flux_m2_s", "positive_ion_rate_m3_s",
    "positive_ion_current_A_m2", "carrier_conduction_A_m2", "conduction_A_m2"))
EXTRA_FIELDS = {"seed_preparation", "seed_preparation_sha256", "representation", "dc_state_role"}
REPLACED_FIELDS = {"schema", "state", "preparation_checks", "qf_references_V", "verification", "sha256"}
DC_STATE_ROLE = "verified_legacy_dc_seed_not_the_fine_physical_state"


def verification_contract():
    return {"schema": "R1PreparationPairVerificationV2", "representation": REPRESENTATION,
        "seed_route": "explicit_legacy_backend_preparation_and_physics_verification",
        "fine_route": "explicit_pair_factory_from_verified_legacy_seed",
        "recomputed": ["seed_preparation", "state", "qf_references_V", "preparation_checks"],
        "precision": "all_seventeen_normalized_hi_lo_pairs_are_exact_state_identity",
        "gates": "original_equilibrium_checks_with_declared_preparation_and_consumer_policy",
        "provenance_only": ["environment", "created_utc", "dc_state.optimizer_history"],
        "scientific_scope": "one_common_state_preparation_not_full_R1_2_or_P2_qualification"}


def _validated_snapshot_arrays(record, *, require_fine=True):
    if not isinstance(record, dict) or set(record) != (SNAPSHOT_FIELDS if require_fine else LEGACY_SNAPSHOT_FIELDS):
        raise R1StateError("production pair snapshot key coverage mismatch")
    arrays = {}
    for name, value in record.items():
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise R1StateError("production snapshot has nonnumeric/nonfinite array: " + name)
        arrays[name] = array
    nodes, interfaces = arrays["n_m3"].size, arrays["occupancy"].size
    shapes = {**dict.fromkeys(("n_m3", "p_m3", "positive_m3", "phi_V", "dqfn_V", "dqfp_V", "positive_rate_m3_s"), (nodes,)),
        **dict.fromkeys(("occupancy", "sheet_charge_C_m2"), (interfaces,)),
        **dict.fromkeys(("electron_current_A_m2", "hole_current_A_m2", "positive_flux_m2_s"), (nodes - 1,)),
        "trace_potential_V": (interfaces, 2), "trace_state_m3": (interfaces, 4),
        "capture_m2_s": (interfaces, 4), "local_residual": (6 * interfaces,),
        "poisson_residual_C_m2": (nodes - 2,), "boundary_flux_m2_s": (2,)}
    if nodes < 3 or interfaces < 1:
        raise R1StateError("invalid production state node/interface dimensions")
    for name, shape in shapes.items():
        if name in arrays and arrays[name].shape != shape:
            raise R1StateError("production snapshot shape mismatch: " + name)
    if require_fine:
        for name in FINE_FIELDS:
            high, low = arrays["precision_" + name + "_hi"], arrays["precision_" + name + "_lo"]
            if high.shape != low.shape or name in shapes and high.shape != shapes[name]:
                raise R1StateError("production pair high/low shape mismatch: " + name)
            if name == "storage" and (high.ndim != 1 or not 2 * (nodes - 2) + interfaces <= high.size <= 3 * nodes - 4 + interfaces):
                raise R1StateError("production pair storage shape mismatch")
            normalized = DD(high, low)
            if not np.array_equal(normalized.hi, high) or not np.array_equal(normalized.lo, low):
                raise R1StateError("production pair words are not normalized: " + name)
            if name in arrays and not np.array_equal(high, arrays[name]):
                raise R1StateError("production pair high word differs from its binary64 snapshot: " + name)
    return arrays


def validate_snapshot(record, *, require_fine=True):
    _validated_snapshot_arrays(record, require_fine=require_fine)
    return record


def _structural_record(record):
    if not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("representation") != REPRESENTATION:
        raise R1StateError("production pair requires its explicit V2 schema and representation")
    seed_record = record.get("seed_preparation")
    if not isinstance(seed_record, dict) or seed_record.get("schema") != "R1CommonStateV1":
        raise R1StateError("production pair lacks its full original seed preparation")
    seed = R1PreparedState.from_dict(seed_record)
    from .one_dimensional_mechanism_r1_result_contract import verify_prepared_metadata
    verify_prepared_metadata(seed_record)
    if set(record) != set(seed_record) | EXTRA_FIELDS:
        raise R1StateError("production pair preparation key coverage mismatch")
    if record.get("seed_preparation_sha256") != seed.sha256 or record.get("dc_state_role") != DC_STATE_ROLE:
        raise R1StateError("production pair seed identity/role mismatch")
    for name in set(seed_record) - REPLACED_FIELDS:
        if canonical(record[name]) != canonical(seed_record[name]):
            raise R1StateError("production pair changed seed metadata: " + name)
    if record.get("verification") != verification_contract():
        raise R1StateError("production pair verification contract mismatch")
    validate_snapshot(record["state"])
    states._require_finite_preparation(record)
    if record.get("sha256") != digest({key: value for key, value in record.items() if key != "sha256"}):
        raise R1StateError("production pair preparation hash mismatch")
    return seed


@dataclass(frozen=True, slots=True)
class R1CommonStatePairV2(R1PreparedState):
    def __post_init__(self):
        record = json.loads(self.canonical_json)
        _structural_record(record)
        object.__setattr__(self, "canonical_json", canonical(record))


def decode_prepared(record, representation=None):
    record = record.to_dict() if hasattr(record, "to_dict") else record
    if representation in ("pair", REPRESENTATION):
        return R1CommonStatePairV2.from_dict(record)
    if representation in (None, "legacy", LEGACY_REPRESENTATION):
        return R1PreparedState.from_dict(record)
    raise R1StateError("unsupported requested preparation representation")


def _fine(baseline, before, fine_factory):
    grid = np.array(baseline.grid, copy=True)
    system, initial = fine_factory(baseline, before)
    if json_data(system.controls) != {"nu_I": 1, "nu_t": 1} or not np.array_equal(system.grid, grid):
        raise R1StateError("production common preparation changed the D controls or grid")
    return system, initial


def _checks(system, initial, policy, *, require_pass=True):
    checks = json_data(states.equilibrium_checks(system, initial, policy))
    if require_pass and (checks.get("certified") is not True or checks.get("reasons")):
        exc = R1StateError("production pair preparation fails original equilibrium gates")
        exc.result = {"schema": "R1FailedPreparationPairV2", "checks": checks, "certified": False}
        raise exc
    return checks


def prepare_pair_state(stack, intervals, binding, *, policy=None, legacy_prepare, legacy_verify, fine_factory, snapshot):
    seed = legacy_prepare(stack, intervals, binding, policy=policy)
    baseline, before = legacy_verify(seed, stack, binding, policy=None)
    seed_record = seed.to_dict()
    if seed_record.get("schema") != "R1CommonStateV1":
        raise R1StateError("production pair seed callback did not return the original schema")
    system, initial = _fine(baseline, before, fine_factory)
    payload = {**seed_record, "schema": SCHEMA, "representation": REPRESENTATION,
        "dc_state_role": DC_STATE_ROLE, "seed_preparation": seed_record, "seed_preparation_sha256": seed.sha256,
        "state": validate_snapshot(json_data(snapshot(system, initial))),
        "qf_references_V": json_data({"electron": system.qfn_reference, "hole": system.qfp_reference}),
        "preparation_checks": _checks(system, initial, states._preparation_policy(seed_record["preparation_policy"]), require_pass=False),
        "verification": verification_contract()}
    payload.pop("sha256", None)
    if payload["source"] != states.execution_source():
        raise R1StateError("production pair source changed during preparation")
    payload["sha256"] = digest(payload)
    if payload["preparation_checks"].get("certified") is not True or payload["preparation_checks"].get("reasons"):
        error = R1StateError("production pair preparation fails original equilibrium gates")
        error.result = {"schema": FAILED_PREPARATION_SCHEMA, "representation": REPRESENTATION,
            "raw_preparation": payload, "certified": False,
            "reasons": payload["preparation_checks"]["reasons"]}
        raise error
    return R1CommonStatePairV2.from_dict(payload)


def verify_pair_state(prepared, stack, binding, *, policy=None, legacy_verify, fine_factory, snapshot, require_pass=True):
    record = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
    pair = R1CommonStatePairV2.from_dict(record)
    record = pair.to_dict()
    from .one_dimensional_mechanism_r1_checkout import current_execution_context
    actual_source = states.execution_source()
    matches = (record["source"] == actual_source if current_execution_context() is not None
               else all(record["source"].get(key) == actual_source[key] for key in ("files", "study_input")))
    if not matches:
        raise R1StateError("production pair equations differ from the executing backend")
    seed = _structural_record(record)
    baseline, before = legacy_verify(seed, stack, binding, policy=policy)
    system, initial = _fine(baseline, before, fine_factory)
    actual = validate_snapshot(json_data(snapshot(system, initial)))
    if canonical(actual) != canonical(record["state"]):
        raise R1StateError("production pair physical fields differ from deterministic reconstruction")
    references = json_data({"electron": system.qfn_reference, "hole": system.qfp_reference})
    if canonical(record["qf_references_V"]) != canonical(references):
        raise R1StateError("production pair QF reference mismatch")
    checks = _checks(system, initial, states._preparation_policy(record["preparation_policy"]), require_pass=require_pass)
    if canonical(checks) != canonical(record["preparation_checks"]):
        raise R1StateError("production pair equilibrium checks differ from reconstruction")
    if policy is not None:
        _checks(system, initial, policy, require_pass=require_pass)
    return system, initial


def finalize_record(record, kind="step"):
    if kind not in ("step", "zero"):
        raise ValueError("unsupported production pair result kind")
    record["schema"] = STEP_SCHEMA if kind == "step" else ZERO_SCHEMA
    record["representation"] = REPRESENTATION
    record["source"] = states.execution_source()
    from .one_dimensional_mechanism_r1_protocol import nonfinite_numeric_paths
    paths = nonfinite_numeric_paths(record)
    if paths and record.get("certificate", {}).get("certified") is False:
        # Preserve the original failure and raw attempted state. Strict JSON
        # tagging plus a lossless diagnostic sidecar happens at the writer;
        # no finite canonical digest is asserted for this failed payload.
        record.pop("sha256", None)
        record["certificate"]["finite_numeric_evidence"] = {
            "passed": False, "nonfinite_numeric_paths": paths,
            "scope": "all_numeric_leaves_before_digest"}
        record.setdefault("failure", {})["nonfinite_numeric_paths"] = paths
        return record
    for row in record.get("accepted_steps", []):
        validate_snapshot(row["state"])
    if kind == "zero":
        for control in record.get("controls", {}).values():
            if "population_identity" in control:
                validate_snapshot(control["population_identity"])
    record.pop("sha256", None)
    record["sha256"] = digest(record)
    return record


def numeric_arrays(value, prefix="data", *, allow_nonfinite=False):
    result = {}
    _collect_numeric_arrays(value, prefix, result, allow_nonfinite)
    return result


def _collect_numeric_arrays(item, path, result, allow_nonfinite):
    # A module-level collector avoids both per-leaf dictionaries and a
    # recursive closure retaining its completed array mapping until cyclic GC.
    if isinstance(item, dict):
        if (len(item) == 2 and "hi" in item and "lo" in item
                and all(isinstance(word, (int, float, np.integer, np.floating))
                        and not isinstance(word, (bool, np.bool_)) for word in item.values())):
            for name, word in item.items():
                array = np.asarray(word, dtype=np.float64)
                if not allow_nonfinite and not np.isfinite(array):
                    raise R1StateError("nonfinite production scalar pair word: " + path + "." + name)
                result[path + "." + name] = array
            return
        for name, child in item.items():
            _collect_numeric_arrays(child, path + "." + str(name), result, allow_nonfinite)
    elif isinstance(item, (np.ndarray, list, tuple)):
        try:
            array = np.asarray(item)
        except ValueError:
            array = np.asarray([], dtype=object)
        if array.dtype.kind in ("fiuc" if allow_nonfinite else "fiu") and array.ndim > 0:
            if not allow_nonfinite and not np.all(np.isfinite(array)):
                raise R1StateError("nonfinite production sidecar field: " + path)
            result[path] = array
        else:
            for index, child in enumerate(item):
                _collect_numeric_arrays(child, path + "." + str(index), result, allow_nonfinite)
    elif allow_nonfinite and isinstance(item, (float, complex, np.floating, np.complexfloating)) and not np.isfinite(item):
        result[path] = np.asarray(item)


def _stack(records, label):
    return _stack_arrays([numeric_arrays(record, prefix="") for record in records], label)


def _stack_arrays(arrays, label):
    if not arrays:
        return {}
    expected = set(arrays[0])
    if any(set(record) != expected for record in arrays):
        raise R1StateError("production sidecar varying field coverage: " + label)
    result = {}
    for name in sorted(expected):
        if any(record[name].shape != arrays[0][name].shape for record in arrays):
            raise R1StateError("production sidecar ragged array: " + label + name)
        result[name.removeprefix(".")] = np.stack([record[name] for record in arrays])
    return result


def state_sidecar_payload(record):
    """Bounded member count; repeated states are stacked by actual row.

    All pair state words, actual eliminated fields/shared inputs/seed, initial
    and upstream state arrays, coordinates, output samples and charge/current
    result arrays have explicit JSON associations. Other equation diagnostics
    remain in the sealed JSON, rather than becoming thousands of ZIP members.
    """
    if isinstance(record, list) and (not record or isinstance(record[0], dict) and "state" in record[0]):
        pair = not record or "precision_phi_V_lo" in record[0]["state"]
        record = {"schema": STEP_SCHEMA if pair else "R1ControlledStepV1", "accepted_steps": record}
    if not isinstance(record, dict):
        return record
    schema = record.get("schema")
    if schema in (SCHEMA, "R1CommonStateV1"):
        validate_snapshot(record["state"], require_fine=schema == SCHEMA)
        payload = {"state": record["state"], "qf_references_V": record["qf_references_V"],
                   "dc_state": record["dc_state"], "grid": record["grid"]}
        if schema == SCHEMA:
            payload["seed_state"] = record["seed_preparation"]["state"]
            payload["seed_qf_references_V"] = record["seed_preparation"]["qf_references_V"]
        return payload
    if schema in (ZERO_SCHEMA, "R1ZeroExcitationV1"):
        for control in record["controls"].values():
            validate_snapshot(control["population_identity"], require_fine=schema == ZERO_SCHEMA)
        return {"controls": record["controls"]}
    if schema not in (STEP_SCHEMA, "R1ControlledStepV1"):
        return record
    pair = schema == STEP_SCHEMA
    rows = record.get("accepted_steps", [])
    accepted_states = _stack_arrays(
        [_validated_snapshot_arrays(row["state"], require_fine=pair) for row in rows], "accepted_states")
    payload = {"accepted_states": accepted_states,
        "row_identity": np.asarray([[row["substeps"], row["time_s"], row["dt_s"]] for row in rows]),
        "times_s": record.get("times_s", []), "voltage_V": record.get("voltage_V", [])}
    if "accepted_state_arrays" in record:
        arrays = record["accepted_state_arrays"]
        if set(arrays) != (SNAPSHOT_FIELDS if pair else LEGACY_SNAPSHOT_FIELDS):
            raise R1StateError("production accepted_state_arrays key coverage mismatch")
        for name, value in arrays.items():
            expected = accepted_states[name] if rows else np.asarray([])
            if not np.array_equal(np.asarray(value), expected):
                raise R1StateError("production accepted_state_arrays differ from actual rows: " + name)
    if "output_states" in record:
        if set(record["output_states"]) != (OUTPUT_FIELDS if pair else LEGACY_OUTPUT_FIELDS):
            raise R1StateError("production output_states key coverage mismatch")
        payload["output_states"] = record["output_states"]
    available = [(index, row["physics_reconstruction"]) for index, row in enumerate(rows)
                 if isinstance(row.get("physics_reconstruction"), dict)
                 and row["physics_reconstruction"].get("available") is not False]
    payload["equation_row_indices"] = np.asarray([index for index, _ in available], dtype=int)
    payload["equation_arrays"] = _stack([{name: value[name] for name in EQUATION_ARRAY_FIELDS}
                                          for _, value in available], "equation_arrays")
    sides = [value.get("eliminated_precision") for _, value in available]
    if pair and any(value is None for value in sides):
        raise R1StateError("production sidecar requires actual eliminated precision records")
    if pair and sides:
        payload["eliminated_fields"] = _stack([value["fields"] for value in sides], "eliminated_fields")
        payload["eliminated_shared_inputs"] = _stack([value["shared_inputs"]["fields"] for value in sides], "eliminated_shared_inputs")
        payload["eliminated_seed_phi_V"] = _stack([{"seed": value["solve"]["seed_phi_V"]} for value in sides], "eliminated_seed")
    for name in ("initial_event", "physics_reconstruction", "finite_step_averages", "failure"):
        if name in record:
            payload[name] = record[name]
    if "charge_integral" in record:
        charge = record["charge_integral"]
        levels = sorted(charge["regular_by_substeps_C_m2"], key=int)
        payload["charge_integral"] = {
            "substeps": np.asarray([int(level) for level in levels]),
            "regular_C_m2": np.asarray([charge["regular_by_substeps_C_m2"][level] for level in levels]),
            "complete_C_m2": np.asarray([charge["complete_by_substeps_C_m2"][level] for level in levels]),
            "impulse_C_m2": np.asarray([charge["impulse_charge_C_m2"]])}
    if "regular_currents" in record:
        payload["regular_current_arrays"] = _stack(record["regular_currents"], "regular_current_arrays")
    return payload


def write_numeric_sidecar(path, record):
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **numeric_arrays(state_sidecar_payload(record)))


def verify_numeric_sidecar(path, record):
    expected = numeric_arrays(state_sidecar_payload(record))
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected) or len(arrays.files) != len(set(arrays.files)):
            raise R1StateError("production JSON/NPZ key coverage mismatch")
        for name, value in expected.items():
            actual = arrays[name]
            if (actual.dtype != value.dtype or actual.shape != value.shape
                    or not np.all(np.isfinite(actual)) or actual.tobytes() != value.tobytes()):
                raise R1StateError("production JSON/NPZ numeric mismatch: " + name)
    return {"verified_array_count": len(expected), "exact_key_coverage": True}


def contains_nonfinite_tags(value):
    if isinstance(value, dict):
        return (set(value) == {"nonfinite"} and value["nonfinite"] in ("nan", "inf", "-inf")) or any(
            contains_nonfinite_tags(child) for child in value.values())
    return isinstance(value, (tuple, list)) and any(contains_nonfinite_tags(child) for child in value)


def verify_failed_numeric_sidecar(path, record):
    """Check a raw unsealable failure, without granting finite-state validity."""
    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"nonfinite"} and value["nonfinite"] in ("nan", "inf", "-inf"):
                return float(value["nonfinite"])
            if set(value) == {"real", "imag"}:
                return complex(restore(value["real"]), restore(value["imag"]))
            return {name: restore(child) for name, child in value.items()}
        if isinstance(value, list):
            return [restore(child) for child in value]
        return value
    if not contains_nonfinite_tags(record) or record.get("certificate", {}).get("certified") is not False or "sha256" in record:
        raise R1StateError("raw nonfinite pair evidence must remain failed and unsealed")
    expected = numeric_arrays(restore(record), allow_nonfinite=True)
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected) or len(arrays.files) != len(set(arrays.files)):
            raise R1StateError("raw failed pair JSON/NPZ key coverage mismatch")
        for name, value in expected.items():
            actual = arrays[name]
            if actual.dtype != value.dtype or actual.shape != value.shape or not np.array_equal(actual, value, equal_nan=True):
                raise R1StateError("raw failed pair JSON/NPZ numeric mismatch: " + name)
    return {"verified_array_count": len(expected), "exact_key_coverage": True, "finite_state_validity": False}
