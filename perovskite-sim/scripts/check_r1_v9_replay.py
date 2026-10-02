#!/usr/bin/env python3
"""Reseal mutations of one actual production short result and call real consumers."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import PAIR
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json
from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import write_numeric_sidecar, verify_numeric_sidecar
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import digest, json_data
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.compensated import DD


def seal(record):
    record.pop("sha256", None)
    record["sha256"] = digest(record)
    return record


def write(path, value):
    path.write_text(json.dumps(json_data(value), sort_keys=True, indent=2, allow_nan=False) + "\n")


def perturb_low(state, name="n_m3"):
    high, low = [np.asarray(state["precision_" + name + "_" + word], dtype=float).copy() for word in ("hi", "lo")]
    index = 1
    low.flat[index] += max(abs(float(low.flat[index])) * .1, abs(float(high.flat[index])) * 1e-25, 1e-30)
    value = DD(high, low)
    assert np.array_equal(value.hi, high), "fault must change only a normalized low word"
    state["precision_" + name + "_lo"] = value.lo.tolist()


def check_case(case, prepared, result, stack, binding, output):
    output.mkdir()
    original_prepared, original_result = prepared, result
    prepared, result = deepcopy(prepared), deepcopy(result)
    format_fault = False
    if case == "prepared_missing_low":
        prepared["state"].pop("precision_n_m3_lo")
        format_fault = True
    elif case == "prepared_changed_low":
        perturb_low(prepared["state"])
    elif case == "snapshot_missing_low":
        result["accepted_steps"][1]["state"].pop("precision_n_m3_lo")
        format_fault = True
    elif case == "snapshot_wrong_shape":
        result["accepted_steps"][1]["state"]["precision_n_m3_lo"] = [0.]
        format_fault = True
    elif case == "accepted_array_missing_low":
        result["accepted_state_arrays"].pop("precision_n_m3_lo")
        format_fault = True
    elif case == "output_unknown_key":
        result["output_states"]["unclassified_state"] = [0.]
        format_fault = True
    elif case == "accepted_row_unknown_key":
        result["accepted_steps"][1]["unclassified_physics"] = 1.
    elif case == "eliminated_unknown_key":
        result["accepted_steps"][1]["physics_reconstruction"]["eliminated_precision"]["unclassified"] = 1.
    elif case == "coordinated_state_low":
        row = result["accepted_steps"][1]
        perturb_low(row["state"])
        key = "precision_n_m3_lo"
        result["accepted_state_arrays"][key][1] = row["state"][key]
    elif case not in ("healthy", "npz_missing_low", "npz_changed_low", "npz_float32", "npz_signed_zero"):
        raise ValueError("unknown negative case")
    seal(prepared)
    result["prepared_sha256"] = prepared["sha256"]
    seal(result)
    write(output / "PreparedV1.json", prepared)
    write(output / "ResultV1.json", result)
    # Malformed JSON must reach the consumer, so retain the bound original
    # sidecar for structural faults; valid coordinated mutations get new NPZs.
    prep_side = output / "PreparedStateArraysV1.npz"
    result_side = output / "StateArraysV1.npz"
    write_numeric_sidecar(prep_side, original_prepared if format_fault else prepared)
    write_numeric_sidecar(result_side, original_result if format_fault else result)
    if case.startswith("npz_"):
        with np.load(result_side, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        name = "data.accepted_states.precision_n_m3_lo"
        if case == "npz_missing_low":
            arrays.pop(name)
        elif case == "npz_float32":
            arrays[name] = arrays[name].astype(np.float32)
        elif case == "npz_signed_zero":
            zero_indices = np.flatnonzero(arrays[name] == 0.)
            if not zero_indices.size:
                name = next(key for key, value in arrays.items() if np.any(value == 0.))
                zero_indices = np.flatnonzero(arrays[name] == 0.)
            index = zero_indices[0]
            arrays[name].flat[index] = -0. if not np.signbit(arrays[name].flat[index]) else 0.
        else:
            arrays[name][1, 1] += max(abs(float(arrays[name][1, 1])) * .1, 1e-20)
        np.savez_compressed(result_side, **arrays)
    started = time.monotonic()
    try:
        saved_prepared, saved_result = read_json(output / "PreparedV1.json"), read_json(output / "ResultV1.json")
        PAIR.decode_prepared(saved_prepared)
        verify_numeric_sidecar(prep_side, saved_prepared)
        verify_numeric_sidecar(result_side, saved_result)
        verified = verify_r1_step_physics(stack, 16, binding, saved_prepared, saved_result,
            backend="pair", expected_prepared_sha256=saved_prepared["sha256"])
        if not verified["certified"]:
            raise ValueError("real physical verifier did not certify result")
        failure = None
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    manifest = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                for p in sorted(output.iterdir()) if p.is_file()}
    write(output / "ManifestV1.json", manifest)
    return {"case": case, "rejected": failure is not None, "failure": failure,
            "elapsed_s": time.monotonic() - started, "record_sha256": result["sha256"],
            "prepared_sha256": prepared["sha256"], "manifest_sha256": hashlib.sha256((output / "ManifestV1.json").read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "result", "fixture", "binding", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    prepared, result = read_json(args.prepared), read_json(args.result)
    if (prepared.get("schema") != "R1CommonStatePairV2"
            or result.get("schema") != "R1ControlledStepV2"
            or len(result["accepted_steps"]) != 10):
        raise ValueError("campaign requires the actual production PairV2 ten-row short result")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stack, binding = load_device_from_yaml(args.fixture), read_json(args.binding)
    cases = ("healthy", "prepared_missing_low", "prepared_changed_low", "snapshot_missing_low",
        "snapshot_wrong_shape", "accepted_array_missing_low", "output_unknown_key",
        "accepted_row_unknown_key", "eliminated_unknown_key", "coordinated_state_low",
        "npz_missing_low", "npz_changed_low", "npz_float32", "npz_signed_zero")
    with threadpool_limits(1):
        results = [check_case(case, prepared, result, stack, binding, args.output_dir / case) for case in cases]
    passed = not results[0]["rejected"] and all(result["rejected"] for result in results[1:])
    report = {"schema": "R1ProductionPairFormatReplayV1", "passed": passed, "actual_accepted_row_count": 10,
              "healthy_recomputed": not results[0]["rejected"], "mutation_count": len(results) - 1,
              "results": results, "source_inputs": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                                       for name in ("prepared", "result", "fixture", "binding")},
              "scope": "saved_production_short_chain_real_format_and_physical_consumers; no_new_integration_or_full_qualification"}
    write(args.output_dir / "ResultV1.json", report)
    print(json.dumps({"passed": passed, "mutation_count": len(results) - 1, "output": str(args.output_dir)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
