"""Two actual lifecycle paths, reused for all publication/read integrity checks."""
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from solarlab.experiments.transient_results import (
    _decode, _encode, publish_transient_result, read_transient_result, run_transient_for_publication,
)
from solarlab.io.artifacts import ArtifactIntegrityError, ArtifactPayload, _json_document, array_artifact, bytes_artifact
from solarlab.io.run_store import RunStore


def words(value):
    """Independent native-word witness; no production serialization calls."""
    if is_dataclass(value):
        return {field.name: words(getattr(value, field.name)) for field in fields(value)}
    if type(value) is np.ndarray:
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "word_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return {"scalar_dtype": value.dtype.str, "words": value.tobytes().hex()}
    if isinstance(value, Enum):
        return words(value.value)
    if type(value) is float:
        return {"binary64": struct.pack("<d", value).hex()}
    if type(value) is dict:
        return {name: words(item) for name, item in value.items()}
    if type(value) in (tuple, list):
        return {type(value).__name__: [words(item) for item in value]}
    return value


def expected(result, nodes):
    arrays = {"requested_times_s": result.protocol.times_s, "requested_voltage_V": result.protocol.voltage_V,
        "requested_generation_scale": result.protocol.generation_scale, "accepted_times_s": result.accepted_times_s,
        "nodes_m": nodes}
    arrays.update({name: np.stack([getattr(s, name) for s in result.states])
        for name in ("potential_V", "electron_density_m3", "hole_density_m3")})
    for prefix in ("", "terminal_"):
        for kind in ("conduction", "displacement", "total"):
            name = f"{prefix}{kind}_current_density_A_m2"
            arrays[name] = np.stack([a.accounting[name] for a in result.attempts if a.accepted])
    diagnostics = {"initial": result.initial, "attempts": result.attempts,
        "accepted_state_coordinates": tuple(state.u for state in result.states),
        "accepted_state_voltage_V": tuple(state.voltage_V for state in result.states),
        "state_device_identities": tuple(state.device_sha256 for state in result.states),
        "solver_controls": result.solver_controls, "initial_controls": result.initial_controls, "wall_s": result.wall_s}
    return {"arrays": words(arrays), "diagnostics": words(diagnostics)}


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = Path(os.environ.get("SOLARLAB_TRANSIENT_RESULT_CASE", str(tmp_path_factory.mktemp("transient-results")))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    witness = root / "Calculation.json"
    if witness.exists():
        return root, json.loads(witness.read_text())  # Local reruns reuse both saved calculations.
    started = root / "calculation.started"
    if started.exists():
        pytest.fail("previous calculation did not finish publication; inspect it before any recomputation")
    started.touch()
    from examples.research_1d.steady import load_device
    from examples.research_1d.transient import controls
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.solvers.nonlinear.problem import SolveBudget
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
    from support.provenance import checked_provenance_owner
    from threadpoolctl import threadpool_limits

    output = {"calculations": 2, "shared_initial_dc_solves": 1, "cases": {}}
    with threadpool_limits(limits=1):
        compiled = compile_device_1d(load_device(), intervals_per_layer=12)
        initial_controls = SteadyOptions()
        initial = solve_steady_state(compiled, voltage_V=0., generation_scale=1., options=initial_controls)
        cases = (
            ("completed", TransientProtocol([0, 1e-7, 2e-7], [0, 0, 0], [1, 1.0025, 1.01]), controls()),
            ("failed-prefix", TransientProtocol([0, 1e-7, 2e-7, 3e-7], [0, 0, 0, 0], [1, 1, 1.5, 1]),
             replace(controls(), newton_budget=SolveBudget(1, 10, 1, 4))),
        )
        with RunStore(root / "store") as store:
            for name, protocol, options in cases:
                with checked_provenance_owner():
                    capture = run_transient_for_publication(compiled, initial, protocol,
                        options=options, initial_options=initial_controls)
                result = capture.result
                assert result.completed == (name == "completed")
                source_words = expected(result, compiled.nodes_m)
                row = store.enqueue(name, opaque_identity=capture.identity,
                    input_metadata={"kind": "actual bounded transient publication fixture", "capture_scope": "solve_transient_call"})
                token = store.claim(name, expected_attempt_id=row["current_attempt_id"],
                    expected_generation=row["generation"], owner_id="transient-publication")
                published = publish_transient_result(store, token, result, nodes_m=compiled.nodes_m, capture=capture)
                with pytest.raises(ValueError, match="committed terminal"):
                    read_transient_result(store, name)
                state = "succeeded" if result.completed else "failed"
                sequence = store.complete(token, state=state, result=published, artifacts=published["artifact_ids"])
                assert store.complete(token, state=state, result=published, artifacts=published["artifact_ids"]) == sequence
                output["cases"][name] = {"document": published, "expected": source_words,
                    "capture": json.loads(capture.capture_json), "terminal_state": state,
                    "requested_times_s": protocol.times_s.tolist(), "accepted_times_s": result.accepted_times_s.tolist(),
                    "accepted_attempts": [a.accepted for a in result.attempts],
                    "failure_termination": [a.termination for a in result.failed_attempts]}
                if name == "completed":
                    # The same real returned result can be published without
                    # capture, but no source/runtime history may be invented.
                    uncaptured = store.enqueue("uncaptured", opaque_identity=capture.identity,
                                               input_metadata={"kind": "same result, no supplied provenance"})
                    other = store.claim("uncaptured", expected_attempt_id=uncaptured["current_attempt_id"],
                        expected_generation=uncaptured["generation"], owner_id="transient-publication")
                    document = publish_transient_result(store, other, result, nodes_m=compiled.nodes_m)
                    store.complete(other, state="succeeded", result=document, artifacts=document["artifact_ids"])
    witness.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return root, output


@pytest.mark.parametrize("name", ["completed", "failed-prefix"])
def test_actual_histories_and_all_diagnostics_match_original_words(saved, name):
    root, witness = saved
    with RunStore(root / "store", read_only=True) as store:
        loaded = read_transient_result(store, name)
        assert words(loaded["arrays"]) == witness["cases"][name]["expected"]["arrays"]
        assert words(loaded["diagnostics"]) == witness["cases"][name]["expected"]["diagnostics"]
        assert loaded["run_state"] == witness["cases"][name]["terminal_state"]
        assert loaded["document"]["physical_certificate_status"] == "not_evaluated"
        assert len(loaded["document"]["artifact_ids"]) == 16
        assert len([e for e in store.events(run_id=name, limit=1000) if e["kind"] == "terminal"]) == 1
        for item in loaded["manifest"]["payloads"].values():
            assert store.artifact(item["artifact_id"])["phase"] == "committed"
        arrays = loaded["arrays"]
        assert arrays["conduction_current_density_A_m2"].shape == (len(arrays["accepted_times_s"]) - 1, 36)
        assert arrays["terminal_displacement_current_density_A_m2"].shape[1] == 2
        assert "no displacement current exists at the initial sample" in loaded["manifest"]["current_alignment"]
        if name == "failed-prefix":
            assert arrays["accepted_times_s"].tolist() == [0., 1e-7]
            assert arrays["requested_times_s"].tolist() == [0., 1e-7, 2e-7, 3e-7]
            assert [a["accepted"] for a in loaded["diagnostics"]["attempts"]] == [True, False]


def test_reader_does_not_import_transient_solver_or_consult_current_sources(saved, tmp_path):
    root, _ = saved
    code = '''
import importlib.abc, json, pathlib, sys
sys.path.insert(0, sys.argv[2])
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in (
            'solarlab.experiments.transient', 'solarlab.solvers', 'solarlab.equations',
            'solarlab.device.compile_1d', 'perovskite_sim')):
            raise AssertionError('reader imported a solver: ' + fullname)
sys.meta_path.insert(0, NoSolver())
from solarlab.io.run_store import RunStore
import solarlab.experiments.transient_results as reader
def no_current_source():
    raise AssertionError('reader tried to reconstruct current source history')
reader._sources = no_current_source
with RunStore(pathlib.Path(sys.argv[1]), read_only=True) as store:
    for name in ('completed', 'failed-prefix', 'uncaptured'):
        loaded = reader.read_transient_result(store, name)
        assert len(loaded['arrays']) == 14
        print(json.dumps(dict(run_id=name, state=loaded['run_state'], provenance=loaded['provenance']['status'])))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root / "store"), str(Path(read_transient_result.__code__.co_filename).resolve().parents[2])],
        cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["state"] for r in rows] == ["succeeded", "failed", "succeeded"]
    assert rows[-1]["provenance"] == "not_recorded"


def test_capture_is_limited_to_this_call_and_unknown_history_stays_unknown(saved):
    root, _ = saved
    with RunStore(root / "store", read_only=True) as store:
        captured = read_transient_result(store, "completed")["provenance"]
        description = captured["description"]
        assert description["scope"] == "solve_transient_call"
        assert all(description[name] is None for name in ("compiled_device_history", "initial_dc_history"))
        job = description["job_provenance"]["record"]
        assert description["build_source_commit"] == job["git_commit"]
        assert job["environment_lock_sha256"] and description["sources"]
        # Complete inventory membership does not enable an ionic physical law.
        assert job["assembler"]["operator"] == "BackwardEuler1DStep"
        unknown = read_transient_result(store, "uncaptured")["provenance"]
        assert unknown["status"] == "not_recorded" and "shape_checked_only" in unknown["node_grid"]
        with pytest.raises(ValueError, match="committed terminal"):
            read_transient_result(store, "completed", attempt_id="a-different-attempt")


def republish_fixture(saved, target, change, *, name="completed"):
    """Re-seal modified fixtures through real RunStore, without another solve."""
    root, _ = saved
    with RunStore(root / "store", read_only=True) as original:
        loaded = read_transient_result(original, name)
        manifest = loaded["manifest"]
        payloads = {key: ArtifactPayload(original.read_artifact(value["artifact_id"]),
            original.artifact(value["artifact_id"])["format"], _json_document(value["metadata"]))
            for key, value in manifest["payloads"].items()}
    with RunStore(target) as store:
        row = store.enqueue("fixture", opaque_identity=manifest["identity"], input_metadata={"fixture": change})
        token = store.claim("fixture", expected_attempt_id=row["current_attempt_id"],
            expected_generation=row["generation"], owner_id="resealed-fixture")
        if change in {"unit", "shape", "capture-grid"}:
            role = "nodes_m" if change == "capture-grid" else "potential_V"
            values = loaded["arrays"][role]
            if change == "shape":
                values = values.reshape(-1)
            if change == "capture-grid":
                values = values * 2
            payloads[role] = array_artifact(values, unit="wrong-unit" if change == "unit" else manifest["payloads"][role]["metadata"]["unit"],
                metadata={"schema": manifest["summary"]["schema"], "compiled_identity": manifest["summary"]["compiled_identity"]})
        if change == "nonfinite-failure":
            diagnostics = _decode(json.loads(payloads["transient_diagnostics"].data))
            failed = diagnostics["attempts"][-1]
            failed["candidate_coordinates"] = failed["candidate_coordinates"].copy()
            failed["candidate_coordinates"][0] = np.frombuffer(bytes.fromhex("420000000000f87f"), dtype="<f8")[0]
            failed["newton"]["scaled_residual"] = float("-inf")
            payloads["transient_diagnostics"] = bytes_artifact(_json_document(_encode(diagnostics)).encode(),
                metadata=manifest["payloads"]["transient_diagnostics"]["metadata"])
        records = {key: store.publish(token, key, payload) for key, payload in payloads.items()}
        manifest.update(store_id=store.store_id, run_id="fixture", attempt_id=token.attempt_id, generation=token.generation,
            payloads={key: {"artifact_id": item["artifact_id"], "sha256": item["sha256"], "metadata": item["metadata"]}
                      for key, item in records.items()})
        if change != "capture-grid":
            manifest["provenance"] = {"status": "not_recorded", "reason": "re-sealed negative fixture, no computation claim"}
        if change == "false-completion":
            manifest["summary"].update(outcome="completed", reason=None)
        if change == "attempt":
            manifest["attempt_id"] = "foreign-attempt"
        raw = _json_document(manifest).encode()
        record = store.publish(token, "transient_manifest", bytes_artifact(raw, metadata={
            "schema": manifest["summary"]["schema"], "compiled_identity": manifest["summary"]["compiled_identity"], "role": "result_manifest"}))
        document = {**manifest["summary"], "store_id": store.store_id, "run_id": "fixture", "attempt_id": token.attempt_id,
            "generation": token.generation, "manifest_artifact_id": record["artifact_id"],
            "manifest_sha256": "0" * 64 if change == "manifest-hash" else hashlib.sha256(raw).hexdigest(),
            "artifact_ids": [*[r["artifact_id"] for r in records.values()], record["artifact_id"]]}
        store.complete(token, state="failed" if name == "failed-prefix" else "succeeded", result=document, artifacts=document["artifact_ids"])
        if change == "terminal-reference":
            terminal = store.attempts("fixture")[-1]["terminal"]
            terminal["artifacts"][0]["manifest_sha256"] = "0" * 64
            store._db.execute("UPDATE attempts SET terminal_json=? WHERE attempt_id=?",
                              (_json_document(terminal), token.attempt_id))


@pytest.mark.parametrize("change", ["unit", "shape", "attempt", "manifest-hash", "capture-grid", "false-completion", "terminal-reference"])
def test_resealed_metadata_or_identity_conflicts_are_rejected(saved, tmp_path, change):
    root = tmp_path / "damaged"
    republish_fixture(saved, root, change, name="failed-prefix" if change == "false-completion" else "completed")
    with RunStore(root, read_only=True) as store, pytest.raises(ValueError):
        read_transient_result(store, "fixture")


def test_nonfinite_failed_candidate_words_remain_failure_data(saved, tmp_path):
    root = tmp_path / "nonfinite"
    republish_fixture(saved, root, "nonfinite-failure", name="failed-prefix")
    with RunStore(root, read_only=True) as store:
        loaded = read_transient_result(store, "fixture")
        failed = loaded["diagnostics"]["attempts"][-1]
        assert failed["candidate_coordinates"][:1].tobytes().hex() == "420000000000f87f"
        assert failed["newton"]["scaled_residual"] == float("-inf")
        assert failed["accepted"] is False and loaded["document"]["outcome"] == "partial"
        assert all(np.all(np.isfinite(array)) for array in loaded["arrays"].values())


def test_corrupt_payload_never_returns_arrays(saved, tmp_path):
    root = tmp_path / "corrupt"
    republish_fixture(saved, root, "unchanged")
    with RunStore(root, read_only=True) as store:
        loaded = read_transient_result(store, "fixture")
        identifier = loaded["manifest"]["payloads"]["potential_V"]["artifact_id"]
        path = root / store.artifact(identifier)["relative_path"]
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with pytest.raises(ArtifactIntegrityError):
            read_transient_result(store, "fixture")


@pytest.mark.parametrize("name", ["completed", "failed-prefix"])
def test_source_edit_does_not_break_saved_readback_and_strict_still_refuses(saved, tmp_path, monkeypatch, name):
    from solarlab.reproducibility import job_provenance as provenance
    root, witness = saved
    source = tmp_path / "later_source.py"
    source.write_bytes(b"refactored implementation")
    with RunStore(root / "store", read_only=True) as store:
        original = read_transient_result(store, name)
        recorded = provenance.read_provenance(original["provenance"]["description"]["job_provenance"])
        arrays_before = words(original["arrays"])
        producer_before = recorded.canonical_json
        current = provenance.capture_job_provenance({}, dirty_policy="record", operator="BackwardEuler1DStep",
            git_commit=recorded.export()["git_commit"], source_files=(("solarlab/later_source.py", source.read_bytes()),),
            source_attested=True, environment_lock=b"different locked environment")
        # A real source edit cannot trigger a read-time file inspection or solve.
        source.write_bytes(b"another source edit after capture")
        monkeypatch.setattr(provenance, "capture_job_provenance", lambda *a, **k: pytest.fail("reader captured current source"))
        result = read_transient_result(store, name, current_identity=current)
        assert result["provenance_comparison"]["status"] == "different"
        assert words(result["arrays"]) == arrays_before == witness["cases"][name]["expected"]["arrays"]
        assert provenance.read_provenance(result["provenance"]["description"]["job_provenance"]).canonical_json == producer_before
        assert result["run_state"] == witness["cases"][name]["terminal_state"]
        with pytest.raises(ValueError, match="strict_same_source"):
            read_transient_result(store, name, current_identity=current, strict_same_source=True)
