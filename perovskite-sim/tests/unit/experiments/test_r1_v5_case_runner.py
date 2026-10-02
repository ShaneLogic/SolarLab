"""Fast request validation; actual device cases run only from a frozen request."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_explicit_short_cases_preserve_axes_and_reject_scope_or_anchor_changes(tmp_path):
    script = Path(__file__).resolve().parents[3] / "scripts/run_r1_v5_cases.py"
    spec = importlib.util.spec_from_file_location("v5_case_runner_validation", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    snapshot = runner.source_files()
    assert snapshot["perovskite-sim/scripts/run_r1_v5_cases.py"] == hashlib.sha256(script.read_bytes()).hexdigest()
    case = {"control": "A", "intervals": 64, "nonlinear_factor": .01,
            "time_substeps": [2, 4, 8], "amplitude_V": .005, "times_s": [0., 1e-9, 1e-8, 1e-6, 1e-4]}
    request = {"schema": "R1V5CaseRequestV1", "cases": [case,
        {**case, "control": "D", "intervals": 128, "time_substeps": [4, 8, 16]},
        {**case, "control": "D", "intervals": 256, "nonlinear_factor": .1, "time_substeps": [4, 8, 16]}]}
    path, output = tmp_path / "CallerRequestV1.json", tmp_path / "Results"
    def save(value):
        path.write_text(json.dumps(value, allow_nan=False) + "\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    anchor = save(request)
    actual, raw = runner.load_request(path, anchor, output)
    assert actual == request and raw == path.read_bytes()
    assert [runner.case_name(value) for value in actual["cases"]] == [
        "A_N64_F0p01_T2", "D_N128_F0p01_T4", "D_N256_F0p1_T4"]
    with pytest.raises(ValueError, match="external digest"):
        runner.load_request(path, "0" * 64, output)
    with pytest.raises(ValueError, match="outside the result"):
        runner.load_request(path, anchor, tmp_path)
    # The frozen study allows these only after a necessary time comparison
    # fails. Parsing supports them without adding them to any request.
    for ladder in ([8, 16, 32], [16, 32, 64]):
        extended = {"schema": "R1V5CaseRequestV1", "cases": [{**case, "time_substeps": ladder}]}
        assert runner.load_request(path, save(extended), output)[0] == extended
    for field, value in [("control", "E"), ("intervals", 64.), ("nonlinear_factor", True),
                         ("time_substeps", [1, 3, 9]), ("times_s", [0., 1e-9, 100.]),
                         ("amplitude_V", .01)]:
        wrong = deepcopy(request)
        wrong["cases"][0][field] = value
        with pytest.raises(ValueError):
            runner.load_request(path, save(wrong), output)
    beyond_limit = {"schema": "R1V5CaseRequestV1", "cases": [{**case, "time_substeps": [32, 64, 128]}]}
    with pytest.raises(ValueError, match="time-substep"):
        runner.load_request(path, save(beyond_limit), output)
    for cases in ([], [case, case], [case] * 33):
        wrong = {"schema": "R1V5CaseRequestV1", "cases": cases}
        with pytest.raises(ValueError):
            runner.load_request(path, save(wrong), output)
    assert not output.exists(), "parameter validation must never produce a physical run"
