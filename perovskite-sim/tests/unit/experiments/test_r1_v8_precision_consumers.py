"""Selection and intervention semantics; real numerical matrix is a CLI run."""
import json
from types import SimpleNamespace

import pytest

from scripts import check_r1_v8_precision_consumers as consumers


def test_fixed_case_matrix_and_directions_are_not_selected_after_faults():
    assert consumers.SELECTIONS == {"ordinary": (4, 2.5e-10), "late": (4, 1.9663570500354033),
                                   "single_path": (4, 0.7173200981104755)}
    assert len(consumers.FAULTS) == 5
    assert consumers.PATH_PHI_PERTURBATION_V == 1e-13
    assert consumers.HEALTHY_PHI_PERTURBATION_V == 1e-23


def test_intervention_is_restored_even_when_a_real_call_raises():
    owner = SimpleNamespace(hook=lambda: "healthy")
    with pytest.raises(RuntimeError):
        with consumers.replace_method(owner, "hook", lambda: "fault"):
            assert owner.hook() == "fault"
            raise RuntimeError("test")
    assert owner.hook() == "healthy"


def test_selection_requires_an_actual_unique_same_tier_predecessor(tmp_path):
    rows = []
    for _, (tier, time) in consumers.SELECTIONS.items():
        rows.extend([{"substeps": tier, "time_s": time / 2}, {"substeps": tier, "time_s": time}])
    path = tmp_path / "AcceptedStepsV1.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert set(consumers.select_rows(tmp_path)) == set(consumers.SELECTIONS)
    rows.append(rows[1])
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="absent or ambiguous"):
        consumers.select_rows(tmp_path)


def test_matching_analysis_scripts_do_not_authorize_a_changed_production_kernel(tmp_path):
    receipt = {"all_tracked_files": {"perovskite-sim/perovskite_sim/physics/compensated.py": "original"}}
    (tmp_path/"SourceReceiptV1.json").write_text(json.dumps(receipt))
    source = {"commit": "same", "git_status": "", "source_files": {
        "perovskite_sim/physics/compensated.py": "modified"}}
    assert not consumers.execution_source_matches(tmp_path, {"source_commit": "same"}, source)["passed"]
    source["source_files"]["perovskite_sim/physics/compensated.py"] = "original"
    assert consumers.execution_source_matches(tmp_path, {"source_commit": "same"}, source)["passed"]


def test_calibration_cannot_move_to_a_different_physical_state():
    calibration = {"schema": "R1V8PathFaultCalibrationV1", "amplitude_V": 1e-13, "signs": [-1,1],
                   "selection": {"substeps": 4, "time_s": consumers.SELECTIONS["single_path"][1], "node": 7}}
    consumers.validate_case_contract({}, calibration, "digest", historical=True)
    calibration["selection"]["node"] = 8
    with pytest.raises(ValueError, match="calibration selection"):
        consumers.validate_case_contract({}, calibration, "digest", historical=True)
