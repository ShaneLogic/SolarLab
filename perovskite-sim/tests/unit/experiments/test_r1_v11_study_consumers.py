"""A target-DC prerequisite can be collected without starting a long window."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("frequencies", [("nan",), ("-1",), ("1", "1"), ("2", "1")])
def test_invalid_ac_axis_is_rejected_before_a_study_starts(tmp_path, monkeypatch, frequencies):
    path = Path(__file__).resolve().parents[3] / "scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_v11_ac_axis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Study", lambda *a: pytest.fail("invalid axis reached a physical study"))
    with pytest.raises(SystemExit) as exc:
        module.main(["--output-dir", str(tmp_path / "absent"), "--section", "ac",
                     "--frequencies-hz", *frequencies])
    assert exc.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_target_dc_plan_has_only_the_selected_endpoint_and_preparation(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_v11_study_consumers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("prepare_common_state", "run_r1_step", "solve_controlled_dc"):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("planning started physical work"))
    study = module.Study.__new__(module.Study)
    study.args = SimpleNamespace(case_filter="", case=["TargetDC/N256/D/A0.005"], window_amplitude=.005)
    study.grids, study.controls = (16, 64, 128, 256), tuple("ABCD")
    study.request = {"source": "frozen", "window_amplitude_V": .005, "linearity_case": None}
    plan = study.make_plan(("target-dc",))
    assert set(plan["cases"]) == {"Preparation/N256", "TargetDC/N256/D/A0.005"}
    assert plan["cases"]["TargetDC/N256/D/A0.005"] == {
        "intervals": 256, "control": "D", "voltage_V": .005,
        "scope": "sealed_same_control_dc_state_for_derived_analysis"}
    with pytest.raises(ValueError, match="not defined"):
        study.make_plan(("target-dc",), {"case_filter": "", "cases": ["Window/Base/N256/A0.005"]})
