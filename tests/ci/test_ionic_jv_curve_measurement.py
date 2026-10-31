"""Pure diagnostic receipt and summary contracts for the dev-only static job."""
import subprocess

import pytest

from scripts.ci import ionic_jv_timing as timing


def history():
    return {"controls": {"steady": {"wall_s": 30.}}, "dark_steady": {
        "wall_s": .1, "accepted": True, "termination": "converged", "error": None},
        "illumination": {"wall_s": .2, "accepted": True, "termination": "converged", "error": None},
        "curve": {"wall_s": 64.001, "attempts": {"$tuple": [{
            "index": 50, "voltage_V": 1.25, "accepted": False, "termination": "wall_budget_exhausted",
            "solver_called": True, "wall_s": .501, "error": "_WaveWallLimit",
            "solution": {"wall_s": .5005, "termination": "wall_budget_exhausted", "error": "_WallLimit",
                         "solver_controls": {"wall_s": .5}}}]}}}


def test_failed_curve_and_clipped_point_timings_remain_visible():
    result = timing.numerical_measurements(history())
    assert result["numerical_seconds"] == {"dark_steady": .1, "illumination": .2, "curve": 64.001}
    point, = result["point_timings"]
    assert point["index"] == 50 and point["wall_s"] == .501 and point["solver_wall_seconds"] == .5005
    assert point["effective_point_wall_seconds"] == .5 and point["wall_censored"] is True
    assert result["censoring"] == {"whole_job_wall": False, "numerical_wall": True,
        "events": [{"phase": "curve", "index": 50, "limit": "curve_wall_s", "reason": "wall_budget_exhausted"}],
        "unobserved_phases": []}


def test_initialization_failure_preserves_unstarted_curve():
    data = history()
    data["illumination"].update(wall_s=25.01, accepted=False, termination="wall_budget_exhausted")
    data["curve"] = None
    result = timing.numerical_measurements(data)
    assert result["numerical_seconds"] == {"dark_steady": .1, "illumination": 25.01, "curve": None}
    assert result["point_timings"] == []
    assert result["censoring"]["unobserved_phases"] == ["curve"]
    assert result["censoring"]["events"] == [
        {"phase": "illumination", "limit": "initialization_wall_s", "reason": "wall_budget_exhausted"}]


@pytest.mark.parametrize("failed", [False, True])
def test_measurement_never_recommends_a_shipping_limit_and_keeps_failures(failed):
    completed = history()
    completed["curve"]["wall_s"] = 20.
    point = completed["curve"]["attempts"]["$tuple"][0]
    point.update(accepted=True, termination="converged", error=None)
    point["solution"].update(termination="converged", error=None, solver_controls={"wall_s": 30.})
    rows = [{"attempt": index, "status": "failed" if failed and index == 3 else "passed",
             "measurement": {"measurement_kind": timing.CURVE_MEASUREMENT, "child_wall_seconds": 35.,
                             **timing.numerical_measurements(history() if failed and index == 3 else completed)}}
            for index in range(1, 4)]
    result = timing.summarize(rows)
    assert result["status"] == ("failed" if failed else "passed")
    assert result["completed"] == 3 - failed and result["requested"] == 3
    assert result["recommended_wall_seconds"] is None
    assert result["recommendation_status"] == "measurement_only"
    assert "allowance_sufficient" not in result
    assert result["observed_numerical_seconds"]["curve"]["observed"] == 3
    assert result["observed_numerical_seconds"]["curve"]["max"] == (64.001 if failed else 20.)



def test_static_harness_imports_without_service_packages():
    code = """
import importlib.abc
import runpy
import sys

class DenyServices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fastapi", "solarlab_server"}:
            raise ImportError("static tests must not import " + fullname)

sys.meta_path.insert(0, DenyServices())
loaded = runpy.run_path(sys.argv[1])
measurement = loaded["timing"].numerical_measurements(loaded["history"]())
assert measurement["censoring"]["numerical_wall"] is True
assert not any(name.split(".")[0] in {"fastapi", "solarlab_server"} for name in sys.modules)
"""
    subprocess.run([timing.sys.executable, "-c", code, __file__], cwd=timing.ROOT, check=True)
