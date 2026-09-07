"""Optical probe orchestration, not a numerical-physics certification."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "scripts/probe_calado_optical_inputs.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("calado_optical_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def request_body():
    return {"kind": "jv", "device": {"device": {"Phi": 2.5e22}, "layers": [
        {"role": "absorber", "alpha": 1e5, "P0": 0, "D_ion": 0},
    ]}, "params": {"waveform": {"uniform_generation_rate_m3_s": None}}}


def test_optical_variations_keep_source_and_ions_unchanged(probe, request_body):
    before = copy.deepcopy(request_body)
    cases = probe.optical_cases(request_body)
    assert request_body == before
    assert len(cases) == 7
    assert cases["alpha_zero"]["device"]["layers"][0]["alpha"] == 0
    assert cases["flux_zero"]["device"]["device"]["Phi"] == 0
    assert cases["uniform_alpha_zero"]["params"]["waveform"]["uniform_generation_rate_m3_s"] == 2.5e27
    for request in cases.values():
        assert request["device"]["layers"][0]["P0"] == 0
        assert request["device"]["layers"][0]["D_ion"] == 0


def test_common_tolerance_override_changes_only_numerical_controls(probe, request_body):
    request_body["params"]["waveform_controls"] = {"rtol": 1e-4, "atol_m3": 1.0}
    before = copy.deepcopy(request_body)
    original_cases = probe.optical_cases(request_body)
    changed_cases = probe.optical_cases(request_body, atol_m3=100.0)
    assert request_body == before
    for name, changed in changed_cases.items():
        assert changed["params"]["waveform_controls"] == {"rtol": 1e-4, "atol_m3": 100.0}
        changed["params"]["waveform_controls"]["atol_m3"] = 1.0
        assert changed == original_cases[name]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_tolerance_override_is_rejected(probe, request_body, value):
    with pytest.raises(ValueError, match="atol_m3"):
        probe.optical_cases(request_body, atol_m3=value)


def test_failed_rows_do_not_skip_remaining_cases_or_pass_matrix(probe, request_body, monkeypatch, tmp_path):
    source, output = tmp_path / "source.json", tmp_path / "matrix"
    probe.write_json(source, {"request": request_body})
    monkeypatch.setattr("sys.argv", ["probe", "--source", str(source), "--out-dir", str(output)])
    submitted = []
    def submit(request):
        submitted.append(request)
        return {"job_id": str(len(submitted))}
    def status(job_id):
        if job_id in {"2", "4"}:
            return probe.JobStatus.ERROR, None, "Retained solver failure"
        return probe.JobStatus.DONE, {}, None
    monkeypatch.setattr(probe, "start_job", submit)
    monkeypatch.setattr(probe, "_JOB_REGISTRY", SimpleNamespace(next_event=lambda *a, **k: None, status=status))
    monkeypatch.setattr(probe, "audit_record", lambda path: (
        {"J_sc_fwd_A_m2": 1.0, "generation_budget_A_m2": 1.0}, json.loads(path.read_text())))
    with pytest.raises(RuntimeError, match="2/7 optical cases failed"):
        probe.main()
    report = json.loads((output / "checks.json").read_text())
    assert len(submitted) == 7
    assert report["status"] == "failed"
    assert set(report["failed_cases"]) == {"alpha_zero", "flux_zero"}
    assert len(report["completed_cases"]) == 5
    assert (output / "uniform_alpha_zero.json").is_file()
    assert not (output / "device_optics.png").exists()
