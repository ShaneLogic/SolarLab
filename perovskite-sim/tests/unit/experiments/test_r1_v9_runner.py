"""The production validation entry preserves scope, failures and all evidence costs."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_r1_v9_prototype as runner


@pytest.fixture
def contract(tmp_path,monkeypatch):
    times=[float(i) for i in range(114)]
    monkeypatch.setattr(runner,"TIMES_SHA256",hashlib.sha256(runner.canonical(times).encode()).hexdigest())
    data={"schema":runner.SCHEMA,"scope":runner.SCOPE,"case":{**runner.CASE_AXES,"times_s":times},
        "inputs":runner.INPUTS,"certificate_limits":runner.LIMITS,"engineering_limits":runner.ENGINEERING_LIMITS,
        "absolute_engineering_limits":runner.ABSOLUTE_LIMITS,"execution":{
            "baseline_repeats":3,"compensated_repeats":1,"paired_baseline_ordinal":3,
            "time_comparator":"minimum_elapsed_of_all_three_complete_same_source_baselines",
            "production_backend_explicit":True,"production_monkeypatch_forbidden":True,
            "expected_total_rows":794,"expected_rows_by_substeps":{"1":114,"2":227,"4":453},
            "single_thread":True,"requires_clean_full_commit":True,"resume_from_old_state":False,
            "new_profile_runs":0,"new_ablation_runs":0,"new_100s_runs":0},
        "scientifically_accepted":False,"formal_qualification":False,"P2_qualified":False}
    return data


@pytest.mark.parametrize("section,key,value",[
    ("case","intervals",256),("case","time_substeps",[4,8,16]),
    ("execution","baseline_repeats",2),("execution","new_100s_runs",1),
    ("execution","production_monkeypatch_forbidden",False),
    ("absolute_engineering_limits","elapsed_s",400),
])
def test_resealed_scope_or_budget_change_is_rejected(tmp_path,contract,section,key,value):
    data=deepcopy(contract);data[section][key]=value
    p=tmp_path/"request.json";runner.write(p,data)
    with pytest.raises(ValueError):
        runner.load_contract(p,runner.sha(p),tmp_path/"result")


def test_contract_accepts_exact_external_identity_and_rejects_changed_bytes(tmp_path,contract):
    p=tmp_path/"request.json";runner.write(p,contract);digest=runner.sha(p)
    assert runner.load_contract(p,digest,tmp_path/"result")[0]==contract
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="external"):
        runner.load_contract(p,digest,tmp_path/"result")


def baselines():
    return [{"schema":"R1V9ProductionValidationRunV1","mode":"baseline","baseline_ordinal":i,
        "request_sha256":"request","source_commit":"source","source_content_sha256":"content",
        "runtime_identity":{"python":"same"},"source_unchanged":True,
        "extent":{"complete":True},"numeric_sidecar_exact":True,
        "replay_completed":True,"module_function_identity":{"unchanged":True},
        "four_predicates":{"observer_matches_result":True},"four_predicates_passed":True,
        "certificate_check":{"limits_match_contract":True},"execution_status":"completed","failure":None,
        "cost":{"elapsed_s":t,"peak_rss_bytes":100,"bytes_per_row":50}}
        for i,t in enumerate((9.,10.,11.),1)]


def test_conservative_comparison_uses_fastest_complete_baseline():
    before=baselines()
    current={"source_commit":"source","source_content_sha256":"content","runtime_identity":{"python":"same"},
        "extent":{"complete":True},"cost":{"elapsed_s":45,"peak_rss_bytes":200,"bytes_per_row":150}}
    report=runner.comparison(current,before,"request")
    assert report["qualified"] and report["ratios"]==runner.ENGINEERING_LIMITS
    current["cost"]["elapsed_s"]=45.01
    assert not runner.comparison(current,before,"request")["qualified"]
    current["cost"]["elapsed_s"]=44
    before[0]["extent"]["complete"]=False
    assert not runner.comparison(current,before,"request")["qualified"]
    assert not runner.comparison(current,before[:2],"request")["qualified"]


def test_forged_or_reordered_baseline_and_wrong_runtime_are_rejected():
    bs=baselines()
    current={**bs[0],"mode":"compensated"}
    assert not runner.comparison(current,list(reversed(bs)),"request")["qualified"]
    bs[1]["runtime_identity"]={"python":"different"}
    assert not runner.comparison(current,bs,"request")["qualified"]


def test_float_reference_keeps_known_final_failure_but_no_other_failure():
    reference=baselines()[0]
    reference.update(execution_status="failed",four_predicates_passed=False,
        original_metric_failures=["eliminated_operator_error"],failure={"type":"R1RunError",
        "message":"R1 controlled-step certificate failed: eliminated_operator_error"})
    assert runner.baseline_readiness(reference)["ready"]
    reference["original_metric_failures"].append("nonlinear_residual")
    assert not runner.baseline_readiness(reference)["ready"]
    reference["original_metric_failures"]=["eliminated_operator_error"]
    reference["replay_completed"]=False
    assert not runner.baseline_readiness(reference)["ready"]


def test_sidecar_and_whole_case_are_counted_and_manifest_reaches_fixed_point(tmp_path):
    (tmp_path/"ResultV1.json").write_bytes(b"{}")
    (tmp_path/"StateArraysV1.npz").write_bytes(b"123456")
    summary={"cost":{"elapsed_s":1,"peak_rss_bytes":1,"accepted_row_payload_bytes":6}}
    runner.finalize_manifest(tmp_path,summary)
    saved=json.loads((tmp_path/"SummaryV1.json").read_text())
    assert saved["cost"]["case_artifact_bytes"]==sum(p.stat().st_size for p in tmp_path.iterdir())
    assert saved["absolute_engineering_check"]["passed"]
    for name,record in json.loads((tmp_path/"ManifestV1.json").read_text()).items():
        assert runner.sha(tmp_path/name)==record["sha256"]


def test_preparation_failure_retains_numerical_witness_without_claiming_complete(tmp_path,contract):
    def fail():
        error=RuntimeError("failed fine initialization")
        error.result={"actual_state":{"precision_phi_V_hi":[1.],"precision_phi_V_lo":[1e-20]}}
        raise error
    report=runner.run_trajectory(tmp_path,contract["case"],"compensated",
        {"prepare":fail,"json_data":lambda value:value},lambda:None)
    assert not report["four_predicates_passed"]
    assert report["extent"]["accepted_rows"]==0
    assert (tmp_path/"FailureV1.json").exists()
    assert (tmp_path/"PhaseFailureResultV1.json").exists()


def test_sidecar_failure_keeps_actual_accepted_prefix_and_result(tmp_path,contract):
    row={"substeps":1,"time_s":0.,"phase":"0+","state":{}}
    result={"accepted_steps":[row],"certificate":{"certified":False,"limits":runner.LIMITS,
            "metrics":{name:0. for name in runner.LIMITS}}}
    def run(prepared,observer):
        observer(row)
        return result
    def fail(*args):
        raise OSError("controlled artifact failure")
    api={"prepare":lambda:SimpleNamespace(to_dict=lambda:{"seed":"actual"}),
         "json_data":lambda value:value,"run":run,"persist_numeric":fail}
    report=runner.run_trajectory(tmp_path,contract["case"],"compensated",api,lambda:None)
    assert report["extent"]["accepted_rows"]==1 and not report["four_predicates_passed"]
    assert json.loads((tmp_path/"ResultV1.json").read_text())==result
    assert len((tmp_path/"AcceptedStepsV1.jsonl").read_text().splitlines())==1
    assert report["phase_timing"]["exclusive_sum_s"]<=report["cost"]["elapsed_s"]


def test_raw_array_owner_released_before_readback_and_replay(tmp_path, contract):
    import weakref
    class RawResult(dict):
        pass
    raw = RawResult(accepted_steps=[], certificate={"certified": False,
        "limits":runner.LIMITS,"metrics":{name:0. for name in runner.LIMITS}})
    reference = weakref.ref(raw)
    pending = [raw]
    del raw
    stages = []
    def persist(path, value):
        assert value is reference() and isinstance(value,RawResult)
        path.write_bytes(b"numeric-sidecar")
        stages.append("persist")
    def readback(path, value):
        assert reference() is None and type(value) is dict
        assert path.read_bytes()==b"numeric-sidecar"
        stages.append("readback")
    def replay(prepared,value,incomplete,observer):
        assert reference() is None and value["certificate"]["limits"]==runner.LIMITS
        observer({"row_index":0,"elapsed_s":0.})
        stages.append("replay")
        return {"certified":False}
    report=runner.run_trajectory(tmp_path,contract["case"],"baseline",{
        "prepare":lambda:SimpleNamespace(to_dict=lambda:{"prepared":True}),
        "run":lambda prepared,observer:pending.pop(),
        "json_data":lambda value:json.loads(json.dumps(value)),
        "persist_numeric":persist,"verify_numeric":readback,"replay":replay},lambda:None)
    assert stages==["persist","readback","replay"]
    assert report["numeric_sidecar_exact"] and report["replay_observed_rows"]==1


def test_streaming_readback_preserves_content_and_rejects_invalid_rows(tmp_path):
    path=tmp_path/"rows.jsonl"
    for text in ('', '{"x":-0.0}\n{"x":[1,2]}\n', '{"x":1}\r\n{"x":2}'):
        path.write_text(text)
        assert runner.read_persisted_rows(path)==[json.loads(line) for line in path.read_text().splitlines()]
    path.write_text('{"x":1}\n\n')
    with pytest.raises(json.JSONDecodeError):
        runner.read_persisted_rows(path)
