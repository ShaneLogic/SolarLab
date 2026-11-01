"""PV metric semantics and source-bound readback; never repeat native curves."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.research_1d import ionic_pv as consumer
from solarlab.experiments.jv import ionic_steady as driver
from solarlab.experiments.jv.steady import steady_jv_metrics
from solarlab.experiments.transient_results import _encode


def summary_input(voltage, current, *, incident=None, failed=False):
    """Synthetic display-only data; never represented as saved solver evidence."""
    v, j = np.asarray(voltage, dtype=float), np.asarray(current, dtype=float)
    attempts = [{"index":i,"voltage_V":float(value),"accepted":True,"solution":None}
                for i,value in enumerate(v)]
    requested = v
    if failed:
        requested = np.r_[v, v[-1] + 0.1]
        attempts.append({"index":len(v),"voltage_V":float(requested[-1]),"accepted":False,
            "termination":"synthetic_failure","error":None,"solver_called":False,"solution":None})
    return {"completed":not failed,"result":{"voltage_V":v,"output_current_A_m2":j,
        "requested_voltage_V":requested,"attempts":attempts,
        "metrics":asdict(steady_jv_metrics(v,j,incident_power_W_m2=incident)),
        "incident_power_W_m2":incident,"solver_calls":0,"evaluation_counts":{},"wall_s":0.0,
        "generation_scale":1.0,"compiled_identity":"synthetic_display_only",
        "source_identity":"synthetic","controls_sha256":"synthetic","request_sha256":"synthetic",
        "physical_certificate_status":"not_evaluated"}}


def test_positive_endpoint_establishes_jsc_but_not_voc_ff_or_efficiency():
    value=consumer.pv_summary(summary_input([0.0,0.2],[10.0,9.0]))
    assert value["sampled_short_circuit_current_A_m2"]==10.0
    assert not value["open_circuit_bracket_established"]
    assert value["open_circuit_bracket"] is None
    for name in ("open_circuit_voltage_V","maximum_power_W_m2","fill_factor","efficiency"):
        assert value["metrics"][name] is None
    assert value["efficiency_status"]=="unknown_incident_power"


@pytest.mark.parametrize("reverse",[False,True])
def test_interpolated_crossing_and_power_between_samples_are_labeled(reverse):
    v,j=[0.0,0.8,1.2],[10.0,2.0,-2.0]
    if reverse:
        v,j=v[::-1],j[::-1]
    value=consumer.pv_summary(summary_input(v,j))
    assert value["open_circuit_bracket"]["voltage_V"]==[0.8,1.2]
    assert value["metrics"]["open_circuit_voltage_V"]==pytest.approx(1.0)
    # I=10(1-V), so the linear-segment power peak is at V=0.5, not a supplied node.
    assert value["metrics"]["maximum_power_W_m2"]==pytest.approx(2.5)
    assert value["metrics"]["fill_factor"]==pytest.approx(0.25)
    assert "piecewise-linear" in value["metric_basis"]
    assert value["voltage_grid_qualification"]=="not_evaluated"
    assert value["physical_certificate_status"]=="not_evaluated"


def test_generation_one_does_not_imply_irradiance():
    absent=consumer.pv_summary(summary_input([0.0,1.0],[10.0,0.0]))
    declared=consumer.pv_summary(summary_input([0.0,1.0],[10.0,0.0],incident=100.0))
    assert absent["generation_scale"]==declared["generation_scale"]==1.0
    assert absent["metrics"]["efficiency"] is None
    assert declared["metrics"]["efficiency"]==pytest.approx(0.025)


def test_no_zero_voltage_observation_prevents_invented_pv_bracket():
    value=consumer.pv_summary(summary_input([0.8,1.2],[2.0,-2.0]))
    assert value["sampled_short_circuit_current_A_m2"] is None
    assert not value["open_circuit_bracket_established"]
    assert value["metrics"]["limitation"]=="zero_voltage_not_bracketed"


def test_failed_protocol_does_not_erase_an_observed_accepted_bracket():
    value=consumer.pv_summary(summary_input([0.0,0.8,1.2],[10.0,2.0,-2.0],failed=True))
    assert not value["completed"] and value["first_failure"] is not None
    assert value["open_circuit_bracket_established"]
    assert value["metrics"]["open_circuit_voltage_V"]==pytest.approx(1.0)


@pytest.fixture(scope="module")
def native_records():
    parent_path=os.environ.get("SOLARLAB_IONIC_PV_RESULT")
    fine_path=os.environ.get("SOLARLAB_IONIC_PV_CONTINUATION_RESULT")
    if not parent_path or not fine_path:
        pytest.skip("bind archived raw curves explicitly; these tests never rerun native physics")
    parent_sha=os.environ["SOLARLAB_IONIC_PV_SHA256"]
    fine_sha=os.environ["SOLARLAB_IONIC_PV_CONTINUATION_SHA256"]
    parent=driver.read_ionic_steady_jv(parent_path,parent_sha)
    fine=driver.read_ionic_steady_jv(fine_path,fine_sha)
    return parent,fine,Path(parent_path),parent_sha,Path(fine_path),fine_sha


def test_original_first_failure_and_complete_requested_grid_remain(native_records):
    parent=native_records[0]
    r=parent["result"]
    assert not parent["completed"]
    np.testing.assert_array_equal(r["requested_voltage_V"],consumer.COARSE_PROBE_VOLTAGES_V)
    np.testing.assert_array_equal(r["voltage_V"],[0.0,0.01,0.025,0.05,0.1,0.2])
    assert len(r["attempts"])==r["solver_calls"]==7
    failed=r["attempts"][-1]
    assert failed["voltage_V"]==0.3 and not failed["accepted"]
    solution=failed["solution"]
    assert solution["newton"]["termination"]=="budget_exhausted"
    assert len(solution["newton"]["iterations"])==60
    assert solution["candidate_coordinates"].tobytes()==solution["newton"]["state"].tobytes()
    assert solution["diagnostics"]["original_residual"].shape==(137,)
    checks={x["name"]:x for x in solution["checks"]}
    assert checks["carrier_balance_A_m2"]["value"]>checks["carrier_balance_A_m2"]["limit"]


def test_continuation_reuses_exact_parent_state_and_actual_controls(native_records):
    parent,fine,_,parent_sha,_,_=native_records
    point=parent["result"]["attempts"][5]["solution"]
    r=fine["result"]
    assert fine["completed"] and r["solver_calls"]==40
    np.testing.assert_array_equal(r["requested_voltage_V"],consumer.CONTINUATION_VOLTAGES_V)
    initial=r["initial_provenance"]
    raw=Path(initial["source_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==initial["source_sha256"]
    seed=json.loads(raw)
    assert seed["summary"]["parent_curve_sha256"]==parent_sha
    assert seed["summary"]["parent_attempt_index"]==5
    assert _encode(initial["state"])==_encode(point["state"])
    assert seed["result"]["solver_controls"]==point["solver_controls"]==r["solver_controls"]["steady"]
    assert 0<r["solver_controls"]["steady"]["wall_s"]<30.0
    assert r["solver_controls"]["wall_s"]==25.0


def test_all_accepted_currents_species_and_original_balances_are_retained(native_records):
    for record in native_records[:2]:
        r=record["result"]
        for attempt in r["attempts"]:
            if not attempt["accepted"]:
                continue
            point=attempt["solution"]
            d=point["diagnostics"]
            assert all(g["passed"] for g in point["checks"])
            assert d["total_current_density_A_m2"].shape==(38,)
            assert np.ptp(d["total_current_density_A_m2"])<=1e-8
            for species in r["ion_species"]:
                density=point["state"][species+"_ion_density_m3"]
                info=d["ion_species"][species]
                assert math.fsum((r["ion_weights_m"]*density).tolist())==info["inventory_m2"]
                assert info["inventory_relative_error"]<=1e-12
                assert info["ion_balance_l1_m2_s"]<=1e6 and info["ion_flux_max_m2_s"]<=1e6
                assert info["original_dropped_balance_m2_s"]==d["original_residual"][info["constraint_row"]]
            assert d["shared_site"]["maximum_summed_site_fraction"]<1.0


def test_joined_read_keeps_failed_parent_and_does_not_extrapolate(native_records):
    parent,fine,pp,ps,fp,fs=native_records
    value=consumer.read_joined_pv(pp,ps,fp,fs,5)
    assert not value["parent_completed"] and value["continuation_completed"]
    assert value["parent_first_failure"]["voltage_V"]==0.3
    np.testing.assert_array_equal(value["accepted_voltage_V"],
        np.r_[parent["result"]["voltage_V"],fine["result"]["voltage_V"]])
    assert len(value["accepted_voltage_V"])==46
    assert value["output_current_A_m2"][-1]>0.0
    assert value["metrics"]["short_circuit_current_A_m2"]==parent["result"]["output_current_A_m2"][0]
    for name in ("open_circuit_voltage_V","maximum_power_W_m2","fill_factor","efficiency"):
        assert value["metrics"][name] is None
    assert value["physical_certificate_status"]=="not_evaluated"


def test_cli_joined_read_is_solver_free(native_records,monkeypatch,capsys):
    _,_,pp,ps,fp,fs=native_records
    def forbidden(*args,**kwargs):
        pytest.fail("read-only PV access started model/solver work")
    for module,name in ((consumer,"compile_case"),(consumer,"run_ionic_steady_jv"),
                        (consumer,"load_ionic_steady_initial"),(driver,"solve_ionic_steady_state")):
        monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(sys,"argv",["ionic_pv","read","--result",str(fp),"--sha256",fs,
        "--parent-result",str(pp),"--parent-sha256",ps,"--parent-index","5"])
    assert consumer.main()==0
    assert json.loads(capsys.readouterr().out)["metrics"]["open_circuit_voltage_V"] is None


@pytest.mark.parametrize("metric",["open_circuit_voltage_V","maximum_power_W_m2","fill_factor","efficiency"])
def test_reader_refuses_metrics_invented_without_brackets(native_records,tmp_path,metric):
    parent=deepcopy(native_records[0])
    parent["result"]["metrics"][metric]=0.5
    raw=(json.dumps(_encode(parent),allow_nan=False)+"\n").encode()
    path=tmp_path/"InventedMetric.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError,match="metrics contradict"):
        driver.read_ionic_steady_jv(path,hashlib.sha256(raw).hexdigest())


def test_failed_parent_candidate_cannot_seed_a_join(native_records):
    _,_,pp,ps,fp,fs=native_records
    with pytest.raises(ValueError,match="accepted parent point"):
        consumer.read_joined_pv(pp,ps,fp,fs,6)


@pytest.fixture(scope="module")
def native_tail():
    path=os.environ.get("SOLARLAB_IONIC_PV_TAIL_RESULT")
    if not path:
        pytest.skip("bind the archived tail explicitly; never rerun native physics")
    sha=os.environ["SOLARLAB_IONIC_PV_TAIL_SHA256"]
    return driver.read_ionic_steady_jv(path,sha),Path(path),sha


def test_tail_retains_whole_parent_result_and_effective_controls(native_records,native_tail):
    fine=native_records[1]["result"]
    tail=native_tail[0]["result"]
    parent=fine["attempts"][39]["solution"]
    assert native_tail[0]["completed"] and tail["solver_calls"]==2
    np.testing.assert_array_equal(tail["requested_voltage_V"],consumer.TAIL_VOLTAGES_V)
    initial=tail["initial_provenance"]
    raw=Path(initial["source_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==initial["source_sha256"]
    exported=json.loads(raw)
    assert exported["result"]==consumer.json_value(parent)
    assert exported["summary"]["parent_curve_sha256"]==native_records[5]
    assert exported["summary"]["parent_attempt_index"]==39
    assert _encode(initial["state"])==_encode(parent["state"])
    assert tail["solver_controls"]["steady"]==parent["solver_controls"]
    assert tail["solver_controls"]["steady"]["wall_s"]==14.116039042011835
    assert tail["evaluation_counts"]=={"residual":504,"jacobian":95}


def test_tail_original_current_and_both_species_inventory_checks(native_tail):
    r=native_tail[0]["result"]
    for attempt in r["attempts"]:
        point=attempt["solution"]
        d=point["diagnostics"]
        assert attempt["accepted"] and all(g["passed"] for g in point["checks"])
        assert d["total_current_density_A_m2"].shape==(38,)
        assert np.ptp(d["total_current_density_A_m2"])<=1e-8
        for species in r["ion_species"]:
            density=point["state"][species+"_ion_density_m3"]
            info=d["ion_species"][species]
            assert math.fsum((r["ion_weights_m"]*density).tolist())==info["inventory_m2"]
            assert info["inventory_relative_error"]<=1e-12
            assert info["ion_balance_l1_m2_s"]<=1e6 and info["ion_flux_max_m2_s"]<=1e6
            assert info["original_dropped_balance_m2_s"]==d["original_residual"][info["constraint_row"]]
        assert d["shared_site"]["maximum_summed_site_fraction"]<1.0


def test_three_wave_metrics_have_actual_bracket_and_keep_original_failure(native_records,native_tail):
    _,_,pp,ps,fp,fs=native_records
    _,tp,ts=native_tail
    value=consumer.read_three_wave_pv(pp,ps,fp,fs,tp,ts)
    assert len(value["accepted_voltage_V"])==48
    assert np.all(np.diff(value["accepted_voltage_V"])>0.0)
    assert value["solver_calls_including_preserved_failures"]==49
    assert not value["coarse_completed"] and value["fine_completed"] and value["tail_completed"]
    assert value["coarse_first_failure"]["voltage_V"]==0.3
    assert value["coarse_first_failure"]["iterations"]==60
    bracket=value["open_circuit_bracket"]
    assert bracket["voltage_V"]==[1.2,1.225]
    j0,j1=bracket["output_current_A_m2"]
    assert j0>0>j1
    metrics=value["metrics"]
    assert metrics["open_circuit_voltage_V"]==pytest.approx(1.2+0.025*j0/(j0-j1))
    assert metrics["short_circuit_current_A_m2"]==144.3836399722872
    assert metrics["maximum_power_W_m2"]==pytest.approx(147.10362262419616)
    assert metrics["fill_factor"]==pytest.approx(0.838815495249181)
    assert metrics["efficiency"] is None and value["incident_power_W_m2"] is None
    assert value["physical_certificate_status"]==value["mesh_qualification"]=="not_evaluated"
    assert "piecewise-linear" in value["metric_basis"]


def test_three_wave_cli_read_never_invokes_compile_initialization_or_solver(native_records,native_tail,monkeypatch,capsys):
    _,_,pp,ps,fp,fs=native_records
    _,tp,ts=native_tail
    def forbidden(*args,**kwargs):
        pytest.fail("saved three-wave metrics started model or solver work")
    for module,name in ((consumer,"compile_case"),(consumer,"run_ionic_steady_jv"),
                        (consumer,"load_ionic_steady_initial"),(driver,"solve_ionic_steady_state")):
        monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(sys,"argv",["ionic_pv","read","--result",str(tp),"--sha256",ts,
        "--parent-result",str(fp),"--parent-sha256",fs,"--parent-index","39",
        "--coarse-result",str(pp),"--coarse-sha256",ps])
    assert consumer.main()==0
    result=json.loads(capsys.readouterr().out)
    assert result["open_circuit_bracket_established"] and result["tail_completed"]
    assert result["coarse_first_failure"]["voltage_V"]==0.3


def test_three_wave_read_refuses_a_changed_parent_export(native_records,native_tail,monkeypatch):
    _,_,pp,ps,fp,fs=native_records
    tail,tp,ts=native_tail
    seed=Path(tail["result"]["initial_provenance"]["source_path"])
    original=Path.read_bytes
    def changed(path):
        raw=original(path)
        return raw+b" " if path==seed else raw
    monkeypatch.setattr(Path,"read_bytes",changed)
    with pytest.raises(ValueError,match="initial artifact SHA-256 mismatch"):
        consumer.read_three_wave_pv(pp,ps,fp,fs,tp,ts)


@pytest.mark.parametrize("command,expected_grid",[
    ("solve",tuple(tick/40.0 for tick in range(51))),
    ("coarse-probe",(0.0,0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.05,1.1,1.2)),
])
def test_primary_and_historical_commands_call_one_declared_protocol(command,expected_grid,monkeypatch,tmp_path,capsys):
    """CLI routing only: no synthetic value is used as native solver evidence."""
    device=SimpleNamespace(ion_node_count=13)
    initial=SimpleNamespace(generation_scale=1.0)
    calls=[]
    loads=[]
    settings=consumer.controls(13)
    def load(device_arg,path,sha,*,controls):
        loads.append((device_arg,path,sha,controls))
        return initial
    def run(device_arg,voltages,**kwargs):
        calls.append((device_arg,voltages,kwargs))
        return SimpleNamespace(completed=True)
    def forbidden(*args,**kwargs):
        pytest.fail("direct curve attempted a parent export or manual join")
    monkeypatch.setattr(consumer,"compile_case",lambda mode:device if mode=="shared" else None)
    monkeypatch.setattr(consumer,"load_ionic_steady_initial",load)
    monkeypatch.setattr(consumer,"run_ionic_steady_jv",run)
    monkeypatch.setattr(consumer,"write_ionic_steady_jv",lambda result,path:"synthetic_display_only")
    monkeypatch.setattr(consumer,"read_ionic_steady_jv",lambda path,sha:summary_input([0.0,1.25],[10.0,-1.0]))
    monkeypatch.setattr(consumer,"export_parent_initial",forbidden)
    monkeypatch.setattr(consumer,"read_joined_pv",forbidden)
    seed=tmp_path/"Accepted.json"
    monkeypatch.setattr(sys,"argv",["ionic_pv",command,"--initial-result",str(seed),
        "--initial-sha256","explicit_source_sha","--output",str(tmp_path/"Curve.json")])
    assert consumer.main()==0
    capsys.readouterr()
    assert loads==[(device,seed,"explicit_source_sha",settings)]
    assert len(calls)==1
    called_device,grid,kwargs=calls[0]
    assert called_device is device and grid==expected_grid
    assert kwargs["initial"] is initial and kwargs["initial_policy"]=="saved_accepted"
    assert kwargs["generation_scale"]==1.0 and kwargs["incident_power_W_m2"] is None
    assert kwargs["options"].max_points==len(expected_grid)
    assert kwargs["options"].wall_s==25.0 and kwargs["options"].steady==settings


@pytest.fixture(scope="module")
def direct_record():
    path=os.environ.get("SOLARLAB_IONIC_PV_DIRECT_RESULT")
    if not path:
        pytest.skip("bind the saved direct curve explicitly; never resimulate it in tests")
    sha=os.environ["SOLARLAB_IONIC_PV_DIRECT_SHA256"]
    return driver.read_ionic_steady_jv(path,sha),Path(path),sha


def test_direct_native_curve_uses_one_fresh_initial_and_all_requested_points(direct_record):
    record,_,_=direct_record
    r=record["result"]
    assert record["completed"] and len(r["attempts"])==r["solver_calls"]==51
    np.testing.assert_array_equal(r["requested_voltage_V"],consumer.VOLTAGES_V)
    np.testing.assert_array_equal(r["voltage_V"],consumer.VOLTAGES_V)
    assert r["initial_provenance"]["source_sha256"]=="55fa5adc94e9fc31866482025f0bade873289680ba4c958ab839b8f9d7f8f9d3"
    assert r["solver_controls"]["steady"]==asdict(consumer.controls(13))
    assert r["solver_controls"]["wall_s"]==25.0 and r["solver_controls"]["max_points"]==51
    for attempt in r["attempts"]:
        solution=attempt["solution"]
        d=solution["diagnostics"]
        assert attempt["accepted"] and all(g["passed"] for g in solution["checks"])
        assert solution["state"]["u"].shape==(137,)
        assert np.ptp(d["total_current_density_A_m2"])<=1e-8
        assert d["shared_site"]["maximum_summed_site_fraction"]<1.0
        for species in ("positive","negative"):
            density=solution["state"][species+"_ion_density_m3"]
            info=d["ion_species"][species]
            assert math.fsum((r["ion_weights_m"]*density).tolist())==info["inventory_m2"]
            assert info["inventory_relative_error"]<=1e-12
    summary=consumer.pv_summary(record)
    assert summary["first_failure"] is None
    assert summary["open_circuit_bracket"]["voltage_V"]==[1.2,1.225]
    assert summary["metrics"]["efficiency"] is None
    assert summary["physical_certificate_status"]=="not_evaluated"


def test_direct_saved_cli_read_requires_no_initial_artifact_or_solver(direct_record,monkeypatch,capsys):
    record,path,sha=direct_record
    def forbidden(*args,**kwargs):
        pytest.fail("direct saved read started a model, initialization, join or solve")
    for module,name in ((consumer,"compile_case"),(consumer,"run_ionic_steady_jv"),
                        (consumer,"load_ionic_steady_initial"),(consumer,"read_joined_pv"),
                        (consumer,"read_three_wave_pv"),(driver,"solve_ionic_steady_state")):
        monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(sys,"argv",["ionic_pv","read","--result",str(path),"--sha256",sha])
    assert consumer.main()==0
    value=json.loads(capsys.readouterr().out)
    assert value==consumer.pv_summary(record)
    assert len(value["accepted_voltage_V"])==51 and value["first_failure"] is None
