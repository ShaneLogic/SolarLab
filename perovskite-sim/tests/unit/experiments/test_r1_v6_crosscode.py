"""Guard units, tier selection and independent budgets in V6 diagnostics."""
import importlib.util
from pathlib import Path

import pytest


def load_tool(name):
    path=Path(__file__).resolve().parents[3]/"scripts"/name
    spec=importlib.util.spec_from_file_location(name.removesuffix(".py"),path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_uses_all_own_solver_changes_and_refuses_single_difference():
    tool=load_tool("analyze_r1_v6_crosscode.py")
    assert tool.envelope([4.,2.,1.])["uncertainty"]==2.
    assert tool.envelope([1.,2.,6.])["last_difference_decreased"] is False
    with pytest.raises(ValueError,match="three"):
        tool.envelope([1.,2.])


def test_fault_that_improves_agreement_is_not_reported_as_detected():
    tool=load_tool("analyze_r1_v6_crosscode.py")
    result=tool.classify_fault(10.,2.,0.,3.)
    assert result["fault_makes_agreement_better"] is True
    assert result["normal_within_budget"] is False
    assert result["fault_outside_budget"] is False
    assert result["detects_under_observed_budget"] is False
    assert tool.classify_fault(1.,10.,0.,3.)["detects_under_observed_budget"] is True
    assert tool.classify_fault(1.,10.,0.,None)["detects_under_observed_budget"] is None


def test_qualification_fails_closed_for_unknown_unconverged_or_small_denominator():
    tool=load_tool("analyze_r1_v6_crosscode.py")
    assert tool.qualification([],[],100.) is True
    assert tool.qualification(["unquantified model"],[],100.) is False
    assert tool.qualification([],["nonmonotone time"],100.) is False
    assert tool.qualification([],[],10.) is False


def test_df_validation_rejects_full_time_array_with_truncated_state():
    tool=load_tool("analyze_r1_v6_crosscode.py")
    profile={"state_time_rows":200,"time_rows":200,"state_nodes":10,"nx":10,"finite_history":True}
    record={"status":"completed","matlab_max_computational_threads":1,"request":{"times_s":[1e-8]},
            **{k:dict(profile) for k in ["equilibrium","ramp_A","ramp_B","ctlA_1em08","ctlB_1em08"]}}
    tool.validate_new_df(record)
    record["ctlB_1em08"]["state_time_rows"]=199
    with pytest.raises(ValueError,match="truncated"):
        tool.validate_new_df(record)
    record["ctlB_1em08"]["state_time_rows"]=200
    record["ctlA_1em08"]["finite_history"]=False
    with pytest.raises(ValueError,match="nonfinite"):
        tool.validate_new_df(record)


def test_ratio_and_increment_are_formed_within_the_requested_tier():
    tool=load_tool("analyze_r1_v6_crosscode.py")
    record={"rows":[{"substeps":n,"t_s":t,"Jion_nm25":v,"Jn_nm50":v}
                    for n,t,v in [(4,1e-8,2.),(4,1e-6,10.),(8,1e-8,3.),(8,1e-6,6.)]]}
    assert tool.observable(tool.r1_rows(record,8),"g_ion25")==2.
    assert tool.observable(tool.r1_rows(record,4),"delta_electron50")==8.
    with pytest.raises(KeyError):
        tool.observable(tool.r1_rows(record,16),"g_ion25")


def test_extractor_uses_flux_charge_frame_and_square_metre_conversion():
    tool=load_tool("run_r1_v6_crosscode.py")
    prepared={"grid":{"faces_m":[0.,25e-9,75e-9,100e-9]}}
    row={"phase":"accepted_regular_step","time_s":1e-8,"substeps":8,
         "physics_reconstruction":{"positive_ion_flux_m2_s":[1e10,3e10],
             "positive_ion_current_A_m2":[999.,999.],
             "electron_current_A_m2":[-2.,-6.],"hole_current_A_m2":[1.,3.]}}
    out=tool.extract_rows(prepared,[row])[0]
    assert out["Jion_nm50"]==pytest.approx(2e10*tool.Q/1e4)
    assert out["Jn_nm50"]==pytest.approx(-4e-4)
    assert out["Jp_nm50"]==pytest.approx(2e-4)


def test_missing_flux_or_incompatible_faces_cannot_create_successful_rows():
    tool=load_tool("run_r1_v6_crosscode.py")
    prepared={"grid":{"faces_m":[0.,25e-9,75e-9,100e-9]}}
    assert tool.extract_rows(prepared,[{"physics_reconstruction":{}}])==[]
    row={"phase":"accepted_regular_step","time_s":1e-8,"physics_reconstruction":{
        "positive_ion_flux_m2_s":[1.],"electron_current_A_m2":[1.],"hole_current_A_m2":[1.]}}
    with pytest.raises(ValueError,match="coordinates"):
        tool.extract_rows(prepared,[row])
