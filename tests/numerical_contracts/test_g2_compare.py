"""Manufactured comparison records and saved-file readers; no physical run."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path

import pytest

from scripts.benchmarks import g2_compare as c
from scripts.benchmarks import t1_gate as t
from .test_t1_gate import base_request, SPEC3
from scripts.benchmarks.g2_route_bindings import sample_labels


def case(case_id="B_N8_Strict_Full"):
    n=16 if "N16" in case_id else 8
    mesh=[F(i,n) for i in range(n+1)]
    edges=[F(0)]+[(a+b)/2 for a,b in zip(mesh,mesh[1:])]+[F(1)]
    labels=sample_labels(base_request());rows=[]
    for tag in labels:
        now=F(float.fromhex(tag["time_hex"]))/F(9.2)
        values={"phi_V":F(0),"n_m3":F(100)+2*now,"p_m3":F(200)+now,"c_m3":F(300),"f":F(1,2)}
        rows.append({"tag":tag,"states":{field:[c.enc(v) for _ in mesh] for field,v in values.items()},
            "ports":{"left":c.enc(2+now),"right":c.enc(-2-now)},"q":c.enc(now),"f":c.enc(now)})
    return {"case_id":case_id,"mesh_intervals":n,"mesh_m":list(map(t.pair,mesh)),
        "cell_edges_m":list(map(t.pair,edges)),"volumes_m3":[t.pair(b-a) for a,b in zip(edges,edges[1:])],
        "thermal_voltage_V":t.pair(1),"problem_identity":"manufactured_same_N8_problem",
        "reference_error_problem_identity":"manufactured_operator_initial_point_protocol",
        "physical_input_identity":"manufactured_same_continuum_input","fixed_electrode_gauge":"same-fixed-gauge",
        "material_defect_support":["manufactured-uniform-support"]*(n+1),"samples":rows}


def write_pin(path,value):
    import hashlib
    from scripts.benchmarks.native_history import integer_json_dumps
    raw=(integer_json_dumps(value,t.INTEGER_POLICY,sort_keys=True)+"\n").encode()
    path.write_bytes(raw)
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def reviewer_fixture(folder,author):
    record=write_pin(folder/(author+"-Review.json"),{"fixture_only":True,"reviewer":"toy-independent-science","evidence_author":author})
    return {"id":"toy-independent-science","role":"independent_science","evidence_author":author,
            "independent":True,"review_record":record}


def acceptance_history_files(folder,subject,method):
    """Complete but explicitly manufactured method-specific metadata, no history read."""
    labels=[row["tag"] for row in subject["samples"]]
    be=subject["case_id"].startswith("BE_")
    identity={"case_id":subject["case_id"],"input_sha256":subject["request_sha256"],"method_contract_sha256":method["sha256"],
        "plan_sha256":t.digest("toy plan"),"source_commit":"d"*40,"environment_identity":"toy environment",
        "schedule_identity":t.digest("toy schedule"),"source_identity":subject["source_id"],
        "initial_point_identity":subject["native_request"]["initial_preparation"]["point_identity"],"controls_sha256":t.digest("toy controls")}
    if be:subject["run_id"]=t.digest(identity)
    prefix=subject["case_id"]
    history={"path":prefix+".jsonl","encoding":"raw" if be else "gzip","logical_sha256":t.digest([prefix,labels]),
        "logical_bytes":2000,"encoded_bytes":2000 if be else 1000,"records":131,"closed":True,"writer_error":None}
    result={"fixture_only":True,"request_sha256":subject["request_sha256"],"run_id":subject["run_id"],
        **identity,"status":"completed" if be else "completed_bounded_voltage_lift_native_pilot",
        "numerical_complete":True,"complete_protocol":True, "observed_history" if be else "history":history}
    if be:
        result['schema']='solarlab.g2-reference-be-history.v1'
        result['method']='conservative-backward-euler'
        reader={"fixture_only":True,"schema":"solarlab.g2-reference-be-history.v1","structural_pass":True,"numerical_complete":True,
            "method":"conservative-backward-euler",
            "case_id":subject["case_id"],"tagged_observations":131,"observed_history":history,"observations":[{"tag":tag} for tag in labels]}
    else:
        reader={"fixture_only":True,"schema":"solarlab.native-history-verification.v1","verified":True,
            "complete_protocol":True,"complete_readback":True,"case_id":subject["case_id"],"request_sha256":subject["request_sha256"],
            "history":dict(history,eof_seen=True,crc_verified=True,container_complete=True,partial_tail_bytes=0)}
    binding={"case_id":subject["case_id"],"run_id":subject["run_id"],"source_identity":subject["source_id"],
        "input_sha256":subject["request_sha256"],"run_result":write_pin(folder/(prefix+"-Result.json"),result),
        "history_manifest":write_pin(folder/(prefix+"-History.json"),history),
        "independent_readback":write_pin(folder/(prefix+"-Readback.json"),reader),
        **{key:history[key] for key in ("logical_sha256","logical_bytes","records","encoded_bytes")},
        "closed":True,"container_complete":True,"labels_sha256":t.digest(labels)}
    subject["sealed_history_binding"]=binding
    subject["independent_readback_pin"]=binding["independent_readback"]
    return binding,identity


def quantity_binding(pin,key,labels,*,strict=False):
    record=t._pin(pin["path"],pin)
    ownership={"r_R_included_in_E_R":True,"r_R_source":"per_record/<tag>/computed_value/radius","tagwise_ownership_record":pin}
    if strict:ownership["r_S_included_in_E_S"]=True
    return {"reviewed_record":pin,"coverage":{"kind":"full_protocol","records":131,"labels_sha256":t.digest(labels),
        "unit":c.quantity_unit(key),"norm":"max_abs_tagged131"},"claim_kind":"conditional_uncertainty_estimate",
        "value":record["conditional_estimate_value"],"radius_accounting":ownership}


def error_files(folder,reference,*,verdict="accepted",include_radius=True,method=None):
    from .test_g2_route_bindings import frozen_definition
    labels=[r["tag"] for r in reference["samples"]]
    method=method or frozen_definition(folder,"ErrorMethod","solarlab.g2-reference-method.v1",1)
    history,identity=acceptance_history_files(folder,reference,method)
    parameters=write_pin(folder/"ErrorParameters.json",{"fixture_only":True,
        "schema":"solarlab.g2-reference-error-parameters.v1","target_problem_identity":reference["reference_error_problem_identity"],
        "method_contract_sha256":method["sha256"]})
    levels=[{"identity":dict(identity,case_id=f"BE_N8_L{i}_Full",schedule_identity=t.digest(["schedule",i]),
              controls_sha256=t.digest(["controls",i])),"observed_history":{key:history[key] for key in
              ("logical_sha256","logical_bytes","encoded_bytes","records")}} for i in range(3)]
    levels[2]["identity"]=identity
    lineage=write_pin(folder/"Lineage.json",{"fixture_only":True,"schema":"solarlab.g2-reference-error.v1",
        "kind":"reference_error_lineage","method_contract":method,"parameters":parameters,"levels":levels})
    records={}
    for key in ("port/left","q","f","state/phi_V/0"):
        values=c.quantity_values(reference,key)
        row={"fixture_only":True,"schema":"solarlab.g2-reference-evidence.v1","status":"computed_pending_review",
            "quantity_key":key,"quantity_and_unit":{"quantity":c.quantity_name(key),"unit":c.quantity_unit(key)},
            "claim_kind":"conditional_uncertainty_estimate","target_problem_identity":reference["reference_error_problem_identity"],
            "norm_and_coverage":{"norm":"max_abs_tagged131","records":131,"labels_sha256":t.digest(labels)},
            "reference_source_and_run_identity":{"case_id":"BE_N8_L2_Full","source_identity":reference["source_id"],
                "request_sha256":reference["request_sha256"],"run_id":reference["run_id"],"history_sha256":history["logical_sha256"]},
            "reviewer_and_decision_authority":{"evidence_author":"toy-author","method_contract":method},
            "assumptions_and_checked_premises":{"E3_parameters":parameters},
            "error_components_and_ownership":[{"source":lineage}],
            "reference_value_radius_ownership":"reported observation radii are included in E_R; coalesce them in E2" if include_radius else
                "reported observation radii are not included in E_R; add them in E2",
            "conditional_estimate_value":t.pair(F(1,1000000)),"rigorous_bound_value":None,
            "estimated_or_enclosed_value":{"value_field":"conditional_estimate_value","value":t.pair(F(1,1000000)),"unit":c.quantity_unit(key)},
            "per_record":[{"tag":tag,"computed_value":{"center":v["center"],"radius":v["absolute_error_bound"]},
                "computation":{"status":"computed_pending_review","method_status":"passed","E_R":t.pair(F(1,1000000))}}
                for tag,v in zip(labels,values,strict=True)]}
        records[key]=write_pin(folder/(t.digest(key)+".json"),row)
    result=write_pin(folder/"ErrorResult.json",{"fixture_only":True,"schema":"solarlab.g2-reference-error.v1",
        "target_problem_identity":reference["reference_error_problem_identity"],"method_contract":method,"quantity_records":records})
    accepted=sorted(records) if verdict=="accepted" else []
    acceptance=write_pin(folder/"ErrorAcceptance.json",{"fixture_only":True,"schema":"solarlab.g2-reference-error-acceptance.v1","version":1,
        "reviewed_error_file":result,"accepted_quantities":accepted,"quantity_bindings":{key:quantity_binding(records[key],key,labels) for key in accepted},
        "claim_kind":"conditional_uncertainty_estimate","reviewer":reviewer_fixture(folder,"toy-author"),
        "verdict":"unknown" if verdict=="conditions" else verdict,"conditions":[],"method_contract":method,"parameters":parameters,"lineage":lineage,
        "target_problem_identity":reference["reference_error_problem_identity"],"reference_history":history,
        "scope":{"route":"G2-R","meaning":"BE reference errors only; no IDA Strict acceptance yet"},"scientific_pass":False,"execution_authorized":False})
    return result,acceptance,method



def referenced_cases():
    from scripts.benchmarks.g2_reference_error import target_identity
    strict=case();reference=case();reference["case_id"]="BE_N8_L2_Full"
    request=base_request()
    request["schema"]="solarlab.voltage-lift-native-request.v1"
    request["numeric_packet"].update(nodes=9,intervals=8,x_m=[i/8 for i in range(9)])
    request["voltage_lift_map"]["physical_model"]=t.digest("toy physical model shared by both methods")
    request["initial_preparation"]={"point_identity":t.digest("toy initial point")}
    for key in ("physical_handoff_words_hex","physical_rate_handoff_words_hex"):
        request["initial_segment_frame"][key]=[[0.0.hex()]*45 for _ in range(12)]
    for row,name in ((strict,"IDA"),(reference,"BE")):
        row["native_request"]=dict(deepcopy(request),fixture_method=name)
        row.update(source_id=request["voltage_lift_map"]["physical_model"],
            request_sha256=t.digest(row["native_request"]),run_id=t.digest(name+"run"),
            problem_identity=t.physical_problem_identity(request),
            reference_error_problem_identity=target_identity(request["voltage_lift_map"]["physical_model"],
                request["initial_preparation"]["point_identity"],request["segments"]))
    return strict,reference



def errors():
    names=[f"state/{field}/{i}" for field in c.FIELDS for i in range(9)]+["port/left","port/right"]
    return {name:{"status":"accepted","claim_kind":"conditional_uncertainty_estimate",
                  "values":[t.pair(0) for _ in range(131)],"fixture_only":True} for name in names}


def time_compare(main,strict,reference=None):
    return c.compare_time(main,strict,errors() if reference is None else reference,
        labels=sample_labels(base_request()),scales={"n_m3":100,"p_m3":200,"c_m3":300})


def test_complete_131_tags_both_ports_and_19_physical_locations():
    result=time_compare(case(c.FULL_CASES[1]),case())
    assert result["status"] == "passed"
    assert len(result["port_tables"]) == 262 and len(result["state_tables"]) == 131*19*5
    assert len(result["initial_bias_checks"]) == 2
    assert result["scientific_qualification"] is False
    labels=[r["tag"] for r in case()["samples"]]
    assert labels[32]["time_hex"] == labels[33]["time_hex"]
    assert (labels[32]["event_side"],labels[33]["event_side"]) == ("left","right")


def test_raw_DC_bias_cannot_be_hidden_by_subtracting_both_initials():
    main,strict=case(c.FULL_CASES[1]),case()
    bias=F(2,100000)
    for row in main["samples"]:
        value=t.q(row["ports"]["left"]["center"])
        row["ports"]["left"]=c.enc(value+bias)
    result=time_compare(main,strict)
    assert result["initial_bias_checks"][0]["status"] == "failed"
    assert result["port_tables"][0]["both_initials_subtracted"] is False
    assert t.q(result["port_tables"][0]["observed_difference_upper"]) == bias


def test_missing_and_diagnostic_reference_error_stays_unknown_per_quantity():
    ref=errors();del ref["port/right"]
    for i in range(9):ref[f"state/c_m3/{i}"]["claim_kind"]="diagnostic_agreement"
    result=time_compare(case(c.FULL_CASES[0]),case(),ref)
    assert all(r["status"] == "unknown" for r in result["port_tables"] if r["key"]["port"] == "right")
    assert all(r["status"] == "passed" for r in result["port_tables"] if r["key"]["port"] == "left")
    assert all(r["status"] == "unknown" for r in result["state_tables"] if r["key"]["field"] == "c_m3")
    assert all(r["status"] == "passed" for r in result["state_tables"] if r["key"]["field"] == "phi_V")


def test_reference_one_third_boundary_and_radius_ownership():
    value=c.qc.Span.of(0)
    exact=c.assess(value,value,F(1,3),(1,1),key="toy")
    assert exact["status"] == "passed"
    excess=c.assess(value,value,F(1,3)+F(1,2**150),(1,1),key="toy")
    assert excess["status"] == "failed" and excess["reference_radius_added_again"] is False
    assert c.assess(value,value,None,(1,1),key="toy")["reason"] == "unknown_reference_error"


@pytest.mark.parametrize("field,delta",[("phi_V",F(1,10000)),("c_m3",F(3,10)),
    ("f",F(1,10000)),("n_m3",F(1,10)),("p_m3",F(1,5))])
def test_state_bands_retain_the_declared_SI_values(field,delta):
    main,strict=case(c.FULL_CASES[1]),case()
    for row in main["samples"]:
        row["states"][field]=[c.enc(t.q(v["center"])+delta) for v in row["states"][field]]
    result=time_compare(main,strict)
    first=next(r for r in result["state_tables"] if r["key"]["field"] == field)
    assert first["status"] == "passed" and t.q(first["allowance_lower"]) == delta
    direct=c.assess(c.qc.Span.of(delta+F(1,2**160)),c.qc.Span.of(0),0,(delta,delta),key="toy")
    assert direct["status"] == "failed"


@pytest.mark.parametrize("damage",["missing","duplicate","event_side","physical_problem"])
def test_comparison_never_masks_missing_or_mismatched_records(damage):
    main,strict=case(c.FULL_CASES[1]),case()
    if damage == "missing":main["samples"].pop()
    elif damage == "duplicate":main["samples"][33]=deepcopy(main["samples"][32])
    elif damage == "event_side":main["samples"][33]["tag"]["event_side"]="left"
    else:main["problem_identity"]="different_initial_words"
    with pytest.raises(c.ComparisonError):time_compare(main,strict)


def test_SG_interval_propagation_contains_all_endpoint_corners_and_low_word():
    tiny=F(1,2**120);S=c.qc.Span
    boxed=c._sg_box(S(2,2+tiny),S(8,8+tiny),S(-1,2),F(1,3),"electron")
    for a in (F(2),2+tiny):
        for b in (F(8),8+tiny):
            for xi in (-1,0,2):
                exact=c.qc.sg_interpolate(a,b,xi,F(1,3),carrier="electron")
                assert boxed.lo <= exact.lo and boxed.hi >= exact.hi
    value=c._sg_box(S(1,1),S(1+tiny,1+tiny),S(0,0),F(1,2),"electron")
    assert value.lo == value.hi == 1+tiny/2


def test_E2_coalesces_BE_radius_and_preserves_claim_kind():
    kwargs=dict(strict_value=c.enc(2,F(1,10)),reference_value=c.enc(1,F(1,5)),
                reference_error=F(3,10),map_error=F(2,5))
    own=c.e2_transfer(**kwargs,includes_reference_radius=True)
    extra=c.e2_transfer(**kwargs,includes_reference_radius=False)
    assert t.q(own["conditional_estimate_value"]) == F(9,5)
    assert t.q(extra["conditional_estimate_value"]) == 2
    assert own["rigorous_bound_value"] is None and own["status"] == "computed_pending_review"
    assert t.q(own["ownership"]["r_R_marginal"]) == 0
    with pytest.raises(c.ComparisonError):c.e2_transfer(**kwargs,includes_reference_radius=None)


def test_N16_projection_preserves_inventory_and_reports_no_order():
    low,high=case(),case(c.FULL_CASES[3])
    for row in high["samples"]:
        row["states"]["c_m3"]=[c.enc(300+i) for i in range(17)]
    matrix=c.spatial_projection(high,low)
    vf,vc=list(map(t.q,high["volumes_m3"])),list(map(t.q,low["volumes_m3"]))
    assert all(sum(vc[i]*matrix[i][j] for i in range(9)) == vf[j] for j in range(17))
    result=c.compare_space(low,high,labels=sample_labels(base_request()))
    assert result["status"] == "reported_spatial_difference"
    assert result["summary"]["spatial_order"] is None and result["summary"]["G2Q_qualification"] is False
    row=next(r for r in result["state_tables"] if r["key"]["field"] == "c_m3")
    assert t.q(row["maximum_difference_upper"]) > 0
    assert t.q(row["inventory_difference"]["center"]) == -8
    high["material_defect_support"][8]="different-defect"
    with pytest.raises(c.ComparisonError,match="QC01"):c.compare_space(low,high,labels=sample_labels(base_request()))


def test_reader_sample_sink_waits_for_actual_certificate_and_keeps_event_sides(tmp_path):
    from .test_g2_route import continued_fixture,publish_fixture,read_fixture
    data=continued_fixture(failure=False,observation="sealed-history")
    publish_fixture(tmp_path,data);samples=[]
    verified=read_fixture(tmp_path,sample_sink=lambda tag,row,certificate:samples.append((tag,row,certificate)))
    assert verified["verified"] and verified["history"]["crc_verified"]
    expected=sum(len(times) for times in data[0]["observation_times"].values())
    assert len(samples) == expected
    assert [r[0] for r in samples] == sample_labels(data[0])
    for tag,row,certificate in samples:
        assert certificate["sample_record_sha256"] == row["record_sha256"]
        assert tag["time_hex"] == row["history"]["time_hex"]
    assert samples[0][0]["event_side"] == "right"
    repeated=[tag for tag,_,_ in samples if tag["time_hex"] == 1.0.hex()]
    assert [r["event_side"] for r in repeated] == ["left","right"]


@pytest.mark.parametrize("verdict",["accepted","conditions","rejected"])
def test_real_pinned_error_envelope_and_acceptance_do_not_promote_E2(tmp_path,verdict):
    strict,reference=referenced_cases();result,review,method=error_files(tmp_path,reference,verdict=verdict)
    records=c.read_error_records(result,review,method_pin=method,problem_identity=reference["reference_error_problem_identity"],labels=sample_labels(base_request()))
    transferred=c.transfer_E2(strict,reference,records,labels=sample_labels(base_request()))
    expected="computed_pending_review" if verdict=="accepted" else "unknown_reference_error"
    assert all(v["status"]==expected for v in transferred["quantities"].values())
    assert transferred["E_R_accepted_does_not_accept_E_S"] is True
    original=json.loads(Path(records["quantities"]["q"]["pin"]["path"]).read_text())
    assert original["status"]=="computed_pending_review"


def test_error_acceptance_wrong_pin_or_same_author_is_rejected(tmp_path):
    strict,reference=referenced_cases();result,review,method=error_files(tmp_path,reference)
    value=json.loads(Path(review["path"]).read_text());value["reviewed_error_file"]["sha256"]="0"*64
    review=write_pin(Path(review["path"]),value)
    found=c.read_error_records(result,review,method_pin=method,problem_identity=reference["reference_error_problem_identity"],labels=sample_labels(base_request()))
    assert found["status"]=="unknown_reference_error" and found["reason"]=="reference_error_review_binding"
    value["reviewed_error_file"]=result;value["reviewer"]["id"]="toy-author";review=write_pin(Path(review["path"]),value)
    found=c.read_error_records(result,review,method_pin=method,problem_identity=reference["reference_error_problem_identity"],labels=sample_labels(base_request()))
    assert found["status"]=="unknown_reference_error" and found["reason"]=="acceptance_independent_reviewer"


def test_BE_adapter_calls_the_owned_reader_and_preserves_all_state_words(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.benchmarks import g2_reference_be as be
    reference=case();labels=sample_labels(base_request());rows=[]
    for i,row in enumerate(reference["samples"]):
        raw={field:{"shape":[9],"words_hex":[[float(t.q(x["center"])).hex() for x in values],
              [(2.0**-100).hex()]*9]} for field,values in row["states"].items()}
        def be_enc(v):return {"center":v["center"],"radius":v["absolute_error_bound"]}
        rows.append({"tag":row["tag"],"physical_state_words":raw,"ports":{k:be_enc(v) for k,v in row["ports"].items()},
            "charges":{"device":be_enc(row["q"])},"cumulative_physical_flux":[be_enc(row["f"])],
            "record_sha256":t.digest(i),"point_identity":t.digest([i,"point"])})
    request={"fixture_only":True,"numeric_packet":{"layout_identity":"toy-layout"}}
    request_pin=write_pin(tmp_path/"BERequest.json",request)
    identity={k:t.digest(k) for k in ("method_contract_sha256","plan_sha256","schedule_identity","source_identity",
        "initial_point_identity","controls_sha256")}
    identity.update(case_id="BE_N8_L2_Full",input_sha256=t.digest(request),source_commit="a"*40,environment_identity="toy")
    layout=SimpleNamespace(identity="toy-layout");calls=[]
    def reader(directory,passed_layout,expected,**kwargs):
        calls.append((directory,passed_layout,expected))
        return {"structural_pass":True,"numerical_complete":True,"scientific_status":"unknown","observations":rows,
                "observed_history":{"path":"ToyHistory.jsonl","logical_bytes":1}}
    monkeypatch.setattr(be,"read_reference_history",reader)
    def metadata(*args):
        value=deepcopy(reference);value["samples"]=[];value["request_sha256"]=t.digest(request)
        value.update(source_id=identity["source_identity"],initial_point_identity=identity["initial_point_identity"])
        return value
    monkeypatch.setattr(c,"case_metadata",metadata)
    actual=c.read_be_case({"identity":identity,"request":request_pin,"directory":str(tmp_path)},labels,
        {"max_record_bytes":1048576,"max_logical_bytes":1048576},lambda:None,layout)
    assert len(calls)==1 and calls[0][1] is layout
    assert t.q(actual["samples"][0]["states"]["n_m3"][0]["center"])==100+F(1,2**100)
    assert t.q(actual["samples"][0]["q"]["center"])==0
    assert actual["run_id"]==t.digest(be.RunIdentity(**identity).payload())
    factories=[]
    monkeypatch.setattr(be,"reference_layout",lambda r:(factories.append(r),layout)[1],raising=False)
    c.read_be_case({"identity":identity,"request":request_pin,"directory":str(tmp_path)},labels,
        {"max_record_bytes":1048576,"max_logical_bytes":1048576},lambda:None)
    assert factories==[request] and len(calls)==2


def loaded_T1(case_id, *, body_defect=F(0)):
    from .test_t1_gate import data_v3,SPEC3
    request,labels,intervals,pieces=data_v3()
    for i,row in enumerate(intervals):
        row["case_id"]=case_id
        if body_defect:
            value=row["channels"]["device"]
            value["integral_F_enclosure_C"]=c.enc(-body_defect)
        piece=pieces[i]
        piece["case_id"]=case_id
        piece["channels"]=deepcopy(row["channels"])
        piece["parent_sha256"]=t.digest(row)
        piece["piece_id"]=t.digest({k:v for k,v in piece.items() if k!="piece_id"})
    return {"manifest":{"case_id":case_id,"run_id":"manufactured-run","source_id":"manufactured-source",
        "problem_identity":t.physical_problem_identity(request)},"request":request,"labels":labels,
        "spec":SPEC3,"intervals":intervals,"pieces":pieces,"checkpoint":lambda:None}


def test_matched_growth_is_latched_across_events_and_does_not_award_order():
    results={key:c.evaluate_T1(loaded_T1(key,body_defect=F(1,100000) if key==c.FULL_CASES[2] else F(0)))
             for key in c.FULL_CASES[:3]}
    result=c.matched_windows(results)
    assert result["status"]=="failed"
    grown=[r for r in result["device_growth"] if r["pair"]==[c.FULL_CASES[1],c.FULL_CASES[2]]]
    assert len(grown)==256 and all(r["status"]=="failed" for r in grown)
    assert all(r["convergence_order_proved"] is False for r in grown)
    assert all(r["systematic_floor_proved"] is False for r in result["metal_ratio_hypothesis"])


def test_decision_slots_remain_unknown_without_Strict_reference_review():
    from .test_t1_gate import SPEC3
    cases={key:case(key) for key in c.FULL_CASES}
    for value in cases.values():
        value["readback"]={"complete_protocol":True,"complete_readback":True,"verified":True,
                           "scientific_progress":{"failed_gates":[],"first_failure":None}}
    data={key:loaded_T1(key) for key in c.FULL_CASES}
    value=c.compare_loaded(cases,data,spec=SPEC3,method_pin={"sha256":"toy-method"},labels=sample_labels(base_request()),
        bindings={"physical_context":{"n_ref_m3":100,"p_ref_m3":200,"c_ref_m3":300}})
    normalized=value["normalized_evidence"]
    assert set(normalized["point_comparisons"])=={"N8_Main_vs_Strict","N8_Coarse_vs_Strict","N8_Strict_vs_N16_Strict"}
    assert normalized["point_comparisons"]["N8_Main_vs_Strict"]["status"]=="unknown"
    assert normalized["matched_window_T1"]["device"][c.FULL_CASES[1]]["status"]=="unknown_reference_error"
    assert value["scientific_pass"] is False and value["route_decision"]=="pending_independent_review"
    assert set(normalized["reference_evidence"])=={"device_q","device_integrated_f","left_current","right_current","phi","n","p","c","f_occupancy"}


def test_comparison_cli_keeps_the_existing_bounded_integer_codec(monkeypatch,capsys,tmp_path):
    from scripts.benchmarks import g2_route
    from scripts.benchmarks.native_history import decode_integer_values
    exact=2**16000+1
    monkeypatch.setattr(g2_route,"compare",lambda path:{"value":[exact,exact+2],"scientific_pass":False})
    assert g2_route.main(["compare","--plan",str(tmp_path/"NoRealPlan.json")])==0
    actual=decode_integer_values(json.loads(capsys.readouterr().out),t.INTEGER_POLICY)
    assert actual["value"]==[exact,exact+2]


def test_unadmitted_compare_cannot_read_any_result(tmp_path,monkeypatch):
    from scripts.benchmarks import g2_route
    from .test_g2_route import plan_fixture
    path=tmp_path/"Pending.json";path.write_text(json.dumps(plan_fixture(tmp_path)))
    monkeypatch.setattr(c,"compare_plan",lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("read unadmitted data")))
    with pytest.raises(ValueError):g2_route.compare(path)


def strict_reference_files(folder,*,zero_charge=False):
    from .test_g2_route_bindings import frozen_definition
    from .test_t1_gate import SPEC3
    strict,reference=referenced_cases()
    if zero_charge:
        for one in (strict,reference):
            for sample in one['samples']:sample['q']=sample['f']=c.enc(0)
    labels=[r['tag'] for r in strict['samples']]
    method=frozen_definition(folder,'E2Method','solarlab.g2-reference-method.v1',1)
    result,er_review,_=error_files(folder,reference,method=method)
    records=c.read_error_records(result,er_review,method_pin=method,
        problem_identity=reference['reference_error_problem_identity'],labels=labels)
    acceptance_history_files(folder,strict,method)
    context={'gate_spec':{'sha256':SPEC3.sha256},'method_contract':method,'evidence_author':'toy-E2-author',
        'inputs':{'E_R_acceptance':er_review,'IDA_Strict_sealed_history':strict['sealed_history_binding'],
                  'BE_reference_history':reference['sealed_history_binding']}}
    transferred=c.transfer_E2(strict,reference,records,labels=labels,context=context)
    transferred['quantity_records']={key:write_pin(folder/('E2-'+t.digest(key)+'.json'),row)
                                    for key,row in transferred['quantities'].items()}
    e2=write_pin(folder/'ToyE2.json',transferred)
    proof=write_pin(folder/'ToySameProblem.json',{'fixture_only':True,'target_problem_identity':strict['problem_identity'],
        'reference_error_problem_identity':reference['reference_error_problem_identity'],
        'IDA_Strict_input':write_pin(folder/'StrictInput.json',strict['native_request']),
        'BE_reference_input':write_pin(folder/'BEInput.json',reference['native_request'])})
    keys=sorted(transferred['quantities'])
    review=write_pin(folder/'ToyStrictAcceptance.json',{'fixture_only':True,
        'schema':'solarlab.g2-strict-reference-acceptance.v1','version':1,'reviewed_e2_file':e2,
        'inputs':context['inputs'],'accepted_quantities':keys,
        'quantity_bindings':{key:quantity_binding(transferred['quantity_records'][key],key,labels,strict=True) for key in keys},
        'claim_kind':'conditional_uncertainty_estimate','reviewer':reviewer_fixture(folder,'toy-E2-author'),
        'verdict':'accepted','conditions':[],'target_problem_identity':strict['problem_identity'],'same_problem_evidence':proof,
        'scope':{'route':'G2-R','reference_case_id':'B_N8_Strict_Full','coverage':'full131, both event sides'},
        'scientific_pass':False,'execution_authorized':False})
    wrapper={'schema':'solarlab.g2-strict-reference.v1','e2_file':e2,'independent_review':review}
    return strict,wrapper,method,transferred



def test_E2_acceptance_binds_three_inputs_and_keeps_computation_pending(tmp_path):
    from .test_t1_gate import SPEC3
    strict,ref,method,transferred=strict_reference_files(tmp_path)
    labels=[r['tag'] for r in strict['samples']]
    qualified=c.comparison_reference_errors(ref,spec=SPEC3,method_pin=method,labels=labels,strict=strict)
    assert qualified['status']=='accepted_reference_error', qualified
    assert set(qualified['quantities'])=={'q','f','port/left','state/phi_V/0'}
    assert all(v['source_status']=='computed_pending_review' for v in qualified['quantities'].values())
    result=t.reference_status_v3(ref,spec=SPEC3,labels=labels,problem_identity=strict['problem_identity'])
    assert result['status']=='eligible' and result['strict_acceptance_sha256']==ref['independent_review']['sha256']
    assert result['claim_kind']=='conditional_uncertainty_estimate'
    assert all(r['status']=='computed_pending_review' for r in transferred['quantities'].values())
    assert t._pin(ref['e2_file']['path'],ref['e2_file'])==transferred
    strict['samples'][10]['ports']['left']=c.enc(99)
    other=c.comparison_reference_errors(ref,spec=SPEC3,method_pin=method,labels=labels,strict=strict)
    assert 'port/left' not in other['quantities'] and 'q' in other['quantities']
    assert other['quantity_failures']['port/left']=='Strict_exact_saved_operands'


@pytest.mark.parametrize('damage',['root','pin','inputs','rejected','conditions','same_author','claim','missing_f','E_R_review'])
def test_E2_acceptance_missing_or_mismatched_evidence_is_unknown(tmp_path,damage):
    from .test_t1_gate import SPEC3
    strict,ref,method,_=strict_reference_files(tmp_path)
    review=t._pin(ref['independent_review']['path'],ref['independent_review'])
    if damage=='root':review=[]
    elif damage=='pin':review['reviewed_e2_file']['sha256']='0'*64
    elif damage=='inputs':review['inputs']['BE_reference_history']['logical_sha256']='0'*64
    elif damage=='rejected':review['verdict']='rejected'
    elif damage=='conditions':review['conditions']=['unresolved proof premise']
    elif damage=='same_author':review['reviewer']['id']='toy-E2-author'
    elif damage=='claim':review['claim_kind']='diagnostic_agreement'
    elif damage=='missing_f':
        review['accepted_quantities'].remove('f');del review['quantity_bindings']['f']
    else:
        er=review['inputs']['E_R_acceptance']
        value=t._pin(er['path'],er);value['verdict']='rejected';write_pin(Path(er['path']),value)
    ref['independent_review']=write_pin(Path(ref['independent_review']['path']),review)
    labels=[r['tag'] for r in strict['samples']]
    result=t.reference_status_v3(ref,spec=SPEC3,labels=labels,problem_identity=strict['problem_identity'])
    assert result['reason']=='unknown_reference_error'
    assert result['Qsig_C'] is None
    errors=c.comparison_reference_errors(ref,spec=SPEC3,method_pin=method,labels=labels,strict=strict)
    assert ('f' not in errors['quantities']) if damage=='missing_f' else errors['quantities']=={}
    if damage!='missing_f':assert errors['status']=='unknown_reference_error' and errors['reason']


def test_E_R_and_IDA_problem_identity_domains_are_distinct(tmp_path):
    strict,reference=referenced_cases();result,review,method=error_files(tmp_path,reference)
    assert strict['problem_identity']!=reference['reference_error_problem_identity']
    found=c.read_error_records(result,review,method_pin=method,problem_identity=strict['problem_identity'],
                               labels=sample_labels(base_request()))
    assert found['status']=='unknown_reference_error' and found['reason']=='reference_error_envelope'


def test_missing_E2_acceptance_cannot_be_replaced_by_E_R_acceptance(tmp_path):
    from .test_t1_gate import SPEC3
    strict,ref,method,transferred=strict_reference_files(tmp_path)
    ref['independent_review']=transferred['inputs']['E_R_acceptance']
    result=t.reference_status_v3(ref,spec=SPEC3,labels=sample_labels(base_request()),problem_identity=strict['problem_identity'])
    assert result['reason']=='unknown_reference_error'


@pytest.mark.parametrize('missing',[0,5])
def test_charge_series_never_rebases_after_a_missing_piece(missing):
    current=case();data=loaded_T1(c.FULL_CASES[2]);checked=c.evaluate_T1(data)
    data['pieces']=[p for p in data['pieces'] if p['window_index']!=missing]
    checked['fixed_windows'][missing]['complete_pieces']=False
    c.attach_charge_series(current,data,checked)
    for row in current['samples']:
        index=next(i for i,label in enumerate(data['labels']) if t.digest(label)==t.digest(row['tag']))
        if missing==0:
            assert row['q'] is None and row['f'] is None
        elif index>missing+1:
            assert row['q'] is not None and row['f'] is None


def sealed_comparison_fixture(folder,monkeypatch,*,missing_sample=False):
    from scripts.benchmarks import g2_route as route
    from scripts.benchmarks.native_history import HistoryWriter
    from .test_t1_gate import document
    from .test_g2_route import supervision_fixture
    labels=sample_labels(base_request());request=base_request()
    doc=document();doc.update(schema=t.SPEC_SCHEMA_V3,version=3)
    doc['physical_context'].update(n_ref_m3=100,p_ref_m3=200,c_ref_m3=300)
    gate=write_pin(folder/'ToyGate.json',doc)
    method=write_pin(folder/'ToyMethod.json',{'fixture_only':True})
    ids=c.FULL_CASES[1:3]
    cases=[{'id':key,'execution':{'effective_input_digest':t.digest(request),
             'native_admission':{'operation_id':'manufactured-run'}}} for key in ids]
    producer={'fixture_only':True,'cases':cases,'definitions':{'gate_spec':gate}}
    producer_pin=write_pin(folder/'ToyProducerPlan.json',producer)
    request_pin=write_pin(folder/'ToyRequest.json',request)
    def metadata(req,key):
        value=case(key);value['samples']=[]
        value.update(request_sha256=t.digest(req),source_id='manufactured-source',problem_identity=t.physical_problem_identity(req))
        return value
    monkeypatch.setattr(c,'case_metadata',metadata)
    def validate(path,**kwargs):
        assert kwargs['purpose']=='evaluate' and kwargs['executable'] is True
        assert kwargs['case_id'] in ids
        return {'plan':producer}
    monkeypatch.setattr(route,'validate',validate)
    descriptors={};histories=[]
    for key in ids:
        data=loaded_T1(key);context=metadata(request,key);context['run_id']='manufactured-run'
        case_folder=folder/key;case_folder.mkdir()
        writer=HistoryWriter(case_folder/'ToyHistory.jsonl.gz',encoding='gzip',total_output_bytes=8*1024**2,
                             max_record_bytes=65536,integer_io_policy=t.INTEGER_POLICY)
        writer.append({'kind':'sealed_g2_context','request_sha256':t.digest(request),'evidence':context})
        for kind,rows in (('t1_interval_evidence',data['intervals']),('t1_fixed_window_piece',data['pieces'])):
            for row in rows:writer.append({'kind':kind,'evidence':row})
        samples=case(key)['samples']
        if missing_sample and key==ids[0]:samples.pop(33)
        for row in samples:writer.append({'kind':'sealed_g2_sample','request_sha256':t.digest(request),'evidence':row})
        writer.finish();histories.append(str(writer.path))
        history={'path':str(writer.path),'encoding':'gzip','closed':True,'container_complete':True,
            'encoded_bytes':writer.encoded_bytes,'logical_bytes':writer.logical_bytes,'records':writer.records,
            'logical_sha256':writer.logical_sha256,'integer_io_policy':t.INTEGER_POLICY}
        manifest={'schema':t.EVIDENCE_SCHEMA,'case_id':key,'run_id':'manufactured-run','source_id':'manufactured-source',
            'problem_identity':t.physical_problem_identity(request),'run_plan':producer_pin,
            'gate_spec':gate,'request':request_pin,'history':history}
        receipt={'fixture_only':True,'schema':'solarlab.native-history-verification.v1','verified':True,
            'complete_protocol':True,'complete_readback':True,'case_id':key,'request_sha256':t.digest(request),
            'plan_sha256':producer_pin['sha256'],'history':dict(history,eof_seen=True,crc_verified=True),
            'observed_history':history,'scientific_progress':{'failed_gates':[],'first_failure':None}}
        descriptors[key]={'evidence':write_pin(folder/(key+'Evidence.json'),manifest),
                          'readback':write_pin(folder/(key+'Readback.json'),receipt)}
    manifest=write_pin(folder/'ToyComparison.json',{'schema':c.REQUEST_SCHEMA,'cases':descriptors,
                                                 'producer_run_plan':producer_pin})
    channel=folder/'Channels';channel.mkdir()
    execution={'manifest':manifest,'max_record_bytes':65536,'max_logical_bytes':32*1024**2,
        'max_records':2000,'wall_s':30,'max_retained_bytes':32*1024**2,'max_output_bytes':32*1024**2,
        'rss_bytes':1024**3,'supervision':supervision_fixture(folder),'channel':0}
    plan={'fixture_only':True,'status':'admitted','comparison_execution':execution,
        'resources':{'reader_logical_byte_cap':32*1024**2,'combined_persisted_case_bytes':1024**3,
                     'rss_bytes_per_case':1024**3,'full_wall_cap_max_s':30},
        'source_freeze':{'frozen_gate_spec':gate,'frozen_evidence_method_contract':method},
        'protocol':{'requested_sample_records':labels},'artifacts':{'channel_directory':str(channel)},'cases':cases}
    return {'plan':plan,'plan_sha256':t.digest(plan)},histories


@pytest.mark.parametrize('missing_sample',[False,True])
def test_compare_plan_reads_each_manufactured_sealed_stream_once(tmp_path,monkeypatch,missing_sample):
    import fcntl
    from scripts.benchmarks import native_history as history
    context,paths=sealed_comparison_fixture(tmp_path,monkeypatch,missing_sample=missing_sample)
    original=history.HistoryReader;opened=[]
    def reader(path,**kwargs):
        opened.append(str(path));assert str(path) in paths
        return original(path,**kwargs)
    monkeypatch.setattr(history,'HistoryReader',reader)
    with (tmp_path/'Channels/0.lock').open('x+b') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        monkeypatch.setenv('SOLARLAB_G2_ROUTE_CHANNEL_FD',str(held.fileno()))
        result=c.compare_plan(context)
    assert opened==paths
    normalized=result['normalized_evidence'];bindings=normalized['binding_check']
    assert bindings['original_native_histories_opened_by_compare'] is False
    assert bindings['observation_records']==(775 if missing_sample else 776)
    if missing_sample:
        assert c.FULL_CASES[1] not in bindings['case_inputs']
        assert any('all_131_ordered_tags' in r['reason'] for r in normalized['missing_records'])
    else:
        compared=normalized['point_comparisons']['N8_Main_vs_Strict']
        assert len(compared['port_tables'])==262 and len(compared['state_tables'])==131*19*5
        assert compared['status']=='unknown'
    assert normalized['matched_window_T1']['device'][c.FULL_CASES[2]]['status']=='unknown_reference_error'
    assert result['scientific_pass'] is False


def test_incomplete_control_family_cannot_keep_Main_unresolved_L_U2(tmp_path):
    from .test_t1_gate import SPEC3
    strict,ref,method,_=strict_reference_files(tmp_path,zero_charge=True)
    main=deepcopy(strict);main['case_id']=c.FULL_CASES[1]
    cases={c.FULL_CASES[1]:main,c.FULL_CASES[2]:strict}
    for value in cases.values():
        value['readback']={'complete_protocol':True,'complete_readback':True,'verified':True,
                          'scientific_progress':{'failed_gates':[],'first_failure':None}}
    data={key:loaded_T1(key) for key in cases}
    for value in data.values():value['manifest']['problem_identity']=strict['problem_identity']
    result=c.compare_loaded(cases,data,spec=SPEC3,method_pin=method,labels=sample_labels(base_request()),reference=ref,
        bindings={'physical_context':{'n_ref_m3':100,'p_ref_m3':200,'c_ref_m3':300}})
    main_result=result['normalized_evidence']['matched_window_T1']['device'][c.FULL_CASES[1]]
    assert main_result['status']=='unknown' and main_result['reason']=='unknown_matched_window_growth'
    assert main_result['local_certificate_status']=='unresolved_signal'
    assert result['L_U2']['eligible'] is False


def test_metal_hypothesis_uses_common_windows_and_never_awards_two_grid_order():
    results={key:c.evaluate_T1(loaded_T1(key)) for key in c.FULL_CASES}
    for key,result in results.items():
        ratio=F(4,100000) if key==c.FULL_CASES[3] else F(15,10**6)
        for window in result['fixed_windows']:
            window['independent_window_charge_enclosures_C']={name:c.enc(1) for name in ('left_metal','right_metal')}
            for name in ('left_metal','right_metal'):
                window['channels'][name].update(signed_balance_C=t.pair(ratio),uncertainty_C=t.pair(0))
    value=c.matched_windows(results)
    assert len(value['metal_ratio_hypothesis'])==4*128*2
    for row in value['metal_ratio_hypothesis']:
        assert row['hypothesis_compatible'] is (row['case_id']!=c.FULL_CASES[3])
        assert row['systematic_floor_proved'] is False and row['convergence_order_estimated'] is False
        assert 'left' in row and 'right' in row
    results[c.FULL_CASES[3]]['fixed_windows'][5]['right']['time_hex']=float(.199).hex()
    with pytest.raises(c.ComparisonError,match='matched_time_window_tags'):
        c.matched_windows(results)


def test_comparison_first_failure_is_ordered_across_ports_and_states():
    main,strict=case(c.FULL_CASES[1]),case()
    main['samples'][0]['states']['phi_V'][0]=c.enc(F(1,100))
    main['samples'][-1]['ports']['left']=c.enc(100)
    result=time_compare(main,strict)
    first=result['summary']['first_failure']
    assert first['tag']==main['samples'][0]['tag'] and first['field']=='phi_V'


def test_sample_normalization_preserves_lowest_mapped_word_and_raw_DC():
    from .test_native_readback import tag as encode
    count=9;size=count*5
    layers=[2.**(-10-55*i) for i in range(12)];tiny=layers[-1]
    roots={'phi_V':0.,'n_m3':100.,'p_m3':200.,'c_m3':300.,'f':.5}
    offsets={name:[i*count,(i+1)*count] for i,name in enumerate(c.FIELDS)}
    request={'z0':[0.]*size,'frame_input_policy':{'mapped_input_words':12,'raw_parent_words':4},
        'voltage_lift_map':{'physical_model':'manufactured-model'},'numeric_packet':{
            'variable_offsets':offsets,'initial_reference':{'payload':{'fields':{
                name:{'high':[value.hex()]*count,'low':[(-0.).hex()]*count} for name,value in roots.items()}}}}}
    words=[[layer.hex()]*size for layer in layers]
    history={'physical_cumulative_words_hex':words,'physical_rate_words_hex':words,
        'physical_model_identity':'manufactured-model','point_identity':'toy-point','map_identity':'toy-map'}
    sample={'record_sha256':'toy-record','history':history,
            'raw':{'total_inward':{'high_hex':[2.0.hex(),(-2.0).hex()],
                                   'low_hex':[tiny.hex(),(-tiny).hex()]}}}
    certificate={'sample_record_sha256':'toy-record','evidence':encode({'point_identity':'toy-point',
        'readback':{'physical_state_absolute_error':(F(0),)*size,'do_not_add_same_input_error_twice':True},
        'source_and_readback_error_A':((F(0),F(0)),)*2,'input_error_included_once':True})}
    key=sample_labels(base_request())[0]
    output=c.normalize_ida_sample(request,key,sample,certificate)
    for name,root in roots.items():
        exact=F(root)+sum(map(F,layers),F(0))
        assert all(t.q(x['center'])==exact for x in output['states'][name])
        assert exact!=F(root)+sum(map(F,layers[:-1]),F(0))
    assert t.q(output['ports']['left']['center'])==2+F(tiny)
    assert t.q(output['ports']['right']['center'])==-2-F(tiny)
    assert output['current_origin']=='raw full mapped RateView; no initial subtraction'


def test_H_F_W01_short_E_R_is_unknown_and_complete_structured_v1_works(tmp_path):
    _,reference=referenced_cases();result,review,method=error_files(tmp_path,reference)
    args=dict(method_pin=method,problem_identity=reference['reference_error_problem_identity'],labels=sample_labels(base_request()))
    complete=c.read_error_records(result,review,**args)
    assert complete['status']=='accepted_for_E2' and len(complete['accepted_quantities'])==4
    value=t._pin(review['path'],review)
    short={key:value[key] for key in ('schema','reviewed_error_file','accepted_quantities','claim_kind','verdict','conditions')}
    short['reviewer']='toy-independent-science'
    for reviewer in ('toy-independent-science',value['reviewer']):
        short['reviewer']=reviewer
        pin=write_pin(tmp_path/'ShortErrorReview.json',short)
        rejected=c.read_error_records(result,pin,**args)
        assert rejected['status']=='unknown_reference_error' and rejected['reason']=='acceptance_required_fields'
        assert all(not row['usable_for_E2'] for row in rejected['quantities'].values())


@pytest.mark.parametrize('status,usable',[('satisfied',True),('supported_under_U1',True),('not_applicable',True),('open',False),('violated',False)])
def test_H_F_W02_condition_status_is_consumed(tmp_path,status,usable):
    _,reference=referenced_cases();result,review,method=error_files(tmp_path,reference)
    value=t._pin(review['path'],review)
    evidence=write_pin(tmp_path/'ConditionEvidence.json',{'fixture_only':True,'statement':'bounded toy premise checked'})
    value['conditions']=[{'id':'C1','statement':'toy premise, no quantitative waiver','status':status,'evidence':[evidence]}]
    review=write_pin(Path(review['path']),value)
    found=c.read_error_records(result,review,method_pin=method,
        problem_identity=reference['reference_error_problem_identity'],labels=sample_labels(base_request()))
    assert all(row['usable_for_E2'] is usable for row in found['quantities'].values())
    if not usable:assert found['status']=='unknown_reference_error' and 'condition_unresolved' in found['reason']


def test_H_F_W03_short_Strict_is_unknown_and_complete_structured_v1_works(tmp_path):
    strict,wrapper,method,_=strict_reference_files(tmp_path)
    args=dict(spec=SPEC3,labels=sample_labels(base_request()),problem_identity=strict['problem_identity'])
    assert t.accepted_strict_E2(wrapper,**args)['status']=='accepted'
    complete=t._pin(wrapper['independent_review']['path'],wrapper['independent_review'])
    short={key:complete[key] for key in ('schema','reviewed_e2_file','inputs','accepted_quantities','claim_kind','verdict','conditions')}
    for reviewer in ('toy-independent-science',complete['reviewer']):
        short['reviewer']=reviewer
        wrapper['independent_review']=write_pin(tmp_path/'ShortStrictReview.json',short)
        found=t.accepted_strict_E2(wrapper,**args)
        assert found['status']=='unknown_reference_error' and found['accepted_quantities']==[] and found['reason']


@pytest.mark.parametrize('damage',['version_missing','version_bool','reviewer_string','reviewer_independent_alias','review_pin',
    'keys_unsorted','keys_duplicate','key_wildcard','missing_binding','unit','value_float','scope','parameters','lineage',
    'history_source','history_count_bool','history_closed','history_pin','BE_result_schema','condition_pin','radius_path','method_status'])
def test_H_E_R_structured_fields_are_fail_closed_without_throwing(tmp_path,damage):
    _,reference=referenced_cases();result,review,method=error_files(tmp_path,reference)
    value=t._pin(review['path'],review)
    if damage=='version_missing':del value['version']
    elif damage=='version_bool':value['version']=True
    elif damage=='reviewer_string':value['reviewer']='toy-independent-science'
    elif damage=='reviewer_independent_alias':value['reviewer']['independent']=1
    elif damage=='review_pin':value['reviewer']['review_record']['sha256']='0'*64
    elif damage=='keys_unsorted':value['accepted_quantities'].reverse()
    elif damage=='keys_duplicate':value['accepted_quantities'].append('q')
    elif damage=='key_wildcard':value['accepted_quantities']=['state/*']
    elif damage=='missing_binding':del value['quantity_bindings']['q']
    elif damage=='unit':value['quantity_bindings']['q']['coverage']['unit']='A'
    elif damage=='value_float':value['quantity_bindings']['q']['value']=0.0
    elif damage=='scope':value['scope']['route']='G2-Q'
    elif damage=='parameters':value['parameters']=None
    elif damage=='lineage':value['lineage']['sha256']='0'*64
    elif damage=='history_source':value['reference_history']['source_identity']=t.digest('other physics')
    elif damage=='history_count_bool':value['reference_history']['records']=True
    elif damage=='history_closed':value['reference_history']['container_complete']=False
    elif damage=='history_pin':value['reference_history']['independent_readback']['sha256']='0'*64
    elif damage=='BE_result_schema':
        pin=value['reference_history']['run_result'];result_body=t._pin(pin['path'],pin)
        result_body['schema']='not-the-BE-result-schema'
        value['reference_history']['run_result']=write_pin(Path(pin['path']),result_body)
    elif damage=='radius_path':value['quantity_bindings']['q']['radius_accounting']['r_R_source']='per_record/computed_value/radius'
    elif damage=='condition_pin':
        value['conditions']=[{'id':'C1','statement':'toy premise','status':'satisfied','evidence':[None]}]
    else:
        payload=t._pin(result['path'],result);pin=payload['quantity_records']['q'];record=t._pin(pin['path'],pin)
        record['per_record'][17]['computation']['method_status']='failed'
        new=write_pin(Path(pin['path']),record);payload['quantity_records']['q']=new
        result=write_pin(Path(result['path']),payload);value['reviewed_error_file']=result
        value['quantity_bindings']['q']=quantity_binding(new,'q',sample_labels(base_request()))
    review=write_pin(Path(review['path']),value)
    found=c.read_error_records(result,review,method_pin=method,
        problem_identity=reference['reference_error_problem_identity'],labels=sample_labels(base_request()))
    assert found['quantities'].get('q',{'usable_for_E2':False})['usable_for_E2'] is False
    assert found['reason'] or found['quantities']['q']['reason']


def test_H_E_R_radius_ownership_is_checked_at_each_tag_not_only_the_maximum(tmp_path):
    _,reference=referenced_cases();result,review,method=error_files(tmp_path,reference)
    payload=t._pin(result['path'],result);value=t._pin(review['path'],review)
    pin=payload['quantity_records']['q'];record=t._pin(pin['path'],pin)
    record['per_record'][0]['computation']['E_R']=[2,1000000]
    record['per_record'][1]['computed_value']['radius']=[2,1000000]
    record['conditional_estimate_value']=[2,1000000]
    record['estimated_or_enclosed_value']['value']=[2,1000000]
    updated=write_pin(Path(pin['path']),record);payload['quantity_records']['q']=updated
    result=write_pin(Path(result['path']),payload);value['reviewed_error_file']=result
    value['quantity_bindings']['q']=quantity_binding(updated,'q',sample_labels(base_request()))
    review=write_pin(Path(review['path']),value)
    found=c.read_error_records(result,review,method_pin=method,
        problem_identity=reference['reference_error_problem_identity'],labels=sample_labels(base_request()))
    assert not found['quantities']['q']['usable_for_E2']
    assert found['quantities']['q']['reason']=='reference_error_tagwise_radius_not_owned'
    assert found['quantities']['f']['usable_for_E2']


def test_H_IDA_reader_tail_count_rejects_boolean_alias(tmp_path):
    strict,wrapper,method,_=strict_reference_files(tmp_path)
    e2=t._pin(wrapper['e2_file']['path'],wrapper['e2_file'])
    binding=e2['inputs']['IDA_Strict_sealed_history']
    pin=binding['independent_readback'];reader=t._pin(pin['path'],pin)
    reader['history']['partial_tail_bytes']=False
    binding['independent_readback']=write_pin(Path(pin['path']),reader)
    with pytest.raises(t.InvalidT1Data,match='acceptance_IDA_container'):
        t._acceptance_history(binding,case_id='B_N8_Strict_Full',labels=sample_labels(base_request()))


@pytest.mark.parametrize('damage',['version','reviewer','conditions','scope','same_problem','different_initial_rates',
    'quantity_record','r_S','r_R','tagwise_pin','different_BE_history','missing_readback','open_E_R','quantity_alias'])
def test_H_Strict_structured_acceptance_rejects_missing_binding_with_reason(tmp_path,damage):
    strict,wrapper,method,_=strict_reference_files(tmp_path)
    review=t._pin(wrapper['independent_review']['path'],wrapper['independent_review'])
    if damage=='version':review['version']=1.0
    elif damage=='reviewer':review['reviewer']=None
    elif damage=='conditions':review['conditions']=[{'id':'C1','statement':'unresolved','status':'open','evidence':[]}]
    elif damage=='scope':review['scope']['reference_case_id']='B_N16_Strict_Full'
    elif damage=='same_problem':review['same_problem_evidence']['sha256']='0'*64
    elif damage=='different_initial_rates':
        ref=review['same_problem_evidence'];proof=t._pin(ref['path'],ref);inp=proof['BE_reference_input']
        request=t._pin(inp['path'],inp);request['initial_segment_frame']['physical_rate_handoff_words_hex'][11][0]=(2.0**-100).hex()
        proof['BE_reference_input']=write_pin(Path(inp['path']),request)
        review['same_problem_evidence']=write_pin(Path(ref['path']),proof)
    elif damage=='quantity_record':review['quantity_bindings']['q']['reviewed_record']['sha256']='0'*64
    elif damage=='r_S':review['quantity_bindings']['q']['radius_accounting']['r_S_included_in_E_S']=False
    elif damage=='r_R':review['quantity_bindings']['q']['radius_accounting']['r_R_included_in_E_R']=1
    elif damage=='tagwise_pin':review['quantity_bindings']['q']['radius_accounting']['tagwise_ownership_record']=None
    elif damage=='different_BE_history':review['inputs']['BE_reference_history']['logical_sha256']='0'*64
    elif damage=='missing_readback':del review['inputs']['IDA_Strict_sealed_history']['independent_readback']
    elif damage=='quantity_alias':review['accepted_quantities']=['state/phi_V/00']
    else:
        pin=review['inputs']['E_R_acceptance'];er=t._pin(pin['path'],pin)
        er['conditions']=[{'id':'C1','statement':'unresolved','status':'open','evidence':[]}]
        write_pin(Path(pin['path']),er)
    wrapper['independent_review']=write_pin(Path(wrapper['independent_review']['path']),review)
    found=t.accepted_strict_E2(wrapper,spec=SPEC3,labels=sample_labels(base_request()),problem_identity=strict['problem_identity'])
    assert found['status']=='unknown_reference_error' and found['reason'] and found['accepted_quantities']==[]


@pytest.mark.parametrize('damage',['both_x100','q_x100','f_x100','missing_table','missing_q','radius','tag'])
def test_I_Strict_responses_must_match_verified_tagwise_values(tmp_path,damage):
    strict,wrapper,_,e2=strict_reference_files(tmp_path)
    e2=t._pin(wrapper['e2_file']['path'],wrapper['e2_file'])
    args=dict(spec=SPEC3,labels=sample_labels(base_request()),problem_identity=strict['problem_identity'])
    assert t.reference_status_v3(wrapper,**args)['status']=='eligible'
    original_quantities=t.digest(e2['quantities'])
    if damage in {'both_x100','q_x100','f_x100'}:
        fields=('q_C','f_C') if damage=='both_x100' else (damage[0]+'_C',)
        for response in e2['strict_responses'][1:]:
            for field in fields:response[field]['center']=t.pair(100*t.q(response[field]['center']))
    elif damage=='missing_table':del e2['strict_responses']
    elif damage=='missing_q':del e2['strict_responses'][73]['q_C']
    elif damage=='radius':e2['strict_responses'][73]['f_C']['absolute_error_bound']=[1,1000000]
    else:e2['strict_responses'][73]['tag']['event_side']='left'
    assert t.digest(e2['quantities'])==original_quantities
    wrapper['e2_file']=write_pin(Path(wrapper['e2_file']['path']),e2)
    review=t._pin(wrapper['independent_review']['path'],wrapper['independent_review'])
    review['reviewed_e2_file']=wrapper['e2_file']
    wrapper['independent_review']=write_pin(Path(wrapper['independent_review']['path']),review)
    accepted=t.accepted_strict_E2(wrapper,**args)
    assert accepted['status']=='unknown_reference_error' and accepted['accepted_quantities']==[] and accepted['reason']
    result=t.reference_status_v3(wrapper,**args)
    assert result['status']=='unknown' and result['reason']=='unknown_reference_error' and result['Qsig_C'] is None
