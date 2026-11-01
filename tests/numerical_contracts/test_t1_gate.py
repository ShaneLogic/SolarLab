"""Manufactured exact GateSpec V2/V3 evidence; no physical/native simulation."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmarks import t1_gate as t
from scripts.benchmarks.g2_route_bindings import sample_labels
from scripts.benchmarks.native_history import HistoryWriter, integer_json_dumps, decode_integer_values


SPEC = t.CandidateSpec("manufactured-spec", F(10**8))  # Q0=one normalized charge unit


def enc(center, radius=0):
    return {"center": t.pair(center), "absolute_error_bound": t.pair(radius)}


def op(delta=0, flux=0, *, errors=(0,0,0,0), l1=None, missing_l1=False):
    errors = tuple(map(F, errors))
    bound = abs(F(flux)) if l1 is None else F(l1)
    return {"delta_Q_enclosure_C": enc(delta, errors[0]), "integral_F_enclosure_C": enc(flux, errors[1]),
            "uncertainty_components_C": {k:t.pair(v) for k,v in zip(t.COMPONENTS,errors,strict=True)},
            "quadrature_component_upper_C": t.pair(errors[1]),
            "integral_abs_F_lower_C": None if missing_l1 else t.pair(bound),
            "integral_abs_F_upper_C": None if missing_l1 else t.pair(bound)}


def base_request():
    segments = [dict(id="dark_equilibrium_hold",start=0.,end=.1,voltage=[0,0],photons=[0,0]),
                dict(id="voltage_ramp",start=.1,end=.2,voltage=[0,.1],photons=[0,0]),
                dict(id="slow_state_hold",start=.2,end=9.2,voltage=[.1,.1],photons=[0,0])]
    times = {s["id"]:[s["start"]+(s["end"]-s["start"])*i/(count-1) for i in range(count)]
             for s,count in zip(segments,(33,33,65))}
    for s in segments: times[s["id"]][-1] = s["end"]
    return {"segments":segments,"observation_times":times,
            "numeric_packet":{"definition":{"fixture":"manufactured exact charge path"},"x_m":[0,1],
                              "physical_reference_fields":{"q":{"high":[0.],"low":[-0.]}}},
            "initial_segment_frame": {key:[["0x0.0p+0"] for _ in range(12)] for key in
                                      ("physical_handoff_words_hex","physical_rate_handoff_words_hex")},
            "voltage_lift_map":{"lift_hex":["0x0.0p+0"],"reference_inputs_hex":["0x0.0p+0"]}}


def full_data(*, failure=None, missing_l1=False):
    request = base_request(); labels = sample_labels(request)
    intervals, pieces = [], []
    for i,(left,right,_,_) in enumerate(t.fixed_windows(labels)):
        channels = {name:op(missing_l1=missing_l1) for name in t.CHANNELS}
        if i == failure:
            channels["left_metal"] = op(1,0,missing_l1=missing_l1)
        row = {"run_id":"manufactured-run","request_sha256":t.digest(request),
               "native_interval_id":f"native-{i}","segment_id":left["segment"],"event_side":right["event_side"],
               "t_left_hex":left["time_hex"],"t_right_hex":right["time_hex"],
               "source_id":"manufactured-source","frame_id":f"frame-{i}","map_id":"manufactured-map","channels":channels}
        intervals.append(row)
        pieces.append(dict(deepcopy(row),window_index=i,parent_sha256=t.digest(row)))
    return request, labels, intervals, pieces


def evaluate(data, **kwargs):
    request, labels, intervals, pieces = data
    return t.evaluate(intervals,pieces,labels=labels,spec=SPEC,case_id="B_N8_Main_Full",run_id="manufactured-run",
                      problem_identity=t.physical_problem_identity(request),**kwargs)


def pin(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode();path.write_bytes(raw)
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def reference_fixture(folder, data, spec=SPEC):
    """Independent exact linear toy q=f=t/T, explicitly NOT a B reference."""
    request, labels, _, _ = data
    rows = [{"tag":tag,"q_C":enc(F(float.fromhex(tag["time_hex"]))/F(9.2)),
             "f_C":enc(F(float.fromhex(tag["time_hex"]))/F(9.2)),"E_q_C":[0,1],"E_f_C":[0,1]} for tag in labels]
    proof = {"fixture_only":True,"exact_toy_IVP":"q'=f'=1/T; q(0)=f(0)=0",
             "problem_identity":t.physical_problem_identity(request),"kind":"analytic_original_problem", "unit":"C",
             "labels_sha256":t.digest(labels),"bound_table_sha256":t.digest(rows),
             "error_coverage":["evaluation","initial_and_algebraic","input_clock_parameter","trajectory_nonlinear","stability_reconstruction"]}
    reference = {"fixture_only":True,"case_id":"B_N8_Strict_Full","problem_identity":proof["problem_identity"],
                 "unit":"C","labels":labels,"proof_kind":proof["kind"],"responses":rows,
                 "proof":pin(folder/"ToyProof.json",proof)}
    review = {"fixture_only":True,"accepted":True,"independent_forward_error_review":True,
              "evidence_sha256":t.digest(reference),"proof_sha256":reference["proof"]["sha256"],"gate_spec_sha256":spec.sha256}
    reference["independent_review"] = pin(folder/"ToyReview.json",review)
    return reference


def test_R1_common_bias_is_not_qualified_by_old_numeric_band():
    delta=F(1,10**7)
    coarse,main,strict,true = 1+3*delta,1+delta,F(1),F(0)
    heuristic=2*abs(main-strict)
    assert abs(main-strict) <= abs(coarse-main)/2
    assert t.resolution_conditions(1,heuristic,heuristic,SPEC)
    assert not t.resolution_conditions(1,abs(strict-true),abs(strict-true),SPEC)
    result=t.reference_status({"proof_kind":"2*dMS","accepted":True},spec=SPEC,labels=sample_labels(base_request()),problem_identity="toy")
    assert result["reason"] == "unknown_reference_error"


def test_R1_upper_upper_false_contraction_and_exact_zero():
    assert F(3,4) <= F(2)/2  # old upper/upper check falsely says contraction
    result=t.observed_contraction((1,2),(F(3,4),F(3,4)))
    assert result["status"] == "unknown_contraction" and not result["forward_error_qualified"]
    assert t.observed_contraction((2,2),(1,1))["status"] == "certified_observed_contraction"
    assert t.observed_contraction((0,0),(0,0))["status"] == "exact_zero"


def test_R2_turnover_does_not_enlarge_device_net_cap():
    ledger=t.ChargeLedger()
    for value in (op(1001,1000),op(999,1000)):
        assert t.assess_interval(value,floor=SPEC.eta,epsilon=SPEC.epsilon_device)["status"] == "passed"
        ledger.append(t.operands(value),SPEC.eta)
    assert ledger.result(floor=SPEC.eta,epsilon=SPEC.epsilon_device)["status"] == "passed"
    result=ledger.result(floor=SPEC.eta,epsilon=SPEC.epsilon_device,cap=F(1))
    assert result["status"] == "failed" and t.q(result["defect_C"]) == 2


def test_R3_repeated_floor_and_fixed_window_counterexample():
    ledger=t.ChargeLedger()
    for _ in range(100):
        value=op(F(1,1000),0)
        assert t.assess_interval(value,floor=1,epsilon=SPEC.epsilon_metal)["status"] == "passed"
        ledger.append(t.operands(value),F(1))
    assert ledger.D <= SPEC.epsilon_metal*ledger.old_scale_lower
    assert ledger.result(floor=1,epsilon=SPEC.epsilon_metal)["status"] == "failed"
    assert ledger.result(floor=1,epsilon=SPEC.epsilon_metal,window=True)["status"] == "failed"


def test_C2_four_components_once_and_quadrature_is_a_subset():
    value=op(1,1,errors=(F(1,100),F(2,100),F(3,100),F(4,100)),l1=1)
    result=t.assess_interval(value,floor=1,epsilon=1)
    assert t.q(result["defect_C"]) == F(1,10)
    assert t.q(result["quadrature_C"]) == F(2,100)
    value["quadrature_component_upper_C"]=[0,1]
    with pytest.raises(t.InvalidT1Data,match="inseparable_quadrature"):
        t.operands(value)


@pytest.mark.parametrize("D,quad,lower,upper,status", [(1,0,1,2,"passed"),(3,0,1,2,"failed"),(F(3,2),0,1,2,"unknown"),
    (0,1,1,2,"failed"),(0,F(1,2),1,2,"unknown"),(1,0,1,None,"passed"),(2,0,1,None,"unknown")])
def test_lower_pass_upper_fail_and_unknown(D,quad,lower,upper,status):
    assert t.verdict(D,quad,lower,upper,F(1))["status"] == status


def test_valid_lower_proves_pass_even_without_L1_upper():
    assert t.assess_interval(op(0,0,missing_l1=True),floor=1,epsilon=SPEC.epsilon_metal)["status"] == "passed"
    assert t.assess_interval(op(1,0,missing_l1=True),floor=1,epsilon=SPEC.epsilon_metal)["status"] == "unknown"


def test_reference_proposal_1_5_metals_pass_device_exceeds_cap_1000_times():
    values=(op(2,1),op(999,F(1999,2)),op(-1001,F(-2001,2)))
    assert sum(t.operands(v)["dc"] for v in values) == 0
    assert sum(t.operands(v)["fc"] for v in values) == 0
    for value in values[1:]:
        assert t.assess_interval(value,floor=SPEC.Q0,epsilon=SPEC.epsilon_metal)["status"] == "passed"
    ledger=t.ChargeLedger();ledger.append(t.operands(values[0]),SPEC.eta)
    result=ledger.result(floor=SPEC.eta,epsilon=SPEC.epsilon_device,cap=F(1))
    assert result["status"] == "failed"
    assert t.q(result["defect_C"])/(SPEC.epsilon_device*t.q(result["scale_upper_C"])) == 1000


def test_all_128_windows_and_unknown_reference_without_reset():
    data=full_data(missing_l1=True)
    result=evaluate(data)
    assert len(result["native"]) == len(result["fixed_windows"]) == 128
    assert result["channels"]["left_metal"]["status"] == result["channels"]["right_metal"]["status"] == "passed"
    assert result["channels"]["device"]["status"] == "unknown"
    assert result["reference"]["reason"] == "unknown_reference_error"
    assert result["native"][-1]["channels"]["left_metal"]["prefix"]["terms"] == 128
    assert result["scientific_qualification"] is False


def test_first_failure_survives_later_windows_events_and_zero_balances():
    result=evaluate(full_data(failure=17))
    assert result["channels"]["left_metal"]["first_failure"]["index"] == 17
    assert result["channels"]["left_metal"]["status"] == "failed"
    assert result["native"][-1]["channels"]["left_metal"]["prefix"]["status"] == "failed"
    assert len(result["native"]) == 128


def test_missing_pieces_are_unknown_not_whole_interval_balance():
    data=full_data();data[-1].pop(5)
    result=evaluate(data)
    assert result["fixed_windows"][5]["channels"]["left_metal"]["reason"] == "unknown_missing_window_pieces"
    assert result["channels"]["left_metal"]["status"] == "unknown"


def test_piece_positive_debt_cannot_cancel_like_its_native_interval():
    data=full_data();parent=data[2][5];original=data[-1].pop(5)
    middle=float((F(float.fromhex(parent["t_left_hex"]))+F(float.fromhex(parent["t_right_hex"])))/2).hex()
    a,b=deepcopy(original),deepcopy(original);a["t_right_hex"]=middle;b["t_left_hex"]=middle
    a["channels"]["left_metal"]=op(1,0);b["channels"]["left_metal"]=op(-1,0)
    data[-1][5:5]=[a,b]
    result=evaluate(data)
    assert result["native"][5]["channels"]["left_metal"]["interval"]["status"] == "passed"
    assert result["fixed_windows"][5]["channels"]["left_metal"]["status"] == "failed"
    assert t.q(result["fixed_windows"][5]["channels"]["left_metal"]["defect_C"]) == 2


def test_pieces_must_split_refresh_boundary():
    data=full_data();parent=data[2][70]
    middle=float((F(float.fromhex(parent["t_left_hex"]))+F(float.fromhex(parent["t_right_hex"])))/2).hex()
    with pytest.raises(t.InvalidT1Data,match="piece_crosses_refresh"):
        evaluate(data,refresh_times=[middle])
    original=data[-1].pop(70);a,b=deepcopy(original),deepcopy(original)
    a["t_right_hex"]=middle;b["t_left_hex"]=middle;data[-1][70:70]=[a,b]
    assert evaluate(data,refresh_times=[middle])["fixed_windows"][70]["complete_pieces"]


@pytest.mark.parametrize("damage", ["gap","overlap","parent","integral","L1","side","index_bool","request"])
def test_piece_clock_identity_and_additivity_are_checked(damage):
    data=full_data()
    if damage == "gap": data[-1][3]["t_left_hex"]=data[-1][3]["t_right_hex"]
    elif damage == "overlap": data[-1].insert(3,deepcopy(data[-1][3]))
    elif damage == "parent": data[-1][3]["parent_sha256"]="foreign"
    elif damage == "integral": data[-1][3]["channels"]["left_metal"]=op(1,0)
    elif damage == "L1": data[-1][3]["channels"]["left_metal"]=op(0,0,l1=1)
    elif damage == "side": data[1][32]["event_side"]="right"
    elif damage == "request": data[-1][3]["request_sha256"]="foreign"
    else: data[1][0]["index"]=False
    with pytest.raises(t.InvalidT1Data): evaluate(data)


def test_bound_reference_contract_and_N16_nonreuse(tmp_path):
    data=full_data();ref=reference_fixture(tmp_path,data)
    result=evaluate(data,reference=ref)
    assert result["reference"]["status"] == "passed"
    assert result["reference"]["proof_recomputed_by_this_gate"] is False
    assert result["channels"]["device"]["status"] == "passed" and not result["scientific_qualification"]
    result=t.evaluate(data[2],data[3],labels=data[1],spec=SPEC,case_id="B_N16_Strict_Full",run_id="manufactured-run",
                      problem_identity=t.physical_problem_identity(data[0]),reference=ref)
    assert result["reference"]["reason"] == "unknown_reference_error"


@pytest.mark.parametrize("damage", ["method","table","labels","review","problem"])
def test_a_label_or_unbound_reference_is_not_qualified(tmp_path,damage):
    data=full_data();ref=reference_fixture(tmp_path,data)
    if damage == "method": ref["proof_kind"]="2*dMS"
    elif damage == "table": ref["responses"][4]["E_q_C"]=[-1,1]
    elif damage == "labels": ref["labels"]=deepcopy(ref["labels"]);ref["labels"][4]["index"]=999
    elif damage == "review": Path(ref["independent_review"]["path"]).write_text('{"accepted":true}')
    else: ref["problem_identity"]="different_initial_problem"
    assert evaluate(data,reference=ref)["reference"]["reason"] == "unknown_reference_error"


@pytest.mark.parametrize("bad", [True,False,0.0,float('nan'),[1,0],[True,1]])
def test_bool_float_and_invalid_rational_are_rejected(bad):
    with pytest.raises(t.InvalidT1Data): t.q(bad)


def test_low_words_and_existing_integer_codec_are_lossless(tmp_path):
    tiny=F(1,2**120)+F(1,2**240)
    result=t.assess_interval(op(1+tiny,1),floor=1,epsilon=SPEC.epsilon_metal)
    assert t.q(result["signed_balance_C"]) == tiny
    value={"ratio":t.pair(F(2**16000,2**16000+1))}
    text=integer_json_dumps(value,t.INTEGER_POLICY)
    assert decode_integer_values(json.loads(text),t.INTEGER_POLICY) == value
    path=tmp_path/"ExactReferenceInteger.json";path.write_text(text)
    assert t._pin(path,{"bytes":len(text.encode()),"sha256":hashlib.sha256(text.encode()).hexdigest()}) == value


def document():
    return {"schema":t.SPEC_SCHEMA,"version":2,"status":"proposed_not_frozen",
        "parameters":{"epsilon_metal":[1,1000],"epsilon_device_proposal":[1,1000],
            "device_signal_floor_fraction":[1,100],"quadrature_share":[1,3],"reference_error_share":[1,3],"legacy_absolute_fraction":[1,10**8],
            "resolution_multiplier":[3,1],"denominator_partition_max_depth":8,"denominator_partition_max_leaves":256},
        "physical_context":{"Qref_C_exact":str(10**8),"legacy_absolute_C_exact":"1"}}


@pytest.mark.parametrize("key,value", [("resolution_multiplier",[4,1]),
    ("denominator_partition_max_depth",8.0),("denominator_partition_max_leaves",257)])
def test_candidate_rejects_changed_resolution_or_partition_parameters(key,value):
    doc=document();doc["parameters"][key]=value
    with pytest.raises(t.InvalidT1Data): t.CandidateSpec.from_document(doc,"manufactured-spec")


def sealed_fixture(folder, *, missing_pieces=False, v3=False):
    request,labels,intervals,pieces=data_v3() if v3 else full_data()
    if missing_pieces: pieces=[]
    writer=HistoryWriter(folder/"Manufactured.jsonl.gz",encoding="gzip",total_output_bytes=4*1024**2,max_record_bytes=16384)
    for kind,rows in (("t1_interval_evidence",intervals),("t1_fixed_window_piece",pieces)):
        for row in rows: writer.append({"kind":kind,"evidence":row})
    writer.finish()
    spec_doc=document()
    if v3: spec_doc.update(schema=t.SPEC_SCHEMA_V3,version=3)
    manifest={"schema":t.EVIDENCE_SCHEMA,"run_id":"manufactured-run","case_id":"B_N8_Main_Full",
        "source_id":"manufactured-source","problem_identity":t.physical_problem_identity(request),
        "gate_spec":pin(folder/"SpecFixture.json",spec_doc),"request":pin(folder/"RequestFixture.json",request),
        "history":{"path":str(writer.path),"encoding":"gzip","closed":True,"container_complete":True,
            "encoded_bytes":writer.encoded_bytes,"logical_bytes":writer.logical_bytes,"records":writer.records,
            "logical_sha256":writer.logical_sha256}}
    path=folder/"EvidenceFixture.json";pin(path,manifest)
    return path,manifest


def read_sealed(path, **kwargs):
    return t.evaluate_sealed(path,max_record_bytes=16384,max_logical_bytes=4*1024**2,max_records=300,wall_s=10,**kwargs)


def test_genuine_HistoryReader_roundtrip_EOF_CRC_all_windows(tmp_path):
    path,_=sealed_fixture(tmp_path)
    result=read_sealed(path)
    assert result["history"]["records"] == 256 and result["history"]["eof_seen"] and result["history"]["crc_verified"]
    assert len(result["fixed_windows"]) == 128 and result["channels"]["left_metal"]["status"] == "passed"
    assert result["channels"]["device"]["status"] == "unknown"


@pytest.mark.parametrize("damage", ["trailer","logical_hash","record_count","source","state_word","retained_limit"])
def test_sealed_data_identity_and_bounds_fail_closed(tmp_path,damage):
    path,manifest=sealed_fixture(tmp_path)
    options={}
    if damage == "trailer":
        p=Path(manifest["history"]["path"]);p.write_bytes(p.read_bytes()[:-5]);manifest["history"]["encoded_bytes"]=p.stat().st_size
    elif damage == "logical_hash": manifest["history"]["logical_sha256"]="0"*64
    elif damage == "record_count": manifest["history"]["records"]-=1
    elif damage == "source": manifest["source_id"]="wrong_source"
    elif damage == "state_word":
        p=Path(manifest["request"]["path"]);request=json.loads(p.read_text());request["initial_segment_frame"]["physical_rate_handoff_words_hex"][-1][0]=(2.**-240).hex()
        manifest["request"]=pin(p,request)
    else: options["max_output_bytes"]=256
    pin(path,manifest)
    with pytest.raises((ValueError,EOFError)): read_sealed(path,**options)


def test_current_admission_entry_stays_disconnected():
    from scripts.benchmarks import g2_route
    assert g2_route.T1_SPEC_SCHEMA == "solarlab.g2-route.t1-gatespec.v3"
    assert "t1_gate.py" in g2_route.EVALUATOR_FILES
    assert g2_route._evaluation_blockers({}) == ["evaluation_not_admitted"]


def sealed_observer_row():
    from scripts.benchmarks.interval_observation import Enclosure
    from scripts.benchmarks.native_observation import observation_record
    clock={"predecessor":[0,1],"tn":t.pair(F(.1)),"segment_end":t.pair(F(.1))}
    zero=(F(0),)*3
    physical={"schema":"solarlab.native-physical-charge.v1",
        "physical_flux_basis":"fixed-uniform-epsilon-Dirichlet-slab-reservoir-carrier-ion-trap-capacitance-v1",
        "source_identity":"manufactured-source","map_identity":"map","frame_identity":"frame",
        "path_identity":"path","clock":clock,"request_sha256":"request",
        "saved_finite_charge_change_C":(Enclosure(0),Enclosure(1),Enclosure(0)),
        "physical_flux_integral_C":tuple(Enclosure(0) for _ in range(3)),
        "error_components_C":{k:zero for k in ("saved_charge_evaluation","physical_flux_integration","input","clock_strip")},
        "signed_balance_C":(F(999),)*3,"passed":True}  # ignored producer claims
    observation={"physical_charge":physical,"path_identity":"path","clock":clock,
                 "raw_tangent_L1_current_bounds":(F(10**9),)*2}
    return {"kind":"sealed_voltage_lift_charge","request_sha256":"request","source_interval_sha256":"interval",
        "interval":{"segment_id":"dark_equilibrium_hold","left":{"time_s":0.},"right":{"time_s":.1},
                    "coefficient_frame_identity":"frame","observation":observation_record(observation)}}


def test_actual_observation_codec_adapter_ignores_flags_and_legacy_L1():
    row=sealed_observer_row()
    value=t.from_sealed_charge(row,run_id="manufactured-run")
    metal=value["channels"]["left_metal"]
    assert metal["integral_abs_F_upper_C"] is None
    assert t.assess_interval(metal,floor=1,epsilon=SPEC.epsilon_metal)["status"] == "unknown"
    assert t.operands(metal)["b"] == 1  # not the claimed producer 999
    scale={"source_record_sha256":t.digest(row),"partition":{"depth":1,"leaves":2},
           "channels":{k:{"lower_C":[1000,1],"upper_C":[1000,1]} for k in t.CHANNELS}}
    metal=t.from_sealed_charge(row,run_id="manufactured-run",scale_evidence=scale)["channels"]["left_metal"]
    assert t.assess_interval(metal,floor=1,epsilon=SPEC.epsilon_metal)["status"] == "passed"


@pytest.mark.parametrize("damage", ["source","depth","leaves","clock","frame","flux_basis"])
def test_actual_adapter_rejects_unbound_scale_clock_and_partition(damage):
    row=sealed_observer_row()
    scale={"source_record_sha256":t.digest(row),"partition":{"depth":1,"leaves":2},
           "channels":{k:{"lower_C":[0,1],"upper_C":[0,1]} for k in t.CHANNELS}}
    if damage == "source": scale["source_record_sha256"]="foreign"
    elif damage == "depth": scale["partition"]["depth"]=9
    elif damage == "leaves": scale["partition"]["leaves"]=257
    elif damage == "clock": row["interval"]["right"]["time_s"] = .09
    elif damage == "frame": row["interval"]["coefficient_frame_identity"]="other"
    else: row["interval"]["observation"]["physical_charge"]["physical_flux_basis"]="raw_metal_derivative"
    with pytest.raises(t.InvalidT1Data):
        t.from_sealed_charge(row,run_id="manufactured-run",scale_evidence=scale)


def test_matched_signed_trend_is_not_a_prefix_or_order_proof():
    assert t.matched_growth(1,F(1,10),F(11,10),0)["status"] == "passed"
    assert t.matched_growth(1,0,F(11,10),0)["status"] == "failed"
    assert t.matched_growth(1,None,1,0)["status"] == "unknown"
    assert t.matched_growth(1,0,-1,0)["convergence_order_proved"] is False


SPEC3 = t.CandidateSpec("manufactured-V3-spec",F(10**8),version=3)
OWNERSHIP = {k:"manufactured independently bounded "+k for k in (*t.COMPONENTS,"Qquad")}


def data_v3():
    request,labels,intervals,pieces=full_data(missing_l1=True)
    for i,row in enumerate(intervals):
        row.update(case_id="B_N8_Main_Full",input_digest=t.digest(request),path_identity=f"path-{i}",
                   owner_generation={"owner":"manufactured-owner","generation":1})
        left,right=map(F,(float.fromhex(row["t_left_hex"]),float.fromhex(row["t_right_hex"])))
        row["native_clock"]={"predecessor":t.pair(left),"tn":t.pair(right),"hused":t.pair(right-left)}
        for value in row["channels"].values():
            value.update(left_charge_enclosure_C=enc(0),right_charge_enclosure_C=enc(0),coverage_ownership=deepcopy(OWNERSHIP))
        piece=deepcopy(row)
        piece.update(schema="solarlab.g2-window-piece.v1",window_index=i,parent_sha256=t.digest(row),
            window_key={"left":t.fixed_windows(labels)[i][0],"right":t.fixed_windows(labels)[i][1]})
        piece["piece_id"]=t.digest(piece);pieces[i]=piece
    return request,labels,intervals,pieces


def evaluate_v3(data,**kwargs):
    return t.evaluate(data[2],data[3],labels=data[1],spec=SPEC3,case_id="B_N8_Main_Full",run_id="manufactured-run",
                      problem_identity=t.physical_problem_identity(data[0]),**kwargs)


def typed_reference(folder,data,*,signal=F(1),error=F(0),kind="rigorous_bound"):
    from .test_g2_route_bindings import frozen_definition
    request,labels,_,_=data;problem=t.physical_problem_identity(request)
    method=frozen_definition(folder,"ReferenceMethod","solarlab.g2-reference-method.v1",1)
    rows=[{"tag":tag,"q_C":enc(signal*F(float.fromhex(tag["time_hex"]))/F(9.2)),
           "f_C":enc(signal*F(float.fromhex(tag["time_hex"]))/F(9.2))} for tag in labels]
    proof=pin(folder/"ToyForwardProof.json",{"fixture_only":True,"kind":"analytic_original_problem",
        "target_problem_identity":problem,"checked_premises":["manufactured exact observable values and declared error"]})
    axes={axis:pin(folder/(axis+"Toy.json"),{"fixture_only":True,"status":"accepted","axis":axis,
        "method_contract_sha256":method["sha256"],"target_problem_identity":problem,
        "checked_premises":["manufactured method support, no actual BE trajectory"]}) for axis in ("E1","E2","E3")}
    evidence={}
    for field,quantity in (("E_q","q"),("E_f","f")):
        slot="rigorous_bound_value" if kind == "rigorous_bound" else "conditional_estimate_value"
        record={"schema":"solarlab.g2-reference-evidence.v1","status":"accepted","evidence_class":"manufactured",
            "claim_kind":kind,"target_problem_identity":problem,"quantity_and_unit":{"quantity":quantity,"unit":"C"},
            "norm_and_coverage":{"norm":"max_abs_tagged131","records":131,"labels_sha256":t.digest(labels)},
            "reference_source_and_run_identity":{"case_id":"B_N8_Strict_Full","run_id":"toy-reference",
                "source_identity":"toy-source","request_sha256":t.digest(request)},
            "error_components_and_ownership":[{"id":"all-toy-errors","claim_kind":kind,"value_C":t.pair(error),
                "covered_axes":["evaluation","initial_and_algebraic","input_clock_parameter","trajectory_nonlinear","stability_reconstruction"],
                "source":proof}],"assumptions_and_checked_premises":["explicit manufactured observable/error"],
            "observed_refinement_and_method_independence":axes if kind == "conditional_uncertainty_estimate" else {},
            "unresolved_common_bias_and_model_error":"manufactured scope only; no physical B assertion",
            "reviewer_and_decision_authority":"manufactured separate reviewer",
            "rigorous_bound_value":None,"conditional_estimate_value":None,"diagnostic_value":None,"proof":proof}
        record[slot]=t.pair(error);record["estimated_or_enclosed_value"]={"value_field":slot,"value":record[slot],"unit":"C"}
        evidence[field]=record
    reference={"fixture_only":True,"case_id":"B_N8_Strict_Full","problem_identity":problem,"unit":"C",
               "labels":labels,"responses":rows,"method_contract":method,"error_evidence":evidence}
    return review_typed_reference(folder,reference)


def review_typed_reference(folder,reference):
    reference.pop("independent_review",None)
    review={"fixture_only":True,"accepted":True,"independent_reference_error_review":True,
        "evidence_sha256":t.digest(reference),"gate_spec_sha256":SPEC3.sha256,
        "method_contract_sha256":reference["method_contract"]["sha256"],
        "reviewer":"toy-independent-reviewer","evidence_author":"toy-author"}
    reference["independent_review"]=pin(folder/"TypedToyReview.json",review)
    return reference


@pytest.mark.parametrize("kind",["rigorous_bound","conditional_uncertainty_estimate"])
def test_V3_typed_reference_claim_is_preserved(tmp_path,kind):
    data=data_v3();ref=typed_reference(tmp_path,data,kind=kind)
    result=evaluate_v3(data,reference=ref)
    assert result["channels"]["device"]["status"] == "passed"
    assert result["reference"]["claim_kind"] == result["channels"]["device"]["claim_kind"] == kind
    assert result["reference"]["E1_E2_E3_recomputed_by_this_gate"] is False
    assert result["L_U2"]["eligible"] is False and not result["scientific_qualification"]


@pytest.mark.parametrize("damage",["missing","rejected","diagnostic","mixed_slot","boolean","uncovered","same_author","missing_E2"])
def test_V3_bad_reference_never_becomes_U2_weak_signal(tmp_path,damage):
    data=data_v3();ref=typed_reference(tmp_path,data,signal=0,kind="conditional_uncertainty_estimate")
    row=ref["error_evidence"]["E_q"]
    if damage == "missing": del ref["error_evidence"]["E_q"]
    elif damage == "rejected": row["status"]="rejected"
    elif damage == "diagnostic": row["claim_kind"]="diagnostic_agreement"
    elif damage == "mixed_slot": row["rigorous_bound_value"]=[0,1]
    elif damage == "boolean": row["conditional_estimate_value"]=False;row["estimated_or_enclosed_value"]["value"]=False
    elif damage == "uncovered": row["error_components_and_ownership"][0]["covered_axes"].pop()
    elif damage == "missing_E2": del row["observed_refinement_and_method_independence"]["E2"]
    review_typed_reference(tmp_path,ref)
    if damage == "same_author":
        path=Path(ref["independent_review"]["path"]);value=json.loads(path.read_text());value["reviewer"]=value["evidence_author"]
        ref["independent_review"]=pin(path,value)
    result=evaluate_v3(data,reference=ref)
    assert result["channels"]["device"]["status"] == "unknown_reference_error"
    assert not result["L_U2"]["device_signal_eligible"] and not result["L_U2"]["eligible"]


@pytest.mark.parametrize("signal,error,status",[(0,0,"unresolved_signal"),(3,1,"unresolved_signal"),
    (1,F(1,10000),"failed"),(1,0,"passed")])
def test_V3_signal_boundary_and_reference_share_are_distinct(tmp_path,signal,error,status):
    data=data_v3();ref=typed_reference(tmp_path,data,signal=F(signal),error=error)
    result=evaluate_v3(data,reference=ref)
    assert result["channels"]["device"]["status"] == status
    if status == "unresolved_signal":
        assert all(row["channels"]["device"]["prefix"]["status"] == "not_evaluated" for row in result["native"])
        assert result["L_U2"]["device_signal_eligible"] and not result["L_U2"]["eligible"]


def test_V3_missing_piece_unknown_precedes_eligible_weak_signal(tmp_path):
    data=data_v3();ref=typed_reference(tmp_path,data,signal=0);data[3].pop(5)
    result=evaluate_v3(data,reference=ref)
    assert result["channels"]["device"]["status"] == "unknown" and not result["L_U2"]["eligible"]


def test_V3_failure_is_latched_ahead_of_missing_reference_or_later_weak_signal(tmp_path):
    data=data_v3();ref=typed_reference(tmp_path,data,signal=0)
    for reference in (None,ref):
        result=evaluate_v3(data,reference=reference,known_device_failure={"kind":"manufactured-independent-device-failure","index":3})
        assert result["channels"]["device"]["status"] == "failed"
        assert result["channels"]["device"]["first_failure"]["index"] == 3 and not result["L_U2"]["eligible"]
    assert t.device_disposition(evaluated=False,reference={"status":"unknown"},non_signal_complete=False)["status"] == "not_evaluated"


def test_V3_U2_requires_all_other_checks_review_and_explicit_restriction(tmp_path):
    data=data_v3();ref=typed_reference(tmp_path,data,signal=0,kind="conditional_uncertainty_estimate")
    result=evaluate_v3(data,reference=ref);scope=result["L_U2"]["result_scope_sha256"]
    checks={k:"passed" for k in ("safety","T2","T3","history_identity","matched_growth","port_reference_comparisons","state_reference_comparisons")}
    review=pin(tmp_path/"ToyRouteReview.json",{"fixture_only":True,"accepted":True,"independent_route_review":True,
        "result_scope_sha256":scope,"checks_sha256":t.digest(checks)})
    restriction=pin(tmp_path/"ToyRestriction.json",{"fixture_only":True,"accepted":True,"id":"L-U2","scope":"G2-R",
        "result_scope_sha256":scope,"transfer":"G2-Q","exit_before":"G9P","user_record":"manufactured-not-real-authorization"})
    evidence={"checks":checks,"independent_review":review,"commander_restriction":restriction}
    actual=evaluate_v3(data,reference=ref,route_evidence=evidence)
    assert actual["L_U2"]["eligible"] and actual["L_U2"]["route_status"] == "passed_with_user_accepted_limitation"
    assert actual["L_U2"]["device_scientific_pass"] is actual["L_U2"]["full_scientific_pass"] is False
    evidence["checks"]["matched_growth"]="unknown"
    assert not evaluate_v3(data,reference=ref,route_evidence=evidence)["L_U2"]["eligible"]


def test_V3_piece_requires_independent_window_endpoint_actions(tmp_path):
    data=data_v3();ref=typed_reference(tmp_path,data,signal=0)
    piece=data[3][2];piece["channels"]["device"]["right_charge_enclosure_C"]=enc(1)
    piece["piece_id"]=t.digest({k:v for k,v in piece.items() if k != "piece_id"})
    with pytest.raises(t.InvalidT1Data,match="piece_endpoint_delta"):
        evaluate_v3(data,reference=ref)


def test_V3_sealed_supplement_cannot_replace_original_signed_operands():
    row=sealed_observer_row();native=t.from_sealed_charge(row,run_id="manufactured-run")
    native["native_clock"]=deepcopy(row["interval"]["observation"]["physical_charge"]["clock"])
    row["t1_evidence"]=native
    assert t.from_sealed_t1(row,run_id="manufactured-run") == native
    native["channels"]["left_metal"]["delta_Q_enclosure_C"]=enc(0)
    with pytest.raises(t.InvalidT1Data,match="original_operand_changed"):
        t.from_sealed_t1(row,run_id="manufactured-run")


def test_L1_retained_cell_proof_detects_rehashed_bound_or_sign_tampering():
    pytest.importorskip("flint")
    from scripts.benchmarks.interval_observation import BallIntegrator
    from scripts.benchmarks.native_observation import observation_record
    a=BallIntegrator();proof=a.absolute_enclosure(lambda z,analytic:a.number(1),-1,0,F(1,10**20))
    value=op(1,1)
    value.update(integral_abs_F_lower_C=t.pair(proof["lower"]),integral_abs_F_upper_C=t.pair(proof["upper"]),
        integral_abs_F_evidence=observation_record(proof),scale_input_clock_widening_C=[0,1],
        L1_scope={"path_identity":"toy","t_left":[0,1],"t_right":[1,1],"hused":[1,1],"native_tn":[1,1]})
    t.operands(value)
    broken=deepcopy(value);broken["integral_abs_F_lower_C"]=[2,1]
    with pytest.raises(t.InvalidT1Data):t.operands(broken)
    broken=deepcopy(value);broken["integral_abs_F_evidence"]["cells"]["tuple"][0]["sign"]="nonpositive"
    with pytest.raises(t.InvalidT1Data,match="L1_sign_certificate"):t.operands(broken)


def test_V3_sealed_evaluation_needs_reviewed_source_before_opening_history(tmp_path,monkeypatch):
    from scripts.benchmarks import native_history
    path,manifest=sealed_fixture(tmp_path)
    doc=document();doc.update(schema=t.SPEC_SCHEMA_V3,version=3)
    manifest["gate_spec"]=pin(tmp_path/"V3Spec.json",doc);pin(path,manifest)
    def forbidden(*args,**kwargs):raise AssertionError("unadmitted evaluator opened history")
    monkeypatch.setattr(native_history,"HistoryReader",forbidden)
    with pytest.raises(t.InvalidT1Data,match="evaluation_admission_required"):
        read_sealed(path)


@pytest.mark.parametrize("damage", [None,"trailer","logical_hash","record_count","admission_input"])
def test_V3_manufactured_admission_then_real_HistoryReader_roundtrip(tmp_path,monkeypatch,damage):
    from scripts.benchmarks import g2_route,native_history
    path,manifest=sealed_fixture(tmp_path,v3=True)
    request=json.loads(Path(manifest["request"]["path"]).read_text())
    plan={"fixture_only":True,"definitions":{"gate_spec":manifest["gate_spec"]},
          "cases":[{"id":manifest["case_id"],"execution":{"effective_input_digest":t.digest(request),
                    "native_admission":{"operation_id":manifest["run_id"]}}}]}
    manifest["run_plan"]=pin(tmp_path/"ToyEvaluationContext.json",plan)
    calls=[]
    def manufactured_admission(plan_path,**kwargs):
        assert str(plan_path) == manifest["run_plan"]["path"]
        assert kwargs == {"executable":True,"case_id":"B_N8_Main_Full","purpose":"evaluate"}
        calls.append(kwargs)
        return {"plan":plan}
    monkeypatch.setattr(g2_route,"validate",manufactured_admission)
    if damage == "trailer":
        history=Path(manifest["history"]["path"])
        history.write_bytes(history.read_bytes()[:-5])
        manifest["history"]["encoded_bytes"]=history.stat().st_size
    elif damage == "logical_hash": manifest["history"]["logical_sha256"]="0"*64
    elif damage == "record_count": manifest["history"]["records"]-=1
    elif damage == "admission_input":
        plan["cases"][0]["execution"]["effective_input_digest"]="wrong-input"
        def forbidden(*args,**kwargs): raise AssertionError("unbound input opened history")
        monkeypatch.setattr(native_history,"HistoryReader",forbidden)
    pin(path,manifest)
    if damage is not None:
        with pytest.raises((t.InvalidT1Data,EOFError)): read_sealed(path)
    else:
        result=read_sealed(path)
        assert result["history"] == {"records":256,"logical_bytes":manifest["history"]["logical_bytes"],
            "logical_sha256":manifest["history"]["logical_sha256"],"eof_seen":True,"crc_verified":True}
        assert len(result["fixed_windows"]) == 128 and all(w["complete_pieces"] for w in result["fixed_windows"])
        assert result["channels"]["left_metal"]["status"] == "passed"
        assert result["channels"]["device"]["status"] == "unknown_reference_error"
        assert result["L_U2"]["eligible"] is False and result["scientific_qualification"] is False
    assert len(calls) == 1


def test_sealed_T1_emitter_preserves_C_and_splits_actual_fixed_windows(monkeypatch):
    """Exercise the real writer adapter with an explicit exact toy action provider."""
    pytest.importorskip("flint")
    from types import SimpleNamespace
    from scripts.benchmarks import physical_tolerance
    from scripts.benchmarks.native_observation import SealedChargeObserver
    request=base_request()
    request["interval_observation"]={"arithmetic":{"bits":128,"evaluations":64,"depth":8}}
    row=sealed_observer_row();row["request_sha256"]=t.digest(request)
    physical=row["interval"]["observation"]["physical_charge"]
    physical["request_sha256"]=t.digest(request)
    duration=F(.1)
    physical["clock"]["hused"]=t.pair(duration)
    row["interval"]["observation"]["clock"]["hused"]=t.pair(duration)
    original=deepcopy(row);emitted=[];cuts=[]
    sealed=object.__new__(SealedChargeObserver)
    sealed.request=request;sealed.emit=emitted.append;sealed.checkpoint=lambda:None
    sealed.t1_context={"spec_sha256":SPEC3.sha256,"run_id":"manufactured-run","case_id":"B_N8_Main_Full"}
    sealed.t1_spec=SPEC3;sealed.t1_windows=t.fixed_windows(sample_labels(request));sealed.t1_pieces=0
    clock=SimpleNamespace(predecessor=F(0),tn=duration,hused=duration,segment_end=duration)
    path=SimpleNamespace(clock=clock,identity="path")
    observer=SimpleNamespace(prepared=SimpleNamespace(path=path),frame=SimpleNamespace(owner="toy-owner",generation=1))
    def exact_toy_piece(left_pair,right_pair,first,last,**kwargs):
        cuts.append((first,last,kwargs))
        channels={}
        for name in t.CHANNELS:
            left,right=(first/duration,last/duration) if name == "left_metal" else (F(0),F(0))
            value=op(right-left,0,missing_l1=True)
            value.update(integral_abs_F_evidence=None,scale_input_clock_widening_C=[0,1],missing_reason=None,
                coverage_ownership=deepcopy(OWNERSHIP),left_charge_enclosure_C=enc(left),right_charge_enclosure_C=enc(right),
                L1_scope={"path_identity":"path","t_left":t.pair(first),"t_right":t.pair(last),
                          "hused":t.pair(duration),"native_tn":t.pair(duration)})
            channels[name]=value
        return {"channels":channels,"endpoint_action_source":{
                    "left":"saved-state" if first == 0 else "exact-retained-polynomial",
                    "right":"saved-state" if last == duration else "exact-retained-polynomial"},
                "form_identities":{"body_charge":"toy-body","metal_charge":"toy-metal"},
                "path_identity":"path","arithmetic_calls":[],"status":"computed"}
    observer.t1_piece_evidence=exact_toy_piece
    monkeypatch.setattr(physical_tolerance,"physical_charge_error_allocation",lambda request,length:F(length)/10000)
    sealed._emit_t1(row,observer,("toy-left",),("toy-right","toy-increment"),F(1,10000))
    assert {k:v for k,v in emitted[0].items() if k != "t1_evidence"} == original
    native=t.from_sealed_t1(emitted[0],run_id="manufactured-run")
    assert native["source_record_sha256"] == t.digest(original)
    pieces=[record["evidence"] for record in emitted[1:]]
    assert sealed.t1_pieces == len(pieces) == 32 and len(cuts) == 33
    assert cuts[0][:2] == (0,duration) and len(cuts[0][2]["scale_deciders"]) == 3
    assert pieces[0]["endpoint_action_source"]["left"] == "saved-state"
    assert pieces[1]["endpoint_action_source"]["left"] == "exact-retained-polynomial"
    assert pieces[-1]["endpoint_action_source"]["right"] == "saved-state"
    assert pieces[-1]["event_side"] == "left"
    assert all(p["native_clock"] == physical["clock"] for p in pieces)
    assert all(p["owner_generation"] == {"owner":"toy-owner","generation":1} for p in pieces)
    for index,piece in enumerate(pieces):
        t._piece_v3(piece,native,sealed.t1_windows[index],"B_N8_Main_Full")
    result=t.evaluate([native],pieces,labels=sample_labels(request),spec=SPEC3,case_id="B_N8_Main_Full",
                      run_id="manufactured-run",problem_identity=t.physical_problem_identity(request))
    assert sum(w["complete_pieces"] for w in result["fixed_windows"]) == 32
    assert result["channels"]["device"]["status"] == "unknown_reference_error"
    assert result["channels"]["left_metal"]["complete_native_coverage"] is False


# Change G: independent D2 review witnesses on manufactured data only.
def three_piece_v3_window(index=5):
    data = data_v3()
    original = data[3][index]
    left, right = map(F, (float.fromhex(original['t_left_hex']), float.fromhex(original['t_right_hex'])))
    cuts = [left, F(float(left+(right-left)/4)), F(float(left+3*(right-left)/4)), right]
    children = []
    for a, b in zip(cuts, cuts[1:]):
        child = deepcopy(original)
        child.update(t_left_hex=float(a).hex(), t_right_hex=float(b).hex(), event_side='interior')
        child['piece_id'] = t.digest({k: v for k, v in child.items() if k != 'piece_id'})
        children.append(child)
    data[3][index:index+1] = children
    return data


@pytest.mark.parametrize('missing', [0, 1, 2])
@pytest.mark.parametrize('earlier_failure', [False, True])
def test_G_missing_first_middle_last_piece_retains_summary(tmp_path, missing, earlier_failure):
    data = three_piece_v3_window()
    reference = typed_reference(tmp_path, data, signal=0)
    assert evaluate_v3(data, reference=reference)['fixed_windows'][5]['complete_pieces'] is True
    del data[3][5+missing]
    failure = {'kind': 'manufactured_earlier_device_failure', 'index': 1} if earlier_failure else None
    result = evaluate_v3(data, reference=reference, known_device_failure=failure)
    window = result['fixed_windows'][5]
    assert window['complete_pieces'] is False
    assert window['signed_device_trend_input'] is None
    for name in t.CHANNELS[1:]:
        assert window['channels'][name] == {'status': 'unknown', 'reason': 'unknown_missing_window_pieces'}
        assert result['channels'][name]['first_unknown']['index'] == 5
    assert len(result['fixed_windows']) == 128 and result['fixed_windows'][6]['complete_pieces'] is True
    assert result['channels']['device']['status'] == ('failed' if earlier_failure else 'unknown')
    if earlier_failure:
        assert result['channels']['device']['first_failure'] == failure
    assert result['L_U2']['eligible'] is False


@pytest.mark.parametrize('damage', ['duplicate', 'out_of_order', 'identity'])
def test_G_piece_corruption_is_not_classified_as_missing(damage):
    data = three_piece_v3_window()
    if damage == 'duplicate':
        data[3].insert(6, deepcopy(data[3][5]))
    elif damage == 'out_of_order':
        data[3][5], data[3][6] = data[3][6], data[3][5]
    else:
        data[3][6]['piece_id'] = 'unbound-piece'
    with pytest.raises(t.InvalidT1Data):
        evaluate_v3(data)


def G_widening_witness(input_error=F(1,2), clock_error=F(0), widening=F(0)):
    pytest.importorskip("flint")
    from scripts.benchmarks.interval_observation import BallIntegrator
    from scripts.benchmarks.native_observation import observation_record
    arithmetic = BallIntegrator()
    proof = arithmetic.absolute_enclosure(lambda z, analytic: arithmetic.number(1000), -1, 0, F(1,10**20))
    value = op(F(1999,2), 1000, errors=(0,0,input_error,clock_error))
    value.update(integral_abs_F_evidence=observation_record(proof),
        integral_abs_F_lower_C=t.pair(max(F(0),proof['lower']-widening)),
        integral_abs_F_upper_C=t.pair(proof['upper']+widening),
        scale_input_clock_widening_C=t.pair(widening), coverage_ownership=deepcopy(OWNERSHIP),
        L1_scope={'path_identity':'manufactured-constant-flux','t_left':[0,1], 't_right':[1,1],
                  'hused':[1,1],'native_tn':[1,1]})
    envelope = {'t_left_hex':float(0).hex(),'t_right_hex':float(1).hex(),
        'native_clock':{'predecessor':[0,1],'tn':[1,1],'hused':[1,1]},
        'path_identity':'manufactured-constant-flux',
        'channels':{name:deepcopy(value) for name in t.CHANNELS}}
    t._channels_v3(envelope)
    return value


@pytest.mark.parametrize('input_error,clock_error', [(F(1,2),F(0)), (F(0),F(1,2)), (F(1,4),F(1,4))])
@pytest.mark.parametrize('widening', [F(0), F(49,100)])
def test_G_L1_widening_cannot_omit_input_or_clock(input_error, clock_error, widening):
    value = G_widening_witness(input_error, clock_error, widening)
    with pytest.raises(t.InvalidT1Data, match='L1_input_clock_coverage'):
        t.assess_interval(value, floor=1, epsilon=F(1,1000))


def test_G_correctly_widened_L1_witness_is_unknown():
    value = G_widening_witness(widening=F(1,2))
    result = t.assess_interval(value, floor=1, epsilon=F(1,1000))
    assert result['status'] == 'unknown'
    assert t.q(result['defect_C']) == 1
    assert t.q(result['scale_lower_C']) == F(1999,2)
    assert t.q(result['scale_upper_C']) == F(2001,2)
    assert t.assess_interval(G_widening_witness(widening=F(3,4)), floor=1,
                            epsilon=F(1,1000))['status'] == 'unknown'
