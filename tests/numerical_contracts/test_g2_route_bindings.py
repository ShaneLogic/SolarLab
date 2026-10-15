"""Manufactured metadata only: no IDA/model/Arb or historical gzip access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmarks import g2_route_bindings as b


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return b.read_json(path)[1]


def request(mesh=8, profile="Main", *, s0=False):
    size = (3 if s0 else 5)*(mesh+1)
    segments = [dict(id="dark_equilibrium_hold", start=0., end=.1, voltage=[0, 0], photons=[0, 0]),
                dict(id="voltage_ramp", start=.1, end=.2, voltage=[0, .1], photons=[0, 0]),
                dict(id="slow_state_hold", start=.2, end=9.2, voltage=[.1, .1], photons=[0, 0])]
    times = {s["id"]: [s["start"] + (s["end"]-s["start"])*i/(n-1) for i in range(n)]
             for s, n in zip(segments, (33, 33, 65))}
    for s in segments:
        times[s["id"]][-1] = s["end"]
    packet = dict(nodes=mesh+1, intervals=mesh, physical_reference_fields={"saved": ["0x0.0p+0", "-0x0.0p+0"]})
    mapping = dict(mapped_input_words=12, fixture="manufactured", nodes=mesh+1)
    policy = dict(profile=b.PROFILES[profile], size=size, numeric_packet_sha256=b.digest(packet),
                  floor_basis="frozen-initial-reference")
    r = dict(schema="solarlab.voltage-lift-native-request.v1", numeric_packet=packet,
        voltage_lift_map=mapping, map_identity=b.digest(mapping), physical_tolerance_policy=policy,
        weight_certificate=deepcopy(policy), controls=dict(max_step=.02 if mesh == 16 else .1,
            rtol=b.PROFILES[profile]["r"], atol=[1e-8]*size, max_order=5, nonlin_conv_coef=1e-8),
        budgets=dict(wall_s=4200, charge_C=1e-12), original_budgets=dict(wall_s=4200, charge_C=1e-12),
        segments=segments, observation_times=times, quadrature={"8": [], "16": [], "32": []},
        frame_input_policy=dict(raw_parent_words=4), z0=[0.]*size, zdot0=[-0.]*size,
        actual_initial_identity="manufactured-physical-initial", initial_preparation={"manufactured": True},
        preparation_context_sha256="manufactured-context",
        initial_segment_frame={k:[["0x0.0p+0"]*size for _ in range(12)] for k in
                               ("physical_handoff_words_hex", "physical_rate_handoff_words_hex")})
    r["interval_observation"] = dict(map_identity=r["map_identity"], controls_sha256=b.digest(r["controls"]),
        budgets_sha256=b.digest(r["budgets"]), protocol_sha256=b.digest(segments),
        sampling_sha256=b.digest((times,r["quadrature"])), physical_charge={"fixture": "not-T1"})
    r["reference_refresh_policy"] = dict(schema="solarlab.accepted-reference-refresh-policy.v1")
    return r


@pytest.fixture
def bundle(tmp_path):
    cases = []
    saved = {}
    for name in sorted(b.CASE_IDS):
        mesh = 16 if "N16" in name else 8
        profile, probe = name.split("_")[2], name.endswith("Probe")
        r = request(mesh, profile)
        reference = write_json(tmp_path / (name+".json"), r)
        saved[name] = r
        cases.append(dict(id=name, stage="probe" if probe else "full", mesh_intervals=mesh, nodes=mesh+1,
                          coordinates=5*(mesh+1), profile=b.PROFILES[profile], input_ancestor=reference,
                          horizon_s=[0,.2 if probe else 9.2], sample_records=66 if probe else 131))
    s0 = request(s0=True)
    s0_request = write_json(tmp_path/"S0.json",s0)
    s0_acceptance = write_json(tmp_path/"S0Acceptance.json",dict(accepted=True,complete_protocol_accepted=True,
                            request_sha256=b.digest(s0),actual_input=s0_request,fixture="manufactured"))
    n16 = next(c for c in cases if "N16" in c["id"])
    n16_path = tmp_path/"N16Acceptance.json"
    write_json(n16_path,dict(accepted=True,source_input_acceptance_only=True,fixture="manufactured",
        cases={"B":dict(request=n16["input_ancestor"],canonical_request_sha256=b.digest(saved[n16["id"]]),
                        actual_initial_identity=saved[n16["id"]]["actual_initial_identity"])}))
    gate = write_json(tmp_path/"Gate.json",dict(status="proposed_not_frozen"))
    root = tmp_path/"repo"
    (root/"scripts/benchmarks").mkdir(parents=True)
    (root/"scripts/benchmarks/g2_route.py").write_text('T1_IMPLEMENTATION = None\nSCHEMA="solarlab.g2-route.run-plan.v2"\n')
    plan = dict(schema=b.PLAN_SCHEMA,version=2,status="design_only_not_admitted",cases=cases,
        authority=dict(scientific_execution_authorized=False),implementation=dict(entrypoint_proposal="scripts/benchmarks/g2_route.py"),
        source_freeze={k:None for k in ("future_execution_commit","frozen_gate_spec","review_record","environment_lock")},
        definitions=dict(gate_spec=gate), conditional_S0=dict(reference=s0_acceptance),
        protocol=dict(segments=s0["segments"], requested_sample_records=b.sample_labels(s0),max_step_s=.02,
            reference_refresh_schedule=dict(time_hex="0x1.7a4c1c85a8b06p-2",segment_id="slow_state_hold",max_refreshes=1,
                native_api="init_step",physical_segment_ordinal=3,same_owner_generation_increment=1,
                preserve_original_segment_inputs=True,discard_native_multistep_history=True,restored_seed_prefix_credit=False)))
    path = tmp_path/"RunPlan.json"
    write_json(path,plan)
    return path,plan,root,n16_path,saved


def resolve(bundle):
    path,plan,root,n16,_ = bundle
    write_json(path,plan)
    return b.resolve(path,repository=root,n16_acceptance=n16)


def test_actual_callable_reports_all_missing_bindings_without_promoting_ancestry(bundle):
    result = resolve(bundle)
    assert result["status"] == "not_admitted" and result["execution_authorized"] is False
    assert "T1_not_frozen" in result["missing"]
    assert "source_freeze.future_execution_commit" in result["missing"]
    assert "reviewed_entry_does_not_support_RunPlanV2" not in result["missing"]
    assert len(result["cases"]) == 7
    for c in result["cases"]:
        assert c["effective_request"] is None and c["effective_case_binding"] is None
        assert c["ancestry"]["resolved"]["coordinates"] == (85 if "N16" in c["case_id"] else 45)
        assert c["planned_active_sampling"]["records"] == (66 if c["stage"] == "probe" else 131)
        assert c["reference_refresh"]["enabled"] == (c["stage"] == "full")
    assert result["conditional_S0"]["observed"]["coordinates"] == 27


def test_sampling_digest_includes_quadrature_and_preserves_event_sides():
    r = request()
    observed = b.inspect_request(r)
    assert observed["sampling_sha256"] == b.digest((r["observation_times"],r["quadrature"]))
    assert observed["sampling_sha256"] != b.digest(r["observation_times"])
    labels = b.sample_labels(r)
    assert len(labels) == 131 and len({x["time_hex"] for x in labels}) == 129
    assert [x["event_side"] for x in labels if x["time_hex"] == float(.1).hex()] == ["left","right"]
    assert b.sample_labels(r,2)[-1]["segment"] == "voltage_ramp"


@pytest.mark.parametrize("damage", ["map", "controls", "quadrature", "shape", "words", "budgets"])
def test_corrupt_saved_input_is_rejected(damage):
    r = request()
    if damage == "map": r["voltage_lift_map"]["nodes"] = 17
    elif damage == "controls": r["controls"]["max_step"] = .02
    elif damage == "quadrature": r["quadrature"]["8"] = [1]
    elif damage == "shape": r["z0"].pop()
    elif damage == "words": r["initial_segment_frame"]["physical_rate_handoff_words_hex"][0].pop()
    elif damage == "budgets": r["budgets"]["charge_C"] *= 2
    with pytest.raises(ValueError): b.inspect_request(r)


def test_file_bytes_and_hash_are_not_canonical_hash(tmp_path):
    path = tmp_path/"Record.json"
    record = write_json(path,dict(a=1))
    assert record["sha256"] != b.digest(dict(a=1))
    path.write_text('{"a": 1}')
    with pytest.raises(ValueError,match="file_pin"): b.pinned(record)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_bad_json_fails_closed(tmp_path,raw):
    path = tmp_path/"Bad.json";path.write_text(raw)
    with pytest.raises(ValueError): b.read_json(path)


def test_gzip_is_rejected_before_opening(tmp_path,monkeypatch):
    def fail(*a,**k): pytest.fail("history was opened")
    monkeypatch.setattr(Path,"open",fail)
    with pytest.raises(ValueError,match="json_metadata_only"): b.read_json(tmp_path/"NativeHistory.jsonl.gz")


def test_request_label_change_cannot_hide_N8_geometry(bundle):
    bundle[1]["cases"][0]["coordinates"] = 45  # sorted first is N16
    with pytest.raises(ValueError,match="case_definition"): resolve(bundle)


def test_N16_saved_acceptance_must_name_exact_request(bundle):
    d,_ = b.read_json(bundle[3]);d["cases"]["B"]["canonical_request_sha256"] = "0"*64
    write_json(bundle[3],d)
    with pytest.raises(ValueError,match="N16_actual_accepted_input"): resolve(bundle)


def test_shared_refresh_must_not_contain_N16_hash(bundle):
    bundle[1]["protocol"]["reference_refresh_schedule"]["controls_sha256"] = "0"*64
    with pytest.raises(ValueError,match="shared_case_specific_identity"): resolve(bundle)


@pytest.mark.parametrize("damage", ["unknown_stage", "swapped_ids", "index", "time", "event_side", "order", "index_bool", "V1"])
def test_bindings_reject_same_V2_case_and_tag_corruption_as_route(bundle, damage):
    plan = bundle[1]
    labels = plan["protocol"]["requested_sample_records"]
    if damage == "unknown_stage": plan["cases"][0]["stage"] = "other"
    elif damage == "swapped_ids":
        a = next(c for c in plan["cases"] if c["id"] == "B_N8_Main_Probe")
        z = next(c for c in plan["cases"] if c["id"] == "B_N8_Main_Full")
        a["id"], z["id"] = z["id"], a["id"]
    elif damage == "index": labels[4]["index"] = 999
    elif damage == "time": labels[4]["time_hex"] = float(42).hex()
    elif damage == "event_side": labels[32]["event_side"] = "right"
    elif damage == "order": labels[4], labels[5] = labels[5], labels[4]
    elif damage == "index_bool": labels[0]["index"] = False
    else: plan.update(schema="solarlab.g2-route.run-plan.v1", version=1)
    with pytest.raises(ValueError): resolve(bundle)


@pytest.mark.parametrize("bad_path", [False, True])
def test_bindings_validate_supervision_as_metadata_only(bundle, bad_path):
    from test_g2_route import supervision_fixture
    plan = bundle[1]
    plan["resources"] = {"rss_bytes_per_case": 4*1024**3, "combined_persisted_case_bytes": 4*1024**3}
    supervision = supervision_fixture(bundle[0].parent)
    if bad_path:
        supervision["cancel_path"] = "relative-cancel"
    plan["cases"][0]["execution"] = {"wall_s": 10., "supervision": supervision}
    if bad_path:
        with pytest.raises(ValueError, match="supervision_cancel_path"):
            resolve(bundle)
    else:
        assert resolve(bundle)["status"] == "not_admitted"


def test_supplied_old_request_is_not_effective_G2_input(bundle):
    c = bundle[1]["cases"][1]
    c["execution"] = dict(request=c["input_ancestor"])
    result = resolve(bundle)
    actual = next(x for x in result["cases"] if x["case_id"] == c["id"])
    assert "controls.max_step_requires_0.02" in actual["missing"]
    assert "D14_D25_certificate_policy" in actual["missing"]
    assert "frozen_T1_request_binding" in actual["missing"]


def test_changed_signed_zero_cannot_hide_behind_initial_identity(bundle):
    r = request();modified = deepcopy(r);modified["zdot0"][0] = 0.
    case = next(c for c in bundle[1]["cases"] if c["id"] == "B_N8_Main_Full")
    assert "saved_physical_input_changed:zdot0" in b.effective_issues(modified,r,case,bundle[1])


def forecast():
    return dict(phase_seconds=dict(bootstrap=10,native=100,online_safety=5,observer=20,io=3,closeout=2),
        slow_hold_model={"range_s":[50,100],"basis":"manufactured"},
        work_counts=dict(accepted_steps=100,residual_calls=200,jacobian_calls=50),probe_evidence="manufactured")


def test_full_forecast_is_phase_based_and_never_silently_clipped():
    f = forecast();assert b.forecast_cap(f) == 420
    f["phase_seconds"]["native"] = 15000
    with pytest.raises(ValueError,match="infeasible_no_silent_cap"): b.forecast_cap(f)


@pytest.mark.parametrize("fault", ["slow_hold_model", "probe_evidence", "negative", "N16"])
def test_incomplete_forecast_rejected(fault):
    f = forecast()
    if fault == "negative": f["phase_seconds"]["native"] = -1
    elif fault == "N16": f["mesh_intervals"] = 16
    else: del f[fault]
    with pytest.raises(ValueError): b.forecast_cap(f)


def test_cli_dry_run_exit_one_is_not_an_execution_failure(bundle,capsys):
    result = b.main(["--plan",str(bundle[0]),"--repository",str(bundle[2]),"--n16-acceptance",str(bundle[3])])
    assert result == 1 and json.loads(capsys.readouterr().out)["status"] == "not_admitted"


@pytest.mark.parametrize("free,sufficient", [(80,True),(20,False)])
def test_four_channel_disk_capacity_counts_one_filesystem_but_all_copies(tmp_path,monkeypatch,free,sufficient):
    resources = dict(WS1_channels=4,threads_per_case=1,rss_bytes_per_case=4*1024**3,
        combined_persisted_case_bytes=4*1024**3,reader_logical_byte_cap=32*1024**3,
        probe_wall_cap_s=3600,global_scientific_channel_cap=8,native_step_cap=200000,
        residual_call_cap=2000000,forecast_requirements=["phase counts"],rolling_forecast="retain hard cap")
    plan = dict(resources=resources,artifacts=dict(active_root=str(tmp_path/"active"),archive_root=str(tmp_path/"archive")),
                cases=[dict(id="B_N8_Main_Probe",stage="probe"),dict(id="B_N16_Strict_Full",stage="full")])
    monkeypatch.setattr(b.shutil,"disk_usage",lambda path:SimpleNamespace(total=100*1024**3,free=free*1024**3))
    result = b.resource_preflight(plan)
    assert len(result["filesystems"]) == 1
    assert result["filesystems"][0]["required_bytes"] == 56*1024**3
    assert result["four_channel_rss_bytes"] == 16*1024**3
    assert result["disk_capacity_sufficient"] is sufficient
    assert result["per_case"][0]["wall_cap_s"] == 3600 and result["per_case"][1]["wall_cap_s"] is None
    assert not (tmp_path/"active").exists() and not (tmp_path/"archive").exists()
