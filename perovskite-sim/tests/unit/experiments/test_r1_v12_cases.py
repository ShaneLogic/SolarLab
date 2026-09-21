"""Contract and cost boundaries for the new, finite P3/P4 case set."""
from copy import deepcopy
import json

import pytest

from scripts import run_r1_v12_cases as runner


LIMITS = {"elapsed_s": 1800., "peak_rss_bytes": 4*1024**3,
          "row_payload_bytes": 256*1024**2, "case_artifact_bytes": 1024**3}


def request(case_id="D_N32_F0p01_T2"):
    return runner.make_request(case_id, "a"*64, LIMITS)


def load(tmp_path, value):
    path = tmp_path / "Request.json"
    path.write_text(json.dumps(value))
    return runner.load_request(path, runner.sha(path), tmp_path / "Result")


@pytest.mark.parametrize("case_id,rows", [("D_N32_F0p01_T2", 59), ("B_N256_F0p01_T8", 227),
                                        ("D_N256_F0p01_T4", 115), ("D_N128_F0p01_T4", 115)])
def test_original_strict_cases_have_exact_finite_extent(tmp_path, case_id, rows):
    value = load(tmp_path, request(case_id))
    assert value["expected_rows"] == rows
    assert value["preparation"] == {"nonlinear_factor": .1, "time_substeps": [1, 2, 4]}
    assert value["case"]["nonlinear_factor"] == .01


@pytest.mark.parametrize("key,value", [
    ("expected_rows", 794), ("pair_repeats", 2), ("numerical_retries", 1),
    ("absolute_engineering_qualification", False), ("scientifically_accepted", True),
    ("preparation", {"nonlinear_factor": .01, "time_substeps": [2, 4, 8]}),
    ("solver_limits", {**runner.SOLVER_LIMITS, "newton": 200}),
    ("certificate_limits", {**runner.LIMITS, "nonlinear_residual": .051}),
    ("relative_engineering_limits", {**runner.RELATIVE_LIMITS, "elapsed_ratio": 5.1}),
])
def test_changed_extent_solver_acceptance_or_qualification_rejected(tmp_path, key, value):
    body = deepcopy(request())
    body[key] = value
    with pytest.raises(ValueError):
        load(tmp_path, body)


@pytest.mark.parametrize("key,value", [("control", "B"), ("intervals", 64),
    ("nonlinear_factor", .1), ("time_substeps", [1, 2, 4]), ("amplitude_V", .01),
    ("times_s", [0., 1e-9, 1e-8, 1e-6, 1e-3]), ("fault", "omit_ion")])
def test_unapproved_physical_axes_rejected(tmp_path, key, value):
    body = deepcopy(request())
    body["case"][key] = value
    with pytest.raises(ValueError):
        load(tmp_path, body)


def test_long_window_cannot_be_satisfied_by_a_short_case(tmp_path):
    with pytest.raises(ValueError, match="134 historical"):
        load(tmp_path, request("P4LongD16"))


def test_digest_and_external_location_are_checked(tmp_path):
    path = tmp_path / "Request.json"
    path.write_text(json.dumps(request()))
    with pytest.raises(ValueError, match="identity"):
        runner.load_request(path, "0" * 64, tmp_path / "Result")
    with pytest.raises(ValueError, match="outside"):
        runner.load_request(path, runner.sha(path), tmp_path)


def summary(mode="baseline"):
    return {"schema": "R1V12CaseRunV1", "case_id": "B_N256_F0p01_T8", "mode": mode,
            "source_commit": "a" * 40, "source_content_sha256": "b" * 64,
            "request_sha256": "c" * 64, "precision_budget_sha256": "d" * 64,
            "runtime_identity": {"numpy": "2.1.3", "scipy": "1.15.3"},
            "baseline_usable": mode == "baseline", "extent": {"complete": True},
            "cost": {"elapsed_s": 10., "peak_rss_bytes": 1000, "bytes_per_row": 200.}}


def test_failed_prefix_never_supplies_a_full_run_denominator():
    baseline, pair = summary(), summary("compensated")
    baseline["extent"]["complete"] = False
    result = runner.relative_cost(pair, baseline)
    assert result["applicable"] is False and result["qualified"] is None and not result["ratios"]


def test_original_ratio_limits_are_inclusive_and_not_the_4_8_target():
    baseline, pair = summary(), summary("compensated")
    pair["cost"] = {"elapsed_s": 50., "peak_rss_bytes": 2000, "bytes_per_row": 600.}
    result = runner.relative_cost(pair, baseline)
    assert result["qualified"] and result["ratios"] == runner.RELATIVE_LIMITS
    pair["cost"]["elapsed_s"] += 1e-10
    result = runner.relative_cost(pair, baseline)
    assert not result["qualified"] and result["failed_metrics"] == ["elapsed_ratio"]


@pytest.mark.parametrize("key", ["case_id", "source_commit", "source_content_sha256", "request_sha256",
                                  "precision_budget_sha256", "runtime_identity"])
def test_reference_identity_cannot_cross_cases_or_sources(key):
    baseline, pair = summary(), summary("compensated")
    pair[key] = "changed"
    with pytest.raises(ValueError, match="same-source"):
        runner.relative_cost(pair, baseline)


def test_counted_artifacts_include_nested_manifests_and_their_own_metadata(tmp_path):
    child = tmp_path / "Analysis"
    child.mkdir()
    (child / "ManifestV1.json").write_text("{}\n")
    (child / "Rows.jsonl").write_text('{"row":0}\n')
    value = {"cost": {}}
    runner.finalize_manifest(tmp_path, value, LIMITS)
    manifest = json.loads((tmp_path / "ManifestV1.json").read_text())
    assert "Analysis/ManifestV1.json" in manifest
    assert value["cost"]["case_artifact_bytes"] == sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert not value["absolute_engineering_check"]["qualified"]


@pytest.mark.parametrize("key", ["intervals", "control_label", "times_s", "policy", "prepared_sha256", "execution_axes"])
def test_actual_producer_axes_are_not_replaced_by_request_metadata(key):
    case = request()["case"]
    record = {"schema": "R1ControlledStepV2", "intervals": 32, "control_label": "D",
              "amplitude_V": .005, "times_s": runner.SHORT_TIMES,
              "prepared_sha256": "a" * 64, "policy": {"newton": 100},
              "execution_axes": {"intervals": 32, "time_substeps": [2, 4, 8]}}
    runner.validate_result_request(record, case, "a" * 64, {"newton": 100}, pair=True)
    record[key] = "changed"
    with pytest.raises(ValueError, match="actual producer"):
        runner.validate_result_request(record, case, "a" * 64, {"newton": 100}, pair=True)


def test_exact_matrix_has_29_original_plus_two_extensions_without_duplicates():
    original = [key for key, axes in runner.CASES.items() if axes[0] == "original_matrix"]
    assert len(original) == 29 and len(set(original)) == 29
    assert len(runner.CASES) == 33
    assert set(key for key, axes in runner.CASES.items() if axes[0] == "strict_extension") == {
        "D_N256_F0p01_T8", "B_N256_F0p01_T8"}
    assert runner.CASES["P4LongD256"][1:] == ("D", 256, .01, [4, 8, 16])


@pytest.mark.parametrize("key,value", [("elapsed_s", float("nan")), ("peak_rss_bytes", True),
    ("row_payload_bytes", 1.5), ("case_artifact_bytes", 0)])
def test_absolute_contract_rejects_missing_nonfinite_or_coerced_limits(tmp_path, key, value):
    body = request()
    body["absolute_engineering_limits"][key] = value
    with pytest.raises(ValueError, match="resource limits"):
        load(tmp_path, body)


def test_absolute_budget_is_inclusive_and_failed_ratio_is_not_overridden():
    cost = {"elapsed_s": LIMITS["elapsed_s"], "peak_rss_bytes": LIMITS["peak_rss_bytes"],
            "accepted_row_payload_bytes": LIMITS["row_payload_bytes"], "case_artifact_bytes": LIMITS["case_artifact_bytes"]}
    assert runner.absolute_cost(cost, LIMITS)["qualified"]
    cost["case_artifact_bytes"] += 1
    assert runner.absolute_cost(cost, LIMITS)["failed_metrics"] == ["case_artifact_bytes"]


def test_matrix_provenance_is_not_replaceable(tmp_path):
    body = request()
    body["historical_matrix_sha256"] = "0"*64
    with pytest.raises(ValueError, match="provenance"):
        load(tmp_path, body)


def test_long_n256_requires_explicit_external_pair_preparation_identity(tmp_path):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import observation_times
    body = runner.make_request("P4LongD256", "a"*64, LIMITS, long_times=observation_times().tolist(),
        required_pair_preparation_identity_sha256="b"*64)
    assert load(tmp_path, body)["expected_rows"] == 3727
    body["required_pair_preparation_identity_sha256"] = None
    with pytest.raises(ValueError, match="prior pair"):
        load(tmp_path, body)
    body = request()
    body["required_pair_preparation_identity_sha256"] = "b"*64
    with pytest.raises(ValueError, match="only N256"):
        load(tmp_path, body)
    body.pop("required_pair_preparation_identity_sha256")
    with pytest.raises(ValueError, match="absent"):
        load(tmp_path, body)


from scripts import analyze_r1_v12_responses as responses


def response_request(command):
    names = responses.DC_CASES if command == 'collect-dc' else responses.D2_CASES
    body = {'schema': responses.SCHEMAS[command], 'source_commit': 'a'*40,
            'source_content_sha256': 'b'*64,
            'cases': [{'case_id': name, 'bundle': {'directory': '/example/'+name,
                       'manifest_file': '/external/'+name, 'manifest_sha256': 'c'*64}}
                      for name in names]}
    if command == 'analyze':
        body['dc_bundle'] = {'directory': '/dc', 'manifest_file': '/external/dc', 'manifest_sha256': 'd'*64}
    return body


@pytest.mark.parametrize('command', ['collect-dc', 'analyze'])
def test_response_contract_exact_case_inventory(tmp_path, command):
    body = response_request(command)
    path = tmp_path/'ResponseRequest.json'
    path.write_text(json.dumps(body))
    actual, _ = responses.load_request(path, responses.sha(path), tmp_path/'Result', command)
    assert actual == body
    assert len(actual['cases']) == (3 if command == 'collect-dc' else 8)
    body['cases'][-1] = body['cases'][0]
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='exact distinct'):
        responses.load_request(path, responses.sha(path), tmp_path/'Result', command)


@pytest.mark.parametrize('mutation', ['history', 'missing_extra', 'changed_schema', 'self_output'])
def test_response_request_cannot_reuse_history_omit_extra_or_write_inside_input(tmp_path, mutation):
    body = response_request('analyze')
    if mutation == 'history':
        body['cases'][0]['case_id'] = 'P3MigrationD32'
    elif mutation == 'missing_extra':
        body['cases'].pop()
    elif mutation == 'changed_schema':
        body['schema'] = 'R1V5ResponseAnalysisRequestV1'
    path = tmp_path/'ResponseRequest.json'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        responses.load_request(path, responses.sha(path), tmp_path if mutation == 'self_output' else tmp_path/'Result', 'analyze')


def test_d2_finest_gate_preserves_coarse_failure_and_requires_new_t8(monkeypatch):
    selected = {name: {'case_id': name, 'available': True, 'request': {'intervals': int(name.split('_N')[1].split('_')[0])}}
                for name in responses.D2_CASES}
    def report(left, right, axis, baselines, source):
        passed = not (axis == 'time_substeps' and left['case_id'].endswith('_T1'))
        return {'available': True, 'comparison': {'convergence_passed': passed},
                'absolute_responses': {key: {'absolute_difference': [1.]}
                    for key in ('regular_current_A_m2', 'integrated_charge_C_m2')}}
    monkeypatch.setattr(responses, 'pair_report', report)
    request = response_request('analyze')
    result = responses.analyze(request, selected, {64: {}, 128: {}, 256: {}}, {})
    assert result['short_window_D2_gate_passed']
    assert not result['pairs']['time_substeps'][0]['comparison']['convergence_passed']
    assert not result['D2_satisfied']
    selected[responses.EXTRA_TIME]['available'] = False
    result = responses.analyze(request, selected, {64: {}, 128: {}, 256: {}}, {})
    assert result['necessary_finest_pairs_passed']
    assert not result['additional_fine_time_pair_passed'] and not result['short_window_D2_gate_passed']


def test_electrical_adapter_passes_complete_dc_and_both_original_preparations(monkeypatch):
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_convergence as convergence
    record = {'prepared_sha256': 'a'*64}
    dc = {'prepared_sha256': 'b'*64, 'precision_current_A_m2': {'hi': [1., 1.], 'lo': [1e-20, -1e-20]}}
    step_prepared, dc_prepared = {'sha256': 'a'*64}, {'sha256': 'b'*64}
    expected = deepcopy((record, dc, step_prepared, dc_prepared))
    sentinel = object()
    def shared(actual_record, actual_dc, **kwargs):
        assert actual_record is record and actual_dc is dc
        assert kwargs == {'expected_times_s': responses.TIMES,
                          'step_prepared': step_prepared, 'dc_prepared': dc_prepared}
        return sentinel
    monkeypatch.setattr(convergence, 'pair_step_current_charge_responses', shared)
    assert responses.electrical_responses(record, dc,
        step_prepared=step_prepared, dc_prepared=dc_prepared) is sentinel
    assert (record, dc, step_prepared, dc_prepared) == expected


def test_electrical_adapter_cannot_accept_bare_pair_contacts():
    from perovskite_sim.physics.compensated import DD
    with pytest.raises((TypeError, ValueError, AttributeError)):
        responses.electrical_responses({'schema': 'R1ControlledStepV2', 'representation': 'float64-pair-v1'},
            DD([1., 1.], [1e-20, -1e-20]), step_prepared={}, dc_prepared={})
