"""Sealed-input boundaries and honest scoped D-response derivation."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import physical_preparation_identity
from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification_workflow import read_only_derivation
from tests.unit.experiments.test_one_dimensional_mechanism_r1_spatial import manufactured_prepared, manufactured_step


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[3]/"scripts/analyze_r1_v5_responses.py"
    spec = importlib.util.spec_from_file_location("r1_v5_response_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sealed(tool, tmp_path):
    directory = tmp_path/"bundle"
    directory.mkdir()
    tool.write(directory/"Data.json", {"value": 1})
    tool.seal(directory)
    external = tmp_path/"ExternalManifest.json"
    external.write_bytes((directory/"ManifestV1.json").read_bytes())
    descriptor = {"directory": str(directory), "manifest_file": str(external), "manifest_sha256": tool.sha(external)}
    return directory, external, descriptor


def test_external_manifest_binds_complete_bytes_and_cannot_be_self_held(tool, tmp_path):
    directory, external, descriptor = sealed(tool, tmp_path)
    bundle = tool.SealedBundle(descriptor)
    assert bundle.read("Data.json") == {"value": 1}
    with pytest.raises(ValueError, match="cannot modify"):
        tool.require_separate_output(directory/"Derived", [bundle])
    with pytest.raises(ValueError, match="outside"):
        tool.SealedBundle({**descriptor, "manifest_file": str(directory/"ManifestV1.json")})
    tool.write(directory/"Data.json", {"value": 2})
    tool.seal(directory)
    with pytest.raises(ValueError, match="external anchor"):
        tool.SealedBundle(descriptor)
    with pytest.raises(ValueError, match="sealed file mismatch"):
        bundle.read("Data.json")


@pytest.mark.parametrize("defect", ("extra_file", "symlink", "changed_byte", "bad_manifest_digest"))
def test_bundle_mutations_do_not_inherit_external_verification(tool, tmp_path, defect):
    directory, external, descriptor = sealed(tool, tmp_path)
    if defect == "extra_file":
        (directory/"Added.json").write_text("{}")
    elif defect == "symlink":
        (directory/"Link.json").symlink_to(external)
    elif defect == "changed_byte":
        (directory/"Data.json").write_text('{"value": 3}')
    else:
        descriptor["manifest_sha256"] = "0"*64
    with pytest.raises(ValueError):
        tool.SealedBundle(descriptor)


def test_collection_requires_external_request_and_three_explicit_baseline_inputs(tool, tmp_path):
    request = {"schema": "R1V5DCBaselineRequestV1", "source_commit": "a"*40,
        "source_content_sha256": "b"*64, "batches": [{}],
        "selections": [{"intervals": n, "case": f"D_N{n}"} for n in (64, 128, 256)]}
    path = tmp_path/"Request.json"
    tool.write(path, request)
    assert tool.load_request(path, tool.sha(path), tmp_path/"out", "collect-dc")[0] == request
    with pytest.raises(ValueError, match="outside output"):
        tool.load_request(path, tool.sha(path), tmp_path, "collect-dc")
    for indices in ([64, 128], [64, 64, 256], [16, 128, 256]):
        wrong = deepcopy(request)
        wrong["selections"] = [{"intervals": n, "case": f"D_N{n}"} for n in indices]
        tool.write(path, wrong)
        with pytest.raises(ValueError, match="exactly one"):
            tool.load_request(path, tool.sha(path), tmp_path/"out", "collect-dc")


def fixture_entry(tool, n=256, factor=.01, time=4, stamp="case"):
    prepared = manufactured_prepared()
    prepared.update(intervals=n, sha256=stamp, source={"sha256": "source"}, kind="equilibrium_D",
        state_time="0-", preparation_controls={"nu_I": 1, "nu_t": 1}, fixed_reference={},
        contact_velocities_m_s={}, contact_certificate={}, dc_state={}, qf_references_V={},
        preparation_policy={}, study_spec_sha256="spec")
    step = manufactured_step(prepared, times=tool.TIMES)
    policy = json.loads(json.dumps(asdict(r1_policy(factor, time_substeps=(time,2*time,4*time)))))
    step.update(source=prepared["source"], policy=policy,
                regular_currents=[{"report_contact_current_A_m2": [1e-5, 1e-5]} for _ in tool.TIMES],
                initial_event={"impulse_charge_C_m2": 2e-6}, certificate={"certified": True})
    rows = step["accepted_steps"]
    for row in rows:
        row["substeps"] = row["substeps"]*time
        row["regular_integrated_charge_C_m2"] = 1e-5*row["time_s"]
    spec = {"control": "D", "intervals": n, "nonlinear_factor": factor,
            "time_substeps": [time,2*time,4*time], "times_s": tool.TIMES, "amplitude_V": .005}
    return {"name": tool.case_name(spec), "request": spec, "status": "completed", "available": True,
            "reason": None, "prepared": prepared, "result": step,
            "replay": {"certified": True, "content_matches_recomputed": True}}


def baseline(entry, stamp="separate_DC_artifact"):
    return {"currents": np.array([0., 0.]), "prepared_sha256": stamp,
            "physical_preparation_identity": physical_preparation_identity(entry["prepared"])}


def test_same_physical_preparation_with_different_artifact_hashes_retains_both(tool):
    left, right = fixture_entry(tool, time=2, stamp="batch_A"), fixture_entry(tool, time=4, stamp="batch_B")
    result = tool.pair_report(left, right, "time_substeps", {256: baseline(left)})
    assert result["comparison"]["required_quantity_count"] == 8
    assert set(result["comparison"]["quantity_passed"]) == {
        "dc_potential", "response_potential", "carrier_log_density", "ion_density_over_p0",
        "trap_occupancy_change", "ion_centroid_change", "regular_current_response", "integrated_charge_response"}
    assert result["comparison"]["convergence_passed"]
    assert [item["case_prepared_sha256"] for item in result["baseline_applications"]] == ["batch_A", "batch_B"]
    assert all(item["baseline_prepared_sha256"] == "separate_DC_artifact" for item in result["baseline_applications"])
    right["prepared"]["dc_state"]["changed_physical_population"] = 1.
    with pytest.raises(ValueError, match="physical case preparation"):
        tool.pair_report(left, right, "time_substeps", {256: baseline(left)})


def test_complete_time_current_and_impulse_charge_differences_are_retained(tool):
    left, right = fixture_entry(tool, time=2), fixture_entry(tool, time=4)
    right["result"]["regular_currents"][1]["report_contact_current_A_m2"][1] += 1e-6
    right["result"]["initial_event"]["impulse_charge_C_m2"] += 1e-7
    result = tool.pair_report(left, right, "time_substeps", {256: baseline(left)})
    assert not result["comparison"]["convergence_passed"]
    values = result["absolute_responses"]
    assert values["regular_current_A_m2"]["absolute_difference"][1][1] == pytest.approx(1e-6)
    assert values["integrated_charge_C_m2"]["absolute_difference"][0] == pytest.approx(1e-7)
    failure = result["comparison"]["electrical"]["regular_current"]["failures"][0]
    assert failure["coordinates"]["time_s"] == 1e-9


def test_missing_finest_pair_is_unknown_and_never_triggers_a_new_solution(tool):
    with read_only_derivation() as attempts:
        report = tool.analyze({"source_commit": "a"*40, "source_content_sha256": "b"*64}, {}, {})
    assert attempts == []
    assert not report["necessary_finest_pairs_passed"]
    assert not report["D2_satisfied"]
    assert report["complete_budget"]["classification"] == "unknown"
    assert report["complete_budget"]["axis_differences_summed"] is False
    assert all(value["classification"] == "unknown" for value in report["empirical_axis_estimates"].values())


def test_even_all_eight_quantity_pairs_passing_cannot_complete_an_unknown_budget(tool):
    entries = [fixture_entry(tool, n=n) for n in (64, 128, 256)]
    entries += [fixture_entry(tool, time=t) for t in (1, 2)]
    entries += [fixture_entry(tool, factor=f) for f in (1., .1)]
    cases = {entry["name"]: entry for entry in entries}
    baselines = {entry["request"]["intervals"]: baseline(entry) for entry in entries[:3]}
    with read_only_derivation() as attempts:
        report = tool.analyze({"source_commit": "a"*40, "source_content_sha256": "b"*64}, cases, baselines)
    assert attempts == [] and report["necessary_finest_pairs_passed"]
    assert all(item["classification"] == "estimate_only" for item in report["empirical_axis_estimates"].values())
    assert report["complete_budget"]["classification"] == "unknown" and not report["D2_satisfied"]
    assert set(report["complete_budget"]["coverage"]["previous_state_propagation_under_axis_changes"]["covered_by"]) == {
        "intervals", "time_substeps", "nonlinear_factor"}
    assert report["D2_assessment_status"] == "not_assessed_by_this_tool"


def test_explicit_extension_chains_require_one_common_finest_endpoint(tool):
    chains = tool.response_chains()
    chains["time_substeps"] = ["D_N256_F0p01_T4", "D_N256_F0p01_T8", "D_N256_F0p01_T16"]
    with pytest.raises(ValueError, match="same finest"):
        tool.validate_chains(chains)
    chains["intervals"] = [name.replace("_T4", "_T16") for name in chains["intervals"]]
    chains["nonlinear_factor"] = [name.replace("_T4", "_T16") for name in chains["nonlinear_factor"]]
    assert tool.validate_chains(chains) == chains
