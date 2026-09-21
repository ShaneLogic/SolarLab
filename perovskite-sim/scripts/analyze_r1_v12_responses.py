"""Collect three source-bound pair DC baselines, then assess the frozen D2 axes.

Every case descriptor holds an external exact manifest. The analysis reads
saved trajectories only; collect-dc explicitly solves and replays D/0 V for
the selected physical preparations. No historical trajectory can satisfy the
current-source request and a coarse comparison failure is retained separately
from the necessary finest comparisons.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from scripts.analyze_r1_v5_responses import (
    PROJECT, THREADS, TIMES, SealedBundle, digest_text, parse, write, sha, seal,
    source_context, response_chains, axis_estimate, require_separate_output,
)
from scripts import run_r1_v12_cases as cases_runner


DC_CASES = tuple(f"D_N{n}_F0p01_T4" for n in (64, 128, 256))
EXTRA_TIME = "D_N256_F0p01_T8"
D2_CASES = tuple(dict.fromkeys(name for chain in response_chains().values() for name in chain)) + (EXTRA_TIME,)
SCHEMAS = {"collect-dc": "R1V12DCBaselineRequestV1", "analyze": "R1V12ResponseAnalysisRequestV1"}


def load_request(path, expected, output, command):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.is_relative_to(output) or sha(path) != digest_text(expected):
        raise ValueError("request must match its external digest outside output")
    request, raw = parse(path.read_bytes()), path.read_bytes()
    required = {"schema", "source_commit", "source_content_sha256", "cases"}
    if command == "analyze":
        required.add("dc_bundle")
    if not isinstance(request, dict) or set(request) != required or request.get("schema") != SCHEMAS[command]:
        raise ValueError("V12 response request schema or fields differ")
    digest_text(request["source_commit"], 40)
    digest_text(request["source_content_sha256"])
    expected_ids = DC_CASES if command == "collect-dc" else D2_CASES
    selected = request["cases"]
    if (not isinstance(selected, list) or len(selected) != len(expected_ids)
            or any(not isinstance(item, dict) or set(item) != {"case_id", "bundle"} for item in selected)
            or set(item["case_id"] for item in selected) != set(expected_ids)):
        raise ValueError("V12 D2 requires the exact distinct source-bound case inventory")
    return request, raw


def preparation_identity(prepared):
    """Physical pair identity plus representation and execution source.

    The physical helper already includes the complete state, including all
    seventeen high/low pairs. The additional source binding forbids transfer
    from formal-study or historical API states with different source identity.
    """
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import physical_preparation_identity, digest
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import decode_prepared, REPRESENTATION
    decode_prepared(prepared, representation=REPRESENTATION)
    content = {"physical": physical_preparation_identity(prepared),
               "representation": prepared["representation"], "source": prepared["source"]}
    return {"schema": "R1V12PairPreparationIdentityV1", "sha256": digest(content),
            "physical_identity": content["physical"], "representation": content["representation"],
            "source_sha256": prepared["source"]["sha256"],
            "scope": "physical_pair_words_and_execution_source; full_artifact_hash_retained_separately"}


def load_cases(request, actual_source, output):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import REPRESENTATION
    bundles, cases = [], {}
    for item in request["cases"]:
        name = item["case_id"]
        bundle = SealedBundle(item["bundle"])
        summary, receipt, prepared = [bundle.read(key) for key in
            ("SummaryV1.json", "SourceReceiptV1.json", "PreparedV1.json")]
        spec = cases_runner.load_request(bundle.checked_path("RequestV1.json"),
            bundle.manifest["RequestV1.json"]["sha256"], output)
        if (summary.get("schema") != "R1V12CaseRunV1" or summary.get("case_id") != name
                or summary.get("case") != spec["case"] or spec.get("case_id") != name
                or summary.get("mode") != "compensated" or summary.get("backend_representation") != REPRESENTATION
                or any(summary.get(key) != request[key] for key in ("source_commit", "source_content_sha256"))
                or summary.get("request_sha256") != bundle.manifest["RequestV1.json"]["sha256"]
                or receipt.get("source_commit") != request["source_commit"]
                or receipt.get("r1_source_content_sha256") != request["source_content_sha256"]
                or prepared.get("source") != actual_source or prepared.get("intervals") != spec["case"]["intervals"]
                or spec["case"]["control"] != "D" or spec["case"]["times_s"] != TIMES):
            raise ValueError("V12 D2 case, source or representation binding differs: " + name)
        available = bool(summary.get("integrity_passed") is True and summary.get("numerical_passed") is True
            and summary.get("four_predicates_passed") is True and summary.get("extent", {}).get("complete") is True
            and summary.get("independent_analysis", {}).get("passed") is True)
        cases[name] = {"case_id": name, "bundle": bundle, "summary": summary, "prepared": prepared,
            "request": spec["case"], "available": available, "identity": preparation_identity(prepared),
            "reason": None if available else "case_not_numerically_complete_with_independent_replay"}
        bundles.append(bundle)
    return bundles, cases


def load_verified_result(entry, actual_source):
    """Read one result lazily; a comparison retains at most two full cases."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import digest
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from scripts.r1_v10_extent import extent
    if not entry["available"]:
        raise ValueError("required D2 case has no complete verified result")
    bundle, spec, prepared = entry["bundle"], entry["request"], entry["prepared"]
    result, replay, ledger = [bundle.read(key) for key in
        ("ResultV1.json", "PhysicsReplayV1.json", "ReplayLedgerV1.json")]
    cases_runner.validate_result_request(result, spec, prepared["sha256"],
        json.loads(json.dumps(asdict(r1_policy(spec["nonlinear_factor"], time_substeps=spec["time_substeps"])))), pair=True)
    rows = result.get("accepted_steps", [])
    if (result.get("source") != actual_source or result.get("sha256") != digest({k: v for k, v in result.items() if k != "sha256"})
            or not extent(rows, spec)["complete"] or result.get("certificate", {}).get("certified") is not True
            or any(replay.get(key) is not True for key in ("certified", "complete", "content_matches_recomputed",
                                                        "physical_limits_satisfied", "independent_physics_passed"))
            or replay.get("checked_row_count") != len(rows)
            or ledger.get("result_sha256") != result["sha256"] or ledger.get("prepared_sha256") != prepared["sha256"]
            or ledger.get("saved_rows_digest") != digest(rows)
            or ledger.get("actual_recomputed_rows_digest") != digest(rows)
            or ledger.get("checked_row_count") != len(rows)):
        raise ValueError("D2 saved result or true replay does not bind all rows")
    with bundle.checked_path("AcceptedStepsV1.jsonl").open() as stream:
        count = 0
        for count, line in enumerate(stream, 1):
            if count > len(rows) or parse(line) != rows[count-1]:
                raise ValueError("D2 observer differs from saved result")
    if count != len(rows):
        raise ValueError("D2 observer coverage differs")
    return result, replay


def collect_dc(request, cases, output, actual_source, context):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data, restore_common_state
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
        solve_controlled_dc, restore_controlled_dc, verify_response_content)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import write_numeric_sidecar, verify_numeric_sidecar
    from perovskite_sim.models.config_loader import load_device_from_yaml
    fixture = context.read_bytes(cases_runner.INPUTS["fixture"]["path"])
    binding = parse(context.read_bytes(cases_runner.INPUTS["reference"]["path"]))
    (output/"SourceFixtureV1.yaml").write_bytes(fixture)
    write(output/"ReferenceBindingV1.json", binding)
    stack = load_device_from_yaml(output/"SourceFixtureV1.yaml")
    summary = {"schema": "R1V12DCBaselineCollectionV1", "source_commit": request["source_commit"],
        "source_content_sha256": request["source_content_sha256"], "actual_source": actual_source,
        "representation": "float64-pair-v1", "entries": [], "all_completed": False}
    for name in DC_CASES:
        entry = cases[name]
        n, prepared = entry["request"]["intervals"], entry["prepared"]
        item = {"case_id": name, "intervals": n, "control": "D", "voltage_V": 0., "status": "not_run",
                "prepared_sha256": prepared["sha256"], "preparation_identity": entry["identity"]}
        summary["entries"].append(item)
        if not entry["available"]:
            item.update(status="unavailable", reason=entry["reason"])
            continue
        # Before using the state, verify the original exact accepted evidence.
        load_verified_result(entry, actual_source)
        directory = output/f"D_N{n}"
        directory.mkdir()
        write(directory/"PreparedV1.json", prepared)
        dc_request = {"intervals": n, "control": "D", "voltage_V": 0., "nonlinear_factor": .1,
            "time_substeps": [1, 2, 4], "prepared_sha256": prepared["sha256"],
            "reference_sha256": binding["sha256"], "source_sha256": actual_source["sha256"],
            "representation": "float64-pair-v1"}
        write(directory/"RequestV1.json", dc_request)
        try:
            restore_common_state(prepared, stack, n, binding, policy=r1_policy(),
                expected_prepared_sha256=prepared["sha256"], backend="pair")
            dc = solve_controlled_dc(stack, n, binding, prepared, control="D", voltage_V=0.,
                policy=r1_policy(), expected_prepared_sha256=prepared["sha256"], backend="pair")
            write(directory/"DCV1.json", json_data(dc.evidence))
            saved = parse((directory/"DCV1.json").read_bytes())
            write_numeric_sidecar(directory/"DCNumericV1.npz", dc.evidence)
            numeric = verify_numeric_sidecar(directory/"DCNumericV1.npz", saved)
            write(directory/"NumericReadbackV1.json", json_data(numeric))
            restored = restore_controlled_dc(saved, stack, n, binding, prepared, backend="pair")
            replay = verify_response_content(saved, stack=stack, intervals=n, binding=binding,
                prepared=prepared, request=dc_request, backend="pair")
            if not restored.evidence["certified"] or not replay["certified"]:
                raise ValueError("DC restore or complete source-bound replay failed")
            write(directory/"DCReplayV1.json", json_data(replay))
            item.update(status="completed", dc_file_sha256=sha(directory/"DCV1.json"),
                replay_file_sha256=sha(directory/"DCReplayV1.json"), numeric_sidecar_exact=True)
        except Exception as exc:
            item.update(status="failed", reason=str(exc), exception_type=type(exc).__name__)
            write(directory/"FailureV1.json", {**item, "result": json_data(getattr(exc, "result", None))})
        write(output/"SummaryV1.json", summary)
    summary["all_completed"] = all(item["status"] == "completed" for item in summary["entries"])
    write(output/"SummaryV1.json", summary)
    return 0 if summary["all_completed"] else 2


def load_baselines(request, actual_source):
    from perovskite_sim.physics.compensated import DD
    bundle = SealedBundle(request["dc_bundle"])
    summary, completion = bundle.read("SummaryV1.json"), bundle.read("CompletionV1.json")
    if (summary.get("schema") != "R1V12DCBaselineCollectionV1" or summary.get("actual_source") != actual_source
            or any(summary.get(key) != request[key] for key in ("source_commit", "source_content_sha256"))
            or completion.get("completed") is not True or completion.get("command") != "collect-dc"
            or completion.get("input_bytes_and_source_unchanged") is not True
            or sorted(item["intervals"] for item in summary.get("entries", [])) != [64, 128, 256]):
        raise ValueError("V12 DC source, completion or case inventory differs")
    baselines = {}
    for item in summary["entries"]:
        if item["status"] != "completed":
            continue
        n = item["intervals"]
        prepared, dc, replay = [bundle.read(f"D_N{n}/"+key) for key in
            ("PreparedV1.json", "DCV1.json", "DCReplayV1.json")]
        identity = preparation_identity(prepared)
        if (dc.get("schema") != "R1ControlledDCResponseV2" or dc.get("representation") != "float64-pair-v1"
                or dc.get("intervals") != n or dc.get("control") != "D" or dc.get("voltage_V") != 0.
                or dc.get("source") != actual_source or prepared.get("source") != actual_source
                or dc.get("prepared_sha256") != prepared["sha256"] or prepared["sha256"] != item["prepared_sha256"]
                or item["preparation_identity"] != identity or dc.get("certified") is not True
                or any(replay.get(key) is not True for key in ("certified", "content_matches_recomputed", "equations_replayed",
                                                            "precision_state_and_observations_recomputed"))
                or replay.get("prepared_sha256") != prepared["sha256"] or replay.get("source") != actual_source
                or item.get("numeric_sidecar_exact") is not True
                or item.get("dc_file_sha256") != bundle.manifest[f"D_N{n}/DCV1.json"]["sha256"]
                or item.get("replay_file_sha256") != bundle.manifest[f"D_N{n}/DCReplayV1.json"]["sha256"]):
            raise ValueError("V12 DC baseline is not bound to exact preparation and full physical replay")
        currents = dc["precision_current_A_m2"]
        values = DD(currents["hi"], currents["lo"])
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("pair DC baseline must retain both contacts")
        baselines[n] = {"currents": values[[0, -1]], "prepared_sha256": prepared["sha256"],
                        "preparation_identity": identity, "dc": dc, "prepared": prepared}
    return bundle, baselines


def pair_report(left, right, axis, baselines, actual_source):
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import compare_pair_step_current_charge
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_spatial import spatial_responses_from_step, compare_convergence_responses
    a, replay_a = load_verified_result(left, actual_source)
    b, replay_b = load_verified_result(right, actual_source)
    dc_records, dc_preparations, applications = [], [], []
    for entry in (left, right):
        baseline = baselines[entry["request"]["intervals"]]
        if baseline["preparation_identity"] != entry["identity"]:
            raise ValueError("pair DC physical high/low preparation or source differs from case")
        dc_records.append(baseline["dc"])
        dc_preparations.append(baseline["prepared"])
        applications.append({"case_prepared_sha256": entry["prepared"]["sha256"],
            "baseline_prepared_sha256": baseline["prepared_sha256"], "preparation_identity": entry["identity"]})
    responses = [electrical_responses(record, dc, step_prepared=entry["prepared"], dc_prepared=dc_prepared)
        for record, dc, entry, dc_prepared in zip((a, b), dc_records, (left, right), dc_preparations)]
    electrical = compare_pair_step_current_charge(a, b, *dc_records, expected_times_s=TIMES,
        left_prepared=left["prepared"], right_prepared=right["prepared"],
        left_dc_prepared=dc_preparations[0], right_dc_prepared=dc_preparations[1])
    comparison = compare_convergence_responses(spatial_responses_from_step(left["prepared"], a),
        spatial_responses_from_step(right["prepared"], b), axis=axis, electrical=electrical,
        input_verifications=(replay_a, replay_b))
    absolute = {}
    for index, key in enumerate(("regular_current_A_m2", "integrated_charge_C_m2")):
        values = [item[index].values for item in responses]
        absolute[key] = {"left": values[0].tolist(), "right": values[1].tolist(),
                         "absolute_difference": np.abs(values[0]-values[1]).tolist()}
    return {"available": True, "comparison": comparison, "absolute_responses": absolute,
        "baseline_applications": applications, "times_s": TIMES,
        "current_components": ["left_contact", "right_contact"],
        "charge_definition": "left_contact_regular_integral_plus_ideal_step_impulse_minus_DC_times_t",
        "verification_scope": "externally_anchored_complete_true_replay; no_new_physical_solve"}


def electrical_responses(record, dc_record, *, step_prepared, dc_prepared):
    """Use the production pair consumer with both original preparations intact."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import pair_step_current_charge_responses
    return pair_step_current_charge_responses(record, dc_record, expected_times_s=TIMES,
        step_prepared=step_prepared, dc_prepared=dc_prepared)


def analyze(request, cases, baselines, actual_source):
    reports, estimates, missing = {}, {}, []
    for axis, names in response_chains().items():
        pairs = []
        for index, (left, right) in enumerate(zip(names, names[1:])):
            dependencies = [name for name in (left, right) if name not in cases
                or not cases[name]["available"] or cases[name]["request"]["intervals"] not in baselines]
            value = ({"available": False, "missing_dependencies": dependencies} if dependencies
                     else pair_report(cases[left], cases[right], axis, baselines, actual_source))
            value.update(axis=axis, left=left, right=right, required_finest_pair=index == 1)
            pairs.append(value)
        reports[axis] = pairs
        estimates[axis] = axis_estimate(*pairs)
        if not pairs[-1]["available"] or not pairs[-1]["comparison"]["convergence_passed"]:
            missing.append(axis+"_required_finest_pair_not_passed")
    left, right = "D_N256_F0p01_T4", EXTRA_TIME
    dependencies = [name for name in (left, right) if name not in cases or not cases[name]["available"] or 256 not in baselines]
    extra = ({"available": False, "missing_dependencies": dependencies} if dependencies else
             pair_report(cases[left], cases[right], "time_substeps", baselines, actual_source))
    extra.update(axis="time_substeps", left=left, right=right, required_finest_pair=True)
    extra_passed = bool(extra["available"] and extra["comparison"]["convergence_passed"])
    if not extra_passed:
        missing.append("additional_T4_to_T8_required_pair_not_passed")
    necessary = all(items[-1]["available"] and items[-1]["comparison"]["convergence_passed"] for items in reports.values())
    return {"schema": "R1V12DResponseAnalysisV1", "source_commit": request["source_commit"],
        "source_content_sha256": request["source_content_sha256"], "representation": "float64-pair-v1",
        "chains": response_chains(), "pairs": reports, "additional_fine_time_pair": extra,
        "empirical_axis_estimates": estimates, "necessary_finest_pairs_passed": bool(necessary),
        "additional_fine_time_pair_passed": extra_passed, "short_window_D2_gate_passed": bool(necessary and extra_passed),
        "short_window_D2_remaining_requirements": missing, "D2_satisfied": False,
        "D2_assessment_status": "necessary_frozen_short_window_comparisons_only; no_total_error_bound_or_unsampled_scope",
        "historical_coarse_time_failure": "T1_to_T2_1ns_failure_retained_as_history; current_coarse_result_reported_separately",
        "scope": "five_sample_D_control_eight_quantity_comparisons_not_full_R1_2_or_long_window_qualification"}


def run(args):
    if not (sys.flags.isolated and sys.flags.no_site) or any(os.environ.get(key) != "1" for key in THREADS):
        raise ValueError("V12 response tools require -I -S and all five thread limits equal to 1")
    output = args.output.resolve()
    request, raw = load_request(args.request_file, args.request_sha256, output, args.command)
    if output.exists():
        raise FileExistsError("V12 response output already exists")
    context, actual_source = source_context(request)
    bundles, cases = load_cases(request, actual_source, output)
    if args.command == "analyze":
        dc_bundle, baselines = load_baselines(request, actual_source)
        bundles.append(dc_bundle)
    require_separate_output(output, bundles)
    output.mkdir(parents=True)
    (output/"RequestV1.json").write_bytes(raw)
    write(output/"SourceReceiptV1.json", {"source_commit": request["source_commit"],
        "source_content_sha256": request["source_content_sha256"], "actual_source": actual_source, "script_sha256": sha(__file__)})
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification_workflow import read_only_derivation
    with threadpool_limits(1):
        import numpy as np
        import scipy
        np.dot(np.ones((2, 2)), np.ones((2, 2)))
        pools = threadpool_info()
        if not pools or any(item["num_threads"] != 1 for item in pools) or np.__version__ != "2.1.3" or scipy.__version__ != "1.15.3":
            raise ValueError("V12 response runtime differs from its frozen dependency lineage")
        write(output/"EnvironmentV1.json", {"numpy": np.__version__, "scipy": scipy.__version__,
            "executable": sys.executable, "python": sys.version, "blas": pools, "threads": {k: os.environ[k] for k in THREADS}})
        if args.command == "collect-dc":
            code = collect_dc(request, cases, output, actual_source, context)
        else:
            with read_only_derivation() as attempted:
                report = analyze(request, cases, baselines, actual_source)
            report["new_physical_solution_attempts"] = attempted
            write(output/"ResponseAnalysisV1.json", json_data(report))
            code = 0 if report["short_window_D2_gate_passed"] else 2
    for bundle in bundles:
        bundle.verify()
    source_context(request)
    if sha(args.request_file) != args.request_sha256:
        raise ValueError("external V12 response request changed during execution")
    write(output/"CompletionV1.json", {"completed": True, "command": args.command, "exit_code": code,
        "input_bytes_and_source_unchanged": True, "D2_qualification_asserted": False})
    seal(output)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(SCHEMAS))
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
