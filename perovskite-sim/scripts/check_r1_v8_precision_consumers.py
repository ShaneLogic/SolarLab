"""Actual V8 DD healthy/fault cases on identified recorded states.

No new time integration is performed.  Saved states are reconstructed through
the actual V8 initializer/evaluator, and exact replay identity is required for
final-source evidence.  Explicit historical inspection remains diagnostic.
Each shared fault traverses both the actual direct evaluator and the actual
independent eliminated solve; this is not the V7 repeated-evaluator surrogate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, localcontext
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from perovskite_sim.physics.compensated import DD
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as production
from scripts.analyze_r1_v7_prototype import canonical, sha, write
from scripts.analyze_r1_v8_prototype import EVIDENCE_TOOLS, initial_checks, verify_inputs
from scripts.check_r1_v7_precision_consumers import (
    decimals, delta_report, jacobian_checks, poisson_low_response_report, state_record, words,
)
from scripts.verify_r1_v7_precision import dec, evaluate_state, verify_side
from scripts.verify_r1_v8_precision import (
    arithmetic_comparison, as_snapshot, observed, precision_fields, represented, verify_two_sides,
)


PROJECT = Path(__file__).resolve().parents[1]
FAULTS = ("diffusion", "thermal_voltage", "drift_sign", "omit_ion", "single_face_sign")
SELECTIONS = {"ordinary": (4, 2.5e-10), "late": (4, 1.9663570500354033),
              "single_path": (4, 0.7173200981104755)}
HEALTHY_PHI_PERTURBATION_V = 1e-23
PATH_PHI_PERTURBATION_V = 1e-13
BOUNDARY_LEAK_M2_S = 1e-6


def validate_case_contract(request, calibration, calibration_sha256, *, historical=False):
    wanted = {"substeps": SELECTIONS["single_path"][0], "time_s": SELECTIONS["single_path"][1], "node": 7}
    if (calibration.get("schema") != "R1V8PathFaultCalibrationV1"
            or calibration.get("amplitude_V") != PATH_PHI_PERTURBATION_V
            or calibration.get("signs") != [-1, 1]
            or any(calibration.get("selection", {}).get(key) != value for key, value in wanted.items())):
        raise ValueError("fixed path calibration selection, amplitude or signs differ")
    if historical:
        return
    contract = request.get("validation_contract", {})
    required = {"schema": "R1V8ValidationContractV1", "path_node": 7,
                "healthy_potential_perturbation_V": [-HEALTHY_PHI_PERTURBATION_V, HEALTHY_PHI_PERTURBATION_V],
                "path_potential_perturbation_V": [-PATH_PHI_PERTURBATION_V, PATH_PHI_PERTURBATION_V],
                "boundary_leak_m2_s": BOUNDARY_LEAK_M2_S, "path_calibration_sha256": calibration_sha256,
                "shared_faults": {"diffusion_multiplier": 1.01, "thermal_voltage_multiplier": 1.01,
                    "drift_sign": -1, "face_sign": "first_active_nonzero_face", "omitted_ion": True}}
    required.update({key: {"substeps": SELECTIONS[label][0], "time_s": SELECTIONS[label][1], "row": row}
        for key, label, row in (("ordinary_state", "ordinary", 342), ("late_state", "late", 791),
                                ("path_state", "single_path", 770))})
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("V8 request validation matrix differs from the predeclared implementation")


def decode(state):
    return {name: DD(value["hi"], value["lo"]) for name, value in precision_fields(state).items()}


def snapshot(state):
    return {"precision_" + name + "_" + part: getattr(value, part).tolist()
            for name, value in state.fine.items() for part in ("hi", "lo")}


@contextmanager
def replace_method(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def select_rows(directory):
    rows = [json.loads(line) for line in (Path(directory) / "AcceptedStepsV1.jsonl").read_text().splitlines()]
    result = {}
    for label, identity in SELECTIONS.items():
        hits = [(index, row) for index, row in enumerate(rows) if (row["substeps"], row["time_s"]) == identity]
        if len(hits) != 1 or hits[0][0] == 0:
            raise ValueError("required actual state is absent or ambiguous: " + label)
        index, row = hits[0]
        previous = rows[index - 1]
        if previous["substeps"] != row["substeps"] or not previous["time_s"] < row["time_s"]:
            raise ValueError("selected state has no valid same-tier predecessor")
        result[label] = (index, previous, row)
    return result


def rebuild_context(directory, request, prepared, *, historical=False):
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision_state import R1CommonStatePair
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    stack = load_device_from_yaml(Path(directory) / "SourceFixtureV1.yaml")
    binding = json.loads((Path(directory) / "ReferenceBindingV1.json").read_text())
    # A historical package cannot be relabelled as current-source preparation.
    # For explicit diagnostic use, build a new current-source common state
    # from the unchanged raw inputs and retain both identities in the report.
    pair = (states.prepare_common_state(stack, request["case"]["intervals"], binding, policy=r1_policy())
            if historical else R1CommonStatePair.from_dict(prepared))
    system, minus = states.verify_prepared_physics(pair, stack, binding, policy=r1_policy())
    event = build_initial_step(system, minus, request["case"]["amplitude_V"], policy=r1_policy(
        request["case"]["nonlinear_factor"], time_substeps=tuple(request["case"]["time_substeps"])))
    return system, minus, event, pair.sha256


def reconstruct_selected(initial_system, previous_row, row, *, require_exact=True):
    import copy
    owner = copy.copy(initial_system)
    owner._fine_reference = decode(previous_row["state"])
    owner._fine_work = {}
    voltage = row["physics_reconstruction"]["voltage_V"]
    previous = owner.evaluate(np.zeros(owner.dimension), voltage)
    working, local_previous = owner.rebase(previous)
    working.set_voltage_lift(voltage, local_previous)
    coordinate = np.asarray(row["physics_reconstruction"]["coordinate"])
    state = working.evaluate(coordinate, voltage)
    saved = decode(row["state"])
    matches = {name: name in state.fine and np.array_equal(value.hi, state.fine[name].hi)
               and np.array_equal(value.lo, state.fine[name].lo) for name, value in saved.items()}
    if require_exact and not all(matches.values()):
        raise ValueError("selected V8 state failed exact reconstruction: " + str([k for k, v in matches.items() if not v]))
    return working, state, {"all_saved_fine_fields_bit_identical": all(matches.values()), "fields": matches}


def actual_two_sides(owner, state, voltage, frozen, upstream):
    diagnostics = owner.eliminated_operator_diagnostics(state, voltage)
    record = diagnostics.precision_evidence
    row = {"state": snapshot(state), "physics_reconstruction": {"eliminated_precision": record}}
    oracle = verify_two_sides(frozen, row, upstream=upstream)
    maximum = max(float(item["relative_error"]) for item in diagnostics.values())
    return {"oracle": oracle, "original_gate_maximum": maximum, "original_gate_passed": maximum <= 1e-6,
            "original_components": {name: {key: value for key, value in item.items()
                if key in ("maximum_absolute_difference", "normalization_scale", "normalization_floor", "relative_error", "unit")}
                for name, item in diagnostics.items()}, "actual_eliminated_record": record,
            "direct_state": snapshot(state)}


def changed_reference(owner, state, voltage, field, node, amount):
    from dataclasses import replace
    fine = {name: value.copy() for name, value in state.fine.items()}
    fine[field] = production.put(fine[field], node, fine[field][node] + DD(amount))
    working, _ = owner.rebase(replace(state, fine=fine))
    return working, working.evaluate(np.zeros(working.dimension), voltage)


def late_ion_column(owner, state, frozen, node):
    with localcontext() as context:
        context.prec = 100
        fields = {name: [str(x) for x in decimals(state.fine[name])] for name in
                  ("phi_V", "n_m3", "p_m3", "positive_m3", "sheet_charge_C_m2")}
        population = dec(fields["positive_m3"][node])
        h = population * Decimal("1e-24")
        plus, minus = ({name: list(value) for name, value in fields.items()} for _ in range(2))
        plus["positive_m3"][node], minus["positive_m3"][node] = str(population + h), str(population - h)
        a, b = evaluate_state(frozen, plus, precision=100), evaluate_state(frozen, minus, precision=100)
        expected = [(dec(x) - dec(y)) / (2 * h) * population
                    for x, y in zip(a["positive_rate_m3_s"], b["positive_rate_m3_s"])]
        owner._fine_work = {name: value.copy() for name, value in state.fine.items()}
        matrix = owner._ion_jacobians(state.phi, state.positive, None)[2]
        column = owner.positive_slice.start + list(owner.positive_nodes).index(node)
        report = delta_report(matrix[:, column].toarray().ravel(), expected, relative_limit="1e-10")
        report.update(node=node, coordinate_column=column, independent_difference_step_m3=str(h),
                      scope="one_declared_late_ion_column_not_all_trajectory_Jacobians")
        return report


def low_part_consumers(system, initial, frozen):
    """Preserve the V7 named low-part budgets on the actual new 0- state."""
    node = int(frozen["coefficients"]["interface_nodes"][0][0])
    owner, low = changed_reference(system, initial, 0., "phi_V", node, HEALTHY_PHI_PERTURBATION_V)
    checks, actual_inputs = {}, {"base": snapshot(initial), "phi_low": snapshot(low)}
    with localcontext() as context:
        context.prec = 100
        expected_phi = [dec(HEALTHY_PHI_PERTURBATION_V) if j == node else Decimal(0)
                        for j in range(system.node_count)]
        checks["rebase_potential"] = delta_report(low.fine["phi_V"]-initial.fine["phi_V"], expected_phi)
        checks["potential_increment"] = delta_report(owner.potential_increment(low, initial), expected_phi)
        q = dec(frozen["constants"]["q_C"])
        widths = [dec(x) for x in frozen["coefficients"]["physical_width_m"]]
        base_oracle = evaluate_state(frozen, state_record(initial))
        low_oracle = evaluate_state(frozen, state_record(low))
        checks["ion_flux_phi_low"] = delta_report(
            low.fine["positive_flux_m2_s"]-initial.fine["positive_flux_m2_s"],
            [dec(a)-dec(b) for a,b in zip(low_oracle["positive_flux_m2_s"], base_oracle["positive_flux_m2_s"])],
            relative_limit="1e-6")
        _, displacement, _ = owner.interface_current_sides(low, initial, 1.)
        cl = production.EPS_0*system.material.eps_r[node]/system.material.iface_qss_left_distances_m[0]
        checks["interface_displacement_phi_low"] = delta_report(displacement[0],
            [dec(system.polarity)*dec(cl)*dec(HEALTHY_PHI_PERTURBATION_V), Decimal(0)])
        for field, change in (("p_m3", 1e-5), ("positive_m3", 1000.)):
            changed_owner, state = changed_reference(system, initial, 0., field, node, change)
            label = "hole" if field == "p_m3" else "ion"
            actual_inputs[label+"_low"] = snapshot(state)
            checks["rebase_"+label] = delta_report(state.fine[field]-initial.fine[field],
                [dec(change) if j == node else Decimal(0) for j in range(system.node_count)])
            storage = changed_owner.storage_increment(state, initial)
            slot = system.interior_count+node-1 if label == "hole" else (
                2*system.interior_count+system.interface_count+list(system.positive_nodes).index(node))
            expected_storage = [Decimal(0)]*len(storage)
            expected_storage[slot] = dec(change)
            checks["storage_"+label+"_low"] = delta_report(storage, expected_storage)
            checks["charge_"+label+"_low"] = delta_report(
                [changed_owner.integrated_charge_increment(state, initial)], [q*dec(change)*widths[node]])
            new_residual = changed_owner._poisson_pair(*(state.fine[name] for name in
                ("phi_V", "n_m3", "p_m3", "positive_m3", "sheet_charge_C_m2")))
            old_residual = system._poisson_pair(*(initial.fine[name] for name in
                ("phi_V", "n_m3", "p_m3", "positive_m3", "sheet_charge_C_m2")))
            checks["poisson_"+label+"_low"] = poisson_low_response_report(
                frozen, state, initial, new_residual-old_residual)
            if label == "ion":
                ion_oracle = evaluate_state(frozen, state_record(state))
                checks["ion_flux_population_low"] = delta_report(
                    state.fine["positive_flux_m2_s"]-initial.fine["positive_flux_m2_s"],
                    [dec(a)-dec(b) for a,b in zip(ion_oracle["positive_flux_m2_s"], base_oracle["positive_flux_m2_s"])],
                    relative_limit="1e-8")
    checks["jacobian"] = jacobian_checks(system, initial, owner, low, frozen, node)
    return {"passed": all(value["passed"] for value in checks.values()), "checks": checks,
            "actual_inputs": actual_inputs, "scope": "V7_named_budgets_on_new_actual_initial_low_part_consumers"}


def run_case_matrix(initial_system, zero_minus_system, zero_minus, zero_plus, selected, frozen, *, historical=False):
    cases, shared, path_cases = {}, [], []
    rebuilt = {}
    upstream = zero_minus_system.precision_arithmetic_context()
    for label, (index, previous, row) in selected.items():
        owner, state, identity = reconstruct_selected(initial_system, previous, row, require_exact=not historical)
        voltage = row["physics_reconstruction"]["voltage_V"]
        checked = actual_two_sides(owner, state, voltage, frozen, upstream)
        cases[label] = {"row": index, "substeps": row["substeps"], "time_s": row["time_s"],
                        "saved_state_sha256": hashlib_state(row["state"]), "reconstruction": identity, "check": checked}
        rebuilt[label] = (owner, state, voltage, row)
    for label in ("ordinary", "late"):
        owner, _, voltage, row = rebuilt[label]
        for fault in FAULTS:
            owner.precision_fault = fault
            try:
                faulty = owner.evaluate(np.asarray(row["physics_reconstruction"]["coordinate"]), voltage)
                checked = actual_two_sides(owner, faulty, voltage, frozen, upstream)
            finally:
                owner.precision_fault = "none"
            direct, eliminated = checked["oracle"].get("direct"), checked["oracle"].get("eliminated")
            detected = (cases[label]["check"]["oracle"]["qualified"] and direct is not None and eliminated is not None
                        and not direct["qualified"] and not eliminated["qualified"])
            shared.append({"case": label, "fault": fault, "detected": detected, "check": checked,
                           "injection": "actual_direct_evaluate_and_actual_independent_eliminated_solver_constitutive_parameter"})
    owner, state, voltage, _ = rebuilt["single_path"]
    node = int(frozen["coefficients"]["interface_nodes"][0][0])
    for sign in (-1, 1):
        def perturb(phi, direction=sign):
            return production.put(phi, node, phi[node] + DD(direction * PATH_PHI_PERTURBATION_V))
        with replace_method(owner, "_eliminated_flux_potential", perturb):
            checked = actual_two_sides(owner, state, voltage, frozen, upstream)
        healthy_case = cases["single_path"]["check"]
        baseline_fields = healthy_case["actual_eliminated_record"]["fields"]
        actual_fields = checked["actual_eliminated_record"]["fields"]
        with localcontext() as context:
            context.prec = 100
            wanted = represented(baseline_fields["phi_V"])
            wanted[node] += dec(sign*PATH_PHI_PERTURBATION_V)
            arithmetic = arithmetic_comparison(actual_fields["phi_V"], wanted, voltage=True)
        input_checks = {"direct_state_unchanged": canonical(checked["direct_state"]) == canonical(healthy_case["direct_state"]),
            "constraint_potential_unchanged": canonical(actual_fields["constraint_phi_V"]) == canonical(baseline_fields["constraint_phi_V"]),
            "actual_injected_potential": arithmetic}
        path_cases.append({"sign": sign, "node": node, "amplitude_V": PATH_PHI_PERTURBATION_V,
                           "detected": healthy_case["original_gate_passed"] and not checked["original_gate_passed"]
                            and input_checks["direct_state_unchanged"] and input_checks["constraint_potential_unchanged"]
                            and arithmetic["passed"], "check": checked, "input_checks": input_checks,
                           "injection": "only_independent_eliminated_potential_after_constraint_solve_before_flux"})
    healthy = []
    for label, owner, state, voltage in (("zero_minus", zero_minus_system, zero_minus, 0.),
                                       ("zero_plus", initial_system, zero_plus, voltage)):
        health = verify_side(frozen, state_record(state), observed(precision_fields(snapshot(state))))
        healthy.append({"case": label, "oracle": health, "actual_state": snapshot(state)})
    for sign in (-1, 1):
        owner, state = changed_reference(zero_minus_system, zero_minus, 0., "phi_V", node,
                                         sign * HEALTHY_PHI_PERTURBATION_V)
        health = verify_side(frozen, state_record(state), observed(precision_fields(snapshot(state))))
        healthy.append({"case": "signed_phi_operator", "sign": sign, "amplitude_V": HEALTHY_PHI_PERTURBATION_V,
                        "oracle": health, "actual_state": snapshot(state),
                        "scope": "controlled_operator_input_not_an_accepted_state"})
    active = list(initial_system.positive_nodes)
    limits = np.asarray(initial_system.material.P_lim_node)
    if not np.all(limits[active] == limits[active[0]]):
        raise ValueError("declared strict-zero sample requires the frozen uniform active site limit")
    zero_fields = {name: value.copy() for name, value in zero_minus.fine.items()}
    zero_fields["phi_V"] = DD(np.zeros(initial_system.node_count))
    zero_fields["positive_m3"] = DD(np.full(initial_system.node_count, 0.1 * np.min(limits)))
    flux = production.ion_flux_pair(zero_fields["phi_V"], zero_fields["positive_m3"], initial_system.material,
                                    np.diff(initial_system.grid))
    boundary = DD(np.zeros(2))
    rate = -production.diff(production.cat(boundary[0], flux, boundary[1])) / DD(initial_system.widths)
    zero_fields.update(positive_flux_m2_s=flux, positive_rate_m3_s=rate, boundary_flux_m2_s=boundary)
    packed = {name: words(value) for name, value in zero_fields.items()}
    zero_check = verify_side(frozen, as_snapshot(packed), observed(packed))
    healthy.append({"case": "analytic_strict_zero_actual_DD_kernel", "oracle": zero_check,
                    "actual_state": as_snapshot(packed),
                    "exact_zero": bool(np.all(flux.hi == 0) and np.all(flux.lo == 0)),
                    "scope": "analytic_constitutive_sample_not_actual_device_equilibrium"})
    owner, late, voltage, _ = rebuilt["late"]
    low_owner, low_state = changed_reference(owner, late, voltage, "phi_V", node, HEALTHY_PHI_PERTURBATION_V)
    potential_column = jacobian_checks(owner, late, low_owner, low_state, frozen, node)
    ion_column = late_ion_column(owner, late, frozen, node)
    with replace_method(owner, "_ion_boundary_flux", lambda: DD([BOUNDARY_LEAK_M2_S, 0.])):
        leaked = owner.evaluate(np.asarray(rebuilt["late"][3]["physics_reconstruction"]["coordinate"]), voltage)
        leakage = actual_two_sides(owner, leaked, voltage, frozen, upstream)
    leakage_detected = all(not leakage["oracle"][side]["checks"]["boundary_flux_m2_s"]["passed"]
                           for side in ("direct", "eliminated"))
    consumers = low_part_consumers(zero_minus_system, zero_minus, frozen)
    evidence_passed = (consumers["passed"] and all(value["check"]["oracle"]["qualified"] for value in cases.values())
                       and all(value["detected"] for value in shared + path_cases)
                       and all(value["oracle"]["qualified"] for value in healthy) and healthy[-1]["exact_zero"]
                       and potential_column["passed"] and ion_column["passed"] and leakage_detected)
    return {"schema": "R1V8ActualPrecisionCasesV1", "P1_qualified": False, "actual_accepted_new_steps": 0,
            "historical_input": historical, "evidence_subset_passed": evidence_passed,
            "final_evidence_eligible": evidence_passed and not historical, "cases": cases,
            "shared_faults": shared, "single_path_faults": path_cases, "healthy": healthy,
            "low_part_consumers": consumers,
            "late_columns": {"potential": potential_column, "ion": ion_column},
            "boundary_fault": {"amplitude_m2_s": BOUNDARY_LEAK_M2_S, "detected": leakage_detected, "check": leakage}}


def hashlib_state(state):
    import hashlib
    return hashlib.sha256(canonical(state).encode()).hexdigest()


def current_source():
    files = sorted((PROJECT / "perovskite_sim").rglob("*.py"))
    files.extend(PROJECT / name for name in (*EVIDENCE_TOOLS, "scripts/check_r1_v8_precision_consumers.py",
                                           "scripts/check_r1_v7_precision_consumers.py"))
    entries = {path.relative_to(PROJECT).as_posix(): sha(path) for path in files}
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True),
            "source_files": entries, "source_files_sha256": hashlib_state(entries)}


def execution_source_matches(directory, summary, source):
    receipt = json.loads((Path(directory)/"SourceReceiptV1.json").read_text())
    tracked = receipt.get("all_tracked_files", {})
    matches = {name: value == tracked.get("perovskite-sim/"+name)
               for name,value in source["source_files"].items()}
    return {"passed": source["commit"] == summary["source_commit"] and not source["git_status"].strip()
                and bool(matches) and all(matches.values()),
            "same_commit": source["commit"] == summary["source_commit"],
            "clean_checkout": not source["git_status"].strip(), "source_files": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--budget-sha256")
    parser.add_argument("--path-calibration", type=Path, required=True)
    parser.add_argument("--path-calibration-sha256", required=True)
    parser.add_argument("--allow-historical-input", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() == args.run_dir.resolve() or args.output.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError("consumer diagnostics cannot write into the sealed run")
    if sha(args.path_calibration) != args.path_calibration_sha256:
        raise ValueError("path-fault calibration differs from its frozen identity")
    calibration = json.loads(args.path_calibration.read_text())
    request, summary, frozen, prepared, _, _, identity = verify_inputs(
        args.run_dir, args.budget_file, args.budget_sha256, historical=args.allow_historical_input,
        evidence_tools=(*EVIDENCE_TOOLS, "scripts/check_r1_v8_precision_consumers.py",
                        "scripts/check_r1_v7_precision_consumers.py"))
    validate_case_contract(request, calibration, args.path_calibration_sha256, historical=args.allow_historical_input)
    selected = select_rows(args.run_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = current_source()
    execution_binding = execution_source_matches(args.run_dir, summary, before)
    report = {"schema": "R1V8ActualPrecisionCasesV1", "P1_qualified": False, "evidence_subset_passed": False,
              "final_evidence_eligible": False, "input_identity": identity, "actual_current_source_before": before,
              "execution_source_matches_run": execution_binding,
              "path_calibration_sha256": args.path_calibration_sha256, "run_class": "development"}
    write(args.output / "InputsV1.json", {"identity": identity, "source": before,
        "path_calibration": calibration, "actual_saved_inputs": {
            label: {"row": index, "previous": previous, "selected": row}
            for label, (index, previous, row) in selected.items()}})
    try:
        if not args.allow_historical_input and not execution_binding["passed"]:
            raise ValueError("actual consumer execution requires the complete clean recorded V8 source")
        with threadpool_limits(1), production.precision_context():
            system, minus, event, actual_preparation_sha256 = rebuild_context(
                args.run_dir, request, prepared, historical=args.allow_historical_input)
            initial_result = {"initial_event": event.event,
                              "physics_reconstruction": {"initial_state": snapshot(event.zero_plus)}}
            write(args.output / "ActualInitialInputsV1.json", initial_result)
            report["initial_arithmetic"] = initial_checks(frozen, {}, initial_result)
            report["actual_current_source_preparation_sha256"] = actual_preparation_sha256
            report.update(run_case_matrix(event.system, system, minus, event.zero_plus, selected, frozen,
                                          historical=args.allow_historical_input))
            report["evidence_subset_passed"] &= report["initial_arithmetic"]["qualified"]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    after = current_source()
    report.update(wall_seconds=time.monotonic()-start, actual_current_source_after=after,
                  actual_current_source_unchanged=before == after, source_script_sha256=sha(__file__))
    report["evidence_subset_passed"] &= before == after
    report["final_evidence_eligible"] = report["evidence_subset_passed"] and identity["final_source_eligible"] and execution_binding["passed"]
    write(args.output / "ResultV1.json", report)
    write(args.output / "ManifestV1.json", {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                             for path in args.output.iterdir() if path.is_file()})
    print(json.dumps({name: report[name] for name in ("evidence_subset_passed", "final_evidence_eligible", "P1_qualified")}, indent=2))
    return 0 if report["evidence_subset_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
