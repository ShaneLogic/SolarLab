"""Current saved-state physics verification and explicit legacy consistency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import _canonical, read_json


def _same(actual, expected, label):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    actual, expected = json_data(actual), json_data(expected)
    if _canonical(actual) != _canonical(expected):
        raise ValueError("result identity mismatch: " + label)


def _sealed(record):
    body = {key: value for key, value in record.items() if key != "sha256"}
    if record.get("sha256") != hashlib.sha256(_canonical(body).encode()).hexdigest():
        raise ValueError("result canonical payload digest mismatch")


def _array(value, label):
    data = np.asarray(value)
    if data.dtype.kind not in "fiu" or not np.all(np.isfinite(data)):
        raise ValueError("nonfinite or nonnumeric result array: " + label)
    return data


def _legacy_contact_closure_scope(record, output):
    """Select the saved producer's definition, never whichever metric passes.

    Older R1 sources inherited the internal-face metric. The 392/815 sources
    include physical contacts; V3 restores separately named statistics. Source anchoring is
    performed by the outer evidence verifier; here its module bytes must also
    match the result's recorded source identity before choosing that meaning.
    """
    import ast
    import zipfile

    relative = "experiments/one_dimensional_mechanism_r1_dynamics.py"
    claimed = record.get("source", {}).get("files", {}).get(relative)
    current = Path(__file__).with_name("one_dimensional_mechanism_r1_dynamics.py").read_bytes()
    if claimed == hashlib.sha256(current).hexdigest():
        producer = current
    else:
        try:
            with zipfile.ZipFile(Path(output) / "SourceV1.zip") as archive:
                producer = archive.read("perovskite-sim/perovskite_sim/" + relative)
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError("legacy current metric requires its recorded producer source") from exc
        if hashlib.sha256(producer).hexdigest() != claimed:
            raise ValueError("legacy current metric producer differs from recorded source identity")

    known = {
        # Reviewed source blobs, not a value selected from the result labels.
        # 04170c4 inherited internal faces; 3928259 and 815c2fe included contacts.
        "94cafc0df08f19c0133d6df713c24b0a14e51fdbef5a577fd9596c0293d81992": False,
        "3dcd8110420868be25c9d6d3afc0d59589381c7e1d46152dc790e66673759062": True,
    }
    identity = hashlib.sha256(producer).hexdigest()
    if identity in known:
        return known[identity]
    if producer == current:
        for node in ast.parse(current).body:
            if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "CURRENT_METRIC_SEMANTICS" for target in node.targets)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value == "r1-v3-separate-internal-contact-interface"):
                return False
    raise ValueError("unsupported recorded R1 current-metric implementation")


def _saved_face_current_spread(row, *, include_contacts):
    physical = row["physical"].get("regular_right_limit") or row["physical"]
    currents = _array(physical["internal_maxwell_A_m2"], "internal Maxwell current")
    # The initial event remains the regular internal right limit. The R1
    # finite-step producer aggregates the same signed internal/contact totals;
    # a common polarity of +/-1 preserves this range and scale exactly.
    if include_contacts and row["dt_s"] > 0:
        currents = np.r_[currents, _array(physical["contact_maxwell_A_m2"], "contact Maxwell current")]
    return float(np.ptp(currents)) / max(float(np.max(np.abs(currents))), 1e-20)


def _npz_matches_json(path, record):
    """Compare every numerical sidecar to the corresponding JSON field."""
    if not path.exists():
        raise ValueError("step result lacks numerical sidecar")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"data.times_s", "data.voltage_V"}
        required.update("data.output_states." + key for key in record["output_states"])
        required.update("data.accepted_state_arrays." + key for key in record["accepted_state_arrays"])
        if not required <= set(arrays.files) or len(arrays.files) != len(set(arrays.files)):
            raise ValueError("step result numerical sidecar coverage mismatch")
        for name in arrays.files:
            parts = name.split(".")
            if parts[0] != "data":
                raise ValueError("unknown result sidecar field: " + name)
            value = record
            try:
                for part in parts[1:]:
                    value = value[int(part)] if isinstance(value, list) else value[part]
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raise ValueError("unknown result sidecar field: " + name) from exc
            actual, expected = arrays[name], _array(value, name)
            if actual.dtype.kind not in "fiu" or actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError("result JSON/NPZ mismatch: " + name)


def _counts(completion, rows):
    expected = {
        "accepted_record_count": len(rows), "observed_record_count": len(rows),
        "persisted_record_count": len(rows),
        "persisted_finite_step_count": sum(row.get("phase") == "accepted_regular_step" for row in rows),
        "physical_passed_finite_step_count": sum(row.get("phase") == "accepted_regular_step"
            and row.get("physical_checks_passed") is True for row in rows),
    }
    # A persistence failure can observe a row that was not written. It is a
    # failed record, not a passed trajectory. Never infer persisted rows from it.
    if completion["status"] == "failed":
        expected.pop("observed_record_count")
        observed = completion.get("observed_record_count")
        if type(observed) is not int or observed < len(rows):
            raise ValueError("invalid failed observed record count")
    for key, value in expected.items():
        if type(completion.get(key)) is not int or completion[key] != value:
            raise ValueError("result completion count mismatch: " + key)


def _system(output, prepared, controls, policy):
    """Rebuild algebraic context from the anchored fixture, without integrating."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import _checked_study_input
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
        _decode_dc, _make_system, build_r1_material, json_data,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml
    fixture = output / "SourceFixtureV1.yaml"
    if not fixture.exists():
        fixture = Path(__file__).resolve().parents[2] / json.loads(_checked_study_input())["fixture"]
    stack = load_device_from_yaml(fixture)
    _same(prepared["physical_stack"], json_data(stack), "prepared physical stack")
    grid, material = build_r1_material(stack, prepared["intervals"])
    system = _make_system(stack, grid, material, _decode_dc(prepared["dc_state"]),
                          prepared["fixed_reference"], controls, policy)
    return system, system.evaluate(system.initial_coordinate(), 0.0)


def _current_report(report, polarity, *, regular=False):
    electron = _array(report["contact_electron_current_A_m2"], "contact electron current")
    hole = _array(report["contact_hole_current_A_m2"], "contact hole current")
    ion = _array(report["contact_ion_current_A_m2"], "contact ion current")
    if electron.shape != (2,) or hole.shape != (2,) or ion.shape != (2,):
        raise ValueError("result contact current shape mismatch")
    _same(ion, np.zeros(2), "blocking contact ion current")
    _same(report["contact_conduction_A_m2"], electron + hole, "contact current decomposition")
    if "contact_maxwell_A_m2" in report:
        total = electron + hole + _array(report["contact_displacement_A_m2"], "contact displacement")
        _same(report["contact_maxwell_A_m2"], total, "Maxwell current decomposition")
        internal = _array(report["internal_maxwell_A_m2"], "internal Maxwell current")
        combined = np.r_[internal, total]
        if regular:
            _same(report["report_contact_current_A_m2"], polarity * total, "reported contact current polarity")
            interface = _array(report["interface_maxwell_A_m2"], "interface Maxwell current")
            combined = np.r_[internal, interface.ravel(), total]
            _same(report["face_current_spread_relative"], float(np.ptp(internal)) /
                  max(float(np.max(np.abs(internal))), 1e-20), "regular internal current spread")
            _same(report["interface_current_spread_relative"], float(np.max(
                np.abs(interface[:, 0] - interface[:, 1]) /
                np.maximum(np.max(np.abs(interface), axis=1), 1e-20))), "regular interface current spread")
        _same(report["contact_internal_current_spread_relative"], float(np.ptp(combined)) /
              max(float(np.max(np.abs(combined))), 1e-20), "physical current spread")
        # Match the source operation order so cancellation remains visible.
        # The 0+ physical record retains an explicitly inapplicable placeholder.
        error = float(report["charge_rate_A_m2"] - ((electron + hole)[0] - (electron + hole)[1]))
        _same(report["charge_balance_error_A_m2"], error, "physical charge balance error")


def _trap_check(row, previous, system, policy):
    from perovskite_sim.constants import Q
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import _TOLERANCE_FIELDS
    check, physical, dt = row["physical"]["trap_storage_check"], row["physical"], row["dt_s"]
    if previous is None:
        _same(check, {"applicable": False, "dt_s": 0.0, "certified": None,
            "reason": "no_positive_duration_time_step", "legacy_error_is_placeholder": True,
            "legacy_error_note": "0.0 is a compatibility placeholder; no finite-step error was computed"},
            "initial trap storage applicability")
        return
    error, storage_scale = physical["trap_storage_error_A_m2"], check["storage_scale_m2"]
    minimum_scale = policy.interface_storage_atol_m2 + policy.storage_relative_tolerance * max(
        abs(float(system.trap_density[0] * previous["state"]["occupancy"][0])),
        abs(float(system.trap_density[0] * system.reference_occupancy[0])))
    if error < 0 or storage_scale < minimum_scale:
        raise ValueError("result trap storage scale or error is invalid")
    before_floor = check["charge_scale_before_floor_A_m2"]
    contact = np.asarray(physical["contact_conduction_A_m2"])
    if before_floor < max(abs(physical["charge_rate_A_m2"]), abs(float(contact[0] - contact[1]))):
        raise ValueError("result trap storage charge scale is understated")
    charge_scale = max(before_floor, 1.0)
    nonlinear = Q * policy.maximum_scaled_nonlinear_residual * storage_scale
    conservation = dt * min(policy.maximum_charge_balance_relative_error, 1e-10) * charge_scale
    charge_limit = min(nonlinear, conservation)
    current_limit = charge_limit / dt
    expected = {
        "applicable": True, "dt_s": dt, "error_A_m2": error, "charge_error_C_m2": error * dt,
        "maximum_scaled_nonlinear_residual": policy.maximum_scaled_nonlinear_residual,
        "nonlinear_tolerances": {key: float(getattr(policy, key)) for key in _TOLERANCE_FIELDS},
        "nonlinear_charge_budget_C_m2": nonlinear, "conservation_charge_budget_C_m2": conservation,
        "charge_balance_scale_A_m2": charge_scale, "charge_scale_floor_A_m2": 1.0,
        "charge_scale_floor_active": before_floor < 1.0,
        "charge_balance_relative_limit": min(policy.maximum_charge_balance_relative_error, 1e-10),
        "charge_error_limit_C_m2": charge_limit, "current_error_limit_A_m2": current_limit,
        "newton_consistency_ratio": error / (nonlinear / dt),
        "local_charge_ratio": error / (conservation / dt), "normalized_error": error / current_limit,
        "normalized_limit": 1.0, "trap_dynamics_active": bool(system.controls.nu_t),
        "dominant_budget": "equal" if nonlinear == conservation else "newton" if nonlinear < conservation else "local_charge",
        "certified": error / current_limit <= 1.0, "failure_reason": None,
    }
    for key, value in expected.items():
        _same(check.get(key), value, "trap storage " + key)
    _same(physical["charge_balance_normalized"], abs(physical["charge_balance_error_A_m2"]) / charge_scale,
          "physical charge balance normalization")


def _scope(stage):
    result = {
        "stage": stage, "integration_replayed": False,
        "numerical_certificate_independently_approved": False,
        "checked": ["identity_and_sidecar_consistency", "controls_and_voltage", "initial_algebraic_event",
                    "accepted_row_charge_integrals", "saved_row_metric_aggregations", "physical_current_decomposition",
                    "population_inventory_and_site_occupancy", "trap_budget_arithmetic"],
        "range_only": ["analytic_jacobian_error", "charge_balance_relative", "interface_current_relative",
                       "eliminated_operator_error", "current_decomposition_error", "analytic_jacobian_nnz",
                       "maximum_absolute_charge_balance_error_A_m2", "maximum_absolute_poisson_residual_C_m2"],
        "provenance_only": ["solver_accepted", "scaled_nonlinear_residual_per_row",
                            "trap_storage_error_A_m2_per_row", "storage_scale_m2_per_row",
                            "charge_scale_before_floor_A_m2_per_row", "frozen_ion_flux_per_row"],
        "limitation": "Saved row diagnostics are internally checked and aggregated; original solver coordinates, "
                      "Jacobian checks and incremental storage cannot be independently recovered without replay.",
    }
    if stage != "step":
        result.update(checked=(["per_control_zero_equations", "zero_initial_algebraic_events", "population_identity"]
                               if stage == "zero-check" else ["failure_record_and_persistence_counts"]
                               if stage == "failed" else []), range_only=[], provenance_only=[])
    return result


def _step(record, rows, protocol, completion, prepared, output):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import _physical_step_checks, r1_policy

    if record.get("schema") != "R1ControlledStepV1":
        raise ValueError("invalid controlled step result schema")
    _same(record.get("accepted_steps"), rows, "accepted step records")
    for key in ("intervals", "amplitude_V", "times_s", "policy"):
        _same(record.get(key), protocol.get(key), key)
    _same(record.get("control_label"), protocol.get("control"), "control")
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    controls = R1DynamicsControls.from_label(protocol["control"])
    _same(record.get("controls"), {"nu_I": controls.nu_I, "nu_t": controls.nu_t}, "control flags")
    time_substeps = protocol["policy"]["refinement_substeps"]
    policy = r1_policy(nonlinear_factor=protocol["nonlinear_factor"], time_substeps=time_substeps)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    _same(record["policy"], json_data(policy), "pinned numerical policy")
    times = _array(record["times_s"], "times_s")
    if times.ndim != 1 or times.size < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("invalid result observation times")
    _same(record.get("voltage_V"), np.full(times.size, record["amplitude_V"]), "voltage waveform")
    system, before = _system(output, prepared, controls, policy)
    _same(record.get("junction_polarity"), system.polarity, "junction polarity")
    initial = build_initial_step(system, before, record["amplitude_V"], policy=policy)
    _same(record.get("initial_event"), initial.event, "initial algebraic event")
    system = initial.system
    if not rows:
        raise ValueError("passed step has no accepted records")
    groups = {level: [] for level in policy.refinement_substeps}
    physical = []
    integrated = {level: 0.0 for level in groups}
    for row in rows:
        level = row.get("substeps")
        if type(level) is not int or level not in groups:
            raise ValueError("result row has undeclared time refinement")
        group = groups[level]
        finite = row.get("phase") == "accepted_regular_step"
        if (not group and (row.get("phase") != "0+" or row.get("time_s") != 0 or row.get("dt_s") != 0)
                or group and (not finite or row["time_s"] <= group[-1]["time_s"]
                or row["dt_s"] <= 0
                or not np.isclose(row["time_s"] - group[-1]["time_s"], row["dt_s"], rtol=1e-12, atol=0))):
            raise ValueError("result accepted time sequence mismatch")
        _same(row.get("solver_accepted"), finite, "accepted row phase")
        _trap_check(row, group[-1] if group else None, system, policy)
        _current_report(row["physical"], system.polarity)
        if not finite:
            _same(row["physical"]["regular_right_limit"], initial.event["regular_current"], "initial regular current")
            _same(row["physical"]["initial_algebraic_certificate"], initial.event["algebraic_certificate"], "initial algebraic certificate")
        else:
            integrated[level] += float(system.polarity * row["physical"]["contact_maxwell_A_m2"][0] * row["dt_s"])
        _same(row.get("regular_integrated_charge_C_m2"), integrated[level], "accepted integrated charge")
        context = {**row, "initial_event": record["initial_event"]}
        checked = _physical_step_checks(row["physical"], finite_step=finite, policy=policy, evidence=context)
        _same(row.get("physical_checks"), checked, "physical row check")
        if checked["passed"] is not True or row.get("physical_checks_passed") is not True:
            raise ValueError("passed result contains a failed physical row")
        if finite:
            physical.append(row["physical"])
        group.append(row)
    for level, group in groups.items():
        if not group or group[-1]["time_s"] != times[-1]:
            raise ValueError("result time refinement lacks the complete window")
        expected_times = [0.0]
        expected_steps = [0.0]
        for start, end in zip(times[:-1], times[1:]):
            dt = float(end - start) / level
            expected_times.extend(float(start + (step + 1) * dt) for step in range(level))
            expected_steps.extend([dt] * level)
        _same([row["time_s"] for row in group], expected_times, "declared substep times")
        _same([row["dt_s"] for row in group], expected_steps, "declared substep durations")
    impulse = initial.event["impulse_charge_C_m2"]
    _same(record["charge_integral"]["impulse_charge_C_m2"], impulse, "charge impulse")
    _same(record["charge_integral"]["regular_by_substeps_C_m2"], {str(k): v for k, v in integrated.items()}, "regular integrated charge")
    _same(record["charge_integral"]["complete_by_substeps_C_m2"], {str(k): impulse + v for k, v in integrated.items()}, "complete integrated charge")
    finest = groups[policy.refinement_substeps[-1]]
    sampled = []
    for time in times:
        matches = [row for row in finest if row["time_s"] == time]
        if len(matches) != 1:
            raise ValueError("result lacks a unique accepted observation time")
        sampled.append(matches[0])
    for key, actual in record["output_states"].items():
        _same(actual, [row["state"][key] for row in sampled], "sampled state " + key)
    for key, actual in record["accepted_state_arrays"].items():
        _same(actual, [row["state"][key] for row in rows], "accepted array " + key)
    if set(record["accepted_state_arrays"]) != set(rows[0]["state"]):
        raise ValueError("result accepted state coverage mismatch")
    expected_state_keys = {"n_m3", "p_m3", "positive_m3", "occupancy", "phi_V", "sheet_charge_C_m2"}
    if set(record["output_states"]) != expected_state_keys:
        raise ValueError("result output state coverage mismatch")
    regular = record["regular_currents"]
    if len(regular) != times.size:
        raise ValueError("result regular current observation count mismatch")
    for report, row in zip(regular, sampled):
        _current_report(report, system.polarity, regular=True)
        for key in ("contact_electron_current_A_m2", "contact_hole_current_A_m2", "contact_conduction_A_m2"):
            _same(report[key], row["physical"][key], "regular/accepted conduction " + key)
    _same(regular[0], initial.event["regular_current"], "regular initial current")
    internal = [initial.event["regular_current"]["internal_maxwell_A_m2"]] + [row["physical"]["internal_maxwell_A_m2"] for row in sampled[1:]]
    _same(record["finite_step_averages"]["internal_total_A_m2"], system.polarity * np.asarray(internal), "finite-step current averages")
    certificate = record["certificate"]
    metrics, limits = certificate["metrics"], certificate["limits"]
    expected_limits = {
        "nonlinear_residual": policy.maximum_scaled_nonlinear_residual,
        "local_carrier_residual": policy.maximum_local_carrier_normalized_residual,
        "local_gauss_residual": policy.maximum_local_gauss_normalized_residual,
        "analytic_jacobian_error": policy.maximum_jacobian_column_relative_error,
        "charge_balance_relative": policy.maximum_charge_balance_relative_error,
        "all_face_current_relative": policy.maximum_all_face_current_spread_relative,
        "interface_current_relative": policy.maximum_two_sided_interface_total_current_relative_error,
        "eliminated_operator_error": policy.maximum_eliminated_operator_relative_error,
        "inventory_relative_drift": min(policy.maximum_ion_inventory_relative_drift, 1e-10),
        "current_decomposition_error": policy.maximum_current_decomposition_relative_error,
        "refinement_state_change": policy.maximum_refinement_state_change,
        "refinement_current_change": policy.maximum_refinement_current_relative_change,
        "full_gauss_normalized": 1e-10, "full_charge_balance_normalized": 1e-10,
        "physical_current_spread_relative": 2e-6, "trap_storage_normalized_error": 1.0,
    }
    _same(limits, expected_limits, "certificate limits")
    if set(metrics) != set(limits):
        raise ValueError("result certificate metric coverage mismatch")
    for name, limit in expected_limits.items():
        value = metrics[name]
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0 or value > limit:
            raise ValueError("result certificate exceeds pinned limit: " + name)
    for name, physical_key in (("full_gauss_normalized", "gauss_normalized"),
            ("full_charge_balance_normalized", "charge_balance_normalized"),
            ("physical_current_spread_relative", "contact_internal_current_spread_relative")):
        _same(metrics[name], max(row[physical_key] for row in physical), "certificate metric " + name)
    _same(metrics["nonlinear_residual"], max(row["scaled_nonlinear_residual"] for row in finest), "certificate nonlinear residual")
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import _trap_storage_summary
    trap = _trap_storage_summary(physical)
    _same(certificate.get("trap_storage"), trap, "trap storage summary")
    _same(metrics["trap_storage_normalized_error"], trap["maximum_normalized_error"], "certificate trap storage metric")
    _same(certificate.get("maximum_trap_storage_error_A_m2"), trap["maximum_error_A_m2"], "maximum trap storage error")
    carrier = gauss = 0.0
    from dataclasses import replace
    from perovskite_sim.constants import Q
    from perovskite_sim.experiments.interface_defect_transient import (
        _material_two_sided_interface_problem, electrostatic_trace_residual_and_jacobian,
    )
    for row in finest:
        residual = np.asarray(row["state"]["local_residual"]).reshape(-1, 6)
        carrier = max(carrier, float(np.max(np.abs(residual[:, 2:]) / system.reference_local_scale[:, 2:])))
        # The stored local residual uses increments; the trace certificate uses
        # the absolute electrostatic residual carried by each local state.
        state = row["state"]
        for index in range(system.interface_count):
            geometry, _, bulk = _material_two_sided_interface_problem(
                system.material, system.stack, np.asarray(state["n_m3"]), np.asarray(state["p_m3"]),
                np.asarray(state["phi_V"]), index, cross_transmission=system.dark_reference.interface_transmission)
            sheet = -Q * system.trap_density[index] * (state["occupancy"][index] - system.equilibrium_occupancy[index])
            charged = replace(geometry, fixed_sheet_charge_C_m2=float(geometry.fixed_sheet_charge_C_m2) + sheet)
            electrostatic, _ = electrostatic_trace_residual_and_jacobian(
                np.asarray(state["trace_potential_V"][index]), charged, bulk)
            gauss = max(gauss, abs(float(electrostatic[1])) / system.reference_local_scale[index, 1])
    _same(metrics["local_carrier_residual"], carrier, "certificate local carrier residual")
    _same(metrics["local_gauss_residual"], gauss, "certificate local Gauss residual")
    include_contacts = _legacy_contact_closure_scope(record, output)
    face_spreads = [_saved_face_current_spread(row, include_contacts=include_contacts) for row in finest]
    _same(metrics["all_face_current_relative"], max(face_spreads), "certificate face current spread")
    sampled_groups = [[next(row for row in group if row["time_s"] == time) for time in times]
                      for group in groups.values()]
    output_rows = [row for group in sampled_groups for row in group]
    inventories = np.asarray([row["state"]["positive_inventory_m2"] for row in output_rows])
    _same(metrics["inventory_relative_drift"], float(np.max(np.abs(inventories / system.positive_targets - 1.0))), "certificate inventory drift")
    for row in rows:
        state = row["state"]
        from perovskite_sim.physics.physical_control_volume import physical_contact_displacement
        positive = _array(state["positive_m3"], "positive population")
        inventory = [float(positive[np.asarray(nodes)] @ system.widths[np.asarray(nodes)]) for nodes in system.ion_layout.positive_components]
        _same(state["positive_inventory_m2"], inventory, "state component inventory")
        _same(row["physical"]["inventory_m2"], float(positive[system.positive_nodes] @ system.widths[system.positive_nodes]), "physical ion inventory")
        _same(row["physical"]["inventory_relative_drift"], abs(row["physical"]["inventory_m2"] / 1e15 - 1), "physical inventory drift")
        if np.any(_array(state["n_m3"], "electron density") <= 0) or np.any(_array(state["p_m3"], "hole density") <= 0):
            raise ValueError("result contains nonpositive carrier density")
        if np.any(positive[system.positive_nodes] <= 0) or np.any(np.asarray(state["occupancy"]) <= 0) or np.any(np.asarray(state["occupancy"]) >= 1):
            raise ValueError("result contains invalid ion/trap population")
        if system._site_fraction(positive, None, reject=False) >= policy.site_occupancy_ceiling:
            raise ValueError("result accepted state exceeds site occupancy ceiling")
        rho, _ = system.system._bulk_space_charge_and_tangent(np.asarray(state["n_m3"]),
            np.asarray(state["p_m3"]), positive_ion_density_m3=positive, negative_ion_density_m3=None)
        displacement = physical_contact_displacement(-system.material.poisson_factor.C * np.diff(state["phi_V"]), rho, system.widths)
        charge = float(np.dot(rho, system.widths) + np.sum(state["sheet_charge_C_m2"]))
        _same(row["physical"]["rho_C_m3"], rho, "physical charge density")
        _same(row["physical"]["physical_displacement_C_m2"], displacement, "physical electric displacement")
        _same(row["physical"]["full_charge_C_m2"], charge, "physical full charge")
        gauss_error = float(displacement[1] - displacement[0] - charge)
        _same(row["physical"]["gauss_error_C_m2"], gauss_error, "physical Gauss error")
        _same(row["physical"]["gauss_normalized"], abs(gauss_error) / (Q * 1e15), "physical Gauss normalization")
    site = max(system._site_fraction(np.asarray(row["state"]["positive_m3"]), None, reject=False) for row in output_rows)
    _same(certificate.get("site_occupancy_fraction"), site, "site occupancy fraction")
    if site >= policy.site_occupancy_ceiling:
        raise ValueError("result site occupancy exceeds physical ceiling")
    coarse, fine = sampled_groups[-2:]
    state_change = 0.0
    for key in ("n_m3", "p_m3", "trace_state_m3"):
        a, b = [np.asarray([row["state"][key] for row in group]) for group in (coarse, fine)]
        state_change = max(state_change, float(np.max(np.abs(np.log(a / b)))))
    a, b = [np.asarray([row["state"]["positive_m3"] for row in group])[:, system.positive_nodes] for group in (coarse, fine)]
    state_change = max(state_change, float(np.max(np.abs(np.log(a / b)))))
    a, b = [np.asarray([row["state"]["occupancy"] for row in group]) for group in (coarse, fine)]
    state_change = max(state_change, float(np.max(np.abs(a - b))))
    a, b = [np.asarray([row["state"]["phi_V"] for row in group]) for group in (coarse, fine)]
    state_change = max(state_change, float(np.max(np.abs(a - b))) / max(float(np.ptp(b)), 0.025))
    _same(metrics["refinement_state_change"], state_change, "state refinement metric")
    a, b = [np.asarray([row["physical"]["internal_maxwell_A_m2"] for row in group[1:]]) for group in (coarse, fine)]
    _same(metrics["refinement_current_change"], float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1.0), "current refinement metric")
    frozen = {}
    if not controls.nu_I:
        frozen["positive_population"] = all(np.array_equal(row["state"]["positive_m3"], prepared["state"]["positive_m3"]) for row in rows)
        frozen["ion_flux"] = True  # The selected operator disables ion flux; raw per-face ion flux is not saved.
    if not controls.nu_t:
        frozen["trap_population"] = all(np.array_equal(row["state"]["occupancy"], prepared["state"]["occupancy"]) for row in rows)
        frozen["trap_charge"] = all(np.array_equal(row["state"]["sheet_charge_C_m2"], prepared["state"]["sheet_charge_C_m2"]) for row in rows)
        frozen["capture_flux"] = all(np.count_nonzero(row["state"]["capture_m2_s"]) == 0 for row in rows)
    _same(certificate["exact_freeze_checks"], frozen, "frozen population checks")
    if any(value is not True for value in frozen.values()):
        raise ValueError("passed result contains a failed frozen-population check")
    _same(certificate["dense_entries"], system.dimension**2, "Jacobian dimension")
    if (type(certificate["analytic_jacobian_nnz"]) is not int
            or not 0 < certificate["analytic_jacobian_nnz"] < certificate["dense_entries"]):
        raise ValueError("passed result lacks a sparse Jacobian")
    for name in ("maximum_absolute_charge_balance_error_A_m2", "maximum_absolute_poisson_residual_C_m2"):
        if type(certificate.get(name)) not in (int, float) or certificate[name] < 0:
            raise ValueError("invalid nonnegative result diagnostic: " + name)
    saved_poisson = max(float(np.max(np.abs(row["state"]["poisson_residual_C_m2"]))) for row in finest)
    if certificate["maximum_absolute_poisson_residual_C_m2"] < saved_poisson:
        raise ValueError("result Poisson diagnostic understates saved residual")
    _counts(completion, rows)


def _verify_legacy_result_records(output: Path, completion: dict) -> dict:
    """Validate saved claims and duplicate representations, without new solves."""
    output = Path(output)
    if completion["status"] == "failed":
        _same(read_json(output / "FailureV1.json"), completion["failure"], "failure record")
        if completion["stage"] == "step" or (output / "AcceptedStepsV1.json").exists():
            _counts(completion, read_json(output / "AcceptedStepsV1.json")
                    if (output / "AcceptedStepsV1.json").exists() else [])
        return _scope("failed")
    if completion["stage"] == "prepare":
        return _scope("prepare")
    prepared = read_json(output / "PreparedStateV1.json")
    protocol = read_json(output / "ProtocolV1.json")
    name = "StepResultV1.json" if completion["stage"] == "step" else "ZeroExcitationV1.json"
    record = read_json(output / name)
    _sealed(record)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import nonfinite_numeric_paths
    if nonfinite_numeric_paths(record):
        raise ValueError("nonfinite result evidence")
    _same(record.get("prepared_sha256"), prepared["sha256"], "preparation")
    _same(record.get("source"), prepared["source"], "source")
    _same(record.get("reference_sha256"), prepared["reference_sha256"], "reference")
    if record.get("certificate", {}).get("certified") is not True or record["certificate"].get("reasons") != []:
        raise ValueError("passed result lacks a passed certificate")
    if completion["stage"] == "step":
        _step(record, read_json(output / "AcceptedStepsV1.json"), protocol, completion, prepared, output)
        _npz_matches_json(output / "StepResultV1.npz", record)
    else:
        if record.get("schema") != "R1ZeroExcitationV1" or set(record["controls"]) != set(protocol["control"]):
            raise ValueError("zero-check control coverage mismatch")
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import equilibrium_checks, snapshot
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
        policy = r1_policy(protocol["nonlinear_factor"], time_substeps=protocol["policy"]["refinement_substeps"])
        _same(protocol["policy"], policy, "zero-check pinned policy")
        for label, value in record["controls"].items():
            controls = R1DynamicsControls.from_label(label)
            _same(value.get("controls"), controls, "zero-check controls " + label)
            system, state = _system(output, prepared, controls, policy)
            _same(value.get("population_identity"), snapshot(system, state), "zero-check population " + label)
            _same(value.get("remaining_equations"), equilibrium_checks(system, state, policy), "zero-check equations " + label)
            _same(value.get("initial_event"), build_initial_step(system, state, 0.0, policy=policy).event,
                  "zero-check initial event " + label)
            if value["remaining_equations"].get("certified") is not True:
                raise ValueError("zero-check contains failed equilibrium equations")
    return _scope("step" if completion["stage"] == "step" else "zero-check")


def _tagged_numeric(value):
    """Recover a lossless failed-array sidecar value from strict tagged JSON."""
    if isinstance(value, dict):
        if set(value) == {"nonfinite"} and value["nonfinite"] in ("nan", "inf", "-inf"):
            return float(value["nonfinite"])
        if set(value) == {"real", "imag"}:
            return complex(_tagged_numeric(value["real"]), _tagged_numeric(value["imag"]))
        raise ValueError("failed result contains invalid numeric tags")
    if isinstance(value, list):
        return [_tagged_numeric(item) for item in value]
    return value


def _failed_sidecar(path, record):
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)):
            raise ValueError("duplicate failed result sidecar arrays")
        for name in arrays.files:
            parts = name.split(".")
            if parts[0] != "data":
                raise ValueError("unknown failed result sidecar field")
            value = record
            try:
                for part in parts[1:]:
                    value = value[int(part)] if isinstance(value, list) else value[part]
                expected = np.asarray(_tagged_numeric(value))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError("unknown failed result sidecar field: " + name) from exc
            actual = arrays[name]
            if (actual.dtype.kind not in "fiuc" or expected.dtype.kind not in "fiuc"
                    or actual.shape != expected.shape
                    or not np.array_equal(actual, expected, equal_nan=True)):
                raise ValueError("failed result JSON/NPZ mismatch: " + name)



def _failed_preparation_physics(output, raw):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import verify_prepared_metadata
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
        _decode_dc, _make_system, _preparation_policy, _recomputed_dc_certificate,
        build_r1_material, equilibrium_checks, snapshot, json_data,
    )
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
    from perovskite_sim.models.config_loader import load_device_from_yaml

    verify_prepared_metadata(raw)
    stack = load_device_from_yaml(output / "SourceFixtureV1.yaml")
    binding = read_json(output / "ReferenceBindingV1.json")
    _same(raw.get("physical_stack"), json_data(stack), "failed preparation material stack")
    _same(raw.get("fixed_reference"), binding, "failed preparation fixed reference")
    _same(raw.get("intervals"), read_json(output / "ProtocolV1.json")["intervals"], "failed preparation intervals")
    policy = _preparation_policy(raw.get("preparation_policy"))
    try:
        grid, material = build_r1_material(stack, raw["intervals"])
        dc = _decode_dc(raw["dc_state"])
        system = _make_system(stack, grid, material, dc, binding, R1DynamicsControls(), policy)
        state = system.evaluate(system.initial_coordinate(), 0.0)
        _same(raw.get("state"), snapshot(system, state), "failed preparation saved physical arrays")
        checks = equilibrium_checks(system, state, policy)
        _same(raw.get("preparation_checks"), checks, "failed preparation reported equations")
        certificate = _recomputed_dc_certificate(system, dc)
        for key, value in certificate.items():
            _same(raw["dc_state"]["certificate"].get(key), value, "failed preparation DC metric " + key)
    except (TypeError, KeyError, FloatingPointError, RuntimeError) as exc:
        raise ValueError("failed preparation physical reevaluation unavailable: " + str(exc)) from exc


def _failed_zero_physics(output, record, prepared):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import verify_zero_metadata
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import equilibrium_checks, snapshot
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step

    verify_zero_metadata(record, same=_same, failed=True)
    protocol = read_json(output / "ProtocolV1.json")
    policy = r1_policy(protocol["nonlinear_factor"], time_substeps=protocol["time_substeps"])
    _same(record.get("prepared_sha256"), prepared["sha256"], "failed zero preparation")
    _same(record.get("reference_sha256"), prepared["reference_sha256"], "failed zero reference")
    if not isinstance(record.get("controls"), dict) or not set(record["controls"]) <= set(protocol["control"]):
        raise ValueError("failed zero-control coverage mismatch")
    for label, value in record["controls"].items():
        controls = R1DynamicsControls.from_label(label)
        system, state = _system(output, prepared, controls, policy)
        _same(value.get("controls"), controls, "failed zero control " + label)
        _same(value.get("population_identity"), snapshot(system, state), "failed zero population " + label)
        _same(value.get("remaining_equations"), equilibrium_checks(system, state, policy), "failed zero equations " + label)
        _same(value.get("initial_event"), build_initial_step(system, state, 0.0, policy=policy).event,
              "failed zero initial event " + label)

def failure_scope_report(record, reconstruction, *, persisted_rows=None, failure=None):
    """Describe exactly the saved prefix checked, not the unwitnessed execution.

    Call only after the row schedule and equations have been validated. A
    shorter, internally consistent prefix does not establish why the original
    computation stopped. It never promotes a failed experiment to acceptance.
    """
    rows = record.get("accepted_steps", [])
    checked = reconstruction.get("checked_row_count", len(rows))
    expected = reconstruction.get("expected_row_count")
    violations = reconstruction.get("physical_limit_violations", [])
    witnesses = [index for index, row in enumerate(rows)
                 if row.get("physical_checks_passed") is False]
    def endpoint(row):
        return {key: row.get(key) for key in ("substeps", "time_s", "dt_s", "phase")}
    return {
        "schema": "R1FailureScopeV1",
        "scientifically_accepted": False,
        "saved_row_count": len(rows), "checked_row_count": checked,
        "expected_row_count": expected,
        "persisted_row_count": len(persisted_rows) if persisted_rows is not None else None,
        "first_saved_row": endpoint(rows[0]) if rows else None,
        "last_saved_row": endpoint(rows[-1]) if rows else None,
        "saved_schedule_checked": bool(checked == len(rows) and rows),
        "complete_requested_schedule": bool(expected is not None and checked == expected),
        "saved_physical_failure_witness_rows": witnesses,
        "recomputed_violation_count": len(violations),
        "saved_physical_failure_demonstrated": bool(witnesses or violations),
        "failure_origin_status": ("saved_physical_violations_reproduced"
                                  if witnesses or violations else "not_reconstructed_from_saved_prefix"),
        "original_execution_extent_verified": False,
        "recorded_failure": failure if failure is not None else record.get("failure"),
        "provenance_only": ["failure_type_and_message", "unpersisted_execution", "I/O_history"],
        "scope": "present_saved_prefix_only; absence_of_a_saved_violation_does_not_refute_the_recorded_failure",
    }


def _verify_physical_result_records(output, completion):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import verify_prepared_physics
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import (
        verify_prepared_metadata, verify_zero_metadata,
    )

    failed, stage = completion["status"] == "failed", completion["stage"]
    base_scope = {
        "stage": stage, "integration_replayed": False,
        "numerical_certificate_independently_approved": False,
        "range_only": [], "provenance_only": ["execution_environment", "timestamps", "solver_iteration_history"],
        "scientifically_accepted": False,
        "scope": "single_recorded_setting_not_full_R1_2_or_mechanism_acceptance",
    }
    if failed:
        _same(read_json(output / "FailureV1.json"), completion["failure"], "failure record")
    elif any((output / name).exists() for name in ("FailureV1.json", "FailedResultV1.json")):
        raise ValueError("passed result contains conflicting failure evidence")
    if stage != "step":
        if failed:
            checked = []
            prepared = None
            if (output / "PreparedStateV1.json").exists():
                prepared = read_json(output / "PreparedStateV1.json")
                verify_prepared_metadata(prepared)
                _failed_sidecar(output / "PreparedStateV1.npz", prepared)
                verify_prepared_physics(prepared, load_device_from_yaml(output / "SourceFixtureV1.yaml"),
                                        read_json(output / "ReferenceBindingV1.json"))
                checked.append("saved_preparation_equations")
            payload_path = output / "FailedResultV1.json"
            if payload_path.exists():
                payload = read_json(payload_path)
                _failed_sidecar(payload_path.with_suffix(".npz"), payload)
                duplicate = output / "ZeroExcitationV1.json"
                if duplicate.exists():
                    _same(read_json(duplicate), payload, "failed zero-excitation duplicate")
                    _failed_sidecar(duplicate.with_suffix(".npz"), payload)
                if payload.get("certificate", {}).get("certified") is True or payload.get("certified") is True:
                    raise ValueError("failed scientific payload cannot assert a passed certificate")
                schema = payload.get("schema")
                if schema == "R1FailedPreparationV1" and "raw_preparation" in payload:
                    _failed_preparation_physics(output, payload["raw_preparation"])
                    checked.append("failed_preparation_state_and_reported_equations")
                elif schema == "R1ZeroExcitationV1" and prepared is not None:
                    _failed_zero_physics(output, payload, prepared)
                    checked.append("saved_partial_zero_excitation_equations")
                else:
                    raise ValueError("present failed scientific payload has no reconstructable physical schema")
            return {**base_scope, "content_matches_recomputed": True if checked else None,
                    "physical_limits_satisfied": False, "checked": checked,
                    "failure_origin_reconstructed": False,
                    "reconstruction_scope": "present_preparation_or_zero_equations_only; failure_history_not_replayed",
                    "unavailable": "failed experiment remains ineligible for scientific acceptance"}
        prepared = read_json(output / "PreparedStateV1.json")
        verify_prepared_metadata(prepared)
        _failed_sidecar(output / "PreparedStateV1.npz", prepared)
        verify_prepared_physics(prepared, load_device_from_yaml(output / "SourceFixtureV1.yaml"),
                                read_json(output / "ReferenceBindingV1.json"))
        if stage == "zero-check":
            verify_zero_metadata(read_json(output / "ZeroExcitationV1.json"), same=_same)
            _verify_legacy_result_records(output, completion)
            _failed_sidecar(output / "ZeroExcitationV1.npz", read_json(output / "ZeroExcitationV1.json"))
        return {**base_scope, "scientifically_accepted": True,
                "content_matches_recomputed": True, "physical_limits_satisfied": True,
                "reconstruction_scope": "saved_preparation_and_zero_equations; declared_historical_metadata_not_replayed",
                "provenance_only": base_scope["provenance_only"] + [
                    "prepared.environment", "prepared.created_utc", "prepared.dc_state.certificate.optimizer_success",
                    "prepared.dc_state.certificate.optimizer_nfev"],
                "checked": ["saved_preparation_equations"] + (["zero_excitation_equations"] if stage == "zero-check" else [])}
    rows_path = output / "AcceptedStepsV1.json"
    rows = read_json(rows_path) if rows_path.exists() else []
    _failed_sidecar(rows_path.with_suffix(".npz"), rows)
    _counts(completion, rows)
    result_path = output / ("FailedResultV1.json" if failed else "StepResultV1.json")
    if not result_path.exists():
        if failed and not rows:
            return {**base_scope, "content_matches_recomputed": None,
                    "physical_limits_satisfied": False, "checked_row_count": 0,
                    "unavailable": "failure before a reconstructable trajectory was saved"}
        raise ValueError("physical result payload unavailable for saved accepted rows")
    record = read_json(result_path)
    if failed and (output / "StepResultV1.json").exists():
        _same(read_json(output / "StepResultV1.json"), record, "failed step duplicate")
        _failed_sidecar(output / "StepResultV1.npz", record)
    if record.get("schema") != "R1ControlledStepV1":
        if failed and not rows:
            return {**base_scope, "content_matches_recomputed": None,
                    "physical_limits_satisfied": False, "checked_row_count": 0,
                    "unavailable": "failed payload contains no controlled trajectory"}
        raise ValueError("physical result schema mismatch")
    record_rows = record.get("accepted_steps")
    if not isinstance(record_rows, list):
        raise ValueError("physical result lacks accepted-state records")
    if failed and len(record_rows) != len(rows):
        # A failed durable write can leave one extra observed row in the raw
        # failure payload. Bind the durable prefix, then verify every raw row.
        if len(record_rows) != completion.get("observed_record_count") or len(record_rows) < len(rows):
            raise ValueError("failed result observed-state coverage mismatch")
        _same(record_rows[:len(rows)], rows, "failed persisted accepted prefix")
    else:
        _same(record_rows, rows, "accepted step records")
    prepared = read_json(output / "PreparedStateV1.json")
    verify_prepared_metadata(prepared)
    protocol = read_json(output / "ProtocolV1.json")
    for name in ("intervals", "amplitude_V", "times_s", "policy"):
        _same(record.get(name), protocol.get(name), "executed " + name)
    _same(record.get("control_label"), protocol.get("control"), "executed control")
    _same(record.get("prepared_sha256"), prepared["sha256"], "preparation identity")
    if not failed:
        _sealed(record)
        _same(record.get("source"), prepared["source"], "source identity")
        _npz_matches_json(result_path.with_suffix(".npz"), record)
    else:
        if "source" in record:
            _same(record["source"], prepared["source"], "failed source identity")
        if "sha256" in record:
            _sealed(record)
        _failed_sidecar(result_path.with_suffix(".npz"), record)
        if record.get("certificate", {}).get("certified") is not False:
            raise ValueError("failed result cannot assert a passed certificate")
        if not rows and not record.get("physics_reconstruction"):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import RESULT_ARRAY_FIELDS
            if any(key in record for key in RESULT_ARRAY_FIELDS):
                raise ValueError("failed result arrays have no reconstructable accepted states")
            return {**base_scope, "content_matches_recomputed": None,
                    "physical_limits_satisfied": False, "checked_row_count": 0,
                    "unavailable_result_fields": list(RESULT_ARRAY_FIELDS),
                    "unavailable": "failure before exact initial-state reconstruction; only outer identities checked"}
    physics = verify_r1_step_physics(
        load_device_from_yaml(output / "SourceFixtureV1.yaml"), protocol["intervals"],
        read_json(output / "ReferenceBindingV1.json"), prepared, record,
        expected_prepared_sha256=prepared["sha256"], allow_incomplete=failed,
    )
    if not failed and not physics["certified"]:
        raise ValueError("passed result fails reconstructed physical gates")
    return {**base_scope, **physics, "scientifically_accepted": not failed and physics["certified"],
            "failure_scope": (failure_scope_report(record, physics, persisted_rows=rows,
                                                   failure=completion.get("failure")) if failed else None),
            "scope": base_scope["scope"], "reconstruction_scope": physics["scope"],
            "provenance_only": sorted(set(base_scope["provenance_only"]) | set(physics["provenance_only"])),
            "checked": ["state_to_transport_and_current", "charge_and_storage_equations",
                        "finite_difference_jacobian", "eliminated_operator_and_scales",
                        "original_physical_certificate_metrics"],
            "numerical_certificate_independently_approved": False,
            "integration_replayed": False, "range_only": ["historical_iteration_maximum_jacobian_nnz"]}


def verify_result_records(output: Path, completion: dict) -> dict:
    """Revision six checks saved physics; legacy formats retain explicit scope."""
    output = Path(output)
    if completion.get("evidence_revision", 5) == 6:
        return _verify_physical_result_records(output, completion)
    scope = _verify_legacy_result_records(output, completion)
    return {**scope, "scientifically_accepted": False, "mode": "historical_inspection"}
