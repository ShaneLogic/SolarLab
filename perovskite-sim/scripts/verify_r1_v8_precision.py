"""Independent V8 two-sided and initialization arithmetic checks.

The V7 same-discrete-coefficient Decimal oracle is reused unchanged.  This
module never imports a production numerical kernel.  Solved eliminated
quantities and shared physical inputs are distinguished explicitly; missing
upstream initialization inputs remain missing evidence, never reconstructed
backwards from the final populations.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json

from scripts.verify_r1_v7_precision import dec, evaluate_state, verify_side


VOLTAGE_LIMIT = Decimal("1e-26")
POPULATION_LIMIT = Decimal("1e-27")
SOLVE_CORRECTION_LIMIT = Decimal("1e-28")
NEEDED_SIDE_FIELDS = ("phi_V", "n_m3", "p_m3", "positive_m3", "sheet_charge_C_m2",
                      "positive_flux_m2_s", "positive_rate_m3_s", "boundary_flux_m2_s")
REFERENCE_FIELDS = ("phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3",
                    "occupancy", "trace_potential_V", "trace_state_m3")
SHARED_FIELDS = ("dqfn_V", "dqfp_V", "positive_m3", "occupancy", "sheet_charge_C_m2")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_identity(state):
    fields = precision_fields(state)
    return digest({name: fields[name] for name in REFERENCE_FIELDS})


def flatten(value):
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in flatten(part)]
    return [dec(value)]


def represented(value):
    if isinstance(value, dict):
        if set(value) != {"hi", "lo"}:
            raise ValueError("represented values require exactly hi and lo")
        high, low = flatten(value["hi"]), flatten(value["lo"])
        if len(high) != len(low):
            raise ValueError("mismatched high and low sizes")
        return [a + b for a, b in zip(high, low)]
    return flatten(value)


def precision_fields(state):
    if "fields" in state:
        return state["fields"]
    if "precision" in state:
        return state["precision"]["fields"]
    result = {}
    for name in state:
        if name.startswith("precision_") and name.endswith("_hi"):
            field = name[len("precision_"):-len("_hi")]
            low = "precision_" + field + "_lo"
            if low not in state:
                raise ValueError("missing low part for " + field)
            result[field] = {"hi": state[name], "lo": state[low]}
    if not result:
        raise ValueError("a recorded fine state is required")
    return result


def as_snapshot(fields):
    return {"precision": {"representation": "float64-pair-v1", "fields": fields}}


def observed(fields):
    return {name: fields[name] for name in ("positive_flux_m2_s", "positive_rate_m3_s", "boundary_flux_m2_s")
            if name in fields}


def arithmetic_comparison(actual, expected, *, voltage=False):
    with localcontext() as context:
        context.prec = 100
        a, b = represented(actual), represented(expected)
        if len(a) != len(b):
            raise ValueError("arithmetic comparison size mismatch")
        differences = [abs(x - y) for x, y in zip(a, b)]
        if voltage:
            scores, limit = differences, VOLTAGE_LIMIT
        else:
            scores = [error / max(abs(x), abs(y)) if x != 0 or y != 0 else Decimal(0)
                      for x, y, error in zip(a, b, differences)]
            limit = POPULATION_LIMIT
        maximum = max(scores, default=Decimal(0))
        report = {"criterion": "absolute_V" if voltage else "relative_without_floor",
                  "limit": str(limit), "maximum_error": str(maximum),
                  "maximum_absolute_error": str(max(differences, default=Decimal(0))),
                  "passed": maximum <= limit}
        if not report["passed"]:
            report.update(actual=[str(x) for x in a], expected=[str(x) for x in b])
        return report


def verify_initial_update(frozen, state, upstream, zero_minus_state):
    """Independently apply the actual recorded 0+ coordinate and lift.

    Endpoints retain the original binary64 contact law. Its actual outputs
    are captured before insertion into the fine state and are checked as
    exact inputs; interior and trace updates use independent Decimal math.
    """
    update = upstream.get("actual_update", {})
    required = ("reference_fields", "coordinate", "coordinate_indices", "voltage_lift_V",
                "trace_voltage_lift_V", "contact_boundary_inputs", "contact_evaluation")
    missing = ["actual_update." + name for name in required if name not in update]
    if missing or zero_minus_state is None:
        return {}, missing or ["actual_zero_minus_state"]
    with localcontext() as context:
        context.prec = 100
        fields = precision_fields(state)
        reference = update["reference_fields"]
        expected = {name: represented(reference[name]) for name in REFERENCE_FIELDS}
        checks = {
            "update_reference_identity": {"passed": digest(reference) == state_identity(zero_minus_state)},
            "update_output_identity": {"passed": update.get("fine_state_identity") == state_identity(state)},
        }
        indices, z = update["coordinate_indices"], represented(update["coordinate"])
        size, vt = len(expected["phi_V"]), dec(frozen["constants"]["thermal_voltage_V"])
        interface_count = len(frozen["coefficients"]["interface_nodes"])
        active_faces = [j for j, value in enumerate(frozen["coefficients"]["D_ion_face_m2_s"]) if dec(value) > 0]
        active = sorted({j for face in active_faces for j in (face, face + 1)})
        lengths = {"electron": size-2, "hole": size-2, "potential": size-2,
                   "trap": interface_count, "positive": len(active)}
        if any(len(indices.get(name, [])) != length for name, length in lengths.items()):
            raise ValueError("initial update coordinate index sizes differ from frozen material")
        local = indices.get("local", [])
        if len(local) != interface_count or any(len(block) != 6 for block in local):
            raise ValueError("initial update local indices differ from frozen material")
        all_indices = [i for name in lengths for i in indices[name]] + [i for block in local for i in block]
        if sorted(all_indices) != list(range(len(z))):
            raise ValueError("initial update indices do not partition its coordinate")
        lift, trace_lift = represented(update["voltage_lift_V"]), represented(update["trace_voltage_lift_V"])
        if len(lift) != size or len(trace_lift) != 2*interface_count:
            raise ValueError("initial voltage lift shapes differ from actual state")
        for j, node in enumerate(range(1, size-1)):
            zp, zn, zh = (z[indices[name][j]] for name in ("potential", "electron", "hole"))
            expected["phi_V"][node] += vt*zp + lift[node]
            expected["dqfn_V"][node] += vt*zn - lift[node]
            expected["dqfp_V"][node] += vt*zh + lift[node]
            expected["n_m3"][node] *= (zn+zp).exp()
            expected["p_m3"][node] *= (zh-zp).exp()
        for j, node in enumerate(active):
            expected["positive_m3"][node] *= z[indices["positive"][j]].exp()
        for j, block in enumerate(local):
            old, exponent = expected["occupancy"][j], z[indices["trap"][j]].exp()
            expected["occupancy"][j] = old*exponent/(1-old+old*exponent)
            for side in range(2):
                expected["trace_potential_V"][2*j+side] += vt*z[block[side]] + trace_lift[2*j+side]
            for carrier in range(4):
                expected["trace_state_m3"][4*j+carrier] *= z[block[2+carrier]].exp()
        contact = update["contact_evaluation"]
        contact_valid = (contact.get("source") == "original_binary64_coordinates_before_fine_endpoint_insertion"
                         and contact.get("inputs_derived_from_final_state") is False)
        if not contact_valid:
            missing.append("upstream_original_contact_evaluation")
        else:
            for name in ("phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3"):
                if name not in contact or len(represented(contact[name])) != 2:
                    raise ValueError("initial original contact field missing or malformed: " + name)
                expected[name][0], expected[name][-1] = represented(contact[name])
            boundary = update["contact_boundary_inputs"]
            # Reproduce only the explicitly retained original binary64
            # boundary subtraction, not an extra fine arithmetic operation.
            right = float(boundary["built_in_voltage_V"]) - float(boundary["junction_polarity"])*float(boundary["voltage_V"])
            checks["original_potential_boundary"] = arithmetic_comparison(contact["phi_V"], [0., right], voltage=True)
        for name, wanted in expected.items():
            checks["actual_update_" + name] = arithmetic_comparison(fields[name], wanted,
                voltage=name in ("phi_V", "dqfn_V", "dqfp_V", "trace_potential_V"))
    return checks, missing


def side_construction(frozen, fields, upstream, *, potential_field="phi_V"):
    """QF populations and sheet charge from actual upstream fixed inputs."""
    anchors = upstream.get("qf_anchors", {})
    required = ("phi0_V", "log_n0", "log_p0", "contact_n_m3", "contact_p_m3")
    missing = ["qf_anchors."+name for name in required if name not in anchors]
    charge = upstream.get("sheet_inputs", {})
    missing.extend("sheet_inputs."+name for name in
                   ("trap_density_m2", "equilibrium_occupancy", "static_sheet_charge_C_m2") if name not in charge)
    if missing:
        return {"qualified": False, "missing": missing, "checks": {}}
    with localcontext() as context:
        context.prec = 100
        phi, qn, qp = (represented(fields[name]) for name in (potential_field, "dqfn_V", "dqfp_V"))
        phi0, ln0, lp0 = (represented(anchors[name]) for name in required[:3])
        if len({len(value) for value in (phi, qn, qp, phi0, ln0, lp0)}) != 1:
            raise ValueError("side QF construction input sizes differ")
        vt, q = (dec(frozen["constants"][name]) for name in ("thermal_voltage_V", "q_C"))
        n = [(ln0[j]+(qn[j]+phi[j]-phi0[j])/vt).exp() for j in range(len(phi))]
        p = [(lp0[j]+(qp[j]-phi[j]+phi0[j])/vt).exp() for j in range(len(phi))]
        n[0], n[-1] = represented(anchors["contact_n_m3"])
        p[0], p[-1] = represented(anchors["contact_p_m3"])
        nt, eq, static = (represented(charge[name]) for name in
                         ("trap_density_m2", "equilibrium_occupancy", "static_sheet_charge_C_m2"))
        occupancy = represented(fields["occupancy"])
        if len({len(value) for value in (nt, eq, static, occupancy)}) != 1:
            raise ValueError("side sheet construction input sizes differ")
        sigma = [a+q*b*(c-d) for a,b,c,d in zip(static, nt, eq, occupancy)]
        checks = {"qf_n_m3": arithmetic_comparison(fields["n_m3"], n),
                  "qf_p_m3": arithmetic_comparison(fields["p_m3"], p),
                  "sheet_charge_C_m2": arithmetic_comparison(fields["sheet_charge_C_m2"], sigma)}
    return {"qualified": all(item["passed"] for item in checks.values()), "checks": checks, "missing": [],
            "potential_input": potential_field, "scope": "QF_and_sheet_arithmetic_from_frozen_upstream_inputs"}


def verify_two_sides(frozen, row, *, upstream=None):
    """Both actual solved paths against independent healthy constitutive laws."""
    direct_fields = precision_fields(row["state"])
    direct = verify_side(frozen, as_snapshot(direct_fields), observed(direct_fields))
    record = row.get("physics_reconstruction", {}).get("eliminated_precision")
    if not isinstance(record, dict) or record.get("schema") != "R1EliminatedPrecisionV1":
        return {"schema": "R1V8TwoSideOracleV1", "qualified": False, "direct": direct,
                "eliminated": None, "missing": ["actual_eliminated_precision_record"]}
    fields = record.get("fields", {})
    missing = [name for name in (*NEEDED_SIDE_FIELDS, *SHARED_FIELDS, "constraint_phi_V") if name not in fields]
    solve = record.get("solve", {})
    provenance_passed = (solve.get("kind") == "independent_fixed_qf_poisson"
                         and solve.get("used_direct_phi") is False and solve.get("converged") is True
                         and type(solve.get("iterations")) is int and 1 <= solve["iterations"] <= 12
                         and "last_correction_max_abs_V" in solve
                         and abs(dec(solve["last_correction_max_abs_V"])) < SOLVE_CORRECTION_LIMIT)
    shared = record.get("shared_inputs", {})
    direct_identity = state_identity(row["state"])
    input_identity_passed = (record.get("state_identity") == direct_identity
                             and shared.get("input_state_identity") == direct_identity)
    shared_inputs_passed = (shared.get("fields") == list(SHARED_FIELDS)
                            and shared.get("source") == "actual_direct_state_fixed_physical_inputs_not_independently_solved"
                            and all(name in fields and canonical(fields[name]) == canonical(direct_fields[name])
                                    for name in SHARED_FIELDS))
    if missing:
        return {"schema": "R1V8TwoSideOracleV1", "qualified": False, "direct": direct,
                "eliminated": None, "missing": missing, "solve_provenance_passed": provenance_passed}
    eliminated = verify_side(frozen, as_snapshot(fields), observed(fields))
    construction = None if upstream is None else {
        "direct": side_construction(frozen, direct_fields, upstream),
        "eliminated": side_construction(frozen, fields, upstream, potential_field="constraint_phi_V")}
    construction_passed = construction is None or all(item["qualified"] for item in construction.values())
    return {"schema": "R1V8TwoSideOracleV1", "qualified": direct["qualified"] and eliminated["qualified"]
                and provenance_passed and input_identity_passed and shared_inputs_passed and construction_passed,
            "direct": direct, "eliminated": eliminated, "missing": [],
            "solve_provenance_passed": provenance_passed, "solve": solve,
            "input_identity_passed": input_identity_passed, "shared_inputs_passed": shared_inputs_passed,
            "upstream_construction": construction,
            "shared_inputs": record.get("shared_inputs"),
            "scope": "per_side_constitutive_arithmetic_separate_from_original_path_gate"}


def verify_initial_arithmetic(frozen, state, upstream, *, phase, zero_minus_state=None):
    """Check 0-/0+ output from explicitly recorded upstream physical inputs.

    qf_anchors contains phi0_V, log_n0, log_p0 and actual contact reservoirs.
    trace_geometry contains the original binary64 capacitances and prescribed
    jump. Trace density arithmetic must identify its raw local-solver input
    or its reference and actual logarithmic increment; no final-output inverse
    transform is accepted as an upstream source.
    """
    checks, missing = {}, []
    if phase not in ("zero_minus", "zero_plus"):
        raise ValueError("initial arithmetic phase must be zero_minus or zero_plus")
    if not isinstance(upstream, dict) or upstream.get("schema") != "R1PrecisionArithmeticContextV1":
        return {"schema": "R1V8InitialArithmeticV1", "phase": phase, "qualified": False,
                "missing": ["upstream_precision_arithmetic_context"], "checks": {}}
    if upstream.get("inputs_derived_from_final_state") is not False:
        missing.append("explicit_non_inverse_upstream_provenance")
    with localcontext() as context:
        context.prec = 100
        fields = precision_fields(state)
        values = {name: represented(value) for name, value in fields.items()}
        phi, qn, qp = (values[name] for name in ("phi_V", "dqfn_V", "dqfp_V"))
        vt, q = (dec(frozen["constants"][name]) for name in ("thermal_voltage_V", "q_C"))
        anchors = upstream.get("qf_anchors", {})
        required = ("phi0_V", "log_n0", "log_p0", "contact_n_m3", "contact_p_m3")
        if not all(name in anchors for name in required):
            missing.extend("qf_anchors." + name for name in required if name not in anchors)
        else:
            phi0, ln0, lp0 = (represented(anchors[name]) for name in required[:3])
            if not len(phi) == len(phi0) == len(ln0) == len(lp0) == len(qn) == len(qp):
                raise ValueError("initial QF anchor shapes disagree")
            n = [(ln0[j] + (qn[j] + phi[j] - phi0[j]) / vt).exp() for j in range(len(phi))]
            p = [(lp0[j] + (qp[j] - phi[j] + phi0[j]) / vt).exp() for j in range(len(phi))]
            n[0], n[-1] = represented(anchors["contact_n_m3"])
            p[0], p[-1] = represented(anchors["contact_p_m3"])
            checks["qf_electron_population"] = arithmetic_comparison(fields["n_m3"], n)
            checks["qf_hole_population"] = arithmetic_comparison(fields["p_m3"], p)
        charge = upstream.get("sheet_inputs", {})
        if all(name in charge for name in ("trap_density_m2", "equilibrium_occupancy", "static_sheet_charge_C_m2")):
            nt, eq, static = (represented(charge[name]) for name in (
                "trap_density_m2", "equilibrium_occupancy", "static_sheet_charge_C_m2"))
            if not len(nt) == len(eq) == len(static) == len(values["occupancy"]):
                raise ValueError("sheet input shapes disagree")
            sheet = [fixed + q * density * (reference - fraction)
                     for fixed, density, reference, fraction in zip(static, nt, eq, values["occupancy"])]
            # This is charge arithmetic, not an invented physical relative gate.
            checks["sheet_construction"] = arithmetic_comparison(fields["sheet_charge_C_m2"], sheet)
        else:
            missing.append("sheet_inputs")
        geometry = upstream.get("trace_geometry", {})
        if "capacitances_F_m2" in geometry and "jump_V" in geometry:
            caps, jumps = geometry["capacitances_F_m2"], represented(geometry["jump_V"])
            nodes = frozen["coefficients"]["interface_nodes"]
            if len(caps) != len(nodes) or len(jumps) != len(nodes):
                raise ValueError("trace geometry shapes disagree")
            expected = []
            for (left, right), pair, jump, sigma in zip(nodes, caps, jumps, values["sheet_charge_C_m2"]):
                cl, cr = represented(pair)
                trace_left = (cl * phi[left] + cr * phi[right] + sigma - cr * jump) / (cl + cr)
                expected.extend((trace_left, trace_left + jump))
            checks["trace_electrostatics"] = arithmetic_comparison(fields["trace_potential_V"], expected, voltage=True)
        else:
            missing.append("trace_geometry")
        trace = upstream.get("trace_state_arithmetic", {})
        if trace.get("inputs_derived_from_final_state") is not False:
            missing.append("trace_state_arithmetic_upstream_provenance")
        elif trace.get("kind") == "raw_binary64_local_solver_input" and "input_m3" in trace:
            checks["trace_population_construction"] = arithmetic_comparison(fields["trace_state_m3"], represented(trace["input_m3"]))
        elif trace.get("kind") == "recorded_log_update" and "reference_m3" in trace and "log_increment" in trace:
            reference, increment = represented(trace["reference_m3"]), represented(trace["log_increment"])
            if len(reference) != len(increment):
                raise ValueError("trace update shape mismatch")
            checks["trace_population_construction"] = arithmetic_comparison(
                fields["trace_state_m3"], [a * b.exp() for a, b in zip(reference, increment)])
        else:
            missing.append("trace_state_arithmetic")
        if phase == "zero_plus":
            if zero_minus_state is None:
                missing.append("actual_zero_minus_state")
            else:
                before = precision_fields(zero_minus_state)
                for name in ("n_m3", "p_m3", "positive_m3", "occupancy"):
                    checks["fixed_population_" + name] = arithmetic_comparison(fields[name], represented(before[name]))
            update_checks, update_missing = verify_initial_update(frozen, state, upstream, zero_minus_state)
            checks.update(update_checks)
            missing.extend(update_missing)
        else:
            fixed = upstream.get("fixed_population_inputs", {})
            for name in ("positive_m3", "occupancy"):
                if name not in fixed:
                    missing.append("fixed_population_inputs." + name)
                else:
                    checks["fixed_initial_" + name] = arithmetic_comparison(fields[name], represented(fixed[name]))
            for name in ("dqfn_V", "dqfp_V"):
                if name not in anchors:
                    missing.append("qf_anchors." + name)
                else:
                    checks["fixed_initial_" + name] = arithmetic_comparison(fields[name], anchors[name], voltage=True)
        equation = evaluate_state(frozen, as_snapshot(fields), precision=100)["poisson"]
    return {"schema": "R1V8InitialArithmeticV1", "phase": phase,
            "qualified": not missing and bool(checks) and all(item["passed"] for item in checks.values()),
            "checks": checks, "missing": missing, "poisson_absolute_equation": equation,
            "upstream_context_sha256": digest(upstream),
            "scope": "recorded_initial_construction_arithmetic_original_initial_physics_certificate_remains_separate"}
