"""V8 evidence routing and initial arithmetic semantics, not a trajectory."""
from copy import deepcopy
from decimal import Decimal, localcontext

import pytest

from scripts import verify_r1_v8_precision as check


def split(values):
    with localcontext() as context:
        context.prec = 100
        high = [float(value) for value in values]
        return {"hi": high, "lo": [float(Decimal(value) - Decimal.from_float(h)) for value, h in zip(values, high)]}


def minimal_fields():
    fields = {name: split([Decimal(0)] * (2 if name == "boundary_flux_m2_s" else 3))
              for name in check.NEEDED_SIDE_FIELDS}
    fields["sheet_charge_C_m2"] = split([])
    fields.update(dqfn_V=split([Decimal(0)] * 3), dqfp_V=split([Decimal(0)] * 3),
                  occupancy=split([]), trace_potential_V=split([]), trace_state_m3=split([]))
    return fields


def valid_solve():
    return {"kind": "independent_fixed_qf_poisson", "used_direct_phi": False, "converged": True,
            "iterations": 2, "last_correction_max_abs_V": 1e-30}


def side_record(direct, eliminated, solve=None):
    identity = check.state_identity(check.as_snapshot(direct))
    eliminated = deepcopy(eliminated)
    eliminated["constraint_phi_V"] = deepcopy(eliminated["phi_V"])
    return {"schema": "R1EliminatedPrecisionV1", "fields": eliminated, "solve": solve or valid_solve(),
            "state_identity": identity, "shared_inputs": {"input_state_identity": identity,
                "fields": list(check.SHARED_FIELDS),
                "source": "actual_direct_state_fixed_physical_inputs_not_independently_solved"}}


def test_two_sides_are_routed_separately_and_missing_elimination_cannot_pass(monkeypatch):
    seen = []
    monkeypatch.setattr(check, "verify_side", lambda frozen, state, output: seen.append(state) or {"qualified": True})
    direct, eliminated = minimal_fields(), minimal_fields()
    eliminated["phi_V"]["lo"][1] = 1e-30
    row = {"state": check.as_snapshot(direct), "physics_reconstruction": {
        "eliminated_precision": side_record(direct, eliminated)}}
    result = check.verify_two_sides({}, row)
    assert result["qualified"] and len(seen) == 2
    assert seen[0]["precision"]["fields"]["phi_V"] != seen[1]["precision"]["fields"]["phi_V"]
    del row["physics_reconstruction"]["eliminated_precision"]
    assert not check.verify_two_sides({}, row)["qualified"]


@pytest.mark.parametrize("field,value", [("used_direct_phi", True), ("converged", False),
                                         ("iterations", 13), ("last_correction_max_abs_V", "1e-28")])
def test_invalid_independent_solve_provenance_does_not_qualify(monkeypatch, field, value):
    monkeypatch.setattr(check, "verify_side", lambda *_: {"qualified": True})
    solve = valid_solve(); solve[field] = value
    fields = minimal_fields()
    row = {"state": check.as_snapshot(fields), "physics_reconstruction": {
        "eliminated_precision": side_record(fields, fields, solve)}}
    assert not check.verify_two_sides({}, row)["qualified"]


def test_independent_record_must_be_bound_to_actual_direct_shared_inputs(monkeypatch):
    monkeypatch.setattr(check, "verify_side", lambda *_: {"qualified": True})
    fields = minimal_fields()
    record = side_record(fields, fields)
    row = {"state": check.as_snapshot(fields), "physics_reconstruction": {"eliminated_precision": record}}
    record["state_identity"] = "unrelated"
    assert not check.verify_two_sides({}, row)["input_identity_passed"]
    record["state_identity"] = record["shared_inputs"]["input_state_identity"]
    record["fields"]["positive_m3"]["lo"][1] = 1e-30
    assert not check.verify_two_sides({}, row)["shared_inputs_passed"]


def test_initial_check_requires_upstream_and_does_not_invert_outputs(monkeypatch):
    assert not check.verify_initial_arithmetic({}, {}, {}, phase="zero_minus")["qualified"]
    fields = minimal_fields()
    fields["n_m3"] = split([Decimal(1)] * 3)
    fields["p_m3"] = split([Decimal(1)] * 3)
    upstream = {"schema": "R1PrecisionArithmeticContextV1", "inputs_derived_from_final_state": False,
        "qf_anchors": {"phi0_V": [0., 0., 0.], "log_n0": [0., 0., 0.], "log_p0": [0., 0., 0.],
                       "contact_n_m3": [1., 1.], "contact_p_m3": [1., 1.],
                       "dqfn_V": [0., 0., 0.], "dqfp_V": [0., 0., 0.]},
        "sheet_inputs": {"trap_density_m2": [], "equilibrium_occupancy": [], "static_sheet_charge_C_m2": []},
        "trace_geometry": {"capacitances_F_m2": [], "jump_V": []},
        "trace_state_arithmetic": {"kind": "raw_binary64_local_solver_input", "input_m3": [],
                                    "inputs_derived_from_final_state": False},
        "fixed_population_inputs": {"positive_m3": [0., 0., 0.], "occupancy": []}}
    frozen = {"constants": {"thermal_voltage_V": "0.025", "q_C": "1.6e-19"},
              "coefficients": {"interface_nodes": []}}
    monkeypatch.setattr(check, "evaluate_state", lambda *a, **k: {"poisson": {"scope": "test_only"}})
    assert check.verify_initial_arithmetic(frozen, check.as_snapshot(fields), upstream, phase="zero_minus")["qualified"]
    derived = deepcopy(upstream); derived["inputs_derived_from_final_state"] = True
    assert not check.verify_initial_arithmetic(frozen, check.as_snapshot(fields), derived, phase="zero_minus")["qualified"]
    faulty = deepcopy(fields); faulty["n_m3"]["lo"][1] = 1e-20
    assert not check.verify_initial_arithmetic(frozen, check.as_snapshot(faulty), upstream, phase="zero_minus")["qualified"]


def test_zero_plus_requires_actual_prior_state_even_when_qf_math_passes(monkeypatch):
    # Missing context is not repaired from final state values.
    result = check.verify_initial_arithmetic({}, {}, None, phase="zero_plus")
    assert not result["qualified"] and result["missing"]


def test_population_check_has_no_floor_that_masks_a_small_fraction():
    assert not check.arithmetic_comparison(["1e-40"], ["2e-40"])["passed"]
    assert check.arithmetic_comparison(["0"], ["0"])["passed"]


def test_initial_update_uses_actual_reference_coordinate_lift_and_preinsertion_contacts():
    fields = minimal_fields()
    fields["n_m3"] = split([Decimal(1)]*3)
    fields["p_m3"] = split([Decimal(1)]*3)
    reference = {name: deepcopy(fields[name]) for name in check.REFERENCE_FIELDS}
    z, vt = Decimal("1e-24"), Decimal("0.025")
    fields["phi_V"] = split([Decimal(0), vt*z, Decimal(0)])
    fields["dqfn_V"] = split([Decimal(0), -vt*z, Decimal(0)])
    fields["dqfp_V"] = split([Decimal(0), vt*z, Decimal(0)])
    state = check.as_snapshot(fields)
    context = {"actual_update": {"reference_fields": reference,
        "coordinate": [str(-z), str(z), str(z)],
        "coordinate_indices": {"electron": [0], "hole": [1], "potential": [2], "positive": [], "trap": [], "local": []},
        "voltage_lift_V": [0., 0., 0.], "trace_voltage_lift_V": [],
        "contact_boundary_inputs": {"built_in_voltage_V": 0., "junction_polarity": 1., "voltage_V": 0.},
        "contact_evaluation": {"source": "original_binary64_coordinates_before_fine_endpoint_insertion",
            "inputs_derived_from_final_state": False, "phi_V": [0.,0.], "dqfn_V": [0.,0.],
            "dqfp_V": [0.,0.], "n_m3": [1.,1.], "p_m3": [1.,1.]},
        "fine_state_identity": check.state_identity(state)}}
    frozen = {"constants": {"thermal_voltage_V": str(vt)},
              "coefficients": {"D_ion_face_m2_s": [0.,0.], "interface_nodes": []}}
    checks, missing = check.verify_initial_update(frozen, state, context, check.as_snapshot(reference))
    assert not missing and all(item["passed"] for item in checks.values())
    context["actual_update"]["coordinate"][2] = "1e-10"
    checks, _ = check.verify_initial_update(frozen, state, context, check.as_snapshot(reference))
    assert not checks["actual_update_phi_V"]["passed"]
