"""Aggregation semantics without an invented new trajectory or gate."""
import copy

from scripts.summarize_r1_v8_gates import LIMITS, aggregate


def sample():
    rows = []
    for tier, when, operator in ((1, 10., 2e-6), (2, 1., 3e-6), (4, 2., 1e-8)):
        for finite in (False, True):
            component = {"maximum_absolute_difference": 2e-8, "normalization_scale": 2.,
                "normalization_floor": 1., "relative_error": 1e-8, "floor_active": False, "unit": "example",
                "direct_maximum_absolute": 2., "eliminated_maximum_absolute": 2.}
            row = {"phase": "accepted_regular_step" if finite else "0+", "substeps": tier,
                "time_s": when if finite else 0., "physical_checks_passed": True,
                "physical_failure_reasons": [], "physical_checks": {"passed": True},
                "physical": {"inventory_relative_drift": 1e-14, "gauss_normalized": 1e-14,
                    "charge_balance_normalized": 1e-14, "contact_internal_current_spread_relative": 1e-8,
                    "trap_storage_check": {"normalized_error": .5},
                    "regular_right_limit": {"face_current_spread_relative": 5e-7, "interface_current_spread_relative": 1e-9}},
                "physics_reconstruction": {"local_carrier_residual": 1e-12, "local_gauss_residual": 1e-12,
                    "eliminated_operator_error": operator if finite else 1e-8,
                    "scaled_nonlinear_residual": .02 if finite else 999.,
                    "charge_balance_relative": 1e-14, "all_face_current_relative": 1e-8,
                    "interface_current_relative": 1e-9, "jacobian_checked": finite,
                    "analytic_jacobian_error": 1e-5 if finite else None,
                    "conduction_A_m2": [3.], "carrier_conduction_A_m2": [1.], "positive_ion_current_A_m2": [2.],
                    "eliminated_operator": {"positive_ion_flux": component, "positive_ion_rate": copy.deepcopy(component)}}}
            rows.append(row)
    certificate = {"metrics": {name: 1e-8 for name in LIMITS}, "limits": dict(LIMITS), "certified": True}
    return rows, certificate


def test_row_applicability_and_certificate_scope_are_separate():
    rows, certificate = sample()
    report = aggregate(rows, certificate)
    gates = report["original_gates"]
    assert set(gates) == set(LIMITS)
    assert gates["nonlinear_residual"]["evaluated_rows"] == 3
    assert gates["nonlinear_residual"]["maximum"] == .02  # excludes row0 compatibility placeholder
    assert gates["analytic_jacobian_error"]["evaluated_rows"] == 3
    assert gates["all_face_current_relative"]["evaluated_rows"] == 6
    assert gates["all_face_current_relative"]["maximum"] == 5e-7  # actual initial right limit
    assert gates["eliminated_operator_error"]["maximum"] == 3e-6
    assert gates["eliminated_operator_error"]["finest_saved_row_maximum"]["maximum"] == 1e-8
    assert not gates["eliminated_operator_error"]["certificate_value_is_all_row_maximum"]
    assert gates["refinement_state_change"]["evaluated_rows"] is None
    assert not gates["refinement_state_change"]["rowwise_aggregation_applicable"]
    assert not gates["refinement_state_change"]["independently_recomputed_here"]


def test_failure_file_order_is_not_confused_with_earliest_physical_time():
    report = aggregate(*sample())
    assert report["failed_rows"] == 2
    assert report["first_failure_in_file_order"]["row"] == 1
    assert report["earliest_failure_by_physical_time"]["row"] == 3
    assert report["per_substeps"]["4"]["failed_rows"] == 0
    detail = report["ion_components"]["positive_ion_flux"]["maximum_absolute_difference"]
    assert detail["maximum_absolute_difference"] == 2e-8
    assert detail["normalization_scale"] == 2. and detail["normalization_floor"] == 1.
    assert report["original_gates"]["eliminated_operator_error"]["headroom_fraction"] < 0.


def test_original_rate_floor_is_dynamic_and_is_not_replaced_by_oracle_floor_one():
    rows, certificate = sample()
    for index, row in enumerate(rows):
        detail = row["physics_reconstruction"]["eliminated_operator"]["positive_ion_rate"]
        floor = 1e9 + index
        detail.update(normalization_floor=floor, normalization_scale=floor,
                      relative_error=detail["maximum_absolute_difference"] / floor, floor_active=True)
    rate = aggregate(rows, certificate)["ion_components"]["positive_ion_rate"]
    assert rate["minimum_recorded_floor"] == 1e9
    assert rate["maximum_recorded_floor"] == 1e9 + 5
    assert rate["floor_active_rows"] == 6
