"""Reanchoring a payload cannot turn fabricated physical metrics into evidence."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import ALLOWED_TIME_SUBSTEPS
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from tests.unit.experiments.test_one_dimensional_mechanism_r1_preparation_chain import formal_context
from tests.unit.experiments.test_one_dimensional_mechanism_r1_state import common_state, single_thread_blas


DC_PHYSICAL_FIELDS = (
    "maximum_normalized_residual", "electron_continuity_bound_A_m2",
    "hole_continuity_bound_A_m2", "maximum_ion_electrochemical_residual",
    "maximum_ionic_face_current_A_m2", "maximum_ion_inventory_relative_error",
    "face_current_spread_A_m2", "poisson_residual",
    "maximum_bulk_trap_balance_relative_error", "maximum_interface_residual",
    "maximum_interface_gauss_residual",
)
PREPARATION_FIELDS = (
    "electron_continuity_A_m2", "hole_continuity_A_m2",
    "electron_normalized_residual", "hole_normalized_residual",
    "ion_equilibrium_residual", "inventory_relative_error",
    "nominal_inventory_relative_error", "ionic_face_current_A_m2",
    "dc_current_spread_A_m2", "poisson_normalized", "full_gauss_normalized",
    "local_carrier_residual", "local_gauss_residual", "trap_storage_rate_A_m2",
    "eliminated_operator_error",
)


def reanchor(record):
    """Attacker may update every self-hash AND supply a matching new anchor."""
    record["source"] = common.execution_source()
    dc = record["dc_state"]
    dc["state_sha256"] = common._state_sha256(*(
        np.asarray(dc[key], dtype=float) for key in (
            "electron_qf_increment_V", "hole_qf_increment_V", "positive_ion_density_m3",
        )
    ))
    record["sha256"] = common.digest({k: v for k, v in record.items() if k != "sha256"})
    return common.R1PreparedState.from_dict(record)


def restore_reanchored(case, record):
    prepared = reanchor(record)
    return common.restore_common_state(
        prepared, case[0], 4, case[1], expected_prepared_sha256=prepared.sha256,
    )


@pytest.mark.parametrize("field", DC_PHYSICAL_FIELDS)
def test_reanchored_dc_physical_certificate_is_recomputed(common_state, formal_context, field):
    record = common_state[2].to_dict()
    actual = record["dc_state"]["certificate"][field]
    record["dc_state"]["certificate"][field] = 0.0 if actual else 1e-30
    with pytest.raises(common.R1StateError, match=f"recomputed physics: {field}"):
        restore_reanchored(common_state, record)


@pytest.mark.parametrize("field", ["tolerance_eV", "fermi_level_span_eV", "potential_mismatch_V",
                                  "metal_work_function_mismatch_eV", "contact_quasi_fermi_levels_eV"])
def test_nested_contact_certificate_is_recomputed(common_state, formal_context, field):
    record = common_state[2].to_dict()
    contact = record["dc_state"]["certificate"]["contact_thermodynamics"]
    contact[field] = [0.0] if field == "contact_quasi_fermi_levels_eV" else 3.0
    with pytest.raises(common.R1StateError, match="recomputed physics: contact_thermodynamics"):
        restore_reanchored(common_state, record)


@pytest.mark.parametrize("field", PREPARATION_FIELDS)
@pytest.mark.parametrize("section", ["metrics", "limits"])
def test_reanchored_preparation_numbers_are_recomputed(common_state, formal_context, section, field):
    record = common_state[2].to_dict()
    actual = record["preparation_checks"][section][field]
    record["preparation_checks"][section][field] = 0.0 if actual else 1e-30
    with pytest.raises(common.R1StateError, match="preparation_checks differ from recomputed physics"):
        restore_reanchored(common_state, record)


@pytest.mark.parametrize("field,value", [("certified", False), ("reasons", ["fabricated_failure"]),
                                        ("extra_metric", 0.0)])
def test_reanchored_preparation_decision_and_schema_are_recomputed(common_state, formal_context, field, value):
    record = common_state[2].to_dict()
    record["preparation_checks"][field] = value
    with pytest.raises(common.R1StateError, match="preparation_checks differ from recomputed physics"):
        restore_reanchored(common_state, record)


@pytest.mark.parametrize("field,value", [
    ("maximum_dc_continuity_bound_A_m2", 1.0),
    ("maximum_dc_normalized_residual", 1.0),
    ("maximum_dc_inventory_error", 1.0),
    ("maximum_newton_iterations", 100000),
    ("dc_max_nfev", 100000),
    ("maximum_newton_iterations", True),
    ("storage_relative_tolerance", "1e-9"),
    ("undeclared", 1.0),
])
def test_reanchored_policy_cannot_loosen_physics_or_change_schema(common_state, formal_context, field, value):
    record = common_state[2].to_dict()
    record["preparation_policy"][field] = value
    # An attacker is also free to revise its reported limits and pass flag.
    record["preparation_checks"]["limits"] = {k: 1e9 for k in PREPARATION_FIELDS}
    record["preparation_checks"].update(certified=True, reasons=[])
    with pytest.raises(common.R1StateError, match="preparation_policy is not a declared"):
        restore_reanchored(common_state, record)


def test_missing_preparation_policy_field_cannot_fall_back_to_default(common_state, formal_context):
    record = common_state[2].to_dict()
    del record["preparation_policy"]["maximum_dc_normalized_residual"]
    with pytest.raises(common.R1StateError, match="preparation_policy is not a declared"):
        restore_reanchored(common_state, record)


def test_qss_certificate_is_distinct_from_canonical_dynamic_embedding(common_state, monkeypatch):
    stack, binding, prepared = common_state
    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not launch another DC optimizer")
    monkeypatch.setattr(common, "solve_r1_dc", forbidden)
    system, state = common.verify_prepared_physics(prepared, stack, binding)
    saved = prepared.to_dict()
    actual = common._recomputed_dc_certificate(system, system.common_dc_state)
    assert actual["electron_continuity_bound_A_m2"] == saved["dc_state"]["certificate"]["electron_continuity_bound_A_m2"]
    # Roundoff-sized QSS and dynamic residuals are intentionally different.
    assert actual["electron_continuity_bound_A_m2"] != saved["preparation_checks"]["metrics"]["electron_continuity_A_m2"]
    np.testing.assert_array_equal(state.n, saved["state"]["n_m3"])
    np.testing.assert_array_equal(state.p, saved["state"]["p_m3"])
    np.testing.assert_array_equal(state.phi, saved["state"]["phi_V"])


def test_bounded_verifier_does_not_claim_live_execution_source_identity(common_state, monkeypatch):
    stack, binding, prepared = common_state
    def forbidden():
        raise AssertionError("bundle verification must use its independently anchored source")
    monkeypatch.setattr(common, "execution_source", forbidden)
    common.verify_prepared_physics(prepared, stack, binding)


def test_saved_policy_is_used_when_consumer_changes_nonlinear_factor(common_state):
    stack, binding, prepared = common_state
    record = prepared.to_dict()
    original = common.verify_prepared_physics(prepared, stack, binding)
    consumer = common.verify_prepared_physics(prepared, stack, binding, policy=r1_policy(0.001))
    assert record == prepared.to_dict()
    assert common.snapshot(*original) == common.snapshot(*consumer)
    strict_dc = replace(r1_policy(), maximum_dc_continuity_bound_A_m2=1e-30)
    with pytest.raises(common.R1StateError, match="consumer D equations"):
        common.verify_prepared_physics(prepared, stack, binding, policy=strict_dc)
    consumer_ceiling = replace(r1_policy(), site_occupancy_ceiling=0.998)
    system, state = common.verify_prepared_physics(prepared, stack, binding, policy=consumer_ceiling)
    assert system.site_occupancy_ceiling == 0.998
    assert common.snapshot(system, state) == common.snapshot(*original)


@pytest.mark.parametrize("factor", [1.0, 0.1, 0.01, 0.001])
@pytest.mark.parametrize("substeps", ALLOWED_TIME_SUBSTEPS)
def test_all_declared_independent_policy_axes_are_admitted(factor, substeps):
    policy = r1_policy(factor, time_substeps=substeps)
    assert common._preparation_policy(common.json_data(policy)) == policy


def test_preparation_with_independent_axes_preserves_saved_checks(common_state):
    stack, binding, _ = common_state
    preparation_policy = r1_policy(0.01, time_substeps=(8, 16, 32))
    prepared = common.prepare_common_state(stack, 4, binding, policy=preparation_policy)
    assert prepared.to_dict()["preparation_policy"] == common.json_data(preparation_policy)
    common.verify_prepared_physics(prepared, stack, binding,
                                   policy=r1_policy(0.001, time_substeps=(2, 4, 8)))


@pytest.mark.parametrize("field", ["environment", "created_utc", "optimizer_nfev"])
def test_historical_metadata_is_explicitly_provenance_only(common_state, formal_context, field):
    record = common_state[2].to_dict()
    if field == "environment":
        record[field]["python"] = "a claimed historical Python environment"
        path = field
    elif field == "created_utc":
        record[field] = "2000-01-01T00:00:00+00:00"
        path = field
    else:
        record["dc_state"]["certificate"][field] += 1
        path = "dc_state.certificate." + field
    assert path in record["verification"]["provenance_only"]
    assert record["verification"]["provenance_assurance"] == "content_bound_claims_not_independently_verified"
    _, restored = restore_reanchored(common_state, record)
    np.testing.assert_array_equal(restored.n, common_state[2].to_dict()["state"]["n_m3"])


def test_reanchoring_cannot_promote_provenance_to_verified_physics(common_state, formal_context):
    record = common_state[2].to_dict()
    record["verification"]["provenance_assurance"] = "independently_verified"
    with pytest.raises(common.R1StateError, match="identity mismatch: verification"):
        restore_reanchored(common_state, record)
