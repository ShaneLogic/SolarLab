"""Classify every controlled-step field and bind its deterministic meaning."""
from __future__ import annotations


STEP_FIELD_ROLES = {
    **dict.fromkeys(("schema", "scope", "version", "scope_note", "execution_axes",
                    "junction_polarity", "current_sign_convention"), "deterministic_metadata"),
    **dict.fromkeys(("prepared_sha256", "reference_sha256", "controls", "control_label",
                    "intervals", "amplitude_V", "times_s", "voltage_V", "policy",
                    "accepted_steps", "initial_event", "physics_reconstruction", "certificate",
                    "output_states", "regular_currents", "finite_step_averages",
                    "accepted_state_arrays", "charge_integral"), "physical_reconstruction"),
    **dict.fromkeys(("source", "sha256"), "outer_source_and_serialization_binding"),
    **dict.fromkeys(("failure", "persistence_failure"), "recorded_failure_provenance"),
}
RESULT_ARRAY_FIELDS = ("output_states", "regular_currents", "finite_step_averages",
                       "accepted_state_arrays", "charge_integral")
SCOPE_NOTE = "Single controlled-step execution at the recorded observation times; no full-window or three-axis convergence claim"
CURRENT_SIGN_CONVENTION = "J_x along +x; reported j = junction_polarity * J_x"
AVERAGE_NOTE = "positive-time values are backward-Euler last-substep averages; row0 is the regular right limit"
CHARGE_QUADRATURE = "sum of each accepted backward-Euler physical-contact current times dt"

# Children compared as whole reconstructed dictionaries are already closed by
# equality. These containers were previously inspected member by member and
# therefore need an explicit field set as well as their numerical checks.
ACCEPTED_ROW_FIELDS = frozenset((
    "substeps", "time_s", "dt_s", "phase", "solver_accepted", "state", "physical",
    "physics_reconstruction", "physical_checks", "physical_checks_passed",
    "physical_failure_reasons", "scaled_nonlinear_residual", "regular_integrated_charge_C_m2",
    "persistence_failure",
))
STEP_CERTIFICATE_FIELDS = frozenset((
    "certified", "reasons", "scope", "metrics", "limits", "analytic_jacobian_nnz",
    "dense_entries", "exact_freeze_checks", "finite_numeric_evidence",
    "maximum_absolute_charge_balance_error_A_m2", "maximum_absolute_poisson_residual_C_m2",
    "maximum_trap_storage_error_A_m2", "prepared_D_equilibrium_certified",
    "site_occupancy_fraction", "trap_storage", "zero_plus_is_equilibrium",
    "failed_record_index", "physical_checks", "nonfinite_numeric_paths", "secondary_reasons",
))
PREPARED_FIELDS = frozenset((
    "schema", "kind", "state_time", "intervals", "preparation_controls", "grid",
    "physical_stack", "stack_sha256", "fixed_reference", "reference_sha256",
    "study_spec_sha256", "contact_velocities_m_s", "history", "verification",
    "dc_state", "state", "qf_references_V", "contact_certificate", "preparation_checks",
    "preparation_policy", "environment", "created_utc", "source", "sha256",
))
DC_STATE_FIELDS = frozenset((
    "bulk_trap_occupancy", "certificate", "electron_density_m3", "electron_qf_increment_V",
    "hole_density_m3", "hole_qf_increment_V", "interface_occupancy", "negative_ion_density_m3",
    "positive_ion_density_m3", "potential_V", "state_sha256",
))
DC_CERTIFICATE_FIELDS = frozenset((
    "certified", "contact_thermodynamics", "electron_continuity_bound_A_m2",
    "face_current_spread_A_m2", "hole_continuity_bound_A_m2",
    "maximum_bulk_trap_balance_relative_error", "maximum_interface_gauss_residual",
    "maximum_interface_residual", "maximum_ion_electrochemical_residual",
    "maximum_ion_inventory_relative_error", "maximum_ionic_face_current_A_m2",
    "maximum_normalized_residual", "optimizer_nfev", "optimizer_success", "poisson_residual", "reasons",
))
ZERO_FIELDS = frozenset((
    "schema", "scope", "version", "prepared_sha256", "reference_sha256", "controls",
    "certificate", "source", "sha256", "failure",
))
FAILURE_FIELDS = frozenset((
    "type", "message", "traceback", "reasons", "reason", "record_index", "time_s", "substeps",
    "nonfinite_numeric_paths", "numerical_evidence", "persistence_failure",
))
PERSISTENCE_FIELDS = frozenset(("reason", "type", "message", "record_index", "time_s", "substeps"))


def _closed(record, fields, label):
    if not isinstance(record, dict):
        raise ValueError(label + " must be a record")
    extra = set(record) - fields
    if extra:
        raise ValueError("unclassified " + label + " fields: " + ", ".join(sorted(extra)))


def verify_failure_metadata(record, *, failed, rows=None, same=None):
    """Bind saved I/O coordinates, without claiming that I/O history was replayed."""
    for name in ("failure", "persistence_failure"):
        if name in record and not failed:
            raise ValueError("passed result contains " + name)
    if "failure" in record:
        _closed(record["failure"], FAILURE_FIELDS, "failure provenance")
        failure = record["failure"]
        if failure.get("type") == "PhysicalCheckFailure":
            index = failure.get("record_index")
            if type(index) is not int or not rows or index != len(rows) - 1:
                raise ValueError("recorded physical failure lacks its terminal saved witness")
            if rows[index].get("physical_checks_passed") is not False or not rows[index].get("physical_failure_reasons"):
                raise ValueError("recorded physical failure has no violating saved row")
            if same is not None:
                same(failure.get("reasons"), rows[index]["physical_failure_reasons"], "terminal failure reasons")
    certificate = record.get("certificate", {})
    if "failed_record_index" in certificate:
        index = certificate["failed_record_index"]
        if type(index) is not int or not rows or index != len(rows) - 1:
            raise ValueError("failure certificate terminal row is absent from the saved prefix")
        if "physical_checks" in certificate and same is not None:
            same(certificate["physical_checks"], rows[index].get("physical_checks"), "terminal failure checks")
    persistence = record.get("persistence_failure")
    row_failures = [(index, row["persistence_failure"]) for index, row in enumerate(rows or [])
                    if "persistence_failure" in row]
    if row_failures and persistence is None:
        raise ValueError("saved row persistence failure lacks result provenance")
    if persistence is None:
        return
    _closed(persistence, PERSISTENCE_FIELDS, "persistence provenance")
    if (not rows or len(row_failures) != 1 or row_failures[0][0] != len(rows) - 1
            or persistence.get("record_index") != len(rows) - 1):
        raise ValueError("persistence failure must identify the last saved raw row")
    if same is not None:
        same(row_failures[0][1], persistence, "row persistence provenance")
        for field in ("time_s", "substeps"):
            same(persistence.get(field), rows[-1].get(field), "persistence " + field)
    if persistence.get("reason") != "accepted_step_persistence_failed":
        raise ValueError("unrecognized persistence failure reason")


def verify_prepared_metadata(record):
    """Close preparation containers; equations and historical types are checked separately."""
    _closed(record, PREPARED_FIELDS, "prepared state")
    if "dc_state" in record:
        _closed(record["dc_state"], DC_STATE_FIELDS, "prepared DC state")
        if "certificate" in record["dc_state"]:
            _closed(record["dc_state"]["certificate"], DC_CERTIFICATE_FIELDS, "prepared DC certificate")
    if "source" in record:
        _closed(record["source"], {"files", "study_input", "run_class", "source_commit",
                                   "source_content_sha256", "sha256"}, "prepared source")


def verify_zero_metadata(record, *, same, failed=False):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import (
        SCOPE, VERSION, nonfinite_numeric_paths,
    )

    _closed(record, ZERO_FIELDS, "zero-excitation")
    for name, expected in (("schema", "R1ZeroExcitationV1"), ("scope", SCOPE), ("version", VERSION)):
        same(record.get(name), expected, "zero metadata " + name)
    certificate = record.get("certificate")
    _closed(certificate, {"certified", "reasons", "scope", "relative_dynamic_current_certified",
                          "finite_numeric_evidence"}, "zero certificate")
    partial = failed and "relative_dynamic_current_certified" not in certificate
    same(certificate.get("scope"), SCOPE if partial else "remaining_equilibrium_equations_and_no_impulse",
         "zero certificate scope")
    if not partial:
        same(certificate.get("relative_dynamic_current_certified"), False, "zero dynamic-current disclaimer")
    if "finite_numeric_evidence" in certificate or not failed:
        body = dict(record, certificate={k: v for k, v in certificate.items() if k != "finite_numeric_evidence"})
        paths = nonfinite_numeric_paths(body)
        same(certificate.get("finite_numeric_evidence"), {
            "passed": not paths, "nonfinite_numeric_paths": paths, "scope": "all_numeric_leaves_before_digest",
        }, "zero finite numeric certificate")
    verify_failure_metadata(record, failed=failed)
    controls = record.get("controls")
    _closed(controls, set("ABCD"), "zero controls")
    for label, item in controls.items():
        _closed(item, {"controls", "remaining_equations", "initial_event", "population_identity"},
                "zero control " + label)


def verify_step_metadata(record, *, intervals, policy, polarity, same):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import SCOPE, VERSION

    unclassified = set(record) - STEP_FIELD_ROLES.keys()
    if unclassified:
        raise ValueError("unclassified controlled-step fields: " + ", ".join(sorted(unclassified)))
    certificate = record.get("certificate", {})
    _closed(certificate, STEP_CERTIFICATE_FIELDS, "step certificate")
    rows = record.get("accepted_steps", [])
    if not isinstance(rows, list):
        raise ValueError("accepted rows must be a list")
    for index, row in enumerate(rows):
        _closed(row, ACCEPTED_ROW_FIELDS, "accepted row " + str(index))
    verify_failure_metadata(record, failed=certificate.get("certified") is False, rows=rows, same=same)
    expected = {
        "schema": "R1ControlledStepV1", "scope": SCOPE, "version": VERSION,
        "scope_note": SCOPE_NOTE,
        "execution_axes": {"intervals": int(intervals), "time_substeps": list(policy.refinement_substeps)},
        "junction_polarity": polarity, "current_sign_convention": CURRENT_SIGN_CONVENTION,
    }
    for name, value in expected.items():
        same(record.get(name), value, "result metadata " + name)
    if "finite_step_averages" in record:
        same(record["finite_step_averages"].get("note"), AVERAGE_NOTE, "finite-step average meaning")
    if "charge_integral" in record:
        same(record["charge_integral"].get("quadrature"), CHARGE_QUADRATURE, "charge quadrature meaning")
