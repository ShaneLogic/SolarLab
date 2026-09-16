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


def verify_step_metadata(record, *, intervals, policy, polarity, same):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import SCOPE, VERSION

    unclassified = set(record) - STEP_FIELD_ROLES.keys()
    if unclassified:
        raise ValueError("unclassified controlled-step fields: " + ", ".join(sorted(unclassified)))
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
