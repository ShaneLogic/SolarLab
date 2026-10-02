"""Common-state identity, exact population reuse, and live-equation gates."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.constants import Q
from perovskite_sim.experiments import defect_ion_combined_impedance as combined
from perovskite_sim.experiments import interface_defect_ion_transient as transient
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments import one_dimensional_mechanism_r1_binding as binding_gate
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import (
    build_r1_material, validate_binding,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import (
    APPROVED_R1_BINDING_PATH, alternate_r1_binding, approved_r1_binding,
)
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1, user_api="blas"):
        yield


@pytest.fixture(scope="module")
def common_state():
    stack = load_device_from_yaml(FIXTURE)
    reference = approved_r1_binding()
    return stack, reference, common.prepare_common_state(stack, 4, reference)


@pytest.fixture(scope="module")
def common_state_16():
    stack = load_device_from_yaml(FIXTURE)
    reference = approved_r1_binding()
    return stack, reference, common.prepare_common_state(stack, 16, reference)


def _restore(case, prepared=None, *, label="D"):
    stack, reference, original = case
    return common.restore_common_state(
        original if prepared is None else prepared,
        stack,
        4,
        reference,
        controls=R1DynamicsControls.from_label(label),
    )


def test_physical_identity_preserves_run_identity_but_ignores_only_provenance(common_state):
    import copy
    original = common_state[2].to_dict()
    changed = copy.deepcopy(original)
    changed["created_utc"] = "2030-01-01T00:00:00+00:00"
    assert common.physical_preparation_identity(original) == common.physical_preparation_identity(changed)
    assert common.digest(original) != common.digest(changed)
    changed["state"]["phi_V"][1] += 1e-6
    assert common.physical_preparation_identity(original) != common.physical_preparation_identity(changed)


def test_saved_dc_restore_checks_physics_without_running_a_new_dc_solve(common_state_16, monkeypatch):
    import copy
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_response as response
    stack, binding, prepared = common_state_16
    dc = response.solve_controlled_dc(stack, 16, binding, prepared)
    record = common.json_data(dc.evidence)
    monkeypatch.setattr(response, "solve_controlled_dc", lambda *a, **k: pytest.fail("new DC solve"))
    restored = response.restore_controlled_dc(record, stack, 16, binding, prepared)
    assert np.array_equal(restored.state.coordinate, dc.state.coordinate)
    changed = copy.deepcopy(record)
    changed["current_A_m2"][0] += 1.
    with pytest.raises(ValueError, match="content mismatch"):
        response.restore_controlled_dc(changed, stack, 16, binding, prepared)


def _reseal(record):
    record["sha256"] = common.digest({k: v for k, v in record.items() if k != "sha256"})
    return common.R1PreparedState.from_dict(record)


def _reseal_dc_coordinates(record):
    record["dc_state"]["state_sha256"] = combined._state_sha256(*(
        np.asarray(record["dc_state"][key], dtype=float)
        for key in (
            "electron_qf_increment_V", "hole_qf_increment_V", "positive_ion_density_m3"
        )
    ))


def test_approved_binding_uses_canonical_identity_not_file_formatting():
    stack = load_device_from_yaml(FIXTURE)
    reference = approved_r1_binding()
    canonical_sha256 = "0a6532a436dd07e6b27f01da3fc39aef906e6fd882057f0109c1bc5bdf77e39b"
    study = json.loads(binding_gate.STUDY_INPUT_PATH.read_text())
    assert study["fixed_reference_binding_sha256"] == reference["sha256"] == canonical_sha256
    raw = APPROVED_R1_BINDING_PATH.read_bytes()
    reformatted = json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == (
        "780e0275a3278e2d0ef2eb0ddf748ac763a7cc38ac0d03796312a952e0a01b4e"
    )
    assert hashlib.sha256(reformatted).digest() != hashlib.sha256(raw).digest()
    binding_gate.validate_r1_study_binding(json.loads(reformatted), stack)


@pytest.mark.parametrize("entrypoint", ["prepare", "restore"])
def test_resealed_alternate_reference_is_rejected_before_state_work(
    common_state, monkeypatch, entrypoint,
):
    stack, _, prepared = common_state
    alternate = alternate_r1_binding()
    # It is admissible to the generic reference validator: the failure must
    # identify this fixed study's approved input, not a malformed ladder.
    validate_binding(alternate, stack)

    def forbidden(*args, **kwargs):
        raise AssertionError("unapproved study reference reached numerical state work")

    monkeypatch.setattr(common, "solve_r1_dc", forbidden)
    monkeypatch.setattr(common, "_make_system", forbidden)
    with pytest.raises(ValueError, match="not the approved study reference"):
        if entrypoint == "prepare":
            common.prepare_common_state(stack, 4, alternate)
        else:
            # Model an externally supplied matching reference/state pair.
            # Admission must reject it before reconstructing any populations.
            external = prepared.to_dict()
            external["fixed_reference"] = alternate
            external["reference_sha256"] = alternate["sha256"]
            common.restore_common_state(_reseal(external), stack, 4, alternate)


def test_claiming_approved_hash_does_not_admit_changed_binding(common_state, monkeypatch):
    stack, approved, _ = common_state
    alternate = alternate_r1_binding()
    alternate["sha256"] = approved["sha256"]

    def forbidden(*args, **kwargs):
        raise AssertionError("forged canonical digest reached a DC solve")

    monkeypatch.setattr(common, "solve_r1_dc", forbidden)
    with pytest.raises(ValueError, match="reference identity mismatch"):
        common.prepare_common_state(stack, 4, alternate)


def test_prepared_source_identity_includes_repository_study_input(
    common_state, monkeypatch, tmp_path,
):
    original_source = common_state[2].to_dict()["source"]
    raw = binding_gate.STUDY_INPUT_PATH.read_bytes()
    assert original_source["study_input"] == {
        "path": binding_gate.STUDY_INPUT_RELATIVE_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    changed = tmp_path / "StudyInputV1.json"
    changed.write_bytes(raw + b"\n")
    changed_identity = {**original_source["study_input"], "sha256": hashlib.sha256(changed.read_bytes()).hexdigest()}
    monkeypatch.setattr(common, "study_input_identity", lambda: changed_identity)
    assert common.execution_source()["sha256"] != original_source["sha256"]
    with pytest.raises(common.R1StateError, match="identity mismatch: source"):
        _restore(common_state)


def test_relocated_study_input_is_rejected_before_identity_import(common_state, monkeypatch, tmp_path):
    relocated = tmp_path / "StudyInputV1.json"
    relocated.write_bytes(binding_gate.STUDY_INPUT_PATH.read_bytes())
    monkeypatch.setattr(binding_gate, "STUDY_INPUT_PATH", relocated)
    with pytest.raises(ValueError, match="tracked|checkout|policy"):
        _restore(common_state)


@pytest.mark.parametrize("label", ["A", "B", "C", "D"])
def test_controls_reuse_exact_common_populations_without_dc_solve(common_state, monkeypatch, label):
    def forbidden(*args, **kwargs):
        raise AssertionError("restoring a common state must not run another DC solve")

    monkeypatch.setattr(combined, "_solve_combined_dc", forbidden)
    monkeypatch.setattr(transient, "_solve_combined_dc", forbidden)
    monkeypatch.setattr(common, "solve_r1_dc", forbidden)
    system, state = _restore(common_state, label=label)
    saved = common_state[2].to_dict()

    assert system.controls.label == label
    for field, key in (
        ("n", "n_m3"), ("p", "p_m3"), ("positive", "positive_m3"),
        ("occupancy", "occupancy"), ("phi", "phi_V"),
        ("sheet_charge", "sheet_charge_C_m2"),
    ):
        np.testing.assert_array_equal(getattr(state, field), saved["state"][key])
    assert np.any(state.sheet_charge != 0.0)
    np.testing.assert_array_equal(system.equilibrium_occupancy, common_state[1]["f_ref"])
    np.testing.assert_array_equal(
        system.dark_reference.equilibrium_occupancy, common_state[1]["f_ref"]
    )
    np.testing.assert_allclose(
        state.sheet_charge,
        -Q * system.trap_density * (state.occupancy - system.equilibrium_occupancy),
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(system.qfn_reference, saved["qf_references_V"]["electron"])
    np.testing.assert_array_equal(system.qfp_reference, saved["qf_references_V"]["hole"])


def test_json_roundtrip_and_exports_do_not_mutate_prepared_state(common_state):
    prepared = common_state[2]
    original = prepared.to_dict()
    exported = prepared.to_dict()
    roundtrip = common.R1PreparedState.from_dict(exported)
    assert roundtrip.canonical_json == prepared.canonical_json
    assert roundtrip.sha256 == prepared.sha256

    exported["state"]["positive_m3"][1] *= 1.1
    exported["fixed_reference"]["f_ref"][0] = 0.5
    assert prepared.to_dict() == original
    assert roundtrip.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        prepared.canonical_json = "{}"

    system, first = _restore(common_state, roundtrip)
    _other_system, second = _restore(common_state, prepared.to_dict())
    assert common.snapshot(system, first) == common.snapshot(system, second)
    assert not np.shares_memory(first.positive, second.positive)
    if first.positive.flags.writeable:
        first.positive[1] *= 1.1
    else:
        with pytest.raises(ValueError):
            first.positive[1] *= 1.1
    np.testing.assert_array_equal(second.positive, original["state"]["positive_m3"])
    assert prepared.to_dict() == original


def test_unsealed_content_change_is_rejected_before_import(common_state):
    record = common_state[2].to_dict()
    record["state"]["occupancy"][0] *= 1.01
    with pytest.raises(common.R1StateError, match="content hash mismatch"):
        common.R1PreparedState.from_dict(record)


@pytest.mark.parametrize("field", [
    "intervals", "grid", "physical_stack", "stack_sha256", "fixed_reference",
    "reference_sha256", "source", "study_spec_sha256", "preparation_controls",
])
def test_resealed_identity_changes_are_rejected(common_state, field):
    record = common_state[2].to_dict()
    if field == "intervals":
        record[field] = 16
    elif field == "grid":
        record[field]["widths_m"][0] *= 2.0
    elif field == "physical_stack":
        record[field] = {}
    elif field == "fixed_reference":
        record[field]["f_ref"][0] *= 1.01
    elif field == "source":
        files = record[field]["files"]
        files[next(iter(files))] = "0" * 64
        record[field]["sha256"] = common.digest(files)
    elif field == "preparation_controls":
        record[field] = {"nu_I": 0, "nu_t": 0}
    else:
        record[field] = "0" * 64
    with pytest.raises(common.R1StateError, match=field):
        _restore(common_state, _reseal(record))


def test_prepared_grid_cannot_be_restored_on_another_grid(common_state):
    stack, reference, prepared = common_state
    with pytest.raises(common.R1StateError, match="identity mismatch"):
        common.restore_common_state(prepared, stack, 16, reference)


@pytest.mark.parametrize("field,value", [
    ("kind", "finite_time_preparation"),
    ("voltage_V", 0.005),
    ("temperature_K", 350.0),
    ("illuminated", True),
    ("duration_s", 10.0),
    ("duration_reason", "finite-time state"),
    ("unrecorded_waveform", [0.0, 0.005]),
])
def test_resealed_preparation_history_change_is_rejected(common_state, field, value):
    record = common_state[2].to_dict()
    record["history"][field] = value
    with pytest.raises(common.R1StateError, match="identity mismatch: history"):
        _restore(common_state, _reseal(record))


@pytest.mark.parametrize("history", [None, {}, [], "dark equilibrium"])
def test_missing_or_malformed_preparation_history_is_rejected(common_state, history):
    record = common_state[2].to_dict()
    if history is None:
        del record["history"]
    else:
        record["history"] = history
    with pytest.raises(common.R1StateError, match="identity mismatch: history"):
        _restore(common_state, _reseal(record))


@pytest.mark.parametrize("field", [
    "n_m3", "p_m3", "positive_m3", "occupancy", "phi_V", "sheet_charge_C_m2",
    "trace_potential_V", "trace_state_m3", "capture_m2_s", "positive_inventory_m2",
    "poisson_residual_C_m2", "local_residual",
])
def test_resealed_physical_snapshot_changes_are_rejected(common_state, field):
    record = common_state[2].to_dict()
    values = np.asarray(record["state"][field], dtype=float)
    assert values.size > 0
    values.flat[0] += max(abs(values.flat[0]) * 0.01, 1.0e-6)
    record["state"][field] = values.tolist()
    with pytest.raises(common.R1StateError, match="physical array mismatch"):
        _restore(common_state, _reseal(record))


def test_resealed_qf_reference_change_is_rejected(common_state):
    record = common_state[2].to_dict()
    record["qf_references_V"]["electron"][1] += 0.01
    with pytest.raises(common.R1StateError, match="QF reference mismatch"):
        _restore(common_state, _reseal(record))


def test_resealed_record_with_invalid_nested_dc_hash_is_rejected(common_state):
    record = common_state[2].to_dict()
    record["dc_state"]["state_sha256"] = "0" * 64
    with pytest.raises(common.R1StateError, match="DC coordinate hash mismatch"):
        _restore(common_state, _reseal(record))


@pytest.mark.parametrize("field,index,increment", [
    ("electron_density_m3", 1, 1.0e9),
    ("hole_density_m3", 1, 1.0e9),
    ("positive_ion_density_m3", 1, 1.0e9),
    ("potential_V", 1, 1.0e-6),
    ("interface_occupancy", 0, 1.0e-6),
])
def test_dc_scaling_reference_cannot_differ_from_canonical_state(
    common_state_16, field, index, increment
):
    stack, reference, prepared = common_state_16
    record = prepared.to_dict()
    # On N16, node 1 is interior and does not touch the interface. Inconsistent
    # reference densities here must not disappear through QF reconstruction.
    _grid, material = build_r1_material(stack, 16)
    assert 1 not in material.iface_qss_left_nodes
    assert 1 not in material.iface_qss_right_nodes
    record["dc_state"][field][index] += increment
    _reseal_dc_coordinates(record)
    with pytest.raises(common.R1StateError, match="DC physical array mismatch"):
        common.restore_common_state(_reseal(record), stack, 16, reference)


@pytest.mark.parametrize("species,block", [("electron", 0), ("hole", 1)])
@pytest.mark.parametrize("gate", ["integrated", "normalized"])
def test_live_carrier_checks_keep_integrated_and_normalized_dc_gates(
    common_state_16, monkeypatch, species, block, gate
):
    stack, reference, prepared = common_state_16
    system, state = common.restore_common_state(prepared, stack, 16, reference)
    policy = transient.InterfaceDefectIonTransientPolicy()
    # A non-unit current scale makes omitting the normalization observable.
    current_scale = 7.0
    monkeypatch.setattr(system.system, "current_scale", current_scale)
    contribution = (
        0.6 * policy.maximum_dc_continuity_bound_A_m2
        if gate == "integrated"
        else 2.0 * current_scale * policy.maximum_dc_normalized_residual
    )
    rate = state.rate.copy()
    rate[:2 * system.interior_count] = 0.0
    start = block * system.interior_count
    rate[start:start + 2] = np.array([contribution, -contribution]) / (
        Q * system.widths[1:3]
    )
    checks = common.equilibrium_checks(system, replace(state, rate=rate), policy)
    integrated = f"{species}_continuity_A_m2"
    normalized = f"{species}_normalized_residual"
    assert checks["metrics"][integrated] == pytest.approx(2.0 * contribution)
    assert checks["metrics"][normalized] == pytest.approx(contribution / current_scale)
    assert not checks["certified"]
    if gate == "integrated":
        assert contribution < policy.maximum_dc_continuity_bound_A_m2
        assert integrated in checks["reasons"]
    else:
        assert 2.0 * contribution < policy.maximum_dc_continuity_bound_A_m2
        assert integrated not in checks["reasons"]
        assert normalized in checks["reasons"]


def test_forged_dc_pass_flag_does_not_replace_live_equations(common_state):
    stack, reference, prepared = common_state
    record = prepared.to_dict()
    record["dc_state"]["positive_ion_density_m3"][1] *= 1.01
    _reseal_dc_coordinates(record)
    record["dc_state"]["certificate"]["certified"] = True
    record["dc_state"]["certificate"]["reasons"] = []
    record["preparation_checks"]["certified"] = True
    record["preparation_checks"]["reasons"] = []
    # Build a matching snapshot of the forged populations. This prevents a
    # duplicate-array mismatch from hiding whether live equations are checked.
    grid, material = build_r1_material(stack, 4)
    dc = common._decode_dc(record["dc_state"])
    policy = transient.InterfaceDefectIonTransientPolicy()
    system = common._make_system(
        stack, grid, material, dc, reference, R1DynamicsControls(), policy
    )
    altered = system.evaluate(system.initial_coordinate(), 0.0)
    record["state"] = common.snapshot(system, altered)
    with pytest.raises(common.R1StateError, match="live D equations"):
        _restore(common_state, _reseal(record))
