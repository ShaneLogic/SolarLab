"""Explicit production backend and real bounded A-D equation replay."""
from dataclasses import FrozenInstanceError
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import interface_defect_transient as transient
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as replay
from perovskite_sim.experiments import one_dimensional_mechanism_r1_step as step
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
from perovskite_sim.experiments.one_dimensional_mechanism_r1_replay_receipt import R1PhysicsReplayReceipt
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.compensated import DD


@pytest.fixture(scope="module")
def pair_preparation():
    project = Path(__file__).resolve().parents[2]
    stack = load_device_from_yaml(project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    with threadpool_limits(1):
        prepared = states.prepare_common_state(stack, 16, binding, backend="pair")
    return stack, binding, prepared


def _module_functions():
    return (states.prepare_common_state, states.verify_prepared_physics, states.snapshot,
            transient._solve_step, step.build_initial_step, step._solve_local_carriers,
            replay.independent_physics_row, protocol.build_initial_step)


def test_backend_is_immutable_and_does_not_replace_modules(pair_preparation, monkeypatch):
    stack, binding, prepared = pair_preparation
    aliases = _module_functions()
    def forbidden(*args, **kwargs):
        raise AssertionError("production cannot enter the historical precision context")
    monkeypatch.setattr(precision, "precision_context", forbidden)
    with pytest.raises(FrozenInstanceError):
        get_backend("pair").representation_id = "float64-baseline"
    with threadpool_limits(1):
        legacy, old = states.verify_prepared_physics(prepared.to_dict()["seed_preparation"], stack, binding)
        before = states.snapshot(legacy, old)
        fine, actual = states.verify_prepared_physics(prepared, stack, binding, backend="pair")
        assert not hasattr(old, "fine") and hasattr(actual, "fine")
        assert states.snapshot(legacy, legacy.evaluate(legacy.initial_coordinate(), 0.)) == before
        late = importlib.import_module("perovskite_sim.experiments.one_dimensional_mechanism_r1_response")
        assert late.snapshot is states.snapshot
        assert late.restore_common_state is states.restore_common_state
        with pytest.raises(ValueError):
            states.verify_prepared_physics(prepared, stack, binding)
        with pytest.raises(ValueError):
            states.snapshot(fine, actual, backend="legacy")
    assert _module_functions() == aliases


def test_zero_excitation_retains_every_control_and_shared_fine_inputs(pair_preparation):
    stack, binding, prepared = pair_preparation
    with threadpool_limits(1):
        record = protocol.check_zero_excitation(stack, 16, binding, prepared, backend="pair")
    assert record["schema"] == "R1ZeroExcitationV2"
    assert record["certificate"]["certified"]
    assert set(record["controls"]) == set("ABCD")
    for control in record["controls"].values():
        for field in precision.REFERENCE_FIELDS:
            for word in ("hi", "lo"):
                key = "precision_" + field + "_" + word
                assert control["population_identity"][key] == prepared.to_dict()["state"][key]


@pytest.mark.parametrize("control", tuple("ABCD"))
def test_actual_short_pair_trajectory_and_replay_receipt(pair_preparation, control):
    stack, binding, prepared = pair_preparation
    aliases = _module_functions()
    timings = []
    with threadpool_limits(1):
        record = protocol.run_r1_step(stack, 16, binding, prepared, control=control,
            times_s=[0., 1e-9], backend="pair")
        verified = replay.verify_r1_step_physics(stack, 16, binding, prepared, record,
            backend="pair", row_observer=timings.append)
    export = os.environ.get("R1_V9_CORE_SHORT_OUTPUT")
    if control == "D" and export:
        from shutil import copyfile
        import scipy
        output = Path(export)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("PreparedV1.json", prepared.to_dict()), ("ResultV1.json", record),
            ("PhysicsReplayV1.json", verified), ("ReplayReceiptV1.json", verified.replay_receipt.to_dict()),
            ("BindingV1.json", binding), ("ReplayTimingsV1.json", timings),
            ("RuntimeV1.json", {"numpy": np.__version__, "scipy": scipy.__version__,
                                "source": states.execution_source(), "scope": "bounded_development_short_chain"}),
        ):
            (output / name).write_text(json.dumps(states.json_data(value), sort_keys=True, allow_nan=False) + "\n")
        copyfile(Path(__file__).resolve().parents[2] /
            "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
            output / "FixtureV1.yaml")
    assert record["schema"] == "R1ControlledStepV2"
    assert record["certificate"]["certified"] and verified["certified"]
    assert len(record["accepted_steps"]) == len(timings) == 10
    assert len(record["output_states"]) == 40
    assert all(value["elapsed_s"] >= 0. for value in timings)
    assert isinstance(verified.replay_receipt, R1PhysicsReplayReceipt)
    ledger = verified.replay_receipt.to_dict()
    assert ledger["saved_rows_digest"] == states.digest(record["accepted_steps"])
    assert ledger["actual_recomputed_rows_digest"] == ledger["saved_rows_digest"]
    assert ledger["result_sha256"] == record["sha256"]
    for row, binding_row in zip(record["accepted_steps"], ledger["row_bindings"]):
        eliminated = row["physics_reconstruction"]["eliminated_precision"]
        assert binding_row["record_digest"] == states.digest(eliminated)
        assert binding_row["fixed_input_digest"] == states.digest(eliminated["shared_inputs"]["fields"])
        assert binding_row["ion_input_digest"] == states.digest(eliminated["ion_inputs"])
        assert eliminated["fields"]["phi_V"] == eliminated["fields"]["constraint_phi_V"]
    assert _module_functions() == aliases


def test_failed_pair_observer_preserves_schema_and_module_identities(pair_preparation):
    stack, binding, prepared = pair_preparation
    aliases = _module_functions()
    def fail(_row):
        raise RuntimeError("deliberate persistence failure")
    with threadpool_limits(1), pytest.raises(protocol.R1RunError) as error:
        protocol.run_r1_step(stack, 16, binding, prepared, times_s=[0., 1e-9],
            backend="pair", accepted_step_observer=fail)
    record = error.value.result
    assert record["schema"] == "R1ControlledStepV2"
    assert len(record["accepted_steps"]) == 1
    assert record["certificate"]["certified"] is False
    assert record["sha256"] == states.digest({k: v for k, v in record.items() if k != "sha256"})
    assert _module_functions() == aliases


def test_pair_failed_newton_witness_replays_without_integrating(pair_preparation, monkeypatch):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
    stack, binding, prepared = pair_preparation
    def fail_at_actual_iterate(system, coordinate, previous, voltage, dt, policy, *,
                               check_jacobian, scaling_system=None):
        scaling = system if scaling_system is None else scaling_system
        storage = scaling.storage_scale(previous.storage, previous, dt, policy)
        poisson, local = scaling.poisson_scale(policy), scaling.local_algebraic_scale(policy)
        residual, _, state = system.residual_and_jacobian(
            coordinate, voltage, previous, dt, storage, poisson, local)
        currents = system.transient_current_metrics(state, previous, dt)
        diagnostics = {"iteration": 0, "scaled_nonlinear_residual": float(np.max(np.abs(residual))),
            "charge_balance_relative": system.charge_balance_metrics(state, previous, dt)[1],
            "solver_current_spread_relative": currents[4], "interface_current_spread_relative": currents[5],
            "linear_backward_error": 0.}
        error = transient.InterfaceDefectTransientError("deliberate failed Newton witness")
        error.result = system.failure_evidence(state, previous, voltage, dt, residual,
                                               storage, poisson, local, diagnostics)
        raise error
    with threadpool_limits(1):
        with monkeypatch.context() as patch:
            patch.setattr(transient, "_solve_step", fail_at_actual_iterate)
            with pytest.raises(protocol.R1RunError) as failure:
                protocol.run_r1_step(stack, 16, binding, prepared, times_s=[0., 1e-9], backend="pair")
        record = failure.value.result
        def forbid_integration(*args, **kwargs):
            raise AssertionError("failed-state reconstruction cannot integrate a new trajectory")
        monkeypatch.setattr(transient, "_solve_step", forbid_integration)
        monkeypatch.setattr(protocol, "_integrate_trace", forbid_integration)
        prefix = replay.verify_r1_step_physics(stack, 16, binding, prepared, record,
                                               allow_incomplete=True, backend="pair")
        terminal = rebuild_failure_witness(stack, 16, binding, prepared, record, backend="pair")
    assert len(record["accepted_steps"]) == 1
    assert prefix["evidence_matches_equations"] and not prefix["certified"]
    assert terminal["available"] and terminal["terminal_state_physics_recomputed"]
    assert terminal["accepted"] is False


def test_independent_constraint_solver_cannot_read_direct_outputs(pair_preparation, monkeypatch):
    stack, binding, prepared = pair_preparation
    with threadpool_limits(1):
        system, initial = states.verify_prepared_physics(prepared, stack, binding, backend="pair")
        fixed = {key: initial.fine[key] for key in precision._INDEPENDENT_FIXED_FIELDS}
        inputs = system.independent_poisson_inputs(fixed, 0.)
        actual, evidence = precision.solve_independent_poisson(inputs)
        with pytest.raises(FrozenInstanceError):
            inputs.fixed_inputs_json = "{}"
        with pytest.raises(ValueError, match="rejects direct outputs"):
            system.independent_poisson_inputs({**fixed, "phi_V": initial.fine["phi_V"]}, 0.)
        def forbidden(*args, **kwargs):
            raise AssertionError("independent constraint read a production method")
        monkeypatch.setattr(system, "_poisson_pair", forbidden)
        monkeypatch.setattr(system, "_refresh_fine_constants", forbidden)
        system._fine_evaluation_cache = {"phi_V": np.full(system.node_count, np.nan)}
        system._poisson_laplacian = None
        initial.fine["phi_V"] = DD(np.full(system.node_count, 123.))
        initial.fine["poisson_residual_C_m2"] = DD(np.full(system.node_count - 2, 999.))
        initial.fine["positive_flux_m2_s"] = DD(np.full(system.node_count - 1, 456.))
        initial.phi[:] = 123.
        initial.positive_flux[:] = 456.
        # Build another input bundle after corrupting every prohibited source.
        # A cached already-built DTO alone would not exercise the adapter.
        second_inputs = system.independent_poisson_inputs(fixed, 0.)
        assert second_inputs == inputs
        again, second = precision.solve_independent_poisson(second_inputs)
    assert second == evidence
    for key in actual:
        assert precision.pair_words(again[key]) == precision.pair_words(actual[key])
    assert second["last_correction_max_abs_V"] < 1e-28
    assert second["iterations"] <= 12
    assert set(inputs.to_dict()["fixed_inputs"]) == set(precision._INDEPENDENT_FIXED_FIELDS)
    with pytest.raises(TypeError):
        R1PhysicsReplayReceipt({"claimed": "JSON alone cannot issue a receipt"})
