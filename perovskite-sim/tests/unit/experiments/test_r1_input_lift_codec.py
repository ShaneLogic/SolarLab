"""Checkpoint/dataflow tests on four nodes; no DC or nonlinear solve."""
from dataclasses import replace
import copy
import hashlib
import json

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift as lift
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift_codec as codec
from perovskite_sim.physics.compensated import DD
from tests.unit.experiments.test_r1_input_lift import synthetic


@pytest.fixture
def accepted(synthetic, monkeypatch):
    baseline, template, working, _, evaluate = synthetic
    coordinate = np.zeros(15)
    coordinate[[0, 2, 5, 7, 9, 10, 11, 12, 13, 14]] = 2.**-59
    state, _, _ = evaluate(coordinate)

    def same_state_evaluate(self, coordinate, voltage):
        # The synthetic geometry omits the full sparse assembly.  Use the
        # production DD hooks for every saved primary/rate/current/local field.
        try:
            qn, qp, phi, n, p, occupancy, trace, logs = self._coordinates(coordinate, voltage)
            positive, _ = self._ion_coordinates(coordinate)
            local, aggregate = self._local_states(n, p, phi, occupancy, trace, logs,
                trace_density_m3=self._trace_density_coordinates(coordinate))
            source = self._source(n, p, phi, voltage, aggregate)
            tn, tp, jn, jp = self._currents(qn, qp, phi, n, p, local)
            self._carrier_rate_fields(source, tn, tp, local)
            irate, _, iflux, _ = self._ion_fields(phi, positive, None)
            skeleton = replace(template, coordinate=coordinate.copy(), dqfn=qn, dqfp=qp,
                phi=phi, n=n, p=p, occupancy=occupancy, local=local, positive=positive,
                positive_rate=irate, positive_flux=iflux, current_n=jn, current_p=jp,
                positive_current=Q*iflux, carrier_conduction=jn+jp, conduction=jn+jp+Q*iflux)
            return self._with_step_electrostatics(skeleton)
        finally:
            self._input_lift_work = None

    monkeypatch.setattr(lift.RebasedInputLiftR1System, "evaluate", same_state_evaluate)
    return baseline, template, working, state


def _rehash(record):
    record["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in record.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return record


def _record(accepted):
    _, _, system, state = accepted
    return codec.save_checkpoint(system, state, time_s=.25, voltage_V=0.)


def test_checkpoint_json_roundtrip_preserves_all_words_and_reconstructs_payload(accepted):
    baseline, template, _, original = accepted
    encoded = json.loads(json.dumps(_record(accepted), allow_nan=False))
    restored_system, restored = codec.restore_checkpoint(baseline, template, encoded)
    assert np.any(restored.input_lift["n_m3"].lo != 0.)
    for name, value in original.input_lift.items():
        assert value.hi.tobytes() == restored.input_lift[name].hi.tobytes()
        assert value.lo.tobytes() == restored.input_lift[name].lo.tobytes()
    assert restored.local[0] is not template.local[0]
    np.testing.assert_array_equal(restored.local[0].carrier_data.inputs.state_density.lo,
                                  restored.input_lift["trace_state_m3"].lo[0])
    assert restored_system._step_reference is restored
    assert np.count_nonzero(restored.coordinate) == 0
    assert restored.coordinate_reference_identity != original.coordinate_reference_identity
    receipt = restored_system.input_lift_restore_receipt
    assert receipt["nonlinear_solves"] == receipt["time_advances"] == 0
    assert receipt["algebraic_evaluations"] == 1
    assert receipt["all_saved_words_exact"] is receipt["derived_reconstruction_exact"] is True
    assert baseline._step_reference is template
    assert template.local[0].tangent.balance.capture_flux_m2_s.tolist() == [0.]*4


def test_snapshot_import_declares_missing_historical_coordinate(accepted):
    baseline, template, system, state = accepted
    snapshot = lift.snapshot(system, state)
    restored_system, restored = codec.import_snapshot(baseline, template, snapshot, time_s=.25, voltage_V=0.)
    assert restored_system.input_lift_restore_receipt["historical_coordinate_available"] is False
    assert restored_system.input_lift_restore_receipt["checkpoint_sha256"] is None
    assert restored_system.input_lift_restore_receipt["prior_coordinate_reference_identity"] == state.coordinate_reference_identity
    assert restored_system.input_lift_restore_receipt["coordinate_role"] == codec.COORDINATE_ROLE
    np.testing.assert_array_equal(restored.coordinate, 0.)
    for name in state.input_lift:
        np.testing.assert_array_equal(restored.input_lift[name].lo, state.input_lift[name].lo)


def test_restored_and_uninterrupted_next_evaluation_are_identical(accepted):
    baseline, template, system, state = accepted
    uninterrupted, _ = lift.from_accepted_state(system, state)
    restored_system, _ = codec.restore_checkpoint(baseline, template, _record(accepted))
    increment = np.zeros(15)
    increment[[1, 3, 6, 8, 9, 10, 11, 12, 13, 14]] = -2.**-60
    uninterrupted_state = uninterrupted.evaluate(increment, 0.)
    resumed_state = restored_system.evaluate(increment, 0.)
    assert lift._words(uninterrupted_state.input_lift) == lift._words(resumed_state.input_lift)


def test_corrupted_checkpoint_hash_rejected_before_evaluation(accepted, monkeypatch):
    baseline, template, _, _ = accepted
    record = _record(accepted)
    record["snapshot"]["represented_fields"]["n_m3"]["lo"][1] += 1e-20
    monkeypatch.setattr(lift.RebasedInputLiftR1System, "evaluate", lambda *a: pytest.fail("corrupt checkpoint evaluated"))
    with pytest.raises(codec.InputLiftCheckpointError, match="SHA256"):
        codec.restore_checkpoint(baseline, template, record)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(schema="float64-pair-v1"), "schema"),
    (lambda r: r.update(state_identity="0"*64), "identity"),
    (lambda r: r.update(new_reference_identity="0"*64), "identity"),
    (lambda r: r["snapshot"].update(coordinate_reference_identity="unknown"), "reference identity"),
    (lambda r: r["snapshot"]["represented_fields"]["n_m3"].update(lo=[0.]), "shape"),
    (lambda r: r["snapshot"]["represented_fields"]["n_m3"].update(lo=[1.]*4), "normalized"),
    (lambda r: r["snapshot"]["high_word_state"]["n_m3"].__setitem__(1, 99.), "high view"),
    (lambda r: r["snapshot"]["represented_fields"].pop("rate"), "coverage"),
    (lambda r: r.update(source_coordinate=[0.]), "shape"),
    (lambda r: r.update(time_s=-1.), "domain"),
])
def test_rehashed_schema_shape_and_identity_corruption_is_rejected(accepted, mutation, match):
    baseline, template, _, _ = accepted
    record = _record(accepted)
    mutation(record)
    with pytest.raises(codec.InputLiftCheckpointError, match=match):
        codec.restore_checkpoint(baseline, template, _rehash(record))


def test_nonfinite_or_string_words_rejected(accepted):
    baseline, template, system, state = accepted
    for invalid in (float("nan"), float("inf"), "0.0", True):
        record = lift.snapshot(system, state)
        record["represented_fields"]["rate"]["lo"] = [invalid] * len(state.rate)
        with pytest.raises(codec.InputLiftCheckpointError):
            codec.import_snapshot(baseline, template, record, time_s=.25, voltage_V=0.)


def test_rehashed_derived_rate_tampering_is_rejected_by_reconstruction(accepted):
    baseline, template, system, state = accepted
    snapshot = lift.snapshot(system, state)
    snapshot["represented_fields"]["rate"]["hi"][0] *= 2.
    snapshot["represented_fields"]["rate"]["lo"][0] = 0.
    with pytest.raises(codec.InputLiftCheckpointError, match="reconstruction.*rate"):
        codec.import_snapshot(baseline, template, snapshot, time_s=.25, voltage_V=0.)


def test_physical_coefficient_mismatch_rejected(accepted):
    baseline, template, _, _ = accepted
    record = _record(accepted)
    baseline.material = copy.copy(baseline.material)
    baseline.material.D_ion_face = 2 * baseline.material.D_ion_face
    with pytest.raises(codec.InputLiftCheckpointError, match="reconstruction|physical model"):
        codec.restore_checkpoint(baseline, template, record)


def test_saved_residual_replay_matches_word_equation_without_evaluate(accepted, monkeypatch):
    baseline, template, system, state = accepted
    restored_system, restored = codec.restore_checkpoint(baseline, template, _record(accepted))
    before = system.evaluate(np.zeros(system.dimension), 0.)
    monkeypatch.setattr(lift.RebasedInputLiftR1System, "evaluate", lambda *a: pytest.fail("replay evaluated the operator"))
    result = codec.replay_saved_step(restored_system, before, restored, dt_s=.01,
        storage_scale=np.ones(state.storage.shape), poisson_scale=np.ones(state.poisson_residual.shape),
        local_scale=np.ones(state.local_residual.shape))
    expected = lift.cat(state.input_lift["storage"]-before.input_lift["storage"]-DD(.01)*state.input_lift["rate"],
                        state.input_lift["poisson_residual_C_m2"], state.input_lift["local_residual"])
    np.testing.assert_array_equal(result["scaled_residual_vector"], expected.hi)
    assert result["nonlinear_solves"] == result["time_advances"] == 0
    assert result["independent_physics"]["operator_representation"] == lift.REPRESENTATION


def test_checkpoint_values_do_not_alias_state_and_restore_is_immutable(accepted):
    baseline, template, _, state = accepted
    record = _record(accepted)
    restored_system, restored = codec.restore_checkpoint(baseline, template, record)
    record["snapshot"]["represented_fields"]["phi_V"]["lo"][1] = 123.
    assert state.input_lift["phi_V"].lo[1] != 123.
    assert restored.input_lift["phi_V"].lo[1] != 123.
    with pytest.raises(ValueError):
        restored.input_lift["phi_V"].lo[1] = 0.
    with pytest.raises(TypeError):
        restored_system.input_lift_restore_receipt["all_saved_words_exact"] = False
