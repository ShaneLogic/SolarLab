"""Format/helper checks; actual physical rejection has its own integration run."""
import copy

import pytest

from scripts.check_r1_v8_replay import (
    LOW_FIELD, bounded_prefix, expect_reconstruction_rejection, mutate_preparation,
    mutate_trajectory, require_sealed, reseal, save_case,
)
from scripts.check_r1_v8_legacy import predecessor_contract, stale_predecessor


def snapshot():
    return {"phi_V": [0., .1, .2], "precision_phi_V_hi": [0., .1, .2],
            LOW_FIELD: [0., 1e-20, 0.], "n_m3": [1., 2., 3.], "p_m3": [3., 2., 1.],
            "positive_m3": [1., 1., 1.], "occupancy": [.5], "sheet_charge_C_m2": [0.]}


def record():
    rows = [{"substeps": 1, "time_s": float(i), "phase": "0+" if i == 0 else "accepted_regular_step",
             "state": snapshot(), "physics_reconstruction": {"coordinate": [float(i)]}}
            for i in range(4)]
    rows[1]["state"][LOW_FIELD][1] = 2e-20
    return reseal({"accepted_steps": rows, "scope": "test", "times_s": [0., 1., 2., 3.],
                  "source": {"bytes": "unchanged"}, "output_states": {}, "regular_currents": {},
                  "charge_integral": {}, "finite_step_averages": {}, "certificate": {"certified": True}})


def test_prefix_retains_actual_coordinates_and_request_without_claiming_completion():
    original = record()
    prefix = bounded_prefix(original)
    assert len(prefix["accepted_steps"]) == 2
    assert prefix["accepted_steps"] == original["accepted_steps"][:2]
    assert prefix["times_s"] == original["times_s"]
    assert prefix["certificate"]["certified"] is False
    assert "output_states" not in prefix
    require_sealed(prefix)


def test_real_low_word_mutation_preserves_high_and_synchronizes_all_snapshot_copies():
    prefix = bounded_prefix(record())
    prefix["same_direct_copy"] = copy.deepcopy(prefix["accepted_steps"][1]["state"])
    prefix["independent_eliminated"] = copy.deepcopy(prefix["accepted_steps"][1]["state"])
    reseal(prefix)
    before = copy.deepcopy(prefix)
    changed, evidence = mutate_trajectory(prefix)
    assert prefix == before
    old = prefix["accepted_steps"][1]["state"]
    new = changed["accepted_steps"][1]["state"]
    assert new[LOW_FIELD][1] != old[LOW_FIELD][1]
    assert new["precision_phi_V_hi"] == old["precision_phi_V_hi"]
    assert new["phi_V"] == old["phi_V"]
    assert changed["same_direct_copy"] == new
    assert changed["independent_eliminated"] == prefix["independent_eliminated"]
    assert changed["accepted_state_arrays"][LOW_FIELD][1] == new[LOW_FIELD]
    assert evidence["coordinate_unchanged"] and evidence["source_unchanged"]
    require_sealed(changed)


def test_preparation_low_word_mutation_preserves_actual_seed_and_source():
    source = reseal({"state": snapshot(), "seed_preparation": {"physical": "legacy"}, "source": {"same": True}})
    changed, evidence = mutate_preparation(source)
    assert changed["sha256"] != source["sha256"]
    assert evidence["seed_unchanged"] and evidence["source_unchanged"]
    require_sealed(changed)


def test_saved_negative_case_reseals_and_reads_all_duplicate_files(tmp_path):
    prepared = reseal({"state": snapshot(), "seed_preparation": {}, "source": {}})
    changed, evidence = mutate_trajectory(bounded_prefix(record()))
    saved_prepared, saved_changed = save_case(tmp_path / "case", prepared, changed, evidence)
    assert saved_prepared == prepared
    assert saved_changed == changed


@pytest.mark.parametrize("message", ["common-state content hash mismatch", "physical reconstruction mismatch: accepted state arrays"])
def test_hash_or_duplicate_rejection_never_fulfills_physical_tamper_requirement(message):
    def reject():
        raise ValueError(message)
    result = expect_reconstruction_rejection(reject, "physical reconstruction mismatch: accepted state 1")
    assert result["hash_or_duplicate_rejection"]
    assert not result["required_physical_rejection"]


def test_fixed_predecessor_is_same_shape_tier_and_earlier_actual_state():
    value = predecessor_contract(record()["accepted_steps"])
    assert value["shape_and_tier_preserved"]
    assert (value["wrong_previous_row"], value["correct_previous_row"], value["target_row"]) == (0, 1, 2)


def test_stale_predecessor_enters_actual_rebase_once_and_restores_inheritance():
    class Parent:
        def rebase(self, previous):
            return {"evaluated_reference": previous}
    class System(Parent):
        pass
    system = System()
    with pytest.raises(RuntimeError):
        with stale_predecessor(System, original_rebase=System.rebase,
                snapshot=lambda system, state: state, expected={"id": 2}, replacement={"id": 1}) as events:
            assert system.rebase({"id": 2}) == {"evaluated_reference": {"id": 1}}
            assert system.rebase({"id": 2}) == {"evaluated_reference": {"id": 2}}
            assert len(events) == 1
            raise RuntimeError("restoration")
    assert "rebase" not in System.__dict__
    assert system.rebase({"id": 2}) == {"evaluated_reference": {"id": 2}}


def test_fixed_mutation_fails_closed_if_it_would_change_high_precision_projection():
    prepared = reseal({"state": snapshot(), "seed_preparation": {}, "source": {}})
    prepared["state"]["precision_phi_V_hi"][1] = 0.
    prepared["state"][LOW_FIELD][1] = 0.
    with pytest.raises(ValueError, match="binary64-invisible"):
        mutate_preparation(prepared)
