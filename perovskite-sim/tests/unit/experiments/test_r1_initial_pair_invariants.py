"""Low-word losses must not pass the ideal-step population/rebase guards."""

import numpy as np
import pytest

from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientError,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import (
    _check_pair_word_identity,
    _pair_word_snapshot,
)
from perovskite_sim.physics.compensated import DD


FIXED_FIELDS = ("n_m3", "p_m3", "positive_m3", "occupancy", "sheet_charge_C_m2")
REBASE_FIELDS = (
    "phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3",
    "occupancy", "trace_potential_V", "trace_state_m3", "sheet_charge_C_m2",
)


def _fine_words():
    # Every low word is below half an ulp: rounding to public arrays hides it.
    return {field: DD([1., 2.], [2.**-56, -2.**-55]) for field in REBASE_FIELDS}


@pytest.mark.parametrize("field", FIXED_FIELDS)
def test_fixed_population_guard_rejects_low_word_loss_with_identical_highs(field):
    fine = _fine_words()
    before = _pair_word_snapshot(fine)
    fine[field] = DD(fine[field].hi)
    np.testing.assert_array_equal(before[field][0], fine[field].hi)
    with pytest.raises(InterfaceDefectIonTransientError) as failure:
        _check_pair_word_identity(before, fine, stage="fixed_population_voltage_jump",
                                  fields=FIXED_FIELDS)
    assert failure.value.result["field"] == field
    assert failure.value.result["word"] == "lo"


@pytest.mark.parametrize("field", REBASE_FIELDS)
def test_rebase_guard_preserves_every_primary_pair_and_sheet_word(field):
    fine = _fine_words()
    before = _pair_word_snapshot(fine)
    # Replace a value in the original mapping after its snapshot was taken.
    # Keeping the mapping by reference would make this mutation undetectable.
    fine[field] = DD(fine[field].hi)
    with pytest.raises(InterfaceDefectIonTransientError) as failure:
        _check_pair_word_identity(before, fine, stage="zero_plus_zero_coordinate_restore")
    assert failure.value.result["field"] == field
    assert failure.value.result["word"] == "lo"


def test_event_guard_allows_algebraic_qf_and_trace_changes_but_rebase_does_not():
    fine = _fine_words()
    before = _pair_word_snapshot(fine)
    for field in ("phi_V", "dqfn_V", "dqfp_V", "trace_potential_V", "trace_state_m3"):
        fine[field] = fine[field] + .25
    _check_pair_word_identity(before, fine, stage="fixed_population_voltage_jump",
                              fields=FIXED_FIELDS)
    with pytest.raises(InterfaceDefectIonTransientError, match="phi_V.hi"):
        _check_pair_word_identity(before, fine, stage="zero_plus_rebase")


@pytest.mark.parametrize("fault", ("missing_field", "changed_shape"))
def test_pair_guard_does_not_accept_incomplete_or_reshaped_words(fault):
    fine = _fine_words()
    before = _pair_word_snapshot(fine)
    if fault == "missing_field":
        del fine["phi_V"]
    else:
        fine["phi_V"] = fine["phi_V"].reshape(1, 2)
    with pytest.raises(InterfaceDefectIonTransientError, match="phi_V.hi"):
        _check_pair_word_identity(before, fine, stage="zero_minus_rebase")
