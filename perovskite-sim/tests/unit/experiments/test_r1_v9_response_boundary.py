"""Pair response paths still reject incomplete or implicit precision input."""
from types import SimpleNamespace

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_response as response
from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import PAIR


@pytest.mark.parametrize("entry", [response.solve_controlled_dc, response.restore_controlled_dc,
    response.dc_conductance_study, response.dc_amplitude_endpoint_study])
@pytest.mark.parametrize("selection", ["explicit", "prepared_schema"])
def test_pair_response_entry_cannot_fall_back_to_float(entry, selection):
    prepared = {"schema": "R1CommonStatePairV2"} if selection == "prepared_schema" else {}
    kwargs = {"backend": PAIR} if selection == "explicit" else {}
    args = (None, 16, {}, prepared)
    if entry is response.restore_controlled_dc:
        args = ({}, *args)
    with pytest.raises(ValueError, match="explicit matching backend|explicit V2 schema"):
        entry(*args, **kwargs)


def test_live_pair_observation_and_frequency_entry_are_guarded():
    system, state = SimpleNamespace(_r1_backend=PAIR), SimpleNamespace(fine={})
    with pytest.raises(ValueError, match="complete compensated state"):
        response.physical_observations(system, state)
    dc = response.R1DCResponse(system, state, {"certified": True})
    with pytest.raises(ValueError, match="complete compensated state"):
        response.small_signal_response(dc, [0.])
    with pytest.raises(ValueError, match="complete compensated state"):
        response.compare_transient_tail(dc, initial_state={}, tail_state={},
                                       tail_regular_current_A_m2=0., time_s=1.)


def test_verifier_rejects_pair_before_reading_saved_pass_flags():
    with pytest.raises(ValueError, match="explicit matching backend"):
        response.verify_response_content({"certified": True}, stack=None, intervals=16,
            binding={}, prepared={"schema": "R1CommonStatePairV2"}, request={})


def test_tail_dictionary_cannot_silently_discard_pair_words():
    dc = response.R1DCResponse(SimpleNamespace(), SimpleNamespace(), {"certified": True})
    with pytest.raises(ValueError, match="explicit matching backend"):
        response.compare_transient_tail(dc,
            initial_state={"precision_phi_V_hi": [1.], "precision_phi_V_lo": [1e-20]},
            tail_state={}, tail_regular_current_A_m2=0., time_s=1.)
