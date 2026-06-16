# tests/unit/autoloop/test_search_types.py
import dataclasses
import pytest
from perovskite_sim.autoloop.search import (
    DesignKnob, Trial, SearchResult, SearchNotTrusted, DEFAULT_DESIGN_SPACE,
)


def test_design_knob_frozen():
    k = DesignKnob(axis="etl_doping_cm3", low=1e15, high=1e19, scale="log")
    assert k.scale == "log"
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.low = 0.0  # type: ignore[misc]


def test_trial_and_result():
    t = Trial(design={"etl_doping_cm3": 1e17}, pce=0.27, bracketed=True)
    r = SearchResult(best=t, trials=(t,), n_evaluated=1, parity_overall=0.9, budget=10)
    assert r.best.pce == 0.27 and r.n_evaluated == 1


def test_search_not_trusted_is_runtimeerror():
    assert issubclass(SearchNotTrusted, RuntimeError)


def test_default_design_space_axes():
    axes = {k.axis for k in DEFAULT_DESIGN_SPACE}
    assert axes == {"etl_delta_ec_eV", "htl_delta_ev_eV",
                    "etl_doping_cm3", "absorber_defect_density_cm3"}
    assert all(k.low < k.high for k in DEFAULT_DESIGN_SPACE)
