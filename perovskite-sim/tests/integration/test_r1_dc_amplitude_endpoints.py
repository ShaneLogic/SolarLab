"""Real controlled DC endpoint values remain bound to their requested study."""
import copy

import numpy as np
import pytest

from tests.fixtures.r1_reference import approved_r1_binding
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import AMPLITUDES_V
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    dc_amplitude_endpoint_study, verify_response_content,
)


@pytest.fixture(scope="module")
def endpoint_study():
    stack, binding = load_device_from_yaml(FIXTURE), approved_r1_binding()
    prepared = prepare_common_state(stack, 16, binding)
    record = dc_amplitude_endpoint_study(stack, 16, binding, prepared, control="D")
    request = {"intervals": 16, "control": "D", "operating_voltage_V": 0.,
               "amplitudes_V": list(AMPLITUDES_V), "prepared_sha256": prepared.sha256,
               "source_sha256": record["source"]["sha256"]}
    return record, dict(stack=stack, intervals=16, binding=binding, prepared=prepared, request=request)


def test_real_dc_endpoint_values_replay_and_do_not_certify_linearity(endpoint_study):
    record, kwargs = endpoint_study
    assert verify_response_content(record, **kwargs)["certified"]
    assert len(record["endpoints"]) == 7 and record["dc_states_certified"]
    baseline = record["baseline"]["terminal_current_A_m2"]
    for amplitude, endpoint, normalized in zip(AMPLITUDES_V, record["endpoints"], record["normalized_endpoint_response_S_m2"]):
        assert endpoint["dc"]["voltage_V"] == amplitude
        assert normalized == (endpoint["dc"]["terminal_current_A_m2"]-baseline)/amplitude
    assert np.all(np.isfinite(record["normalized_endpoint_response_S_m2"]))
    assert record["absolute_current_error_bounds_A_m2"] is None
    assert not record["linearity_certified"] and not record["full_transient_linearity_certified"]


@pytest.mark.parametrize("mutation", ["dc_value", "amplitude", "source"])
def test_replay_rejects_changed_real_dc_value_amplitude_or_source(endpoint_study, mutation):
    original, kwargs = endpoint_study
    record = copy.deepcopy(original)
    if mutation == "dc_value":
        record["endpoints"][1]["dc"]["terminal_current_A_m2"] += 1e-8
    elif mutation == "amplitude":
        record["endpoints"][1]["amplitude_V"] = .0025
    else:
        record["source"]["sha256"] = "other"
    with pytest.raises(ValueError, match="response content mismatch"):
        verify_response_content(record, **kwargs)


def test_replay_rejects_request_source_or_amplitude_mismatch(endpoint_study):
    record, kwargs = endpoint_study
    request = {**kwargs["request"], "amplitudes_V": [.005, .0025]}
    with pytest.raises(ValueError, match="declared amplitude ladder"):
        verify_response_content(record, **{**kwargs, "request": request})
    request = {**kwargs["request"], "source_sha256": "other"}
    with pytest.raises(ValueError, match="source identity"):
        verify_response_content(record, **{**kwargs, "request": request})
