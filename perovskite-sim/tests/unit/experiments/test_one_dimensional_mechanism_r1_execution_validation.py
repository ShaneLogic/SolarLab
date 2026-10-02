"""Relabelling executed settings cannot pass current physical validation."""

from copy import deepcopy

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_execution_validation import (
    verify_execution_parameters,
)
from tests.unit.experiments.test_one_dimensional_mechanism_r1_execution_axes import (
    runner, solver_boundary,
)


@pytest.fixture
def recorded_settings(runner, solver_boundary):
    arguments, _, output = solver_boundary
    assert runner.main([*arguments, "--physics-evidence", "--window", "full",
                        "--time-substeps", "2", "4", "8"]) == 1
    # This test stops at the solver boundary. It tests the executed settings;
    # separate integration tests reconstruct real states and currents.
    runner.write_json(output / "PreparedStateV1.json", {
        "physical_stack": runner.read_json(output / "ResolvedStackV1.json"),
    })
    completion = runner.read_json(output / "CompletionV1.json")
    protocol = runner.read_json(output / "ProtocolV1.json")
    result = verify_execution_parameters(output, completion)
    assert result["times_s"][-1] == 100.0
    assert result["convergence_claimed"] is False
    return output, completion, protocol


@pytest.mark.parametrize("mutation", [
    "window_end", "window_count", "window_spacing", "output_time", "time_axis",
    "nonlinear", "physical_limit", "control_definition", "material", "scope",
    "physics_evidence", "failure_waiver",
])
def test_relabelled_settings_are_rejected(runner, recorded_settings, mutation):
    output, completion, protocol = recorded_settings
    protocol = deepcopy(protocol)
    if mutation == "window_end":
        protocol["observation_window"]["last_time_s"] = 1000.
    elif mutation == "window_count":
        protocol["observation_window"]["output_point_count"] += 1
    elif mutation == "window_spacing":
        protocol["observation_window"]["logarithmic_intervals_per_decade"] = 24
    elif mutation == "output_time":
        protocol["times_s"][10] *= 1.01
    elif mutation == "time_axis":
        protocol["time_substeps"] = [4, 8, 16]
    elif mutation == "nonlinear":
        protocol["nonlinear_factor"] = .01
    elif mutation == "physical_limit":
        protocol["policy"]["maximum_eliminated_operator_relative_error"] = 1.0
    elif mutation == "control_definition":
        protocol["control_definitions"]["D"] = protocol["control_definitions"]["A"]
    elif mutation == "material":
        runner.write_json(output / "ResolvedStackV1.json", {"temperature_K": 301.})
    elif mutation == "scope":
        protocol["stage_scope"] = completion["stage_scope"] = "R1-1"
    elif mutation == "physics_evidence":
        protocol["physics_evidence"] = False
    elif mutation == "failure_waiver":
        protocol["historical_failure_case_count"] = 0
    runner.write_json(output / "ProtocolV1.json", protocol)
    with pytest.raises(ValueError, match="parameter mismatch|requires saved"):
        verify_execution_parameters(output, completion)


def test_extended_formal_execution_is_a_single_setting_not_convergence(runner, recorded_settings):
    output, completion, protocol = recorded_settings
    completion.update(run_class="formal", stage_scope="R1-2-physics")
    protocol.update(run_class="formal", stage_scope="R1-2-physics")
    runner.write_json(output / "ProtocolV1.json", protocol)
    result = verify_execution_parameters(output, completion)
    assert result["scope"] == "R1-2-physics"
    assert result["convergence_claimed"] is False
    assert result["mechanism_identification_claimed"] is False
