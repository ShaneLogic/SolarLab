"""Actual short-run evidence before the sole complete V8 acceptance run."""
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import precision_context
from perovskite_sim.models.config_loader import load_device_from_yaml
from scripts.analyze_r1_v7_prototype import canonical, compare_state_arithmetic
from scripts.analyze_r1_v8_prototype import initial_checks
from scripts.verify_r1_v7_precision import freeze_healthy_material
from scripts.verify_r1_v8_precision import precision_fields, verify_two_sides


def test_actual_short_run_has_both_sides_initial_construction_and_regular_updates():
    project = Path(__file__).resolve().parents[2]
    stack = load_device_from_yaml(project/"tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((project/"tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    grid, material = build_r1_material(stack, 16)
    frozen = freeze_healthy_material(grid, material, source_identity="actual_short_integration_test_before_precision_context")
    with threadpool_limits(1), precision_context():
        prepared = states.prepare_common_state(stack, 16, binding)
        result = states.json_data(protocol.run_r1_step(stack, 16, binding, prepared,
            times_s=[0., 1e-9], policy=protocol.r1_policy(.1), physics_evidence=True))
    assert result["certificate"]["certified"]
    assert len(result["accepted_steps"]) == 10
    initial = initial_checks(frozen, prepared.to_dict(), result)
    assert initial["qualified"], initial
    upstream = result["initial_event"]["precision_arithmetic_context"]
    budget = {"local_state_arithmetic": {"potential_qf_trace_absolute_error_V": 1e-26,
                                       "population_occupancy_relative_error": 1e-27}}
    anchors = regular = 0
    previous = None
    for row in result["accepted_steps"]:
        sides = verify_two_sides(frozen, row, upstream=upstream)
        assert sides["qualified"], sides
        assert sides["upstream_construction"] is not None
        if row["time_s"] == 0:
            anchors += 1
            assert canonical(precision_fields(row["state"])) == canonical(initial["zero_plus_fields"])
        else:
            regular += 1
            arithmetic = compare_state_arithmetic(previous, row, frozen, budget)
            assert arithmetic["passed"], arithmetic
        previous = row
    assert (anchors, regular) == (3, 7)
