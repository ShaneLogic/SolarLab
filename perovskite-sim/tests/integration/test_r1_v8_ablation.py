"""Real single-tier reference intervention, truthful physical history and replay."""
import json
import copy
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch

from threadpoolctl import threadpool_limits
from scripts.run_r1_v8_ablation import run_trace, replay_trace, validate_replay_header, bounded_scientific_outcome
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import precision_context


def test_actual_short_ablation_and_replay_share_intervention():
    project = Path(__file__).resolve().parents[2]
    stack = load_device_from_yaml(project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    with threadpool_limits(1), precision_context():
        prepared = states.prepare_common_state(stack, 16, binding, policy=protocol.r1_policy())
        result = run_trace(stack, binding, prepared, [0., 1e-9], protocol.r1_policy(.1))
        assert result["expected_rows"] == 5
        assert result["observed_rows"] >= 3, result.get("failure")
        rows = result["accepted_steps"]
        assert "reference_quantization" not in rows[0]["physics_reconstruction"]
        assert "reference_quantization" not in rows[1]["physics_reconstruction"]
        assert "reference_quantization" in rows[2]["physics_reconstruction"]
        intervention = rows[2]["physics_reconstruction"]["reference_quantization"]
        assert intervention["physical_previous_unchanged"]
        assert intervention["reference_before_identity"] == intervention["physical_previous_identity"]
        for field, words in intervention["reference_after"].items():
            assert all(value == 0. for value in np.asarray(words["lo"]).ravel())
            assert words["hi"] == intervention["reference_before"][field]["hi"]
        assert result["intervention_rows"] >= 1
        assert not result["refinement_evidence"]["available"]
        metric_names = set().union(*(row["applicable_original_gates"]["metrics"] for row in rows))
        from scripts.run_r1_v8_prototype import LIMITS
        assert metric_names == set(LIMITS) - {"refinement_state_change", "refinement_current_change"}
        replay = replay_trace(stack, binding, prepared, result, expected_times=[0., 1e-9])
        assert replay["evidence_exact"]
        assert replay["external_time_array_bound"]
        assert not replay["P1_qualified"]
        with pytest.raises(ValueError, match="external ablation time array"):
            replay_trace(stack, binding, prepared, result, expected_times=[0., 2e-9])
        for field, value in (("observed_rows", 0), ("expected_rows", 9), ("intervention_rows", 99),
                             ("amplitude_V", .004), ("control", "B"), ("P1_qualified", True),
                             ("complete", not result["complete"]), ("source", {"tampered": True})):
            changed = copy.deepcopy(result)
            changed[field] = value
            changed["sha256"] = states.digest({key: item for key, item in changed.items() if key != "sha256"})
            with pytest.raises((ValueError, TypeError)):
                validate_replay_header(prepared, binding, changed)
        # Exercise the first-intervention failure boundary with real inputs.
        # This controlled stop is not a claimed physical/Newton failure.
        from perovskite_sim.experiments import interface_defect_transient as transient
        original_solver = transient._solve_step
        def stop_at_first_quantized_attempt(system, *args, **kwargs):
            if system.rebase_evidence is not None:
                raise RuntimeError("controlled test stop before quantized solve")
            return original_solver(system, *args, **kwargs)
        with patch.object(transient, "_solve_step", stop_at_first_quantized_attempt):
            partial = run_trace(stack, binding, prepared, [0., 1e-9], protocol.r1_policy(.1))
        assert transient._solve_step is original_solver
        assert partial["execution_status"] == "failed" and partial["observed_rows"] == 2
        assert partial["intervention_rows"] == 0
        assert partial["last_solver_attempt"]["row_index"] == 2
        assert partial["last_solver_attempt"]["reference_quantization"]["physical_previous_unchanged"]
        partial_replay = replay_trace(stack, binding, prepared, partial)
        assert partial_replay["evidence_exact"]
        assert partial_replay["last_solver_attempt_inputs_reconstructed"]
        assert partial_replay["intervention_attempt_reconstructed"]
        assert not partial_replay["failure_provenance_reexecuted"]
        assert not bounded_scientific_outcome(partial)
