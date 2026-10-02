"""Actual N16 state/trajectory and replay, separate from long-window evidence."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as replay
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import precision_context
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_actual_fine_initialization_short_steps_and_exact_replay():
    project=Path(__file__).resolve().parents[2]
    stack=load_device_from_yaml(project/"tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding=json.loads((project/"tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    prepare_before=states.prepare_common_state
    snapshot_before=states.snapshot
    with threadpool_limits(1),precision_context():
        prepared=states.prepare_common_state(stack,16,binding)
        assert prepared.to_dict()["schema"]=="R1CommonStatePairV1"
        system,initial=states.verify_prepared_physics(prepared,stack,binding)
        assert np.max(np.abs(initial.poisson_residual))<1e-30
        assert np.any(initial.fine["phi_V"].lo!=0)
        policy=protocol.r1_policy(.1)
        result=protocol.run_r1_step(stack,16,binding,prepared,times_s=[0,1e-9],policy=policy,physics_evidence=True)
        assert result["certificate"]["certified"]
        assert len(result["accepted_steps"])==10  # 2+3+5 across all three levels.
        verified=replay.verify_r1_step_physics(stack,16,binding,prepared,result)
        assert verified["certified"]
    assert states.prepare_common_state is prepare_before
    assert states.snapshot is snapshot_before
