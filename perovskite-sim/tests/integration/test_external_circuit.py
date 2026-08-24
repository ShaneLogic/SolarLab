from __future__ import annotations

import numpy as np

from perovskite_sim.experiments.external_circuit import (
    ExternalCircuitProtocol,
    apply_external_circuit,
)
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_real_transient_jv_closes_zero_and_nonzero_external_circuit():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    intrinsic = run_jv_sweep(
        stack,
        N_grid=20,
        n_points=12,
        V_max=1.2,
        v_rate=20.0,
    )
    assert intrinsic.certified

    zero = apply_external_circuit(intrinsic, ExternalCircuitProtocol())
    assert zero.certified
    assert np.array_equal(zero.forward.terminal_voltage_V, intrinsic.V_fwd)
    assert np.array_equal(zero.forward.terminal_current_A_m2, intrinsic.J_fwd)
    assert zero.metrics_fwd == intrinsic.metrics_fwd

    parasitic = apply_external_circuit(
        intrinsic,
        ExternalCircuitProtocol(
            series_resistance_ohm_m2=2.0e-4,
            shunt_resistance_ohm_m2=0.2,
        ),
    )
    assert parasitic.certified
    assert parasitic.forward.max_current_balance_error_A_m2 == 0.0
    assert parasitic.forward.max_voltage_balance_error_V == 0.0
    assert parasitic.metrics_fwd.FF < intrinsic.metrics_fwd.FF
    assert parasitic.metrics_fwd.PCE < intrinsic.metrics_fwd.PCE
    assert parasitic.source_experiment_protocol_sha256 == intrinsic.protocol.sha256
