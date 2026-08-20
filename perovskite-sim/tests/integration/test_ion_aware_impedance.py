from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    build_ion_aware_impedance_protocol,
    run_ion_aware_impedance,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_ionmonger_certified_dc_drives_reference_frequency_operator():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 12)
    dc_protocol = build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
    )
    dc_state = solve_ion_aware_dc(x, stack, dc_protocol)
    protocol = build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0e-3, 1.0, 1.0e3]),
    )

    result = run_ion_aware_impedance(
        x,
        stack,
        protocol,
        dc_state=dc_state,
    )

    certificate = result.certificate
    assert dc_state.numerically_certified
    assert certificate.numerically_certified
    assert not certificate.thermodynamically_certified
    assert not certificate.certified
    assert not certificate.reasons
    assert certificate.max_relative_face_spread < protocol.max_relative_face_spread
    assert certificate.max_backward_error < protocol.max_backward_error
    assert certificate.max_ion_inventory_response_relative < (
        protocol.max_ion_inventory_response_relative
    )
    assert all(item.passed for item in certificate.perturbation_assessments)
    assert np.max(np.abs(result.positive_ion_admittance_faces_S_m2)) > 0.0
    assert np.ptp(result.Z.real) > 0.0
    assert result.protocol.dc_protocol_sha256 == dc_state.protocol_hash
    assert result.reference_linearization.mass_matrix.shape == (
        result.coordinate_layout.size,
        result.coordinate_layout.size,
    )


def test_symmetric_dual_ion_state_keeps_both_blocking_inventory_constraints():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layers = []
    for layer in stack.layers:
        params = layer.params
        if layer.role == "absorber":
            params = replace(
                params,
                D_ion_neg=params.D_ion,
                P0_neg=params.P0,
                P_lim_neg=params.P_lim,
            )
        layers.append(replace(layer, params=params))
    dual_stack = replace(stack, layers=tuple(layers))
    x = build_electrical_grid(dual_stack, 12)
    dc_state = solve_ion_aware_dc(
        x,
        dual_stack,
        build_ion_aware_dc_protocol(
            dual_stack,
            V_dc=0.9,
            illuminated=True,
        ),
    )
    protocol = build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0e-3, 1.0, 1.0e3]),
    )

    result = run_ion_aware_impedance(
        x,
        dual_stack,
        protocol,
        dc_state=dc_state,
    )

    assert result.certificate.numerically_certified
    assert result.negative_ion_admittance_faces_S_m2 is not None
    assert result.negative_ion_storage_response_F_m2 is not None
    assert result.coordinate_layout.negative_ion_state_indices
    assert result.certificate.max_ion_inventory_response_relative < (
        protocol.max_ion_inventory_response_relative
    )
    assert np.max(np.abs(result.positive_ion_admittance_faces_S_m2)) > 0.0
    assert np.max(np.abs(result.negative_ion_admittance_faces_S_m2)) > 0.0
