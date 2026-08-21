"""Fast, real-RHS health probes for the P1.2 edge-regime contract."""

import dataclasses

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.recombination import _observe_srh_denominators
from perovskite_sim.solver import mol
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsMonitor,
    NumericalDiagnosticsPolicy,
    StateLayout,
)


def _replace_material(stack, layer_index: int, **changes):
    layers = list(stack.layers)
    layer = layers[layer_index]
    layers[layer_index] = dataclasses.replace(
        layer,
        params=dataclasses.replace(layer.params, **changes),
    )
    return dataclasses.replace(stack, layers=tuple(layers))


def _edge_case(name: str):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    illuminated = False
    biases = (0.0,)
    carrier_scale = 1.0

    if name == "dark_depletion":
        carrier_scale = 1.0e-6
        biases = (-0.8,)
    elif name == "strong_injection":
        carrier_scale = 1.0e5
        illuminated = True
        biases = (1.2,)
    elif name == "deep_cliff":
        stack = _replace_material(stack, 1, chi=4.0, Eg=1.6)
        stack = _replace_material(stack, 2, chi=4.8, Eg=3.2)
        biases = (-0.5,)
    elif name == "deep_spike":
        stack = _replace_material(stack, 1, chi=4.0, Eg=1.6)
        stack = _replace_material(stack, 2, chi=3.2, Eg=3.2)
    elif name == "low_temperature":
        stack = dataclasses.replace(stack, T=180.0)
    elif name == "extremely_low_ni":
        stack = _replace_material(
            stack,
            1,
            ni=1.0e-12,
            n1=1.0e-12,
            p1=1.0e-12,
        )
    elif name == "high_trap_density":
        stack = _replace_material(
            stack,
            1,
            tau_n=1.0e-14,
            tau_p=1.0e-14,
            trap_N_t_interface=1.0e26,
            trap_N_t_bulk=1.0e20,
            trap_decay_length=1.0e-8,
        )
    elif name == "rapid_bias_step":
        # Two real RHS evaluations straddle a 0 -> 1.2 V step over 1 ps.
        # This is intentionally an RHS health probe, not a claim that a
        # two-sample trajectory resolves the physical step response.
        biases = (0.0, 1.2)
    else:  # pragma: no cover - parameter table is closed below
        raise AssertionError(name)
    return stack, illuminated, biases, carrier_scale


@pytest.mark.parametrize(
    "case_name",
    [
        "dark_depletion",
        "strong_injection",
        "deep_cliff",
        "deep_spike",
        "low_temperature",
        "extremely_low_ni",
        "high_trap_density",
        "rapid_bias_step",
    ],
)
def test_real_rhs_edge_regime_is_finite_and_strictly_reported(case_name):
    stack, illuminated, biases, carrier_scale = _edge_case(case_name)
    x = multilayer_grid([Layer(layer.thickness, 3) for layer in stack.layers])
    mat = mol.build_material_arrays(x, stack)
    intrinsic = np.sqrt(
        np.maximum(mat.ni_sq, np.finfo(float).tiny)
    )
    carriers = np.maximum(intrinsic * carrier_scale, 1.0e-30)
    state = mol.StateVec.pack(carriers, carriers.copy(), mat.P_ion0)

    monitor = NumericalDiagnosticsMonitor(
        StateLayout(
            len(x),
            positive_ion_active=tuple(mat.P_ion0 > 0.0),
        ),
        NumericalDiagnosticsPolicy.research_strict(
            bulk_srh_denominator_floor_s_m3=0.0
        ),
    )

    for index, bias in enumerate(biases):
        monitor.observe_trial_state(state)
        with _observe_srh_denominators(monitor.observe_srh_denominator):
            rhs = mol.assemble_rhs(
                float(index) * 1.0e-12,
                state,
                x,
                stack,
                mat,
                illuminated=illuminated,
                V_app=bias,
            )
        monitor.observe_rhs(rhs)
        assert np.all(np.isfinite(rhs))

    report = monitor.finalize(state, solver_success=True)

    assert report.would_pass_strict is True
    assert report.nonfinite_trial_evaluations == 0
    assert report.nonfinite_rhs_evaluations == 0
    assert report.minimum_bulk_srh_denominator_s_m3 is not None
    assert report.minimum_bulk_srh_denominator_s_m3 > 0.0
    assert report.trial_evaluations == len(biases)

    # Prove each named fixture actually activated its intended edge parameter.
    if case_name == "dark_depletion":
        assert np.max(carriers) < 1.0e9
    elif case_name == "strong_injection":
        assert np.max(carriers) > 1.0e18
    elif case_name == "deep_cliff":
        assert mat.chi[-1] - mat.chi[len(x) // 2] > 0.5
    elif case_name == "deep_spike":
        assert mat.chi[-1] - mat.chi[len(x) // 2] < -0.5
    elif case_name == "low_temperature":
        assert mat.T_device == 180.0
    elif case_name == "extremely_low_ni":
        assert np.min(mat.ni_sq[mat.P_ion0 > 0.0]) <= 1.0e-24
    elif case_name == "high_trap_density":
        assert np.min(mat.tau_n[mat.P_ion0 > 0.0]) <= 1.0e-14
    elif case_name == "rapid_bias_step":
        assert biases == (0.0, 1.2)
