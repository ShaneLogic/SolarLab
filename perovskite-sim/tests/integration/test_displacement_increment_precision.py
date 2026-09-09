"""Displacement tangents against independent capacitance and Gauss-law oracles."""

import math

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0
from perovskite_sim.experiments.ion_aware_dc import build_ion_aware_dc_protocol, solve_ion_aware_dc
from perovskite_sim.experiments.ion_aware_impedance import (
    _build_reference_evaluator,
    _state_coordinate_layout,
    build_ion_aware_impedance_protocol,
)
from perovskite_sim.experiments.ion_aware_structured_jacobian import (
    _build_poisson_implicit_sensitivity,
    _structured_evaluator,
    build_ion_aware_structured_jacobian_protocol,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


@pytest.fixture(scope="module")
def displacement_case():
    stack = load_device_from_yaml("tests/fixtures/configs/ionmonger_benchmark.yaml")
    grid = build_electrical_grid(stack, 90)
    material = build_material_arrays(grid, stack)
    dc = solve_ion_aware_dc(grid, stack, build_ion_aware_dc_protocol(stack, V_dc=0.9, illuminated=True))
    protocol = build_ion_aware_impedance_protocol(dc, np.array([1.0]))
    layout = _state_coordinate_layout(material, grid.size)
    poisson = _build_poisson_implicit_sensitivity(
        grid, stack, dc, material, layout,
        build_ion_aware_structured_jacobian_protocol(protocol), progress=None,
    )
    reference = _build_reference_evaluator(grid, stack, protocol, material, layout, dc.y)
    structured = _structured_evaluator(grid, stack, dc, protocol, material, layout, poisson)
    eps_face = EPS_0 * 2 * material.eps_r[:-1] * material.eps_r[1:] / (material.eps_r[:-1] + material.eps_r[1:])
    capacitance = 1 / math.fsum(np.diff(grid) / eps_face)
    return {"reference": reference, "structured": structured}, layout, capacitance


@pytest.mark.parametrize("method", ["reference", "structured"])
@pytest.mark.parametrize("step", [1e-3, 1e-5, 1e-7])
def test_voltage_tangent_resolves_series_capacitance(displacement_case, method, step):
    evaluators, layout, capacitance = displacement_case
    evaluate = evaluators[method]
    coordinate = np.zeros(layout.size)
    high_voltage, low_voltage = 0.9 + step, 0.9 - step
    high = evaluate(coordinate, high_voltage).displacement_charge_faces
    low = evaluate(coordinate, low_voltage).displacement_charge_faces
    derivative = (high - low) / (high_voltage - low_voltage)
    np.testing.assert_allclose(derivative, capacitance, rtol=2e-10, atol=0.0)
