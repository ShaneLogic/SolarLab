"""Declared physical scales and a real p-i-n operating-state comparison."""

from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.validation.foundation_reference_refinement import (
    foundation_reference_protocol,
    run_foundation_reference_steady_state,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE = "foundation-physical-homojunction-steady-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/NumericalRefinementRegistry.yaml", project_root=ROOT
    ).lane(LANE)


def test_resolved_reference_keeps_all_original_physical_limits():
    registry = load_refinement_registry(
        ROOT / "reproducibility/NumericalRefinementRegistry.yaml", project_root=ROOT
    )
    original = registry.lane(LANE)
    resolved = registry.lane("foundation-physical-homojunction-resolved-v2")
    assert resolved.config_sha256 == original.config_sha256
    assert resolved.observables == original.observables
    assert resolved.quality_gates == original.quality_gates
    assert resolved.grid_values == (88, 176, 352)
    protocol = foundation_reference_protocol(resolved)
    assert protocol["profile_interpolation"] == "scharfetter_gummel_constant_face_flux"
    assert protocol["profile_points_per_layer"] == 33


def test_definition_keeps_physical_scales_and_independent_refinements():
    lane = _lane()
    protocol = foundation_reference_protocol(lane)
    assert lane.grid_values == (44, 88, 176)
    assert lane.tolerance_factors == (1.0, 0.1, 0.01)
    assert protocol["voltage_steps_V"] == [0.02, 0.01, 0.005]
    assert protocol["photon_energy_eV"] == 2.0
    assert protocol["current_normalization"] == "incident_photon_charge_flux"
    assert "tolerance_factor" not in protocol
    gates = {gate.metric: gate.limit for gate in lane.observables}
    assert gates["voc_V"] == 0.001
    assert gates["jsc_A_m2"] == 0.002
    assert gates["fill_factor"] == 0.005
    assert gates["open_circuit_electron_log_density"] == pytest.approx(np.log(1.01))


def test_equilibrium_preserves_subunit_contact_density_and_mass_action():
    from perovskite_sim.experiments.quasi_fermi_steady_state import (
        solve_quasi_fermi_steady_state,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.mol import build_material_arrays

    stack = load_device_from_yaml(ROOT / _lane().config_path)
    grid = np.linspace(0.0, sum(layer.thickness for layer in stack.layers), 45)
    material = build_material_arrays(grid, stack)
    state = solve_quasi_fermi_steady_state(
        grid, stack, illuminated=False, mat=material, require_contact_certificate=True
    )
    assert 0.0 < material.n_L < 1.0 and 0.0 < material.p_R < 1.0
    np.testing.assert_allclose(
        state.y[[0, 44, 45, 89]],
        [material.n_L, material.n_R, material.p_L, material.p_R],
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.y[:45] * state.y[45:90], material.ni_sq, rtol=1e-10, atol=0.0
    )


@pytest.mark.parametrize("model", ["frozen_ions", "tunnelling"])
def test_density_predictor_refuses_uncovered_physics_before_density_solve(
    monkeypatch, model
):
    import perovskite_sim.experiments.steady_state as density_solver
    from perovskite_sim.experiments.quasi_fermi_steady_state import (
        QuasiFermiSteadyStateError,
        solve_quasi_fermi_steady_state,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    path = (
        _lane().config_path
        if model == "frozen_ions"
        else "tests/fixtures/configs/wkb_resolved_electron_barrier.yaml"
    )
    stack = load_device_from_yaml(ROOT / path)
    if model == "frozen_ions":
        layers = list(stack.layers)
        layers[1] = replace(layers[1], params=replace(layers[1].params, P0=1e20))
        stack = replace(stack, layers=tuple(layers))
        grid = np.linspace(0.0, sum(layer.thickness for layer in stack.layers), 45)
    else:
        from perovskite_sim.discretization.grid import Layer, multilayer_grid

        grid = multilayer_grid(
            [Layer(layer.thickness, 12) for layer in stack.layers], alpha=5.0
        )

    def unexpected_seed(*args, **kwargs):
        raise AssertionError("unsupported physics reached the density predictor")

    monkeypatch.setattr(density_solver, "solve_steady_state", unexpected_seed)
    reason = (
        "nonzero ionic background"
        if model == "frozen_ions"
        else "bulk density prediction requires"
    )
    with pytest.raises(QuasiFermiSteadyStateError, match=reason):
        solve_quasi_fermi_steady_state(grid, stack, use_density_predictor=True)


@pytest.mark.slow
def test_real_reference_cell_certifies_all_operating_points_and_optical_power():
    lane = _lane()
    result = run_foundation_reference_steady_state(lane, MatrixPoint(44, 1.0), ROOT)
    quality = {item.name: item.values[0] for item in result.quality}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert {item.name for item in result.observables} == {
        gate.metric for gate in lane.observables
    }
    for gate in lane.quality_gates:
        value = quality[gate.metric]
        if gate.operator == "eq":
            assert value == gate.limit, gate.metric
        elif gate.operator == "le":
            assert value <= gate.limit, gate.metric
        else:
            assert value >= gate.limit, gate.metric
    actual = json.loads(result.metadata_json)["actual"]
    assert actual["incident_power_W_m2"] == pytest.approx(2.0 * Q * 1e21)
    assert actual["intervals"] == 44
    assert actual["density_basin_initializations"] == 1
    raw = actual["raw"]
    assert set(raw["operating_states"]) == {
        "equilibrium",
        "short_circuit",
        "open_circuit",
        "maximum_power",
    }
    assert len(raw["voltage_metrics"]) == 3
    assert 0.0 < raw["voltage_metrics"][-1]["PCE"] < 1.0
