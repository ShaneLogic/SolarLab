"""Restricted charged bulk-trap equilibrium solver tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.contacts import build_semiconductor_contact_state
from perovskite_sim.solver.bulk_trap_equilibrium import (
    _density_state,
    solve_bulk_trap_pn_equilibrium,
)
from perovskite_sim.solver.mol import (
    BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    BulkTrapChargeCapabilityError,
    build_material_arrays,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/csi_gaussian_bulk_trap_pn_research.yaml"


def _stack():
    return load_device_from_yaml(str(CONFIG))


def _grid(intervals: int):
    stack = _stack()
    return multilayer_grid(
        tuple(Layer(layer.thickness, intervals) for layer in stack.layers),
        alpha=3.0,
    )


def test_real_equilibrium_closes_poisson_transport_traps_and_gauss_law():
    result = solve_bulk_trap_pn_equilibrium(
        _grid(20),
        _stack(),
        quadrature_order=32,
        poisson_tolerance=1.0e-8,
    )

    assert result.newton_iterations <= 20
    assert result.maximum_normalized_poisson_residual <= 1.0e-8
    assert result.maximum_relative_face_current < 1.0e-12
    assert result.maximum_mass_action_relative_error < 1.0e-12
    assert result.gauss_law_relative_error < 5.0e-8
    assert 0.0 <= result.minimum_trap_occupancy
    assert result.maximum_trap_occupancy <= 1.0
    assert result.integrated_bulk_trap_charge_C_m2 < 0.0
    assert result.left_contact.bulk_trap_state is not None
    assert result.right_contact.bulk_trap_state is not None


def test_poisson_trap_charge_tangent_matches_finite_difference():
    stack = _stack()
    grid = _grid(8)
    material = build_material_arrays(
        grid,
        stack,
        bulk_trap_charge_closure=BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    )
    contact = build_semiconductor_contact_state(
        stack.layers[0].params,
        temperature_K=material.T_device,
        use_temperature_scaling=True,
        bulk_trap_quadrature_order=32,
    )
    potential = np.linspace(
        stack.phi_left,
        stack.phi_left + material.V_bi_bc,
        grid.size,
    )
    state = _density_state(
        potential,
        material,
        contact.work_function_eV,
        quadrature_order=32,
    )
    analytic = state.charge_derivative_potential_C_m3_V(
        material.V_T_device
    )
    step = 1.0e-7
    plus = _density_state(
        potential + step,
        material,
        contact.work_function_eV,
        quadrature_order=32,
    ).charge_density_C_m3(material)
    minus = _density_state(
        potential - step,
        material,
        contact.work_function_eV,
        quadrature_order=32,
    ).charge_density_C_m3(material)

    np.testing.assert_allclose(
        analytic,
        (plus - minus) / (2.0 * step),
        rtol=3.0e-8,
        atol=0.0,
    )


def test_energy_and_spatial_refinement_are_stable_on_real_equilibrium():
    stack = _stack()
    order_16 = solve_bulk_trap_pn_equilibrium(
        _grid(20),
        stack,
        quadrature_order=16,
        poisson_tolerance=1.0e-8,
    )
    order_32 = solve_bulk_trap_pn_equilibrium(
        _grid(20),
        stack,
        quadrature_order=32,
        poisson_tolerance=1.0e-8,
    )
    grid_40 = solve_bulk_trap_pn_equilibrium(
        _grid(40),
        stack,
        quadrature_order=32,
        poisson_tolerance=1.0e-8,
    )

    charge_scale = abs(order_32.integrated_bulk_trap_charge_C_m2)
    assert abs(
        order_32.integrated_bulk_trap_charge_C_m2
        - order_16.integrated_bulk_trap_charge_C_m2
    ) / charge_scale < 5.0e-3
    assert abs(
        grid_40.integrated_bulk_trap_charge_C_m2
        - order_32.integrated_bulk_trap_charge_C_m2
    ) / charge_scale < 2.0e-2
    assert abs(
        grid_40.peak_electric_field_V_m
        - order_32.peak_electric_field_V_m
    ) / grid_40.peak_electric_field_V_m < 5.0e-2


def test_research_topology_rejects_nonhomogeneous_distribution():
    stack = _stack()
    right = stack.layers[1]
    altered_distribution = replace(
        right.params.bulk_trap_distribution,
        total_density_m3=2.0e22,
    )
    altered = replace(
        stack,
        layers=(
            stack.layers[0],
            replace(
                right,
                params=replace(
                    right.params,
                    bulk_trap_distribution=altered_distribution,
                ),
            ),
        ),
    )

    with pytest.raises(
        BulkTrapChargeCapabilityError,
        match="non-homojunction fields=bulk_trap_distribution",
    ):
        solve_bulk_trap_pn_equilibrium(_grid(8), altered)


def test_equilibrium_solver_rejects_a_grid_that_omits_part_of_the_stack():
    truncated = _grid(8)[:-1]

    with pytest.raises(ValueError, match="span the full electrical stack"):
        solve_bulk_trap_pn_equilibrium(truncated, _stack())
