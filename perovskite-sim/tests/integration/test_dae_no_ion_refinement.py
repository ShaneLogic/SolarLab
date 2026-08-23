"""No-ion temporal refinement of the research DAE against production MoL."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae import (
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
)
from perovskite_sim.solver.dae_integrator import run_backward_euler_reference
from perovskite_sim.solver.mol import StateVec, _charge_density, run_transient
from perovskite_sim.solver.newton import solve_equilibrium


def _illuminated_no_ion_problem():
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source_layer = source.layers[1]
    assert source_layer.params is not None
    layer = replace(
        source_layer,
        params=replace(source_layer.params, alpha=2.0e4),
    )
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=1.0e17,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, 6)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_no_ion_no_interface_dae(
        grid,
        stack,
        reference,
        illuminated=True,
        reference_time_s=1.0e-9,
    )
    initial = build_consistent_initial_condition(model)
    return grid, stack, model, initial


def test_backward_euler_contracts_to_high_accuracy_mol_reference():
    grid, stack, model, initial = _illuminated_no_ion_problem()
    final_time_s = 1.0e-9
    mol = run_transient(
        grid,
        initial.physical_state,
        (0.0, final_time_s),
        np.array([0.0, final_time_s]),
        stack,
        illuminated=True,
        V_app=0.0,
        rtol=1.0e-10,
        atol=1.0e-3,
        max_step=final_time_s / 100.0,
        mat=model.material,
    )
    assert mol.success
    mol_final = StateVec.unpack(mol.y[:, -1], grid.size)
    rho = _charge_density(
        mol_final.p,
        mol_final.n,
        mol_final.P,
        model.material.P_ion0,
        model.material.N_A,
        model.material.N_D,
    )
    mol_potential = solve_poisson_prefactored(
        model.material.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=0.0,
    )

    errors = []
    potential_errors = []
    trajectories = []
    for step_count in (2, 4, 8):
        result = run_backward_euler_reference(
            model,
            np.linspace(0.0, final_time_s, step_count + 1),
            initial=initial,
        )
        final = StateVec.unpack(result.physical_states[-1], grid.size)
        errors.append(
            max(
                float(np.max(np.abs(np.log(final.n / mol_final.n)))),
                float(np.max(np.abs(np.log(final.p / mol_final.p)))),
            )
        )
        potential_errors.append(
            float(np.max(np.abs(result.potentials_V[-1] - mol_potential)))
        )
        trajectories.append(result)

    errors = np.asarray(errors)
    contraction = errors[1:] / errors[:-1]
    assert np.all(np.diff(errors) < 0.0)
    np.testing.assert_allclose(contraction, 0.5, rtol=0.03, atol=0.0)
    assert errors[-1] < 2.0e-4
    assert max(potential_errors) < 2.0e-13
    assert all(
        result.max_normalized_differential_residual <= 1.0e-9
        and result.max_normalized_algebraic_residual <= 1.0e-9
        and result.max_electron_balance_defect_A_m2 < 2.0e-9
        and result.max_hole_balance_defect_A_m2 < 2.0e-9
        for result in trajectories
    )
