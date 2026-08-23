"""Dual-ion DAE temporal refinement against the production MoL solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae_dual_ion_integrator import (
    run_dual_ion_backward_euler_reference,
)
from perovskite_sim.solver.dae_dual_ions import (
    build_dual_ion_consistent_initial_condition,
    build_dual_ion_dae,
)
from perovskite_sim.solver.mol import (
    StateVec,
    _charge_density,
    poisson_right_boundary,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _biased_dual_ion_problem():
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layer = source.layers[1]
    assert layer.params is not None
    dual_layer = replace(
        layer,
        params=replace(
            layer.params,
            D_ion_neg=3.2e-18,
            P0_neg=layer.params.P0,
            P_lim_neg=layer.params.P_lim,
        ),
    )
    stack = replace(
        source,
        layers=(dual_layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=0.0,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
        ion_steric_diffusion_only=True,
        ion_steric_shared_site=True,
        mode="full",
    )
    grid = multilayer_grid([Layer(dual_layer.thickness, 6)], alpha=1.0)
    model = build_dual_ion_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=0.01,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    initial = build_dual_ion_consistent_initial_condition(model)
    return grid, stack, model, initial


def test_dual_ion_be_contracts_to_production_mol_reference():
    grid, stack, model, initial = _biased_dual_ion_problem()
    final_time_s = 1.0e-2
    mol = run_transient(
        grid,
        initial.physical_state,
        (0.0, final_time_s),
        np.array([0.0, final_time_s]),
        stack,
        illuminated=False,
        V_app=0.01,
        rtol=1.0e-10,
        atol=1.0e2,
        max_step=2.5e-5,
        mat=model.material,
    )
    assert mol.success
    mol_final = StateVec.unpack(mol.y[:, -1], grid.size)
    assert mol_final.P_neg is not None
    assert model.material.P_ion0_neg is not None
    rho = _charge_density(
        mol_final.p,
        mol_final.n,
        mol_final.P,
        model.material.P_ion0,
        model.material.N_A,
        model.material.N_D,
        P_neg=mol_final.P_neg,
        P_neg0=model.material.P_ion0_neg,
    )
    mol_potential = solve_poisson_prefactored(
        model.material.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, 0.01),
    )

    carrier_errors = []
    positive_ion_errors = []
    negative_ion_errors = []
    potential_errors = []
    trajectories = []
    for step_count in (2, 4, 8):
        result = run_dual_ion_backward_euler_reference(
            model,
            np.linspace(0.0, final_time_s, step_count + 1),
            initial=initial,
            residual_tolerance=1.0e-9,
            max_newton_iterations=24,
        )
        terminal = StateVec.unpack(result.physical_states[-1], grid.size)
        assert terminal.P_neg is not None
        carrier_errors.append(
            max(
                float(np.max(np.abs(np.log(terminal.n / mol_final.n)))),
                float(np.max(np.abs(np.log(terminal.p / mol_final.p)))),
            )
        )
        positive_ion_errors.append(
            float(np.max(np.abs((terminal.P - mol_final.P) / mol_final.P)))
        )
        negative_ion_errors.append(
            float(
                np.max(
                    np.abs((terminal.P_neg - mol_final.P_neg) / mol_final.P_neg)
                )
            )
        )
        potential_errors.append(
            float(np.max(np.abs(result.potentials_V[-1] - mol_potential)))
        )
        trajectories.append(result)

    for errors in (
        carrier_errors,
        positive_ion_errors,
        negative_ion_errors,
        potential_errors,
    ):
        values = np.asarray(errors)
        assert np.all(np.diff(values) < 0.0)
        np.testing.assert_allclose(values[1:] / values[:-1], 0.5, rtol=0.05)
    assert carrier_errors[-1] < 6.0e-9
    assert positive_ion_errors[-1] < 6.0e-12
    assert negative_ion_errors[-1] < 3.0e-12
    assert potential_errors[-1] < 1.3e-10
    assert all(
        result.max_normalized_differential_residual <= 1.0e-9
        and result.max_normalized_algebraic_residual <= 1.0e-9
        and result.max_relative_positive_ion_inventory_drift < 2.0e-15
        and result.max_relative_negative_ion_inventory_drift < 2.0e-15
        and result.minimum_site_vacancy_fraction > 0.0
        for result in trajectories
    )
