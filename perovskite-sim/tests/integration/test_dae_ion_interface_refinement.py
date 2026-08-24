"""Combined ion/interface DAE temporal refinement against production MoL."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.interface_plane import solve_interface_states_live_qss
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae_ion_interface_integrator import (
    run_ion_interface_backward_euler_reference,
)
from perovskite_sim.solver.dae_ion_interface_states import (
    build_single_ion_algebraic_interface_consistent_initial_condition,
    build_single_ion_algebraic_interface_dae,
)
from perovskite_sim.solver.mol import (
    StateVec,
    _charge_density,
    poisson_right_boundary,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import NumericalDiagnosticsPolicy


def _combined_problem():
    layer = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=1.0e-16,
        P_lim=1.0e24,
        P0=1.0e22,
        ni=1.0e12,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e12,
        p1=1.0e12,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=1.5,
        Nc300=1.0e25,
        Nv300=1.0e25,
    )
    stack = DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, layer, role="absorber"),
            LayerSpec("right", 1.0e-7, replace(layer, chi=4.1), role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )
    grid = multilayer_grid(
        [Layer(value.thickness, 4) for value in stack.layers],
        alpha=1.0,
    )
    model = build_single_ion_algebraic_interface_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=0.01,
        carrier_reference_time_s=1.0e-7,
        ion_reference_time_s=1.0,
    )
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    return grid, stack, model, initial


def test_combined_be_contracts_to_eliminated_qss_mol_reference():
    grid, stack, model, initial = _combined_problem()
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
        numerical_diagnostics=NumericalDiagnosticsPolicy.research_strict(
            terminal_density_floor_m3=0.0,
            bulk_srh_denominator_floor_s_m3=0.0,
        ),
    )

    assert mol.success
    assert mol.numerical_diagnostics.would_pass_strict
    assert mol.numerical_diagnostics.nonfinite_rhs_evaluations == 0
    mol_terminal = StateVec.unpack(mol.y[:, -1], grid.size)
    rho = _charge_density(
        mol_terminal.p,
        mol_terminal.n,
        mol_terminal.P,
        model.material.P_ion0,
        model.material.N_A,
        model.material.N_D,
    )
    mol_potential = solve_poisson_prefactored(
        model.material.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, 0.01),
    )
    mol_interface = solve_interface_states_live_qss(
        model.material,
        stack,
        mol_terminal.n,
        mol_terminal.p,
        mol_potential,
        V_app=0.01,
        v_th_eff=model.material.iface_state_v_th,
        cross_transmission=model.material.iface_qss_cross_transmission,
        interface_transport_model=model.material.iface_qss_transport_model,
        residual_tolerance=model.interface_residual_tolerance,
    )

    results = tuple(
        run_ion_interface_backward_euler_reference(
            model,
            np.linspace(0.0, final_time_s, steps + 1),
            initial=initial,
            residual_tolerance=1.0e-8,
            max_newton_iterations=24,
        )
        for steps in (2, 4, 8)
    )
    carrier_errors: list[float] = []
    positive_ion_errors: list[float] = []
    interface_errors: list[float] = []
    potential_errors: list[float] = []
    for result in results:
        terminal = StateVec.unpack(result.physical_states[-1], grid.size)
        carrier_errors.append(
            max(
                float(np.max(np.abs(np.log(terminal.n / mol_terminal.n)))),
                float(np.max(np.abs(np.log(terminal.p / mol_terminal.p)))),
            )
        )
        positive_ion_errors.append(
            float(np.max(np.abs((terminal.P - mol_terminal.P) / mol_terminal.P)))
        )
        interface_errors.append(
            float(
                np.max(
                    np.abs(result.interface_states_m3[-1] - mol_interface.state_m3)
                    / mol_interface.state_m3
                )
            )
        )
        potential_errors.append(
            float(np.max(np.abs(result.potentials_V[-1] - mol_potential)))
        )
        assert result.max_normalized_carrier_residual <= 1.0e-8
        assert result.max_normalized_positive_ion_residual <= 1.0e-8
        assert result.max_normalized_interface_residual <= 1.0e-8
        assert result.max_normalized_algebraic_residual <= 1.0e-8
        assert result.max_relative_positive_ion_inventory_drift < 2.0e-15
        assert Q * result.max_interface_state_balance_m2_s < 1.0e-12

    for errors in (
        carrier_errors,
        positive_ion_errors,
        interface_errors,
        potential_errors,
    ):
        values = np.asarray(errors)
        assert np.all(np.diff(values) < 0.0), values
        np.testing.assert_allclose(values[1:] / values[:-1], 0.5, rtol=0.05)

    initial_state = StateVec.unpack(initial.physical_state, grid.size)
    assert np.max(np.abs(mol_terminal.P / initial_state.P - 1.0)) > 1.0e-6
    assert (
        np.max(np.abs(mol_interface.state_m3 / initial.interface_state_m3 - 1.0))
        > 1.0e-3
    )
    assert carrier_errors[-1] < 1.0e-7
    assert positive_ion_errors[-1] < 1.0e-5
    assert interface_errors[-1] < 1.0e-7
    assert potential_errors[-1] < 1.0e-8

    structured = run_ion_interface_backward_euler_reference(
        model,
        np.linspace(0.0, final_time_s, 9),
        initial=initial,
        residual_tolerance=1.0e-8,
        max_newton_iterations=24,
        jacobian_mode="structured_analytic",
    )
    np.testing.assert_allclose(
        structured.physical_states,
        results[-1].physical_states,
        rtol=2.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.interface_states_m3,
        results[-1].interface_states_m3,
        rtol=2.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.potentials_V,
        results[-1].potentials_V,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert structured.total_residual_evaluations < (
        results[-1].total_residual_evaluations / 20
    )
