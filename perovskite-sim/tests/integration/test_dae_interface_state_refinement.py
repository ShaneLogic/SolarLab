from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.interface_plane import solve_interface_states_live_qss
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae_interface_integrator import (
    run_algebraic_interface_backward_euler_reference,
)
from perovskite_sim.solver.dae_interface_states import (
    build_algebraic_interface_consistent_initial_condition,
    build_algebraic_interface_state_dae,
)
from perovskite_sim.solver.mol import (
    StateVec,
    _charge_density,
    poisson_right_boundary,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import NumericalDiagnosticsPolicy


def _charge_off_interface_slice():
    source = load_device_from_yaml("configs/interface_charge_research.yaml")
    stack = replace(
        source,
        interface_defects=(),
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=False,
        Phi=0.0,
    )
    grid = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in stack.layers],
        alpha=1.0,
    )
    model = build_algebraic_interface_state_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=0.01,
        carrier_reference_time_s=1.0e-7,
    )
    initial = build_algebraic_interface_consistent_initial_condition(model)
    return stack, grid, model, initial


def test_backward_euler_contracts_to_eliminated_qss_mol_reference():
    stack, grid, model, initial = _charge_off_interface_slice()
    final_time_s = 1.0e-8
    mol = run_transient(
        grid,
        initial.physical_state,
        (0.0, final_time_s),
        np.array([0.0, final_time_s]),
        stack,
        illuminated=False,
        V_app=0.01,
        rtol=1.0e-8,
        atol=1.0e4,
        max_step=final_time_s / 20.0,
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
        run_algebraic_interface_backward_euler_reference(
            model,
            np.linspace(0.0, final_time_s, steps + 1),
            initial=initial,
            residual_tolerance=1.0e-8,
        )
        for steps in (2, 4, 8)
    )
    carrier_errors: list[float] = []
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
        assert result.max_normalized_interface_residual <= 1.0e-8
        assert result.max_normalized_algebraic_residual <= 1.0e-8

    for errors in (carrier_errors, interface_errors, potential_errors):
        assert errors[1] < 0.1 * errors[0]
        assert errors[2] < 0.1 * errors[1]
    assert carrier_errors[-1] < 1.0e-5
    assert interface_errors[-1] < 1.0e-5
    assert potential_errors[-1] < 1.0e-12
    initial_state = StateVec.unpack(initial.physical_state, grid.size)
    assert np.max(
        np.abs(np.log(mol_terminal.n / initial_state.n))
    ) > 1.0e-4
    assert Q * results[-1].max_interface_state_balance_m2_s < 1.0e-12

    structured = run_algebraic_interface_backward_euler_reference(
        model,
        np.linspace(0.0, final_time_s, 9),
        initial=initial,
        residual_tolerance=1.0e-8,
        jacobian_mode="structured_analytic",
    )
    np.testing.assert_allclose(
        structured.physical_states,
        results[-1].physical_states,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.interface_states_m3,
        results[-1].interface_states_m3,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.potentials_V,
        results[-1].potentials_V,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert structured.total_residual_evaluations < (
        results[-1].total_residual_evaluations / 20
    )
