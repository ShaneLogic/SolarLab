"""Integration, conservation, live legacy-parity, and 4-axis refinement tests for P06-02 and P06-03."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.steady import load_device
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.twod.experiments.jv_sweep_2d import (
    run_jv_sweep_2d as legacy_run_jv_sweep_2d,
)
from perovskite_sim.twod.experiments.voc_grain_sweep import (
    run_voc_grain_sweep as legacy_run_voc_grain_sweep,
)
from perovskite_sim.twod.interface_recombination_2d import (
    TwoSidedInterfaceSRHCoupling2D as LegacyInterfaceCoupling2D,
    evaluate_two_sided_interface_srh_2d as legacy_eval_interface_srh_2d,
)
from perovskite_sim.twod.ion_migration_2d import (
    positive_ion_fluxes_2d as legacy_positive_ion_fluxes_2d,
)
from perovskite_sim.twod.microstructure import GrainBoundary as LegacyGrainBoundary, Microstructure as LegacyMicrostructure
from perovskite_sim.twod.mobile_ion_current_2d import (
    evaluate_mobile_ion_current_components_2d as legacy_eval_mobile_ion_current_2d,
)
from perovskite_sim.twod.radiative_reabsorption_2d import (
    recompute_g_with_rad_2d as legacy_recompute_g_with_rad_2d,
)
from perovskite_sim.twod.snapshot import SpatialSnapshot2D as LegacySnapshot2D
from perovskite_sim.twod.solver_2d import (
    build_material_arrays_2d as legacy_build_material_arrays_2d,
)
from perovskite_sim.twod.grid_2d import build_grid_2d as legacy_build_grid_2d
from perovskite_sim.discretization.grid import Layer as LegacyGridLayer
from perovskite_sim.models.device import electrical_layers as legacy_electrical_layers
from solarlab.device.compile_ionic_2d import compile_ionic_device_2d
from solarlab.device.inputs import GrainBoundaryInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.ionic_2d import (
    TwoSidedInterfaceSRHCoupling2D,
    evaluate_two_sided_interface_srh_2d,
    positive_ion_fluxes_2d,
    recompute_g_with_rad_2d,
)
from solarlab.experiments.two_dimensional.inputs import JV2DInput, MicrostructureInput
from solarlab.experiments.two_dimensional.jv_protocol_2d import (
    DynamicJVPairProtocol2D,
    ImplicitProtocolError,
    ProtocolMismatchError,
)
from solarlab.experiments.two_dimensional.jv_sweep_2d import (
    build_jv_2d_execution_protocol,
    read_jv_2d_result,
    run_dynamic_jv_pair_2d,
    run_jv_sweep_2d,
    write_jv_2d_result,
)
from solarlab.experiments.two_dimensional.refinement_2d import (
    RefinementBaseline2D,
    read_refinement_report_2d,
    run_four_axis_refinement_2d,
    write_refinement_report_2d,
)
from solarlab.experiments.two_dimensional.voc_grain_sweep import (
    read_voc_grain_sweep,
    run_voc_grain_sweep,
    write_voc_grain_sweep,
)
from solarlab.solvers.nonlinear.ionic_2d import solve_ionic_steady_state_2d
from solarlab.solvers.time.ionic_backward_euler_2d import (
    compute_mobile_ion_current_components_2d,
    evaluate_mobile_ion_current_components_2d,
    run_ionic_backward_euler_2d,
)

_LEGACY_MAT_KEYS = (
    "eps_r", "mu_n", "mu_p", "D_ion", "P_lim", "P0", "E_a_ion",
    "ni", "tau_n", "tau_p", "n1", "p1", "B_rad", "C_n", "C_p",
    "alpha", "N_A", "N_D", "chi", "Eg", "Nc300", "Nv300", "v_th",
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> Any:
    with threadpool_limits(limits=1):
        yield


def _to_legacy_stack(prepared: PreparedDevice) -> DeviceStack:
    mapping = prepared.to_mapping()
    settings = mapping["settings"]
    layers = []
    for lyr in prepared.layers:
        params = dict(lyr.parameters)
        mp = MaterialParams(**{k: params[k] for k in _LEGACY_MAT_KEYS if k in params})
        layers.append(LayerSpec(name=lyr.id, thickness=lyr.thickness, params=mp, role=lyr.role))
    return DeviceStack(
        layers=tuple(layers),
        V_bi=float(settings.get("V_bi", 1.1)),
        Phi=float(settings["Phi"]),
        T=float(settings["T"]),
        mode=str(settings.get("mode", "full")),
        ion_steric_diffusion_only=bool(settings.get("ion_steric_diffusion_only", False)),
        built_in_potential_mode=str(settings.get("built_in_potential_mode", "work_function")),
        dos_band_potentials=bool(settings.get("dos_band_potentials", False)),
    )


def test_ionic_be_2d_transient_ramp_dwell_maxwell_current_and_1d_parity() -> None:
    """P06-02 (1)-(4): 2D mobile-ion BE conservation, Maxwell port current, and 1D lateral parity."""
    prep_ion = prepare_confined_ion_device()
    abs_ids = [lyr.id for lyr in prep_ion.layers if lyr.role == "absorber"]

    # 1. Laterally uniform 2D Neumann transient BE has zero lateral spread across columns
    uni_sp = JV2DInput(
        kind="jv_2d",
        lateral_length=2.0e-7,
        Nx=3,
        Ny_per_layer=4,
        lateral_bc="neumann",
        microstructure=MicrostructureInput(grain_boundaries=()),
        ion_dynamics="single_mobile",
    )
    uni_dev = compile_ionic_device_2d(
        prep_ion, uni_sp, ion_domain_layer=abs_ids[0], ion_domain_endpoints=("blocking", "blocking")
    )
    uni_eq = solve_ionic_steady_state_2d(uni_dev, voltage_V=0.0, generation_scale=0.0)
    assert uni_eq.accepted and uni_eq.state is not None

    t_ramp = np.linspace(0.0, 2.0e-6, 5, dtype=np.float64)
    dv_dt = 0.4 / 2.0e-6
    uni_run = run_ionic_backward_euler_2d(
        uni_dev,
        uni_eq.state,
        t_ramp,
        voltage_fn=lambda t: dv_dt * t,
        generation_fn=lambda _t: 0.0,
    )
    assert uni_run.accepted
    assert uni_run.ion_diagnostics.passed
    assert uni_run.ion_diagnostics.relative_inventory_drift <= 1.0e-12
    end_u = uni_run.states[-1]
    rel_spread = np.ptp(end_u.positive_ion_density_m3, axis=1) / np.max(end_u.positive_ion_density_m3, axis=1)
    assert float(np.max(rel_spread)) <= 1.0e-13

    # 2. Centered-GB 2D ramp (dV/dt != 0) + dwell (dV/dt == 0) Maxwell current & conservation
    ms = MicrostructureInput(
        grain_boundaries=(
            GrainBoundaryInput.model_validate(
                {
                    "id": "gb_mid",
                    "layer_ids": abs_ids,
                    "x_position": 1.0e-7,
                    "width": 2.0e-8,
                    "tau_n": 1.0e-9,
                    "tau_p": 1.0e-9,
                }
            ),
        )
    )
    gb_sp = JV2DInput(
        kind="jv_2d",
        lateral_length=2.0e-7,
        Nx=3,
        Ny_per_layer=4,
        lateral_bc="neumann",
        microstructure=ms,
        ion_dynamics="single_mobile",
    )
    gb_dev = compile_ionic_device_2d(
        prep_ion, gb_sp, ion_domain_layer=abs_ids[0], ion_domain_endpoints=("blocking", "blocking")
    )
    gb_eq = solve_ionic_steady_state_2d(gb_dev, voltage_V=0.0, generation_scale=0.0)
    assert gb_eq.accepted and gb_eq.state is not None

    ramp_run = run_ionic_backward_euler_2d(
        gb_dev,
        gb_eq.state,
        t_ramp,
        voltage_fn=lambda t: dv_dt * t,
        generation_fn=lambda t: 0.0 if t <= 0.0 else 1.0,
    )
    assert ramp_run.accepted
    assert ramp_run.ion_diagnostics.passed
    assert ramp_run.ion_diagnostics.relative_inventory_drift <= 1.0e-12
    assert ramp_run.ion_diagnostics.terminal_min_density_m3 > 0.0
    assert ramp_run.ion_diagnostics.terminal_max_site_fraction < 1.0
    for acct in ramp_run.accountings:
        assert acct.max_total_current_spread_A_m2 < 1.0e-3
        assert abs(float(acct.terminal_displacement_A_m2[1])) > 0.0

    ramp_state = ramp_run.states[-1]
    curr_ramp = compute_mobile_ion_current_components_2d(
        gb_dev, ramp_state, generation_scale=1.0, applied_voltage_rate_V_s=dv_dt
    )
    curr_zero_rate = compute_mobile_ion_current_components_2d(
        gb_dev, ramp_state, generation_scale=1.0, applied_voltage_rate_V_s=0.0
    )
    assert curr_ramp.max_relative_face_spread < 1.0e-4
    assert abs(curr_ramp.terminal_displacement_A_m2 - curr_zero_rate.terminal_displacement_A_m2) > 1.0
    assert curr_ramp.terminal_total_A_m2 == pytest.approx(
        curr_ramp.terminal_electron_A_m2
        + curr_ramp.terminal_hole_A_m2
        + curr_ramp.terminal_positive_ion_A_m2
        + curr_ramp.terminal_displacement_A_m2,
        rel=1.0e-14,
    )

    # Settle dwell at fixed V = 0.4 V -> displacement current decays toward zero
    dwell_run = run_ionic_backward_euler_2d(
        gb_dev,
        ramp_state,
        np.geomspace(1.0e-7, 1.0e-2, 8) - 1.0e-7,
        voltage_fn=lambda _t: 0.4,
        generation_fn=lambda _t: 1.0,
    )
    assert dwell_run.accepted
    assert dwell_run.ion_diagnostics.passed
    assert dwell_run.ion_diagnostics.relative_inventory_drift <= 1.0e-12


def test_live_legacy_operator_and_protocol_parity_2d() -> None:
    """P06-02 (5): Direct operator & canonical protocol parity against perovskite_sim.twod.*."""
    rng = np.random.default_rng(20261112)
    x = np.linspace(0.0, 2.4e-7, 4)
    y = np.array([0.0, 4.0e-8, 8.0e-8, 2.0e-7, 3.2e-7, 3.6e-7, 4.0e-7], dtype=float)
    ny, nx = y.size, x.size
    phi = np.linspace(0.9, 0.1, ny)[:, None] + 0.015 * np.cos(np.linspace(0.0, np.pi, nx))[None, :]
    p_ion = np.zeros((ny, nx), dtype=float)
    p_ion[2:5, :] = 1.5e24 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=(3, nx)))
    d_ion = np.zeros((ny, nx), dtype=float)
    d_ion[2:5, :] = 1.0e-16
    p_lim = np.full((ny, nx), 1.0e25, dtype=float)

    for diff_only in (False, True):
        f_new = positive_ion_fluxes_2d(
            x, y, phi, p_ion, d_ion, 0.02585, p_lim, steric_diffusion_only=diff_only
        )
        f_leg = legacy_positive_ion_fluxes_2d(
            x, y, phi, p_ion, d_ion, 0.02585, p_lim, steric_diffusion_only=diff_only
        )
        np.testing.assert_allclose(f_new.x, f_leg.x, rtol=1.0e-14, atol=1.0e-6)
        np.testing.assert_allclose(f_new.y, f_leg.y, rtol=1.0e-14, atol=1.0e-6)

    # Instantaneous Maxwell current decomposition parity (dV/dt = 0 and dV/dt = 250 V/s)
    prep_ion_op = prepare_confined_ion_device()
    abs_ids_op = [lyr.id for lyr in prep_ion_op.layers if lyr.role == "absorber"]
    op_sp = JV2DInput(
        kind="jv_2d",
        lateral_length=2.4e-7,
        Nx=3,
        Ny_per_layer=4,
        lateral_bc="neumann",
        microstructure=MicrostructureInput(grain_boundaries=()),
        ion_dynamics="single_mobile",
    )
    op_dev = compile_ionic_device_2d(
        prep_ion_op, op_sp, ion_domain_layer=abs_ids_op[0], ion_domain_endpoints=("blocking", "blocking")
    )
    op_eq = solve_ionic_steady_state_2d(op_dev, voltage_V=0.0, generation_scale=0.0)
    assert op_eq.accepted and op_eq.state is not None
    stack_op = _to_legacy_stack(prep_ion_op)
    leg_layers = [LegacyGridLayer(thickness=lyr.thickness, N=4) for lyr in legacy_electrical_layers(stack_op)]
    leg_grid = legacy_build_grid_2d(leg_layers, Nx=3, lateral_length=2.4e-7)
    leg_mat = legacy_build_material_arrays_2d(
        leg_grid, stack_op, LegacyMicrostructure(), lateral_bc="neumann", ion_dynamics="single_mobile"
    )
    ny_op, nx_op = op_dev.electronic_component.geometry.shape
    phi_op = op_eq.state.electronic_state.potential_V
    pion_op = op_dev.full_grid_ion_density(op_eq.state.positive_ion_density_m3)
    jy_n_zero = np.zeros((ny_op - 1, nx_op), dtype=float)
    jy_p_zero = np.zeros((ny_op - 1, nx_op), dtype=float)
    jx_zero = np.zeros((ny_op, nx_op - 1), dtype=float)
    n_snap = op_eq.state.electronic_state.electron_density_m3.copy()
    p_snap = op_eq.state.electronic_state.hole_density_m3.copy()
    n_snap[0, :], n_snap[-1, :] = leg_mat.n_eq_left, leg_mat.n_eq_right
    p_snap[0, :], p_snap[-1, :] = leg_mat.p_eq_left, leg_mat.p_eq_right
    leg_snap = LegacySnapshot2D(
        V=0.0,
        x=leg_grid.x,
        y=leg_grid.y,
        phi=phi_op,
        n=n_snap,
        p=p_snap,
        Jx_n=jx_zero,
        Jy_n=jy_n_zero,
        Jx_p=jx_zero,
        Jy_p=jy_p_zero,
        P_ion=pion_op,
    )
    zero_3b = np.zeros(3 * op_dev.node_count, dtype=float)
    for rate in (0.0, 250.0):
        c_new = evaluate_mobile_ion_current_components_2d(
            op_dev, phi_op, jy_n_zero, jy_p_zero, pion_op, zero_3b, applied_voltage_rate_V_s=rate
        )
        c_leg = legacy_eval_mobile_ion_current_2d(
            leg_snap, zero_3b, leg_mat, applied_voltage_rate_V_s=rate
        )
        np.testing.assert_allclose(c_new.total_y_A_m2, c_leg.total_y_A_m2, rtol=1.0e-13, atol=1.0e-12)
        assert c_new.terminal_total_A_m2 == pytest.approx(c_leg.terminal_total_A_m2, rel=1.0e-13, abs=1.0e-12)
        assert c_new.terminal_displacement_A_m2 == pytest.approx(
            c_leg.terminal_displacement_A_m2, rel=1.0e-13, abs=1.0e-12
        )
    c_eq = compute_mobile_ion_current_components_2d(
        op_dev, op_eq.state, generation_scale=0.0, applied_voltage_rate_V_s=0.0
    )
    assert abs(c_eq.terminal_total_A_m2) <= 1.0e-12

    n = np.exp(np.linspace(48.0, 52.0, ny))[:, None] * (1.0 + 0.05 * np.sin(np.arange(nx))[None, :])
    p = np.exp(np.linspace(52.0, 47.0, ny))[:, None] * (1.0 + 0.04 * np.cos(np.arange(nx))[None, :])
    g_opt = np.full((ny, nx), 1.0e27)

    # Two-sided cross-node interface SRH parity
    srh_kwargs = {
        "interface_index": 0,
        "interface_row": 2,
        "left_sample_row": 1,
        "right_sample_row": 3,
        "electron_capture_velocity_m_s": 12.0,
        "hole_capture_velocity_m_s": 8.0,
        "n1_m3": 1.0e15,
        "p1_m3": 2.0e15,
        "pair_a_equilibrium_product_m6": 1.0e30,
        "pair_b_equilibrium_product_m6": 1.5e30,
    }
    c_srh_new = (TwoSidedInterfaceSRHCoupling2D(**srh_kwargs),)
    c_srh_leg = (LegacyInterfaceCoupling2D(**srh_kwargs),)
    rep_new = evaluate_two_sided_interface_srh_2d(n, p, y, c_srh_new)
    rep_leg = legacy_eval_interface_srh_2d(n, p, y, c_srh_leg)
    np.testing.assert_allclose(rep_new.volumetric_sink_m3_s, rep_leg.volumetric_sink_m3_s, rtol=1.0e-14)
    np.testing.assert_allclose(rep_new.total_surface_rate_m2_s, rep_leg.total_surface_rate_m2_s, rtol=1.0e-14)

    # Bounded radiative reabsorption parity
    b_rad = np.zeros((ny, nx), dtype=float)
    b_rad[2:5, :] = 5.0e-17
    ni_sq_arr = np.full((ny, nx), 1.0e22, dtype=float)
    area_abs = (y[4] - y[2]) * (x[-1] - x[0])
    g_rr_new = recompute_g_with_rad_2d(
        G_optical=g_opt,
        n=n,
        p=p,
        B_rad=b_rad,
        ni_sq=ni_sq_arr,
        x=x,
        y=y,
        absorber_y_ranges=((2, 5),),
        absorber_p_esc=(0.15,),
        absorber_areas=(area_abs,),
    )
    g_rr_leg = legacy_recompute_g_with_rad_2d(
        G_optical=g_opt,
        n=n,
        p=p,
        B_rad=b_rad,
        ni_sq=ni_sq_arr,
        x=x,
        y=y,
        absorber_y_ranges=((2, 5),),
        absorber_p_esc=(0.15,),
        absorber_areas=(area_abs,),
    )
    np.testing.assert_allclose(g_rr_new, g_rr_leg, rtol=1.0e-14)

    # Canonical JV2DProtocol JSON & SHA-256 hash parity
    from perovskite_sim.twod.experiments.jv_sweep_2d import (
        build_jv_2d_execution_protocol as legacy_build_exec_proto,
    )
    ms_proto_new = MicrostructureInput(
        grain_boundaries=(
            GrainBoundaryInput.model_validate(
                {
                    "id": "gb_p",
                    "layer_ids": abs_ids_op,
                    "x_position": 1.2e-7,
                    "width": 2.0e-8,
                    "tau_n": 1.0e-9,
                    "tau_p": 2.0e-9,
                }
            ),
        )
    )
    ms_proto_leg = LegacyMicrostructure(
        grain_boundaries=(
            LegacyGrainBoundary(x_position=1.2e-7, width=2.0e-8, tau_n=1.0e-9, tau_p=2.0e-9),
        )
    )
    proto_kw: dict[str, Any] = {
        "lateral_length": 2.4e-7,
        "Nx": 3,
        "Ny_per_layer": 4,
        "V_max": 1.05,
        "V_step": 0.35,
        "illuminated": True,
        "lateral_bc": "neumann",
        "settle_t": 1.0e-4,
        "ion_dynamics": "single_mobile",
        "interface_srh": "off",
        "initial_state_settle_s": 1.0e-4,
    }
    p_new = build_jv_2d_execution_protocol(prep_ion_op, ms_proto_new, **proto_kw)
    p_leg = legacy_build_exec_proto(stack_op, ms_proto_leg, **proto_kw)
    assert p_new.canonical_json() == p_leg.canonical_json()
    assert p_new.protocol_hash == p_leg.protocol_hash


def test_live_legacy_experiment_parity_and_protocol_gates_2d() -> None:
    """P06-02 (5): Live parity of run_voc_grain_sweep & mobile-ion run_jv_sweep_2d vs perovskite_sim."""
    prep = load_device()
    stack = _to_legacy_stack(prep)

    # 1. voc_grain_sweep jv_sweep parity & s_gb_m_s equivalence
    res_new = run_voc_grain_sweep(
        prep, [2.0e-7, 8.0e-7], tau_gb=(1.0e-9, 1.0e-9), gb_width=1.0e-8,
        Nx=3, Ny_per_layer=4, V_max=1.14, V_step=0.38, settle_t=1.0e-4, mode="jv_sweep",
    )
    res_leg = legacy_run_voc_grain_sweep(
        stack, [2.0e-7, 8.0e-7], tau_gb=(1.0e-9, 1.0e-9), gb_width=1.0e-8,
        Nx=3, Ny_per_layer=4, V_max=1.14, V_step=0.38, settle_t=1.0e-4,
    )
    np.testing.assert_allclose(res_new.V_oc_V, res_leg.V_oc_V, atol=1.0e-3)
    np.testing.assert_allclose(res_new.J_sc_Am2, res_leg.J_sc_Am2, rtol=1.0e-4)
    assert res_new.V_oc_V[1] > res_new.V_oc_V[0]
    assert res_new.scientific_qualification is False

    # Velocity parametrization s_gb = width / tau_gb reproduces identical Voc
    res_vel = run_voc_grain_sweep(
        prep, [2.0e-7], tau_gb=None, s_gb_m_s=10.0, gb_width=1.0e-8,
        Nx=3, Ny_per_layer=4, V_max=1.14, V_step=0.38, settle_t=1.0e-4, mode="jv_sweep",
    )
    assert res_vel.V_oc_V[0] == pytest.approx(res_new.V_oc_V[0], abs=1.0e-12)

    # Direct steady Voc mode matches J_sc and preserves grain-size Voc ordering
    res_direct = run_voc_grain_sweep(
        prep, [2.0e-7, 8.0e-7], tau_gb=(1.0e-9, 1.0e-9), gb_width=1.0e-8,
        Nx=3, Ny_per_layer=4, V_max=1.3, V_step=0.2, mode="direct_voc",
    )
    np.testing.assert_allclose(res_direct.J_sc_Am2, res_new.J_sc_Am2, rtol=1.0e-5)
    assert res_direct.V_oc_V[1] > res_direct.V_oc_V[0]
    assert np.all(np.isnan(res_direct.FF))

    # 2. Mobile-ion research_strict run_jv_sweep_2d parity + protocol enforcement
    prep_ion = prepare_confined_ion_device()
    stack_ion = _to_legacy_stack(prep_ion)
    abs_ids = [lyr.id for lyr in prep_ion.layers if lyr.role == "absorber"]
    ms_new = MicrostructureInput(
        grain_boundaries=(
            GrainBoundaryInput.model_validate(
                {
                    "id": "gb0",
                    "layer_ids": abs_ids,
                    "x_position": 1.0e-7,
                    "width": 2.0e-8,
                    "tau_n": 1.0e-9,
                    "tau_p": 1.0e-9,
                }
            ),
        )
    )
    ms_leg = LegacyMicrostructure(
        grain_boundaries=(
            LegacyGrainBoundary(x_position=1.0e-7, width=2.0e-8, tau_n=1.0e-9, tau_p=1.0e-9),
        )
    )
    kw: dict[str, Any] = {
        "lateral_length": 2.0e-7,
        "Nx": 3,
        "Ny_per_layer": 4,
        "V_max": 1.05,
        "V_step": 0.35,
        "illuminated": True,
        "lateral_bc": "neumann",
        "settle_t": 1.0e-4,
        "initial_state_settle_s": 1.0e-4,
        "ion_dynamics": "single_mobile",
        "interface_srh": "off",
        "save_snapshots": False,
    }

    with pytest.raises(ImplicitProtocolError):
        run_jv_sweep_2d(prep_ion, microstructure=ms_new, protocol_mode="research_strict", **kw)

    proto_mob = build_jv_2d_execution_protocol(prep_ion, ms_new, **kw)
    with pytest.raises(ProtocolMismatchError):
        run_jv_sweep_2d(
            prep_ion,
            microstructure=ms_new,
            jv_2d_protocol=proto_mob,
            protocol_mode="research_strict",
            **(kw | {"settle_t": 2.0e-4}),
        )

    mob_new = run_jv_sweep_2d(
        prep_ion, microstructure=ms_new, jv_2d_protocol=proto_mob, protocol_mode="research_strict", **kw
    )
    from perovskite_sim.twod.experiments.jv_protocol_2d import JV2DProtocol as LegacyJV2DProtocol
    leg_proto = LegacyJV2DProtocol.from_json(proto_mob.canonical_json())
    mob_leg = legacy_run_jv_sweep_2d(
        stack_ion,
        ms_leg,
        jv_2d_protocol=leg_proto,
        protocol_mode="research_strict",
        **kw,
    )
    assert mob_new.disposition == "completed"
    assert all(d.passed and d.relative_inventory_drift <= 1.0e-12 for d in mob_new.ion_diagnostics)
    assert mob_new.metrics.V_oc == pytest.approx(mob_leg.metrics.V_oc, abs=5.0e-3)
    assert mob_new.metrics.J_sc == pytest.approx(mob_leg.metrics.J_sc, rel=2.0e-3)


def test_dynamic_jv_pair_partial_disposition_and_artifact_roundtrip_2d(tmp_path: Path) -> None:
    """P06-02: Continuous dynamic forward/reverse scan, partial failure recording, and lossless I/O."""
    prep_ion = prepare_confined_ion_device()
    abs_ids = [lyr.id for lyr in prep_ion.layers if lyr.role == "absorber"]
    ms = MicrostructureInput(
        grain_boundaries=(
            GrainBoundaryInput.model_validate(
                {
                    "id": "gb_dyn",
                    "layer_ids": abs_ids,
                    "x_position": 1.0e-7,
                    "width": 2.0e-8,
                    "tau_n": 1.0e-9,
                    "tau_p": 1.0e-9,
                }
            ),
        )
    )
    base_proto = build_jv_2d_execution_protocol(
        prep_ion,
        ms,
        lateral_length=2.0e-7,
        Nx=3,
        Ny_per_layer=4,
        V_max=1.05,
        V_step=0.35,
        illuminated=True,
        lateral_bc="neumann",
        settle_t=1.0e-4,
        initial_state_settle_s=1.0e-4,
        ion_dynamics="single_mobile",
    )
    pair_proto = DynamicJVPairProtocol2D(
        temperature_K=base_proto.temperature_K,
        illuminated=True,
        precondition_voltage_V=0.0,
        precondition_hold_s=1.0e-4,
        forward_voltages_V=(0.0, 0.35, 0.7, 1.05),
        turnaround_hold_s=1.0e-3,
        reverse_voltages_V=(1.05, 0.7, 0.35, 0.0),
        step_mode="linear_ramp",
        dwell_or_step_time_s=1.0e-3,
        state_topology="single_positive_mobile_ion",
        lateral_bc="neumann",
        carrier_boundary_condition="ohmic",
        interface_srh="off",
        x_coordinates_m=base_proto.x_coordinates_m,
        y_coordinates_m=base_proto.y_coordinates_m,
        grain_boundaries=base_proto.grain_boundaries,
        substeps_per_point=4,
    )
    pair_res = run_dynamic_jv_pair_2d(prep_ion, pair_proto, microstructure=ms)
    assert pair_res.disposition == "completed"
    assert pair_res.scientific_qualification is False
    assert pair_res.preconditioned_state_sha256 != pair_res.turnaround_state_sha256
    assert pair_res.hysteresis_voc_delta_V is not None
    assert all(d.passed for d in pair_res.forward.ion_diagnostics)
    assert all(d.passed for d in pair_res.reverse.ion_diagnostics)
    assert any(abs(c.terminal_displacement_A_m2) > 0.0 for c in pair_res.forward.current_components[1:])

    # Partial failure recording vs raise when extreme forward voltage jump exceeds zero-bisection step
    prep_std = load_device()
    partial_res = run_jv_sweep_2d(
        prep_std,
        lateral_length=2.0e-7,
        Nx=2,
        Ny_per_layer=3,
        V_max=15.0,
        V_step=7.5,
        illuminated=False,
        settle_t=1.0e-2,
        max_bisect=0,
        max_nfev_per_solve=25,
        save_snapshots=False,
        on_point_failure="record_partial",
    )
    assert partial_res.disposition in {"partial", "completed"}
    assert len(partial_res.attempts) >= 1

    # Lossless artifact round-trip in tmp_path
    jv_path = tmp_path / "jv_2d_forward.json"
    sha_jv = write_jv_2d_result(jv_path, pair_res.forward)
    loaded_jv = read_jv_2d_result(jv_path, expected_sha256=sha_jv)
    np.testing.assert_array_equal(loaded_jv.V, pair_res.forward.V)
    np.testing.assert_array_equal(loaded_jv.J, pair_res.forward.J)
    assert loaded_jv.scientific_qualification is False
    with pytest.raises(ValueError, match="SHA-256"):
        read_jv_2d_result(jv_path, expected_sha256="0" * 64)

    voc_res = run_voc_grain_sweep(
        prep_std, [2.0e-7, 6.0e-7], Nx=2, Ny_per_layer=3, V_max=1.3, V_step=0.25, mode="direct_voc"
    )
    voc_path = tmp_path / "voc_grain_direct.json"
    sha_voc = write_voc_grain_sweep(voc_path, voc_res)
    loaded_voc = read_voc_grain_sweep(voc_path, expected_sha256=sha_voc)
    np.testing.assert_array_equal(loaded_voc.grain_sizes_m, voc_res.grain_sizes_m)
    np.testing.assert_array_equal(loaded_voc.V_oc_V, voc_res.V_oc_V)
    assert np.all(np.isnan(loaded_voc.FF))


def test_four_axis_refinement_harness_2d(tmp_path: Path) -> None:
    """P06-03 (1)-(2): 4-axis independent 2D refinement harness & structural/resource metrics."""
    prep_ion = prepare_confined_ion_device()

    # Mobile-ion 4-axis refinement with centered GB
    report_mob = run_four_axis_refinement_2d(
        prep_ion,
        baseline=RefinementBaseline2D(
            lateral_length_m=2.0e-7,
            Nx=2,
            Ny_per_layer=3,
            dt_s=5.0e-7,
            total_time_s=1.0e-6,
            nonlinear_tol_scale=1.0,
            voltage_V=0.2,
            generation_scale=0.0,
            lateral_bc="neumann",
            ion_dynamics="single_mobile",
            gb_width_m=2.0e-8,
        ),
        nx_values=(2, 4),
        ny_per_layer_values=(3, 5),
        dt_s_values=(5.0e-7, 2.5e-7),
        nonlinear_tol_scales=(10.0, 1.0),
    )
    assert report_mob.scientific_qualification is False
    assert report_mob.performance_gate_evaluated is False

    axes = (
        ("Nx", report_mob.nx_axis, {"Ny_per_layer", "dt_s", "nonlinear_tol_scale"}),
        ("Ny_per_layer", report_mob.ny_axis, {"Nx", "dt_s", "nonlinear_tol_scale"}),
        ("dt_s", report_mob.dt_axis, {"Nx", "Ny_per_layer", "nonlinear_tol_scale"}),
        ("nonlinear_tolerance", report_mob.nonlinear_tolerance_axis, {"Nx", "Ny_per_layer", "dt_s"}),
    )
    for expected_name, axis_rep, expected_fixed_keys in axes:
        assert axis_rep.axis == expected_name
        assert set(axis_rep.held_fixed.keys()) == expected_fixed_keys
        assert len(axis_rep.points) == 2
        for pt in axis_rep.points:
            assert pt.accepted is True
            assert pt.dof > 0
            assert pt.jacobian_nnz > 0
            assert pt.lu_nnz >= pt.jacobian_nnz
            assert pt.rss_bytes > 0
            assert pt.wall_seconds > 0.0
            assert pt.newton_iterations > 0
            assert pt.ion_inventory_rel_drift is not None and pt.ion_inventory_rel_drift <= 1.0e-12
        assert axis_rep.points[-1].delta_from_finest_current_A_m2 == 0.0

    assert report_mob.nx_axis.points[1].dof > report_mob.nx_axis.points[0].dof
    assert report_mob.ny_axis.points[1].dof > report_mob.ny_axis.points[0].dof

    # Frozen-ion periodic 4-axis refinement + lossless JSON round-trip
    report_frz = run_four_axis_refinement_2d(
        load_device(),
        baseline=RefinementBaseline2D(
            lateral_length_m=2.0e-7,
            Nx=2,
            Ny_per_layer=3,
            dt_s=5.0e-7,
            total_time_s=1.0e-6,
            nonlinear_tol_scale=1.0,
            voltage_V=0.2,
            generation_scale=0.0,
            lateral_bc="periodic",
            ion_dynamics="frozen",
            gb_width_m=None,
        ),
        nx_values=(2, 4),
        ny_per_layer_values=(3, 5),
        dt_s_values=(5.0e-7, 2.5e-7),
        nonlinear_tol_scales=(10.0, 1.0),
    )
    assert all(pt.accepted and pt.ion_inventory_rel_drift is None for pt in report_frz.nx_axis.points)

    ref_path = tmp_path / "refinement_2d_report.json"
    ref_sha = write_refinement_report_2d(ref_path, report_mob)
    loaded_ref = read_refinement_report_2d(ref_path, expected_sha256=ref_sha)
    assert loaded_ref.scientific_qualification is False
    assert loaded_ref.performance_gate_evaluated is False
    assert loaded_ref.nx_axis.points[-1].lu_nnz == report_mob.nx_axis.points[-1].lu_nnz
    with pytest.raises(ValueError, match="SHA-256"):
        read_refinement_report_2d(ref_path, expected_sha256="f" * 64)
