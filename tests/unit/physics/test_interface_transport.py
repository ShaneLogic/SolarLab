"""Unit tests for 1D thermionic capping, supply integrals, and two-sided QSS traces (P04-09)."""

from __future__ import annotations

from collections.abc import Iterator
import math

import numpy as np
import pytest
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M as EPS_0
from solarlab.physics.electrostatics.interface_traces import (
    TraceElectrostatics,
    interface_displacement_sides,
    solve_electrostatic_traces,
)
from solarlab.physics.interface_transport import (
    A_STAR_FREE_ELECTRON_A_M2_K2,
    ChargedTwoSidedInterfaceQSSResult,
    EquilibriumReferencedSheetCharge,
    TwoSidedBulkState,
    TwoSidedInterfacePhysics,
    cap_particle_flux_thermionic,
    effective_mass_from_dos,
    emission_velocity,
    fermi_dirac_one,
    fermi_dirac_richardson_flux,
    fermi_dirac_zero,
    fermi_richardson_flux,
    is_free_electron_default,
    remove_shared_interface_nodes,
    resolve_layer_richardson,
    richardson_from_dos,
    scaps_thermionic_flux,
    solve_charged_two_sided_interface_qss,
    solve_two_sided_interface_qss,
    thermionic_emission_bound_m2_s,
)
from solarlab.physics.recombination.interface import resolve_interface_defect_references
from solarlab.physics.statistics.fermi_dirac import fermi_dirac_half
from solarlab.physics.statistics.thermal import thermal_voltage


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def test_richardson_dos_relations_and_layer_resolution() -> None:
    assert is_free_electron_default(A_STAR_FREE_ELECTRON_A_M2_K2)
    assert is_free_electron_default(1.2017e6)
    assert not is_free_electron_default(0.0)
    assert not is_free_electron_default(2.5e5)

    dos = 2.5e24
    m_ratio = effective_mass_from_dos(dos, 300.0)
    assert 0.05 < m_ratio < 1.0
    a_star = richardson_from_dos(dos, 300.0)
    assert a_star == pytest.approx(A_STAR_FREE_ELECTRON_A_M2_K2 * m_ratio, rel=1e-15)
    assert emission_velocity(300.0, m_ratio) > emission_velocity(300.0, 1.0)

    assert resolve_layer_richardson(0.0, dos, 300.0) == 0.0
    assert resolve_layer_richardson(1.2017e6, dos, 300.0) == pytest.approx(a_star, rel=1e-15)
    assert resolve_layer_richardson(4.2e5, dos, 300.0) == pytest.approx(4.2e5, rel=1e-15)


@pytest.mark.parametrize("eta", [-60.0, -8.0, -4.0, -1.5, 0.0, 2.5, 15.0, 45.0])
def test_fermi_dirac_zero_and_one_quadrature_and_derivative(eta: float) -> None:
    f0 = fermi_dirac_zero(eta)
    f1 = fermi_dirac_one(eta)
    assert f0 >= 0.0
    assert f1 >= 0.0

    if -15.0 <= eta <= 25.0:
        quad_val, _ = quad(
            lambda x: 0.0 if (x - eta) > 700.0 else x / (1.0 + math.exp(x - eta)),
            0.0,
            math.inf,
            epsrel=1e-12,
        )
        assert f1 == pytest.approx(quad_val, rel=1e-11)

    h = 1.0e-5
    df1 = (fermi_dirac_one(eta + h) - fermi_dirac_one(eta - h)) / (2.0 * h)
    assert df1 == pytest.approx(f0, rel=1e-8, abs=1e-28)


@pytest.mark.parametrize("mode", ["hard", "legacy_logistic", "compact_support"])
@pytest.mark.parametrize("sg_flux", [-4.2e22, -1.5e20, 1.8e20, 5.6e22])
def test_thermionic_cap_direction_equilibrium_and_fd_derivatives(
    mode: str, sg_flux: float
) -> None:
    temp = 300.0
    vt = thermal_voltage(temp)
    barrier = 0.18
    a_star = 3.2e5

    # Exact zero bound at Boltzmann equilibrium across delta_barrier_eV:
    n_l_eq = 2.0e21
    n_r_eq = n_l_eq * math.exp(-barrier / vt)
    bound_eq, _, _ = thermionic_emission_bound_m2_s(n_l_eq, n_r_eq, barrier, temp, a_star)
    assert bound_eq == pytest.approx(0.0, abs=1e-8)

    # Non-equilibrium trial state where |sg_flux| and |te_bound| span both uncapped and capped
    # regimes (with dos_norm_m3 = 2.5e24 so |te_bound| ~ 2.9e22 m^-2 s^-1 is comparable to |sg_flux|!)
    n_l = 1.7e21
    n_r = 6.3e19
    dos_norm = 2.5e24
    soft = 0.0 if mode == "hard" else 0.05
    reg_mode = "legacy_logistic" if mode == "hard" else mode
    res = cap_particle_flux_thermionic(
        sg_flux,
        n_left_m3=n_l,
        n_right_m3=n_r,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    )
    assert math.copysign(1.0, res.flux_m2_s) == math.copysign(1.0, sg_flux)
    te_val, _, _ = thermionic_emission_bound_m2_s(
        n_l, n_r, barrier, temp, a_star, dos_norm_m3=dos_norm
    )
    if soft == 0.0:
        assert abs(res.flux_m2_s) == pytest.approx(min(abs(sg_flux), abs(te_val)), rel=1e-15)

    # Verify analytical partials wrt (sg_flux, n_left, n_right) against central FD.
    h_sg = 1.0e-6 * abs(sg_flux)
    fp_sg = cap_particle_flux_thermionic(
        sg_flux + h_sg,
        n_left_m3=n_l,
        n_right_m3=n_r,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    ).flux_m2_s
    fm_sg = cap_particle_flux_thermionic(
        sg_flux - h_sg,
        n_left_m3=n_l,
        n_right_m3=n_r,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    ).flux_m2_s

    h_nl = 1.0e-4 * n_l
    fp_nl = cap_particle_flux_thermionic(
        sg_flux,
        n_left_m3=n_l + h_nl,
        n_right_m3=n_r,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    ).flux_m2_s
    fm_nl = cap_particle_flux_thermionic(
        sg_flux,
        n_left_m3=n_l - h_nl,
        n_right_m3=n_r,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    ).flux_m2_s

    h_nr = 1.0e-4 * n_r
    fp_nr = cap_particle_flux_thermionic(
        sg_flux,
        n_left_m3=n_l,
        n_right_m3=n_r + h_nr,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    ).flux_m2_s
    fm_nr = cap_particle_flux_thermionic(
        sg_flux,
        n_left_m3=n_l,
        n_right_m3=n_r - h_nr,
        delta_barrier_eV=barrier,
        temperature_K=temp,
        a_star_A_m2_K2=a_star,
        te_softness=soft,
        regularization_mode=reg_mode,  # type: ignore[arg-type]
        dos_norm_m3=dos_norm,
    ).flux_m2_s

    assert res.d_flux_d_sg_flux == pytest.approx((fp_sg - fm_sg) / (2.0 * h_sg), rel=1e-7, abs=1e-12)
    assert res.d_flux_d_n_left_m_s == pytest.approx(
        (fp_nl - fm_nl) / (2.0 * h_nl),
        rel=1e-6,
        abs=1.0e-8 * max(abs(res.flux_m2_s) / n_l, 1.0e-12),
    )
    assert res.d_flux_d_n_right_m_s == pytest.approx(
        (fp_nr - fm_nr) / (2.0 * h_nr),
        rel=1e-6,
        abs=1.0e-8 * max(abs(res.flux_m2_s) / n_r, 1.0e-12),
    )


def test_cross_plane_flux_kernels_equilibrium_and_fd_jacobians() -> None:
    temp = 300.0
    vt = thermal_voltage(temp)
    nc_l, nc_r = 4.0e24, 2.5e24
    nv_l, nv_r = 6.0e24, 3.5e24
    cb_step = 0.14  # chi_L - chi_R
    vb_step = 0.22  # (chi_R + Eg_R) - (chi_L + Eg_L)
    phi_l, phi_r = 0.12, 0.21
    dphi = phi_r - phi_l

    # Fermi-Dirac thermal equilibrium: eta_n_R = eta_n_L - (cb_step - dphi)/vt,
    # eta_p_R = eta_p_L - (vb_step + dphi)/vt.
    eta_nl = -3.0
    eta_nr = eta_nl - (cb_step - dphi) / vt
    eta_pl = -4.0
    eta_pr = eta_pl - (vb_step + dphi) / vt
    eq_fd_state = np.array(
        [
            nc_l * fermi_dirac_half(eta_nl),
            nv_l * fermi_dirac_half(eta_pl),
            nc_r * fermi_dirac_half(eta_nr),
            nv_r * fermi_dirac_half(eta_pr),
        ],
        dtype=np.float64,
    )
    fd_eq = fermi_dirac_richardson_flux(
        eq_fd_state,
        phi_trace_left_V=phi_l,
        phi_trace_right_V=phi_r,
        conduction_band_step_eV=cb_step,
        hole_transport_step_eV=vb_step,
        N_C_left_m3=nc_l,
        N_C_right_m3=nc_r,
        N_V_left_m3=nv_l,
        N_V_right_m3=nv_r,
        richardson_n_A_m2_K2=2.5e5,
        richardson_p_A_m2_K2=3.5e5,
        temperature_K=temp,
        thermal_voltage_V=vt,
    )
    for c_idx in range(2):
        assert fd_eq.cross_flux_m2_s[c_idx] == pytest.approx(
            0.0, abs=1.0e-13 * float(fd_eq.one_way_scale_m2_s[c_idx])
        )

    # Boltzmann equilibrium for fermi_richardson_flux and scaps_thermionic_flux:
    n_l_b = 1.0e18
    n_r_b = n_l_b * (nc_r / nc_l) * math.exp(-(cb_step - dphi) / vt)
    p_l_b = 2.0e18
    p_r_b = p_l_b * (nv_r / nv_l) * math.exp(-(vb_step + dphi) / vt)
    eq_mb_state = np.array([n_l_b, p_l_b, n_r_b, p_r_b], dtype=np.float64)
    mb_eq = fermi_richardson_flux(
        eq_mb_state,
        phi_trace_left_V=phi_l,
        phi_trace_right_V=phi_r,
        conduction_band_step_eV=cb_step,
        hole_transport_step_eV=vb_step,
        N_C_left_m3=nc_l,
        N_C_right_m3=nc_r,
        N_V_left_m3=nv_l,
        N_V_right_m3=nv_r,
        richardson_n_A_m2_K2=2.5e5,
        richardson_p_A_m2_K2=3.5e5,
        temperature_K=temp,
        thermal_voltage_V=vt,
    )
    sc_eq = scaps_thermionic_flux(
        eq_mb_state,
        phi_trace_left_V=phi_l,
        phi_trace_right_V=phi_r,
        conduction_band_step_eV=cb_step,
        hole_transport_step_eV=vb_step,
        N_C_left_m3=nc_l,
        N_C_right_m3=nc_r,
        N_V_left_m3=nv_l,
        N_V_right_m3=nv_r,
        richardson_n_left_A_m2_K2=2.5e5,
        richardson_n_right_A_m2_K2=1.8e5,
        richardson_p_left_A_m2_K2=3.5e5,
        richardson_p_right_A_m2_K2=2.9e5,
        temperature_K=temp,
        thermal_voltage_V=vt,
    )
    for c_idx in range(2):
        assert mb_eq.cross_flux_m2_s[c_idx] == pytest.approx(
            0.0, abs=1.0e-13 * float(mb_eq.one_way_scale_m2_s[c_idx])
        )
        assert sc_eq.cross_flux_m2_s[c_idx] == pytest.approx(
            0.0, abs=1.0e-13 * float(sc_eq.one_way_scale_m2_s[c_idx])
        )

    # Check FD derivatives of all three kernels on a non-equilibrium state.
    trial = np.array([4.2e20, 1.9e21, 8.7e19, 6.4e20], dtype=np.float64)
    for eval_fn in (
        lambda s, pl, pr: fermi_dirac_richardson_flux(
            s,
            phi_trace_left_V=pl,
            phi_trace_right_V=pr,
            conduction_band_step_eV=cb_step,
            hole_transport_step_eV=vb_step,
            N_C_left_m3=nc_l,
            N_C_right_m3=nc_r,
            N_V_left_m3=nv_l,
            N_V_right_m3=nv_r,
            richardson_n_A_m2_K2=2.5e5,
            richardson_p_A_m2_K2=3.5e5,
            temperature_K=temp,
            thermal_voltage_V=vt,
            transmission=0.85,
        ),
        lambda s, pl, pr: fermi_richardson_flux(
            s,
            phi_trace_left_V=pl,
            phi_trace_right_V=pr,
            conduction_band_step_eV=cb_step,
            hole_transport_step_eV=vb_step,
            N_C_left_m3=nc_l,
            N_C_right_m3=nc_r,
            N_V_left_m3=nv_l,
            N_V_right_m3=nv_r,
            richardson_n_A_m2_K2=2.5e5,
            richardson_p_A_m2_K2=3.5e5,
            temperature_K=temp,
            thermal_voltage_V=vt,
            transmission=0.85,
        ),
        lambda s, pl, pr: scaps_thermionic_flux(
            s,
            phi_trace_left_V=pl,
            phi_trace_right_V=pr,
            conduction_band_step_eV=cb_step,
            hole_transport_step_eV=vb_step,
            N_C_left_m3=nc_l,
            N_C_right_m3=nc_r,
            N_V_left_m3=nv_l,
            N_V_right_m3=nv_r,
            richardson_n_left_A_m2_K2=2.5e5,
            richardson_n_right_A_m2_K2=1.8e5,
            richardson_p_left_A_m2_K2=3.5e5,
            richardson_p_right_A_m2_K2=2.9e5,
            temperature_K=temp,
            thermal_voltage_V=vt,
            transmission=0.85,
        ),
    ):
        base = eval_fn(trial, phi_l, phi_r)
        log_trial = np.log(trial)
        fd_log = np.zeros((2, 4))
        for j in range(4):
            h = 1.0e-6
            lp = log_trial.copy()
            lm = log_trial.copy()
            lp[j] += h
            lm[j] -= h
            fd_log[:, j] = (
                eval_fn(np.exp(lp), phi_l, phi_r).cross_flux_m2_s
                - eval_fn(np.exp(lm), phi_l, phi_r).cross_flux_m2_s
            ) / (2.0 * h)
        fd_phi = np.zeros((2, 2))
        h_v = 1.0e-6
        fd_phi[:, 0] = (
            eval_fn(trial, phi_l + h_v, phi_r).cross_flux_m2_s
            - eval_fn(trial, phi_l - h_v, phi_r).cross_flux_m2_s
        ) / (2.0 * h_v)
        fd_phi[:, 1] = (
            eval_fn(trial, phi_l, phi_r + h_v).cross_flux_m2_s
            - eval_fn(trial, phi_l, phi_r - h_v).cross_flux_m2_s
        ) / (2.0 * h_v)

        scale_atol = 1.0e-8 * float(np.max(base.one_way_scale_m2_s))
        np.testing.assert_allclose(base.d_cross_d_log_state_m2_s, fd_log, rtol=1e-8, atol=scale_atol)
        np.testing.assert_allclose(
            base.d_cross_d_trace_potentials_m2_s_V, fd_phi, rtol=1e-8, atol=scale_atol
        )


def test_remove_shared_interface_nodes_and_displacement_jump() -> None:
    grid = np.array([0.0, 20e-9, 50e-9, 90e-9, 450e-9, 500e-9])
    reduced = remove_shared_interface_nodes(grid, [50e-9, 450e-9])
    np.testing.assert_array_equal(reduced, np.array([0.0, 20e-9, 90e-9, 500e-9]))

    with pytest.raises(ValueError, match="outer contact"):
        remove_shared_interface_nodes(grid, [0.0])
    with pytest.raises(ValueError, match="expected exactly one"):
        remove_shared_interface_nodes(grid, [123e-9])

    sigma = -2.4e-4
    geom = TraceElectrostatics(
        left_distance_m=4.0e-9,
        right_distance_m=6.0e-9,
        eps_r_left=3.5,
        eps_r_right=24.0,
        fixed_sheet_charge_C_m2=sigma,
    )
    phi_bulk_l, phi_bulk_r = 0.15, 0.42
    phi_tl, phi_tr = solve_electrostatic_traces(geom, phi_bulk_l, phi_bulk_r)
    c_l = EPS_0 * geom.eps_r_left / geom.left_distance_m
    c_r = EPS_0 * geom.eps_r_right / geom.right_distance_m
    c_series = (c_l * c_r) / (c_l + c_r)
    d_series = -c_series * (phi_bulk_r - phi_bulk_l)
    d_l, d_r = interface_displacement_sides(
        d_series,
        eps_r_left=geom.eps_r_left,
        left_distance_m=geom.left_distance_m,
        eps_r_right=geom.eps_r_right,
        right_distance_m=geom.right_distance_m,
        sheet_charge_C_m2=sigma,
    )
    assert d_l == pytest.approx(-c_l * (phi_tl - phi_bulk_l), rel=1e-14)
    assert d_r == pytest.approx(-c_r * (phi_bulk_r - phi_tr), rel=1e-14)
    assert (d_r - d_l) == pytest.approx(sigma, rel=1e-14)


def _sample_two_sided_setup() -> tuple[TraceElectrostatics, TwoSidedInterfacePhysics, float]:
    temp = 300.0
    vt = thermal_voltage(temp)
    geom = TraceElectrostatics(
        left_distance_m=5.0e-9,
        right_distance_m=7.0e-9,
        eps_r_left=4.0,
        eps_r_right=24.1,
    )
    chi_l, eg_l, nc_l, nv_l = 3.75, 2.1, 4.0e24, 6.0e24
    chi_r, eg_r, nc_r, nv_r = 3.92, 1.6, 2.2e24, 3.1e24
    refs = resolve_interface_defect_references(
        left_role="HTL",
        right_role="absorber",
        left_chi_phys_eV=chi_l,
        left_Eg_phys_eV=eg_l,
        left_Nc_m3=nc_l,
        left_Nv_m3=nv_l,
        right_chi_phys_eV=chi_r,
        right_Eg_phys_eV=eg_r,
        right_Nc_m3=nc_r,
        right_Nv_m3=nv_r,
        thermal_voltage_V=vt,
        trap_depth_eV=0.58,
        energy_reference="below_conduction_band",
    )
    phys = TwoSidedInterfacePhysics(
        thermal_voltage_V=vt,
        temperature_K=temp,
        D_n_left_m2_s=1.0e-6,
        D_n_right_m2_s=5.0e-5,
        D_p_left_m2_s=2.0e-5,
        D_p_right_m2_s=4.0e-5,
        N_C_left_m3=nc_l,
        N_C_right_m3=nc_r,
        N_V_left_m3=nv_l,
        N_V_right_m3=nv_r,
        richardson_n_A_m2_K2=min(richardson_from_dos(nc_l), richardson_from_dos(nc_r)),
        richardson_p_A_m2_K2=min(richardson_from_dos(nv_l), richardson_from_dos(nv_r)),
        conduction_band_step_eV=chi_l - chi_r,
        hole_transport_step_eV=(chi_r + eg_r) - (chi_l + eg_l),
        transmission=1.0,
        surface_recombination_velocity_n_m_s=30.0,
        surface_recombination_velocity_p_m_s=45.0,
        n1_left_m3=refs.n1_left_m3,
        n1_right_m3=refs.n1_right_m3,
        p1_left_m3=refs.p1_left_m3,
        p1_right_m3=refs.p1_right_m3,
    )
    return geom, phys, vt


def test_two_sided_qss_ift_jacobian_equilibrium_and_mirror_symmetry() -> None:
    geom, phys, vt = _sample_two_sided_setup()

    # Non-equilibrium bulk state: check 4x6 bulk_flux_jacobian_bulk and 1x6 occupancy_jacobian_bulk vs FD.
    coords = np.array(
        [0.18, 0.34, math.log(1.8e18), math.log(4.5e20), math.log(6.2e19), math.log(2.3e19)],
        dtype=np.float64,
    )

    def _run_uncharged(c: np.ndarray):
        return solve_two_sided_interface_qss(
            geom,
            phys,
            TwoSidedBulkState(
                phi_left_V=float(c[0]),
                phi_right_V=float(c[1]),
                n_left_m3=math.exp(float(c[2])),
                p_left_m3=math.exp(float(c[3])),
                n_right_m3=math.exp(float(c[4])),
                p_right_m3=math.exp(float(c[5])),
            ),
            residual_tolerance=1.0e-12,
        )

    base = _run_uncharged(coords)
    assert base.converged
    assert base.normalized_residual < 1.0e-12
    assert base.occupancy is not None

    fd_flux = np.zeros((4, 6))
    fd_occ = np.zeros(6)
    for j in range(6):
        h = 1.0e-4 if j >= 2 else 1.0e-5
        cp = coords.copy()
        cm = coords.copy()
        cp[j] += h
        cm[j] -= h
        rp = _run_uncharged(cp)
        rm = _run_uncharged(cm)
        fd_flux[:, j] = (rp.bulk_flux_m2_s - rm.bulk_flux_m2_s) / (2.0 * h)
        assert rp.occupancy is not None and rm.occupancy is not None
        fd_occ[j] = (rp.occupancy - rm.occupancy) / (2.0 * h)

    for row in range(4):
        row_atol = 1.0e-7 * max(float(np.max(np.abs(base.bulk_flux_jacobian_bulk[row]))), 1.0)
        np.testing.assert_allclose(
            base.bulk_flux_jacobian_bulk[row], fd_flux[row], rtol=1e-6, atol=row_atol
        )
    np.testing.assert_allclose(base.occupancy_jacobian_bulk, fd_occ, rtol=1e-6, atol=1e-10)

    # Left/right layer reversal symmetry check!
    geom_rev = TraceElectrostatics(
        left_distance_m=geom.right_distance_m,
        right_distance_m=geom.left_distance_m,
        eps_r_left=geom.eps_r_right,
        eps_r_right=geom.eps_r_left,
    )
    phys_rev = TwoSidedInterfacePhysics(
        thermal_voltage_V=phys.thermal_voltage_V,
        temperature_K=phys.temperature_K,
        D_n_left_m2_s=phys.D_n_right_m2_s,
        D_n_right_m2_s=phys.D_n_left_m2_s,
        D_p_left_m2_s=phys.D_p_right_m2_s,
        D_p_right_m2_s=phys.D_p_left_m2_s,
        N_C_left_m3=phys.N_C_right_m3,
        N_C_right_m3=phys.N_C_left_m3,
        N_V_left_m3=phys.N_V_right_m3,
        N_V_right_m3=phys.N_V_left_m3,
        richardson_n_A_m2_K2=phys.richardson_n_A_m2_K2,
        richardson_p_A_m2_K2=phys.richardson_p_A_m2_K2,
        conduction_band_step_eV=-phys.conduction_band_step_eV,
        hole_transport_step_eV=-phys.hole_transport_step_eV,
        transmission=phys.transmission,
        surface_recombination_velocity_n_m_s=phys.surface_recombination_velocity_n_m_s,
        surface_recombination_velocity_p_m_s=phys.surface_recombination_velocity_p_m_s,
        n1_left_m3=phys.n1_right_m3,
        n1_right_m3=phys.n1_left_m3,
        p1_left_m3=phys.p1_right_m3,
        p1_right_m3=phys.p1_left_m3,
    )
    rev = solve_two_sided_interface_qss(
        geom_rev,
        phys_rev,
        TwoSidedBulkState(
            phi_left_V=float(coords[1]),
            phi_right_V=float(coords[0]),
            n_left_m3=math.exp(float(coords[4])),
            p_left_m3=math.exp(float(coords[5])),
            n_right_m3=math.exp(float(coords[2])),
            p_right_m3=math.exp(float(coords[3])),
        ),
        residual_tolerance=1.0e-12,
    )
    assert rev.occupancy == pytest.approx(base.occupancy, rel=1e-12)
    np.testing.assert_allclose(rev.cross_flux_m2_s, -base.cross_flux_m2_s, rtol=1e-11)
    # Bulk flux into trace has ordering (n_L, p_L, n_R, p_R): swapping sides swaps (0,1) <-> (2,3)!
    np.testing.assert_allclose(
        rev.bulk_flux_m2_s,
        base.bulk_flux_m2_s[[2, 3, 0, 1]],
        rtol=1e-11,
    )


def test_charged_two_sided_qss_gauss_jump_equilibrium_and_fd_jacobians() -> None:
    geom, phys, _ = _sample_two_sided_setup()

    # First compute an uncharged equilibrium-like reference occupancy, then verify charged QSS
    # at that same state gives zero incremental sheet charge!
    eq_bulk = TwoSidedBulkState(
        phi_left_V=0.10,
        phi_right_V=0.25,
        n_left_m3=2.0e17,
        p_left_m3=3.0e20,
        n_right_m3=8.0e18,
        p_right_m3=4.0e18,
    )
    uncharged = solve_two_sided_interface_qss(geom, phys, eq_bulk, residual_tolerance=1.0e-12)
    assert uncharged.occupancy is not None
    charge_spec = EquilibriumReferencedSheetCharge(
        trap_density_m2=4.0e15,
        equilibrium_occupancy=uncharged.occupancy,
    )
    at_ref = solve_charged_two_sided_interface_qss(
        geom, phys, eq_bulk, charge_spec, residual_tolerance=1.0e-12
    )
    assert at_ref.converged
    assert at_ref.incremental_sheet_charge_C_m2 == pytest.approx(0.0, abs=1e-16)
    assert at_ref.phi_trace_left_V == pytest.approx(at_ref.phi_trace_right_V, abs=1e-14)

    # Non-equilibrium biased state: check Gauss jump D_R - D_L == sigma and all 3 IFT Jacobians vs FD!
    coords = np.array(
        [0.21, 0.29, math.log(5.5e18), math.log(2.1e20), math.log(1.9e20), math.log(7.5e18)],
        dtype=np.float64,
    )

    def _run_charged(c: np.ndarray) -> ChargedTwoSidedInterfaceQSSResult:
        return solve_charged_two_sided_interface_qss(
            geom,
            phys,
            TwoSidedBulkState(
                phi_left_V=float(c[0]),
                phi_right_V=float(c[1]),
                n_left_m3=math.exp(float(c[2])),
                p_left_m3=math.exp(float(c[3])),
                n_right_m3=math.exp(float(c[4])),
                p_right_m3=math.exp(float(c[5])),
            ),
            charge_spec,
            residual_tolerance=1.0e-12,
        )

    charged = _run_charged(coords)
    assert charged.converged
    assert charged.normalized_electrostatic_residual < 1.0e-12
    assert charged.normalized_carrier_residual < 1.0e-12
    assert abs(charged.incremental_sheet_charge_C_m2) > 1.0e-6

    c_l = EPS_0 * geom.eps_r_left / geom.left_distance_m
    c_r = EPS_0 * geom.eps_r_right / geom.right_distance_m
    c_series = (c_l * c_r) / (c_l + c_r)
    d_series = -c_series * (float(coords[1]) - float(coords[0]))
    d_l, d_r = interface_displacement_sides(
        d_series,
        eps_r_left=geom.eps_r_left,
        left_distance_m=geom.left_distance_m,
        eps_r_right=geom.eps_r_right,
        right_distance_m=geom.right_distance_m,
        sheet_charge_C_m2=charged.incremental_sheet_charge_C_m2,
    )
    assert d_l == pytest.approx(-c_l * (charged.phi_trace_left_V - float(coords[0])), rel=1e-12)
    assert d_r == pytest.approx(-c_r * (float(coords[1]) - charged.phi_trace_right_V), rel=1e-12)
    assert (d_r - d_l) == pytest.approx(charged.incremental_sheet_charge_C_m2, rel=1e-12)

    fd_flux = np.zeros((4, 6))
    fd_sigma = np.zeros(6)
    fd_occ = np.zeros(6)
    for j in range(6):
        h = 1.0e-4 if j >= 2 else 1.0e-5
        cp = coords.copy()
        cm = coords.copy()
        cp[j] += h
        cm[j] -= h
        rp = _run_charged(cp)
        rm = _run_charged(cm)
        fd_flux[:, j] = (rp.bulk_flux_m2_s - rm.bulk_flux_m2_s) / (2.0 * h)
        fd_sigma[j] = (
            rp.incremental_sheet_charge_C_m2 - rm.incremental_sheet_charge_C_m2
        ) / (2.0 * h)
        fd_occ[j] = (rp.occupancy - rm.occupancy) / (2.0 * h)

    for row in range(4):
        row_atol = 1.0e-7 * max(float(np.max(np.abs(charged.bulk_flux_jacobian_bulk[row]))), 1.0)
        np.testing.assert_allclose(
            charged.bulk_flux_jacobian_bulk[row], fd_flux[row], rtol=1e-6, atol=row_atol
        )
    np.testing.assert_allclose(charged.sheet_charge_jacobian_bulk, fd_sigma, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(charged.occupancy_jacobian_bulk, fd_occ, rtol=1e-6, atol=1e-10)
