"""Unit tests for heterointerface SRH, plane closure, and sheet-charge laws (P04-09)."""

from __future__ import annotations

from collections.abc import Iterator
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.physics.recombination.interface import (
    InterfaceDefectSpec,
    S_SUPPLY_M_S,
    TwoDimensionalInterfaceSRHMode,
    build_plane_closure_params,
    equilibrium_referenced_sheet_charge,
    evaluate_cross_carrier_cell_pair_srh,
    evaluate_shared_interface_trap,
    interface_srh_rate,
    resolve_interface_defect_references,
    solve_plane_closure,
)
from solarlab.physics.statistics.thermal import thermal_voltage


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def test_interface_srh_rate_equilibrium_zero_and_disabled_velocities() -> None:
    ni = 2.5e16
    ni_sq = ni * ni
    n_eq = 5.0e18
    p_eq = ni_sq / n_eq
    res = interface_srh_rate(
        n_eq,
        p_eq,
        v_n_m_s=15.0,
        v_p_m_s=40.0,
        ni_sq_m6=ni_sq,
        n1_m3=1.2e16,
        p1_m3=5.2e16,
    )
    assert float(res.rate_m2_s) == 0.0
    assert float(res.d_rate_dn_m_s) > 0.0
    assert float(res.d_rate_dp_m_s) > 0.0

    for vn, vp in ((0.0, 25.0), (25.0, 0.0), (0.0, 0.0)):
        off = interface_srh_rate(
            1.0e20,
            2.0e20,
            v_n_m_s=vn,
            v_p_m_s=vp,
            ni_sq_m6=ni_sq,
            n1_m3=ni,
            p1_m3=ni,
        )
        assert float(off.rate_m2_s) == 0.0
        assert float(off.d_rate_dn_m_s) == 0.0
        assert float(off.d_rate_dp_m_s) == 0.0


def test_interface_srh_areal_scaling_and_nonnegative_clamp() -> None:
    ni_sq = 1.0e32
    r1 = interface_srh_rate(
        1.0e19, 4.0e18, v_n_m_s=3.0, v_p_m_s=7.0, ni_sq_m6=ni_sq, n1_m3=1.0e16, p1_m3=1.0e16
    )
    r2 = interface_srh_rate(
        1.0e19, 4.0e18, v_n_m_s=6.0, v_p_m_s=14.0, ni_sq_m6=ni_sq, n1_m3=1.0e16, p1_m3=1.0e16
    )
    np.testing.assert_allclose(r2.rate_m2_s, 2.0 * r1.rate_m2_s, rtol=1e-15, atol=0.0)

    # Depletion n * p < ni^2: signed vs clamped.
    signed = interface_srh_rate(
        1.0e13,
        1.0e14,
        v_n_m_s=10.0,
        v_p_m_s=20.0,
        ni_sq_m6=ni_sq,
        n1_m3=1.0e16,
        p1_m3=1.0e16,
        clamp_nonnegative=False,
    )
    clamped = interface_srh_rate(
        1.0e13,
        1.0e14,
        v_n_m_s=10.0,
        v_p_m_s=20.0,
        ni_sq_m6=ni_sq,
        n1_m3=1.0e16,
        p1_m3=1.0e16,
        clamp_nonnegative=True,
    )
    assert float(signed.rate_m2_s) < 0.0
    assert float(clamped.rate_m2_s) == 0.0
    assert float(clamped.d_rate_dn_m_s) == 0.0
    assert float(clamped.d_rate_dp_m_s) == 0.0


@pytest.mark.parametrize(
    ("n_val", "p_val"),
    [
        (1.0e14, 1.0e15),  # deep depletion (generation)
        (4.0e17, 6.25e14),  # exact thermal equilibrium n*p == 2.5e32
        (4.0e17 * (1.0 + 1e-11), 6.25e14),  # near-equilibrium cancellation test
        (3.5e20, 8.2e19),  # strong forward injection
    ],
)
def test_interface_srh_derivatives_match_complex_step_and_fd(n_val: float, p_val: float) -> None:
    ni_sq = 2.5e32
    vn, vp = 18.5, 42.0
    n1, p1 = 5.0e16, 5.0e15
    ev = interface_srh_rate(
        n_val, p_val, v_n_m_s=vn, v_p_m_s=vp, ni_sq_m6=ni_sq, n1_m3=n1, p1_m3=p1
    )

    eps_cs = 1.0e-30
    cs_n = interface_srh_rate(
        n_val + 1j * eps_cs * n_val,
        p_val,
        v_n_m_s=vn,
        v_p_m_s=vp,
        ni_sq_m6=ni_sq,
        n1_m3=n1,
        p1_m3=p1,
    )
    cs_p = interface_srh_rate(
        n_val,
        p_val + 1j * eps_cs * p_val,
        v_n_m_s=vn,
        v_p_m_s=vp,
        ni_sq_m6=ni_sq,
        n1_m3=n1,
        p1_m3=p1,
    )
    dn_cs = float(np.imag(cs_n.rate_m2_s)) / (eps_cs * n_val)
    dp_cs = float(np.imag(cs_p.rate_m2_s)) / (eps_cs * p_val)
    np.testing.assert_allclose(float(ev.d_rate_dn_m_s), dn_cs, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(float(ev.d_rate_dp_m_s), dp_cs, rtol=1e-14, atol=0.0)


def test_resolve_interface_defect_references_rules_and_detailed_balance() -> None:
    vt = thermal_voltage(300.0)
    # Unique absorber wins even if transport layer had smaller gap; test standard HTL/absorber.
    refs_cb = resolve_interface_defect_references(
        left_role="HTL",
        right_role="absorber",
        left_chi_phys_eV=3.6,
        left_Eg_phys_eV=2.2,
        left_Nc_m3=1.0e25,
        left_Nv_m3=2.0e25,
        right_chi_phys_eV=3.9,
        right_Eg_phys_eV=1.6,
        right_Nc_m3=3.0e24,
        right_Nv_m3=5.0e24,
        thermal_voltage_V=vt,
        trap_depth_eV=0.55,
        energy_reference="below_conduction_band",
        require_in_gap=True,
    )
    refs_vb = resolve_interface_defect_references(
        left_role="HTL",
        right_role="absorber",
        left_chi_phys_eV=3.6,
        left_Eg_phys_eV=2.2,
        left_Nc_m3=1.0e25,
        left_Nv_m3=2.0e25,
        right_chi_phys_eV=3.9,
        right_Eg_phys_eV=1.6,
        right_Nc_m3=3.0e24,
        right_Nv_m3=5.0e24,
        thermal_voltage_V=vt,
        trap_depth_eV=1.6 - 0.55,
        energy_reference="above_valence_band",
        require_in_gap=True,
    )
    assert refs_cb.reference_side == "right"
    np.testing.assert_allclose(refs_cb.n1_reference_m3, refs_vb.n1_reference_m3, rtol=1e-14)
    np.testing.assert_allclose(refs_cb.p1_reference_m3, refs_vb.p1_reference_m3, rtol=1e-14)
    np.testing.assert_allclose(refs_cb.n1_left_m3, refs_vb.n1_left_m3, rtol=1e-14)
    np.testing.assert_allclose(refs_cb.p1_left_m3, refs_vb.p1_left_m3, rtol=1e-14)

    # Detailed balance on each physical band structure: n1_s * p1_s == Nc_s * Nv_s * exp(-Eg_s/vt)
    ni_sq_left = 1.0e25 * 2.0e25 * math.exp(-2.2 / vt)
    ni_sq_right = 3.0e24 * 5.0e24 * math.exp(-1.6 / vt)
    np.testing.assert_allclose(
        refs_cb.n1_left_m3 * refs_cb.p1_left_m3, ni_sq_left, rtol=1e-13
    )
    np.testing.assert_allclose(
        refs_cb.n1_right_m3 * refs_cb.p1_right_m3, ni_sq_right, rtol=1e-13
    )

    # Smaller Eg wins when neither (or both) is absorber; left wins exact ties.
    tie = resolve_interface_defect_references(
        left_role="absorber",
        right_role="absorber",
        left_chi_phys_eV=3.9,
        left_Eg_phys_eV=1.55,
        left_Nc_m3=2.0e24,
        left_Nv_m3=2.0e24,
        right_chi_phys_eV=4.0,
        right_Eg_phys_eV=1.55,
        right_Nc_m3=2.0e24,
        right_Nv_m3=2.0e24,
        thermal_voltage_V=vt,
        trap_depth_eV=0.4,
        energy_reference="below_conduction_band",
    )
    assert tie.reference_side == "left"

    with pytest.raises(ValueError, match="outside reference physical band gap"):
        resolve_interface_defect_references(
            left_role="HTL",
            right_role="absorber",
            left_chi_phys_eV=3.6,
            left_Eg_phys_eV=2.2,
            left_Nc_m3=1.0e25,
            left_Nv_m3=2.0e25,
            right_chi_phys_eV=3.9,
            right_Eg_phys_eV=1.6,
            right_Nc_m3=3.0e24,
            right_Nv_m3=5.0e24,
            thermal_voltage_V=vt,
            trap_depth_eV=1.7,
            energy_reference="below_conduction_band",
            require_in_gap=True,
        )

    spec = InterfaceDefectSpec(
        areal_density_m2=2.0e15,
        trap_depth_eV=0.5,
        energy_reference="below_conduction_band",
        sigma_n_m2=1.0e-19,
        sigma_p_m2=3.0e-19,
        v_th_n_m_s=1.0e5,
        v_th_p_m_s=1.0e5,
        calibration_factor=0.5,
    )
    assert spec.microscopic_capture_velocities_m_s == pytest.approx((20.0, 60.0))
    assert spec.effective_capture_velocities_m_s() == pytest.approx((10.0, 30.0))


def test_shared_interface_trap_detailed_balance_sheet_charge_and_fd_jacobians() -> None:
    vt = thermal_voltage(300.0)
    refs = resolve_interface_defect_references(
        left_role="HTL",
        right_role="absorber",
        left_chi_phys_eV=3.7,
        left_Eg_phys_eV=2.0,
        left_Nc_m3=5.0e24,
        left_Nv_m3=8.0e24,
        right_chi_phys_eV=3.9,
        right_Eg_phys_eV=1.6,
        right_Nc_m3=2.5e24,
        right_Nv_m3=3.5e24,
        thermal_voltage_V=vt,
        trap_depth_eV=0.65,
        energy_reference="below_conduction_band",
    )
    # Thermal equilibrium across potential jump delta_phi = phi_R - phi_L:
    # wait: trace emission densities n1_L, n1_R are referenced to zero local electrostatic shift;
    # at zero potential jump across the zero-thickness plane (phi_tL == phi_tR),
    # equilibrium Fermi level E_F gives n_s / n1_s = exp((E_F - E_t)/vt) = p1_s / p_s = lambda_eq!
    lam = 12.5
    n_eq_l = lam * refs.n1_left_m3
    p_eq_l = refs.p1_left_m3 / lam
    n_eq_r = lam * refs.n1_right_m3
    p_eq_r = refs.p1_right_m3 / lam
    f_expected = lam / (1.0 + lam)

    eq_eval = evaluate_shared_interface_trap(
        n_eq_l,
        p_eq_l,
        n_eq_r,
        p_eq_r,
        v_n_left_m_s=15.0,
        v_p_left_m_s=35.0,
        v_n_right_m_s=25.0,
        v_p_right_m_s=45.0,
        n1_left_m3=refs.n1_left_m3,
        p1_left_m3=refs.p1_left_m3,
        n1_right_m3=refs.n1_right_m3,
        p1_right_m3=refs.p1_right_m3,
        areal_density_m2=4.0e15,
        equilibrium_occupancy=f_expected,
    )
    assert eq_eval.occupancy == pytest.approx(f_expected, rel=1e-14)
    one_way_scale = max(25.0 * n_eq_r, 45.0 * p_eq_r, 15.0 * n_eq_l, 35.0 * p_eq_l)
    np.testing.assert_allclose(eq_eval.capture_rates_m2_s, 0.0, atol=1e-14 * one_way_scale)
    assert eq_eval.incremental_sheet_charge_C_m2 == pytest.approx(0.0, abs=1e-22)

    # Non-equilibrium trial state: check C_nL + C_nR == C_pL + C_pR, sheet charge formula, and FD Jacobians.
    dens = np.array([2.4e18, 9.1e19, 7.3e19, 1.8e18], dtype=np.float64)
    nt = 3.5e15
    f_ref = 0.28
    base = evaluate_shared_interface_trap(
        float(dens[0]),
        float(dens[1]),
        float(dens[2]),
        float(dens[3]),
        v_n_left_m_s=12.0,
        v_p_left_m_s=28.0,
        v_n_right_m_s=19.0,
        v_p_right_m_s=33.0,
        n1_left_m3=refs.n1_left_m3,
        p1_left_m3=refs.p1_left_m3,
        n1_right_m3=refs.n1_right_m3,
        p1_right_m3=refs.p1_right_m3,
        areal_density_m2=nt,
        equilibrium_occupancy=f_ref,
    )
    assert 0.0 < base.occupancy < 1.0
    np.testing.assert_allclose(
        base.capture_rates_m2_s[0] + base.capture_rates_m2_s[2],
        base.capture_rates_m2_s[1] + base.capture_rates_m2_s[3],
        rtol=1e-14,
    )
    expected_sigma = equilibrium_referenced_sheet_charge(
        base.occupancy, areal_density_m2=nt, equilibrium_occupancy=f_ref
    )
    assert base.incremental_sheet_charge_C_m2 == pytest.approx(expected_sigma, rel=1e-15)
    assert expected_sigma == pytest.approx(-Q * nt * (base.occupancy - f_ref), rel=1e-15)

    fd_df = np.zeros(4)
    fd_dcap = np.zeros((4, 4))
    fd_dsigma = np.zeros(4)
    for j in range(4):
        step = 1.0e-6 * dens[j]
        dp = dens.copy()
        dm = dens.copy()
        dp[j] += step
        dm[j] -= step
        ep = evaluate_shared_interface_trap(
            float(dp[0]),
            float(dp[1]),
            float(dp[2]),
            float(dp[3]),
            v_n_left_m_s=12.0,
            v_p_left_m_s=28.0,
            v_n_right_m_s=19.0,
            v_p_right_m_s=33.0,
            n1_left_m3=refs.n1_left_m3,
            p1_left_m3=refs.p1_left_m3,
            n1_right_m3=refs.n1_right_m3,
            p1_right_m3=refs.p1_right_m3,
            areal_density_m2=nt,
            equilibrium_occupancy=f_ref,
        )
        em = evaluate_shared_interface_trap(
            float(dm[0]),
            float(dm[1]),
            float(dm[2]),
            float(dm[3]),
            v_n_left_m_s=12.0,
            v_p_left_m_s=28.0,
            v_n_right_m_s=19.0,
            v_p_right_m_s=33.0,
            n1_left_m3=refs.n1_left_m3,
            p1_left_m3=refs.p1_left_m3,
            n1_right_m3=refs.n1_right_m3,
            p1_right_m3=refs.p1_right_m3,
            areal_density_m2=nt,
            equilibrium_occupancy=f_ref,
        )
        fd_df[j] = (ep.occupancy - em.occupancy) / (2.0 * step)
        fd_dcap[:, j] = (ep.capture_rates_m2_s - em.capture_rates_m2_s) / (2.0 * step)
        fd_dsigma[j] = (
            ep.incremental_sheet_charge_C_m2 - em.incremental_sheet_charge_C_m2
        ) / (2.0 * step)

    np.testing.assert_allclose(base.d_occupancy_d_densities_m3, fd_df, rtol=5e-8, atol=0.0)
    np.testing.assert_allclose(base.d_capture_d_densities_m_s, fd_dcap, rtol=5e-8, atol=1e-18)
    np.testing.assert_allclose(base.d_sheet_charge_d_densities_C_m, fd_dsigma, rtol=5e-8, atol=0.0)


@pytest.mark.parametrize("allow_generation", [False, True])
def test_plane_closure_qss_balance_equilibrium_and_ift_derivatives(
    allow_generation: bool,
) -> None:
    vt = thermal_voltage(300.0)
    chi_l, chi_r = 3.72, 3.91
    ceg_l, ceg_r = 5.85, 5.51
    nc_ref, nv_ref = 2.8e24, 3.6e24
    chi_ref_phys = 3.90
    et_depth = 0.62

    params = build_plane_closure_params(
        chi_left_folded_eV=chi_l,
        chi_right_folded_eV=chi_r,
        valence_left_folded_eV=ceg_l,
        valence_right_folded_eV=ceg_r,
        Nc_ref_m3=nc_ref,
        Nv_ref_m3=nv_ref,
        chi_ref_phys_eV=chi_ref_phys,
        trap_depth_below_ref_cb_eV=et_depth,
        thermal_voltage_V=vt,
    )
    assert params.s_supply_m_s == S_SUPPLY_M_S
    assert params.n1_s_m3 * params.p1_s_m3 == pytest.approx(params.ni_s_sq_m6, rel=1e-13)

    # Supply-balanced equilibrium point (nb = 2 * n_s_eq, pb = 2 * p_s_eq, n_s_eq * p_s_eq == ni_s^2)
    # gives R = 0 under both clamped and signed-generation branches.
    n_s_eq = math.sqrt(params.ni_s_sq_m6) * 2.0
    p_s_eq = params.ni_s_sq_m6 / n_s_eq
    nl_bal = n_s_eq / params.bn_L
    nr_bal = n_s_eq / params.bn_R
    pl_bal = p_s_eq / params.bp_L
    pr_bal = p_s_eq / params.bp_R
    bal_res = solve_plane_closure(
        nl_bal, nr_bal, pl_bal, pr_bal, params, 25.0, 55.0, allow_generation=allow_generation
    )
    assert bal_res.converged
    eq_supply_scale = S_SUPPLY_M_S * (
        params.bn_L * nl_bal + params.bn_R * nr_bal + params.bp_L * pl_bal + params.bp_R * pr_bal
    )
    assert bal_res.rate_m2_s == pytest.approx(0.0, abs=1e-13 * eq_supply_scale)

    # Under default allow_generation=False (NOGEN clamp), Boltzmann equilibrium across the
    # heterojunction step also gives exact zero recombination.
    if not allow_generation:
        chi_s = max(chi_l, chi_r)
        ceg_s = min(ceg_l, ceg_r)
        nl_eq = n_s_eq * math.exp((chi_l - chi_s) / vt)
        nr_eq = n_s_eq * math.exp((chi_r - chi_s) / vt)
        pl_eq = p_s_eq * math.exp((ceg_s - ceg_l) / vt)
        pr_eq = p_s_eq * math.exp((ceg_s - ceg_r) / vt)
        eq_res = solve_plane_closure(
            nl_eq, nr_eq, pl_eq, pr_eq, params, 25.0, 55.0, allow_generation=False
        )
        assert eq_res.converged
        assert eq_res.rate_m2_s == 0.0

    # Test both injection (n*p > ni_s^2) and depletion (n*p < ni_s^2)
    for proj in (
        np.array([1.2e19, 4.5e19, 3.1e19, 8.0e18], dtype=np.float64),
        np.array([1.0e12, 2.0e12, 3.0e12, 1.5e12], dtype=np.float64),
    ):
        res = solve_plane_closure(
            float(proj[0]),
            float(proj[1]),
            float(proj[2]),
            float(proj[3]),
            params,
            25.0,
            55.0,
            allow_generation=allow_generation,
        )
        assert res.converged
        nb = params.bn_L * proj[0] + params.bn_R * proj[1]
        pb = params.bp_L * proj[2] + params.bp_R * proj[3]
        qss_scale = S_SUPPLY_M_S * (nb + pb)
        assert S_SUPPLY_M_S * (nb - 2.0 * res.n_plane_m3) == pytest.approx(
            res.rate_m2_s, rel=1e-10, abs=1e-11 * qss_scale
        )
        assert S_SUPPLY_M_S * (pb - 2.0 * res.p_plane_m3) == pytest.approx(
            res.rate_m2_s, rel=1e-10, abs=1e-11 * qss_scale
        )

        analytic = np.array(
            [
                res.d_rate_d_n_left_proj_m_s,
                res.d_rate_d_n_right_proj_m_s,
                res.d_rate_d_p_left_proj_m_s,
                res.d_rate_d_p_right_proj_m_s,
            ]
        )
        fd = np.zeros(4)
        for j in range(4):
            h = 1.0e-4 * proj[j]
            pp = proj.copy()
            pm = proj.copy()
            pp[j] += h
            pm[j] -= h
            rp = solve_plane_closure(
                float(pp[0]),
                float(pp[1]),
                float(pp[2]),
                float(pp[3]),
                params,
                25.0,
                55.0,
                allow_generation=allow_generation,
            ).rate_m2_s
            rm = solve_plane_closure(
                float(pm[0]),
                float(pm[1]),
                float(pm[2]),
                float(pm[3]),
                params,
                25.0,
                55.0,
                allow_generation=allow_generation,
            ).rate_m2_s
            fd[j] = (rp - rm) / (2.0 * h)
        row_atol = 1.0e-8 * max(float(np.max(np.abs(analytic))), 1.0e-18)
        np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=row_atol)


def test_two_dimensional_cell_pair_srh_mode() -> None:
    n_other = np.array([1.0e18, 5.0e19])
    p_self = np.array([2.5e14, 4.0e18])
    ni_sq = 2.5e32
    off = evaluate_cross_carrier_cell_pair_srh(
        TwoDimensionalInterfaceSRHMode.OFF,
        n_other_m3=n_other,
        p_self_m3=p_self,
        v_n_m_s=10.0,
        v_p_m_s=20.0,
        ni_sq_m6=ni_sq,
        n1_m3=1.0e16,
        p1_m3=1.0e16,
    )
    np.testing.assert_array_equal(off.areal_rate_m2_s, np.zeros(2))
    np.testing.assert_array_equal(off.half_rate_m2_s, np.zeros(2))

    active = evaluate_cross_carrier_cell_pair_srh(
        TwoDimensionalInterfaceSRHMode.CROSS_CARRIER_CELL_PAIR_V1,
        n_other_m3=n_other,
        p_self_m3=p_self,
        v_n_m_s=10.0,
        v_p_m_s=20.0,
        ni_sq_m6=ni_sq,
        n1_m3=1.0e16,
        p1_m3=1.0e16,
    )
    assert active.areal_rate_m2_s[0] == 0.0  # exact equilibrium on cell 0
    assert active.areal_rate_m2_s[1] > 0.0
    np.testing.assert_allclose(active.half_rate_m2_s, 0.5 * active.areal_rate_m2_s, rtol=1e-15)


def test_scaps_interface_extrema_band_reference_mode() -> None:
    from solarlab.physics.recombination import (
        REFERENCE_LAYER_BAND_ANCHOR,
        SCAPS_INTERFACE_EXTREMA_ANCHOR,
    )

    vt = thermal_voltage(300.0)
    # Deep electron cliff at absorber/ETL: chi_ETL (4.69 eV) > chi_PVK (3.94 eV)
    kw_cliff = dict(
        left_role="absorber",
        right_role="etl",
        left_chi_phys_eV=3.94,
        left_Eg_phys_eV=1.53,
        left_Nc_m3=1.0e25,
        left_Nv_m3=1.0e25,
        right_chi_phys_eV=4.69,
        right_Eg_phys_eV=1.90,
        right_Nc_m3=8.0e25,
        right_Nv_m3=8.0e25,
        thermal_voltage_V=vt,
        trap_depth_eV=0.60,
        energy_reference="below_conduction_band",
    )
    ref_default = resolve_interface_defect_references(**kw_cliff)  # type: ignore[arg-type]
    ref_explicit = resolve_interface_defect_references(
        **kw_cliff, band_reference_mode=REFERENCE_LAYER_BAND_ANCHOR  # type: ignore[arg-type]
    )
    assert ref_default == ref_explicit

    ref_scaps = resolve_interface_defect_references(
        **kw_cliff, band_reference_mode=SCAPS_INTERFACE_EXTREMA_ANCHOR  # type: ignore[arg-type]
    )
    # Under SCAPS lowest-Ec anchor, E_t = -max(3.94, 4.69) - 0.60 = -5.29 eV,
    # so E_{c,ETL} - E_t == +0.60 eV regardless of cliff depth.
    assert ref_scaps.n1_right_m3 == pytest.approx(
        8.0e25 * np.exp(-0.60 / vt), rel=1e-13
    )
    assert ref_scaps.n1_right_m3 < ref_default.n1_right_m3

    # Under an electron spike (chi_ETL < chi_PVK), lowest Ec is PVK so both modes agree identically!
    kw_spike = {**kw_cliff, "right_chi_phys_eV": 3.74}
    assert resolve_interface_defect_references(
        **kw_spike, band_reference_mode=SCAPS_INTERFACE_EXTREMA_ANCHOR  # type: ignore[arg-type]
    ) == resolve_interface_defect_references(**kw_spike)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="band_reference_mode"):
        resolve_interface_defect_references(
            **kw_cliff, band_reference_mode="invalid"  # type: ignore[arg-type]
        )
