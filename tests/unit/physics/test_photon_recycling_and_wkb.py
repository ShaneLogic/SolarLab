"""T1 unit tests for photon recycling, radiative reabsorption, and WKB tunnelling."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from solarlab.physics.generation.optical_binding import (
    OpticalLayerSpec,
    OpticalStackBinding,
    constant_index_optical_constants,
)
from solarlab.physics.generation.photon_recycling import (
    TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
    TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
    YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    AbsorberEscapeInput,
    PhotonRecyclingCapabilityError,
    bind_absorber_escape_from_optical_stack,
    compute_p_esc,
    compute_p_esc_for_absorber,
    effective_radiative_coefficient,
    wavelength_at_gap,
)
from solarlab.physics.generation.radiative_reabsorption import (
    AbsorberReabsorptionSpec,
    assess_switching_surfaces,
    dense_reabsorption_jacobian,
    evaluate_radiative_reabsorption,
    radiative_reabsorption_directional_derivative,
    require_smooth_ac_surfaces,
    thermionic_cap_directional_derivative,
    thermionic_cap_value,
)
from solarlab.physics.generation.spectrum import SpectralPhotonFlux
from solarlab.physics.transport.field_mobility_tangents import (
    field_mobility_directional_derivative,
)
from solarlab.physics.transport.tunneling_channels import (
    BandToBandTunnellingChannel,
    ContactTunnellingChannel,
    InterfaceDefectAssistedTunnellingChannel,
    IntrabandBarrierRegion,
    IntrabandTunnellingChannel,
    TunnellingChannelCapabilityError,
    TunnellingChannelDocument,
    TunnellingChannelSchemaError,
    band_to_band_flux,
    compile_tunnelling_channels,
    conduction_band_eV,
    contact_tunnelling_flux,
    interface_defect_assisted_rate,
    intraband_flux,
    intraband_path_flux,
    intraband_path_flux_and_jacobian,
    valence_band_eV,
)
from solarlab.physics.transport.wkb_tunneling import (
    kane_uniform_field_action,
    piecewise_linear_wkb_segment_action_derivatives,
    tfe_gamma,
    triangular_barrier_action,
    two_band_action,
    wkb_action,
)


@pytest.fixture(scope="module", autouse=True)
def blas1() -> object:
    with threadpool_limits(limits=1):
        yield


# ---------------------------------------------------------------------------
# 1. Escape probability limits & explicit TMM absorbance modes
# ---------------------------------------------------------------------------


def test_escape_probability_asymptotic_limits_and_guardrails() -> None:
    assert wavelength_at_gap(1.55) == pytest.approx(7.99898014e-7, rel=1e-6)
    with pytest.raises(ValueError, match="Eg_eV"):
        wavelength_at_gap(0.0)

    # Non-positive inputs clamp cleanly to 1.0 (no trapping).
    assert compute_p_esc(0.0, 500e-9, 2.5) == 1.0
    assert compute_p_esc(1e5, 0.0, 2.5) == 1.0
    assert compute_p_esc(1e5, 500e-9, 0.0) == 1.0

    # Optically thin limit (OD -> 0) clamps to 1.0 rather than > 1.0.
    assert compute_p_esc(1.0, 1.0e-9, 2.5) == 1.0

    # Optically thick limit (OD >> 1) follows 1 / (4 n_r^2 alpha d).
    alpha = 1.0e7
    d = 1.0e-5
    n_r = 2.6
    expected = 1.0 / (4.0 * n_r * n_r * alpha * d)
    assert compute_p_esc(alpha, d, n_r) == pytest.approx(expected, rel=1e-15)


def test_absorber_escape_modes_separate_intrinsic_trapezoid_and_control_volume() -> None:
    wl_nm = np.linspace(400.0, 950.0, 56)
    wl_m = wl_nm * 1e-9
    n_arr, k_arr = constant_index_optical_constants(
        wl_nm, absorption_coefficient_m1=2.0e6, refractive_index=2.5
    )
    spectrum = SpectralPhotonFlux(wl_nm, np.full_like(wl_nm, 1.0e27), "synthetic_flat")
    stack = OpticalStackBinding(
        (OpticalLayerSpec("abs", 400.0e-9, n_arr, k_arr, incoherent=False, electrical=True),),
        spectrum,
    )
    inp = (AbsorberEscapeInput("abs", 400.0e-9, band_gap_front_eV=1.6, band_gap_back_eV=1.55),)

    b_intr = bind_absorber_escape_from_optical_stack(
        stack, inp, mode=YABLONOVITCH_INTRINSIC_ALPHA_D_V1
    )[0]
    assert b_intr.band_gap_eV == 1.55
    assert 0.0 < b_intr.escape_probability < 1.0

    # Non-uniform grid distinguishes trapezoid vs control-volume weights explicitly.
    x_abs = np.array([0.0, 20e-9, 90e-9, 240e-9, 400e-9])
    mask = np.ones(x_abs.size, dtype=bool)
    # Deliberately distinct positive control-volume weights summing to 400 nm.
    cv_weights = np.array([15e-9, 60e-9, 125e-9, 130e-9, 70e-9])
    assert math.isclose(float(np.sum(cv_weights)), 400.0e-9, rel_tol=1e-15)

    b_trap = bind_absorber_escape_from_optical_stack(
        stack,
        inp,
        mode=TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
        electrical_nodes_m=x_abs,
        absorber_node_masks=(mask,),
    )[0]
    b_cv = bind_absorber_escape_from_optical_stack(
        stack,
        inp,
        mode=TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
        electrical_nodes_m=x_abs,
        absorber_node_masks=(mask,),
        absorber_weights_m=(cv_weights,),
    )[0]
    assert b_trap.sha256 != b_cv.sha256
    assert b_trap.escape_probability != b_cv.escape_probability

    # Direct compute_p_esc_for_absorber refuses CV mode without weights.
    absorbance_dummy = np.full((x_abs.size, wl_m.size), 1.0e5)
    with pytest.raises(ValueError, match="weights_m"):
        compute_p_esc_for_absorber(
            absorbance_dummy,
            wl_m,
            x_abs,
            1.55,
            2.5,
            mode=TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
        )


# ---------------------------------------------------------------------------
# 2. Radiative reabsorption conservation, uniform equivalence & rank-1 Jacobian
# ---------------------------------------------------------------------------


def test_reabsorption_photon_conservation_and_uniform_limit_equivalence() -> None:
    rng = np.random.default_rng(20261113)
    node_count = 18
    # Non-uniform control volumes across 3 regions: [0..4], [5..12], [13..17].
    dx = rng.uniform(5e-9, 45e-9, size=node_count)
    emit_nodes: NDArray[np.intp] = np.arange(5, 13, dtype=np.intp)
    # Cross-region receiver support (partially overlapping + downstream region).
    recv_nodes: NDArray[np.intp] = np.arange(8, 17, dtype=np.intp)
    recv_shape = np.exp(-0.25 * np.arange(recv_nodes.size, dtype=float))

    b_rad = np.full(node_count, 4.5e-16)
    ni_sq = np.full(node_count, 1.0e20)
    p_esc = 0.14

    spec_cross = AbsorberReabsorptionSpec.control_volume_conservative(
        layer_id="abs_cross",
        node_count=node_count,
        emitter_nodes=emit_nodes,
        cell_measures_m=dx,
        escape_probability=p_esc,
        intrinsic_radiative_B_m3_s=b_rad,
        intrinsic_product_m6=ni_sq,
        receiver_nodes=recv_nodes,
        receiver_shape=recv_shape,
    )
    n = rng.uniform(1e18, 5e21, size=node_count)
    p = rng.uniform(1e18, 5e21, size=node_count)

    ev = evaluate_radiative_reabsorption((spec_cross,), n, p, derivatives=True)
    budget = ev.budgets[0]
    assert ev.factors[0].active is True
    assert budget.emitted_flux_m2_s > 0.0
    assert abs(budget.closure_residual_m2_s) <= 1e-14 * budget.emitted_flux_m2_s
    # Explicit cell-weighted integral check: sum(dx * G_rad) == (1 - P_esc) * emitted
    integrated_reabs = float(np.dot(dx, ev.generation_m3_s))
    assert integrated_reabs == pytest.approx((1.0 - p_esc) * budget.emitted_flux_m2_s, rel=1e-14)

    # Spatially uniform n*p equivalence on self-reabsorbing uniform receiver:
    spec_self = AbsorberReabsorptionSpec.control_volume_conservative(
        layer_id="abs_self",
        node_count=node_count,
        emitter_nodes=emit_nodes,
        cell_measures_m=dx,
        escape_probability=p_esc,
        intrinsic_radiative_B_m3_s=b_rad,
        intrinsic_product_m6=ni_sq,
    )
    np_const = 2.5e38
    n_uni = np.linspace(1.0e19, 5.0e19, node_count)
    p_uni = np_const / n_uni
    ev_uni = evaluate_radiative_reabsorption((spec_self,), n_uni, p_uni, derivatives=False)
    net_full_minus_reabs = b_rad[emit_nodes] * (n_uni[emit_nodes] * p_uni[emit_nodes] - ni_sq[emit_nodes]) - ev_uni.generation_m3_s[emit_nodes]
    b_eff = effective_radiative_coefficient(
        b_rad, (np.isin(np.arange(node_count), emit_nodes),), (p_esc,)
    )
    local_scaled = b_eff[emit_nodes] * (n_uni[emit_nodes] * p_uni[emit_nodes] - ni_sq[emit_nodes])
    np.testing.assert_allclose(net_full_minus_reabs, local_scaled, rtol=1e-14, atol=0.0)


def test_reabsorption_rank1_jacobian_matches_central_fd() -> None:
    rng = np.random.default_rng(42)
    node_count = 14
    x: NDArray[np.float64] = np.cumsum(np.r_[0.0, rng.uniform(8e-9, 30e-9, size=node_count - 1)])
    dx = np.empty(node_count)
    dx[0] = 0.5 * (x[1] - x[0])
    dx[-1] = 0.5 * (x[-1] - x[-2])
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])

    spec_cv = AbsorberReabsorptionSpec.control_volume_conservative(
        layer_id="abs_cv",
        node_count=node_count,
        emitter_nodes=np.arange(2, 9),
        cell_measures_m=dx,
        escape_probability=0.22,
        intrinsic_radiative_B_m3_s=np.full(node_count, 3.0e-16),
        intrinsic_product_m6=np.full(node_count, 1.0e20),
        receiver_nodes=np.arange(5, 12),
    )
    mask_hist: NDArray[np.bool_] = np.zeros(node_count, dtype=bool)
    mask_hist[3:11] = True
    spec_hist = AbsorberReabsorptionSpec.historical_trapezoid_uniform(
        layer_id="abs_hist",
        nodes_m=x,
        absorber_mask=mask_hist,
        thickness_m=float(x[10] - x[3]),
        escape_probability=0.35,
        intrinsic_radiative_B_m3_s=np.full(node_count, 2.0e-16),
        intrinsic_product_m6=np.full(node_count, 1.0e20),
    )

    n0 = rng.uniform(2e19, 8e20, size=node_count)
    p0 = rng.uniform(2e19, 8e20, size=node_count)
    specs = (spec_cv, spec_hist)
    ev = evaluate_radiative_reabsorption(specs, n0, p0, derivatives=True)
    dn_ana, dp_ana = dense_reabsorption_jacobian(ev, node_count)

    dn_fd = np.zeros_like(dn_ana)
    dp_fd = np.zeros_like(dp_ana)
    for j in range(node_count):
        hn = 1e-4 * n0[j]
        np_plus = n0.copy()
        np_minus = n0.copy()
        np_plus[j] += hn
        np_minus[j] -= hn
        g_p = evaluate_radiative_reabsorption(specs, np_plus, p0, derivatives=False).generation_m3_s
        g_m = evaluate_radiative_reabsorption(specs, np_minus, p0, derivatives=False).generation_m3_s
        dn_fd[:, j] = (g_p - g_m) / (2.0 * hn)

        hp = 1e-4 * p0[j]
        pp_plus = p0.copy()
        pp_minus = p0.copy()
        pp_plus[j] += hp
        pp_minus[j] -= hp
        gp_p = evaluate_radiative_reabsorption(specs, n0, pp_plus, derivatives=False).generation_m3_s
        gp_m = evaluate_radiative_reabsorption(specs, n0, pp_minus, derivatives=False).generation_m3_s
        dp_fd[:, j] = (gp_p - gp_m) / (2.0 * hp)

    rel_n = np.linalg.norm(dn_ana - dn_fd) / np.linalg.norm(dn_fd)
    rel_p = np.linalg.norm(dp_ana - dp_fd) / np.linalg.norm(dp_fd)
    assert rel_n <= 1e-10
    assert rel_p <= 1e-10


# ---------------------------------------------------------------------------
# 3. Switching-surface directional derivatives (reabsorption, TE cap, PF cusp)
# ---------------------------------------------------------------------------


def test_switching_surfaces_expose_both_side_directional_derivatives_and_gate_ac() -> None:
    node_count = 6
    dx = np.full(node_count, 50e-9)
    ni_sq = np.full(node_count, 1.0e20)
    spec = AbsorberReabsorptionSpec.control_volume_conservative(
        layer_id="abs",
        node_count=node_count,
        emitter_nodes=np.arange(1, 5),
        cell_measures_m=dx,
        escape_probability=0.1,
        intrinsic_radiative_B_m3_s=np.full(node_count, 5.0e-16),
        intrinsic_product_m6=ni_sq,
    )
    # Exact dark equilibrium: n * p == ni^2 => R_tot == 0 on clamp surface!
    n_eq = np.full(node_count, 1.0e10)
    p_eq = np.full(node_count, 1.0e10)
    dn = np.full(node_count, 1.0e9)
    dp = np.full(node_count, 1.0e9)

    dd_fwd = radiative_reabsorption_directional_derivative((spec,), n_eq, p_eq, dn, dp)
    dd_rev = radiative_reabsorption_directional_derivative((spec,), n_eq, p_eq, -dn, -dp)
    assert dd_fwd.smooth is False
    assert dd_rev.smooth is False
    assert dd_fwd.non_smooth_layers == ("abs",)
    assert np.any(dd_fwd.directional_derivative_m3_s > 0.0)
    assert np.all(dd_rev.directional_derivative_m3_s == 0.0)

    # Verify directional derivative against one-sided FD along +d and -d.
    eps = 1.0e-4
    g_fwd = evaluate_radiative_reabsorption((spec,), n_eq + eps * dn, p_eq + eps * dp).generation_m3_s
    g_rev = evaluate_radiative_reabsorption((spec,), n_eq - eps * dn, p_eq - eps * dp).generation_m3_s
    np.testing.assert_allclose(g_fwd / eps, dd_fwd.directional_derivative_m3_s, rtol=2e-4)
    np.testing.assert_array_equal(g_rev, 0.0)

    # Thermionic hard-cap crossover |J_sg| == |J_te|:
    j_sg = np.array([10.0, -5.0])
    j_te = np.array([10.0, 20.0])
    dj_sg = np.array([2.0, 1.0])
    dj_te = np.array([0.5, 0.5])
    dd_te_fwd = thermionic_cap_directional_derivative(j_sg, j_te, dj_sg, dj_te)
    dd_te_rev = thermionic_cap_directional_derivative(j_sg, j_te, -dj_sg, -dj_te)
    assert bool(dd_te_fwd.smooth[0]) is False
    assert bool(dd_te_fwd.smooth[1]) is True
    assert dd_te_fwd.directional_derivative_A_m2[0] == pytest.approx(0.5)  # +eps -> |J_te| smaller -> dJ_te
    assert dd_te_rev.directional_derivative_A_m2[0] == pytest.approx(-2.0)  # -eps -> |J_sg| smaller -> -dJ_sg
    val_plus = thermionic_cap_value(j_sg + 1e-6 * dj_sg, j_te + 1e-6 * dj_te)
    val_0 = thermionic_cap_value(j_sg, j_te)
    assert (val_plus[0] - val_0[0]) / 1e-6 == pytest.approx(
        dd_te_fwd.directional_derivative_A_m2[0], rel=1e-6
    )

    # Poole-Frenkel cusp at E == 0:
    dd_pf = field_mobility_directional_derivative(
        np.array([1.0e-4]),
        np.array([0.0]),
        np.array([1.0e5]),
        np.array([0.0]),
        np.array([2.0]),
        np.array([1.0e-4]),
    )
    assert bool(dd_pf.smooth[0]) is False
    assert bool(dd_pf.finite[0]) is False

    ev_eq = evaluate_radiative_reabsorption((spec,), n_eq, p_eq, derivatives=True)
    assessment = assess_switching_surfaces(
        reabsorption=ev_eq,
        thermionic_electron=dd_te_fwd,
        field_mobility_electron=dd_pf,
    )
    assert assessment.smooth is False
    assert len(assessment.reasons) == 3
    with pytest.raises(PhotonRecyclingCapabilityError, match="non_smooth_ac_switching_surface"):
        require_smooth_ac_surfaces(assessment)


# ---------------------------------------------------------------------------
# 4. WKB closed-form actions, segment derivatives & intraband path Jacobian
# ---------------------------------------------------------------------------


def test_wkb_closed_form_fowler_nordheim_and_kane_zener_benchmarks() -> None:
    # Fowler-Nordheim linear triangular barrier: piecewise-linear WKB is exact!
    width = 6.0e-9
    height = 0.45
    mass = 0.25
    x = np.linspace(0.0, width, 25)
    barrier = height * (1.0 - x / width)
    s_num = wkb_action(x, barrier, 0.0, mass)
    s_ref = triangular_barrier_action(height, width, mass, 0.0)
    assert s_num == pytest.approx(s_ref, rel=1e-14)

    # Uniform-field Kane two-band Zener action:
    gap_eV = 1.2
    field_V_m = 8.0e7
    span = 40.0e-9
    x_z = np.linspace(0.0, span, 4001)
    ec = gap_eV / 2.0 - field_V_m * (x_z - 0.5 * span)
    ev = ec - gap_eV
    s_two = two_band_action(x_z, ec, ev, 0.0, 0.15, anchor_face=2000)
    s_kane = kane_uniform_field_action(gap_eV, field_V_m, 0.15)
    assert s_two == pytest.approx(s_kane, rel=1e-4)

    # Static TFE factor limits:
    assert tfe_gamma(0.0, 1.0e24, 0.2, 10.0, 0.02585) == 1.0
    g_pos = tfe_gamma(0.35, 2.0e24, 0.2, 10.0, 0.02585)
    assert g_pos > 1.0
    assert tfe_gamma(-0.35, 2.0e24, 0.2, 10.0, 0.02585) == g_pos


def test_wkb_segment_action_derivatives_and_intraband_path_jacobian_vs_fd() -> None:
    x = np.array([0.0, 1.2e-9, 2.5e-9, 4.1e-9, 6.0e-9, 8.3e-9])
    # Include both sloped, flat (U_L == U_R), and crossing segments!
    barrier = np.array([-0.10, 0.25, 0.25, 0.42, 0.12, -0.08])
    energies = np.array([0.0, 0.15, 0.28])
    mass = 0.22

    derivs = piecewise_linear_wkb_segment_action_derivatives(x, barrier, energies, mass)
    eps = 1.0e-6
    for j in range(x.size):
        bp = barrier.copy()
        bm = barrier.copy()
        bp[j] += eps
        bm[j] -= eps
        sp = piecewise_linear_wkb_segment_action_derivatives(x, bp, energies, mass).segment_actions
        sm = piecewise_linear_wkb_segment_action_derivatives(x, bm, energies, mass).segment_actions
        fd_j = (sp - sm) / (2.0 * eps)
        ana_j = np.zeros_like(fd_j)
        if j < x.size - 1:
            ana_j[:, j] += derivs.d_action_d_left_barrier_eV[:, j]
        if j > 0:
            ana_j[:, j - 1] += derivs.d_action_d_right_barrier_eV[:, j - 1]
        np.testing.assert_allclose(ana_j, fd_j, rtol=1e-9, atol=1e-12)

    # Full intraband_path_flux_and_jacobian FD verification wrt BOTH barrier & quasi_fermi!
    pts = 37
    x_bar = np.linspace(0.0, 12.0e-9, pts)
    u_bar = 0.32 * np.maximum(1.0 - np.abs(x_bar - 6.0e-9) / 2.2e-9, 0.0) + 0.015 * (
        x_bar / x_bar[-1]
    )
    start, stop = 12, 25
    region = IntrabandBarrierRegion(
        "resolved_spike", 0, start, stop, pts, float(x_bar[start]), float(x_bar[stop])
    )
    qf = -0.22 - 0.035 * (x_bar / x_bar[-1]) + 0.004 * np.sin(2.0 * math.pi * x_bar / x_bar[-1])
    ch = IntrabandTunnellingChannel(
        enabled=True, electron_effective_mass_rel=0.2, carrier="electron", energy_quadrature_order=32
    )
    vt = 0.02585
    lin = intraband_path_flux_and_jacobian(
        x_bar, u_bar, ch, region=region, quasi_fermi_eV=qf, thermal_voltage_V=vt
    )

    fd_u = np.zeros_like(lin.d_face_flux_d_barrier_eV)
    fd_qf = np.zeros_like(lin.d_face_flux_d_quasi_fermi_eV)
    h = 1.0e-7
    for j in range(pts):
        up = u_bar.copy()
        um = u_bar.copy()
        up[j] += h
        um[j] -= h
        fp = intraband_path_flux(
            x_bar, up, ch, region=region, quasi_fermi_eV=qf, thermal_voltage_V=vt
        ).particle_face_flux_m2_s
        fm = intraband_path_flux(
            x_bar, um, ch, region=region, quasi_fermi_eV=qf, thermal_voltage_V=vt
        ).particle_face_flux_m2_s
        fd_u[:, j] = (fp - fm) / (2.0 * h)

        qp = qf.copy()
        qm = qf.copy()
        qp[j] += h
        qm[j] -= h
        fqp = intraband_path_flux(
            x_bar, u_bar, ch, region=region, quasi_fermi_eV=qp, thermal_voltage_V=vt
        ).particle_face_flux_m2_s
        fqm = intraband_path_flux(
            x_bar, u_bar, ch, region=region, quasi_fermi_eV=qm, thermal_voltage_V=vt
        ).particle_face_flux_m2_s
        fd_qf[:, j] = (fqp - fqm) / (2.0 * h)

    rel_u = np.linalg.norm(lin.d_face_flux_d_barrier_eV - fd_u) / np.linalg.norm(fd_u)
    rel_qf = np.linalg.norm(lin.d_face_flux_d_quasi_fermi_eV - fd_qf) / np.linalg.norm(fd_qf)
    assert rel_u <= 1e-9
    assert rel_qf <= 1e-9


# ---------------------------------------------------------------------------
# 5. Structural equilibrium reciprocity & explicit capability refusals
# ---------------------------------------------------------------------------


def test_all_four_wkb_channels_have_zero_net_flux_at_equilibrium_and_fail_closed() -> None:
    x = np.linspace(0.0, 20.0e-9, 81)
    phi = 0.6 * (x / x[-1])
    chi = np.where((x >= 8e-9) & (x <= 12e-9), 3.65, 4.0)
    eg = np.where((x >= 8e-9) & (x <= 12e-9), 2.0, 1.1)
    ec = conduction_band_eV(phi, chi)
    ev = valence_band_eV(phi, chi, eg)
    ef_eq = np.full_like(x, -4.25)
    vt = 0.02585

    # 1) Intraband local & path flux at flat E_F == 0 net flux:
    ch_intra = IntrabandTunnellingChannel(enabled=True, carrier="electron", energy_quadrature_order=24)
    f_intra = intraband_flux(x, ec, ch_intra, anchor_face=32, carrier="electron", quasi_fermi_eV=ef_eq, thermal_voltage_V=vt)
    assert f_intra.net_flux_m2_s == 0.0

    # 2) Band-to-band Zener at flat E_F == 0 net flux:
    x_btbt = np.linspace(0.0, 30.0e-9, 121)
    ec_btbt = 0.8 - 1.0e8 * x_btbt
    ev_btbt = ec_btbt - 1.0
    ch_btbt = BandToBandTunnellingChannel(enabled=True, energy_quadrature_order=24)
    f_btbt = band_to_band_flux(
        x_btbt, ec_btbt, ev_btbt, ch_btbt, anchor_face=60, quasi_fermi_eV=np.full_like(x_btbt, -0.5), thermal_voltage_V=vt
    )
    assert f_btbt.net_flux_m2_s == 0.0

    # 3) Contact Schottky tunnelling at E_Fm == E_Fs == 0 net flux:
    x_c = np.linspace(0.0, 15.0e-9, 61)
    bar_c = 0.4 * np.exp(-x_c / 4.0e-9)
    ch_c = ContactTunnellingChannel(enabled=True, barrier_height_eV=0.4, side="left", energy_quadrature_order=24)
    f_c = contact_tunnelling_flux(
        x_c, bar_c, ch_c, anchor_face=0, carrier="electron", metal_fermi_eV=-0.1, quasi_fermi_eV=np.full_like(x_c, -0.1), thermal_voltage_V=vt
    )
    assert f_c.net_flux_m2_s == 0.0

    # 4) Interface defect-assisted rate: refuses None occupancy; vanishes at equilibrium occupancy!
    ch_tat = InterfaceDefectAssistedTunnellingChannel(enabled=True)
    with pytest.raises(TunnellingChannelCapabilityError, match="explicit interface occupancy"):
        interface_defect_assisted_rate(
            x, ec, ev, ch_tat,
            anchor_face=32, trap_energy_eV=float(ec[32]) - 0.35, occupancy=None,
            trap_density_m2=1e16, electron_capture_velocity_m_s=1e3, hole_capture_velocity_m_s=1e3,
            electron_density_m3=1e20, hole_density_m3=1e18,
            electron_reference_density_m3=1e19, hole_reference_density_m3=1e19,
        )
    # Detailed-balance equilibrium where n*p == n1*p1 => f_eq gives electron_net == hole_net == 0!
    tat_eq = interface_defect_assisted_rate(
        x, ec, ev, ch_tat,
        anchor_face=32, trap_energy_eV=float(ec[32]) - 0.35, occupancy=0.5,
        trap_density_m2=1e16, electron_capture_velocity_m_s=1e3, hole_capture_velocity_m_s=1e3,
        electron_density_m3=1e19, hole_density_m3=1e19,
        electron_reference_density_m3=1e19, hole_reference_density_m3=1e19,
    )
    assert tat_eq.electron_net_rate_m2_s == 0.0
    assert tat_eq.hole_net_rate_m2_s == 0.0
    assert tat_eq.net_rate_m2_s == 0.0

    # Schema & device capability refusals:
    with pytest.raises(TunnellingChannelSchemaError, match="algebraically eliminated"):
        InterfaceDefectAssistedTunnellingChannel(enabled=True, requires_explicit_occupancy=False)  # type: ignore[arg-type]
    with pytest.raises(TunnellingChannelSchemaError, match="barrier_height_eV"):
        ContactTunnellingChannel(enabled=True, barrier_height_eV=0.0)

    doc_intra = TunnellingChannelDocument(intraband=ch_intra)
    with pytest.raises(TunnellingChannelCapabilityError, match="quasi_fermi_steady_state"):
        compile_tunnelling_channels(doc_intra, node_count=20, interface_nodes=(8,), driver="mol_transient")
