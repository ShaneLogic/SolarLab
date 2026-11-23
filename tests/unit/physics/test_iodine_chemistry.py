"""Unit tests for SolarLab iodine reaction-diffusion constitutive kernels (M1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.contracts import ContractError
from solarlab.physics.chemistry import (
    CANONICAL_IODINE_NETWORK,
    IODINE_REACTIONS,
    IODINE_SPECIES,
    BoundedSpeciesMap,
    IodineKineticsParameters,
    PositiveSpeciesMap,
    SoftplusSpeciesMap,
    StoichiometricNetwork,
    equilibrium_iodine_populations_m3,
    evaluate_iodine_kinetics,
    harmonic_face_diffusivity_m2_s,
    smoluchowski_rate_m3_s,
    species_sg_face_flux,
)


def _table_s1_perovskite_params(
    size: int = 5,
    *,
    thermodynamic_mode: str = "wegscheider_detailed_balance",
    reaction_e_order: str = "bimolecular_n",
) -> IodineKineticsParameters:
    """Bitton & Tessler (2023) SI Table S1 perovskite kinetic parameters in SI."""
    vt_eV = 0.02585199
    nc_m3 = 7.0e18 * 1e6
    nv_m3 = 2.5e18 * 1e6
    eg_eV = 1.6
    ni_sq_m6 = nc_m3 * nv_m3 * math.exp(-eg_eV / vt_eV)
    return IodineKineticsParameters(
        k_a_m3_s=np.full(size, 1.0e-11 * 1e-6),
        k_b_s_1=np.full(size, 3.0e5),
        k_c_m3_s=np.full(size, 5.0e-7 * 1e-6),
        k_d_m3_s=np.full(size, 5.0e-15 * 1e-6),
        k_e_m3_s=np.full(size, 1.3e-8 * 1e-6),
        k_e_3body_m6_s=np.full(size, 1.3e-24 * 1e-12),
        n_i_sq_m6=np.full(size, ni_sq_m6),
        K_diss_m3=np.full(size, 1.0e8 * 1e6),
        n_e_star_m3=np.full(size, 1.0e8 * 1e6),
        thermodynamic_mode=thermodynamic_mode,  # type: ignore[arg-type]
        reaction_e_order=reaction_e_order,  # type: ignore[arg-type]
    )


def test_canonical_stoichiometric_network_nullspaces_and_bug_rejection() -> None:
    net = CANONICAL_IODINE_NETWORK
    assert net.species_ids == IODINE_SPECIES
    assert net.reaction_ids == IODINE_REACTIONS
    assert np.array_equal(net.charge_numbers @ net.nu, np.zeros(5, dtype=np.int64))
    assert np.array_equal(net.iodine_atom_counts @ net.nu, np.zeros(5, dtype=np.int64))

    # Verify exact source formula S_- = -R_a + R_b + R_c + 2 R_e
    rates = np.array([2.0, 3.0, 5.0, 7.0, 11.0], dtype=np.float64)
    sources = net.project_sources(rates)
    assert sources[0] == pytest.approx(-5.0 - 2.0 * 11.0)
    assert sources[1] == pytest.approx(-2.0 + 3.0)
    assert sources[2] == pytest.approx(-2.0 + 3.0 + 5.0 + 2.0 * 11.0)
    assert sources[3] == pytest.approx(2.0 - 3.0 - 5.0 - 2.0 * 7.0)
    assert sources[4] == pytest.approx(7.0 - 11.0)
    assert float(net.charge_numbers @ sources) == 0.0
    assert float(net.iodine_atom_counts @ sources) == 0.0

    # Reject the non-conserving derivation typo S_- = -R_a - R_b + 2 R_e
    bad_nu = np.array(net.nu, copy=True)
    bad_nu[2, :] = [-1, -1, 0, 0, 2]
    with pytest.raises(ContractError, match="conservation_violated"):
        StoichiometricNetwork(
            species_ids=IODINE_SPECIES,
            reaction_ids=IODINE_REACTIONS,
            nu=bad_nu,
            charge_numbers=net.charge_numbers,
            iodine_atom_counts=net.iodine_atom_counts,
        )


def test_smoluchowski_rates_and_si_figure_s2a_quantitative_match() -> None:
    vt_V = 0.02585199
    r_ab_m = 0.2e-9  # 0.2 nm in Bitton & Tessler Table S1
    d_e_m2_s = 2.0e-4 * vt_V  # mu_e = 2 cm^2/(V s) -> 2e-4 m^2/(V s)
    d_0_m2_s = 1.0e-8 * 1e-4  # D_0 = 1e-8 cm^2/s -> 1e-12 m^2/s

    k_e_si = float(smoluchowski_rate_m3_s(d_e_m2_s, 1.0e-9 * 1e-4, r_ab_m)[0])
    k_d_si = float(smoluchowski_rate_m3_s(d_0_m2_s, d_0_m2_s, r_ab_m)[0])
    assert k_e_si * 1e6 == pytest.approx(1.30e-8, rel=1e-2)
    assert k_d_si * 1e6 == pytest.approx(5.03e-15, rel=1e-2)

    # SI Fig. S2(a) local steady balance k_d * c_0^2 = k_e * c_2 * n:
    # Bulk: c_0 ~ 2e11 cm^-3, n ~ 2e15 cm^-3 -> c_2 < 1e2 cm^-3 (below 1e5 axis)
    # Right interface: c_0 ~ 5e13 cm^-3, n ~ 1e11 cm^-3 -> c_2 ~ 9.6e9 cm^-3
    k_d_cm3_s = 5.0e-15
    k_e_cm3_s = 1.3e-8
    c2_bulk_cm3 = k_d_cm3_s * (2.0e11) ** 2 / (k_e_cm3_s * 2.0e15)
    c2_peak_cm3 = k_d_cm3_s * (5.0e13) ** 2 / (k_e_cm3_s * 1.0e11)
    assert c2_bulk_cm3 < 1.0e2
    assert 5.0e9 <= c2_peak_cm3 <= 2.0e10


@pytest.mark.parametrize("reaction_e_order", ["bimolecular_n", "termolecular_n2"])
def test_wegscheider_dark_thermal_equilibrium_vanishes_across_doping(
    reaction_e_order: str,
) -> None:
    params = _table_s1_perovskite_params(
        size=7,
        thermodynamic_mode="wegscheider_detailed_balance",
        reaction_e_order=reaction_e_order,
    )
    ni = np.sqrt(params.n_i_sq_m6)
    factors = np.logspace(-6.0, 6.0, 7)
    n_eq = ni * factors
    p_eq = params.n_i_sq_m6 / n_eq
    c_tot = np.full(7, 1.0e24)  # 1e18 cm^-3 in SI

    cm_eq, c0_eq, c2_eq = equilibrium_iodine_populations_m3(c_tot, n_eq, params)
    np.testing.assert_allclose(cm_eq + c0_eq + 2.0 * c2_eq, c_tot, rtol=1e-15, atol=0.0)

    ev = evaluate_iodine_kinetics(n_eq, p_eq, cm_eq, c0_eq, c2_eq, params)
    forward_scale = np.maximum(ev.rates_m3_s[0], 1.0)
    # Net hole oxidation R_a - R_b and reversible channels R_c, R_d, R_e vanish
    np.testing.assert_allclose(
        (ev.rates_m3_s[0] - ev.rates_m3_s[1]) / forward_scale, 0.0, atol=1e-14
    )
    for idx in (2, 3, 4):
        np.testing.assert_allclose(
            ev.rates_m3_s[idx] / forward_scale, 0.0, atol=1e-14
        )
    for sp in range(5):
        np.testing.assert_allclose(
            ev.sources_m3_s[sp] / forward_scale, 0.0, atol=1e-14
        )


@pytest.mark.parametrize(
    ("thermodynamic_mode", "reaction_e_order"),
    [
        ("wegscheider_detailed_balance", "bimolecular_n"),
        ("wegscheider_detailed_balance", "termolecular_n2"),
        ("irreversible_mass_action", "bimolecular_n"),
        ("irreversible_mass_action", "termolecular_n2"),
    ],
)
def test_analytical_kinetics_jacobian_matches_central_finite_differences(
    thermodynamic_mode: str,
    reaction_e_order: str,
) -> None:
    params = _table_s1_perovskite_params(
        size=4,
        thermodynamic_mode=thermodynamic_mode,
        reaction_e_order=reaction_e_order,
    )
    # Test on BOTH a near-dark-equilibrium regime (where forward and Wegscheider
    # reverse terms have comparable magnitude so FD resolves every reverse partial
    # to <= 1e-9 without float64 cancellation) AND a strong-injection regime.
    assert params.n_1_iodine_m3 is not None
    n_near = np.array([1.0e5, 5.0e6, 2.0e8, 1.0e10], dtype=np.float64)
    p_near = params.n_i_sq_m6 / n_near
    cm_eq, c0_eq, c2_eq = equilibrium_iodine_populations_m3(
        np.full(4, 1.0e24), n_near, params
    )
    near_eq_state = [
        1.7 * n_near,
        0.6 * p_near,
        0.9 * cm_eq,
        1.4 * c0_eq + 1.0,
        2.3 * c2_eq + 1.0,
    ]
    high_inj_state = [
        np.array([2.0e18, 2.5e18, 8.0e19, 3.0e20], dtype=np.float64),
        np.array([4.0e17, 1.2e18, 5.0e17, 9.0e16], dtype=np.float64),
        np.array([9.5e22, 1.0e23, 6.0e22, 2.0e22], dtype=np.float64),
        np.array([8.0e18, 4.0e18, 2.0e19, 7.0e19], dtype=np.float64),
        np.array([5.0e16, 3.0e17, 9.0e17, 1.5e18], dtype=np.float64),
    ]

    eps = np.finfo(np.float64).eps
    net = CANONICAL_IODINE_NETWORK
    abs_nu = np.abs(net.nu.astype(np.float64))

    for state in (near_eq_state, high_inj_state):
        ev = evaluate_iodine_kinetics(
            state[0], state[1], state[2], state[3], state[4], params, derivatives=True
        )
        assert ev.dsources_dstate is not None

        # Exact tangent nullspaces: z^T dS/dy == 0 and m_I^T dS/dy == 0
        charge_tan = np.einsum(
            "s,sjm->jm", net.charge_numbers.astype(float), ev.dsources_dstate
        )
        atom_tan = np.einsum(
            "s,sjm->jm", net.iodine_atom_counts.astype(float), ev.dsources_dstate
        )
        scale = np.max(np.abs(ev.dsources_dstate), axis=0) + 1.0e-30
        np.testing.assert_allclose(charge_tan / scale, 0.0, atol=1e-14)
        np.testing.assert_allclose(atom_tan / scale, 0.0, atol=1e-14)

        gross_source = abs_nu @ np.abs(ev.rates_m3_s)
        # Gross internal operands inside each reaction rate (forward + reverse)
        assert params.k_e_3body_m6_s is not None
        r_internal = np.stack(
            (
                np.abs(ev.rates_m3_s[0]),
                np.abs(ev.rates_m3_s[1]),
                params.k_c_m3_s * (state[3] * state[0] + params.n_1_iodine_m3 * state[2]),
                params.k_d_m3_s * (state[3] ** 2 + params.K_diss_m3 * state[4]),
                np.abs(ev.rates_m3_s[4])
                + params.k_e_m3_s * state[4] * state[0]
                + params.k_e_3body_m6_s * state[4] * (state[0] ** 2),
            ),
            axis=0,
        )
        gross_total = np.maximum(gross_source, abs_nu @ r_internal)

        for dep in range(5):
            step = 1.0e-6 * np.maximum(state[dep], 1.0e-20)
            plus = [arr.copy() for arr in state]
            minus = [arr.copy() for arr in state]
            plus[dep] += step
            minus[dep] -= step
            ev_plus = evaluate_iodine_kinetics(
                plus[0], plus[1], plus[2], plus[3], plus[4], params, derivatives=False
            )
            ev_minus = evaluate_iodine_kinetics(
                minus[0],
                minus[1],
                minus[2],
                minus[3],
                minus[4],
                params,
                derivatives=False,
            )
            fd = (ev_plus.sources_m3_s - ev_minus.sources_m3_s) / (2.0 * step[None, :])
            roundoff = 32.0 * eps * gross_total / step[None, :]
            assert np.all(
                np.abs(ev.dsources_dstate[:, dep, :] - fd)
                <= 1.0e-8 * np.abs(fd) + roundoff
            )


def test_positive_and_bounded_species_maps_overflow_free_and_cancellation_safe() -> None:
    pmap = PositiveSpeciesMap(
        scale_m3=np.full(5, 1.0e24),
        eta=np.full(5, 1.0e-9),
        vacuum_floor_m3=1.0,
    )
    # Extreme coordinates (from deep empty layer -1e5 to high density +1e5)
    u_ext = np.array([-1.0e5, -1.0, 0.0, 1.0, 1.0e5], dtype=np.float64)
    dens, tan = pmap.density_and_tangent_m3(u_ext)
    assert np.all(np.isfinite(dens)) and np.all(dens > 0.0)
    assert np.all(np.isfinite(tan)) and np.all(tan > 0.0)

    # Exact round-trip including c = 0 mapped to vacuum_floor_m3
    targets = np.array([0.0, 1.0, 1.0e12, 1.0e18, 1.0e24], dtype=np.float64)
    coords = pmap.coordinate_from_density(targets)
    recovered = pmap.density_m3(coords)
    np.testing.assert_allclose(recovered, np.maximum(targets, 1.0), rtol=1e-14)

    # Cancellation-free secant at tiny representable step matches exact tangent * du
    raw_step = 1.0e-12 * np.maximum(np.abs(u_ext), 1.0e-9)
    u_next = u_ext + raw_step
    actual_du = u_next - u_ext
    secant = pmap.secant_increment_m3(u_next, u_ext)
    np.testing.assert_allclose(secant, tan * actual_du, rtol=1e-9)

    # Bounded logistic map round-trip and cancellation-free secant
    bmap = BoundedSpeciesMap(
        site_limit_m3=np.full(4, 1.0e25),
        logit_reference=np.full(4, -2.1972245773362196),
    )
    u_b = np.array([-40.0, -1.5, 0.5, 35.0], dtype=np.float64)
    dens_b, tan_b = bmap.density_and_tangent_m3(u_b)
    assert np.all((dens_b > 0.0) & (dens_b < 1.0e25))
    u_b_next = u_b + 1.0e-11 * np.maximum(np.abs(u_b), 1.0)
    actual_du_b = u_b_next - u_b
    sec_b = bmap.secant_increment_m3(u_b_next, u_b)
    np.testing.assert_allclose(sec_b, tan_b * actual_du_b, rtol=1e-9)
    # Large step identity check against direct subtraction where uncancelled
    u_b2 = np.array([-2.0, 1.0, 3.0, -1.0], dtype=np.float64)
    np.testing.assert_allclose(
        bmap.secant_increment_m3(u_b2, u_b),
        bmap.density_m3(u_b2) - dens_b,
        rtol=1e-13,
    )

    # Smooth log-linear Softplus map round-trip and cancellation-free secant
    spmap = SoftplusSpeciesMap(scale_m3=np.full(5, 1.0e20), vacuum_floor_m3=1.0)
    u_sp = np.array([-60.0, -5.0, 0.0, 8.0, 1.0e5], dtype=np.float64)
    dens_sp, tan_sp = spmap.density_and_tangent_m3(u_sp)
    assert np.all(np.isfinite(dens_sp)) and np.all(dens_sp > 0.0)
    assert np.all(np.isfinite(tan_sp)) and np.all(tan_sp > 0.0)
    sp_targets = np.array([0.0, 1.0, 1.0e14, 1.0e20, 1.0e25], dtype=np.float64)
    sp_coords = spmap.coordinate_from_density(sp_targets)
    np.testing.assert_allclose(
        spmap.density_m3(sp_coords), np.maximum(sp_targets, 1.0), rtol=1e-13
    )
    u_sp_next = u_sp + 1.0e-12 * np.maximum(np.abs(u_sp), 1.0)
    np.testing.assert_allclose(
        spmap.secant_increment_m3(u_sp_next, u_sp),
        tan_sp * (u_sp_next - u_sp),
        rtol=1e-9,
    )


def test_species_sg_face_flux_equilibrium_and_analytical_derivatives() -> None:
    vt = 0.02585199
    phi = np.array([0.0, 0.35, -0.25, 0.60, 0.15], dtype=np.float64)
    dx = np.array([5.0e-9, 1.2e-8, 2.0e-8, 8.0e-9], dtype=np.float64)
    d_node = np.array([1.0e-16, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-16], dtype=np.float64)
    d_face = harmonic_face_diffusivity_m2_s(d_node[:-1], d_node[1:])
    k_part_nodes = np.array([0.02, 1.0, 1.0, 1.0, 0.05], dtype=np.float64)
    log_step = np.log(k_part_nodes[1:] / k_part_nodes[:-1])

    # Equilibrium profile for z = -1 with partition jump:
    # c_eq_i = c_ref * K_part_i * exp(+phi_i / V_T)
    c_eq = 1.0e22 * k_part_nodes * np.exp((phi - phi[0]) / vt)
    ev_eq = species_sg_face_flux(
        phi,
        c_eq,
        dx,
        d_face,
        vt,
        charge_number=-1,
        log_partition_step=log_step,
    )
    conductance_scale = (d_face / dx) * np.maximum(c_eq[:-1], c_eq[1:])
    np.testing.assert_allclose(ev_eq.flux_m2_s / conductance_scale, 0.0, atol=1e-14)

    # Out-of-equilibrium derivative check against central finite differences
    c_neq = np.array([2.0e19, 9.0e23, 7.5e23, 1.1e24, 4.0e18], dtype=np.float64)
    for z in (-1, 0):
        ev = species_sg_face_flux(
            phi,
            c_neq,
            dx,
            d_face,
            vt,
            charge_number=z,
            log_partition_step=log_step,
            derivatives=True,
        )
        assert (
            ev.dflux_dc_left is not None
            and ev.dflux_dc_right is not None
            and ev.dflux_dphi_left is not None
            and ev.dflux_dphi_right is not None
        )
        for face in range(dx.size):
            for side, dc_exact, dphi_exact in (
                (face, ev.dflux_dc_left[face], ev.dflux_dphi_left[face]),
                (face + 1, ev.dflux_dc_right[face], ev.dflux_dphi_right[face]),
            ):
                hc = 1.0e-6 * c_neq[side]
                cp, cm = c_neq.copy(), c_neq.copy()
                cp[side] += hc
                cm[side] -= hc
                fp = species_sg_face_flux(
                    phi, cp, dx, d_face, vt, charge_number=z,
                    log_partition_step=log_step, derivatives=False,
                ).flux_m2_s[face]
                fm = species_sg_face_flux(
                    phi, cm, dx, d_face, vt, charge_number=z,
                    log_partition_step=log_step, derivatives=False,
                ).flux_m2_s[face]
                assert dc_exact == pytest.approx((fp - fm) / (2.0 * hc), rel=1e-8, abs=1e-12)

                hphi = 1.0e-6
                pp, pm = phi.copy(), phi.copy()
                pp[side] += hphi
                pm[side] -= hphi
                gp = species_sg_face_flux(
                    pp, c_neq, dx, d_face, vt, charge_number=z,
                    log_partition_step=log_step, derivatives=False,
                ).flux_m2_s[face]
                gm = species_sg_face_flux(
                    pm, c_neq, dx, d_face, vt, charge_number=z,
                    log_partition_step=log_step, derivatives=False,
                ).flux_m2_s[face]
                assert dphi_exact == pytest.approx(
                    (gp - gm) / (2.0 * hphi), rel=1e-8, abs=1e-12
                )
