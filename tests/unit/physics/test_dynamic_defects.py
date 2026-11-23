"""Unit tests (T1) for dynamic, energy-distributed, multivalent, and metastable defect physics (P04-10)."""

from __future__ import annotations

from collections.abc import Iterator
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.physics.defects import (
    DIST_CONDUCTION_BAND_TAIL,
    DIST_GAUSSIAN,
    DIST_SINGLE_LEVEL,
    DIST_UNIFORM,
    DIST_VALENCE_BAND_TAIL,
    DOUBLE_ELECTRON_CAPTURE,
    DOUBLE_HOLE_CAPTURE,
    ELECTRON_CAPTURE_PLUS_HOLE_EMISSION,
    HOLE_CAPTURE_PLUS_ELECTRON_EMISSION,
    TRANSITION_ACCEPTOR,
    TRANSITION_DONOR,
    TRANSITION_NEUTRAL,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
    DefectDistributionSpec,
    DynamicInterfaceTrapSpec,
    FrozenMetastableRegion,
    MetastableConversionKineticsSpec,
    MetastableDefectSpec,
    MonovalentDefectRegion,
    MonovalentSpeciesSpec,
    MultivalentConfigurationSpec,
    MultivalentSpeciesSpec,
    MultivalentTransitionKineticsSpec,
    SpatialDensityProfileSpec,
    SpatialProfileKnot,
    advance_metastable_configuration,
    build_defect_energy_quadrature,
    compile_dynamic_bulk_trap_layout,
    configuration_charge_transfer_C_m3,
    configuration_from_logit,
    configuration_logit,
    density_weighted_mean_occupancy,
    distribution_shape_integral_eV,
    evaluate_dynamic_bulk_traps,
    evaluate_dynamic_bulk_traps_about_qss,
    evaluate_dynamic_interface_trap,
    evaluate_frozen_metastable_region,
    evaluate_metastable_configuration_closure,
    evaluate_monovalent_defect_closure,
    evaluate_monovalent_region_closure,
    evaluate_multivalent_defect_closure,
    evaluate_spatial_density_profile,
    integrated_density_from_peak_density,
    logit_occupancy_secant,
    occupancy_from_logit_increment,
    occupancy_logit,
    peak_density_from_integrated_density,
    quasi_steady_bulk_trap_occupancy,
    solve_frozen_metastable_charge_neutrality,
    solve_monovalent_defect_charge_neutrality,
    solve_multivalent_defect_charge_neutrality,
    stable_expit,
    validate_defect_energy_quadrature_order,
)
from solarlab.physics.electrostatics.interface_traces import TraceElectrostatics
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    TwoSidedBulkState,
    TwoSidedInterfacePhysics,
    solve_charged_two_sided_interface_qss,
    solve_fixed_occupancy_two_sided_interface,
)
from solarlab.physics.statistics import thermal_voltage


@pytest.fixture(scope="module", autouse=True)
def _blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def test_energy_distributions_quadrature_and_refinement_convergence() -> None:
    """Verify exact zeroth moment, peak/integrated conversions, and quadrature convergence."""
    assert validate_defect_energy_quadrature_order(16) == 16
    with pytest.raises(ValueError):
        validate_defect_energy_quadrature_order(2)
    with pytest.raises(ValueError):
        validate_defect_energy_quadrature_order(True)

    gap = 1.60
    nt = 2.5e22
    specs = [
        DefectDistributionSpec(DIST_SINGLE_LEVEL, center_eV_above_vb=0.75, total_density_m3=nt),
        DefectDistributionSpec(
            DIST_UNIFORM,
            center_eV_above_vb=0.80,
            total_density_m3=nt,
            width_eV=0.20,
            width_convention=WIDTH_UNIFORM_FULL,
        ),
        DefectDistributionSpec(
            DIST_GAUSSIAN,
            center_eV_above_vb=0.80,
            total_density_m3=nt,
            width_eV=0.06,
            width_convention=WIDTH_GAUSSIAN_SIGMA,
            support_width_multiplier=6.0,
        ),
        DefectDistributionSpec(
            DIST_GAUSSIAN,
            center_eV_above_vb=0.80,
            total_density_m3=nt,
            width_eV=0.08,
            width_convention=WIDTH_SCAPS_CHARACTERISTIC,
            support_width_multiplier=None,  # Legacy band-gap-truncated [0, Eg]
        ),
        DefectDistributionSpec(
            DIST_CONDUCTION_BAND_TAIL,
            center_eV_above_vb=1.55,
            total_density_m3=nt,
            width_eV=0.04,
            support_width_multiplier=6.0,
        ),
        DefectDistributionSpec(
            DIST_VALENCE_BAND_TAIL,
            center_eV_above_vb=0.05,
            total_density_m3=nt,
            width_eV=0.04,
            support_width_multiplier=6.0,
        ),
    ]
    for spec in specs:
        quad = build_defect_energy_quadrature(spec, order=16, band_gap_eV=gap)
        assert math.isclose(float(np.sum(quad.density_weights_m3)), nt, rel_tol=1e-14, abs_tol=0.0)
        assert math.isclose(float(np.sum(quad.normalized_weights)), 1.0, rel_tol=0.0, abs_tol=1e-14)
        assert np.all(quad.energies_eV_above_vb >= 0.0)
        assert np.all(quad.energies_eV_above_vb <= gap)
        mean_occ = density_weighted_mean_occupancy(
            quad, np.full((quad.order, 3), 0.37, dtype=np.float64)
        )
        np.testing.assert_allclose(mean_occ, 0.37, rtol=1e-14, atol=0.0)

    shape = distribution_shape_integral_eV(
        DIST_GAUSSIAN,
        0.05,
        width_convention=WIDTH_GAUSSIAN_SIGMA,
        support_width_multiplier=6.0,
    )
    peak = peak_density_from_integrated_density(nt, shape)
    assert math.isclose(integrated_density_from_peak_density(peak, shape), nt, rel_tol=1e-15)

    # Independent quadrature-order convergence on smooth Gaussian trap closure:
    gauss_spec = MonovalentSpeciesSpec(
        "gauss_acc",
        TRANSITION_ACCEPTOR,
        specs[2],
        sigma_n_m2=1.0e-19,
        sigma_p_m2=2.0e-19,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=1.0e5,
    )
    n = np.array([3.0e20, 8.0e19])
    p = np.array([1.2e20, 5.0e20])
    ref = evaluate_monovalent_defect_closure(
        n,
        p,
        (gauss_spec,),
        band_gap_eV=gap,
        effective_conduction_dos_m3=1.0e25,
        effective_valence_dos_m3=1.0e25,
        temperature_K=300.0,
        energy_quadrature_order=64,
    )
    err_8 = float(
        np.max(
            np.abs(
                evaluate_monovalent_defect_closure(
                    n,
                    p,
                    (gauss_spec,),
                    band_gap_eV=gap,
                    effective_conduction_dos_m3=1.0e25,
                    effective_valence_dos_m3=1.0e25,
                    temperature_K=300.0,
                    energy_quadrature_order=8,
                ).total_recombination_rate_m3_s
                - ref.total_recombination_rate_m3_s
            )
            / ref.total_recombination_rate_m3_s
        )
    )
    err_24 = float(
        np.max(
            np.abs(
                evaluate_monovalent_defect_closure(
                    n,
                    p,
                    (gauss_spec,),
                    band_gap_eV=gap,
                    effective_conduction_dos_m3=1.0e25,
                    effective_valence_dos_m3=1.0e25,
                    temperature_K=300.0,
                    energy_quadrature_order=24,
                ).total_recombination_rate_m3_s
                - ref.total_recombination_rate_m3_s
            )
            / ref.total_recombination_rate_m3_s
        )
    )
    err_32 = float(
        np.max(
            np.abs(
                evaluate_monovalent_defect_closure(
                    n,
                    p,
                    (gauss_spec,),
                    band_gap_eV=gap,
                    effective_conduction_dos_m3=1.0e25,
                    effective_valence_dos_m3=1.0e25,
                    temperature_K=300.0,
                    energy_quadrature_order=32,
                ).total_recombination_rate_m3_s
                - ref.total_recombination_rate_m3_s
            )
            / ref.total_recombination_rate_m3_s
        )
    )
    assert err_32 < err_24 < err_8 and err_32 < 1.0e-10


def test_spatial_density_profile_conservation_and_validation() -> None:
    profile = SpatialDensityProfileSpec(
        (
            SpatialProfileKnot(0.0, 1.8),
            SpatialProfileKnot(0.5, 1.0),
            SpatialProfileKnot(1.0, 0.2),
        )
    )
    pos = np.linspace(0.0, 1.0, 201)
    vals = evaluate_spatial_density_profile(profile, pos)
    assert math.isclose(float(np.trapezoid(vals, pos)), 1.0, rel_tol=0.0, abs_tol=1e-14)
    assert vals[0] == pytest.approx(1.8)
    assert vals[-1] == pytest.approx(0.2)

    with pytest.raises(ValueError, match="unit layer average"):
        SpatialDensityProfileSpec((SpatialProfileKnot(0.0, 2.0), SpatialProfileKnot(1.0, 2.0)))


def test_monovalent_qss_and_dynamic_derivatives_and_secant_stability() -> None:
    """Verify cancellation-free logit secants, zero-density identity, and FD derivatives."""
    base = np.array([-50.0, -10.0, 0.0, 10.0, 50.0])
    shifted = base + np.array([1.0e-12, 1.0e-13, 1.0e-13, 1.0e-13, 1.0e-12])
    delta_exact = shifted - base
    sec = logit_occupancy_secant(shifted, base)
    expected_slope = stable_expit(base) * stable_expit(-base) * delta_exact
    np.testing.assert_allclose(sec, expected_slope, rtol=1e-11, atol=0.0)

    f_vals = np.array([1.0e-9, 0.25, 0.5, 0.999999])
    logits = occupancy_logit(f_vals)
    f_back = occupancy_from_logit_increment(logits, np.zeros_like(logits))
    np.testing.assert_allclose(f_back, f_vals, rtol=1e-14)

    gap = 1.60
    nc, nv, temp = 1.2e25, 1.8e25, 300.0
    vt = thermal_voltage(temp)
    ni_sq = nc * nv * math.exp(-gap / vt)
    n_eq = math.sqrt(ni_sq) * 3.0
    p_eq = ni_sq / n_eq

    species = (
        MonovalentSpeciesSpec(
            "acc",
            TRANSITION_ACCEPTOR,
            DefectDistributionSpec(DIST_SINGLE_LEVEL, 0.72, 2.0e22),
            1.5e-19,
            2.5e-19,
            1.0e5,
            1.1e5,
        ),
        MonovalentSpeciesSpec(
            "don",
            TRANSITION_DONOR,
            DefectDistributionSpec(
                DIST_UNIFORM, 0.90, 1.5e22, width_eV=0.16, width_convention=WIDTH_UNIFORM_FULL
            ),
            2.0e-19,
            1.0e-19,
            1.0e5,
            1.0e5,
        ),
        MonovalentSpeciesSpec(
            "neu",
            TRANSITION_NEUTRAL,
            DefectDistributionSpec(DIST_SINGLE_LEVEL, 0.80, 3.0e22),
            1.0e-19,
            1.0e-19,
            1.0e5,
            1.0e5,
        ),
        MonovalentSpeciesSpec(
            "zero",
            TRANSITION_ACCEPTOR,
            DefectDistributionSpec(DIST_SINGLE_LEVEL, 0.65, 0.0),
            1.0e-19,
            1.0e-19,
            1.0e5,
            1.0e5,
        ),
    )

    eq_res = evaluate_monovalent_defect_closure(
        np.array([n_eq]),
        np.array([p_eq]),
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    assert float(np.max(np.abs(eq_res.total_recombination_rate_m3_s))) <= 1.0e-8
    assert eq_res.charge_density_C_m3[2, 0] == 0.0  # neutral
    assert eq_res.charge_density_C_m3[3, 0] == 0.0 and eq_res.recombination_rate_m3_s[3, 0] == 0.0

    # Non-equilibrium analytical derivatives vs central FD:
    n0 = np.array([2.5e20, 7.0e19, 4.0e21])
    p0 = np.array([1.1e20, 9.0e20, 3.0e19])
    res = evaluate_monovalent_defect_closure(
        n0,
        p0,
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    eps = 1.0e-6
    hi_n = evaluate_monovalent_defect_closure(
        n0 * (1.0 + eps),
        p0,
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    lo_n = evaluate_monovalent_defect_closure(
        n0 * (1.0 - eps),
        p0,
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    hi_p = evaluate_monovalent_defect_closure(
        n0,
        p0 * (1.0 + eps),
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    lo_p = evaluate_monovalent_defect_closure(
        n0,
        p0 * (1.0 - eps),
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    np.testing.assert_allclose(
        res.total_charge_derivative_n_C,
        (hi_n.total_charge_density_C_m3 - lo_n.total_charge_density_C_m3) / (2.0 * eps * n0),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        res.total_charge_derivative_p_C,
        (hi_p.total_charge_density_C_m3 - lo_p.total_charge_density_C_m3) / (2.0 * eps * p0),
        rtol=1e-8,
    )
    rec_scale = float(np.max(np.abs(res.total_recombination_rate_m3_s)))
    fd_rec_n = (
        hi_n.total_recombination_rate_m3_s - lo_n.total_recombination_rate_m3_s
    ) / (2.0 * eps * n0)
    fd_rec_p = (
        hi_p.total_recombination_rate_m3_s - lo_p.total_recombination_rate_m3_s
    ) / (2.0 * eps * p0)
    assert np.all(
        np.abs(res.total_recombination_derivative_n_s1 - fd_rec_n)
        <= 1.0e-8 * np.abs(res.total_recombination_derivative_n_s1)
        + 1.0e-13 * rec_scale / (eps * n0)
    )
    assert np.all(
        np.abs(res.total_recombination_derivative_p_s1 - fd_rec_p)
        <= 1.0e-8 * np.abs(res.total_recombination_derivative_p_s1)
        + 1.0e-13 * rec_scale / (eps * p0)
    )
    # Fixed quasi-Fermi potential shift dphi: dn/dphi = n/Vt, dp/dphi = -p/Vt
    dphi = 1.0e-7
    hi_phi = evaluate_monovalent_defect_closure(
        n0 * math.exp(dphi / vt),
        p0 * math.exp(-dphi / vt),
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    lo_phi = evaluate_monovalent_defect_closure(
        n0 * math.exp(-dphi / vt),
        p0 * math.exp(dphi / vt),
        species,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    np.testing.assert_allclose(
        res.total_charge_derivative_fixed_qf_C_m3_V,
        (hi_phi.total_charge_density_C_m3 - lo_phi.total_charge_density_C_m3) / (2.0 * dphi),
        rtol=1e-9,
    )

    # Dynamic bulk trap layout & derivatives vs FD:
    mask = np.array([False, True, True, True, False])
    region = MonovalentDefectRegion(
        identifier="abs",
        document_sha256="a" * 64,
        active_nodes=mask,
        band_gap_eV=np.full(3, gap),
        effective_conduction_dos_m3=np.full(3, nc),
        effective_valence_dos_m3=np.full(3, nv),
        temperature_K=temp,
        species=species[:3],
        energy_quadrature_order=8,
    )
    reg_qss = evaluate_monovalent_region_closure(
        region, np.r_[1e20, n0, 1e20], np.r_[1e20, p0, 1e20]
    )
    np.testing.assert_allclose(
        reg_qss.total_recombination_rate_m3_s, res.total_recombination_rate_m3_s, rtol=1e-12
    )

    layout = compile_dynamic_bulk_trap_layout((region,), node_count=5)
    n_grid = np.r_[1e20, n0, 1e20]
    p_grid = np.r_[1e20, p0, 1e20]
    f_qss = quasi_steady_bulk_trap_occupancy(n_grid, p_grid, layout)
    qss_dyn = evaluate_dynamic_bulk_traps_about_qss(
        n_grid,
        p_grid,
        f_qss,
        layout,
        reference_electron_density_m3=n_grid,
        reference_hole_density_m3=p_grid,
        reference_occupancy=f_qss,
        occupancy_increment=np.zeros(layout.size),
    )
    assert float(np.max(np.abs(qss_dyn.trap_storage_rate_m3_s))) == 0.0

    f_test = np.clip(f_qss * 0.73 + 0.11, 1.0e-4, 1.0 - 1.0e-4)
    dyn = evaluate_dynamic_bulk_traps(n_grid, p_grid, f_test, layout)
    df = 1.0e-7
    dyn_hi_f = evaluate_dynamic_bulk_traps(n_grid, p_grid, f_test + df, layout)
    dyn_lo_f = evaluate_dynamic_bulk_traps(n_grid, p_grid, f_test - df, layout)
    np.testing.assert_allclose(
        dyn.d_trap_storage_rate_df_m3_s,
        (dyn_hi_f.trap_storage_rate_m3_s - dyn_lo_f.trap_storage_rate_m3_s) / (2.0 * df),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        dyn.d_charge_df_C_m3,
        (dyn_hi_f.charge_density_C_m3 - dyn_lo_f.charge_density_C_m3) / (2.0 * df),
        rtol=1e-9,
        atol=1e-20,
    )

    # Contact neutrality with monovalent traps:
    neut = solve_monovalent_defect_charge_neutrality(
        temperature_K=temp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        donor_density_m3=5.0e22,
        acceptor_density_m3=0.0,
        species=species[:2],
        electron_affinity_eV=3.9,
    )
    assert neut.normalized_charge_residual <= 1.0e-12
    assert neut.work_function_eV is not None

    # Dynamic interface trap derivatives vs FD:
    ispec = DynamicInterfaceTrapSpec(
        interface_id="htl_abs",
        left_node=1,
        right_node=2,
        charge_type=TRANSITION_ACCEPTOR,
        sheet_density_m2=2.0e15,
        v_n_m_s=12.0,
        v_p_m_s=8.0,
        n1_left_m3=1.0e14,
        p1_left_m3=2.0e14,
        n1_right_m3=3.0e14,
        p1_right_m3=4.0e14,
        equilibrium_occupancy=0.42,
    )
    iev = evaluate_dynamic_interface_trap(
        ispec,
        n_left_m3=2.0e19,
        p_left_m3=5.0e20,
        n_right_m3=8.0e19,
        p_right_m3=3.0e19,
        occupancy=0.61,
    )
    iev_hi = evaluate_dynamic_interface_trap(
        ispec,
        n_left_m3=2.0e19,
        p_left_m3=5.0e20,
        n_right_m3=8.0e19,
        p_right_m3=3.0e19,
        occupancy=0.61 + df,
    )
    iev_lo = evaluate_dynamic_interface_trap(
        ispec,
        n_left_m3=2.0e19,
        p_left_m3=5.0e20,
        n_right_m3=8.0e19,
        p_right_m3=3.0e19,
        occupancy=0.61 - df,
    )
    assert iev.d_storage_rate_df == pytest.approx(
        (iev_hi.trap_storage_rate_m2_s - iev_lo.trap_storage_rate_m2_s) / (2.0 * df), rel=1e-9
    )
    assert iev.d_sheet_charge_df == pytest.approx(
        (iev_hi.sheet_charge_C_m2 - iev_lo.sheet_charge_C_m2) / (2.0 * df), rel=1e-9
    )


def test_multivalent_simplex_detailed_balance_and_fd_derivatives() -> None:
    """Verify simplex bounds across extreme Fermi shifts, detailed balance, and FD Jacobians."""
    kin1 = MultivalentTransitionKineticsSpec(1.0e-19, 2.0e-19, 1.0e5, 1.2e5)
    kin2 = MultivalentTransitionKineticsSpec(3.0e-19, 1.5e-19, 1.0e5, 1.0e5)
    cfg = MultivalentConfigurationSpec(
        family="amphoteric",
        charge_states_e=(1, 0, -1),
        degeneracy_convention="scaps_binomial",
        state_degeneracies=(1.0, 2.0, 1.0),
        transition_energies_eV_above_vb=(0.55, 1.05),
        transition_kinetics=(kin1, kin2),
    )
    sp = MultivalentSpeciesSpec("amp_1", 4.0e22, cfg)

    gap, nc, nv, temp = 1.60, 1.0e25, 1.5e25, 300.0
    vt = thermal_voltage(temp)
    ni_sq = nc * nv * math.exp(-gap / vt)

    # Extreme Fermi-level sweep [-25 V, +25 V] on exact n*p == ni^2 equilibrium manifold:
    shifts = np.linspace(-25.0, 25.0, 21)
    n_ext = nc * np.exp(shifts)
    p_ext = nv * np.exp(-gap / vt - shifts)
    ext_res = evaluate_multivalent_defect_closure(
        n_ext,
        p_ext,
        sp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    assert np.all(ext_res.state_probability >= 0.0)
    np.testing.assert_allclose(np.sum(ext_res.state_probability, axis=0), 1.0, rtol=0.0, atol=1e-14)
    # Detailed balance on n*p == ni^2 manifold (rate normalized by characteristic capture scale):
    assert ext_res.maximum_master_residual_s1 <= 1.0e-10
    rate_scale = sp.total_density_m3 * kin1.capture_n_m3_s * np.maximum(n_ext, p_ext)
    assert float(np.max(np.abs(ext_res.total_recombination_rate_m3_s) / rate_scale)) <= 1.0e-15

    # Non-equilibrium FD tangents:
    n0 = np.array([3.0e20, 5.0e18, 9.0e21])
    p0 = np.array([2.0e20, 8.0e21, 4.0e18])
    res = evaluate_multivalent_defect_closure(
        n0,
        p0,
        sp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    eps = 1.0e-6
    hi_n = evaluate_multivalent_defect_closure(
        n0 * (1.0 + eps),
        p0,
        sp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    lo_n = evaluate_multivalent_defect_closure(
        n0 * (1.0 - eps),
        p0,
        sp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    hi_p = evaluate_multivalent_defect_closure(
        n0,
        p0 * (1.0 + eps),
        sp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    lo_p = evaluate_multivalent_defect_closure(
        n0,
        p0 * (1.0 - eps),
        sp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    rho_scale = float(np.max(np.abs(res.charge_density_C_m3)))
    np.testing.assert_allclose(
        res.state_probability_derivative_n_m3,
        (hi_n.state_probability - lo_n.state_probability) / (2.0 * eps * n0),
        rtol=1e-8,
        atol=1e-26,
    )
    fd_rho_n = (hi_n.charge_density_C_m3 - lo_n.charge_density_C_m3) / (2.0 * eps * n0)
    fd_rho_p = (hi_p.charge_density_C_m3 - lo_p.charge_density_C_m3) / (2.0 * eps * p0)
    assert np.all(
        np.abs(res.charge_derivative_n_C - fd_rho_n)
        <= 1.0e-8 * np.abs(res.charge_derivative_n_C) + 1.0e-13 * rho_scale / (eps * n0)
    )
    assert np.all(
        np.abs(res.charge_derivative_p_C - fd_rho_p)
        <= 1.0e-8 * np.abs(res.charge_derivative_p_C) + 1.0e-13 * rho_scale / (eps * p0)
    )
    rec_scale = float(np.max(np.abs(res.total_recombination_rate_m3_s)))
    fd_rec_n = (
        hi_n.total_recombination_rate_m3_s - lo_n.total_recombination_rate_m3_s
    ) / (2.0 * eps * n0)
    fd_rec_p = (
        hi_p.total_recombination_rate_m3_s - lo_p.total_recombination_rate_m3_s
    ) / (2.0 * eps * p0)
    assert np.all(
        np.abs(res.total_recombination_derivative_n_s1 - fd_rec_n)
        <= 1.0e-8 * np.abs(res.total_recombination_derivative_n_s1)
        + 1.0e-13 * rec_scale / (eps * n0)
    )
    assert np.all(
        np.abs(res.total_recombination_derivative_p_s1 - fd_rec_p)
        <= 1.0e-8 * np.abs(res.total_recombination_derivative_p_s1)
        + 1.0e-13 * rec_scale / (eps * p0)
    )

    mv_neut = solve_multivalent_defect_charge_neutrality(
        temperature_K=temp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        donor_density_m3=2.0e22,
        acceptor_density_m3=0.0,
        species=(sp,),
    )
    assert mv_neut.normalized_charge_residual <= 1.0e-12
    assert ni_sq > 0.0


def test_metastable_closure_frozen_region_and_exponential_advance() -> None:
    """Verify detailed-balance barrier gate, FD tangents, semigroup advance, and charge transfer."""
    gap, nc, nv, temp = 1.60, 1.0e25, 1.0e25, 300.0
    kin = MultivalentTransitionKineticsSpec(1.0e-19, 1.0e-19, 1.0e5, 1.0e5)
    donor_cfg = MultivalentConfigurationSpec(
        family="donor",
        charge_states_e=(2, 1, 0),
        degeneracy_convention="unity",
        state_degeneracies=(1.0, 1.0, 1.0),
        transition_energies_eV_above_vb=(1.10, 1.35),
        transition_kinetics=(kin, kin),
    )
    acc_cfg = MultivalentConfigurationSpec(
        family="acceptor",
        charge_states_e=(0, -1, -2),
        degeneracy_convention="unity",
        state_degeneracies=(1.0, 1.0, 1.0),
        transition_energies_eV_above_vb=(0.25, 0.50),
        transition_kinetics=(kin, kin),
    )
    tr = 0.85
    ec = 0.30
    hc = 0.25
    conv = MetastableConversionKineticsSpec(
        transition_energy_eV_above_vb=tr,
        electron_capture_activation_eV=ec,
        electron_emission_activation_eV=ec + 2.0 * (gap - tr),
        hole_capture_activation_eV=hc,
        hole_emission_activation_eV=hc + 2.0 * tr,
        electron_capture_path=DOUBLE_ELECTRON_CAPTURE,
        hole_capture_path=DOUBLE_HOLE_CAPTURE,
        capture_n_m3_s=1.0e-20,
        capture_p_m3_s=1.5e-20,
        phonon_frequency_Hz=1.0e13,
    )
    spec = MetastableDefectSpec(
        identifier="vse_vo",
        total_density_m3=5.0e22,
        donor_configuration=donor_cfg,
        acceptor_configuration=acc_cfg,
        donor_conversion_state_index=0,  # +2
        acceptor_conversion_state_index=0,  # 0 (differs by +2e)
        conversion_kinetics=conv,
    )
    spec.validate_band_gap(gap)

    # Also verify alternate mixed capture paths:
    conv_alt = MetastableConversionKineticsSpec(
        transition_energy_eV_above_vb=0.70,
        electron_capture_activation_eV=0.40,
        electron_emission_activation_eV=0.40 + gap - 2.0 * 0.70,
        hole_capture_activation_eV=0.50,
        hole_emission_activation_eV=0.50 + 2.0 * 0.70 - gap,
        electron_capture_path=ELECTRON_CAPTURE_PLUS_HOLE_EMISSION,
        hole_capture_path=HOLE_CAPTURE_PLUS_ELECTRON_EMISSION,
        capture_n_m3_s=1.0e-20,
        capture_p_m3_s=1.0e-20,
        phonon_frequency_Hz=1.0e13,
    )
    conv_alt.validate_band_gap(gap)

    # Detailed-balance barrier violation must fail:
    bad_conv = MetastableConversionKineticsSpec(
        transition_energy_eV_above_vb=tr,
        electron_capture_activation_eV=ec,
        electron_emission_activation_eV=ec + 0.1,
        hole_capture_activation_eV=hc,
        hole_emission_activation_eV=hc + 2.0 * tr,
        electron_capture_path=DOUBLE_ELECTRON_CAPTURE,
        hole_capture_path=DOUBLE_HOLE_CAPTURE,
        capture_n_m3_s=1.0e-20,
        capture_p_m3_s=1.5e-20,
        phonon_frequency_Hz=1.0e13,
    )
    with pytest.raises(ValueError, match="detailed balance"):
        bad_conv.validate_band_gap(gap)

    n0 = np.array([2.0e20, 6.0e19])
    p0 = np.array([4.0e19, 3.0e20])
    cl = evaluate_metastable_configuration_closure(
        n0,
        p0,
        spec,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    eps = 1.0e-6
    cl_hn = evaluate_metastable_configuration_closure(
        n0 * (1.0 + eps),
        p0,
        spec,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    cl_ln = evaluate_metastable_configuration_closure(
        n0 * (1.0 - eps),
        p0,
        spec,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    cl_hp = evaluate_metastable_configuration_closure(
        n0,
        p0 * (1.0 + eps),
        spec,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    cl_lp = evaluate_metastable_configuration_closure(
        n0,
        p0 * (1.0 - eps),
        spec,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=temp,
    )
    np.testing.assert_allclose(
        cl.donor_fraction_derivative_n_m3,
        (cl_hn.donor_fraction - cl_ln.donor_fraction) / (2.0 * eps * n0),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cl.donor_fraction_derivative_p_m3,
        (cl_hp.donor_fraction - cl_lp.donor_fraction) / (2.0 * eps * p0),
        rtol=1e-8,
    )

    # Exact exponential advance semigroup, slow/fast limits, and two-charge transfer:
    y0 = np.array([0.25, 0.80])
    np.testing.assert_allclose(configuration_from_logit(configuration_logit(y0)), y0, rtol=1e-15)
    tau = 1.0 / cl.total_conversion_rate_s1
    dt1 = float(0.3 * np.min(tau))
    dt2 = float(0.5 * np.min(tau))
    step1 = advance_metastable_configuration(y0, cl, dt1)
    step2 = advance_metastable_configuration(step1.next_donor_fraction, cl, dt2)
    step12 = advance_metastable_configuration(y0, cl, dt1 + dt2)
    np.testing.assert_allclose(step2.next_donor_fraction, step12.next_donor_fraction, rtol=1e-14)

    slow = advance_metastable_configuration(y0, cl, float(1.0e-8 * np.min(tau)))
    np.testing.assert_allclose(
        slow.delta_donor_fraction,
        (cl.donor_fraction - y0) * cl.total_conversion_rate_s1 * slow.dt_s,
        rtol=1e-7,
    )
    fast = advance_metastable_configuration(y0, cl, float(60.0 * np.max(tau)))
    np.testing.assert_allclose(fast.next_donor_fraction, cl.donor_fraction, rtol=1e-13)

    mask = np.array([False, True, True, False])
    reg_before = FrozenMetastableRegion(
        "abs", "b" * 64, mask, np.full(2, gap), np.full(2, nc), np.full(2, nv), temp, (spec,), y0[None, :]
    )
    reg_after = FrozenMetastableRegion(
        "abs",
        "b" * 64,
        mask,
        np.full(2, gap),
        np.full(2, nc),
        np.full(2, nv),
        temp,
        (spec,),
        step1.next_donor_fraction[None, :],
    )
    n_full = np.r_[1e20, n0, 1e20]
    p_full = np.r_[1e20, p0, 1e20]
    ev_before = evaluate_frozen_metastable_region(reg_before, n_full, p_full)
    ev_after = evaluate_frozen_metastable_region(reg_after, n_full, p_full)
    d_rho_pred = configuration_charge_transfer_C_m3(
        spec.total_density_m3,
        ev_before.donor_mean_charge_e[0],
        ev_before.acceptor_mean_charge_e[0],
        step1.delta_donor_fraction,
    )
    np.testing.assert_allclose(
        d_rho_pred,
        (ev_after.total_charge_density_C_m3 - ev_before.total_charge_density_C_m3)[mask],
        rtol=1e-12,
    )

    f_neut = solve_frozen_metastable_charge_neutrality(
        temperature_K=temp,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        donor_density_m3=0.0,
        acceptor_density_m3=2.0e22,
        species=(spec,),
        donor_fractions=(0.4,),
    )
    assert f_neut.normalized_charge_residual <= 1.0e-12


def test_fixed_occupancy_two_sided_interface_qss_identity_and_ift_jacobians() -> None:
    """Verify QSS embedding f == f_qss and all 7 IFT derivatives vs central FD."""
    temp = 300.0
    vt = thermal_voltage(temp)
    geom = TraceElectrostatics(
        eps_r_left=4.0,
        eps_r_right=24.0,
        left_distance_m=5.0e-9,
        right_distance_m=8.0e-9,
        potential_jump_right_minus_left_V=0.0,
        fixed_sheet_charge_C_m2=0.0,
    )
    phys = TwoSidedInterfacePhysics(
        thermal_voltage_V=vt,
        temperature_K=temp,
        D_n_left_m2_s=1.0e-6,
        D_n_right_m2_s=5.0e-5,
        D_p_left_m2_s=2.0e-6,
        D_p_right_m2_s=4.0e-5,
        N_C_left_m3=2.0e25,
        N_C_right_m3=1.0e25,
        N_V_left_m3=3.0e25,
        N_V_right_m3=1.0e25,
        richardson_n_A_m2_K2=6.0e5,
        richardson_p_A_m2_K2=6.0e5,
        conduction_band_step_eV=-0.30,
        hole_transport_step_eV=0.10,
        surface_recombination_velocity_n_m_s=15.0,
        surface_recombination_velocity_p_m_s=10.0,
        n1_left_m3=1.0e12,
        p1_left_m3=2.0e14,
        n1_right_m3=5.0e14,
        p1_right_m3=8.0e14,
    )
    charge = EquilibriumReferencedSheetCharge(
        equilibrium_occupancy=0.35,
        trap_density_m2=3.0e15,
    )

    def make_bulk(coords: np.ndarray) -> TwoSidedBulkState:
        phi_l, phi_r, ln_nl, ln_pl, ln_nr, ln_pr = (float(v) for v in coords)
        return TwoSidedBulkState(
            phi_left_V=phi_l,
            phi_right_V=phi_r,
            n_left_m3=math.exp(ln_nl),
            p_left_m3=math.exp(ln_pl),
            n_right_m3=math.exp(ln_nr),
            p_right_m3=math.exp(ln_pr),
        )

    coords0 = np.array(
        [-0.12, -0.05, math.log(4.0e16), math.log(2.0e21), math.log(3.0e19), math.log(1.5e19)]
    )
    bulk0 = make_bulk(coords0)
    qss = solve_charged_two_sided_interface_qss(geom, phys, bulk0, charge)
    at_qss = solve_fixed_occupancy_two_sided_interface(
        geom, phys, bulk0, charge, occupancy=qss.occupancy
    )
    np.testing.assert_allclose(at_qss.bulk_flux_m2_s, qss.bulk_flux_m2_s, rtol=1e-11)
    assert abs(at_qss.trap_storage_rate_m2_s) <= 1.0e-7 * max(
        float(np.max(np.abs(at_qss.bulk_flux_m2_s))), 1.0
    )
    assert at_qss.incremental_sheet_charge_C_m2 == pytest.approx(
        -Q * charge.trap_density_m2 * (qss.occupancy - charge.equilibrium_occupancy), rel=1e-14
    )

    f0 = 0.62
    res = solve_fixed_occupancy_two_sided_interface(geom, phys, bulk0, charge, occupancy=f0)
    assert res.converged
    step = 1.0e-5
    flux_atol = 1.0e-12 * float(np.max(np.abs(res.bulk_flux_m2_s))) / step
    sr_atol = 1.0e-12 * abs(res.trap_storage_rate_m2_s) / step
    for col in range(6):
        c_hi = coords0.copy()
        c_lo = coords0.copy()
        c_hi[col] += step
        c_lo[col] -= step
        r_hi = solve_fixed_occupancy_two_sided_interface(
            geom, phys, make_bulk(c_hi), charge, occupancy=f0
        )
        r_lo = solve_fixed_occupancy_two_sided_interface(
            geom, phys, make_bulk(c_lo), charge, occupancy=f0
        )
        np.testing.assert_allclose(
            res.bulk_flux_jacobian_bulk[:, col],
            (r_hi.bulk_flux_m2_s - r_lo.bulk_flux_m2_s) / (2.0 * step),
            rtol=1e-6,
            atol=flux_atol,
        )
        assert res.trap_storage_rate_jacobian_bulk[col] == pytest.approx(
            (r_hi.trap_storage_rate_m2_s - r_lo.trap_storage_rate_m2_s) / (2.0 * step),
            rel=1e-6,
            abs=sr_atol,
        )

    r_f_hi = solve_fixed_occupancy_two_sided_interface(
        geom, phys, bulk0, charge, occupancy=f0 + step
    )
    r_f_lo = solve_fixed_occupancy_two_sided_interface(
        geom, phys, bulk0, charge, occupancy=f0 - step
    )
    np.testing.assert_allclose(
        res.bulk_flux_derivative_occupancy,
        (r_f_hi.bulk_flux_m2_s - r_f_lo.bulk_flux_m2_s) / (2.0 * step),
        rtol=1e-8,
        atol=1e-5,
    )
    assert res.trap_storage_rate_derivative_occupancy == pytest.approx(
        (r_f_hi.trap_storage_rate_m2_s - r_f_lo.trap_storage_rate_m2_s) / (2.0 * step),
        rel=1e-8,
    )
