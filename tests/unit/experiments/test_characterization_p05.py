"""T1 analytic and contract unit tests for P05-02b and P05-03a characterization."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M
from solarlab.experiments.dark_jv import (
    DarkJVOptions,
    extract_dark_diode_parameters,
    select_dark_diode_window,
)
from solarlab.experiments.el_spectrum import (
    ELOptions,
    blackbody_photon_flux_m2_s_m,
    run_el_spectrum,
)
from solarlab.experiments.eqe import EQEOptions, compute_eqe
from solarlab.experiments.mott_schottky import (
    CFOptions,
    MottSchottkyOptions,
    _capacitance_from_admittance,
    fit_mott_schottky_curve,
    select_mott_schottky_window,
)
from solarlab.experiments.suns_voc import (
    SunsVocOptions,
    compute_pseudo_ff,
    compute_suns_voc_ideality,
)
from solarlab.experiments.voc_t import (
    VocTOptions,
    fit_voc_t_extrapolation,
    rebind_prepared_temperature,
)
from solarlab.experiments.waveform_jv import (
    JVWaveform,
    WaveformJVOptions,
    WaveformSegment,
    build_piecewise_waveform,
    build_triangular_jv_waveform,
)

K_B = 1.380649e-23


ROOT = Path(__file__).resolve().parents[3]


def load_device() -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    mod = importlib.import_module("examples.research_1d.steady")
    return mod.load_device()


@pytest.fixture(scope="module", autouse=True)
def blas1() -> object:
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("n_true", [1.0, 1.5, 2.0])
def test_suns_voc_analytic_ideality_and_missing_point_exclusion(n_true: float) -> None:
    temperature_K = 300.0
    v_thermal = K_B * temperature_K / ELEMENTARY_CHARGE_C
    suns = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    voc_1sun = 1.08
    voc = voc_1sun + n_true * v_thermal * np.log(suns)

    local_n, n_fit, r_sq = compute_suns_voc_ideality(
        suns, voc, temperature_K=temperature_K
    )
    np.testing.assert_allclose(local_n, np.full_like(suns, n_true), rtol=1e-12)
    assert n_fit == pytest.approx(n_true, rel=1e-12)
    assert r_sq == pytest.approx(1.0, abs=1e-14)

    # Masking an unbracketed point preserves NaN at that index without corrupting fit.
    voc_masked = voc.copy()
    voc_masked[2] = np.nan
    valid_mask = np.ones(suns.shape, dtype=bool)
    valid_mask[2] = False
    local_masked, n_fit_masked, r_sq_masked = compute_suns_voc_ideality(
        suns, voc_masked, temperature_K=temperature_K, valid_mask=valid_mask
    )
    assert math.isnan(local_masked[2])
    np.testing.assert_allclose(
        local_masked[valid_mask], np.full(suns.size - 1, n_true), rtol=1e-11
    )
    assert n_fit_masked == pytest.approx(n_true, rel=1e-12)
    assert r_sq_masked == pytest.approx(1.0, abs=1e-14)


def test_suns_voc_pseudo_ff_matches_analytic_ideal_diode_and_refuses_empty_quadrant() -> None:
    temperature_K = 300.0
    v_thermal = K_B * temperature_K / ELEMENTARY_CHARGE_C
    suns = np.geomspace(1e-4, 10.0, 400)
    voc_ref = 1.05
    jsc_ref = 220.0
    voc = voc_ref + v_thermal * np.log(suns)
    jsc = jsc_ref * suns
    j_pseudo = jsc_ref - jsc

    pff, reason = compute_pseudo_ff(
        voc, j_pseudo, V_oc_ref_V=voc_ref, J_sc_ref_A_m2=jsc_ref
    )
    assert reason is None
    voc_norm = voc_ref / v_thermal
    ff_green = (voc_norm - math.log(voc_norm + 0.72)) / (voc_norm + 1.0)
    assert pff == pytest.approx(ff_green, abs=5e-4)

    # Without valid sub-1-sun points in the power quadrant, never fabricate 0.25.
    pff_bad, reason_bad = compute_pseudo_ff(
        np.array([voc_ref, voc_ref + 0.04]),
        np.array([0.0, -jsc_ref]),
        V_oc_ref_V=voc_ref,
        J_sc_ref_A_m2=jsc_ref,
    )
    assert math.isnan(pff_bad)
    assert reason_bad == "no_valid_power_quadrant_points"


def test_voc_t_extrapolation_recovers_exact_activation_energy_and_excludes_unbracketed() -> None:
    temps = np.array([240.0, 270.0, 300.0, 330.0, 360.0], dtype=float)
    e_a_true = 1.58
    beta = 1.65e-3
    voc_exact = e_a_true - beta * temps
    all_ok = np.ones(temps.shape, dtype=bool)

    slope, intercept_0k, e_a, r_sq, reason = fit_voc_t_extrapolation(
        temps, voc_exact, converged_mask=all_ok, voc_bracketed_mask=all_ok
    )
    assert reason is None
    assert slope == pytest.approx(-beta, rel=1e-12)
    assert intercept_0k == pytest.approx(e_a_true, rel=1e-12)
    assert e_a == pytest.approx(e_a_true, rel=1e-12)
    assert r_sq == pytest.approx(1.0, abs=1e-14)

    # Critical regression: an unbracketed high-T point is NaN + masked False, never 0.0.
    voc_partial = voc_exact.copy()
    voc_partial[-1] = np.nan
    brack_partial = np.array([True, True, True, True, False], dtype=bool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slope_p, int_p, ea_p, rsq_p, reason_p = fit_voc_t_extrapolation(
            temps,
            voc_partial,
            converged_mask=all_ok,
            voc_bracketed_mask=brack_partial,
        )
    assert len(caught) == 1
    assert "360.0K" in str(caught[0].message)
    assert reason_p is None
    assert slope_p == pytest.approx(-beta, rel=1e-12)
    assert int_p == pytest.approx(e_a_true, rel=1e-12)
    assert ea_p == pytest.approx(e_a_true, rel=1e-12)
    assert rsq_p == pytest.approx(1.0, abs=1e-14)

    # Fewer than 2 valid bracketed points refuses by default and records NaN when partial.
    one_ok = np.array([True, False, False, False, False], dtype=bool)
    with pytest.raises(ValueError, match="requires >= 2 valid bracketed temperatures"):
        fit_voc_t_extrapolation(
            temps, voc_partial, converged_mask=all_ok, voc_bracketed_mask=one_ok
        )
    s_nan, i_nan, ea_nan, r_nan, fail_reason = fit_voc_t_extrapolation(
        temps,
        voc_partial,
        converged_mask=all_ok,
        voc_bracketed_mask=one_ok,
        allow_partial=True,
    )
    assert all(math.isnan(v) for v in (s_nan, i_nan, ea_nan, r_nan))
    assert fail_reason == "insufficient_valid_bracketed_points:1_of_5"


def test_rebind_prepared_temperature_updates_temperature_and_identity() -> None:
    base = load_device()
    rebound = rebind_prepared_temperature(base, 275.0)
    assert rebound.to_mapping()["settings"]["T"] == 275.0
    assert rebound.content_sha256 != base.content_sha256


def test_dark_jv_analytic_shockley_diode_with_series_resistance() -> None:
    temperature_K = 300.0
    v_thermal = K_B * temperature_K / ELEMENTARY_CHARGE_C
    n_true = 1.45
    j0_true = 2.5e-9
    rs_true = 2.0e-4  # Ohm * m^2

    # Construct parametric Shockley curve with R_s: V_term = V_j + J_conj * R_s.
    v_j = np.linspace(0.0, 1.05, 140)
    j_conj = j0_true * (np.exp(v_j / (n_true * v_thermal)) - 1.0)
    v_term = v_j + j_conj * rs_true
    v_uniform = np.linspace(0.0, 1.15, 80)
    j_conj_uniform = np.interp(v_uniform, v_term, j_conj)
    j_pv_uniform = -j_conj_uniform

    (
        local_n,
        fit_mask,
        n_fit,
        j0_fit,
        v_lo,
        v_hi,
        r_sq,
        rs_est,
        reason,
    ) = extract_dark_diode_parameters(
        v_uniform, j_pv_uniform, temperature_K=temperature_K
    )
    assert reason is None
    assert np.count_nonzero(fit_mask) >= 8
    assert 0.1 < v_lo < v_hi < 0.95
    assert n_fit == pytest.approx(n_true, rel=1.5e-2)
    assert math.log10(j0_fit) == pytest.approx(math.log10(j0_true), abs=0.08)
    assert r_sq > 0.9999
    assert rs_est == pytest.approx(rs_true, rel=0.15)
    assert np.nanmedian(local_n[fit_mask]) == pytest.approx(n_true, rel=1.5e-2)

    # Non-injecting curve returns explicit failure reason and NaNs (never 0.0).
    empty_mask, empty_reason = select_dark_diode_window(
        np.linspace(0.0, 0.5, 10), np.zeros(10)
    )
    assert not np.any(empty_mask)
    assert empty_reason is not None and "insufficient_forward_injection" in empty_reason


def test_piecewise_and_triangular_waveform_exact_corners_and_continuity() -> None:
    wf_dict = {
        "start_voltage_V": -0.1,
        "dark_seed_s": 0.0,
        "dark_prep_s": 0.05,
        "branch_dwell_s": 0.02,
        "turnaround_s": 0.03,
        "turnaround_dark": True,
    }
    spec = JVWaveform.from_dict(wf_dict)
    assert spec.to_dict() == wf_dict

    tri = build_triangular_jv_waveform(
        spec, V_max=1.1, v_rate=0.4, n_points=13, generation_scale=1.0
    )
    # Every corner index lands on an exact segment transition in protocol.times_s.
    for idx, t_corner in zip(tri.corner_indices, tri.corner_times_s, strict=True):
        assert tri.protocol.times_s[idx] == pytest.approx(t_corner, abs=1e-15)

    roles = [seg.role for seg in tri.segments]
    assert roles == [
        "preconditioning",
        "hold",
        "forward_ramp",
        "hold",
        "hold",
        "reverse_ramp",
    ]
    fwd_slice = tri.segment_step_slices[2]
    rev_slice = tri.segment_step_slices[5]
    np.testing.assert_allclose(tri.step_dV_dt_V_s[fwd_slice], 0.4, rtol=1e-12)
    np.testing.assert_allclose(tri.step_dV_dt_V_s[rev_slice], -0.4, rtol=1e-12)

    # Discontinuous voltage jump between segments is refused explicitly.
    with pytest.raises(ValueError, match="discontinuous waveform voltage"):
        build_piecewise_waveform(
            [
                WaveformSegment("s1", 0.1, 0.0, 0.5, n_steps=4),
                WaveformSegment("s2", 0.1, 0.6, 1.0, n_steps=4),
            ]
        )


def test_mott_schottky_analytic_abrupt_junction_and_inductive_guards() -> None:
    temperature_K = 300.0
    v_thermal = K_B * temperature_K / ELEMENTARY_CHARGE_C
    eps_r = 24.0
    v_bi_true = 1.05
    n_d_true = 3.5e22  # m^-3

    v_bias = np.linspace(-0.4, 0.75, 24)
    # Linear depletion region over [-0.4, 0.45] V, followed by exponential
    # chemical-capacitance surge above 0.5 V so 1/C^2 bends sharply downward.
    w_sq_factor = 2.0 / (
        ELEMENTARY_CHARGE_C * eps_r * VACUUM_PERMITTIVITY_F_M * n_d_true
    )
    one_over_c2_ideal = w_sq_factor * (v_bi_true - v_bias - 2.0 * v_thermal)
    c_depl = 1.0 / np.sqrt(one_over_c2_ideal)
    c_chem = np.where(
        v_bias > 0.48, 4e-3 * (np.exp((v_bias - 0.48) / (2.0 * v_thermal)) - 1.0), 0.0
    )
    c_total = c_depl + c_chem

    (
        one_over_c2,
        fit_mask,
        app_vbi,
        app_n,
        slope,
        _intercept,
        v_lo,
        v_hi,
        r_sq,
        reason,
    ) = fit_mott_schottky_curve(
        v_bias, c_total, eps_r=eps_r, temperature_K=temperature_K
    )
    assert reason is None
    assert slope < 0.0
    assert v_hi <= 0.50
    assert v_lo == pytest.approx(-0.4, abs=1e-12)
    assert np.count_nonzero(fit_mask) >= 15
    assert app_vbi == pytest.approx(v_bi_true, abs=1e-4)
    assert app_n == pytest.approx(n_d_true, rel=1e-4)
    assert r_sq > 0.999999

    # Flat and nearly-flat geometric capacitors are rejected rather than fabricating V_bi.
    c_const = np.full(12, 2.5e-4, dtype=float)
    _, _, vbi_const, n_const, _, _, _, _, _, const_reason = fit_mott_schottky_curve(
        np.linspace(-0.3, 0.3, 12), c_const, eps_r=eps_r, temperature_K=temperature_K
    )
    assert math.isnan(vbi_const) and math.isnan(n_const)
    assert const_reason == "no_contiguous_linear_depletion_window"

    v_near = np.linspace(-0.3, 0.3, 12)
    c_near_flat = 2.5e-4 / np.sqrt(1.0 - 0.02 * v_near)
    _, _, vbi_flat, n_flat, _, _, _, _, _, flat_reason = fit_mott_schottky_curve(
        v_near, c_near_flat, eps_r=eps_r, temperature_K=temperature_K
    )
    assert math.isnan(vbi_flat) and math.isnan(n_flat)
    assert flat_reason == "flat_geometric_capacitance_insufficient_variation"
    assert (
        select_mott_schottky_window(np.linspace(0.0, 0.2, 3), np.array([1.0, 2.0, 3.0]))
        is None
    )

    # Inductive and 0 Hz susceptance points return NaN (never abs(Im(Y))) or raise if strict.
    c_ind, g_ind, loss_ind, ok_ind, r_ind = _capacitance_from_admittance(
        1000.0, 0.5 - 2.0j, strict_capacitive=False
    )
    assert math.isnan(c_ind) and math.isnan(loss_ind)
    assert g_ind == 0.5 and not ok_ind and r_ind == "inductive_susceptance"
    with pytest.raises(RuntimeError, match="inductive susceptance"):
        _capacitance_from_admittance(1000.0, 0.5 - 2.0j, strict_capacitive=True)

    c_dc, _, _, ok_dc, r_dc = _capacitance_from_admittance(
        0.0, 0.5 + 0.0j, strict_capacitive=False
    )
    assert math.isnan(c_dc) and not ok_dc and r_dc == "zero_frequency"


def test_blackbody_reciprocity_and_optical_refusal_guards() -> None:
    wl_m = np.linspace(400e-9, 900e-9, 11)
    bb_300 = blackbody_photon_flux_m2_s_m(wl_m, 300.0)
    bb_350 = blackbody_photon_flux_m2_s_m(wl_m, 350.0)
    assert np.all(np.isfinite(bb_300)) and np.all(bb_300 > 0.0)
    assert np.all(bb_350 > bb_300)

    # Scalar Beer-Lambert PreparedDevice (no nk optical_material table) is refused
    # before solver work by both compute_eqe and run_el_spectrum.
    scalar_alpha_device = load_device()
    with pytest.raises(ValueError, match="tabulated wavelength-resolved optical_material"):
        compute_eqe(scalar_alpha_device)
    with pytest.raises(ValueError, match="tabulated wavelength-resolved optical_material"):
        run_el_spectrum(scalar_alpha_device, V_inj_V=1.0)
    with pytest.raises(ValueError, match="Phi_incident must be finite and positive"):
        compute_eqe(scalar_alpha_device, Phi_incident=0.0)
    with pytest.raises(ValueError, match="V_inj_V must be finite and positive"):
        run_el_spectrum(scalar_alpha_device, V_inj_V=0.0)
    with pytest.raises(ValueError, match=">= 2 distinct wavelengths"):
        run_el_spectrum(scalar_alpha_device, V_inj_V=1.0, wavelengths_nm=[550.0])

    # Option validation guards.
    with pytest.raises(ValueError, match="voc_voltage_tol_V"):
        SunsVocOptions(voc_voltage_tol_V=0.0)
    with pytest.raises(ValueError, match="intervals_per_layer"):
        VocTOptions(intervals_per_layer=1)
    with pytest.raises(ValueError, match="min_diode_window_points"):
        DarkJVOptions(min_diode_window_points=2)
    with pytest.raises(ValueError, match="Backward Euler"):
        WaveformJVOptions(method="bdf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ion_mode"):
        CFOptions(ion_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_window_points"):
        MottSchottkyOptions(min_window_points=2)
    with pytest.raises(ValueError, match="perturbation_mode"):
        EQEOptions(perturbation_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absorptance_mode"):
        ELOptions(absorptance_mode="invalid")  # type: ignore[arg-type]
