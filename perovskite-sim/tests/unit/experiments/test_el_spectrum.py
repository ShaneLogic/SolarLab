"""Unit tests for run_el_spectrum (Phase 4.3 reciprocity EL)."""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments.el_spectrum import (
    _absorber_mask,
    _blackbody_photon_flux,
    run_el_spectrum,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.el import ELResult


# ----------------------------- fixtures -------------------------------------

TMM_CONFIG = "configs/nip_MAPbI3_tmm.yaml"
BEER_LAMBERT_CONFIG = "configs/nip_MAPbI3.yaml"


@pytest.fixture(scope="module")
def tmm_stack():
    return load_device_from_yaml(TMM_CONFIG)


@pytest.fixture(scope="module")
def bl_stack():
    return load_device_from_yaml(BEER_LAMBERT_CONFIG)


@pytest.fixture(scope="module")
def el_result(tmm_stack):
    """Shared EL result on a coarse grid — keeps the suite fast."""
    return run_el_spectrum(
        tmm_stack,
        V_inj=1.0,
        wavelengths_nm=np.linspace(400.0, 900.0, 6),
        N_grid=30,
        n_points_dark=10,
    )


# ----------------------------- input validation -----------------------------


def test_non_tmm_stack_is_rejected(bl_stack):
    """Beer-Lambert stacks have no wavelength-resolved optics → ValueError."""
    with pytest.raises(ValueError, match="optical_material"):
        run_el_spectrum(bl_stack, V_inj=1.0, N_grid=20, n_points_dark=5)


def test_V_inj_must_be_positive(tmm_stack):
    with pytest.raises(ValueError, match="V_inj"):
        run_el_spectrum(tmm_stack, V_inj=0.0, N_grid=20, n_points_dark=5)
    with pytest.raises(ValueError, match="V_inj"):
        run_el_spectrum(tmm_stack, V_inj=-0.5, N_grid=20, n_points_dark=5)


def test_N_grid_lower_bound(tmm_stack):
    with pytest.raises(ValueError, match="N_grid"):
        run_el_spectrum(tmm_stack, V_inj=1.0, N_grid=2, n_points_dark=5)


def test_wavelengths_need_two_points(tmm_stack):
    with pytest.raises(ValueError, match="at least 2 points"):
        run_el_spectrum(
            tmm_stack, V_inj=1.0,
            wavelengths_nm=np.array([500.0]),
            N_grid=20, n_points_dark=5,
        )


def test_wavelengths_must_be_positive(tmm_stack):
    with pytest.raises(ValueError, match="positive"):
        run_el_spectrum(
            tmm_stack, V_inj=1.0,
            wavelengths_nm=np.array([-1.0, 500.0, 800.0]),
            N_grid=20, n_points_dark=5,
        )


# ----------------------------- result shape / sanity ------------------------


def test_result_is_correct_type(el_result):
    assert isinstance(el_result, ELResult)


def test_wavelength_axis_is_sorted(el_result):
    assert np.all(np.diff(el_result.wavelengths_nm) > 0)


def test_array_shapes_match(el_result):
    n_wl = len(el_result.wavelengths_nm)
    assert el_result.EL_spectrum.shape == (n_wl,)
    assert el_result.absorber_absorptance.shape == (n_wl,)


def test_all_fields_finite(el_result):
    assert np.all(np.isfinite(el_result.EL_spectrum))
    assert np.all(np.isfinite(el_result.absorber_absorptance))
    assert np.isfinite(el_result.J_em_rad)
    assert np.isfinite(el_result.J_inj)
    assert np.isfinite(el_result.EQE_EL)
    assert np.isfinite(el_result.delta_V_nr_mV)


def test_absorptance_bounded_to_unit_interval(el_result):
    """A_abs(λ) is a fraction of incident photons — must be in [0, 1]."""
    assert np.all(el_result.absorber_absorptance >= 0.0)
    assert np.all(el_result.absorber_absorptance <= 1.0)


def test_absorptance_peaks_inside_MAPbI3_band(el_result):
    """MAPbI3 absorbs strongly between 500-780 nm, then collapses past 820 nm."""
    lam = el_result.wavelengths_nm
    A = el_result.absorber_absorptance
    mid = A[(lam >= 500) & (lam <= 780)]
    tail = A[lam >= 850]
    assert mid.size >= 1
    # Strong absorber response in the visible band.
    assert mid.max() > 0.5
    # Well past the bandgap — absorptance should be small. Use a loose
    # upper bound to accommodate any weak out-of-gap transport-layer
    # absorption that leaks into the absorber mask on coarse grids.
    if tail.size:
        assert tail.max() < 0.2


def test_EL_spectrum_nonnegative(el_result):
    """Photon flux must be ≥ 0 at every wavelength."""
    assert np.all(el_result.EL_spectrum >= 0.0)


def test_J_em_rad_positive(el_result):
    """q · ∫ Φ_EL dλ is manifestly positive for V_inj > 0."""
    assert el_result.J_em_rad > 0.0


def test_J_inj_is_injection(el_result):
    """Dark forward bias injects carriers — J is negative under our convention."""
    assert el_result.J_inj < 0.0


def test_EQE_EL_inside_physical_band(el_result):
    """0 < EQE_EL ≤ 1 for a well-behaved device."""
    assert 0.0 < el_result.EQE_EL <= 1.0


def test_delta_V_nr_nonnegative(el_result):
    """EQE_EL ≤ 1 ⇒ -kT/q · ln(EQE_EL) ≥ 0."""
    assert el_result.delta_V_nr_mV >= 0.0


def test_delta_V_nr_in_expected_range_for_MAPbI3(el_result):
    """MAPbI3 typically sits in 100-400 mV non-radiative loss territory."""
    assert 50.0 < el_result.delta_V_nr_mV < 500.0


def test_temperature_propagates_into_result(tmm_stack):
    """ELResult.T should match stack.T."""
    res = run_el_spectrum(
        tmm_stack, V_inj=1.0,
        wavelengths_nm=np.linspace(500.0, 800.0, 4),
        N_grid=20, n_points_dark=5,
    )
    assert res.T == pytest.approx(tmm_stack.T, rel=1e-12)


# ----------------------------- scaling behaviour ----------------------------


def test_increasing_V_inj_raises_J_em_rad(tmm_stack):
    """J_em_rad ∝ exp(qV/kT) — a 0.2 V bump at 300 K is a ~2000x factor."""
    wl = np.linspace(500.0, 800.0, 4)
    kwargs = dict(wavelengths_nm=wl, N_grid=20, n_points_dark=5)
    low = run_el_spectrum(tmm_stack, V_inj=0.8, **kwargs)
    high = run_el_spectrum(tmm_stack, V_inj=1.0, **kwargs)
    # Boltzmann factor ratio: exp((0.2)/(kT/q)) at 300 K ~ exp(7.73) ~ 2273.
    # Allow a wide tolerance because A_abs is identical, so this just
    # verifies the Boltzmann factor was applied and the direction is right.
    ratio = high.J_em_rad / low.J_em_rad
    assert ratio > 500.0


def test_input_stack_not_mutated(tmm_stack):
    """Frozen dataclass contract — stack fields survive the run unchanged."""
    T_before = tmm_stack.T
    V_bi_before = tmm_stack.V_bi
    run_el_spectrum(
        tmm_stack, V_inj=1.0,
        wavelengths_nm=np.linspace(500.0, 800.0, 4),
        N_grid=20, n_points_dark=5,
    )
    assert tmm_stack.T == T_before
    assert tmm_stack.V_bi == V_bi_before


def test_progress_callback_invoked(tmm_stack):
    """Callback should fire a strictly-increasing current count."""
    calls: list[tuple[str, int, int, str]] = []

    def cb(stage, cur, tot, msg):
        calls.append((stage, cur, tot, msg))

    run_el_spectrum(
        tmm_stack, V_inj=1.0,
        wavelengths_nm=np.linspace(500.0, 800.0, 4),
        N_grid=20, n_points_dark=5,
        progress=cb,
    )
    assert len(calls) >= 3
    assert all(stage == "el" for stage, _, _, _ in calls)
    counts = [cur for _, cur, _, _ in calls]
    assert counts == sorted(counts)


# ----------------------------- helper functions -----------------------------


def test_blackbody_flux_positive_finite():
    """φ_bb(λ, 300K) should be strictly positive and finite across 400-1000 nm."""
    lam = np.linspace(400e-9, 1000e-9, 50)
    phi = _blackbody_photon_flux(lam, 300.0)
    assert np.all(np.isfinite(phi))
    assert np.all(phi > 0.0)


def test_blackbody_flux_increases_with_temperature():
    """At fixed λ, φ_bb grows monotonically in T (Wien-side behaviour)."""
    lam = np.array([800e-9])
    phi_300 = _blackbody_photon_flux(lam, 300.0)[0]
    phi_400 = _blackbody_photon_flux(lam, 400.0)[0]
    phi_500 = _blackbody_photon_flux(lam, 500.0)[0]
    assert phi_300 < phi_400 < phi_500


def test_blackbody_handles_extreme_wavelengths():
    """No overflow at 300 nm, no NaN at 2000 nm."""
    lam = np.array([300e-9, 2000e-9])
    phi = _blackbody_photon_flux(lam, 300.0)
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0.0)


def test_absorber_mask_matches_absorber_layer(tmm_stack):
    """A TMM preset has exactly one role=absorber layer; the mask should
    cover a contiguous block of nodes that sum to positive area."""
    # Build a coarse grid matching what run_el_spectrum would use internally.
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.models.device import electrical_layers
    elec = electrical_layers(tmm_stack)
    n_per = max(30 // len(elec), 2)
    x = multilayer_grid([Layer(l.thickness, n_per) for l in elec])
    mask = _absorber_mask(x, tmm_stack)
    assert mask.any()
    # Mask should represent a connected region (no islands).
    idx = np.where(mask)[0]
    assert np.all(np.diff(idx) == 1), "absorber mask must be contiguous"
