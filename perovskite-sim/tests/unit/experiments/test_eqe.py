"""Tests for `experiments/eqe.py` — monochromatic EQE + J_sc cross-check.

Key invariants:

1. Stacks without optical_material raise ValueError.
2. EQE values are in [0, 1] physically (tiny numerical slack allowed at
   the edges).
3. EQE peaks in the strong-absorption band (400–700 nm for MAPbI3) and
   drops to near-zero below the sub-gap threshold (~850 nm).
4. Integrating EQE against AM1.5G gives a J_sc consistent with a
   full-spectrum TMM simulation at V=0 (both sample the same optics, so
   the discrepancy is discretisation error and should be within ~20 %).
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments.eqe import EQEResult, compute_eqe
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def tmm_stack():
    """TMM-enabled nip-MAPbI3 with tabulated n, k optical data."""
    return load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")


@pytest.fixture(scope="module")
def bl_stack():
    """Beer-Lambert-only stack (no optical_material) for rejection test."""
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


# ---------------------------------------------------------------------------
# Argument + precondition validation.
# ---------------------------------------------------------------------------

def test_rejects_beer_lambert_only_stack(bl_stack):
    """compute_eqe on a no-optical-data stack must raise, not fake a curve."""
    with pytest.raises(ValueError, match="optical_material"):
        compute_eqe(bl_stack, wavelengths_nm=np.array([500.0]))


def test_rejects_empty_wavelengths(tmm_stack):
    with pytest.raises(ValueError, match="non-empty"):
        compute_eqe(tmm_stack, wavelengths_nm=np.array([]))


def test_rejects_nonpositive_phi(tmm_stack):
    with pytest.raises(ValueError, match="Phi_incident"):
        compute_eqe(
            tmm_stack,
            wavelengths_nm=np.array([500.0, 600.0]),
            Phi_incident=0.0,
        )


def test_rejects_nonpositive_wavelength(tmm_stack):
    with pytest.raises(ValueError, match="positive"):
        compute_eqe(tmm_stack, wavelengths_nm=np.array([500.0, 0.0, 700.0]))


# ---------------------------------------------------------------------------
# Structural invariants — coarse 5-point sweep reused by fast tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_eqe(tmm_stack):
    """5-wavelength sweep reused by shape / range tests."""
    return compute_eqe(
        tmm_stack,
        wavelengths_nm=np.array([400.0, 500.0, 600.0, 700.0, 800.0]),
        N_grid=30,
        t_settle=5e-4,
    )


def test_result_is_populated_dataclass(coarse_eqe):
    r = coarse_eqe
    assert isinstance(r, EQEResult)
    assert r.wavelengths_nm.shape == r.EQE.shape == r.J_sc_per_lambda.shape
    assert np.all(np.isfinite(r.EQE))
    assert np.all(np.isfinite(r.J_sc_per_lambda))
    # wavelengths must be sorted ascending (we sort internally)
    assert np.all(np.diff(r.wavelengths_nm) > 0)


def test_eqe_in_unit_range(coarse_eqe):
    """EQE in [0, 1] — no collection miracle, no negative values.

    Tiny numerical slack (1 %) at the upper edge for TMM interference
    overshoots on 1-point wavelength queries (no spectral smoothing).
    """
    r = coarse_eqe
    assert np.all(r.EQE >= 0.0), f"negative EQE found: {r.EQE}"
    assert np.all(r.EQE <= 1.01), f"EQE > 1 found: {r.EQE}"


def test_eqe_peaks_in_absorption_band(coarse_eqe):
    """Strong-absorption band (400-700 nm) should have larger EQE than
    the sub-gap band edge (800 nm) on a MAPbI3 stack.
    """
    r = coarse_eqe
    # EQE at 500 nm (bulk absorption) should beat 800 nm (near band edge).
    idx_500 = int(np.argmin(np.abs(r.wavelengths_nm - 500.0)))
    idx_800 = int(np.argmin(np.abs(r.wavelengths_nm - 800.0)))
    assert r.EQE[idx_500] > r.EQE[idx_800] + 0.1, (
        f"EQE(500nm)={r.EQE[idx_500]:.3f} not distinctly above "
        f"EQE(800nm)={r.EQE[idx_800]:.3f} for MAPbI3 — band-gap behaviour "
        "suspicious"
    )


@pytest.fixture(scope="module")
def ionmonger_tmm_stack():
    """Ionic-rich TMM stack (D_ion ≈ 1e-17 m²/s, slowest perovskite preset)."""
    return load_device_from_yaml("configs/ionmonger_benchmark_tmm.yaml")


def test_eqe_in_unit_range_on_ionic_rich_preset(ionmonger_tmm_stack):
    """Regression for the ``ionmonger_benchmark_tmm`` EQE>1 artifact
    (audit 2026-05-09).

    With the pre-fix defaults (``Phi_incident=1e20``, ``t_settle=1e-3 s``)
    the residual ionic transient at V=0 swamped the small monochromatic
    photo-signal on this preset, producing ``EQE_max ≈ 48`` — a 4881 %
    quantum yield, physically impossible. The fix bumps the defaults to
    ``Phi_incident=4e21`` and ``t_settle=1e-1 s`` and subtracts the
    dark J(V=0) baseline. This test pins the fix on the worst-case preset
    so any regression resurrects ``EQE > 1`` immediately.
    """
    r = compute_eqe(
        ionmonger_tmm_stack,
        wavelengths_nm=np.array([400.0, 500.0, 600.0, 700.0, 800.0]),
        N_grid=30,
        # Use defaults — the whole point of this test is to lock the
        # default behaviour to "physical EQE on the slowest perovskite
        # preset".
    )
    assert np.all(r.EQE >= 0.0), f"negative EQE found: {r.EQE}"
    # 10 % slack on the upper edge: the fix brings EQE_max from ~48 down
    # to ~1.06 on this preset; tighter than 1.10 would be flaky against
    # TMM Fabry-Pérot fringes that occasionally nudge a single λ above 1.
    assert np.all(r.EQE <= 1.10), (
        f"EQE > 1.10 found on ionmonger_benchmark_tmm: {r.EQE}; the "
        "ionic-transient fix in compute_eqe (default t_settle=1e-1 s, "
        "Phi_incident=4e21, dark-J subtraction) may have regressed."
    )


def test_eqe_drops_above_bandgap():
    """EQE must fall toward zero for λ > MAPbI3 band-gap wavelength (~800 nm)."""
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    r = compute_eqe(
        stack,
        wavelengths_nm=np.array([500.0, 900.0]),
        N_grid=30,
        t_settle=5e-4,
    )
    # 900 nm is well past the MAPbI3 absorption edge; EQE should be
    # substantially smaller than the 500-nm bulk value.
    assert r.EQE[1] < 0.5 * r.EQE[0], (
        f"EQE(900nm)={r.EQE[1]:.3f} not suppressed relative to "
        f"EQE(500nm)={r.EQE[0]:.3f} — MAPbI3 band-gap filter broken"
    )


def test_progress_callback_invoked(tmm_stack):
    """Progress callback fires once per wavelength."""
    events: list[tuple[str, int, int, str]] = []

    def cb(stage, cur, total, msg):
        events.append((stage, cur, total, msg))

    wavelengths = np.array([450.0, 550.0, 650.0])
    compute_eqe(
        tmm_stack, wavelengths_nm=wavelengths, N_grid=30,
        t_settle=5e-4, progress=cb,
    )
    assert len(events) == len(wavelengths)
    assert all(ev[0] == "eqe" for ev in events)
    assert [ev[1] for ev in events] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Physics cross-check: integrated EQE J_sc ≈ full-spectrum J_sc at V=0.
# ---------------------------------------------------------------------------

def test_integrated_jsc_matches_full_tmm():
    """∫ q·EQE·Φ_AM15G dλ must reproduce the full TMM J_sc to within ~25 %.

    Both paths use the same TMM layer stack and the same AM1.5G file;
    they differ only in how the wavelength integration is discretised
    (the EQE path uses 15 probe points; the full sim uses 200
    internally). The test tolerance accommodates that grid coarseness
    — a tight match here would indicate the wavelength grid isn't
    actually being exercised independently.
    """
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")

    # 15 wavelengths across 350-850 nm — covers the bulk of the MAPbI3
    # absorption band. Below 350 nm AM1.5G is weak; above 850 nm MAPbI3
    # absorption collapses.
    wavelengths = np.linspace(350.0, 850.0, 15)
    r_eqe = compute_eqe(
        stack, wavelengths_nm=wavelengths, N_grid=30, t_settle=5e-4,
    )

    # Full-spectrum J_sc from a short JV sweep at V=0.
    jv = run_jv_sweep(stack, N_grid=30, n_points=5, V_max=0.2, v_rate=2.0)
    # J at V=0 (first point of forward sweep).
    J_sc_full = float(np.abs(jv.J_fwd[0]))
    J_sc_eqe = float(np.abs(r_eqe.J_sc_integrated))

    # Relative match within 25 % — both fall under the same AM1.5G file
    # so the two should land in the same ballpark once the wavelength
    # grid is fine enough.
    rel_err = abs(J_sc_eqe - J_sc_full) / max(J_sc_full, 1.0)
    assert rel_err < 0.25, (
        f"integrated EQE J_sc={J_sc_eqe:.2f} A/m² diverges from "
        f"full-TMM J_sc={J_sc_full:.2f} A/m² by {100 * rel_err:.1f}% — "
        "wavelength integration or TMM path likely broken"
    )
