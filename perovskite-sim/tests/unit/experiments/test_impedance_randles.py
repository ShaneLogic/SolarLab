"""Analytic Randles-circuit regression for the lock-in impedance extractor.

Tests the lock-in algorithm inside ``experiments/impedance.py`` against a
textbook Randles equivalent-circuit response (R_s in series with R_p in
parallel with C). The test bypasses the drift-diffusion solver and feeds
the lock-in helper synthetic time-domain currents whose phasors correspond
exactly to Z_Randles(ω). A correct lock-in must return |Z_sim| within 1 %
of |Z_analytic| and arg(Z_sim) within 1° of arg(Z_analytic) across the
0.01 Hz – 1 MHz sweep range.

Why this matters
----------------
The impedance pipeline (``run_impedance``) has two failure modes that are
otherwise invisible: (1) sign/phase-convention bugs that flip the
capacitive arc to the wrong half-plane, and (2) bad window truncation
that leaks the AC component into the DC offset and biases the magnitude.
Validating against a closed-form Randles response isolates the lock-in
arithmetic from every other part of the pipeline, so any later regression
in this helper shows up here before it corrupts end-to-end experiments.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments.impedance import _lockin_extract


# ── Randles circuit parameters (passive convention) ─────────────────────
_R_S  = 2.0       # series resistance [Ω·m²]
_R_P  = 50.0      # polarisation resistance [Ω·m²]
_C    = 1.0e-6    # double-layer capacitance [F·m⁻²]
_DV   = 1.0e-3    # small-signal AC amplitude [V]

# ── Lock-in sampling parameters ─────────────────────────────────────────
_N_CYCLES       = 5
_PTS_PER_CYCLE  = 40       # matches run_impedance's default sampling
_N_EXTRACT      = 2        # only the last two cycles go into the lock-in


def _Z_randles(freqs: np.ndarray) -> np.ndarray:
    """Analytic Z(ω) for R_s + (R_p || C)."""
    omega = 2 * np.pi * freqs
    return _R_S + _R_P / (1.0 + 1j * omega * _R_P * _C)


def _synth_current_series(freq: float, Z_an: complex) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_ext, I_ext) mimicking run_impedance's sampling pattern.

    We emulate the exact timebase used in ``run_impedance``: ``t_eval`` is
    ``n_cycles·pts_per_cycle + 1`` uniformly spaced nodes spanning
    ``n_cycles`` full periods, and the extraction grid sits at the midpoints
    of the last ``n_extract·pts_per_cycle`` intervals. Feeding synthetic
    currents through this same sampler makes the test immediately sensitive
    to any change in windowing or midpoint alignment inside the driver.
    """
    T = 1.0 / freq
    dt = T / _PTS_PER_CYCLE
    n_intervals = _N_CYCLES * _PTS_PER_CYCLE
    t_eval = np.arange(n_intervals + 1, dtype=float) * dt
    n_ext = _N_EXTRACT * _PTS_PER_CYCLE
    t_ext = 0.5 * (t_eval[-n_ext - 1 : -1] + t_eval[-n_ext:])
    # Phasor: Î = V̂ / Z  with V̂ = δV under the imag-part convention.
    I_hat = _DV / Z_an
    I_in_true = float(np.real(I_hat))
    I_quad_true = float(np.imag(I_hat))
    I_t = I_in_true * np.sin(2 * np.pi * freq * t_ext) + \
          I_quad_true * np.cos(2 * np.pi * freq * t_ext)
    return t_ext, I_t


@pytest.mark.parametrize(
    "freq",
    [0.01, 0.1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6],
)
def test_lockin_matches_randles(freq: float):
    """Lock-in Z matches analytic Randles Z to 1 % magnitude, 1° phase."""
    Z_an = _Z_randles(np.array([freq]))[0]
    t_ext, I_ext = _synth_current_series(freq, Z_an)
    Z_sim = _lockin_extract(I_ext, t_ext, freq, _DV)

    mag_err = abs(abs(Z_sim) - abs(Z_an)) / abs(Z_an)
    phase_err_deg = abs(np.angle(Z_sim, deg=True) - np.angle(Z_an, deg=True))
    # Wrap phase error into [0, 180]
    if phase_err_deg > 180.0:
        phase_err_deg = 360.0 - phase_err_deg

    assert mag_err < 0.01, (
        f"f={freq:g} Hz: magnitude error {mag_err*100:.3f}% exceeds 1% "
        f"(|Z_sim|={abs(Z_sim):.4g}, |Z_an|={abs(Z_an):.4g})"
    )
    assert phase_err_deg < 1.0, (
        f"f={freq:g} Hz: phase error {phase_err_deg:.3f}° exceeds 1° "
        f"(arg(Z_sim)={np.angle(Z_sim, deg=True):.3f}°, "
        f"arg(Z_an)={np.angle(Z_an, deg=True):.3f}°)"
    )


def test_lockin_rejects_linear_drift():
    """A large linear drift added on top of the AC response is detrended out."""
    freq = 100.0
    Z_an = _Z_randles(np.array([freq]))[0]
    t_ext, I_ext = _synth_current_series(freq, Z_an)
    # Inject a slow drift comparable to the AC amplitude itself.
    drift = np.linspace(0.0, 1.0e-4, len(t_ext))
    Z_sim = _lockin_extract(I_ext + drift, t_ext, freq, _DV)
    mag_err = abs(abs(Z_sim) - abs(Z_an)) / abs(Z_an)
    assert mag_err < 0.01, (
        f"drift not detrended: mag err {mag_err*100:.3f}% "
        f"(|Z_sim|={abs(Z_sim):.4g}, |Z_an|={abs(Z_an):.4g})"
    )


def test_lockin_dc_limit_is_Rs_plus_Rp():
    """At f → 0 the Randles circuit degenerates to R_s + R_p (pure real).

    The lock-in at the low end of our sweep is sampling over ~5 periods of
    a 0.01 Hz excitation (500 s of simulated time) — capacitive admittance
    is near-zero there, so the returned Z should be within 1 % of R_s+R_p
    with phase near 0°.
    """
    freq = 0.01
    Z_an = _Z_randles(np.array([freq]))[0]
    t_ext, I_ext = _synth_current_series(freq, Z_an)
    Z_sim = _lockin_extract(I_ext, t_ext, freq, _DV)
    assert abs(Z_sim.real - (_R_S + _R_P)) / (_R_S + _R_P) < 0.01, (
        f"DC-limit real part drift: Re(Z_sim)={Z_sim.real:.4f} vs "
        f"R_s+R_p={_R_S + _R_P:.4f}"
    )


def test_lockin_hf_limit_is_Rs():
    """At high f, Re(Z) → R_s. Im(Z) is non-zero but small and must match analytic.

    At f = 1 MHz with R_p = 50, C = 1e-6, ωR_pC = 2π·1e6·50·1e-6 ≈ 314.
    Z_analytic = R_s + R_p/(1 + jωR_pC)
               ≈ 2.00 + 0.0005 − 0.159j.
    So Re(Z) is effectively R_s to 0.03%, but Im(Z) is ≈ −0.159, not zero.
    We therefore check Re(Z_sim) ≈ R_s and Im(Z_sim) ≈ Im(Z_an) separately.
    """
    freq = 1.0e6
    Z_an = _Z_randles(np.array([freq]))[0]
    t_ext, I_ext = _synth_current_series(freq, Z_an)
    Z_sim = _lockin_extract(I_ext, t_ext, freq, _DV)
    assert abs(Z_sim.real - _R_S) / _R_S < 0.05, (
        f"HF-limit real part drift: Re(Z_sim)={Z_sim.real:.4f} vs R_s={_R_S:.4f}"
    )
    # Im(Z_sim) should agree with the analytic capacitive-branch residual
    # (−R_p·ωR_pC / (1+(ωR_pC)²) ≈ −0.159 Ω·m²) to 1 %.
    imag_err = abs(Z_sim.imag - Z_an.imag) / abs(Z_an.imag)
    assert imag_err < 0.01, (
        f"HF-limit imag part drift: Im(Z_sim)={Z_sim.imag:.4f} vs "
        f"Im(Z_an)={Z_an.imag:.4f} (rel err {imag_err*100:.2f}%)"
    )
