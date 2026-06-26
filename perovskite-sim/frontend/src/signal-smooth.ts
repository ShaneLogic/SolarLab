/**
 * Display-only smoothing for noisy spectral curves (EQE).
 *
 * Each EQE point is an independent steady-state solve whose terminal current is
 * the small difference of near-cancelling drift/diffusion fluxes minus the dark
 * baseline, so the plateau carries ~±10% high-frequency NUMERICAL noise that is
 * uncorrelated point-to-point and does not damp with a longer settle. Real TMM
 * interference is ~180 nm period, far broader than this ~14 nm noise, so a
 * two-stage median-5 (kills spike outliers) then Savitzky-Golay-11/quadratic
 * (~77 nm window on the 80-point grid) removes the noise without touching the
 * physics or the sharp band edge.
 *
 * Mirrors scripts/plot_eqe.py (scipy medfilt + savgol_filter) so the UI and the
 * publication figure agree. Edges use clamp padding rather than scipy's
 * polynomial interp — the 5 edge points differ negligibly (the band-edge tail
 * and the short-λ rise), and this keeps the kernel a fixed convolution.
 */

// Savitzky-Golay quadratic/cubic smoothing weights, window length 11.
// Sum = 429, so dividing by 429 preserves a constant signal.
const SG11 = [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36]
const SG11_SUM = 429
const SG11_HALF = 5

/** Running median over an odd window; the window shrinks at the array edges. */
export function medfilt(y: readonly number[], w: number): number[] {
  const m = Math.floor(w / 2)
  return y.map((_, i) => {
    const win = y.slice(Math.max(0, i - m), Math.min(y.length, i + m + 1))
      .slice()
      .sort((a, b) => a - b)
    return win[Math.floor(win.length / 2)]
  })
}

/** Savitzky-Golay quadratic smooth, fixed window 11, edge-clamp padding. */
export function savgol11(y: readonly number[]): number[] {
  if (y.length < SG11.length) return y.slice()
  const at = (i: number): number =>
    y[i < 0 ? 0 : i >= y.length ? y.length - 1 : i]
  return y.map((_, i) => {
    let s = 0
    for (let k = -SG11_HALF; k <= SG11_HALF; k++) s += SG11[k + SG11_HALF] * at(i + k)
    return s / SG11_SUM
  })
}

/**
 * Smooth an EQE curve for display: median-5 then Savitzky-Golay-11. Does not
 * mutate the input. Returns the raw curve unchanged when too short to smooth.
 */
export function smoothEQE(eqe: readonly number[]): number[] {
  return savgol11(medfilt(eqe, 5))
}
