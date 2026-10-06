import type { DeviceConfig } from './types'

/** True when at least one electrical layer carries wavelength-resolved n,k.
 * This may be a tabulated `optical_material` or an active graded-CIGS model.
 * Spectral experiments (EQE, EL) require one of these sources; scalar
 * Beer-Lambert presets do not satisfy the contract. */
export function hasTMMOptics(config: DeviceConfig): boolean {
  const hasTabulated = config.layers.some(
    (l) => typeof l.optical_material === 'string' && l.optical_material.length > 0,
  )
  const hasGradedCigs = config.device.graded_optics === true
    && config.device.band_grading === true
    && config.layers.some((l) => l.cigs_graded_optics != null)
  return hasTabulated || hasGradedCigs
}
