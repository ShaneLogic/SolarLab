import type { DeviceConfig } from './types'

/** True when at least one electrical layer carries tabulated optical n,k
 * (i.e. `optical_material` is a non-empty string). Spectral experiments
 * (EQE, EL) require this; Beer-Lambert presets do not satisfy it and will
 * be rejected by the backend with a ValueError. */
export function hasTMMOptics(config: DeviceConfig): boolean {
  return config.layers.some(
    (l) => typeof l.optical_material === 'string' && l.optical_material.length > 0,
  )
}
