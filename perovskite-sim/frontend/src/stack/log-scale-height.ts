/**
 * Map a layer thickness in metres to a card height in pixels using a
 * log10 scale clamped to a fixed range. The full spectrum of perovskite
 * stacks spans 1 nm (interface) to 1 mm (glass substrate); a linear
 * mapping would make the substrate dwarf every other layer, so we
 * compress the range to keep every card visible while preserving order.
 */

export const MIN_M = 1e-9    // 1 nm
export const MAX_M = 1e-3    // 1 mm
export const MIN_PX = 18
export const MAX_PX = 96

const LOG_MIN = Math.log10(MIN_M)
const LOG_MAX = Math.log10(MAX_M)
const LOG_SPAN = LOG_MAX - LOG_MIN
const PX_SPAN = MAX_PX - MIN_PX

export function logScaleHeight(thicknessMetres: number): number {
  if (!Number.isFinite(thicknessMetres)) {
    return thicknessMetres === Infinity ? MAX_PX : MIN_PX
  }
  if (thicknessMetres <= 0) return MIN_PX
  const t = Math.max(MIN_M, Math.min(MAX_M, thicknessMetres))
  const frac = (Math.log10(t) - LOG_MIN) / LOG_SPAN
  return Math.round(MIN_PX + frac * PX_SPAN)
}
