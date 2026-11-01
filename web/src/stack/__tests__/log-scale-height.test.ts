import { describe, it, expect } from 'vitest'
import { logScaleHeight, MIN_PX, MAX_PX, MIN_M, MAX_M } from '../log-scale-height'

describe('logScaleHeight', () => {
  it('maps minimum thickness to MIN_PX', () => {
    expect(logScaleHeight(MIN_M)).toBe(MIN_PX)
  })

  it('maps maximum thickness to MAX_PX', () => {
    expect(logScaleHeight(MAX_M)).toBe(MAX_PX)
  })

  it('clamps thicknesses below MIN_M to MIN_PX', () => {
    expect(logScaleHeight(MIN_M / 100)).toBe(MIN_PX)
  })

  it('clamps thicknesses above MAX_M to MAX_PX', () => {
    expect(logScaleHeight(MAX_M * 100)).toBe(MAX_PX)
  })

  it('preserves ordering across the valid range', () => {
    const samples = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 4e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    const heights = samples.map(logScaleHeight)
    for (let i = 1; i < heights.length; i++) {
      expect(heights[i]).toBeGreaterThanOrEqual(heights[i - 1])
    }
  })

  it('returns finite, positive integers', () => {
    for (const t of [1e-9, 4e-7, 1e-3]) {
      const h = logScaleHeight(t)
      expect(Number.isFinite(h)).toBe(true)
      expect(h).toBeGreaterThan(0)
      expect(Number.isInteger(h)).toBe(true)
    }
  })

  it('returns MIN_PX for non-finite or non-positive input', () => {
    expect(logScaleHeight(0)).toBe(MIN_PX)
    expect(logScaleHeight(-1)).toBe(MIN_PX)
    expect(logScaleHeight(NaN)).toBe(MIN_PX)
    expect(logScaleHeight(Infinity)).toBe(MAX_PX)
  })
})
