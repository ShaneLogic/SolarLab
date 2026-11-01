import { describe, it, expect } from 'vitest'
import { medfilt, savgol11, smoothEQE } from './signal-smooth'

describe('medfilt', () => {
  it('removes a lone spike (window 5)', () => {
    const out = medfilt([1, 1, 1, 9, 1, 1, 1], 5)
    expect(out[3]).toBe(1)
  })

  it('shrinks the window at the edges, leaving a flat signal flat', () => {
    expect(medfilt([2, 2, 2, 2, 2], 5)).toEqual([2, 2, 2, 2, 2])
  })
})

describe('savgol11', () => {
  it('preserves a constant (kernel weights sum to 1)', () => {
    const y = Array(20).fill(0.9)
    for (const v of savgol11(y)) expect(v).toBeCloseTo(0.9, 12)
  })

  it('returns the input unchanged when shorter than the window', () => {
    const y = [0.1, 0.6, 0.85, 0.8, 0.05]
    expect(savgol11(y)).toEqual(y)
  })

  it('attenuates an alternating sawtooth toward its mean', () => {
    const y = Array.from({ length: 20 }, (_, i) => (i % 2 === 0 ? 0.8 : 1.0))
    const s = savgol11(y)
    const interior = s.slice(6, 14) // away from clamped edges
    for (const v of interior) {
      expect(v).toBeGreaterThan(0.82)
      expect(v).toBeLessThan(0.98)
    }
  })
})

describe('smoothEQE', () => {
  it('does not mutate the input array', () => {
    const y = [0.1, 0.65, 1.5, 0.85, 0.8, 0.05]
    const before = [...y]
    smoothEQE(y)
    expect(y).toEqual(before)
  })
})
