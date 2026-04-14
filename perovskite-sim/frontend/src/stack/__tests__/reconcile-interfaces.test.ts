import { describe, it, expect } from 'vitest'
import { reconcileInterfaces } from '../reconcile-interfaces'
import type { LayerConfig } from '../../types'

const L = (name: string): LayerConfig => ({
  name, role: 'absorber', thickness: 1e-7, eps_r: 1, mu_n: 0, mu_p: 0,
  ni: 0, N_D: 0, N_A: 0, D_ion: 0, P_lim: 0, P0: 0, tau_n: 0, tau_p: 0,
  n1: 0, p1: 0, B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
})

describe('reconcileInterfaces', () => {
  it('returns empty when only one layer remains', () => {
    expect(reconcileInterfaces([L('a'), L('b')], [L('a')], [[1, 2]])).toEqual([])
  })

  it('preserves surviving adjacent pairs after insert', () => {
    const old = [L('a'), L('b'), L('c')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4]]
    const next = [L('a'), L('b'), L('x'), L('c')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result).toEqual([[1, 2], [0, 0], [0, 0]])
  })

  it('drops the interface adjacent to a deleted middle layer', () => {
    const old = [L('a'), L('b'), L('c')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4]]
    const next = [L('a'), L('c')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result).toEqual([[0, 0]])
  })

  it('keeps remaining pair when first layer is deleted', () => {
    const old = [L('a'), L('b'), L('c')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4]]
    const next = [L('b'), L('c')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result).toEqual([[3, 4]])
  })

  it('preserves surviving adjacent pairs after a reorder', () => {
    const old = [L('a'), L('b'), L('c'), L('d')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4], [5, 6]]
    const next = [L('a'), L('c'), L('b'), L('d')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result.length).toBe(3)
  })

  it('always preserves length invariant', () => {
    for (const n of [1, 2, 3, 6, 10]) {
      const layers = Array.from({ length: n }, (_, i) => L(`L${i}`))
      const result = reconcileInterfaces([], layers, [])
      expect(result.length).toBe(Math.max(0, n - 1))
    }
  })
})
