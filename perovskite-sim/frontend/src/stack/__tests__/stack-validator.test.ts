import { describe, it, expect } from 'vitest'
import { validate } from '../stack-validator'
import type { DeviceConfig, LayerConfig } from '../../types'

const layer = (overrides: Partial<LayerConfig>): LayerConfig => ({
  name: 'L', role: 'absorber', thickness: 1e-7, eps_r: 1,
  mu_n: 0, mu_p: 0, ni: 0, N_D: 0, N_A: 0,
  D_ion: 0, P_lim: 0, P0: 0,
  tau_n: 0, tau_p: 0, n1: 0, p1: 0,
  B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  ...overrides,
})

const cfg = (layers: LayerConfig[]): DeviceConfig => ({
  device: { V_bi: 1, Phi: 1, mode: 'full', interfaces: [] },
  layers,
})

describe('validate', () => {
  it('passes a valid n-i-p stack', () => {
    const c = cfg([
      layer({ name: 'TiO2', role: 'ETL' }),
      layer({ name: 'MAPbI3', role: 'absorber' }),
      layer({ name: 'spiro', role: 'HTL' }),
    ])
    const r = validate(c)
    expect(r.errors).toEqual([])
  })

  it('passes a valid p-i-n stack (orientation symmetric)', () => {
    const c = cfg([
      layer({ name: 'PEDOT', role: 'HTL' }),
      layer({ name: 'MAPbI3', role: 'absorber' }),
      layer({ name: 'C60', role: 'ETL' }),
    ])
    expect(validate(c).errors).toEqual([])
  })

  it('errors when there is no absorber', () => {
    const c = cfg([
      layer({ name: 'TiO2', role: 'ETL' }),
      layer({ name: 'spiro', role: 'HTL' }),
    ])
    const r = validate(c)
    expect(r.errors.some(e => e.message.includes('absorber'))).toBe(true)
  })

  it('errors when there are two absorbers', () => {
    const c = cfg([
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('absorber'))).toBe(true)
  })

  it('errors on duplicate layer names', () => {
    const c = cfg([
      layer({ name: 'X', role: 'ETL' }),
      layer({ name: 'X', role: 'absorber' }),
    ])
    const r = validate(c)
    expect(r.errors.some(e => e.message.includes('Duplicate'))).toBe(true)
  })

  it('errors when a thickness is zero', () => {
    const c = cfg([
      layer({ name: 'A', role: 'ETL', thickness: 0 }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('positive'))).toBe(true)
  })

  it('errors when more than one substrate is present', () => {
    const c = cfg([
      layer({ name: 'g1', role: 'substrate', incoherent: true, optical_material: 'glass' }),
      layer({ name: 'g2', role: 'substrate', incoherent: true, optical_material: 'glass' }),
      layer({ name: 'A', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('At most one substrate'))).toBe(true)
  })

  it('errors when the substrate is not the first layer', () => {
    const c = cfg([
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'g', role: 'substrate', incoherent: true, optical_material: 'glass' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('first'))).toBe(true)
  })

  it('errors when the substrate is not incoherent', () => {
    const c = cfg([
      layer({ name: 'g', role: 'substrate', incoherent: false, optical_material: 'glass' }),
      layer({ name: 'A', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('incoherent'))).toBe(true)
  })

  it('errors when the substrate has no optical material', () => {
    const c = cfg([
      layer({ name: 'g', role: 'substrate', incoherent: true, optical_material: null }),
      layer({ name: 'A', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('optical material'))).toBe(true)
  })

  it('warns when an interface row is the (0,0) default — surfaced by callers', () => {
    const c: DeviceConfig = {
      device: { V_bi: 1, Phi: 1, mode: 'full', interfaces: [[0, 0]] },
      layers: [
        layer({ name: 'A', role: 'ETL' }),
        layer({ name: 'B', role: 'absorber' }),
      ],
    }
    expect(validate(c).errors).toEqual([])
  })

  it('warns on mixed TMM / Beer-Lambert layers', () => {
    const c = cfg([
      layer({ name: 'A', role: 'ETL', optical_material: 'TiO2' }),
      layer({ name: 'B', role: 'absorber', optical_material: null }),
    ])
    const r = validate(c)
    expect(r.warnings.some(w => w.message.includes('Mixed'))).toBe(true)
  })
})
