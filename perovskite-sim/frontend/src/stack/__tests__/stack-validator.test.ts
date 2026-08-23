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

  it('errors when thickness is NaN', () => {
    const c = cfg([
      layer({ name: 'A', role: 'ETL', thickness: NaN }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('positive'))).toBe(true)
  })

  it('errors when thickness is negative', () => {
    const c = cfg([
      layer({ name: 'A', role: 'ETL', thickness: -1e-7 }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('positive'))).toBe(true)
  })

  it('errors when thickness is Infinity', () => {
    // Infinity > 0 is true, so the current guard accepts it. We do not
    // reject Infinity here on purpose: the visualizer log-scale clamps it,
    // and physical-sanity backend validation will catch it.
    const c = cfg([
      layer({ name: 'A', role: 'ETL', thickness: Infinity }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.filter(e => e.field === 'thickness')).toEqual([])
  })

  it('does not emit dormant-TMM warning on an empty stack', () => {
    const c = cfg([])
    const r = validate(c)
    expect(r.warnings.some(w => w.message.includes('dormant'))).toBe(false)
    // Empty stack still hard-errors on the absorber rule.
    expect(r.errors.some(e => e.message.includes('absorber'))).toBe(true)
  })

  it('accepts an active, in-domain graded-CIGS optical model', () => {
    const c = cfg([layer({
      name: 'CIGS',
      Eg: 1.15,
      Eg_back: 1.4,
      cigs_graded_optics: {
        model: 'minoura_2015',
        ggi_front: 0.225,
        ggi_back: 0.6,
        cgi: 0.9,
        slices: 25,
        kk_quadrature_order: 192,
      },
    })])
    c.device.band_grading = true
    c.device.graded_optics = true
    const r = validate(c)
    expect(r.errors).toEqual([])
    expect(r.warnings.some(w => w.message.includes('dormant'))).toBe(false)
  })

  it('rejects graded-CIGS activation without the electrical grade', () => {
    const c = cfg([layer({
      name: 'CIGS',
      cigs_graded_optics: {
        ggi_front: 0.225,
        ggi_back: 0.6,
        cgi: 0.9,
      },
    })])
    c.device.graded_optics = true
    const fields = validate(c).errors.map(error => error.field)
    expect(fields).toContain('graded_optics')
    expect(fields).toContain('cigs_graded_optics')
  })

  it('rejects a CIGS model outside its published composition domain', () => {
    const c = cfg([layer({
      name: 'CIGS',
      Eg_back: 1.4,
      cigs_graded_optics: {
        ggi_front: -0.1,
        ggi_back: 0.6,
        cgi: 0.7,
        slices: 0,
        kk_quadrature_order: 24,
      },
    })])
    c.device.band_grading = true
    c.device.graded_optics = true
    const fields = validate(c).errors.map(error => error.field)
    expect(fields).toEqual(expect.arrayContaining([
      'ggi_front', 'cgi', 'slices', 'kk_quadrature_order',
    ]))
  })

  it('requires both explicit metal work functions', () => {
    const c = cfg([
      layer({ name: 'H', role: 'HTL' }),
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'E', role: 'ETL' }),
    ])
    c.device = {
      Phi: 1,
      mode: 'full',
      built_in_potential_mode: 'metal_work_function',
      work_function_left_eV: 5.2,
    }

    const fields = validate(c).errors.map(error => error.field)
    expect(fields).toContain('work_function_right_eV')
  })

  it('accepts a complete semiconductor-work-function contact pair', () => {
    const contactFields = { chi: 4, Eg: 2, Nc300: 1e25, Nv300: 1e25 }
    const c = cfg([
      layer({ name: 'H', role: 'HTL', N_A: 1e23, ...contactFields }),
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'E', role: 'ETL', N_D: 1e23, ...contactFields }),
    ])
    c.device = {
      Phi: 1,
      mode: 'full',
      built_in_potential_mode: 'semiconductor_work_function',
    }

    expect(validate(c).errors).toEqual([])
  })

  it('reports missing DOS at a semiconductor contact', () => {
    const c = cfg([
      layer({ name: 'H', role: 'HTL', chi: 4, Eg: 2 }),
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'E', role: 'ETL', chi: 4, Eg: 2 }),
    ])
    c.device = {
      Phi: 1,
      mode: 'full',
      built_in_potential_mode: 'semiconductor_work_function',
    }

    const fields = validate(c).errors.map(error => error.field)
    expect(fields).toContain('Nc300')
    expect(fields).toContain('Nv300')
  })
})
