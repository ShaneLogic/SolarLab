import { describe, expect, it } from 'vitest'

import {
  dynamicDefectImpedancePreset,
  hasActiveMobileIons,
  hasExplicitBulkDefects,
  hasExplicitInterfaceDefects,
} from './explicit-defect-capability'
import type { DeviceConfig, LayerConfig } from './types'

function layer(overrides: Partial<LayerConfig> = {}): LayerConfig {
  return {
    name: 'absorber',
    role: 'absorber',
    thickness: 500e-9,
    eps_r: 20,
    mu_n: 1e-3,
    mu_p: 1e-3,
    ni: 1e10,
    N_D: 0,
    N_A: 0,
    D_ion: 0,
    P_lim: 0,
    P0: 0,
    tau_n: 1e-6,
    tau_p: 1e-6,
    n1: 1e10,
    p1: 1e10,
    B_rad: 0,
    C_n: 0,
    C_p: 0,
    alpha: 0,
    ...overrides,
  }
}

function config(
  layers: LayerConfig[],
  interfaceDefects: DeviceConfig['device']['interface_defects'] = [],
): DeviceConfig {
  return {
    device: { Phi: 0, interface_defects: interfaceDefects },
    layers,
  }
}

const explicitBulk = {
  defect_schema_version: 'solarlab-explicit-bulk-defects-v3' as const,
  defect_model: 'explicit_quasi_steady' as const,
  bulk_defects: [{}] as unknown as NonNullable<LayerConfig['bulk_defects']>,
}

describe('dynamic-defect impedance presets', () => {
  it('selects the bulk trap window for an ion-free bulk defect', () => {
    const device = config([layer(explicitBulk)])

    expect(hasExplicitBulkDefects(device)).toBe(true)
    expect(dynamicDefectImpedancePreset(device)).toEqual({
      N_grid: 12,
      n_freq: 33,
      f_min: 1e-4,
      f_max: 1e12,
    })
  })

  it('selects the wider interface trap window', () => {
    const device = config(
      [layer(), layer({ name: 'right' })],
      [{
        sigma_n_cm2: 1e-15,
        sigma_p_cm2: 2e-15,
        N_t_cm2: 1e11,
        v_th_cm_s: 1e7,
        E_t_eV_below_cb: 0.5,
      }],
    )

    expect(hasExplicitInterfaceDefects(device)).toBe(true)
    expect(dynamicDefectImpedancePreset(device)).toEqual({
      N_grid: 8,
      n_freq: 45,
      f_min: 1e-8,
      f_max: 1e14,
    })
  })

  it('selects the combined ion and trap window and honors grid hints', () => {
    const device = config([
      layer({ ...explicitBulk, D_ion: 1e-14, P0: 1e22, P_lim: 2e22 }),
    ])
    device.simulation_hints = { min_N_grid: 24 }

    expect(hasActiveMobileIons(device)).toBe(true)
    expect(dynamicDefectImpedancePreset(device)).toEqual({
      N_grid: 24,
      n_freq: 19,
      f_min: 1e-3,
      f_max: 1e6,
    })
  })

  it('does not treat all-null interface slots as active defects', () => {
    const device = config([layer(), layer({ name: 'right' })], [null])

    expect(hasExplicitInterfaceDefects(device)).toBe(false)
  })
})
