import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { readDeviceEditor, renderDeviceEditor } from './config-editor'
import type { BulkDefectSpecies, DeviceConfig, LayerConfig } from './types'

function layer(name: string, role: LayerConfig['role']): LayerConfig {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 3,
    mu_n: 1e-4,
    mu_p: 1e-4,
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
    chi: 4,
    Eg: 1.6,
  }
}

function defect(overrides: Partial<BulkDefectSpecies> = {}): BulkDefectSpecies {
  return {
    name: 'V_I',
    distribution: {
      kind: 'single_level',
      normalization: 'integrated_total',
      total_density_m3: 2.5e21,
      center_eV_above_vb: 0.62,
    },
    charge_transition: 'acceptor',
    neutral_reference: 'empty',
    kinetics: {
      sigma_n_m2: 3e-19,
      sigma_p_m2: 7e-20,
      thermal_velocity_n_m_s: 2e5,
      thermal_velocity_p_m_s: 1.5e5,
    },
    degeneracy: 2,
    ...overrides,
  }
}

function config(absorber: Partial<LayerConfig> = {}): DeviceConfig {
  return {
    device: { V_bi: 1.1, Phi: 2.5e21, mode: 'full' },
    layers: [
      layer('HTL', 'HTL'),
      { ...layer('PVK', 'absorber'), ...absorber },
      layer('ETL', 'ETL'),
    ],
  }
}

function explicitConfig(): DeviceConfig {
  return config({
    defect_schema_version: 'solarlab-explicit-bulk-defects-v1',
    defect_model: 'explicit_quasi_steady',
    bulk_defects: [defect()],
  })
}

let container: HTMLElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})

afterEach(() => document.body.replaceChildren())

describe('explicit bulk-defect layer editor', () => {
  it('keeps legacy layers free of versioned defect keys when untouched', () => {
    const original = config()
    renderDeviceEditor(container, original, 'full', 1)
    expect(document.getElementById('layer-1-defect-enabled')).not.toBeNull()
    const output = readDeviceEditor(original, 1).layers[1] as unknown as Record<string, unknown>
    expect('defect_schema_version' in output).toBe(false)
    expect('defect_model' in output).toBe(false)
    expect('bulk_defects' in output).toBe(false)
  })

  it('round-trips the canonical single-level SI document exactly', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'full', 1)
    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_schema_version).toBe(original.layers[1].defect_schema_version)
    expect(output.defect_model).toBe(original.layers[1].defect_model)
    expect(output.bulk_defects).toEqual(original.layers[1].bulk_defects)
  })

  it('derives the thermodynamic neutral reference from the charge transition', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'full', 1)
    const transition = container.querySelector<HTMLSelectElement>(
      '[data-defect-field="charge_transition"]',
    )!
    const reference = container.querySelector<HTMLInputElement>(
      '[data-defect-field="neutral_reference"]',
    )!
    transition.value = 'donor'
    transition.dispatchEvent(new Event('change', { bubbles: true }))
    expect(reference.value).toBe('filled')
    const output = readDeviceEditor(original, 1).layers[1].bulk_defects![0]
    expect(output.charge_transition).toBe('donor')
    expect(output.neutral_reference).toBe('filled')
  })

  it('converts SCAPS cgs display values back to canonical SI', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'full', 1)
    const units = document.getElementById('layer-1-defect-units') as HTMLSelectElement
    units.value = 'scaps_cgs'
    units.dispatchEvent(new Event('change'))
    const density = container.querySelector<HTMLInputElement>('[data-defect-field="total_density"]')!
    const sigmaN = container.querySelector<HTMLInputElement>('[data-defect-field="sigma_n"]')!
    const velocityN = container.querySelector<HTMLInputElement>('[data-defect-field="thermal_velocity_n"]')!
    expect(Number(density.value)).toBeCloseTo(2.5e15)
    expect(Number(sigmaN.value)).toBeCloseTo(3e-15)
    expect(Number(velocityN.value)).toBeCloseTo(2e7)
    density.value = '4e15'
    sigmaN.value = '8e-15'
    velocityN.value = '3e7'
    const output = readDeviceEditor(original, 1).layers[1].bulk_defects![0]
    expect(output.distribution.total_density_m3).toBeCloseTo(4e21)
    expect(output.kinetics.sigma_n_m2).toBeCloseTo(8e-19)
    expect(output.kinetics.thermal_velocity_n_m_s).toBeCloseTo(3e5)
  })

  it('rejects an empty numeric field instead of coercing it to physical zero', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'full', 1)
    container.querySelector<HTMLInputElement>('[data-defect-field="sigma_n"]')!.value = ''
    expect(() => readDeviceEditor(original, 1)).toThrow(/cross-section must not be empty/)
  })

  it('adds and removes species without changing the explicit model selector', () => {
    const original = config()
    renderDeviceEditor(container, original, 'full', 1)
    const enabled = document.getElementById('layer-1-defect-enabled') as HTMLInputElement
    enabled.checked = true
    enabled.dispatchEvent(new Event('change'))
    expect(container.querySelectorAll('[data-defect-species]')).toHaveLength(1)
    container.querySelector<HTMLButtonElement>('[data-defect-add]')!.click()
    expect(container.querySelectorAll('[data-defect-species]')).toHaveLength(2)
    container.querySelector<HTMLButtonElement>('[data-defect-remove]')!.click()
    expect(container.querySelectorAll('[data-defect-species]')).toHaveLength(1)
    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_model).toBe('explicit_quasi_steady')
    expect(output.bulk_defects).toHaveLength(1)
  })

  it('preserves explicit metadata when the editor is hidden outside FULL tier', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'fast', 1)
    expect(document.getElementById('layer-1-bulk-defect-editor')).toBeNull()
    expect(readDeviceEditor(original, 1).layers[1].bulk_defects).toEqual(
      original.layers[1].bulk_defects,
    )
  })

  it('preserves unsupported Gaussian metadata in a read-only panel', () => {
    const gaussian = defect({
      distribution: {
        kind: 'gaussian',
        normalization: 'integrated_total',
        total_density_m3: 1e21,
        center_eV_above_vb: 0.7,
        width_eV: 0.08,
        width_convention: 'gaussian_standard_deviation',
      },
    })
    const original = config({
      defect_schema_version: 'solarlab-explicit-bulk-defects-v1',
      defect_model: 'effective_lifetime',
      bulk_defects: [gaussian],
    })
    renderDeviceEditor(container, original, 'full', 1)
    expect(container.querySelector('.bulk-defect-readonly')).not.toBeNull()
    expect(document.getElementById('layer-1-defect-enabled')).toBeNull()
    expect(readDeviceEditor(original, 1).layers[1].bulk_defects).toEqual([gaussian])
  })
})
