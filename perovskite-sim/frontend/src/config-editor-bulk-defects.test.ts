import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { readDeviceEditor, renderDeviceEditor } from './config-editor'
import type {
  BulkDefectDistribution,
  BulkDefectSpecies,
  BulkDefectSpatialProfile,
  DeviceConfig,
  LayerConfig,
} from './types'

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

function v2Distribution(
  kind: BulkDefectDistribution['kind'],
): BulkDefectDistribution {
  const base = {
    kind,
    normalization: 'integrated_total' as const,
    total_density_m3: 2.5e21,
    center_eV_above_vb: 0.8,
    energy_reference: 'above_valence_band' as const,
  }
  if (kind === 'single_level') return base
  if (kind === 'uniform') {
    return { ...base, width_eV: 0.4, width_convention: 'uniform_full_width' }
  }
  if (kind === 'gaussian') {
    return {
      ...base,
      width_eV: 0.1,
      width_convention: 'gaussian_standard_deviation',
      support_width_multiplier: 6,
    }
  }
  return {
    ...base,
    center_eV_above_vb: kind === 'conduction_band_tail' ? 1.2 : 0.4,
    width_eV: 0.1,
    width_convention: 'scaps_characteristic_energy',
    support_width_multiplier: 4,
  }
}

function profile(knots: BulkDefectSpatialProfile['knots'] = [
  { position_fraction: 0, density_multiplier: 0.6 },
  { position_fraction: 0.5, density_multiplier: 1 },
  { position_fraction: 1, density_multiplier: 1.4 },
]): BulkDefectSpatialProfile {
  return {
    coordinate: 'normalized_layer_coordinate',
    interpolation: 'piecewise_linear',
    density_normalization: 'layer_average_unity',
    knots,
  }
}

function distributedConfig(version: 'v2' | 'v3'): DeviceConfig {
  const kinds: BulkDefectDistribution['kind'][] = [
    'single_level',
    'gaussian',
    'uniform',
    'conduction_band_tail',
    'valence_band_tail',
  ]
  return config({
    defect_schema_version: `solarlab-explicit-bulk-defects-${version}`,
    defect_model: 'explicit_quasi_steady',
    bulk_defects: kinds.map((kind, index) => defect({
      name: `${kind}_${index}`,
      distribution: v2Distribution(kind),
      ...(version === 'v3' && index === 1 ? { spatial_profile: profile() } : {}),
    })),
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

  it('round-trips every canonical v2 energy distribution exactly', () => {
    const original = distributedConfig('v2')
    renderDeviceEditor(container, original, 'full', 1)

    expect(container.querySelector('.bulk-defect-readonly')).toBeNull()
    expect(container.querySelector('[data-defect-schema-label]')?.textContent).toBe('v2')
    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_schema_version).toBe('solarlab-explicit-bulk-defects-v2')
    expect(output.bulk_defects).toEqual(original.layers[1].bulk_defects)
  })

  it('round-trips a mixed canonical v3 profile document exactly', () => {
    const original = distributedConfig('v3')
    renderDeviceEditor(container, original, 'full', 1)

    expect(container.querySelector('.bulk-defect-readonly')).toBeNull()
    expect(container.querySelector('[data-defect-schema-label]')?.textContent).toBe('v3')
    expect(container.querySelector('[data-defect-profile-average]')?.textContent).toContain('1.000000')
    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_schema_version).toBe('solarlab-explicit-bulk-defects-v3')
    expect(output.bulk_defects).toEqual(original.layers[1].bulk_defects)
  })

  it('preserves a v2 single-level document instead of silently downgrading it', () => {
    const original = config({
      defect_schema_version: 'solarlab-explicit-bulk-defects-v2',
      defect_model: 'explicit_quasi_steady',
      bulk_defects: [defect({ distribution: v2Distribution('single_level') })],
    })
    renderDeviceEditor(container, original, 'full', 1)

    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_schema_version).toBe('solarlab-explicit-bulk-defects-v2')
    expect(output.bulk_defects).toEqual(original.layers[1].bulk_defects)
  })

  it.each([
    ['gaussian', 'gaussian_standard_deviation', true],
    ['uniform', 'uniform_full_width', false],
    ['conduction_band_tail', 'scaps_characteristic_energy', true],
    ['valence_band_tail', 'scaps_characteristic_energy', true],
  ] as const)(
    'promotes a v1 single level to v2 when selecting %s',
    (kind, convention, hasSupport) => {
      const original = explicitConfig()
      renderDeviceEditor(container, original, 'full', 1)
      const kindInput = container.querySelector<HTMLSelectElement>(
        '[data-defect-field="distribution_kind"]',
      )!
      kindInput.value = kind
      kindInput.dispatchEvent(new Event('change', { bubbles: true }))

      expect(
        container.querySelector<HTMLElement>('[data-defect-support-field]')!.hidden,
      ).toBe(!hasSupport)
      const output = readDeviceEditor(original, 1).layers[1]
      const distribution = output.bulk_defects![0].distribution
      expect(output.defect_schema_version).toBe('solarlab-explicit-bulk-defects-v2')
      expect(distribution.energy_reference).toBe('above_valence_band')
      expect(distribution.width_convention).toBe(convention)
      expect(distribution.support_width_multiplier !== undefined).toBe(hasSupport)
    },
  )

  it('lets Gaussian inputs explicitly select the SCAPS characteristic-energy convention', () => {
    const original = distributedConfig('v2')
    renderDeviceEditor(container, original, 'full', 1)
    const gaussian = container.querySelectorAll<HTMLElement>('[data-defect-species]')[1]
    const convention = gaussian.querySelector<HTMLSelectElement>(
      '[data-defect-field="width_convention"]',
    )!
    convention.value = 'scaps_characteristic_energy'

    const output = readDeviceEditor(original, 1).layers[1].bulk_defects![1]
    expect(output.distribution.width_convention).toBe('scaps_characteristic_energy')
  })

  it('promotes to v3 and inserts an integral-preserving interpolated profile knot', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'full', 1)
    const profileEnabled = container.querySelector<HTMLInputElement>(
      '[data-defect-profile-enabled]',
    )!
    profileEnabled.checked = true
    profileEnabled.dispatchEvent(new Event('change', { bubbles: true }))
    container.querySelector<HTMLButtonElement>('[data-defect-knot-add]')!.click()

    expect(container.querySelectorAll('[data-defect-profile-knot]')).toHaveLength(3)
    expect(container.querySelector('[data-defect-profile-average]')?.getAttribute('data-valid')).toBe('true')
    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_schema_version).toBe('solarlab-explicit-bulk-defects-v3')
    expect(output.bulk_defects![0].distribution.energy_reference).toBe('above_valence_band')
    expect(output.bulk_defects![0].spatial_profile?.knots).toEqual([
      { position_fraction: 0, density_multiplier: 1 },
      { position_fraction: 0.5, density_multiplier: 1 },
      { position_fraction: 1, density_multiplier: 1 },
    ])
  })

  it('downgrades v3 to complete v2 when the last profile is explicitly disabled', () => {
    const original = distributedConfig('v3')
    renderDeviceEditor(container, original, 'full', 1)
    const profileEnabled = container.querySelectorAll<HTMLInputElement>(
      '[data-defect-profile-enabled]',
    )[1]
    profileEnabled.checked = false
    profileEnabled.dispatchEvent(new Event('change', { bubbles: true }))

    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.defect_schema_version).toBe('solarlab-explicit-bulk-defects-v2')
    expect(output.bulk_defects?.every(item => item.spatial_profile === undefined)).toBe(true)
    expect(output.bulk_defects?.every(item => (
      item.distribution.energy_reference === 'above_valence_band'
    ))).toBe(true)
  })

  it('preserves sub-milliscale knot coordinates exactly when untouched', () => {
    const exactPosition = 0.000123456789
    const original = config({
      defect_schema_version: 'solarlab-explicit-bulk-defects-v3',
      defect_model: 'explicit_quasi_steady',
      bulk_defects: [defect({
        distribution: v2Distribution('single_level'),
        spatial_profile: profile([
          { position_fraction: 0, density_multiplier: 1 },
          { position_fraction: exactPosition, density_multiplier: 1 },
          { position_fraction: 1, density_multiplier: 1 },
        ]),
      })],
    })
    renderDeviceEditor(container, original, 'full', 1)

    const output = readDeviceEditor(original, 1).layers[1]
    expect(output.bulk_defects![0].spatial_profile?.knots[1].position_fraction).toBe(exactPosition)
    expect(output.bulk_defects).toEqual(original.layers[1].bulk_defects)
  })

  it('rejects a distribution whose finite support leaves the minimum graded gap', () => {
    const original = config({
      Eg: 1.6,
      Eg_back: 1.4,
      grading_bowing: 0.2,
      defect_schema_version: 'solarlab-explicit-bulk-defects-v2',
      defect_model: 'explicit_quasi_steady',
      bulk_defects: [defect({ distribution: v2Distribution('uniform') })],
    })
    original.device.band_grading = true
    renderDeviceEditor(container, original, 'full', 1)
    container.querySelector<HTMLInputElement>('[data-defect-field="width"]')!.value = '1.5'

    expect(() => readDeviceEditor(original, 1)).toThrow(/support lies outside the local band gap/)
  })

  it('rejects a non-conservative or non-monotone spatial profile', () => {
    const original = distributedConfig('v3')
    renderDeviceEditor(container, original, 'full', 1)
    const profiled = container.querySelectorAll<HTMLElement>('[data-defect-species]')[1]
    const multipliers = profiled.querySelectorAll<HTMLInputElement>(
      '[data-defect-field="profile_multiplier"]',
    )
    multipliers[1].value = '1.2'
    expect(() => readDeviceEditor(original, 1)).toThrow(/layer average must equal unity/)

    multipliers[1].value = '1'
    const positions = profiled.querySelectorAll<HTMLInputElement>(
      '[data-defect-field="profile_position"]',
    )
    positions[1].value = '1'
    expect(() => readDeviceEditor(original, 1)).toThrow(/strictly increasing/)
  })

  it('requires a named species for v3 even in effective-lifetime metadata mode', () => {
    const original = explicitConfig()
    renderDeviceEditor(container, original, 'full', 1)
    ;(document.getElementById('layer-1-defect-model') as HTMLSelectElement).value = 'effective_lifetime'
    container.querySelector<HTMLInputElement>('[data-defect-field="name"]')!.value = ''
    const profileEnabled = container.querySelector<HTMLInputElement>(
      '[data-defect-profile-enabled]',
    )!
    profileEnabled.checked = true
    profileEnabled.dispatchEvent(new Event('change', { bubbles: true }))

    expect(() => readDeviceEditor(original, 1)).toThrow(/v3 spatial profile requires a non-empty name/)
  })

  it('preserves unknown v3 nested metadata in a read-only panel', () => {
    const original = distributedConfig('v3')
    const distribution = original.layers[1].bulk_defects![1].distribution as unknown as Record<string, unknown>
    distribution.quadrature_order = 64

    renderDeviceEditor(container, original, 'full', 1)
    expect(container.querySelector('.bulk-defect-readonly')).not.toBeNull()
    expect(document.getElementById('layer-1-defect-enabled')).toBeNull()
    expect(readDeviceEditor(original, 1).layers[1].bulk_defects).toEqual(
      original.layers[1].bulk_defects,
    )
  })
})
