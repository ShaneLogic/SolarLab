import { describe, expect, it } from 'vitest'
import type { BulkDefectInput, DeviceInput, TandemInput } from '../generated/configuration-inputs'
import { cellInput, selections, selectedEntity, writeField, renameConnection } from './device-edits'
import type { DefectSelection, Selection } from './device-edits'
import { changeProfileRows, clearInterfaceDefect, defectSections, removeBulkDefect,
  renameDefect, setDefectSection, writeDefectField } from './defect-edits'

// Declaration-only inputs: no numerical model or default catalog is evaluated.
function device(): DeviceInput {
  return { schema_version: 'solarlab.device-preparation.v1', id: 'test', source_format: 'canonical',
    layers: [{ id: 'a', name: 'A', role: 'HTL', thickness: '1e-7' }, { id: 'b', name: 'B', role: 'ETL', thickness: '2e-7' }],
    contacts: [{ id: 'left', side: 'left', layer: 'a', S_n: -0, S_p: null }, { id: 'right', side: 'right', layer: 'b' }],
    interfaces: [{ id: 'ab', left: 'a', right: 'b', v_n: -0, v_p: 0, defect: {
      id: 'trap', trap_depth_eV: '0.6000000000000000001 eV', energy_reference: 'below_conduction_band',
      total_density_m2: '1e12 cm^-2', calibration_factor: 0, iface_state_calibration_factor: '0.02',
      kinetics: { sigma_n_m2: -0, sigma_p_m2: '1e-19 m^2', thermal_velocity_n_m_s: '1e5 m/s', thermal_velocity_p_m_s: 1e5 },
      partner_metadata: { distribution: 'gaussian', energy_reference: 'below_conduction_band', trap_depth_eV: '0.6', defect_id: 'trap', E_char_eV: '0.1' },
    } }], settings: { phi_left: -0 }, grain_boundaries: [], electrical_grid: [], simulation_hints: null }
}
const connection: Selection = { cell: 'device', kind: 'interface', id: 'ab', occurrence: 0 }
const defect: DefectSelection = { cell: 'device', kind: 'interface_defect', parent: { id: 'ab', occurrence: 0 }, id: 'trap' }

describe('interface and nested-defect draft transformations', () => {
  it('reaches all supplied connections and nested interface defects without writing the input', () => {
    const input = device(), before = structuredClone(input), items = selections(input, 'device')
    expect(items.filter(item => item.kind === 'contact')).toHaveLength(2)
    expect(items.filter(item => item.kind === 'interface')).toHaveLength(1)
    expect(items).toContainEqual(defect)
    expect(selectedEntity(input, defect).path).toEqual(['interfaces', 0, 'defect'])
    expect(input).toStrictEqual(before)
  })
  it('preserves contact number/string/null/omission and does not couple interface edits', () => {
    const input = device(), contact: Selection = { cell: 'device', kind: 'contact', id: 'left', occurrence: 0 }
    expect(writeField(input, contact, 'S_n', { kind: 'value', value: -0 })).toStrictEqual(input)
    const changed = writeField(input, contact, 'S_n', { kind: 'value', value: '20 cm/s' })
    expect(changed.contacts![0].S_n).toBe('20 cm/s')
    expect(changed.contacts![0].S_p).toBeNull()
    const omitted = writeField(changed, contact, 'S_p', { kind: 'omit' })
    expect(Object.hasOwn(omitted.contacts![0], 'S_p')).toBe(false)
    expect(omitted.interfaces).toStrictEqual(input.interfaces)
    const invalidPair = writeField(input, connection, 'v_n', { kind: 'value', value: null })
    expect(invalidPair.interfaces![0].v_n).toBeNull()
    expect(invalidPair.interfaces![0].v_p).toBe(0)
  })
  it('keeps duplicate connection occurrences selectable and changes IDs without rewiring', () => {
    const input = device(); input.interfaces!.push({ id: 'ab', left: 'b', right: 'a' })
    const second: Selection = { ...connection, occurrence: 1 }
    expect(selectedEntity(input, second).path).toEqual(['interfaces', 1])
    const result = renameConnection(input, second, 'second')
    expect(result.input.interfaces![0]).toStrictEqual(input.interfaces![0])
    expect(result.input.interfaces![1]).toStrictEqual({ id: 'second', left: 'b', right: 'a' })
  })
  it('uses generated interface area-density units and never regenerates adjacent data', () => {
    const input = device(), sections = defectSections(input, defect)
    expect(sections[0].schema.properties!.total_density_m2.unit).toBe('m^-2')
    const result = writeDefectField(input, defect, 'defect', 'total_density_m2', { kind: 'value', value: '2e12 cm^-2' })
    const expected = structuredClone(input); expected.interfaces![0].defect!.total_density_m2 = '2e12 cm^-2'
    expect(result).toStrictEqual(expected)
    expect(writeDefectField(input, defect, 'kinetics', 'sigma_n_m2', { kind: 'value', value: -0 })).toStrictEqual(input)
    expect(() => writeDefectField(input, defect, 'defect', 'total_density_m2', { kind: 'omit' })).toThrow('required')
    expect(() => writeDefectField(input, defect, 'defect', 'kinetics', { kind: 'omit' })).toThrow('no scalar')
  })
  it('keeps unknown tagged kinetics/metadata untouched while editing known supplied fields', () => {
    const input = device(), kinetics = input.interfaces![0].defect!.kinetics
    Object.assign(kinetics, { kind: 'future_variant', other_branch: { zero: -0, enabled: false, data: null } })
    const edited = writeDefectField(input, defect, 'kinetics', 'sigma_p_m2', { kind: 'value', value: '3e-19' })
    const expected = structuredClone(input); expected.interfaces![0].defect!.kinetics.sigma_p_m2 = '3e-19'
    expect(edited).toStrictEqual(expected)
    expect(() => renameDefect(input, defect, 'renamed')).toThrow('unrecognized')
  })
  it('renames an interface defect and only its typed partner reference', () => {
    const input = device(), result = renameDefect(input, defect, 'renamed'), expected = structuredClone(input)
    expected.interfaces![0].defect!.id = 'renamed'; expected.interfaces![0].defect!.partner_metadata!.defect_id = 'renamed'
    expect(result.input).toStrictEqual(expected)
    expect(result.selection).toEqual({ ...defect, id: 'renamed' })
    expect(result.updatedPaths).toEqual([['interfaces', 0, 'defect', 'id'], ['interfaces', 0, 'defect', 'partner_metadata', 'defect_id']])
  })
  it('explicitly clears or omits only the selected interface defect', () => {
    const input = device(), cleared = clearInterfaceDefect(input, connection, 'null')
    expect(cleared.interfaces![0].defect).toBeNull()
    const omitted = clearInterfaceDefect(input, connection, 'omit')
    expect(Object.hasOwn(omitted.interfaces![0], 'defect')).toBe(false)
    expect(omitted.contacts).toStrictEqual(input.contacts)
    expect(input.interfaces![0].defect!.id).toBe('trap')
  })
  it.each(['top_cell', 'bottom_cell'] as const)('patches a nested defect only in %s', cell => {
    const input: TandemInput = { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1,
      device_type: 'tandem_2T_monolithic', top_cell: device(), bottom_cell: device(), top_cell_reference: 'top.yaml', bottom_cell_reference: 'bottom.yaml',
      junction_model: 'ideal_ohmic', light_direction: 'top_first', junction_stack: [], benchmark: null, back_reflector: null }
    const selection = { ...defect, cell }, next = writeDefectField(input, selection, 'defect', 'trap_depth_eV', { kind: 'value', value: '9 eV' })
    const expected = structuredClone(input); expected[cell].interfaces![0].defect!.trap_depth_eV = '9 eV'
    expect(next).toStrictEqual(expected)
    expect(cellInput(next, cell).interfaces![0].defect!.trap_depth_eV).toBe('9 eV')
  })
})

function bulk(): BulkDefectInput {
  return { id: 'bulk', name: 'Supplied bulk', charge_transition: 'acceptor', neutral_reference: 'empty', degeneracy: '1',
    distribution: { kind: 'gaussian', normalization: 'integrated_total', total_density_m3: '8e20', center_eV_above_vb: '0.39',
      energy_reference: 'above_valence_band', width_eV: '0.05', width_convention: 'gaussian_standard_deviation', support_width_multiplier: '6' },
    kinetics: { sigma_n_m2: -0, sigma_p_m2: '1e-19', thermal_velocity_n_m_s: '1e5', thermal_velocity_p_m_s: '8e4' },
    spatial_profile: { coordinate: 'normalized_layer_coordinate', interpolation: 'piecewise_linear', density_normalization: 'layer_average_unity',
      knots: [{ position_fraction: -0, density_multiplier: '1.4' }, { position_fraction: '0.5', density_multiplier: '1' }, { position_fraction: 1, density_multiplier: '0.6' }] } }
}
function bulkDevice() {
  const input = device(); input.layers[0].bulk_defects = [bulk()]
  input.layers[0].defect_schema_version = 'solarlab-explicit-bulk-defects-v3'; input.layers[0].defect_model = 'explicit_quasi_steady'
  return input
}
const selectedBulk: DefectSelection = { cell: 'device', kind: 'bulk_defect', parent: { id: 'a', occurrence: 0 }, id: 'bulk', occurrence: 0 }

describe('bulk defect, distribution and profile edits', () => {
  it('exposes the actual bulk density, energy, width and profile metadata without initializing fields', () => {
    const input = bulkDevice(), before = structuredClone(input), sections = defectSections(input, selectedBulk)
    expect(sections.find(section => section.key === 'distribution')!.schema.properties!.total_density_m3.unit).toBe('m^-3')
    expect(sections.find(section => section.key === 'knot:0')!.schema.properties!.position_fraction.unit).toBe('1')
    expect(sections.map(section => section.key)).toContain('energy_level')
    expect(input).toStrictEqual(before)
    expect(writeDefectField(input, selectedBulk, 'knot:0', 'position_fraction', { kind: 'value', value: -0 })).toStrictEqual(input)
  })
  it('retains incompatible branch fields, energy authority and exact unknown payloads after selected edits', () => {
    const input = bulkDevice(), original = input.layers[0].bulk_defects![0] as BulkDefectInput
    Object.assign(original.kinetics, { kind: 'future', other_branch: { flag: false, zero: -0, optional: null } })
    const changed = writeDefectField(input, selectedBulk, 'distribution', 'kind', { kind: 'value', value: 'single_level' })
    const expected = structuredClone(input); (expected.layers[0].bulk_defects![0] as BulkDefectInput).distribution.kind = 'single_level'
    expect(changed).toStrictEqual(expected)
    expect((changed.layers[0].bulk_defects![0] as BulkDefectInput).distribution.width_eV).toBe('0.05')
    expect((changed.layers[0].bulk_defects![0] as BulkDefectInput).kinetics).toStrictEqual(original.kinetics)
    expect((changed.layers[0].bulk_defects![0] as BulkDefectInput).distribution.center_eV_above_vb).toBe('0.39')
  })
  it('creates optional sections empty and distinguishes explicit null from omission', () => {
    const input = bulkDevice(), empty = setDefectSection(input, selectedBulk, 'energy_level', 'empty')
    expect((empty.layers[0].bulk_defects![0] as BulkDefectInput).energy_level).toStrictEqual({})
    expect((empty.layers[0].bulk_defects![0] as BulkDefectInput).distribution).toStrictEqual(bulk().distribution)
    const value = writeDefectField(empty, selectedBulk, 'energy_level', 'value_eV', { kind: 'value', value: '400 meV' })
    expect((value.layers[0].bulk_defects![0] as BulkDefectInput).energy_level!.value_eV).toBe('400 meV')
    const cleared = setDefectSection(value, selectedBulk, 'energy_level', 'null')
    expect((cleared.layers[0].bulk_defects![0] as BulkDefectInput).energy_level).toBeNull()
    const omitted = setDefectSection(cleared, selectedBulk, 'energy_level', 'omit')
    expect(omitted).toStrictEqual(input)
    expect(() => setDefectSection(input, selectedBulk, 'spatial_profile', 'empty')).toThrow('already supplied')
  })
  it('changes and removes exact knot rows without sorting, normalization or affecting other records', () => {
    const input = bulkDevice(), appended = changeProfileRows(input, selectedBulk,
      { kind: 'append', knot: { position_fraction: '', density_multiplier: '' } })
    const appendedDefect = appended.layers[0].bulk_defects![0] as BulkDefectInput
    expect(appendedDefect.spatial_profile!.knots.slice(0, 3)).toStrictEqual(bulk().spatial_profile!.knots)
    expect(appendedDefect.spatial_profile!.knots[3]).toStrictEqual({ position_fraction: '', density_multiplier: '' })
    const changed = writeDefectField(appended, selectedBulk, 'knot:3', 'position_fraction', { kind: 'value', value: '0.1' })
    expect((changed.layers[0].bulk_defects![0] as BulkDefectInput).spatial_profile!.knots[2].position_fraction).toBe(1)
    expect(changeProfileRows(changed, selectedBulk, { kind: 'remove', index: 3 })).toStrictEqual(input)
    expect(input.layers[0].bulk_defects).toHaveLength(1)
  })
  it('renames bulk IDs through partner metadata, retains duplicate drafts and preserves metadata on removal', () => {
    const input = bulkDevice()
    input.layers[0].scaps_defect_metadata = [{ defect_id: 'bulk', distribution: 'gaussian', energy_reference: 'above_valence_band', trap_depth_eV: '0.39' }]
    const renamed = renameDefect(input, selectedBulk, 'renamed')
    expect(renamed.input.layers[0].scaps_defect_metadata![0].defect_id).toBe('renamed')
    expect(renamed.selection).toMatchObject({ id: 'renamed', occurrence: 0 })
    const duplicate = bulkDevice(); duplicate.layers[0].bulk_defects!.push({ ...bulk(), id: 'second' })
    const result = renameDefect(duplicate, selectedBulk, 'second')
    expect(result.input.layers[0].bulk_defects!.map(item => item.id)).toEqual(['second', 'second'])
    expect(selections(result.input, 'device').filter(item => item.kind === 'bulk_defect')).toHaveLength(2)
    const removed = removeBulkDefect(input, selectedBulk)
    expect(removed.layers[0].bulk_defects).toStrictEqual([])
    expect(removed.layers[0].scaps_defect_metadata).toStrictEqual(input.layers[0].scaps_defect_metadata)
  })
  it('keeps multivalent family and per-transition kinetics separate from ordinary defects', () => {
    const input = bulkDevice(); input.layers[0].bulk_defects = [{ id: 'bulk', name: 'Multivalent', total_density_m3: '2e21', configuration: {
      family: 'double_donor', charge_states_e: [2, 1, 0], degeneracy_convention: 'scaps_binomial', state_degeneracies: [1, '2', 1],
      energy_levels: { first_transition_eV_above_vb: '0.3', correlation_energies_eV: ['-0.1'], energy_reference: 'above_valence_band' },
      transition_kinetics: [bulk().kinetics, { ...bulk().kinetics, sigma_n_m2: '1e-19' }],
    } }]
    const sections = defectSections(input, selectedBulk)
    expect(sections.map(section => section.key)).toContain('transition:1')
    expect(sections.map(section => section.key)).not.toContain('distribution')
    const changed = writeDefectField(input, selectedBulk, 'transition:1', 'sigma_n_m2', { kind: 'value', value: '0' })
    const expected = structuredClone(input)
    const item = expected.layers[0].bulk_defects![0]; if ('configuration' in item) item.configuration.transition_kinetics[1].sigma_n_m2 = '0'
    expect(changed).toStrictEqual(expected)
    expect(() => writeDefectField(input, selectedBulk, 'distribution', 'kind', { kind: 'value', value: 'gaussian' })).toThrow('no scalar')
    Object.assign(input.layers[0].bulk_defects![0], { distribution: bulk().distribution })
    expect(defectSections(input, selectedBulk)[0].key).toBe('retained')
  })
})
