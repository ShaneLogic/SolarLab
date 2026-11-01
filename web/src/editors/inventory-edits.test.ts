import { describe, expect, it } from 'vitest'
import type { DeviceInput, MetastableDocumentInput, MultivalentDefectInput, TandemInput } from '../generated/configuration-inputs'
import { movedRow, renameMaterial, selectedEntity, selections } from './device-edits'
import type { DefectSelection, Selection } from './device-edits'
import { changeDefectRows, defectSections, renameDefect, writeDefectField, writeDefectRow } from './defect-edits'
import { createRecord, recordKind, removeRecord, setLayerDocument } from './inventory-edits'

function configuration(): MultivalentDefectInput['configuration'] {
  return { family: 'double_donor', charge_states_e: [2, 1, -0], degeneracy_convention: 'explicit', state_degeneracies: [1, '2.00000000000000001', 1],
    energy_levels: { first_transition_eV_above_vb: '300 meV', correlation_energies_eV: ['-0.1 eV'], energy_reference: 'above_valence_band' },
    transition_kinetics: [{ sigma_n_m2: -0, sigma_p_m2: '1e-19 m^2', thermal_velocity_n_m_s: '1e5 m/s', thermal_velocity_p_m_s: 1e5 },
      { sigma_n_m2: '2e-19 m^2', sigma_p_m2: 0, thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: '8e4 m/s' }] }
}
function device(): DeviceInput {
  const multivalent: MultivalentDefectInput = { id: 'multi', name: 'Multivalent', total_density_m3: '1e18 cm^-3', configuration: configuration() }
  const metastable: MetastableDocumentInput = { schema_version: 'solarlab-metastable-bulk-defects-v1', defect_model: 'explicit_metastable_frozen', metastable_defects: [{
    id: 'meta', name: 'Metastable', total_density_m3: 3e21, donor_configuration: configuration(), acceptor_configuration: configuration(),
    donor_conversion_state_index: 0, acceptor_conversion_state_index: 2,
    conversion_kinetics: { transition_energy_eV_above_vb: '0.19 eV', electron_capture_activation_eV: '0.1', electron_emission_activation_eV: '0.76',
      hole_capture_activation_eV: '0.35', hole_emission_activation_eV: '0.73', electron_capture_path: 'electron_capture_plus_hole_emission', hole_capture_path: 'double_hole_capture',
      capture_n_m3_s: '2e-14', capture_p_m3_s: '3e-14', phonon_frequency_Hz: '1e13' },
  }] }
  return { schema_version: 'solarlab.device-preparation.v1', id: 'test', source_format: 'canonical',
    layers: [{ id: 'a', name: 'A', role: 'HTL', thickness: '1e-7', material: 'mat', parameters: { mu_n: -0, incoherent: false, Nc300: null },
      bulk_defects: [multivalent], metastable_document: metastable, metastable_preparation: null, cigs_graded_optics: null },
    { id: 'b', name: 'B', role: 'ETL', thickness: 2e-7 }],
    materials: [{ id: 'mat', name: 'Material', parameters: { mu_n: '2 cm^2/(V s)' } }],
    contacts: [{ id: 'left', side: 'left', layer: 'a', S_n: -0, S_p: null }],
    interfaces: [{ id: 'ab', left: 'a', right: 'b', v_n: null }], grain_boundaries: [], simulation_hints: null }
}
const layer: Selection = { cell: 'device', kind: 'layer', id: 'a', occurrence: 0 }
const multi: DefectSelection = { cell: 'device', kind: 'bulk_defect', parent: { id: 'a', occurrence: 0 }, id: 'multi', occurrence: 0 }
const meta: DefectSelection = { ...multi, kind: 'metastable_defect', id: 'meta' }
const material: Selection = { cell: 'device', kind: 'material', id: 'mat', occurrence: 0 }

describe('explicit structured record creation and removal', () => {
  it('starts empty declarations without physical values, tags or inherited object regeneration', () => {
    const input = device(), before = structuredClone(input)
    expect(selectedEntity(createRecord(input, layer, 'layer', 'new').input, { ...layer, id: 'new' }).value).toStrictEqual({ id: 'new' })
    const created = createRecord(input, layer, 'multivalent_defect', 'empty')
    expect(selectedEntity(created.input, created.selection).value).toStrictEqual({ id: 'empty', configuration: {
      charge_states_e: [], state_degeneracies: [], energy_levels: { correlation_energies_eV: [] }, transition_kinetics: [] } })
    expect(created.input.layers[0].metastable_document).toStrictEqual(input.layers[0].metastable_document)
    expect(input).toStrictEqual(before)
    expect(() => createRecord(input, layer, 'layer', '')).toThrow('explicit ID')
  })
  it.each(['layer', 'material', 'contact', 'interface', 'multivalent_defect', 'metastable_defect'] as const)('copies a %s exactly except its explicit new ID', kind => {
    const input = device(), source = selections(input, 'device').find(item => recordKind(input, item) === kind)!
    const original = selectedEntity(input, source).value!
    Object.assign(original, { unknown_history: [{ id: 'a', zero: -0, enabled: false, data: null }] })
    const created = createRecord(input, layer, kind, 'copy', source)
    expect(selectedEntity(created.input, created.selection).value).toStrictEqual({ ...original, id: 'copy' })
    expect(selectedEntity(created.input, source).value).toStrictEqual(original)
    expect(created.input.contacts!.slice(0, 1)).toStrictEqual(input.contacts)
    Reflect.set(selectedEntity(created.input, created.selection).value!, 'name', 'isolated')
    expect(Reflect.get(original, 'name')).not.toBe('isolated')
  })
  it('keeps duplicate IDs as distinct selectable occurrences without repairing references', () => {
    const input = device(), result = createRecord(input, layer, 'layer', 'a', layer)
    expect(result.selection).toStrictEqual({ ...layer, occurrence: 1 })
    expect(result.input.layers[2]).toStrictEqual(input.layers[0])
    expect(result.input.interfaces).toStrictEqual(input.interfaces)
    expect(result.input.contacts).toStrictEqual(input.contacts)
    expect(removeRecord(result.input, result.selection).input).toStrictEqual(input)
  })
  it('retains dangling references and an explicit empty inventory after removal', () => {
    const input = device(), result = removeRecord(input, material)
    expect(result.input.materials).toStrictEqual([])
    expect(result.input.layers[0].material).toBe('mat')
    const removedLayer = removeRecord(input, layer)
    expect(removedLayer.input.contacts).toStrictEqual(input.contacts)
    expect(removedLayer.input.interfaces).toStrictEqual(input.interfaces)
    const last = removeRecord(input, meta)
    expect(last.input.layers[0].metastable_document!.metastable_defects).toStrictEqual([])
    expect(last.input.layers[0].metastable_preparation).toBeNull()
  })
  it('can explicitly remove an unsupported mixed record without converting or deleting partner data', () => {
    const input = device(); Object.assign(input.layers[0].bulk_defects![0], { distribution: { unknown: -0 } })
    input.layers[0].scaps_defect_metadata = [{ defect_id: 'multi', distribution: 'single', energy_reference: 'above_valence_band', trap_depth_eV: '0.3' }]
    expect(recordKind(input, multi)).toBeUndefined()
    const removed = removeRecord(input, multi)
    expect(removed.input.layers[0].bulk_defects).toStrictEqual([])
    expect(removed.input.layers[0].scaps_defect_metadata).toStrictEqual(input.layers[0].scaps_defect_metadata)
  })
  it('creates an interface defect explicitly and never overwrites an existing one', () => {
    const input = device(), target: Selection = { cell: 'device', kind: 'interface', id: 'ab', occurrence: 0 }
    const empty = createRecord(input, target, 'interface_defect', 'new')
    expect(empty.input.interfaces![0].defect).toStrictEqual({ id: 'new', kinetics: {} })
    expect(input.interfaces![0].defect).toBeUndefined()
    expect(() => createRecord(empty.input, target, 'interface_defect', 'another')).toThrow('already has')
    expect(removeRecord(empty.input, empty.selection).input).toStrictEqual(input)
  })
  it('never synthesizes a metastable document while adding a defect and distinguishes document null/omission', () => {
    const input = device(); delete input.layers[0].metastable_document
    expect(() => createRecord(input, layer, 'metastable_defect', 'new')).toThrow('inventory declaration')
    const empty = setLayerDocument(input, 'device', { id: 'a', occurrence: 0 }, 'metastable_document', 'empty')
    expect(empty.layers[0].metastable_document).toStrictEqual({ metastable_defects: [] })
    expect(() => setLayerDocument(empty, 'device', { id: 'a', occurrence: 0 }, 'metastable_document', 'empty')).toThrow('already supplied')
    const cleared = setLayerDocument(empty, 'device', { id: 'a', occurrence: 0 }, 'metastable_document', 'null')
    expect(cleared.layers[0].metastable_document).toBeNull()
    expect(setLayerDocument(cleared, 'device', { id: 'a', occurrence: 0 }, 'metastable_document', 'omit')).toStrictEqual(input)
  })
  it('changes only typed material references and rejects ambiguous or unknown identity scopes', () => {
    const input = device(), changed = renameMaterial(input, material, 'renamed')
    expect(changed.input.layers[0].material).toBe('renamed')
    expect(changed.input.layers[0].bulk_defects).toStrictEqual(input.layers[0].bulk_defects)
    const duplicate = createRecord(input, layer, 'material', 'mat', material)
    expect(() => renameMaterial(duplicate.input, duplicate.selection, 'renamed')).toThrow('ambiguous')
    Object.assign(input, { extra: { material: 'mat' } })
    expect(() => renameMaterial(input, material, 'renamed')).toThrow('unrecognized')
  })
  it.each(['top_cell', 'bottom_cell'] as const)('limits creation/removal to %s and preserves sibling and root declarations', cell => {
    const input: TandemInput = { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1,
      device_type: 'tandem_2T_monolithic', top_cell: device(), bottom_cell: device(), top_cell_reference: 'top', bottom_cell_reference: 'bottom',
      junction_model: 'ideal_ohmic', light_direction: 'top_first', junction_stack: [], benchmark: null, back_reflector: null }
    const result = createRecord(input, { ...layer, cell }, 'metastable_defect', 'copy', { ...meta, cell })
    const expected = structuredClone(input); expected[cell].layers[0].metastable_document!.metastable_defects.push({ ...structuredClone(input[cell].layers[0].metastable_document!.metastable_defects[0]), id: 'copy' })
    expect(result.input).toStrictEqual(expected)
    expect(removeRecord(result.input, result.selection).input).toStrictEqual(input)
  })
})

describe('ordered complex defect arrays', () => {
  it('derives item units and integer semantics from the generated schema', () => {
    const sections = defectSections(device(), multi)
    expect(sections.find(item => item.key === 'charges')!.array!.item.type).toBe('integer')
    expect(sections.find(item => item.key === 'degeneracies')!.array!.item.unit).toBe('1')
    expect(sections.find(item => item.key === 'correlations')!.array!.item.unit).toBe('eV')
  })
  it('preserves exact no-op words and changes one array without coupling any other array', () => {
    const input = device()
    expect(writeDefectRow(input, multi, 'charges', 2, { kind: 'value', value: -0 })).toStrictEqual(input)
    const changed = writeDefectRow(input, multi, 'correlations', 0, { kind: 'value', value: '-150.00000000000000001 meV' })
    const expected = structuredClone(input), item = expected.layers[0].bulk_defects![0] as MultivalentDefectInput
    item.configuration.energy_levels.correlation_energies_eV[0] = '-150.00000000000000001 meV'
    expect(changed).toStrictEqual(expected)
    expect(() => writeDefectRow(input, multi, 'charges', 0, { kind: 'omit' })).toThrow('explicit value')
  })
  it('inserts/removes/reorders only chosen rows and keeps invalid lengths and charge ordering', () => {
    const input = device(), inserted = changeDefectRows(input, multi, 'charges', { kind: 'insert', index: 1 }, -2)
    expect((inserted.layers[0].bulk_defects![0] as MultivalentDefectInput).configuration.charge_states_e).toStrictEqual([2, -2, 1, -0])
    expect(changeDefectRows(inserted, multi, 'charges', { kind: 'remove', index: 1 })).toStrictEqual(input)
    const moved = changeDefectRows(input, multi, 'transitions', { kind: 'move', from: 0, to: 1 })
    const expected = structuredClone(input), item = expected.layers[0].bulk_defects![0] as MultivalentDefectInput
    item.configuration.transition_kinetics.reverse()
    expect(moved).toStrictEqual(expected)
    expect(() => changeDefectRows(input, multi, 'charges', { kind: 'move', from: 0, to: 9 })).toThrow('no longer')
  })
  it('maps pending row indices across insertion, movement and explicit deletion', () => {
    expect([0, 1, 2].map(index => movedRow(index, { kind: 'move', from: 0, to: 2 }))).toStrictEqual([2, 0, 1])
    expect([0, 1, 2].map(index => movedRow(index, { kind: 'move', from: 2, to: 0 }))).toStrictEqual([1, 2, 0])
    expect([0, 1, 2].map(index => movedRow(index, { kind: 'remove', index: 1 }))).toStrictEqual([0, undefined, 1])
  })
  it('edits both metastable configurations independently and retains conversion indices and barriers', () => {
    const input = device(), expected = structuredClone(input)
    const changed = writeDefectRow(input, meta, 'donor:charges', 0, { kind: 'value', value: 3 })
    expected.layers[0].metastable_document!.metastable_defects[0].donor_configuration.charge_states_e[0] = 3
    expect(changed).toStrictEqual(expected)
    const unit = writeDefectField(input, meta, 'conversion', 'capture_n_m3_s', { kind: 'value', value: '2e-8 cm^3/s' })
    expect(unit.layers[0].metastable_document!.metastable_defects[0].conversion_kinetics.capture_n_m3_s).toBe('2e-8 cm^3/s')
    expect(renameDefect(input, meta, 'new').input.layers[0].metastable_document!.metastable_defects[0].id).toBe('new')
    expect(selectedEntity(input, meta).path).toStrictEqual(['layers', 0, 'metastable_document', 'metastable_defects', 0])
  })
})
