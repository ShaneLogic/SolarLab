import { describe, expect, it } from 'vitest'
import { configurationSchema } from '../generated/configuration-schema'
import type { DeviceInput, TandemInput } from '../generated/configuration-inputs'
import {
  cellInput, DeviceEditError, exactInputEqual, layerReferences, moveLayer, renameLayer,
  selections, writeConnection, writeField,
} from './device-edits'
import type { Selection } from './device-edits'

// Small declaration fixtures only; no model or physical defaults are evaluated.
function device(): DeviceInput {
  return {
    schema_version: 'solarlab.device-preparation.v1', id: 'device', source_format: 'canonical',
    settings: { phi_left: -0, flat_band_contacts: false },
    layers: [
      { id: 'front', name: 'Front', role: 'HTL', thickness: '2.00000000000000001e-7 m', material: 'shared',
        parameters: { mu_n: -0, Nc300: null, incoherent: false }, bulk_defects: [{
          id: 'front', name: 'front', distribution: { kind: 'single_level', normalization: 'integrated_total', total_density_m3: '1e18' },
          charge_transition: 'unresolved', neutral_reference: 'unresolved',
          kinetics: { sigma_n_m2: '1e-19', sigma_p_m2: '1e-19', thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: 1e5 },
          degeneracy: 1, energy_level: { reference: 'below_conduction_band', value_eV: '0.1' },
        }], cigs_graded_optics: null, metastable_document: null },
      { id: 'middle', name: 'Middle', role: 'absorber', thickness: 5e-7, parameters: {} },
      { id: 'back', name: 'Back', role: 'ETL', thickness: 1e-7 },
    ],
    materials: [{ id: 'shared', name: 'Shared', parameters: { mu_n: '2 cm^2/(V s)' }, cigs_graded_optics: null }],
    interfaces: [{ id: 'fm', left: 'front', right: 'middle', v_n: null }, { id: 'mb', left: 'middle', right: 'back' }],
    contacts: [{ id: 'left', side: 'left', layer: 'front', S_n: 0 }, { id: 'right', side: 'right', layer: 'back' }],
    electrical_grid: [{ layer: 'front', interval_weight: 1, alpha: '2' }],
    grain_boundaries: [{ id: 'grain', layer_ids: ['front', 'middle'], width: '1 nm', x_position: 0, tau_n: '1 ns', tau_p: '1 ns' }],
    simulation_hints: { notes: 'front' },
  }
}
function tandem(): TandemInput {
  return { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1,
    device_type: 'tandem_2T_monolithic', top_cell: device(), bottom_cell: { ...device(), id: 'bottom' },
    top_cell_reference: 'top.yaml', bottom_cell_reference: 'bottom.yaml', junction_model: 'ideal_ohmic',
    light_direction: 'top_first', junction_stack: [], benchmark: null, back_reflector: null }
}
const front: Selection = { cell: 'device', kind: 'layer', id: 'front', occurrence: 0 }

describe('copied input transformations', () => {
  it('preserves exact no-op numbers, strings, null, empty and omitted containers', () => {
    const input = device(), before = structuredClone(input)
    expect(writeField(input, front, 'mu_n', { kind: 'value', value: -0 }, true)).toStrictEqual(input)
    expect(writeField(input, front, 'thickness', { kind: 'value', value: input.layers[0].thickness })).toStrictEqual(input)
    const back: Selection = { ...front, id: 'back' }
    expect(writeField(input, back, 'mu_n', { kind: 'omit' }, true)).toStrictEqual(input)
    expect(Object.hasOwn(writeField(input, back, 'mu_n', { kind: 'omit' }, true).layers[2], 'parameters')).toBe(false)
    expect(writeField(input, { ...front, id: 'middle' }, 'mu_n', { kind: 'omit' }, true).layers[1].parameters).toStrictEqual({})
    expect(input).toStrictEqual(before)
    expect(exactInputEqual(0, -0)).toBe(false)
    expect(exactInputEqual({}, { description: null })).toBe(false)
  })

  it('patches a display name without changing IDs or sharing mutable response data', () => {
    const input = device(), expected = structuredClone(input)
    expected.layers[0].name = 'New display name'
    const edited = writeField(input, front, 'name', { kind: 'value', value: 'New display name' })
    expect(edited).toStrictEqual(expected)
    edited.layers[0].parameters!.mu_n = '5'
    expect(Object.is(input.layers[0].parameters!.mu_n, -0)).toBe(true)
    expect(edited.layers[0].bulk_defects![0].id).toBe('front')
  })

  it('removes only a local override and edits named material values independently', () => {
    const input = device(), expected = structuredClone(input)
    delete expected.layers[0].parameters!.mu_n
    expect(writeField(input, front, 'mu_n', { kind: 'omit' }, true)).toStrictEqual(expected)
    const material: Selection = { cell: 'device', kind: 'material', id: 'shared', occurrence: 0 }
    const next = writeField(input, material, 'mu_n', { kind: 'value', value: '0' }, true)
    expect(next.layers).toStrictEqual(input.layers)
    expect(next.materials![0].parameters.mu_n).toBe('0')
    expect(next.materials![0].id).toBe('shared')
  })

  it('uses generated required/nullability policy and protects identity and complex fields from scalar patches', () => {
    expect(() => writeField(device(), front, 'thickness', { kind: 'omit' })).toThrow(DeviceEditError)
    expect(() => writeField(device(), front, 'mu_n', { kind: 'value', value: null }, true)).toThrow('does not accept null')
    expect(() => writeField(device(), front, 'id', { kind: 'value', value: 'new' })).toThrow('no scalar editor')
    expect(() => writeField(device(), front, 'bulk_defects', { kind: 'omit' })).toThrow('no scalar editor')
    expect(writeField(device(), front, 'Nc300', { kind: 'value', value: null }, true).layers[0].parameters!.Nc300).toBeNull()
  })

  it('rewrites every current typed layer reference, preserving nested IDs, names and physical content', () => {
    const input = device(), expected = structuredClone(input), result = renameLayer(input, front, 'renamed')
    expected.layers[0].id = 'renamed'; expected.interfaces![0].left = 'renamed'
    expected.contacts![0].layer = 'renamed'; expected.electrical_grid![0].layer = 'renamed'
    expected.grain_boundaries![0].layer_ids[0] = 'renamed'
    expect(result.input).toStrictEqual(expected)
    expect(result.selection).toStrictEqual({ ...front, id: 'renamed' })
    expect(result.updatedPaths).toHaveLength(5)
    expect(input.layers[0].id).toBe('front')
    expect(result.input.layers[0].bulk_defects![0].id).toBe('front')
    expect(result.input.simulation_hints!.notes).toBe('front')
    expect(layerReferences(result.input).some(ref => ref.value === 'front')).toBe(false)
  })

  it('guards the reviewed layer-reference coverage against new schema reference fields', () => {
    const defs = configurationSchema.dto_schemas.DeviceInput.schema.$defs
    const observed: string[] = []
    for (const [name, schema] of Object.entries(defs)) {
      for (const field of Object.keys(schema.properties)) if (['layer', 'layer_ids', 'left', 'right'].includes(field)) observed.push(`${name}.${field}`)
    }
    const currentReferences = ['ContactInput.layer', 'GrainBoundaryInput.layer_ids', 'GridLayerInput.layer', 'InterfaceInput.left', 'InterfaceInput.right']
    // These IDs describe the original source bytes, so current layer edits must preserve them.
    const retainedHistory = ['LegacyDeviceFieldsInput.layer_ids']
    expect(observed.sort()).toEqual([...currentReferences, ...retainedHistory].sort())
  })

  it('preserves original legacy layer IDs and source evidence when renaming a current layer', () => {
    const input = device()
    input.legacy_fields = {
      schema_version: 'solarlab.standard-loader-fields.v1', source_id: 'original', source_sha256: 'a'.repeat(64),
      layer_ids: ['front', 'middle', 'back'], temperature: 0, interfaces: [[0, '1 m/s']],
    }
    const original = structuredClone(input), result = renameLayer(input, front, 'renamed')
    expect(result.input.layers[0].id).toBe('renamed')
    expect(result.input.contacts![0].layer).toBe('renamed')
    expect(result.input.legacy_fields).toStrictEqual(original.legacy_fields)
    expect(result.updatedPaths.some(path => path.includes('legacy_fields'))).toBe(false)
    expect(input).toStrictEqual(original)
  })

  it('rejects unknown extension references with exact paths before any ID mutation', () => {
    const input = { ...device(), future: { layer: 'front' } }
    try { renameLayer(input, front, 'renamed'); throw new Error('expected rejection') }
    catch (error) { expect(error).toBeInstanceOf(DeviceEditError); expect((error as DeviceEditError).paths).toEqual([['future']]) }
    expect(input.layers[0].id).toBe('front')
    const nested = device()
    Object.assign(nested.layers[0].parameters!, { future_link: { layer: 'front' } })
    expect(() => renameLayer(nested, front, 'renamed')).toThrow('unrecognized fields')
  })

  it('retains an explicit duplicate-ID draft and refuses an ambiguous subsequent rewrite with reference paths', () => {
    const duplicate = renameLayer(device(), front, 'middle')
    expect(duplicate.input.layers.map(layer => layer.id)).toEqual(['middle', 'middle', 'back'])
    expect(selections(duplicate.input, 'device').filter(item => item.kind === 'layer')).toHaveLength(3)
    try { renameLayer(duplicate.input, duplicate.selection, 'fixed'); throw new Error('expected rejection') }
    catch (error) {
      expect(error).toBeInstanceOf(DeviceEditError)
      expect((error as DeviceEditError).paths).toContainEqual(['interfaces', 0, 'left'])
    }
  })

  it('moves exact layers and preserves invalid connection topology for explicit correction', () => {
    const input = device(), result = moveLayer(input, front, 1)
    expect(result.input.layers).toStrictEqual([input.layers[1], input.layers[0], input.layers[2]])
    expect(result.input.interfaces).toStrictEqual(input.interfaces)
    expect(result.input.contacts).toStrictEqual(input.contacts)
    expect(result.input.electrical_grid).toStrictEqual(input.electrical_grid)
    expect(result.selection).toStrictEqual(front)
    const dangling = writeConnection(result.input, 'device', 'interfaces', 0, 'right', 'missing')
    expect(dangling.interfaces![0].right).toBe('missing')
    expect(dangling.interfaces![0].v_n).toBeNull()
    expect(() => writeConnection(input, 'device', 'contacts', 0, 'left', 'back')).toThrow('supplied connection')
  })

  it.each(['top_cell', 'bottom_cell'] as const)('edits %s without rewriting its sibling, optical, junction or benchmark data', cell => {
    const input = tandem(), selected: Selection = { ...front, cell }
    const result = renameLayer(input, selected, 'renamed')
    const sibling = cell === 'top_cell' ? 'bottom_cell' : 'top_cell'
    expect(result.input[sibling]).toStrictEqual(input[sibling])
    const expected = structuredClone(input); expected[cell] = result.input[cell]
    expect(result.input).toStrictEqual(expected)
    expect(result.updatedPaths.every(path => path[0] === cell)).toBe(true)
    expect(cellInput(result.input, cell).layers[0].id).toBe('renamed')
    expect(() => cellInput(result.input, 'device')).toThrow('Select a cell')
  })
})
