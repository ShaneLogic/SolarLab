import { describe, expect, it } from 'vitest'
import type { DeviceInput, TandemInput } from '../generated/configuration-inputs'
import { exactInputEqual, renameLayer, selectedEntity, selections } from './device-edits'
import type { ExtendedSelection } from './device-edits'
import { changeExtendedRows, extendedReferenceNotices, extendedSections, setExtendedSection, writeExtendedField, writeExtendedRow } from './extended-edits'
import { removeRecord } from './inventory-edits'

function device(): DeviceInput {
  return { schema_version: 'solarlab.device-preparation.v1', id: 'sample', source_format: 'canonical',
    layers: [{ id: 'front', name: 'Front', role: 'absorber', thickness: '100 nm', parameters: { mu_n: -0, incoherent: false },
      material: 'cigs', cigs_graded_optics: { ggi_front: -0, ggi_back: '0.250000000000000001', cgi: 0.9 } },
    { id: 'back', name: 'Back', role: 'ETL', thickness: 1e-7 }],
    materials: [{ id: 'cigs', name: 'Named material', parameters: {}, cigs_graded_optics: { ggi_front: 0, ggi_back: 0.2, cgi: 0.9 } }],
    grain_boundaries: [{ id: 'band', layer_ids: ['front', 'back'], x_position: '1 um', width: '20 nm', tau_n: '1 ns', tau_p: 1e-9, source_layer_role: null }],
    electrical_grid: [{ layer: 'front', interval_weight: '1', alpha: 2 }, { layer: 'back', interval_weight: 2, alpha: '2' }],
    tunnelling_channels: { intraband: { enabled: false }, contact: { barrier_height_eV: -0 } }, spectrum: null }
}
function tandem(): TandemInput {
  return { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1, device_type: 'tandem_2T_monolithic',
    top_cell: device(), bottom_cell: { ...device(), id: 'bottom' }, top_cell_reference: 'top-source', bottom_cell_reference: 'bottom-source',
    junction_model: 'ideal_ohmic', light_direction: 'top_first', junction_stack: [
      { id: 'a', name: 'A', thickness: '2 nm', optical_material: 'nk', incoherent: false },
      { id: 'b', name: 'B', thickness: 3e-9, optical_material: 'nk', incoherent: true }], back_reflector: null, benchmark: null }
}
const optics: ExtendedSelection = { cell: 'device', kind: 'layer_optics', id: 'front', occurrence: 0 }
const micro: ExtendedSelection = { cell: 'device', kind: 'microstructure' }
const tunnel: ExtendedSelection = { cell: 'device', kind: 'tunnelling' }
const stack: ExtendedSelection = { cell: 'top_cell', kind: 'optical_stack' }

describe('structured optical, tunnelling and microstructure edits', () => {
  it('selects every declaration without converting input or conflating optical and physical layers', () => {
    const input = tandem(), before = structuredClone(input)
    for (const cell of ['top_cell', 'bottom_cell'] as const) for (const selected of selections(input, cell)) expect(selectedEntity(input, selected)).toHaveProperty('path')
    expect(extendedSections(input, stack).map(item => item.key)).toEqual(['stack', 'junction', 'optical:0', 'optical:1', 'reflector'])
    expect(exactInputEqual(input, before)).toBe(true)
    expect(() => selectedEntity(device(), { ...stack, cell: 'device' })).toThrow(/tandem/)
  })
  it('changes one quantity without coercing decimal/unit strings, signed zero or unrelated unknown fields', () => {
    const input = device(); Reflect.set(input.layers[0], 'extension', { history: [false, null, -0, '1.00000000000000001'] })
    const next = writeExtendedField(input, optics, 'cigs', 'ggi_front', { kind: 'value', value: '0.00000000000000001' })
    expect(next.layers[0].cigs_graded_optics!.ggi_front).toBe('0.00000000000000001')
    expect(Object.is(input.layers[0].cigs_graded_optics!.ggi_front, -0)).toBe(true)
    const expected = structuredClone(input); expected.layers[0].cigs_graded_optics!.ggi_front = '0.00000000000000001'
    expect(next).toStrictEqual(expected); expect(next.layers[0]).not.toBe(input.layers[0])
    expect(exactInputEqual(writeExtendedField(input, optics, 'cigs', 'model', { kind: 'omit' }), input)).toBe(true)
  })
  it('keeps explicit clear separate from omission and refuses null or omission on required CIGS scalars', () => {
    const input = device(), cleared = setExtendedSection(input, optics, 'cigs', 'null')
    expect(cleared.layers[0].cigs_graded_optics).toBeNull()
    const inherited = setExtendedSection(cleared, optics, 'cigs', 'omit')
    expect(Object.hasOwn(inherited.layers[0], 'cigs_graded_optics')).toBe(false)
    expect(inherited.materials).toStrictEqual(input.materials)
    expect(() => writeExtendedField(input, optics, 'cigs', 'model', { kind: 'value', value: null })).toThrow(/null/)
    expect(() => writeExtendedField(input, optics, 'cigs', 'cgi', { kind: 'omit' })).toThrow(/required/)
  })
  it('starts only explicit empty declarations and copies actual same-cell CIGS input', () => {
    const input = setExtendedSection(device(), optics, 'cigs', 'omit')
    const empty = setExtendedSection(input, optics, 'cigs', 'empty')
    expect(empty.layers[0].cigs_graded_optics).toStrictEqual({})
    expect(() => setExtendedSection(empty, optics, 'cigs', 'empty')).toThrow(/retained/)
    const source: ExtendedSelection = { ...optics, kind: 'material_optics', id: 'cigs' }
    const copied = setExtendedSection(input, optics, 'cigs', 'empty', { selection: source, key: 'cigs' })
    expect(copied.layers[0].cigs_graded_optics).toStrictEqual(input.materials![0].cigs_graded_optics)
    expect(copied.layers[0].cigs_graded_optics).not.toBe(copied.materials![0].cigs_graded_optics)
  })
  it('edits four independent channel contracts with false, signed zero and inherited omitted controls', () => {
    const input = device()
    const zero = writeExtendedField(input, tunnel, 'contact', 'barrier_height_eV', { kind: 'value', value: '0 meV' })
    expect(zero.tunnelling_channels!.intraband).toStrictEqual({ enabled: false })
    expect(Object.is(input.tunnelling_channels!.contact!.barrier_height_eV, -0)).toBe(true)
    const inherited = writeExtendedField(zero, tunnel, 'intraband', 'enabled', { kind: 'omit' })
    expect(inherited.tunnelling_channels!.intraband).toStrictEqual({})
    expect(() => writeExtendedField(input, tunnel, 'intraband', 'enabled', { kind: 'value', value: null })).toThrow(/null/)
    expect(extendedSections(input, tunnel).map(item => item.key)).toEqual(['tunnelling', 'band_to_band', 'intraband', 'interface_defect_assisted', 'contact'])
    const absent = setExtendedSection(input, tunnel, 'tunnelling', 'omit')
    expect(() => setExtendedSection(absent, tunnel, 'contact', 'empty')).toThrow(/containing/)
    expect(setExtendedSection(input, tunnel, 'intraband', 'null').tunnelling_channels!.intraband).toBeNull()
  })
  it('preserves ordered optical contents, duplicate IDs and both complete cells on copy/move/remove', () => {
    const input = tandem(); Reflect.set(input.junction_stack[0], 'extension', { unknown: -0 })
    const copied = changeExtendedRows(input, stack, 'junction', { kind: 'insert', index: 1 }, 0)
    expect(copied.junction_stack[1]).toStrictEqual(input.junction_stack[0])
    expect(copied.junction_stack[1]).not.toBe(copied.junction_stack[0])
    expect(extendedReferenceNotices(copied, stack)).toContainEqual({ path: ['junction_stack', 1, 'id'], message: 'Duplicate optical-layer ID a' })
    const moved = changeExtendedRows(copied, stack, 'junction', { kind: 'move', from: 2, to: 0 })
    expect(moved.junction_stack.map(row => row.id)).toEqual(['b', 'a', 'a'])
    const removed = changeExtendedRows(moved, stack, 'junction', { kind: 'remove', index: 1 })
    expect(removed.top_cell).toStrictEqual(input.top_cell); expect(removed.bottom_cell).toStrictEqual(input.bottom_cell)
    expect(removed.benchmark).toBeNull(); expect(removed.top_cell_reference).toBe(input.top_cell_reference)
    expect(() => setExtendedSection(input, stack, 'junction', 'omit')).toThrow(/presence/)
  })
  it('creates a blank optical row or explicitly copies a reflector without choosing values', () => {
    const input = tandem(), empty = changeExtendedRows(input, stack, 'junction', { kind: 'insert', index: 0 })
    expect(empty.junction_stack[0]).toStrictEqual({})
    const copy = setExtendedSection(input, stack, 'reflector', 'empty', { selection: stack, key: 'optical:0' })
    expect(copy.back_reflector).toStrictEqual(input.junction_stack[0])
    expect(extendedReferenceNotices(copy, stack)).toContainEqual({ path: ['back_reflector', 'id'], message: 'Duplicate optical-layer ID a' })
  })
  it('retains invalid grain/grid reference drafts and does not cascade-delete their references', () => {
    const input = device(), edited = writeExtendedRow(input, micro, 'grain-layers:0', 1, { kind: 'value', value: 'missing' })
    expect(extendedReferenceNotices(edited, micro)[0].path).toEqual(['grain_boundaries', 0, 'layer_ids', 1])
    const removed = removeRecord(input, { cell: 'device', kind: 'layer', id: 'front', occurrence: 0 }).input
    expect(removed.grain_boundaries).toStrictEqual(input.grain_boundaries)
    expect(removed.electrical_grid).toStrictEqual(input.electrical_grid)
    expect(extendedReferenceNotices(removed, micro).map(item => item.path)).toEqual([
      ['grain_boundaries', 0, 'layer_ids', 0], ['electrical_grid', 0, 'layer']])
  })
  it('updates known grid and grain references on explicit physical-layer rename without touching source role', () => {
    const input = device(), renamed = renameLayer(input, { cell: 'device', kind: 'layer', id: 'front', occurrence: 0 }, 'renamed').input
    expect(renamed.grain_boundaries![0].layer_ids).toEqual(['renamed', 'back'])
    expect(renamed.electrical_grid![0].layer).toBe('renamed')
    expect(renamed.grain_boundaries![0].source_layer_role).toBeNull()
  })
  it('keeps optional lists omitted until explicit creation and preserves independent scalar reference rows', () => {
    const input = device(); delete input.grain_boundaries
    const absent = setExtendedSection(input, micro, 'grains', 'omit')
    expect(Object.hasOwn(absent, 'grain_boundaries')).toBe(false)
    const empty = setExtendedSection(absent, micro, 'grains', 'empty')
    const created = changeExtendedRows(empty, micro, 'grains', { kind: 'insert', index: 0 })
    expect(created.grain_boundaries).toStrictEqual([{ layer_ids: [] }])
    const moved = changeExtendedRows(device(), micro, 'grain-layers:0', { kind: 'move', from: 1, to: 0 })
    expect(moved.grain_boundaries![0].layer_ids).toEqual(['back', 'front'])
    expect(() => writeExtendedRow(moved, micro, 'grain-layers:0', 0, { kind: 'omit' })).toThrow(/remove/)
  })
  it('edits bottom-cell geometry and resource presence without rewriting top or root payloads', () => {
    const input = tandem(), bottom: ExtendedSelection = { cell: 'bottom_cell', kind: 'microstructure' }
    const changed = writeExtendedField(input, bottom, 'grain:0', 'width', { kind: 'value', value: '30 nm' })
    const expected = structuredClone(input); expected.bottom_cell.grain_boundaries![0].width = '30 nm'
    expect(changed).toStrictEqual(expected)
    const cleared = writeExtendedField(changed, { cell: 'bottom_cell', kind: 'resources' }, 'resources', 'spectrum', { kind: 'omit' })
    expect(Object.hasOwn(cleared.bottom_cell, 'spectrum')).toBe(false); expect(cleared.top_cell.spectrum).toBeNull()
    expect(() => changeExtendedRows(input, stack, 'junction', { kind: 'insert', index: 1 }, 9)).toThrow(/copied row/)
  })
})
