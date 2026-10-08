import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DeviceInput, TandemInput } from '../generated/configuration-inputs'
import { mountDeviceEditor } from './device-editor'
import type { DeviceEditor } from './device-editor'
import type { EditorInput, Selection } from './device-edits'

function input(): TandemInput {
  const cell: DeviceInput = { schema_version: 'solarlab.device-preparation.v1', id: 'top', source_format: 'canonical',
    layers: [{ id: 'absorber', name: 'Absorber', role: 'absorber', thickness: '100 nm', parameters: { mu_n: -0 },
      cigs_graded_optics: { ggi_front: -0, ggi_back: '0.200000000000000001', cgi: 0.9 } }],
    grain_boundaries: [{ id: 'gb', layer_ids: ['absorber'], x_position: '1 um', width: '10 nm', tau_n: '1 ns', tau_p: '2 ns' }],
    electrical_grid: [{ layer: 'absorber', interval_weight: 1, alpha: 2 }],
    tunnelling_channels: { intraband: { enabled: false }, contact: { barrier_height_eV: -0 } } }
  return { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1, device_type: 'tandem_2T_monolithic',
    top_cell: cell, bottom_cell: { ...structuredClone(cell), id: 'bottom' }, top_cell_reference: 'original top', bottom_cell_reference: 'original bottom',
    junction_model: 'ideal_ohmic', light_direction: 'top_first', junction_stack: [
      { id: 'optical', name: 'Optical <unsafe>', thickness: '2.00000000000000001 nm', optical_material: 'nk', incoherent: false },
      { id: 'other', name: 'Other', thickness: 3e-9, optical_material: 'nk', incoherent: true }], back_reflector: null, benchmark: null }
}
const mounted: DeviceEditor[] = []
function mount(value: EditorInput = input()) {
  const root = document.createElement('div'); document.body.append(root)
  const onApply = vi.fn(), editor = mountDeviceEditor(root, { input: value, onApply }); mounted.push(editor)
  const query = <T extends HTMLElement = HTMLElement>(selector: string) => root.querySelector<T>(selector)!
  const choose = (selector: string, value: string) => { const node = query<HTMLSelectElement>(selector); node.value = value; node.dispatchEvent(new Event('change')) }
  const entity = (kind: Selection['kind'], id?: string) => {
    const option = [...query<HTMLSelectElement>('[data-role=entity]').options].find(option => { const value = JSON.parse(option.value); return value.kind === kind && (!id || value.id === id) })!
    choose('[data-role=entity]', option.value)
  }
  const section = (key: string) => choose('[data-role=extended-section]', key)
  const type = (field: string, value: string) => { const node = query<HTMLInputElement>(`[data-field="${field}"] input`); node.value = value; node.dispatchEvent(new Event('input')); return node }
  const click = (action: string, parent = '') => query<HTMLButtonElement>(`${parent} [data-action=${action}]`).click()
  return { root, editor, onApply, query, choose, entity, section, type, click }
}
afterEach(() => { for (const editor of mounted.splice(0)) editor.dispose(); document.body.replaceChildren() })

describe('extended device draft controls', () => {
  it('keeps exact no-op data through all extended sections and both tandem cells', () => {
    const value = input(); Reflect.set(value.bottom_cell, 'future_extension', { flags: [false, null, -0] })
    const view = mount(value)
    for (const cell of ['top_cell', 'bottom_cell']) {
      view.choose('[data-role=cell]', cell)
      for (const kind of ['layer_optics', 'tunnelling', 'microstructure', 'resources', 'optical_stack'] as const) {
        view.entity(kind)
        const keys = [...view.query<HTMLSelectElement>('[data-role=extended-section]').options].map(option => option.value)
        for (const key of keys) view.section(key)
      }
    }
    expect(view.editor.read().input).toStrictEqual(value); expect(view.editor.read().dirty).toBe(false)
    expect(view.editor.apply()).toBe(true); expect(view.onApply).toHaveBeenLastCalledWith(value)
    expect(Object.is((view.editor.read().input as TandemInput).top_cell.layers[0].cigs_graded_optics!.ggi_front, -0)).toBe(true)
  })
  it('preserves focused input and caret on labels, IDs and quantity changes without re-render mutation', () => {
    const view = mount(); view.entity('optical_stack'); view.section('optical:0')
    const name = view.query<HTMLInputElement>('[data-field=name] input'); name.focus(); name.value = '<img onerror=alert(1)> renamed'; name.setSelectionRange(4, 8)
    name.dispatchEvent(new Event('input'))
    expect(document.activeElement).toBe(name); expect(name.selectionStart).toBe(4); expect(name.selectionEnd).toBe(8)
    view.type('id', 'new-optical')
    expect(view.query<HTMLSelectElement>('[data-role=extended-section]').selectedOptions[0].textContent).toContain('new-optical')
    const thickness = view.type('thickness', '0 nm'); thickness.focus(); thickness.setSelectionRange(1, 1); thickness.dispatchEvent(new Event('input'))
    expect(document.activeElement).toBe(thickness); expect(thickness.selectionStart).toBe(1)
    const value = view.editor.read().input as TandemInput
    expect(value.junction_stack[0].thickness).toBe('0 nm'); expect(value.top_cell).toStrictEqual(input().top_cell)
    expect(view.root.querySelector('img,script')).toBeNull(); expect(view.editor.apply()).toBe(true)
  })
  it('retains incomplete row text through copying and movement and clears only explicitly removed rows', () => {
    const view = mount(); view.entity('optical_stack'); view.section('optical:0'); view.type('thickness', ' ')
    view.click('extended-copy'); view.click('extended-later')
    expect(view.query<HTMLSelectElement>('[data-role=extended-section]').value).toBe('optical:1')
    expect(view.query<HTMLInputElement>('[data-field=thickness] input').value).toBe(' ')
    expect(view.editor.read().incomplete).toEqual([['junction_stack', 1, 'thickness'], ['junction_stack', 0, 'thickness']])
    view.click('extended-remove')
    expect(view.editor.read().incomplete).toEqual([['junction_stack', 0, 'thickness']])
    view.section('optical:0'); view.type('thickness', '4 nm')
    expect(view.editor.apply()).toBe(true); expect((view.editor.read().input as TandemInput).junction_stack.map(row => row.id)).toEqual(['optical', 'other'])
  })
  it('keeps a scalar layer-reference edit with its intended row through insert, reorder and remove', () => {
    const view = mount(); view.entity('microstructure'); view.section('grain-layers:0')
    const value = view.query<HTMLInputElement>('[data-role=extended-row-value] input'); value.value = ''; value.dispatchEvent(new Event('input'))
    view.click('extended-insert', '[data-role=extended-array] [data-row="0"]')
    expect(view.editor.read().incomplete.map(path => path.at(-1))).toEqual([1, 0])
    view.click('extended-earlier', '[data-role=extended-array] [data-row="1"]')
    expect(view.editor.read().incomplete.map(path => path.at(-1))).toEqual([0, 1])
    expect((view.editor.read().input as TandemInput).top_cell.grain_boundaries![0].layer_ids).toEqual(['absorber', ''])
    view.click('extended-remove', '[data-role=extended-array] [data-row="1"]')
    expect(view.editor.read().incomplete).toEqual([['top_cell', 'grain_boundaries', 0, 'layer_ids', 0]])
    const restored = view.query<HTMLInputElement>('[data-role=extended-row-value] input'); expect(restored.value).toBe('')
    restored.value = 'absent'; restored.dispatchEvent(new Event('input'))
    expect(view.root.textContent).toContain('absent must reference one existing electrical layer')
    expect(view.editor.apply()).toBe(true)
  })
  it('distinguishes clear, omitted and empty CIGS input and retains incomplete fields across navigation', () => {
    const view = mount(); view.entity('layer_optics'); view.click('extended-null')
    expect((view.editor.read().input as TandemInput).top_cell.layers[0].cigs_graded_optics).toBeNull()
    view.click('extended-omit'); expect(Object.hasOwn((view.editor.read().input as TandemInput).top_cell.layers[0], 'cigs_graded_optics')).toBe(false)
    view.click('extended-empty'); expect((view.editor.read().input as TandemInput).top_cell.layers[0].cigs_graded_optics).toStrictEqual({})
    expect(view.editor.apply()).toBe(false)
    view.type('ggi_front', '0'); view.entity('tunnelling'); expect(view.editor.apply()).toBe(false)
    view.entity('layer_optics'); view.type('ggi_back', '0.3'); view.type('cgi', '0.9'); expect(view.editor.apply()).toBe(true)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(input())
  })
  it('offers generated boolean/integer channel controls and preserves explicit false/zero versus inheritance', () => {
    const view = mount(); view.entity('tunnelling'); view.section('intraband')
    view.choose('[data-field=enabled] [data-role=scalar-value]', 'true')
    view.choose('[data-field=enabled] [data-role=scalar-value]', 'false')
    expect((view.editor.read().input as TandemInput).top_cell.tunnelling_channels!.intraband!.enabled).toBe(false)
    expect([...view.query<HTMLSelectElement>('[data-field=enabled] [data-role=value-source]').options].map(option => option.value)).not.toContain('null')
    view.choose('[data-field=enabled] [data-role=value-source]', 'omit')
    expect((view.editor.read().input as TandemInput).top_cell.tunnelling_channels!.intraband).toStrictEqual({})
    view.section('contact'); view.type('barrier_height_eV', '0 meV')
    expect((view.editor.read().input as TandemInput).top_cell.tunnelling_channels!.contact!.barrier_height_eV).toBe('0 meV')
    view.choose('[data-field=energy_quadrature_order] [data-role=value-source]', 'value'); view.type('energy_quadrature_order', '1e')
    expect(view.editor.apply()).toBe(false); view.type('energy_quadrature_order', '4'); expect(view.editor.apply()).toBe(true)
  })
  it('creates required blank rows explicitly and preserves invalid duplicate/reference drafts for preview', () => {
    const view = mount(); view.entity('microstructure'); view.section('grains'); view.click('extended-append')
    expect(view.editor.read().incomplete).toContainEqual(['top_cell', 'grain_boundaries', 1, 'id'])
    expect((view.editor.read().input as TandemInput).top_cell.grain_boundaries![1]).toStrictEqual({ layer_ids: [] })
    view.type('id', 'gb'); view.type('x_position', '2 um'); view.type('width', '20 nm'); view.type('tau_n', '1 ns'); view.type('tau_p', '2 ns')
    expect(view.root.textContent).toContain('Duplicate ID gb'); expect(view.editor.apply()).toBe(true)
    view.section('grid-row:0'); view.type('layer', 'missing')
    expect(view.root.textContent).toContain('missing must reference'); expect(view.editor.apply()).toBe(true)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(input())
  })
  it('copies the supplied optical row into a reflector only by explicit choice, retaining its duplicate ID', () => {
    const view = mount(); view.entity('optical_stack'); view.section('reflector')
    view.choose('[data-role=extended-template]', '0'); view.click('extended-copy-document')
    expect((view.editor.read().input as TandemInput).back_reflector).toStrictEqual(input().junction_stack[0])
    expect(view.root.textContent).toContain('Duplicate optical-layer ID optical')
    view.type('id', 'reflector'); expect(view.editor.apply()).toBe(true)
    expect((view.editor.read().input as TandemInput).top_cell).toStrictEqual(input().top_cell)
  })
  it('isolates bottom-cell mutations and makes last submission, dirty draft and disposal observable', () => {
    const view = mount(); view.choose('[data-role=cell]', 'bottom_cell'); view.entity('microstructure'); view.section('grain:0'); view.type('width', '20 nm')
    expect(view.editor.apply()).toBe(true)
    const sent = view.onApply.mock.calls.at(-1)![0] as TandemInput
    expect(sent.top_cell).toStrictEqual(input().top_cell); expect(sent.junction_stack).toStrictEqual(input().junction_stack)
    view.type('width', '30 nm')
    expect(view.query('[data-editor=device]').dataset.previewRelation).toBe('not-submitted')
    view.editor.setInput(sent); expect(view.editor.read().dirty).toBe(false)
    const before = view.onApply.mock.calls.length, field = view.query<HTMLInputElement>('[data-field=width] input')
    view.editor.dispose(); field.value = '99 nm'; field.dispatchEvent(new Event('input'))
    expect(view.editor.apply()).toBe(false); expect(view.onApply).toHaveBeenCalledTimes(before); expect(view.root.childElementCount).toBe(0)
  })
})
