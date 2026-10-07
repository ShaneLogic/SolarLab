import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DeviceInput, TandemInput } from '../generated/configuration-inputs'
import { mountDeviceEditor } from './device-editor'
import type { DeviceEditor } from './device-editor'
import type { EditorInput, Selection } from './device-edits'

const mounted: DeviceEditor[] = []
function input(): DeviceInput {
  return { schema_version: 'solarlab.device-preparation.v1', id: 'sample', source_format: 'canonical',
    layers: [{ id: 'front', name: 'Front', role: 'HTL', thickness: '2.00000000000000001e-7 m', material: 'shared',
      parameters: { mu_n: -0, Nc300: null, incoherent: false, grading_N_mult: 3 }, bulk_defects: [], cigs_graded_optics: null },
    { id: 'back', name: 'Back', role: 'ETL', thickness: 1e-7 }],
    materials: [{ id: 'shared', name: 'Shared', parameters: { mu_n: '2 cm^2/(V s)' } }],
    interfaces: [{ id: 'pair', left: 'front', right: 'back', v_n: null }],
    contacts: [{ id: 'left', side: 'left', layer: 'front' }, { id: 'right', side: 'right', layer: 'back' }],
    settings: { phi_left: -0 }, grain_boundaries: [], simulation_hints: null }
}
function mount(value: EditorInput = input()) {
  const root = document.createElement('div'); document.body.append(root)
  const onApply = vi.fn(), editor = mountDeviceEditor(root, { input: value, onApply }); mounted.push(editor)
  const query = <T extends HTMLElement = HTMLElement>(selector: string) => root.querySelector<T>(selector)!
  function choose(selector: string, value: string) {
    const select = query<HTMLSelectElement>(selector); select.value = value; select.dispatchEvent(new Event('change'))
  }
  function entity(kind: Selection['kind'], id?: string, occurrence = 0) {
    const select = query<HTMLSelectElement>('[data-role=entity]')
    const option = [...select.options].find(option => {
      const item = JSON.parse(option.value)
      return item.kind === kind && (kind === 'settings' || item.id === id && item.occurrence === occurrence)
    })!
    choose('[data-role=entity]', option.value)
  }
  function type(field: string, value: string, area = 'basic') {
    const node = query<HTMLInputElement>(`[data-field="${field}"][data-area=${area}] input`)
    node.value = value; node.dispatchEvent(new Event('input'))
  }
  function source(field: string, value: string, area = 'parameters') {
    choose(`[data-field="${field}"][data-area=${area}] select`, value)
  }
  const parameter = (name: string) => choose('[data-role=parameter]', name)
  const click = (action: string) => query<HTMLButtonElement>(`[data-action=${action}]`).click()
  return { root, editor, onApply, query, choose, entity, type, source, parameter, click }
}
afterEach(() => { for (const editor of mounted.splice(0)) editor.dispose(); document.body.replaceChildren() })

describe('existing layer/material draft editor', () => {
  it('applies exact no-op input without mutating or aliasing the caller or callback', () => {
    const original = input(), view = mount(original)
    expect(view.editor.read().dirty).toBe(false)
    expect(view.editor.apply()).toBe(true)
    expect(view.onApply).toHaveBeenLastCalledWith(original)
    const sent = view.onApply.mock.calls[0][0]
    expect(Object.is(sent.layers[0].parameters.mu_n, -0)).toBe(true)
    sent.layers[0].name = 'callback mutation'
    expect(view.editor.read().input).toStrictEqual(original)
    expect(original.layers[0].name).toBe('Front')
  })

  it('keeps selection by stable ID while editing safe names and rerendering parameter fields', () => {
    const view = mount(); view.entity('layer', 'back')
    view.type('name', '<img src=x onerror=alert(1)>')
    view.parameter('mu_n'); view.source('mu_n', 'value'); view.type('mu_n', '2.00000000000000001 cm^2/(V s)', 'parameters')
    view.parameter('Nc300'); view.parameter('mu_n')
    expect(view.editor.read().selection).toMatchObject({ kind: 'layer', id: 'back' })
    const draft = view.editor.read().input as DeviceInput
    expect(draft.layers[1].id).toBe('back')
    expect(draft.layers[0]).toStrictEqual(input().layers[0])
    expect(draft.layers[1].parameters!.mu_n).toBe('2.00000000000000001 cm^2/(V s)')
    expect(view.root.querySelector('img,script')).toBeNull()
    expect(view.root.textContent).toContain('<img src=x onerror=alert(1)>')
  })

  it('edits supplied material values and removes a local override while retaining references', () => {
    const view = mount(); view.entity('material', 'shared'); view.type('name', 'Renamed display')
    view.parameter('mu_n'); view.type('mu_n', '3 cm^2/(V s)', 'parameters')
    view.entity('layer', 'front'); view.parameter('mu_n'); view.source('mu_n', 'omit')
    const draft = view.editor.read().input as DeviceInput
    expect(draft.layers[0].material).toBe('shared')
    expect(Object.hasOwn(draft.layers[0].parameters!, 'mu_n')).toBe(false)
    expect(draft.materials![0]).toStrictEqual({ id: 'shared', name: 'Renamed display', parameters: { mu_n: '3 cm^2/(V s)' } })
  })

  it('retains incomplete text across navigation and prevents Apply until corrected or discarded', () => {
    const view = mount(); view.parameter('mu_n'); view.type('mu_n', ' ', 'parameters')
    view.entity('layer', 'back'); expect(view.editor.apply()).toBe(false)
    expect(view.onApply).not.toHaveBeenCalled()
    view.entity('layer', 'front'); view.parameter('mu_n')
    expect(view.query<HTMLInputElement>('[data-field=mu_n][data-area=parameters] input').value).toBe(' ')
    view.type('mu_n', '0', 'parameters'); expect(view.editor.apply()).toBe(true)
    expect((view.editor.read().input as DeviceInput).layers[0].parameters!.mu_n).toBe('0')
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(input())
    expect(view.editor.read().dirty).toBe(false)
  })

  it('does not lose an incomplete required name on a rerender', () => {
    const view = mount(); view.type('name', ''); view.parameter('mu_n')
    expect(view.editor.apply()).toBe(false)
    expect(view.query<HTMLInputElement>('[data-field=name] input').value).toBe('')
    view.type('name', 'Valid name'); expect(view.editor.apply()).toBe(true)
  })

  it('derives null, enum, boolean and number controls from generated metadata', () => {
    const view = mount(); view.parameter('Nc300'); view.source('Nc300', 'null')
    expect((view.editor.read().input as DeviceInput).layers[0].parameters!.Nc300).toBeNull()
    view.source('Nc300', 'omit'); expect(Object.hasOwn((view.editor.read().input as DeviceInput).layers[0].parameters!, 'Nc300')).toBe(false)
    view.parameter('incoherent')
    view.choose('[data-field=incoherent] [data-role=scalar-value]', 'true')
    expect((view.editor.read().input as DeviceInput).layers[0].parameters!.incoherent).toBe(true)
    view.parameter('carrier_statistics'); view.source('carrier_statistics', 'value')
    view.choose('[data-field=carrier_statistics] [data-role=scalar-value]', 'fermi_dirac')
    expect((view.editor.read().input as DeviceInput).layers[0].parameters!.carrier_statistics).toBe('fermi_dirac')
    view.parameter('grading_N_mult'); view.type('grading_N_mult', '12 nm', 'parameters')
    expect(view.editor.apply()).toBe(false)
    view.type('grading_N_mult', '12', 'parameters')
    expect((view.editor.read().input as DeviceInput).layers[0].parameters!.grading_N_mult).toBe(12)
    expect([...view.query<HTMLSelectElement>('[data-field=thickness] select').options].map(item => item.value)).toEqual(['value'])
  })

  it('makes explicit ID changes separate from names and shows duplicate drafts for resolver validation', () => {
    const view = mount(); const id = view.query<HTMLInputElement>('[data-role=layer-id]'); id.value = 'new_front'; view.click('rename-layer')
    const renamed = view.editor.read().input as DeviceInput
    expect(renamed.layers[0].name).toBe('Front')
    expect(renamed.interfaces![0].left).toBe('new_front')
    expect(renamed.contacts![0].layer).toBe('new_front')
    expect(view.editor.read().selection).toMatchObject({ id: 'new_front' })
    expect(view.query('[data-role=edit-error]').getAttribute('role')).toBe('status')
    expect(view.query('[data-role=edit-error]').classList.contains('error')).toBe(false)
    view.query<HTMLInputElement>('[data-role=layer-id]').value = 'back'; view.click('rename-layer')
    expect(view.root.textContent).toContain('duplicate back')
    expect(view.editor.apply()).toBe(true)
    expect(view.onApply.mock.lastCall![0].layers.map((item: { id: string }) => item.id)).toEqual(['back', 'back'])
    view.query<HTMLInputElement>('[data-role=layer-id]').value = 'corrected'; view.click('rename-layer')
    expect(view.query('[data-role=edit-error]').textContent).toContain('ambiguous')
    expect(view.query('[data-role=edit-error]').getAttribute('role')).toBe('alert')
    expect(view.query('[data-role=edit-error]').classList.contains('error')).toBe(true)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(input())
  })

  it('preserves invalid reorder drafts and allows only explicit connection correction', () => {
    const original = input(), view = mount(original); view.click('move-later')
    const draft = view.editor.read().input as DeviceInput
    expect(draft.layers).toStrictEqual([original.layers[1], original.layers[0]])
    expect(draft.interfaces).toStrictEqual(original.interfaces)
    expect(draft.contacts).toStrictEqual(original.contacts)
    expect(view.query('[data-section=connections]').hasAttribute('open')).toBe(true)
    expect(view.root.textContent).toContain('not a directed adjacent pair')
    const endpoint = view.query<HTMLInputElement>('[data-connection="interfaces:0"] [data-endpoint=left]')
    endpoint.value = 'missing'; endpoint.dispatchEvent(new Event('input'))
    expect(view.editor.apply()).toBe(true)
    expect(view.onApply.mock.lastCall![0].interfaces[0]).toStrictEqual({ id: 'pair', left: 'missing', right: 'back', v_n: null })
  })

  it('reaches duplicate and dangling supplied records without repairing them on render', () => {
    const original = input(); original.layers[1].id = 'front'; original.layers[0].material = 'missing'
    const view = mount(original); view.entity('layer', 'front', 1)
    expect(view.editor.read().selection).toMatchObject({ occurrence: 1 })
    expect(view.editor.read().input).toStrictEqual(original)
    expect(view.root.textContent).toContain('not a supplied material')
    expect(view.root.textContent).toContain('not a supplied layer')
  })

  it('refreshes material-reference notices in both directions without replacing the focused input', () => {
    const view = mount(), material = view.query<HTMLInputElement>('[data-field=material][data-area=basic] input')
    material.focus(); material.value = 'missing'; material.setSelectionRange(3, 3)
    material.dispatchEvent(new Event('input'))
    expect(view.query('[data-role=reference-notices]').textContent).toContain('missing is not a supplied material')
    expect(document.activeElement).toBe(material); expect(material.selectionStart).toBe(3)
    material.value = 'shared'; material.setSelectionRange(2, 2); material.dispatchEvent(new Event('input'))
    expect(view.query('[data-role=reference-notices]').textContent).not.toContain('not a supplied material')
    expect(document.activeElement).toBe(material); expect(material.selectionStart).toBe(2)
  })

  it('clears corrected endpoint notices and shows newly dangling/reversed endpoints without replacing the input', () => {
    const initial = input(); initial.interfaces![0].right = 'missing'
    const view = mount(initial), endpoint = view.query<HTMLInputElement>('[data-connection="interfaces:0"] [data-endpoint=right]')
    expect(view.query('[data-role=reference-notices]').textContent).toContain('not a supplied layer')
    endpoint.focus(); endpoint.value = 'back'; endpoint.setSelectionRange(2, 2); endpoint.dispatchEvent(new Event('input'))
    expect(view.query('[data-role=reference-notices]').textContent).toBe('')
    expect(document.activeElement).toBe(endpoint); expect(endpoint.selectionStart).toBe(2)
    endpoint.value = 'missing'; endpoint.dispatchEvent(new Event('input'))
    expect(view.query('[data-role=reference-notices]').textContent).toContain('not a supplied layer')
    expect(view.query('[data-role=reference-notices]').textContent).toContain('not a directed adjacent pair')
    expect(view.query('[data-connection="interfaces:0"] [data-endpoint=right]')).toBe(endpoint)
  })

  it('selects both tandem cells and retains sibling/root data and selection on a caller refresh', () => {
    const original: TandemInput = { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1,
      device_type: 'tandem_2T_monolithic', top_cell: input(), bottom_cell: { ...input(), id: 'bottom' },
      top_cell_reference: 'top.yaml', bottom_cell_reference: 'bottom.yaml', junction_model: 'ideal_ohmic',
      light_direction: 'top_first', junction_stack: [], benchmark: null, back_reflector: null }
    const view = mount(original); view.choose('[data-role=cell]', 'bottom_cell'); view.entity('layer', 'back'); view.type('name', 'Bottom back')
    const draft = view.editor.read().input as TandemInput
    expect(draft.top_cell).toStrictEqual(original.top_cell)
    expect(draft.bottom_cell.layers[1].name).toBe('Bottom back')
    const expected = structuredClone(original); expected.bottom_cell.layers[1].name = 'Bottom back'
    expect(draft).toStrictEqual(expected)
    view.editor.setInput(draft)
    expect(view.editor.read().selection).toMatchObject({ cell: 'bottom_cell', id: 'back' })
    view.choose('[data-role=cell]', 'top_cell'); view.entity('layer', 'front'); view.type('role', 'changed')
    expect((view.editor.read().input as TandemInput).bottom_cell).toStrictEqual(draft.bottom_cell)
  })

  it('keeps device settings and input signed zero without creating omitted values', () => {
    const view = mount(); view.entity('settings'); view.parameter('phi_left'); view.editor.apply()
    expect(Object.is(view.onApply.mock.lastCall![0].settings.phi_left, -0)).toBe(true)
    view.type('phi_left', '-25 mV'); view.editor.apply()
    expect(view.onApply.mock.lastCall![0].settings.phi_left).toBe('-25 mV')
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(input())
  })

  it('detaches controls and callbacks on disposal without removing a replacement view', () => {
    const view = mount(), button = view.query<HTMLButtonElement>('[data-action=apply]')
    const name = view.query<HTMLInputElement>('[data-field=name] input'), replacement = document.createElement('p')
    view.root.replaceChildren(replacement); view.editor.dispose()
    name.value = 'late'; name.dispatchEvent(new Event('input')); button.click()
    expect(view.onApply).not.toHaveBeenCalled(); expect(view.editor.apply()).toBe(false)
    expect(view.root.firstChild).toBe(replacement)
  })
})
