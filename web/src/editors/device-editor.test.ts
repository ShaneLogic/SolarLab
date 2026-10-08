import { afterEach, describe, expect, it, vi } from 'vitest'
import type { BulkDefectInput, DeviceInput, MetastableDocumentInput, MetastablePreparationInput, MultivalentDefectInput, TandemInput } from '../generated/configuration-inputs'
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
      return item.kind === kind && (kind === 'settings' || item.id === id && (item.occurrence ?? 0) === occurrence)
    })!
    choose('[data-role=entity]', option.value)
  }
  function type(field: string, value: string, area = 'basic') {
    const node = query<HTMLInputElement>(`[data-field="${field}"][data-area="${area}"] input`)
    node.value = value; node.dispatchEvent(new Event('input'))
  }
  function source(field: string, value: string, area = 'parameters') {
    choose(`[data-field="${field}"][data-area="${area}"] select`, value)
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

  it('edits supplied contact scalars and references, preserving null/omission and incomplete text', () => {
    const original = input(), view = mount(original); view.entity('contact', 'left')
    view.source('S_n', 'value', 'basic'); view.type('S_n', '0'); view.editor.apply()
    expect(view.onApply.mock.lastCall![0].contacts[0].S_n).toBe('0')
    view.source('S_n', 'null', 'basic'); expect((view.editor.read().input as DeviceInput).contacts![0].S_n).toBeNull()
    view.source('S_n', 'omit', 'basic'); expect(Object.hasOwn((view.editor.read().input as DeviceInput).contacts![0], 'S_n')).toBe(false)
    view.type('layer', 'missing'); view.editor.apply()
    expect(view.onApply.mock.lastCall![0].contacts[0].layer).toBe('missing')
    expect(view.root.textContent).toContain('not a supplied layer')
    view.source('S_n', 'value', 'basic'); view.type('S_n', ' ')
    view.entity('contact', 'right'); expect(view.editor.apply()).toBe(false)
    view.entity('contact', 'left'); expect(view.query<HTMLInputElement>('[data-field=S_n] input').value).toBe(' ')
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })

  it('edits nested interface defect energy, density and kinetics without changing retained partner data', () => {
    const original = input(); original.interfaces![0].defect = {
      id: 'trap', trap_depth_eV: '0.6 eV', energy_reference: 'below_conduction_band', total_density_m2: '1e12 cm^-2',
      calibration_factor: 0, iface_state_calibration_factor: '0.02',
      kinetics: { sigma_n_m2: -0, sigma_p_m2: '1e-19 m^2', thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: '1e5 m/s' }, partner_metadata: null,
    }
    const view = mount(original); view.entity('interface', 'pair'); view.click('select-interface-defect')
    expect(view.editor.read().selection).toMatchObject({ kind: 'interface_defect', id: 'trap' })
    expect(view.root.textContent).toContain('m^-2')
    view.type('total_density_m2', '2e12 cm^-2', 'defect')
    view.type('trap_depth_eV', '9 eV', 'defect')
    view.choose('[data-role=defect-section]', 'kinetics')
    expect(Object.is((view.editor.read().input as DeviceInput).interfaces![0].defect!.kinetics.sigma_n_m2, -0)).toBe(true)
    view.type('sigma_p_m2', '3e-19 m^2', 'kinetics')
    view.choose('[data-role=defect-section]', 'defect'); view.editor.apply()
    const result = view.onApply.mock.lastCall![0] as DeviceInput
    expect(result.interfaces![0].defect!.total_density_m2).toBe('2e12 cm^-2')
    expect(result.interfaces![0].defect!.trap_depth_eV).toBe('9 eV')
    expect(result.interfaces![0].defect!.partner_metadata).toBeNull()
    expect(result.contacts).toStrictEqual(original.contacts)
    view.entity('interface', 'pair'); view.click('defect-null')
    expect((view.editor.read().input as DeviceInput).interfaces![0].defect).toBeNull()
    view.click('defect-omit'); expect(Object.hasOwn((view.editor.read().input as DeviceInput).interfaces![0], 'defect')).toBe(false)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })

  it('keeps bulk branch conflicts, optional energy and incomplete profile rows explicit in the same draft', () => {
    const original = input(); original.layers[0].bulk_defects = [{ id: 'bulk', name: 'Bulk', charge_transition: 'acceptor', neutral_reference: 'empty', degeneracy: 1,
      distribution: { kind: 'gaussian', normalization: 'integrated_total', total_density_m3: '8e20', center_eV_above_vb: '0.39',
        width_eV: '0.05', width_convention: 'gaussian_standard_deviation', support_width_multiplier: '6', energy_reference: 'above_valence_band' },
      kinetics: { sigma_n_m2: 0, sigma_p_m2: '1e-19', thermal_velocity_n_m_s: '1e5', thermal_velocity_p_m_s: '8e4' },
      spatial_profile: { coordinate: 'normalized_layer_coordinate', interpolation: 'piecewise_linear', density_normalization: 'layer_average_unity',
        knots: [{ position_fraction: -0, density_multiplier: '1' }, { position_fraction: 1, density_multiplier: '1' }] },
    }]; original.layers[0].defect_schema_version = 'solarlab-explicit-bulk-defects-v3'; original.layers[0].defect_model = 'explicit_quasi_steady'
    const view = mount(original); view.entity('bulk_defect', 'bulk'); view.choose('[data-role=defect-section]', 'distribution')
    expect(view.root.textContent).toContain('m^-3'); expect(view.root.textContent).toContain('Gaussian width convention')
    view.choose('[data-area=distribution][data-field=kind] [data-role=scalar-value]', 'single_level')
    const changed = () => (view.editor.read().input as DeviceInput).layers[0].bulk_defects![0] as BulkDefectInput
    expect(changed().distribution.width_eV).toBe('0.05'); expect(view.editor.apply()).toBe(true)
    view.choose('[data-role=defect-section]', 'energy_level'); view.click('section-empty')
    expect(changed().energy_level).toStrictEqual({}); expect(view.editor.apply()).toBe(false)
    view.choose('[data-area=energy_level][data-field=reference] [data-role=scalar-value]', 'above_valence_band')
    view.type('value_eV', '400 meV', 'energy_level'); expect(view.editor.apply()).toBe(true)
    expect(changed().distribution.center_eV_above_vb).toBe('0.39')
    view.click('section-null'); expect(changed().energy_level).toBeNull()
    view.click('section-omit'); expect(Object.hasOwn(changed(), 'energy_level')).toBe(false)
    view.choose('[data-role=defect-section]', 'spatial_profile'); view.click('add-knot')
    expect(view.query<HTMLSelectElement>('[data-role=defect-section]').value).toBe('knot:2')
    expect(changed().spatial_profile!.knots[2]).toStrictEqual({ position_fraction: '', density_multiplier: '' })
    view.type('position_fraction', '0.5', 'knot:2'); view.entity('contact', 'left'); expect(view.editor.apply()).toBe(false)
    view.entity('bulk_defect', 'bulk'); view.choose('[data-role=defect-section]', 'knot:2')
    expect(view.query<HTMLInputElement>('[data-area="knot:2"][data-field=position_fraction] input').value).toBe('0.5')
    view.type('density_multiplier', '1', 'knot:2'); expect(view.editor.apply()).toBe(true)
    expect(changed().spatial_profile!.knots.map(knot => knot.position_fraction)).toStrictEqual([-0, 1, '0.5'])
    view.click('remove-knot'); expect(changed().spatial_profile!.knots).toHaveLength(2)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })

  it('retains model/schema declarations and lets explicit duplicate bulk IDs remain previewable', () => {
    const original = input(), item: BulkDefectInput = { id: 'one', distribution: { kind: 'single_level', normalization: 'integrated_total', total_density_m3: '1e18', center_eV_above_vb: '0.3' },
      charge_transition: 'unresolved', neutral_reference: 'unresolved', degeneracy: 1,
      kinetics: { sigma_n_m2: 0, sigma_p_m2: '1e-19', thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: 1e5 } }
    original.layers[0].bulk_defects = [item, { ...item, id: 'two' }]
    original.layers[0].defect_schema_version = 'solarlab-explicit-bulk-defects-v1'; original.layers[0].defect_model = 'effective_lifetime'
    const view = mount(original); view.entity('bulk_defect', 'one')
    view.query<HTMLInputElement>('[data-role=item-id]').value = 'two'; view.click('rename-item')
    expect(view.root.textContent).toContain('duplicate two'); expect(view.editor.apply()).toBe(true)
    expect((view.editor.read().input as DeviceInput).layers[0].defect_schema_version).toBe(original.layers[0].defect_schema_version)
    view.entity('bulk_defect', 'two', 1); expect(view.editor.read().selection).toMatchObject({ id: 'two', occurrence: 1 })
    view.click('remove-bulk-defect'); expect((view.editor.read().input as DeviceInput).layers[0].bulk_defects).toHaveLength(1)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })

  it('distinguishes bulk defects in duplicate parent layers and edits only the selected occurrence', () => {
    const original = input(), item: BulkDefectInput = { id: 'trap', name: 'Trap',
      distribution: { kind: 'single_level', normalization: 'integrated_total', total_density_m3: '1e18', center_eV_above_vb: '0.3' },
      charge_transition: 'acceptor', neutral_reference: 'empty', degeneracy: 1,
      kinetics: { sigma_n_m2: -0, sigma_p_m2: '1e-19', thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: 1e5 } }
    original.layers[0].bulk_defects = [item]
    original.layers[1].id = 'front'; original.layers[1].bulk_defects = [structuredClone(item), structuredClone(item)]
    const view = mount(original), options = [...view.query<HTMLSelectElement>('[data-role=entity]').options]
      .filter(option => JSON.parse(option.value).kind === 'bulk_defect')
    expect(options.map(option => option.textContent)).toStrictEqual([
      'Bulk defect trap: Trap — layer front',
      'Bulk defect trap: Trap — layer front (duplicate 2)',
      'Bulk defect trap (duplicate 2): Trap — layer front (duplicate 2)',
    ])
    view.choose('[data-role=entity]', options[2].value); view.type('name', 'Selected trap', 'defect')
    const draft = view.editor.read().input as DeviceInput
    expect(draft.layers[0]).toStrictEqual(original.layers[0])
    expect(draft.layers[1].bulk_defects![0]).toStrictEqual(original.layers[1].bulk_defects![0])
    expect(draft.layers[1].bulk_defects![1]).toStrictEqual({ ...item, name: 'Selected trap' })
    view.choose('[data-role=defect-section]', 'kinetics')
    expect(view.editor.read().selection).toMatchObject({ id: 'trap', occurrence: 1, parent: { id: 'front', occurrence: 1 } })
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })
})

function complexInput(): DeviceInput {
  const original = input()
  original.layers[0].bulk_defects = [{ id: 'multi', name: 'Multivalent', total_density_m3: '2e21', configuration: {
    family: 'double_donor', charge_states_e: [2, 1, -0], degeneracy_convention: 'explicit', state_degeneracies: [1, '2.00000000000000001', 1],
    energy_levels: { first_transition_eV_above_vb: '300 meV', correlation_energies_eV: ['-0.10000000000000001 eV'], energy_reference: 'above_valence_band' },
    transition_kinetics: [{ sigma_n_m2: -0, sigma_p_m2: '1e-19', thermal_velocity_n_m_s: '1e5', thermal_velocity_p_m_s: 1e5 },
      { sigma_n_m2: '2e-19', sigma_p_m2: 0, thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: '8e4' }],
  } }]
  return original
}
function arrayInput(view: ReturnType<typeof mount>, index: number) {
  return view.query<HTMLInputElement>(`[data-role=defect-array] > [data-row="${index}"] [data-role=array-value] input`)
}

function metastableInput(): DeviceInput {
  const original = complexInput(), configuration = (original.layers[0].bulk_defects![0] as MultivalentDefectInput).configuration
  const document: MetastableDocumentInput = { schema_version: 'solarlab-metastable-bulk-defects-v1', defect_model: 'explicit_metastable_frozen',
    metastable_defects: [{ id: 'meta', name: '<img src=x onerror=alert(1)>', total_density_m3: '3e21',
      donor_configuration: structuredClone(configuration), acceptor_configuration: structuredClone(configuration), donor_conversion_state_index: 0, acceptor_conversion_state_index: 2,
      conversion_kinetics: { transition_energy_eV_above_vb: '0.19', electron_capture_activation_eV: '0.1', electron_emission_activation_eV: '0.76',
        hole_capture_activation_eV: '0.35', hole_emission_activation_eV: '0.73', electron_capture_path: 'electron_capture_plus_hole_emission', hole_capture_path: 'double_hole_capture',
        capture_n_m3_s: '2e-14', capture_p_m3_s: '3e-14', phonon_frequency_Hz: '1e13' } }] }
  const preparation: MetastablePreparationInput = { schema_version: 'solarlab-metastable-preparation-v1', preparation_limit: 'stationary_infinite_time',
    preparation_temperature_K: 330, preparation_voltage_V: -0, preparation_illumination_suns: 0, voltage_continuation_steps: 20,
    illumination_continuation_steps: 0, measurement_temperature_K: '200 K', configuration_freeze_stage: 'after_stationary_preparation_before_measurement',
    freeze_configuration_during_measurement: true, measurement_protocol_sha256: 'a'.repeat(64), numerics: {
      initial_donor_fraction_guess: '0.5', max_iterations: 250, relative_tolerance: '1e-6', clamping_factor: '0.05', final_unclamped_refinement: true } }
  original.layers[0].metastable_document = document; original.layers[0].metastable_preparation = preparation
  return original
}
function arrayType(view: ReturnType<typeof mount>, index: number, value: string) {
  const node = arrayInput(view, index); node.value = value; node.dispatchEvent(new Event('input'))
}
function rowAction(view: ReturnType<typeof mount>, index: number, action: string) {
  const row = view.query(`[data-role=defect-array] > [data-row="${index}"]`)
  const button = row?.querySelector<HTMLButtonElement>(`[data-action="${action}"]`)
  expect(button, row?.outerHTML ?? 'Missing ordered row').toBeTruthy()
  button!.click()
}
function create(view: ReturnType<typeof mount>, kind: string, id: string, source?: Selection) {
  view.choose('[data-role=new-record-kind]', kind)
  if (source) view.choose('[data-role=record-template]', JSON.stringify(source))
  const node = view.query<HTMLInputElement>('[data-role=new-record-id]'); node.value = id; node.dispatchEvent(new Event('input'))
  view.click('create-record')
}

describe('complex array and inventory draft controls', () => {
  it('keeps exact scalar array words on no-op and sends changed unit strings without conversion', () => {
    const original = complexInput(), view = mount(original); view.entity('bulk_defect', 'multi')
    view.choose('[data-role=defect-section]', 'charges'); expect(arrayInput(view, 2).value).toBe('-0')
    view.choose('[data-role=defect-section]', 'degeneracies'); expect(arrayInput(view, 1).value).toBe('2.00000000000000001')
    expect(view.editor.apply()).toBe(true); expect(view.onApply.mock.lastCall![0]).toStrictEqual(original)
    view.choose('[data-role=defect-section]', 'correlations'); arrayType(view, 0, '-150 meV')
    const changed = view.editor.read().input as DeviceInput, item = changed.layers[0].bulk_defects![0] as MultivalentDefectInput
    expect(item.configuration.energy_levels.correlation_energies_eV).toStrictEqual(['-150 meV'])
    expect(item.configuration.charge_states_e).toStrictEqual([2, 1, -0])
  })
  it('keeps incomplete scalar text tied to its row across navigation, insertion, movement and removal', () => {
    const view = mount(complexInput()); view.entity('bulk_defect', 'multi'); view.choose('[data-role=defect-section]', 'degeneracies')
    arrayType(view, 0, ' '); expect(view.editor.apply()).toBe(false)
    rowAction(view, 0, 'row-later')
    expect(view.editor.read().incomplete).toContainEqual(['layers', 0, 'bulk_defects', 0, 'configuration', 'state_degeneracies', 1])
    expect(arrayInput(view, 1).value).toBe(' ')
    rowAction(view, 1, 'row-insert')
    expect(arrayInput(view, 2).value).toBe(' '); expect(view.editor.read().incomplete).toHaveLength(2)
    rowAction(view, 1, 'row-remove'); expect(arrayInput(view, 1).value).toBe(' ')
    view.entity('contact', 'left'); expect(view.editor.apply()).toBe(false)
    view.entity('bulk_defect', 'multi'); view.choose('[data-role=defect-section]', 'degeneracies')
    expect(arrayInput(view, 1).value).toBe(' ')
    arrayType(view, 1, '0'); expect(view.editor.apply()).toBe(true)
    const item = (view.editor.read().input as DeviceInput).layers[0].bulk_defects![0] as MultivalentDefectInput
    expect(item.configuration.state_degeneracies).toStrictEqual(['2.00000000000000001', '0', 1])
    expect(item.configuration.charge_states_e).toStrictEqual([2, 1, -0])
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(complexInput())
  })
  it('copies the current edited row and duplicates its pending text without sharing state', () => {
    const view = mount(complexInput()); view.entity('bulk_defect', 'multi'); view.choose('[data-role=defect-section]', 'degeneracies')
    arrayType(view, 0, '3'); rowAction(view, 0, 'row-copy')
    expect(arrayInput(view, 1).value).toBe('3')
    arrayType(view, 1, ''); rowAction(view, 1, 'row-copy')
    expect(view.editor.read().incomplete).toHaveLength(2)
    arrayType(view, 1, '4'); expect(view.editor.read().incomplete).toHaveLength(1)
    expect(arrayInput(view, 2).value).toBe('')
    rowAction(view, 2, 'row-remove'); expect(view.editor.apply()).toBe(true)
  })
  it('moves transition rows with pending fields and preserves independent capture and charge arrays', () => {
    const original = complexInput(), view = mount(original); view.entity('bulk_defect', 'multi')
    view.choose('[data-role=defect-section]', 'transition:0'); view.type('sigma_n_m2', '', 'transition:0')
    view.click('row-later')
    expect(view.query<HTMLSelectElement>('[data-role=defect-section]').value).toBe('transition:1')
    expect(view.query<HTMLInputElement>('[data-area="transition:1"][data-field=sigma_n_m2] input').value).toBe('')
    expect(view.editor.read().incomplete).toContainEqual(['layers', 0, 'bulk_defects', 0, 'configuration', 'transition_kinetics', 1, 'sigma_n_m2'])
    view.type('sigma_n_m2', '0', 'transition:1'); expect(view.editor.apply()).toBe(true)
    const item = (view.editor.read().input as DeviceInput).layers[0].bulk_defects![0] as MultivalentDefectInput
    expect(item.configuration.transition_kinetics[0].sigma_n_m2).toBe('2e-19')
    expect(item.configuration.transition_kinetics[1].sigma_n_m2).toBe('0')
    expect(item.configuration.charge_states_e).toStrictEqual([2, 1, -0])
  })
  it('starts explicit empty records with all unfilled declarations pending and removes them reversibly', () => {
    const original = complexInput(), view = mount(original)
    create(view, 'multivalent_defect', 'new')
    expect(view.editor.read().selection).toMatchObject({ kind: 'bulk_defect', id: 'new' })
    expect(view.editor.apply()).toBe(false)
    expect(view.editor.read().incomplete).toContainEqual(['layers', 0, 'bulk_defects', 1, 'configuration', 'family'])
    const newItem = (view.editor.read().input as DeviceInput).layers[0].bulk_defects![1] as MultivalentDefectInput
    expect(newItem.configuration.charge_states_e).toStrictEqual([])
    expect(Object.hasOwn(newItem.configuration, 'family')).toBe(false)
    view.entity('layer', 'back'); expect(view.editor.apply()).toBe(false)
    view.entity('bulk_defect', 'new'); view.click('remove-record')
    expect(view.editor.read().input).toStrictEqual(original); expect(view.editor.apply()).toBe(true)
  })
  it('copies and removes a selected record without changing topology or losing pending fields on another record', () => {
    const original = complexInput(), view = mount(original)
    view.type('name', '')
    const source = view.editor.read().selection
    create(view, 'layer', 'copy', source)
    expect(view.editor.read().selection).toMatchObject({ kind: 'layer', id: 'copy' })
    expect(view.editor.read().incomplete).toContainEqual(['layers', 2, 'name'])
    view.type('name', 'Copy'); view.click('remove-record')
    expect(view.editor.read().incomplete).toContainEqual(['layers', 0, 'name'])
    expect((view.editor.read().input as DeviceInput).contacts).toStrictEqual(original.contacts)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })
  it('keeps created duplicate material IDs and dangling references visible for resolver validation', () => {
    const original = input(), view = mount(original)
    create(view, 'material', 'shared', { cell: 'device', kind: 'material', id: 'shared', occurrence: 0 })
    expect(view.editor.read().selection).toMatchObject({ kind: 'material', occurrence: 1 })
    expect(view.editor.apply()).toBe(true)
    view.query<HTMLInputElement>('[data-role=item-id]').value = 'other'; view.click('rename-item')
    expect(view.query('[data-role=edit-error]').textContent).toContain('ambiguous')
    view.click('remove-record'); view.entity('material', 'shared'); view.click('remove-record')
    expect((view.editor.read().input as DeviceInput).layers[0].material).toBe('shared')
    expect(view.root.textContent).toContain('not a supplied material'); expect(view.editor.apply()).toBe(true)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })
  it('creates and clears metastable documents explicitly and requires generated tags without choosing them', () => {
    const view = mount(input()); view.click('start-metastable_document')
    expect(view.editor.read().selection.kind).toBe('metastable_document')
    expect(view.editor.apply()).toBe(false)
    const document = (view.editor.read().input as DeviceInput).layers[0].metastable_document
    expect(document).toStrictEqual({ metastable_defects: [] })
    view.choose('[data-area=document][data-field=schema_version] [data-role=scalar-value]', 'solarlab-metastable-bulk-defects-v1')
    view.choose('[data-area=document][data-field=defect_model] [data-role=scalar-value]', 'explicit_metastable_frozen')
    expect(view.editor.apply()).toBe(true) // Empty inventory is an authoritative semantic error, not auto-filled.
    view.click('null-metastable_document'); expect((view.editor.read().input as DeviceInput).layers[0].metastable_document).toBeNull()
    view.click('omit-metastable_document'); expect(Object.hasOwn((view.editor.read().input as DeviceInput).layers[0], 'metastable_document')).toBe(false)
    view.click('start-metastable_preparation'); view.choose('[data-role=defect-section]', 'numerics')
    const values = [...view.query<HTMLSelectElement>('[data-area=numerics][data-field=final_unclamped_refinement] [data-role=scalar-value]').options].map(item => item.value)
    expect(values).toStrictEqual(['', 'true']); expect(view.editor.apply()).toBe(false)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(input())
  })
  it('shows last submitted input separately from a new draft and ignores disposed row controls', () => {
    const view = mount(complexInput()); view.entity('bulk_defect', 'multi'); view.choose('[data-role=defect-section]', 'charges')
    expect(view.editor.apply()).toBe(true)
    arrayType(view, 0, '3'); expect(view.query('[data-editor=device]').getAttribute('data-preview-relation')).toBe('not-submitted')
    expect(view.editor.apply()).toBe(true); view.choose('[data-role=defect-section]', 'degeneracies')
    expect(view.root.textContent).toContain('matches the last input sent')
    const control = arrayInput(view, 0), prior = view.editor.read()
    view.editor.dispose(); control.value = '9'; control.dispatchEvent(new Event('input'))
    expect(view.editor.read()).toStrictEqual(prior)
  })
  it('updates the override presence label immediately without replacing the focused input', () => {
    const view = mount(input()); view.entity('material', 'shared'); view.parameter('mu_p')
    const picker = view.query<HTMLSelectElement>('[data-role=parameter]')
    expect(picker.selectedOptions[0].textContent).toContain('omitted')
    view.source('mu_p', 'value')
    const control = view.query<HTMLInputElement>('[data-area=parameters][data-field=mu_p] input')
    control.focus(); control.value = '0.0003 m^2/(V s)'; control.setSelectionRange(3, 3); control.dispatchEvent(new Event('input'))
    expect(picker.selectedOptions[0].textContent).toContain('supplied')
    expect(document.activeElement).toBe(control); expect(control.selectionStart).toBe(3)
    view.source('mu_p', 'omit'); expect(picker.selectedOptions[0].textContent).toContain('omitted')
    expect(Object.hasOwn((view.editor.read().input as DeviceInput).materials![0].parameters, 'mu_p')).toBe(false)
  })
  it('edits metastable inventories, both configurations and preparation values without rewriting any other branch', () => {
    const original = metastableInput(), view = mount(original); view.entity('metastable_defect', 'meta')
    expect(view.root.querySelector('img,script')).toBeNull()
    expect(view.root.textContent).toContain('<img src=x onerror=alert(1)>')
    view.choose('[data-role=defect-section]', 'donor:charges'); arrayType(view, 0, '3')
    view.choose('[data-role=defect-section]', 'acceptor:correlations'); arrayType(view, 0, '150 meV')
    view.choose('[data-role=defect-section]', 'conversion'); view.type('capture_n_m3_s', '2e-8 cm^3/s', 'conversion')
    const expected = structuredClone(original), meta = expected.layers[0].metastable_document!.metastable_defects[0]
    meta.donor_configuration.charge_states_e[0] = 3; meta.acceptor_configuration.energy_levels.correlation_energies_eV[0] = '150 meV'; meta.conversion_kinetics.capture_n_m3_s = '2e-8 cm^3/s'
    expect(view.editor.read().input).toStrictEqual(expected)
    view.entity('metastable_preparation', 'front'); view.choose('[data-role=defect-section]', 'preparation')
    expect(view.query<HTMLInputElement>('[data-area=preparation][data-field=preparation_voltage_V] input').value).toBe('-0')
    view.type('preparation_voltage_V', '0 mV', 'preparation'); expected.layers[0].metastable_preparation!.preparation_voltage_V = '0 mV'
    expect(view.editor.read().input).toStrictEqual(expected)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })
  it.each(['top_cell', 'bottom_cell'] as const)('edits complex declarations only in the selected %s', cell => {
    const original: TandemInput = { schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem', source_schema_version: 1,
      device_type: 'tandem_2T_monolithic', top_cell: metastableInput(), bottom_cell: metastableInput(), top_cell_reference: 'top', bottom_cell_reference: 'bottom',
      junction_model: 'ideal_ohmic', light_direction: 'top_first', junction_stack: [], benchmark: null, back_reflector: null }
    const view = mount(original); view.choose('[data-role=cell]', cell); view.entity('metastable_defect', 'meta'); view.choose('[data-role=defect-section]', 'donor:degeneracies')
    arrayType(view, 1, '0')
    const expected = structuredClone(original); expected[cell].layers[0].metastable_document!.metastable_defects[0].donor_configuration.state_degeneracies[1] = '0'
    expect(view.editor.read().input).toStrictEqual(expected); expect(view.editor.apply()).toBe(true)
    view.click('discard'); expect(view.editor.read().input).toStrictEqual(original)
  })
})
