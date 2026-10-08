import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SpatialExperimentInput, SpatialExperimentInputDefinitions } from '../generated/configuration-inputs'
import { configurationSchemaSha256 } from '../generated/configuration-schema'
import { mountSpatialExperimentEditor, mountSpatialExperimentPreview } from './spatial-experiment'
import type { SpatialExperimentEditor } from './spatial-experiment'

const mounted: SpatialExperimentEditor[] = []
function input(): SpatialExperimentInput {
  return { schema_version: 'solarlab.experiment-preparation.v1', id: 'test',
    device: { schema_version: 'solarlab.device-preparation.v1', id: 'device', source_format: 'canonical',
      layers: [{ id: 'active', name: '<img src=x onerror=alert(1)>', role: 'absorber', thickness: '400 nm',
        parameters: { mu_n: '0.00020000000000000001 m^2/(V s)', Nc300: null }, bulk_defects: [] }],
      settings: { phi_left: -0 }, grain_boundaries: [], electrical_grid: [] },
    experiment: { kind: 'jv_2d', lateral_length: '500.00000000000001 nm', Nx: 10, V_max: -0,
      illuminated: false, lateral_bc: null, microstructure: {} } }
}
function protocol(): SpatialExperimentInputDefinitions.JV2DProtocolInput {
  return { schema_version: 'jv-2d-execution-protocol-v1', temperature_K: '300 K', illuminated: false,
    illumination_source: null, initial_state_source: 'one_dimensional_dark_equilibrium', initial_state_voltage_V: -0,
    initial_state_settle_s: null, voltage_values_V: [0, '100 mV'], dwell_time_per_voltage_s: '1 ns',
    state_topology: 'frozen_ion_background', ion_boundary_condition: 'frozen', carrier_boundary_condition: 'ohmic',
    interface_srh: 'off', lateral_bc: 'periodic', x_coordinates_m: [0, '250 nm', '500 nm'],
    y_coordinates_m: [0, '200 nm', '400 nm'], grain_boundaries: [], current_composition: 'electron_hole_conduction',
    current_sampling: 'instantaneous_dwell_endpoint', applied_voltage_rate_at_sampling_V_s: '-0 mV/s', solver_method: 'Radau',
    solver_rtol: 1e-6, solver_atol: { mode: 'scalar', scalar_atol: 1e-8, carrier_fraction: null, ion_fraction: null,
      interface_fraction: null, minimum_atol: null, refinement_factor: null }, solver_max_step_divisor: 50,
    max_nfev_per_solve: 200000, max_bisect: 6, ion_inventory_rtol: 0, save_snapshots: false, implicit_legacy_protocol: false }
}
function mount(value = input()) {
  const root = document.createElement('div'); document.body.append(root)
  const onApply = vi.fn(), editor = mountSpatialExperimentEditor(root, { input: value, onApply }); mounted.push(editor)
  const path = (parts: (string | number)[]) => [...root.querySelectorAll<HTMLElement>('[data-path]')].find(node => node.dataset.path === JSON.stringify(parts))!
  const choose = (node: HTMLSelectElement, value: string) => { node.value = value; node.dispatchEvent(new Event('change', { bubbles: true })) }
  const text = (node: HTMLInputElement, value: string) => { node.value = value; node.dispatchEvent(new Event('input', { bubbles: true })) }
  const section = (value: string) => choose(root.querySelector<HTMLSelectElement>('[data-role=experiment-section]')!, value)
  return { root, editor, onApply, path, choose, text, section }
}
afterEach(() => { for (const editor of mounted.splice(0)) editor.dispose(); document.body.replaceChildren(); vi.unstubAllGlobals() })

describe('experiment-owned metadata controls', () => {
  it('keeps exact no-op input, types, negative zero, unknown payloads and safe labels', () => {
    const raw = input(); Reflect.set(raw.device, 'extension', { n: -0, value: null, list: [], units: '3 nm' })
    const before = structuredClone(raw), ui = mount(raw)
    expect(ui.editor.read().input).toStrictEqual(before)
    expect(ui.editor.apply()).toBe(true); expect(ui.onApply.mock.calls[0][0]).toStrictEqual(before)
    expect(ui.root.querySelector('img')).toBeNull()
    ui.onApply.mock.calls[0][0].device.layers.length = 0
    expect(ui.editor.read().input.device.layers.length).toBe(1)
    expect(raw).toStrictEqual(before)
    ui.section('microstructure')
    ui.choose(ui.path(['experiment', 'microstructure']).querySelector('select')!, 'value')
    expect(ui.editor.read().input).toStrictEqual(before)
  })

  it('edits only the chosen control, preserving focus/caret, words and absence', () => {
    const ui = mount(), original = ui.editor.read().input
    const control = ui.path(['experiment', 'lateral_length']).querySelector('input')!
    control.focus(); control.value = '0.5 um'; control.setSelectionRange(3, 3); control.dispatchEvent(new Event('input'))
    expect(document.activeElement).toBe(control); expect(control.selectionStart).toBe(3)
    expect(ui.editor.read().input).toStrictEqual({ ...original, experiment: { ...original.experiment, lateral_length: '0.5 um' } })
    const numeric = ui.path(['experiment', 'Nx']).querySelector('input')!
    ui.text(numeric, '0'); expect(ui.editor.read().input.experiment.Nx).toBe(0)
    ui.text(numeric, ''); expect(ui.editor.read().incomplete).toContainEqual(['experiment', 'Nx'])
    expect(ui.editor.apply()).toBe(false); expect(ui.editor.read().input.experiment.Nx).toBe(0)
    ui.text(numeric, '12'); expect(ui.editor.apply()).toBe(true)
    ui.choose(ui.path(['experiment', 'lateral_length']).querySelector('select')!, 'omit')
    expect(Object.hasOwn(ui.editor.read().input.experiment, 'lateral_length')).toBe(false)
    ui.editor.discard(); expect(ui.editor.read().input).toStrictEqual(original)
  })

  it('distinguishes null, empty overrides and omitted inheritance without physical initialization', () => {
    const ui = mount()
    ui.choose(ui.path(['experiment', 'lateral_bc']).querySelector('select')!, 'omit')
    expect(Object.hasOwn(ui.editor.read().input.experiment, 'lateral_bc')).toBe(false)
    ui.choose(ui.path(['experiment', 'lateral_bc']).querySelector('select')!, 'null')
    expect(Reflect.get(ui.editor.read().input.experiment, 'lateral_bc')).toBeNull()
    ui.section('microstructure')
    ui.choose(ui.path(['experiment', 'microstructure', 'grain_boundaries']).querySelector('select')!, 'value')
    expect(Reflect.get(ui.editor.read().input.experiment, 'microstructure')).toStrictEqual({ grain_boundaries: [] })
    ui.choose(ui.path(['experiment', 'microstructure']).querySelector('select')!, 'omit')
    expect(Object.hasOwn(ui.editor.read().input.experiment, 'microstructure')).toBe(false)
    ui.section('tolerance')
    ui.choose(ui.path(['experiment', 'componentwise_atol']).querySelector('select')!, 'value')
    expect(Reflect.get(ui.editor.read().input.experiment, 'componentwise_atol')).toStrictEqual({})
    expect(ui.editor.read().incomplete).toContainEqual(['experiment', 'componentwise_atol', 'minimum_atol'])
  })

  it('keeps pending protocol array text with its row through move, copy, remove and navigation', () => {
    const raw = input(); if (raw.experiment.kind !== 'jv_2d') throw Error('fixture')
    raw.experiment.jv_2d_protocol = protocol()
    const ui = mount(raw); ui.section('protocol')
    const row = ['experiment', 'jv_2d_protocol', 'x_coordinates_m', 1]
    ui.text(ui.path(row).querySelector('input')!, '')
    const array = () => [...ui.root.querySelectorAll<HTMLElement>('[data-array]')].find(node => node.dataset.array === JSON.stringify(row.slice(0, -1)))!
    array().querySelectorAll<HTMLElement>('[data-action=move-up]')[1].click()
    expect(ui.editor.read().incomplete).toContainEqual([...row.slice(0, -1), 0])
    array().querySelectorAll<HTMLElement>('[data-action=copy-row]')[0].click()
    ui.section('controls'); ui.section('protocol')
    expect(ui.path([...row.slice(0, -1), 0]).querySelector('input')!.value).toBe('')
    expect(Reflect.get(ui.editor.read().input.experiment, 'jv_2d_protocol').x_coordinates_m).toStrictEqual(['250 nm', '250 nm', 0, '500 nm'])
    array().querySelectorAll<HTMLElement>('[data-action=remove-row]')[0].click()
    expect(ui.editor.read().incomplete).not.toContainEqual([...row.slice(0, -1), 0])
    expect(ui.editor.apply()).toBe(true) // Invalid coordinate order reaches authoritative validation.
  })

  it('edits both size aliases independently and preserves explicit unit strings', () => {
    const raw = input(); raw.experiment = { kind: 'voc_grain_sweep', grain_sizes_nm: [500, '1 um'], grain_sizes: null }
    const ui = mount(raw)
    ui.text(ui.path(['experiment', 'grain_sizes_nm', 0]).querySelector('input')!, '0.5 um')
    expect(ui.editor.read().input.experiment).toStrictEqual({ kind: 'voc_grain_sweep', grain_sizes_nm: ['0.5 um', '1 um'], grain_sizes: null })
    expect(ui.root.textContent).toContain('Bare numbers use nm')
    ui.editor.setInput(input()); expect(ui.editor.read().input).toStrictEqual(input())
  })

  it('joins device edits without overwriting experiment controls and disposes safely', () => {
    const ui = mount(), name = ui.root.querySelector<HTMLInputElement>('[data-role=experiment-device] [data-field=name][data-area=basic] input')!
    ui.text(name, 'Edited absorber')
    expect(ui.editor.read().dirty).toBe(true)
    expect(ui.editor.apply()).toBe(true)
    expect(ui.onApply.mock.calls[0][0].device.layers[0].name).toBe('Edited absorber')
    expect(Object.is(ui.onApply.mock.calls[0][0].experiment.V_max, -0)).toBe(true)
    ui.editor.dispose(); ui.text(name, 'late'); expect(ui.editor.apply()).toBe(false)
    expect(ui.onApply).toHaveBeenCalledTimes(1)
  })
})

function response(raw: SpatialExperimentInput) {
  return JSON.stringify({ schema: 'solarlab.configuration-preview.v1', kind: 'experiment', status: 'prepared_pending_dependencies',
    can_execute: false, input: raw, resolved: { schema: 'solarlab.resolved-spatial-experiment-preparation.v1', id: raw.id, can_execute: false },
    identity: { scope: 'configuration_content', content_sha256: 'a'.repeat(64), default_catalog_sha256: 'b'.repeat(64),
      resource_library_sha256: 'c'.repeat(64), configuration_schema_sha256: configurationSchemaSha256 } })
    .replace('"V_max":0', '"V_max":-0.0').replace('"phi_left":0', '"phi_left":-0.0')
}

describe('composed experiment preparation surface', () => {
  it('uses the accepted schema-bound client, preserves original response on reopen and displays actual errors safely', async () => {
    const raw = input(), json = response(raw), fetcher = vi.fn().mockResolvedValue(new Response(json, { headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetcher)
    const root = document.createElement('div'); document.body.append(root)
    const ui = mountSpatialExperimentPreview(root, { endpoint: '/configuration-preview/experiment', input: raw }); mounted.push(ui)
    await ui.ready
    expect(fetcher.mock.calls[0][1].headers['X-Solarlab-Configuration-Schema']).toBe(configurationSchemaSha256)
    expect(root.querySelector('[data-panel=configuration-preview]')?.getAttribute('data-state')).toBe('ready')
    expect(root.querySelector('pre')?.textContent).toBe(json)
    fetcher.mockResolvedValueOnce(new Response(JSON.stringify({ detail: { status: 'unresolved', can_execute: false,
      code: 'configuration_validation', message: '<img src=x>', field_errors: [{ loc: ['experiment', 'Nx'], msg: '<script>bad</script>' }] } }),
      { status: 422, headers: { 'Content-Type': 'application/json' } }))
    expect(ui.apply()).toBe(true); await ui.ready
    expect(root.querySelector('[data-panel=configuration-preview]')?.getAttribute('data-state')).toBe('unresolved')
    expect(root.querySelector('script,img')).toBeNull(); expect(root.textContent).toContain('Nx')
    await ui.reopen(json); expect(ui.read().input).toStrictEqual(raw)
  })

  it('aborts only owned preview fetches on replacement/disposal and ignores late responses', async () => {
    const requests: { signal: AbortSignal; resolve: (value: Response) => void }[] = []
    vi.stubGlobal('fetch', vi.fn((_endpoint, init) => new Promise<Response>(resolve => requests.push({ signal: init.signal, resolve }))))
    const root = document.createElement('div'); document.body.append(root)
    const ui = mountSpatialExperimentPreview(root, { endpoint: '/a', input: input() }); mounted.push(ui)
    const second = { ...input(), id: 'second' }, ready = ui.select(second, '/b')
    expect(requests[0].signal.aborted).toBe(true)
    requests[1].resolve(new Response(response(second), { headers: { 'Content-Type': 'application/json' } })); await ready
    requests[0].resolve(new Response(response(input()), { headers: { 'Content-Type': 'application/json' } }))
    await Promise.resolve(); expect(root.textContent).toContain('second')
    const third = ui.select(input(), '/c'); ui.dispose()
    expect(requests[2].signal.aborted).toBe(true)
    requests[2].resolve(new Response(response(input()), { headers: { 'Content-Type': 'application/json' } })); await third
    expect(root.childElementCount).toBe(0)
  })
})
