import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SweepInput, DeviceInput, JVExperimentInput } from '../generated/configuration-inputs'
import { configurationSchemaSha256 } from '../generated/configuration-schema'
import { mountSweepEditor, mountSweepPreview, sweepTargets } from './sweep'
import type { SweepEditor } from './sweep'

const mounted: SweepEditor[] = []
function device(): DeviceInput {
  return { schema_version: 'solarlab.device-preparation.v1', id: 'device', source_format: 'canonical',
    layers: [{ id: 'pvk', name: '<img src=x> PVK', role: 'absorber', thickness: '400 nm', material: 'named', parameters: { D_ion: -0, P0: 10, chi: '4 eV' } },
      { id: 'etl', name: 'ETL', role: 'ETL', thickness: '20 nm', parameters: { chi: '4.1 eV' } }],
    settings: { phi_left: -0 }, materials: [{ id: 'named', name: 'Supplied material', parameters: { mu_n: '0.00020000000000000001 m^2/(V s)' } }], interfaces: [], electrical_grid: [] }
}
function input(): SweepInput {
  return { schema_version: 'solarlab.sweep-preparation.v1', id: 'sweep', base: device(), reference_id: null,
    axes: [{ id: 'diffusion', target: { family: 'layer_parameter', owner_id: 'pvk', parameter: ['D_ion'] }, coordinates: [{ kind: 'value', value: -0 }, { kind: 'value', value: '1e-18 m^2/s' }, { kind: 'omit' }] },
      { id: 'concentration', target: { family: 'layer_parameter', owner_id: 'pvk', parameter: ['P0'] }, coordinates: [{ kind: 'value', value: 0 }] }] }
}
function mount(raw = input()) {
  const root = document.createElement('div'); document.body.append(root)
  const onApply = vi.fn(), editor = mountSweepEditor(root, { input: raw, onApply }); mounted.push(editor)
  const path = (parts: (string | number)[]) => [...root.querySelectorAll<HTMLElement>('[data-path]')].find(node => node.dataset.path === JSON.stringify(parts))!
  const select = (role: string, value: string) => { const element = root.querySelector<HTMLSelectElement>(`[data-role=${role}]`)!; element.value = value; element.dispatchEvent(new Event('change', { bubbles: true })) }
  const text = (parts: (string | number)[], value: string) => { const element = path(parts).querySelector('input')!; element.value = value; element.dispatchEvent(new Event('input', { bubbles: true })) }
  const click = (action: string) => root.querySelector<HTMLElement>(`[data-action=${action}]`)!.click()
  return { root, editor, onApply, path, select, text, click }
}
afterEach(() => { for (const value of mounted.splice(0)) value.dispose(); document.body.replaceChildren(); vi.unstubAllGlobals() })

describe('explicit sweep editing', () => {
  it('preserves no-op input, unknown members, units and numeric signed zero without rendering markup', () => {
    const raw = input(); Reflect.set(raw.base, 'future_payload', { retained: [-0, null, false] }); const before = structuredClone(raw), ui = mount(raw)
    ui.select('sweep-axis', '1'); ui.select('sweep-axis', '0'); ui.select('sweep-coordinate', '1'); ui.select('sweep-coordinate', '2')
    expect(ui.editor.read().input).toStrictEqual(before); expect(ui.editor.apply()).toBe(true)
    expect(ui.onApply.mock.calls[0][0]).toStrictEqual(before); expect(raw).toStrictEqual(before)
    expect(ui.root.querySelector('img,script')).toBeNull()
    ui.onApply.mock.calls[0][0].axes.length = 0; expect(ui.editor.read().input.axes).toHaveLength(2)
  })
  it('edits coordinate text without moving focus/caret and keeps blank edits incomplete', () => {
    const ui = mount(), path = ['axes', 0, 'coordinates', 0]
    const element = ui.path(path).querySelector('input')!; element.focus(); element.value = '0 m^2/s'; element.setSelectionRange(2, 2); element.dispatchEvent(new Event('input'))
    expect(document.activeElement).toBe(element); expect(element.selectionStart).toBe(2)
    const coordinates = ui.root.querySelector<HTMLSelectElement>('[data-role=sweep-coordinate]')!
    expect(coordinates.selectedOptions[0].textContent).toBe('1: 0 m^2/s')
    expect(ui.editor.read().input.axes[0].coordinates[0]).toStrictEqual({ kind: 'value', value: '0 m^2/s' })
    ui.text(['axes', 0, 'id'], 'mobility control')
    expect(ui.root.querySelector<HTMLSelectElement>('[data-role=sweep-axis]')!.selectedOptions[0].textContent).toBe('1: mobility control')
    ui.text(path, ''); expect(ui.editor.apply()).toBe(false)
    expect(coordinates.selectedOptions[0].textContent).toBe('1: (empty) (incomplete)')
    expect(ui.editor.read().incomplete).toContainEqual(path)
    ui.select('sweep-axis', '1'); ui.select('sweep-axis', '0')
    expect(ui.path(path).querySelector('input')!.value).toBe('')
    ui.editor.discard(); expect(ui.editor.read().input).toStrictEqual(input())
  })
  it('keeps pending coordinates with their intended row through reorder, copy and removal', () => {
    const ui = mount(); ui.text(['axes', 0, 'coordinates', 0], '')
    ui.click('move-coordinate-down'); expect(ui.editor.read().incomplete).toContainEqual(['axes', 0, 'coordinates', 1])
    ui.click('copy-coordinate'); expect(ui.editor.read().incomplete).toContainEqual(['axes', 0, 'coordinates', 1])
    ui.select('sweep-coordinate', '1'); expect(ui.path(['axes', 0, 'coordinates', 1]).querySelector('input')!.value).toBe('')
    ui.click('remove-coordinate'); expect(ui.editor.read().incomplete).toEqual([])
    expect(ui.editor.read().input.axes[0].coordinates[1]).toStrictEqual({ kind: 'value', value: -0 })
    ui.click('move-axis-down'); expect(ui.root.querySelector<HTMLSelectElement>('[data-role=sweep-axis]')!.value).toBe('1')
    expect(ui.editor.read().input.axes[1]!.id).toBe('diffusion')
  })
  it('uses generated nullable/required metadata, stable targets and explicit empty creation', () => {
    const raw = input(); raw.axes = [{ id: 'contact', target: { family: 'setting', parameter: ['work_function_left_eV'] }, coordinates: [{ kind: 'value', value: null }, { kind: 'omit' }] }]
    const ui = mount(raw); expect(ui.editor.apply()).toBe(true)
    expect(ui.editor.read().input.axes[0].coordinates).toStrictEqual(raw.axes[0].coordinates)
    ui.click('add-coordinate'); expect(ui.editor.read().incomplete).toContainEqual(['axes', 0, 'coordinates', 2])
    ui.click('remove-coordinate'); ui.click('add-axis')
    expect(ui.editor.read().input.axes[1]).toStrictEqual({ id: '', target: {}, coordinates: [] })
    expect(ui.editor.apply()).toBe(false)
    ui.text(['axes', 1, 'id'], 'cbo')
    const index = sweepTargets(raw.base).findIndex(value => value.target.family === 'cbo')
    ui.select('sweep-target', String(index)); ui.click('add-coordinate'); ui.text(['axes', 1, 'coordinates', 0], '-200 meV')
    expect(ui.editor.apply()).toBe(true)
    expect(ui.editor.read().input.axes[1]!.target).toStrictEqual({ family: 'cbo', owner_id: 'etl', reference_id: 'pvk', parameter: ['chi'] })
    expect(ui.editor.read().input.base).toStrictEqual(raw.base)
  })
  it('preserves base J-V history and joins selected device edits without normalizing the sweep', () => {
    const raw = input(), base: JVExperimentInput = { schema_version: 'solarlab.experiment-preparation.v1', id: 'jv', device: device(), experiment: { kind: 'jv', v_rate: '40 mV/s', V_max: null, illuminated: false } }
    raw.base = base; const ui = mount(raw)
    const name = ui.root.querySelector<HTMLInputElement>('[data-field=name][data-area=basic] input')!
    name.value = 'Changed name'; name.dispatchEvent(new Event('input', { bubbles: true }))
    expect(ui.editor.apply()).toBe(true)
    expect((ui.onApply.mock.calls[0][0].base as JVExperimentInput).experiment).toStrictEqual(base.experiment)
    expect((ui.onApply.mock.calls[0][0].base as JVExperimentInput).device.layers[0].id).toBe('pvk')
    ui.editor.dispose(); name.value = 'late'; name.dispatchEvent(new Event('input', { bubbles: true })); expect(ui.editor.apply()).toBe(false)
    expect(ui.onApply).toHaveBeenCalledTimes(1)
  })
})

function response(raw: SweepInput) {
  const point = { id: 'point:abc', status: 'prepared_pending_dependencies', can_execute: false, simulation_status: 'not_started', execution_identity: null,
    applied_input: raw.base, coordinates: { diffusion: { kind: 'value', value: -0 } }, reference: { status: 'reference_missing', reason: '<script>blank source</script>' }, effective_targets: { diffusion: { value: -0, origin: 'layer_input:pvk' } } }
  return JSON.stringify({ schema: 'solarlab.configuration-preview.v1', kind: 'sweep', status: 'prepared_pending_dependencies', can_execute: false, input: raw,
    resolved: { schema: 'solarlab.resolved-sweep-preparation.v1', id: raw.id, can_execute: false, points: [point],
      reference_scope: { interpretation: { interface_density_choice: 'historical_areal_input', interface_density_axes: [{ axis_id: 'Nt', reported_unit: 'cm^-3', input_unit: 'cm^-2', resolved_unit: 'm^-2' }] } } }, identity: { scope: 'configuration_content', content_sha256: 'a'.repeat(64), default_catalog_sha256: 'b'.repeat(64), resource_library_sha256: 'c'.repeat(64), configuration_schema_sha256: configurationSchemaSha256 } })
    .replaceAll('"D_ion":0', '"D_ion":-0.0').replaceAll('"phi_left":0', '"phi_left":-0.0').replaceAll('"value":0}', '"value":-0.0}').replace('"value":0,"origin"', '"value":-0.0,"origin"')
}
describe('sweep preview integration', () => {
  it('labels the conduction-band offset separately from the resolved electron affinities', async () => {
    const raw = input()
    raw.axes = [{ id: 'CBO', target: { family: 'cbo', owner_id: 'etl', reference_id: 'pvk', parameter: ['chi'] }, coordinates: [{ kind: 'value', value: '-1 eV' }] }]
    const document = JSON.parse(response(raw))
    document.resolved.points[0].coordinates = { CBO: { kind: 'value', value: '-1 eV' } }
    document.resolved.points[0].effective_targets = { CBO: { value: 5, offset_eV: -1, reference_affinity_eV: 4, origin: 'layer_input:etl' } }
    document.resolved.points[0].applied_input.layers[1].parameters.chi = '5 eV'
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(document), { headers: { 'Content-Type': 'application/json' } })))
    const root = window.document.createElement('div'); window.document.body.append(root)
    const ui = mountSweepPreview(root, { input: raw, endpoints: { sweep: '/sweep', device: '/device', experiment: '/experiment' } }); mounted.push(ui)
    await ui.ready
    const text = root.querySelector('[data-role=point-effective-values]')!.textContent
    expect(text).toContain('CBO: -1 eV')
    expect(text).toContain('ETL electron affinity: 5 eV')
    expect(text).toContain('absorber electron affinity: 4 eV')
    expect(text).not.toContain('CBO: 5')
  })
  it('uses original response bytes for reopen and selected-point input through the accepted client', async () => {
    const raw = input(); raw.axes = [raw.axes[0]]
    const json = response(raw), fetcher = vi.fn().mockImplementation(async () => new Response(json, { headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetcher)
    const root = document.createElement('div'); document.body.append(root)
    const ui = mountSweepPreview(root, { input: raw, endpoints: { sweep: '/sweep', device: '/device', experiment: '/experiment' } }); mounted.push(ui)
    await ui.ready
    expect(fetcher.mock.calls[0][1].headers['X-Solarlab-Configuration-Schema']).toBe(configurationSchemaSha256)
    expect(root.textContent).toContain('reference_missing'); expect(root.querySelector('script')).toBeNull()
    const interpretation = root.querySelector<HTMLElement>('[data-role=sweep-reference-interpretation]')!
    expect(interpretation.hidden).toBe(false)
    expect(interpretation.textContent).toContain('historical_areal_input')
    expect(interpretation.textContent).toContain('reported cm^-3; declared cm^-2; resolved m^-2')
    expect(interpretation.textContent).toContain('no verified native SCAPS input')
    expect(root.querySelector('[data-role=point-effective-values]')!.textContent).toContain('-0')
    expect(root.querySelector('pre')!.textContent).toBe(json)
    await ui.openPoint('point:abc')
    expect(fetcher.mock.calls[1][0]).toBe('/device')
    expect(fetcher.mock.calls[1][1].body).toContain('"D_ion":-0.0')
    await ui.reopen(json); expect(ui.read().input).toStrictEqual(raw)
  })
  it('ignores late responses and closes only owned previews on replacement/disposal', async () => {
    const requests: { signal: AbortSignal; resolve(value: Response): void }[] = []
    vi.stubGlobal('fetch', vi.fn((_url, init) => new Promise<Response>(resolve => requests.push({ signal: init.signal, resolve }))))
    const root = document.createElement('div'); document.body.append(root)
    const raw = input(), ui = mountSweepPreview(root, { input: raw, endpoints: { sweep: '/sweep', device: '/device', experiment: '/experiment' } }); mounted.push(ui)
    const second = { ...raw, id: 'second' }, waiting = ui.select(second)
    expect(requests[0].signal.aborted).toBe(true)
    requests[1].resolve(new Response(response(second), { headers: { 'Content-Type': 'application/json' } })); await waiting
    requests[0].resolve(new Response(response(raw), { headers: { 'Content-Type': 'application/json' } })); await Promise.resolve()
    expect(ui.read().input.id).toBe('second')
    const third = ui.select(raw); ui.dispose(); expect(requests[2].signal.aborted).toBe(true)
    requests[2].resolve(new Response(response(raw), { headers: { 'Content-Type': 'application/json' } })); await third
    expect(root.childElementCount).toBe(0)
  })
})
