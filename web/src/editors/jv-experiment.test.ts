import { afterEach, describe, expect, it, vi } from 'vitest'
import type { JVExperimentInput, JVExperimentInputDefinitions } from '../generated/configuration-inputs'
import { configurationSchemaSha256 } from '../generated/configuration-schema'
import { mountJVExperimentEditor, mountJVExperimentPreview } from './jv-experiment'
import type { JVExperimentEditor } from './jv-experiment'

const mounted: JVExperimentEditor[] = []
function protocol(): JVExperimentInputDefinitions.JVProtocolInput {
  return { experiment: 'jv_hysteresis', initial_state_source: 'finite_time_dc_preconditioned', pre_bias_V: -0,
    soak_duration_s: '0 s', dwell_duration_s: 0.5, temperature_K: '300 K',
    illumination_history: [
      { phase: 'dark seed <script>x</script>', condition: 'dark', duration_s: 0, intensity_suns: null,
        photon_flux_m2_s: null, relative_generation_change: null, source_reference: null },
      { phase: 'forward', condition: 'baseline', duration_s: '2 s', intensity_suns: 1,
        photon_flux_m2_s: null, relative_generation_change: null, source_reference: 'device_optics' }],
    scan: { axis: 'voltage_V', direction: 'ascending_then_descending', start: -0, stop: '1.2 V', rate_V_s: '40 mV/s' },
    ac_excitation: null, voc_search: null,
    dc_settle: { kind: 'not_applicable', duration_s: null, max_carrier_area_rate_A_m2: null,
      max_ion_area_rate_A_m2: null, max_ionic_face_current_A_m2: null, max_face_current_spread_A_m2: null },
    sampling: { axis: 'voltage_V', mode: 'piecewise_linear', values: [-0, '1.2 V', '1.2 V', 0] },
    implicit_legacy_protocol: false, schema_version: 1 }
}
function input(): JVExperimentInput {
  return { schema_version: 'solarlab.experiment-preparation.v1', id: 'jv_test',
    device: { schema_version: 'solarlab.device-preparation.v1', id: 'device', source_format: 'canonical',
      layers: [{ id: 'active', name: '<img src=x onerror=alert(1)>', role: 'absorber', thickness: '400 nm',
        parameters: { mu_n: '0.00020000000000000001 m^2/(V s)', Nc300: null, D_ion: -0, P0: 1e25 }, bulk_defects: [] }],
      settings: { phi_left: -0 }, grain_boundaries: [], electrical_grid: [] },
    experiment: { kind: 'jv', N_grid: 60, V_max: '1.2 V', v_rate: '0.040000000000000001 V/s', illuminated: false,
      waveform: { schema_version: 1, start_voltage_V: -0, dark_seed_s: 0, dark_prep_s: '0 s',
        branch_dwell_s: '500 ms', turnaround_s: 0, turnaround_dark: false, uniform_generation_rate_m3_s: null },
      waveform_controls: null, experiment_protocol: protocol() } }
}
function mount(value = input()) {
  const root = document.createElement('div'); document.body.append(root)
  const onApply = vi.fn(), editor = mountJVExperimentEditor(root, { input: value, onApply }); mounted.push(editor)
  const path = (parts: (string | number)[]) => [...root.querySelectorAll<HTMLElement>('[data-path]')].find(node => node.dataset.path === JSON.stringify(parts))!
  const choose = (node: HTMLSelectElement, value: string) => { node.value = value; node.dispatchEvent(new Event('change', { bubbles: true })) }
  const text = (node: HTMLInputElement, value: string) => { node.value = value; node.dispatchEvent(new Event('input', { bubbles: true })) }
  const section = (value: string) => choose(root.querySelector<HTMLSelectElement>('[data-role=experiment-section]')!, value)
  const array = (parts: (string | number)[]) => [...root.querySelectorAll<HTMLElement>('[data-array]')].find(node => node.dataset.array === JSON.stringify(parts))!
  return { root, editor, onApply, path, choose, text, section, array }
}
afterEach(() => { for (const editor of mounted.splice(0)) editor.dispose(); document.body.replaceChildren(); vi.unstubAllGlobals() })

describe('J-V declaration drafts', () => {
  it('retains exact no-op types, zero signs, absent fields, unknown payloads and safe supplied labels', () => {
    const raw = input(); Reflect.set(raw.device, 'extension', { n: -0, list: [], nullable: null })
    Reflect.set(raw.experiment, 'future_control', { order: [2, 1] })
    const before = structuredClone(raw), ui = mount(raw)
    for (const section of ['controls', 'waveform', 'tolerance', 'protocol', 'history', 'sampling']) ui.section(section)
    expect(ui.editor.read().input).toStrictEqual(before); expect(ui.editor.read().incomplete).toEqual([])
    expect(ui.editor.apply()).toBe(true); expect(ui.onApply.mock.calls[0][0]).toStrictEqual(before)
    expect(ui.root.querySelector('script,img')).toBeNull()
    ui.onApply.mock.calls[0][0].device.layers.length = 0
    expect(ui.editor.read().input.device.layers).toHaveLength(1); expect(raw).toStrictEqual(before)
  })

  it('changes only explicit input without moving focus or coercing units and empty text', () => {
    const ui = mount(), original = ui.editor.read().input
    const rate = ui.path(['experiment', 'v_rate']).querySelector('input')!
    rate.focus(); rate.value = '40 mV/s'; rate.setSelectionRange(4, 4); rate.dispatchEvent(new Event('input'))
    expect(document.activeElement).toBe(rate); expect(rate.selectionStart).toBe(4)
    expect(ui.editor.read().input).toStrictEqual({ ...original, experiment: { ...original.experiment, v_rate: '40 mV/s' } })
    ui.choose(ui.path(['experiment', 'request_api']).querySelector('[data-role=value-source]')!, 'value')
    ui.choose(ui.path(['experiment', 'request_api']).querySelector('[data-role=scalar-value]')!, 'jv_endpoint')
    expect(Reflect.get(ui.editor.read().input.experiment, 'request_api')).toBe('jv_endpoint')
    expect(Object.hasOwn(ui.editor.read().input.experiment, 'n_points')).toBe(false)
    const count = ui.path(['experiment', 'N_grid']).querySelector('input')!
    ui.text(count, '0'); expect(ui.editor.read().input.experiment.N_grid).toBe(0)
    expect(ui.editor.apply()).toBe(true) // Semantic errors belong to the server.
    ui.text(count, ''); expect(ui.editor.apply()).toBe(false)
    expect(ui.editor.read().incomplete).toContainEqual(['experiment', 'N_grid'])
    ui.section('waveform'); ui.section('controls')
    expect(ui.path(['experiment', 'N_grid']).querySelector('input')!.value).toBe('')
    ui.editor.discard(); expect(ui.editor.read().input).toStrictEqual(original)
  })

  it('distinguishes explicit generation zero, null, omitted tolerances and empty declaration creation', () => {
    const ui = mount(); ui.section('waveform')
    const generation = ui.path(['experiment', 'waveform', 'uniform_generation_rate_m3_s'])
    ui.choose(generation.querySelector('select')!, 'value'); ui.text(generation.querySelector('input')!, '0 m^-3/s')
    expect(Reflect.get(ui.editor.read().input.experiment, 'waveform').uniform_generation_rate_m3_s).toBe('0 m^-3/s')
    ui.choose(generation.querySelector('select')!, 'null')
    expect(Reflect.get(ui.editor.read().input.experiment, 'waveform').uniform_generation_rate_m3_s).toBeNull()
    expect(Object.is(Reflect.get(ui.editor.read().input.experiment, 'waveform').start_voltage_V, -0)).toBe(true)
    ui.section('tolerance')
    ui.choose(ui.path(['experiment', 'waveform_controls']).querySelector('select')!, 'omit')
    expect(Object.hasOwn(ui.editor.read().input.experiment, 'waveform_controls')).toBe(false)
    ui.choose(ui.path(['experiment', 'waveform_controls']).querySelector('select')!, 'value')
    expect(Reflect.get(ui.editor.read().input.experiment, 'waveform_controls')).toStrictEqual({})
    expect(ui.editor.read().incomplete).toContainEqual(['experiment', 'waveform_controls', 'rtol'])
    expect(ui.editor.apply()).toBe(false)
    ui.choose(ui.path(['experiment', 'waveform_controls']).querySelector('select')!, 'null')
    expect(ui.editor.read().incomplete).toEqual([]); expect(ui.editor.apply()).toBe(true)
  })

  it('keeps unfinished history edits attached through explicit move, stored-row copy and removal', () => {
    const ui = mount(); ui.section('history')
    const rows = ['experiment', 'experiment_protocol', 'illumination_history']
    ui.text(ui.path([...rows, 0, 'duration_s']).querySelector('input')!, '')
    ui.array(rows).querySelector<HTMLElement>('[data-action=move-down]')!.click()
    expect(ui.editor.read().incomplete).toContainEqual([...rows, 1, 'duration_s'])
    expect(ui.array(rows).querySelector<HTMLSelectElement>('[data-role=row-selection]')!.value).toBe('1')
    ui.array(rows).querySelector<HTMLElement>('[data-action=copy-row]')!.click()
    expect(ui.editor.read().incomplete).toContainEqual([...rows, 1, 'duration_s'])
    ui.choose(ui.array(rows).querySelector('[data-role=row-selection]')!, '1')
    ui.section('controls'); ui.section('history')
    expect(ui.path([...rows, 1, 'duration_s']).querySelector('input')!.value).toBe('')
    ui.array(rows).querySelector<HTMLElement>('[data-action=remove-row]')!.click()
    expect(ui.editor.read().incomplete).toEqual([])
    const history = Reflect.get(ui.editor.read().input.experiment, 'experiment_protocol').illumination_history
    expect(history.map((row: { phase: string }) => row.phase)).toStrictEqual(['forward', 'dark seed <script>x</script>'])
    expect(ui.editor.apply()).toBe(true) // The intentional order mismatch is retained for validation.
  })

  it('edits ordered sample units and empty rows without repairing a mismatched protocol', () => {
    const ui = mount(); ui.section('sampling')
    const rows = ['experiment', 'experiment_protocol', 'sampling', 'values']
    ui.text(ui.path([...rows, 0]).querySelector('input')!, '0 mV')
    ui.array(rows).querySelector<HTMLElement>('[data-action=add-row]')!.click()
    expect(ui.editor.read().incomplete).toContainEqual([...rows, 4])
    ui.text(ui.path([...rows, 4]).querySelector('input')!, '-0 mV')
    expect(Reflect.get(ui.editor.read().input.experiment, 'experiment_protocol').sampling.values).toStrictEqual(['0 mV', '1.2 V', '1.2 V', 0, '-0 mV'])
    expect(ui.editor.apply()).toBe(true)
    ui.editor.discard(); expect(ui.editor.read().input).toStrictEqual(input())
  })

  it('joins device edits, retains unsupported dark payloads and stops callbacks on disposal', () => {
    const raw = input(); raw.experiment = { kind: 'dark_jv', V_max: -0 }
    Reflect.set(raw.experiment, 'waveform', { retained: true })
    const ui = mount(raw), name = ui.root.querySelector<HTMLInputElement>('[data-role=experiment-device] [data-field=name][data-area=basic] input')!
    ui.text(name, 'Changed absorber')
    expect(ui.editor.read().dirty).toBe(true); expect(ui.editor.apply()).toBe(true)
    expect(ui.onApply.mock.calls[0][0].device.layers[0].name).toBe('Changed absorber')
    expect(ui.onApply.mock.calls[0][0].experiment).toStrictEqual(raw.experiment)
    expect(ui.root.textContent).toContain('Additional supplied fields retained')
    ui.editor.dispose(); ui.text(name, 'late'); expect(ui.editor.apply()).toBe(false)
    expect(ui.onApply).toHaveBeenCalledTimes(1)
  })
})

function response(raw: JVExperimentInput) {
  return JSON.stringify({ schema: 'solarlab.configuration-preview.v1', kind: 'experiment', status: 'prepared_pending_dependencies',
    can_execute: false, input: raw, resolved: { schema: 'solarlab.resolved-jv-experiment-preparation.v1', id: raw.id, can_execute: false },
    identity: { scope: 'configuration_content', content_sha256: 'a'.repeat(64), default_catalog_sha256: 'b'.repeat(64),
      resource_library_sha256: 'c'.repeat(64), configuration_schema_sha256: configurationSchemaSha256 } })
    .replaceAll('"start_voltage_V":0', '"start_voltage_V":-0.0').replace('"phi_left":0', '"phi_left":-0.0')
    .replace('"D_ion":0', '"D_ion":-0.0').replace('"pre_bias_V":0', '"pre_bias_V":-0.0')
    .replace('"start":0', '"start":-0.0').replace('"values":[0,', '"values":[-0.0,')
}
describe('J-V preview composition', () => {
  it('posts through the schema-bound client, shows exact errors, exports original text and reopens input', async () => {
    const raw = input(), json = response(raw), fetcher = vi.fn().mockResolvedValue(new Response(json, { headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetcher)
    const root = document.createElement('div'); document.body.append(root)
    const ui = mountJVExperimentPreview(root, { endpoint: '/configuration-preview/experiment', input: raw }); mounted.push(ui)
    await ui.ready
    expect(fetcher.mock.calls[0][1].headers['X-Solarlab-Configuration-Schema']).toBe(configurationSchemaSha256)
    expect(fetcher.mock.calls[0][1].body).toContain('"start_voltage_V":-0.0')
    expect(root.querySelector('[data-panel=configuration-preview]')?.getAttribute('data-state')).toBe('ready')
    expect(root.querySelector('pre')?.textContent).toBe(json)
    fetcher.mockResolvedValueOnce(new Response(JSON.stringify({ detail: { status: 'unresolved', can_execute: false,
      code: 'configuration_validation', message: '<img src=x>', field_errors: [{ loc: ['experiment', 'v_rate'], msg: '<script>bad rate</script>' }] } }),
      { status: 422, headers: { 'Content-Type': 'application/json' } }))
    expect(ui.apply()).toBe(true); await ui.ready
    expect(root.querySelector('[data-panel=configuration-preview]')?.getAttribute('data-state')).toBe('unresolved')
    expect(root.querySelector('script,img')).toBeNull(); expect(root.textContent).toContain('v_rate')
    await ui.reopen(json); expect(ui.read().input).toStrictEqual(raw)
  })

  it('keeps new branch selection and only cancels owned previews when replaced or disposed', async () => {
    const requests: { signal: AbortSignal; resolve: (value: Response) => void }[] = []
    vi.stubGlobal('fetch', vi.fn((_endpoint, init) => new Promise<Response>(resolve => requests.push({ signal: init.signal, resolve }))))
    const root = document.createElement('div'); document.body.append(root)
    const ui = mountJVExperimentPreview(root, { endpoint: '/a', input: input() }); mounted.push(ui)
    const second: JVExperimentInput = { ...input(), id: 'dark', experiment: { kind: 'dark_jv' } }, ready = ui.select(second, '/b')
    expect(requests[0].signal.aborted).toBe(true)
    requests[1].resolve(new Response(response(second), { headers: { 'Content-Type': 'application/json' } })); await ready
    requests[0].resolve(new Response(response(input()), { headers: { 'Content-Type': 'application/json' } }))
    await Promise.resolve(); expect(ui.read().input).toStrictEqual(second)
    const third = ui.select(input(), '/c'); ui.dispose(); expect(requests[2].signal.aborted).toBe(true)
    requests[2].resolve(new Response(response(input()), { headers: { 'Content-Type': 'application/json' } })); await third
    expect(root.childElementCount).toBe(0)
  })
})
