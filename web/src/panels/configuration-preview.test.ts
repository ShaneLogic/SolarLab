import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ConfigurationPreviewError, previewDeviceConfiguration, previewTandemConfiguration } from '../configuration-client'
import type { ConfigurationPreviewDocument } from '../configuration-client'
import { mountConfigurationPreviewPanel } from './configuration-preview'
import type { ConfigurationPreviewPanel, ConfigurationPreviewSelection } from './configuration-preview'

vi.mock('../configuration-client', async importOriginal => ({
  ...await importOriginal<typeof import('../configuration-client')>(),
  previewDeviceConfiguration: vi.fn(), previewTandemConfiguration: vi.fn(),
}))

const panels: ConfigurationPreviewPanel[] = []
let blobs: Map<string, Blob>
let downloads: { url: string; name: string }[]
const makeURL = vi.fn((_blob: Blob) => 'blob:unconfigured')
const revokeURL = vi.fn((_url: string) => {})

function selection(id = 'device-a'): Extract<ConfigurationPreviewSelection, { kind: 'device' }> {
  return { endpoint: 'http://127.0.0.1:8123/configuration-preview/device', kind: 'device', input: {
    schema_version: 'solarlab.device-preparation.v1', id, source_format: 'canonical',
    layers: [], settings: { phi_left: -0, Phi: 0 },
  } }
}

function tandem(): ConfigurationPreviewSelection {
  const cell = selection().input
  return { endpoint: 'http://127.0.0.1:9555/configuration-preview/tandem', kind: 'tandem', input: {
    schema_version: 'solarlab.tandem-preparation.v1', id: 'tandem-c', source_schema_version: 1,
    device_type: 'tandem_2T_monolithic', light_direction: 'top_first', junction_model: 'ideal_ohmic',
    top_cell: cell, bottom_cell: cell, top_cell_reference: 'top', bottom_cell_reference: 'bottom', junction_stack: [],
  } }
}

function documentOf(id = 'device-a', kind: 'device' | 'tandem' = 'device'): ConfigurationPreviewDocument {
  const value: ConfigurationPreviewDocument['value'] = {
    schema: 'solarlab.configuration-preview.v1', kind, status: 'prepared_pending_dependencies', can_execute: false,
    input: { id, schema_version: `solarlab.${kind}-preparation.v1`, settings: { phi_left: -0, Phi: 0 }, clear: null },
    resolved: { id, schema: `solarlab.resolved-${kind}-preparation.v1`, can_execute: false,
      settings: { phi_left: 0, Phi: 0, T: 300 },
      layers: [{ id: 'layer-a', name: '<img src=x onerror="window.injected=true">',
        material_parameters: { optical_material: null, mu_n: 0 },
        provenance: [{ parameter: 'mu_n', effective_value: 0, origin: 'layer' }] }],
      interfaces: [{ id: 'interface-a', v_n: 0 }],
      unknown: { integer: 18446744073709551617n, explicit_null: null, zero: 0 },
      future_rows: [{ first: null }, { second: -0 }], capability_gaps: ['unqualified fixture'],
    },
    identity: { scope: 'configuration_content', content_sha256: 'a'.repeat(64),
      default_catalog_sha256: 'b'.repeat(64), resource_library_sha256: 'c'.repeat(64) },
  }
  // Controlled wire fixture, not a panel serializer or a scientific result.
  const json = JSON.stringify(value, (_key, item) => typeof item === 'bigint' ? `fixture-bigint:${item}`
    : Object.is(item, -0) ? 'fixture-negative-zero' : item)
    .replace(/"fixture-bigint:(-?\d+)"/g, '$1').replace(/"fixture-negative-zero"/g, '-0.0') + '\n'
  return { value, json }
}

function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function mount(input = selection()) {
  const root = document.createElement('div'); document.body.append(root)
  const panel = mountConfigurationPreviewPanel(root, input); panels.push(panel)
  return { root, panel }
}
function phase(root: HTMLElement) { return root.querySelector<HTMLElement>('[data-panel]')!.dataset.state }
function button(root: HTMLElement, label: string) {
  const found = [...root.querySelectorAll('button')].find(node => node.textContent === label)
  if (!found) throw new Error(`Missing button ${label}`)
  return found
}
function blobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  blobs = new Map(); downloads = []
  makeURL.mockImplementation(blob => { const key = `blob:preview-${blobs.size}`; blobs.set(key, blob); return key })
  const NativeURL = URL
  vi.stubGlobal('URL', class extends NativeURL {
    static createObjectURL = makeURL
    static revokeObjectURL = revokeURL
  })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ url: this.href, name: this.download })
  })
  vi.mocked(previewDeviceConfiguration).mockImplementation(async (_url, input) => documentOf(input.id))
  vi.mocked(previewTandemConfiguration).mockImplementation(async (_url, input) => documentOf(input.id, 'tandem'))
})

afterEach(() => {
  for (const panel of panels.splice(0)) panel.dispose()
  document.body.replaceChildren()
  vi.restoreAllMocks(); vi.unstubAllGlobals()
})

describe('configuration preview panel', () => {
  it('shows loading and submits exactly the explicit caller selection with an abort signal', async () => {
    const pending = deferred<ConfigurationPreviewDocument>()
    vi.mocked(previewDeviceConfiguration).mockReturnValue(pending.promise)
    const input = selection(), { root, panel } = mount(input)
    expect(phase(root)).toBe('loading')
    expect(root.textContent).toContain(String(input.endpoint))
    expect(root.textContent).toContain('physical qualification is not established')
    expect(previewDeviceConfiguration).toHaveBeenCalledWith(input.endpoint, input.input, expect.any(AbortSignal))
    pending.resolve(documentOf()); await panel.ready
    expect(phase(root)).toBe('ready')
  })

  it('displays effective settings, material/interface values and origins without losing unknown values', async () => {
    const { root, panel } = mount(); await panel.ready
    expect(root.textContent).toContain('prepared_pending_dependencies')
    expect(root.textContent).toContain('can_execute: false')
    expect(root.querySelector('[data-field=settings]')?.textContent).toContain('300')
    expect(root.querySelector('[data-field=material_parameters]')?.textContent).toContain('optical_material')
    expect(root.querySelector('[data-field=provenance]')?.textContent).toContain('effective_value')
    expect(root.querySelector('[data-field=interfaces]')?.textContent).toContain('interface-a')
    expect(root.querySelector('[data-field=unknown]')?.textContent).toContain('18446744073709551617')
    expect(root.querySelector('[data-section=input]')?.textContent).toContain('-0')
    expect(root.querySelector('[data-field=future_rows]')?.textContent).toContain('omitted')
    expect(root.querySelector('img,script')).toBeNull()
    expect(root.textContent).toContain('<img src=x onerror=')
  })

  it('inspects and downloads the exact original JSON, then revokes its URL', async () => {
    const doc = documentOf(), { root, panel } = mount(); await panel.ready
    expect(root.querySelector('[data-section=original-response] pre')?.textContent).toBe(doc.json)
    button(root, 'Download original JSON').click()
    expect(downloads).toHaveLength(1)
    expect(await blobText(blobs.get(downloads[0].url)!)).toBe(doc.json)
    expect(downloads[0].name).toBe('device-device-a-configuration-preview.json')
    panel.dispose()
    expect(revokeURL).toHaveBeenCalledWith(downloads[0].url)
  })

  it('fences out-of-order input, endpoint and kind changes and old cancel controls', async () => {
    const first = deferred<ConfigurationPreviewDocument>(), second = deferred<ConfigurationPreviewDocument>()
    const third = deferred<ConfigurationPreviewDocument>()
    vi.mocked(previewDeviceConfiguration).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    vi.mocked(previewTandemConfiguration).mockReturnValueOnce(third.promise)
    const { root, panel } = mount(), firstReady = panel.ready
    const oldCancel = button(root, 'Cancel preview request')
    const secondReady = panel.select({ ...selection('device-b'), endpoint: 'http://127.0.0.1:9001/preview' })
    oldCancel.click()
    expect(vi.mocked(previewDeviceConfiguration).mock.calls[0][2]!.aborted).toBe(true)
    expect(vi.mocked(previewDeviceConfiguration).mock.calls[1][2]!.aborted).toBe(false)
    const latest = panel.select(tandem())
    third.resolve(documentOf('tandem-c', 'tandem')); await latest
    first.resolve(documentOf()); second.reject(new Error('late endpoint error'))
    await Promise.all([firstReady, secondReady])
    expect(root.textContent).toContain('Configuration: tandem-c (tandem)')
    expect(root.textContent).not.toContain('late endpoint error')
    expect(root.textContent).not.toContain('device-b')
    expect(phase(root)).toBe('ready')
    expect(vi.mocked(previewDeviceConfiguration).mock.calls[1][2]!.aborted).toBe(true)
    expect(previewTandemConfiguration).toHaveBeenCalledWith(tandem().endpoint, tandem().input, expect.any(AbortSignal))
  })

  it('refetches changed input even when its id and endpoint stay the same', async () => {
    const { root, panel } = mount(); await panel.ready
    const next = selection()
    if (next.kind === 'device') next.input.settings!.Phi = 123
    const changed = documentOf()
    ;(changed.value.resolved.settings as Record<string, unknown>).Phi = 123
    vi.mocked(previewDeviceConfiguration).mockResolvedValueOnce(changed)
    await panel.select(next)
    expect(previewDeviceConfiguration).toHaveBeenCalledTimes(2)
    expect(root.querySelector('[data-section=resolved] [data-field=settings]')?.textContent).toContain('123')
  })

  it('cancels only its pending preview and ignores a late success, then permits a new selection', async () => {
    const pending = deferred<ConfigurationPreviewDocument>()
    vi.mocked(previewDeviceConfiguration).mockReturnValueOnce(pending.promise)
    const { root, panel } = mount(), oldReady = panel.ready
    button(root, 'Cancel preview request').click()
    expect(vi.mocked(previewDeviceConfiguration).mock.calls[0][2]!.aborted).toBe(true)
    expect(phase(root)).toBe('cancelled')
    pending.resolve(documentOf()); await oldReady
    expect(phase(root)).toBe('cancelled')
    expect(root.querySelector('pre')).toBeNull()
    expect(previewDeviceConfiguration).toHaveBeenCalledTimes(1)
    await panel.select(selection('next'))
    expect(phase(root)).toBe('ready')
  })

  it.each(['success', 'failure'])('disposal ignores late %s and later selections', async outcome => {
    const pending = deferred<ConfigurationPreviewDocument>()
    vi.mocked(previewDeviceConfiguration).mockReturnValueOnce(pending.promise)
    const { root, panel } = mount(), oldReady = panel.ready
    panel.dispose()
    expect(vi.mocked(previewDeviceConfiguration).mock.calls[0][2]!.aborted).toBe(true)
    if (outcome === 'success') pending.resolve(documentOf())
    else pending.reject(new Error('late disposal error'))
    await oldReady; await panel.select(selection('ignored'))
    expect(root.childNodes).toHaveLength(0)
    expect(previewDeviceConfiguration).toHaveBeenCalledTimes(1)
  })

  it('shows exact structured field paths and unresolved errors with safe text and raw error JSON', async () => {
    const raw = '{"detail":{"status":"unresolved","can_execute":false,"field_errors":[{"loc":["layers",0,"parameters","mu_n"],"msg":"<script>bad()</script>","input":null}]}}'
    vi.mocked(previewDeviceConfiguration).mockRejectedValueOnce(new ConfigurationPreviewError('http', 'HTTP 422', {
      status: 422, data: JSON.parse(raw), json: raw,
    }))
    const { root, panel } = mount(); await panel.ready
    expect(phase(root)).toBe('unresolved')
    expect(root.querySelector('[data-role=field-errors]')?.textContent).toContain('["layers", 0, "parameters", "mu_n"]')
    expect(root.textContent).toContain('<script>bad()</script>')
    expect(root.querySelector('script')).toBeNull()
    button(root, 'Download original JSON').click()
    expect(await blobText(blobs.get(downloads[0].url)!)).toBe(raw)
  })

  it('shows a transport error without reporting a prepared result or recreating JSON', async () => {
    vi.mocked(previewDeviceConfiguration).mockRejectedValueOnce(new TypeError('<img src=x> connection closed'))
    const { root, panel } = mount(); await panel.ready
    expect(phase(root)).toBe('error')
    expect(root.textContent).toContain('<img src=x> connection closed')
    expect(root.querySelector('img,pre')).toBeNull()
    expect(root.textContent).not.toContain('prepared_pending_dependencies')
  })

  it('retains an available non-JSON HTTP error body as response text', async () => {
    vi.mocked(previewDeviceConfiguration).mockRejectedValueOnce(new ConfigurationPreviewError('http', 'HTTP 503', {
      status: 503, json: '<html>Unavailable</html>',
    }))
    const { root, panel } = mount(); await panel.ready
    button(root, 'Download original response text').click()
    expect(await blobText(blobs.get(downloads[0].url)!)).toBe('<html>Unavailable</html>')
    expect(downloads[0].name.endsWith('.txt')).toBe(true)
  })

  it('reports download failure without changing preparation state', async () => {
    const { root, panel } = mount(); await panel.ready
    makeURL.mockImplementationOnce(() => { throw new Error('download unavailable') })
    button(root, 'Download original JSON').click()
    expect(root.textContent).toContain('Download failed: download unavailable')
    expect(phase(root)).toBe('ready')
    expect(root.querySelector('pre')?.textContent).toBe(documentOf().json)
  })

  it('revokes old download URLs and fences detached download controls on selection changes', async () => {
    const { root, panel } = mount(); await panel.ready
    const old = button(root, 'Download original JSON'); old.click()
    await panel.select(selection('new'))
    expect(revokeURL).toHaveBeenCalledWith(downloads[0].url)
    old.click()
    expect(downloads).toHaveLength(1)
  })

  it('keeps separate mounted panels independent', async () => {
    const one = mount(), two = mount(selection('other'))
    await Promise.all([one.panel.ready, two.panel.ready])
    one.panel.dispose()
    expect(two.root.textContent).toContain('Configuration: other')
    expect(phase(two.root)).toBe('ready')
  })

  it('does not clear a replacement panel when old disposal is repeated', async () => {
    const first = mount(); await first.panel.ready
    first.panel.dispose()
    const replacement = mountConfigurationPreviewPanel(first.root, selection('replacement'))
    panels.push(replacement); await replacement.ready
    first.panel.dispose()
    expect(first.root.textContent).toContain('Configuration: replacement')
    expect(phase(first.root)).toBe('ready')
  })
})
