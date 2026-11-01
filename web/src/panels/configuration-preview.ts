import {
  ConfigurationPreviewError, previewDeviceConfiguration, previewTandemConfiguration, previewExperimentConfiguration, previewSweepConfiguration, previewLegacyWire,
} from '../configuration-client'
import type { ConfigurationPreviewDocument, LegacyWirePreviewDocument, LegacyWirePreviewRequest, LegacyWireSourceIdentity } from '../configuration-client'
import type { DeviceInput, TandemInput, SpatialExperimentInput, JVExperimentInput, SweepInput } from '../generated/configuration-inputs'

export interface LegacyWireExportSelection {
  endpoint: string | URL
  source: LegacyWireSourceIdentity
  references?: Readonly<Record<string, LegacyWireSourceIdentity>>
}

export type ConfigurationPreviewSelection = { endpoint: string | URL } & (
  { kind: 'device'; input: DeviceInput; legacyWire?: LegacyWireExportSelection }
  | { kind: 'tandem'; input: TandemInput; legacyWire?: LegacyWireExportSelection }
  | { kind: 'experiment'; input: SpatialExperimentInput | JVExperimentInput }
  | { kind: 'sweep'; input: SweepInput }
)

export interface ConfigurationPreviewPanel {
  readonly ready: Promise<void>
  readonly document: ConfigurationPreviewDocument | undefined
  readonly wireReady: Promise<void>
  readonly wireDocument: LegacyWirePreviewDocument | undefined
  select(selection: ConfigurationPreviewSelection): Promise<void>
  cancel(): void
  dispose(): void
}

interface View {
  abort: AbortController
  card: HTMLElement
  status: HTMLElement
  content: HTMLElement
  cancel: HTMLButtonElement
  filename: string
  urls: Map<string, ReturnType<typeof setTimeout>>
  document?: ConfigurationPreviewDocument
  wireAbort?: AbortController
  wireReady?: Promise<void>
  wireDocument?: LegacyWirePreviewDocument
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function scalar(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value) // Quote a display scalar only.
  if (typeof value === 'bigint') return value.toString()
  if (Object.is(value, -0)) return '-0'
  return String(value)
}

function scalarElement(value: unknown) {
  const node = element('code', scalar(value))
  if (typeof value !== 'string') node.style.whiteSpace = 'nowrap'
  return node
}

/** Inspect the supplied mappings without interpreting a field as a physical
 * quantity. Original JSON is kept separately and is never reconstructed here. */
function inspect(label: string, value: unknown, open = false): HTMLElement {
  if (value === null || typeof value !== 'object') return scalarElement(value)
  const details = element('details')
  details.className = 'device-settings'
  details.dataset.field = label
  details.open = open
  details.append(element('summary', `${label} (${Array.isArray(value) ? value.length : Object.keys(value).length})`))
  const table = element('table')
  table.className = 'param-table mode-table'
  const body = element('tbody')
  const nested: HTMLElement[] = []
  table.append(body)
  const rows = Array.isArray(value) && value.length && value.every(item => record(item)
    && Object.values(item).every(part => part === null || typeof part !== 'object'))
    ? value as Record<string, unknown>[] : undefined
  if (rows) {
    const keys = [...new Set(rows.flatMap(row => Object.keys(row)))]
    const head = element('thead'), line = element('tr')
    for (const key of keys) line.append(element('th', key))
    head.append(line); table.prepend(head)
    for (const row of rows) {
      const line = element('tr')
      for (const key of keys) {
        const cell = element('td')
        cell.append(Object.hasOwn(row, key) ? scalarElement(row[key]) : element('em', 'omitted'))
        line.append(cell)
      }
      body.append(line)
    }
  } else {
    const entries = Object.entries(value)
    const first = ['settings', 'layers', 'interfaces', 'contacts', 'top_cell', 'bottom_cell']
    entries.sort(([a], [b]) => (first.includes(a) ? first.indexOf(a) : first.length)
      - (first.includes(b) ? first.indexOf(b) : first.length))
    for (const [key, item] of entries) {
      if (item !== null && typeof item === 'object') {
        nested.push(inspect(key, item, key === 'settings'))
        continue
      }
      const line = element('tr'), name = element('td', key), cell = element('td')
      name.dataset.key = key
      cell.append(scalarElement(item))
      line.append(name, cell); body.append(line)
    }
  }
  if (!Object.keys(value).length) details.append(element('code', Array.isArray(value) ? '[]' : '{}'))
  else {
    details.append(...nested)
    if (body.childElementCount) {
      const scroll = element('div')
      scroll.style.overflowX = 'auto'
      scroll.append(table); details.append(scroll)
    }
  }
  return details
}

function message(error: unknown) {
  return error instanceof ConfigurationPreviewError ? `${error.code}: ${error.message}`
    : error instanceof Error ? error.message : String(error)
}

export function mountConfigurationPreviewPanel(
  root: HTMLElement, initial: ConfigurationPreviewSelection,
): ConfigurationPreviewPanel {
  let active: View | undefined
  let disposed = false
  let ready: Promise<void>
  const current = (view: View) => !disposed && active === view && !view.abort.signal.aborted

  function close(view: View | undefined) {
    if (!view) return
    view.abort.abort()
    view.wireAbort?.abort()
    for (const [url, timer] of view.urls) {
      clearTimeout(timer); URL.revokeObjectURL(url)
    }
    view.urls.clear()
  }

  function state(view: View, phase: string, text: string) {
    view.card.dataset.state = phase
    view.status.textContent = text
    view.status.className = phase === 'error' || phase === 'unresolved' ? 'status error' : 'status'
    view.cancel.hidden = phase !== 'loading'
  }

  function download(view: View, parent: HTMLElement, text: string, filename: string, mediaType: string,
    label: string, isCurrent = () => current(view)) {
    const button = element('button', label)
    button.type = 'button'; button.className = 'btn btn-ghost'
    const error = element('p')
    error.className = 'status error'; error.setAttribute('role', 'alert')
    button.addEventListener('click', () => {
      if (!isCurrent()) return
      try {
        error.textContent = ''
        const url = URL.createObjectURL(new Blob([text], { type: mediaType }))
        const timer = setTimeout(() => { URL.revokeObjectURL(url); view.urls.delete(url) }, 1000)
        view.urls.set(url, timer)
        const anchor = element('a')
        anchor.href = url; anchor.download = filename
        anchor.hidden = true; root.append(anchor)
        try { anchor.click() } finally { anchor.remove() }
      } catch (failure) { error.textContent = `Download failed: ${message(failure)}` }
    })
    parent.append(button, error)
  }

  function original(view: View, json: string, isJson = true) {
    const details = element('details')
    details.className = 'device-settings'
    details.dataset.section = 'original-response'
    details.append(element('summary', isJson ? 'Original response JSON' : 'Original response text'))
    const pre = element('pre', json)
    pre.style.whiteSpace = 'pre-wrap'; pre.style.overflowWrap = 'anywhere'
    details.append(pre)
    download(view, details, json, `${view.filename}.${isJson ? 'json' : 'txt'}`,
      isJson ? 'application/json;charset=utf-8' : 'text/plain;charset=utf-8',
      isJson ? 'Download original JSON' : 'Download original response text')
    view.content.append(details)
  }

  function legacyControls(view: View, selection: Extract<ConfigurationPreviewSelection, { kind: 'device' | 'tandem' }>) {
    const binding = selection.legacyWire
    if (!binding) return
    const endpoint = String(binding.endpoint)
    const request: LegacyWirePreviewRequest = structuredClone({
      schema_version: 'solarlab.legacy-wire-preview-request.v1', input: selection.input,
      source: binding.source, references: binding.references ?? {},
    })
    const section = element('section'); section.dataset.section = 'legacy-wire'; section.dataset.state = 'idle'
    section.append(element('h4', 'Legacy YAML export'), element('p', `Original source: ${request.source.id}`),
      element('p', 'Preview changes before downloading. Export does not run a simulation.'))
    const status = element('p', 'Export has not been previewed'); status.setAttribute('role', 'status')
    const start = element('button', 'Preview legacy YAML'), stop = element('button', 'Cancel legacy preview')
    start.type = stop.type = 'button'; start.className = stop.className = 'btn btn-ghost'; stop.hidden = true
    const output = element('div'); section.append(status, start, stop, output); view.card.append(section)
    const state = (phase: string, text: string) => {
      section.dataset.state = phase; status.textContent = text; stop.hidden = phase !== 'loading'
    }
    stop.addEventListener('click', () => {
      if (!current(view) || !view.wireAbort || section.dataset.state !== 'loading') return
      view.wireAbort.abort(); view.wireDocument = undefined; output.replaceChildren()
      state('cancelled', 'Legacy preview request cancelled')
    })
    async function previewWire() {
      if (!current(view)) return
      view.wireAbort?.abort()
      const abort = new AbortController(); view.wireAbort = abort; view.wireDocument = undefined
      const isCurrent = () => current(view) && view.wireAbort === abort && !abort.signal.aborted
      output.replaceChildren(); state('loading', 'Checking legacy YAML export…')
      try {
        const doc = await previewLegacyWire(endpoint, request, abort.signal)
        if (!isCurrent()) return
        view.wireDocument = doc
        state(doc.value.report.supported ? 'ready' : 'unsupported', doc.value.report.supported
          ? 'Legacy YAML ready to download' : 'This edit cannot be represented in legacy YAML')
        output.append(element('p', 'Export preview only. Simulation is disabled.'))
        if (!doc.value.report.supported) {
          const fields = element('ul'); fields.dataset.role = 'legacy-unsupported'
          for (const issue of doc.value.report.unsupported) fields.append(element('li', `${issue.path}: ${issue.message} (${issue.code})`))
          output.append(fields)
        }
        output.append(inspect('Declared changes', doc.value.report.differences, true), inspect('Migration details', doc.value.report))
        for (const file of doc.value.documents) {
          const main = file.role === 'source'
          const details = element('details'); details.dataset.role = 'legacy-document'
          details.append(element('summary', main ? 'Legacy configuration YAML' : `Reference: ${file.reference}`))
          const pre = element('pre', file.utf8); pre.style.whiteSpace = 'pre-wrap'; details.append(pre)
          const name = (file.reference ?? file.source_id).split(/[\\/]/).at(-1)!.replace(/[^a-z0-9_.-]+/gi, '_') || 'configuration.yaml'
          download(view, details, file.utf8, name, 'application/yaml;charset=utf-8',
            main ? 'Download legacy YAML' : `Download reference ${file.reference}`, isCurrent)
          output.append(details)
        }
        if (doc.value.documents.length > 1) output.append(element('p',
          'Download the configuration and every reference. Reference paths in the YAML remain unchanged.'))
        download(view, output, doc.json, `${view.filename}-legacy-wire.json`, 'application/json;charset=utf-8', 'Download migration report', isCurrent)
      } catch (error) {
        if (!isCurrent()) return
        state('error', 'Legacy export preview failed')
        const alert = element('p', message(error)); alert.setAttribute('role', 'alert'); output.append(alert)
        if (error instanceof ConfigurationPreviewError && record(error.data) && record(error.data.detail)) {
          output.append(inspect('Field errors', error.data.detail.field_errors, true))
        }
      }
    }
    start.addEventListener('click', () => { view.wireReady = previewWire() })
  }

  function cancel() {
    const view = active
    if (!view || !current(view) || view.card.dataset.state !== 'loading') return
    state(view, 'cancelled', 'Preview request cancelled')
    view.abort.abort() // Only this fetch is aborted; there is no simulation control API.
    view.wireAbort?.abort(); view.wireDocument = undefined
  }

  async function select(selection: ConfigurationPreviewSelection) {
    close(active); active = undefined
    if (disposed) return
    const card = element('section')
    card.className = 'card'; card.dataset.panel = 'configuration-preview'
    card.style.minWidth = '0'; card.style.overflowWrap = 'anywhere'
    const endpoint = String(selection.endpoint), id = String(selection.input.id), kind = selection.kind
    card.append(element('h3', 'Configuration preview'), element('p', `Configuration: ${id} (${kind})`),
      element('p', 'Preparation only. Execution is disabled; physical qualification is not established.'))
    const source = element('details')
    source.append(element('summary', 'Preview endpoint'), element('p', endpoint)); card.append(source)
    const status = element('p'); status.setAttribute('role', 'status'); status.dataset.role = 'preparation-state'
    const stop = element('button', 'Cancel preview request')
    stop.type = 'button'; stop.className = 'btn btn-ghost'
    const content = element('div')
    card.append(status, stop, content); root.replaceChildren(card)
    const view: View = { abort: new AbortController(), card, status, content, cancel: stop,
      filename: `${kind}-${id.replace(/[^a-z0-9_.-]+/gi, '_')}-configuration-preview`, urls: new Map() }
    active = view
    stop.addEventListener('click', () => { if (current(view)) cancel() })
    state(view, 'loading', 'Loading configuration preview…')
    if (selection.kind === 'device' || selection.kind === 'tandem') legacyControls(view, selection)
    try {
      const doc = selection.kind === 'device'
        ? await previewDeviceConfiguration(endpoint, selection.input, view.abort.signal)
        : selection.kind === 'tandem' ? await previewTandemConfiguration(endpoint, selection.input, view.abort.signal)
          : selection.kind === 'sweep' ? await previewSweepConfiguration(endpoint, selection.input, view.abort.signal)
          : await previewExperimentConfiguration(endpoint, selection.input, view.abort.signal)
      if (!current(view)) return
      view.document = doc
      state(view, 'ready', doc.value.status)
      const identity = inspect('Current configuration content identity', doc.value.identity)
      const displayed = selection.kind === 'sweep' && Array.isArray(doc.value.resolved.points)
        ? { ...doc.value.resolved, points: doc.value.resolved.points.map(point => record(point)
          ? Object.fromEntries(Object.entries(point).filter(([name]) => name !== 'applied_input')) : point) }
        : doc.value.resolved
      const effective = inspect('Effective parameters and field origins', displayed, true)
      effective.dataset.section = 'resolved'
      const input = inspect('Supplied input', doc.value.input)
      input.dataset.section = 'input'
      content.append(element('p', 'Execution is unavailable in this preview.'), identity, effective, input,
        element('p', 'Fields retain their supplied names. Explicit null and -0 remain distinct; omitted input fields are not added.'))
      original(view, doc.json)
      if (selection.kind === 'sweep') content.append(element('p', 'Each point’s complete applied input is retained in the original response JSON and can be opened in the point preview. No point was simulated.'))
    } catch (error) {
      if (!current(view)) return
      const detail = error instanceof ConfigurationPreviewError && record(error.data) && record(error.data.detail)
        ? error.data.detail : undefined
      state(view, detail?.status === 'unresolved' ? 'unresolved' : 'error',
        detail?.status === 'unresolved' ? 'unresolved' : 'Preview error')
      const alert = element('p', message(error)); alert.setAttribute('role', 'alert'); content.append(alert)
      if (Array.isArray(detail?.field_errors)) {
        const paths = element('ul'); paths.dataset.role = 'field-errors'
        for (const field of detail.field_errors) {
          if (!record(field)) continue
          const path = Array.isArray(field.loc) ? '[' + field.loc.map(scalar).join(', ') + ']' : scalar(field.loc)
          paths.append(element('li', `${path}: ${scalar(field.msg)}`))
        }
        content.append(paths)
      }
      if (error instanceof ConfigurationPreviewError) {
        if (error.data !== undefined) content.append(inspect('Error details', error.data, true))
        if (error.json !== undefined) original(view, error.json, error.data !== undefined)
      }
    }
  }

  ready = select(initial)
  return {
    get ready() { return ready },
    get document() { return active && current(active) ? active.document : undefined },
    get wireReady() { return active?.wireReady ?? Promise.resolve() },
    get wireDocument() { return active && current(active) && !active.wireAbort?.signal.aborted ? active.wireDocument : undefined },
    select(selection) { ready = select(selection); return ready },
    cancel,
    dispose() {
      if (disposed) return
      disposed = true; close(active); active = undefined; root.replaceChildren()
    },
  }
}
