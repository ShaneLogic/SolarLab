import {
  ConfigurationPreviewError, previewDeviceConfiguration, previewTandemConfiguration,
} from '../configuration-client'
import type { DeviceInput, TandemInput } from '../generated/configuration-inputs'

export type ConfigurationPreviewSelection = { endpoint: string | URL } & (
  { kind: 'device'; input: DeviceInput } | { kind: 'tandem'; input: TandemInput }
)

export interface ConfigurationPreviewPanel {
  readonly ready: Promise<void>
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

  function original(view: View, json: string, isJson = true) {
    const details = element('details')
    details.className = 'device-settings'
    details.dataset.section = 'original-response'
    details.append(element('summary', isJson ? 'Original response JSON' : 'Original response text'))
    const pre = element('pre', json)
    pre.style.whiteSpace = 'pre-wrap'
    pre.style.overflowWrap = 'anywhere'
    const button = element('button', isJson ? 'Download original JSON' : 'Download original response text')
    button.type = 'button'; button.className = 'btn btn-ghost'
    const error = element('p')
    error.className = 'status error'; error.setAttribute('role', 'alert')
    button.addEventListener('click', () => {
      if (!current(view)) return
      try {
        error.textContent = ''
        const url = URL.createObjectURL(new Blob([json], {
          type: isJson ? 'application/json;charset=utf-8' : 'text/plain;charset=utf-8',
        }))
        const timer = setTimeout(() => { URL.revokeObjectURL(url); view.urls.delete(url) }, 1000)
        view.urls.set(url, timer)
        const anchor = element('a')
        anchor.href = url; anchor.download = `${view.filename}.${isJson ? 'json' : 'txt'}`
        anchor.hidden = true; root.append(anchor)
        try { anchor.click() } finally { anchor.remove() }
      } catch (failure) { error.textContent = `Download failed: ${message(failure)}` }
    })
    details.append(pre, button, error); view.content.append(details)
  }

  function cancel() {
    const view = active
    if (!view || !current(view) || view.card.dataset.state !== 'loading') return
    state(view, 'cancelled', 'Preview request cancelled')
    view.abort.abort() // Only this fetch is aborted; there is no simulation control API.
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
    try {
      const doc = selection.kind === 'device'
        ? await previewDeviceConfiguration(endpoint, selection.input, view.abort.signal)
        : await previewTandemConfiguration(endpoint, selection.input, view.abort.signal)
      if (!current(view)) return
      state(view, 'ready', doc.value.status)
      const identity = inspect('Current configuration content identity', doc.value.identity)
      const effective = inspect('Effective parameters and field origins', doc.value.resolved, true)
      effective.dataset.section = 'resolved'
      const input = inspect('Supplied input', doc.value.input)
      input.dataset.section = 'input'
      content.append(element('p', `can_execute: ${scalar(doc.value.can_execute)}`), identity, effective, input,
        element('p', 'Fields retain their supplied names. Explicit null and -0 remain distinct; omitted input fields are not added.'))
      original(view, doc.json)
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
    select(selection) { ready = select(selection); return ready },
    cancel,
    dispose() {
      if (disposed) return
      disposed = true; close(active); active = undefined; root.replaceChildren()
    },
  }
}
