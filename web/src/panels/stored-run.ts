import { StoredRunClient, RunClientError } from '../run-client'
import type {
  ArtifactReference, AttemptIdentity, ReplayCursor, RunSubscription,
  StoredDocument, StoredEvent, TerminalRecord,
} from '../run-client'

export interface StoredRunSelection extends AttemptIdentity {
  endpoint: string | URL
  storeId: string
}

export interface StoredRunPanel {
  readonly ready: Promise<void>
  readonly cursor: ReplayCursor | undefined
  select(selection: StoredRunSelection): Promise<void>
  reconnect(): void
  dispose(): void
}

interface SelectionView {
  client: StoredRunClient
  target: Readonly<AttemptIdentity>
  abort: AbortController
  subscription?: RunSubscription
  connection: number
  loaded: boolean
  loading: boolean
  terminal?: TerminalRecord
  urls: Map<string, ReturnType<typeof setTimeout>>
  documents: HTMLElement
  events: HTMLElement
  artifacts: HTMLElement
  state: HTMLElement
  runState: HTMLElement
  reason: HTMLElement
  transport: HTMLElement
  readError: HTMLElement
  readHistory: HTMLElement
  lastEvent: HTMLElement
  reconnect: HTMLButtonElement
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}

function message(error: unknown): string {
  return error instanceof RunClientError ? `${error.code}: ${error.message}`
    : error instanceof Error ? error.message : String(error)
}

function storedReason(result: Record<string, unknown>): string {
  if (!Object.hasOwn(result, 'reason')) return 'Omitted from the stored result'
  const reason = result.reason
  if (reason === null) return 'null (stored)'
  if (Object.is(reason, -0)) return '-0'
  if (reason === '') return 'Empty string (stored)'
  if (typeof reason === 'object') return 'Structured value; see original terminal JSON'
  return String(reason)
}

/** Read-only preparation for an explicitly selected stored attempt.
 * Raw documents are displayed/exported verbatim; opaque results are not decoded
 * as physics. A successful execution record is not a qualification decision.
 */
export function mountStoredRunPanel(root: HTMLElement, initial: StoredRunSelection): StoredRunPanel {
  let active: SelectionView | undefined
  let disposed = false
  let ready: Promise<void>
  const current = (view: SelectionView) => !disposed && active === view

  function close(view: SelectionView | undefined) {
    if (!view) return
    view.connection++
    view.abort.abort()
    view.subscription?.unsubscribe()
    for (const [url, timer] of view.urls) {
      clearTimeout(timer)
      URL.revokeObjectURL(url)
    }
    view.urls.clear()
  }

  function download(view: SelectionView, blob: Blob, filename: string) {
    if (!current(view)) return
    const url = URL.createObjectURL(blob)
    const timer = setTimeout(() => {
      URL.revokeObjectURL(url)
      view.urls.delete(url)
    }, 1000)
    view.urls.set(url, timer)
    const anchor = element('a')
    anchor.href = url
    anchor.download = filename
    anchor.hidden = true
    root.append(anchor)
    try { anchor.click() } finally { anchor.remove() }
  }

  function rawDocument(view: SelectionView, parent: HTMLElement, title: string, doc: StoredDocument<unknown>) {
    const details = element('details')
    details.dataset.record = title
    details.append(element('summary', title))
    const content = element('pre', doc.json)
    content.style.whiteSpace = 'pre-wrap'
    content.style.overflowWrap = 'anywhere'
    const exportButton = element('button', 'Download original JSON')
    exportButton.type = 'button'
    exportButton.className = 'btn btn-ghost'
    const error = element('span')
    error.className = 'status error'
    error.setAttribute('role', 'alert')
    exportButton.addEventListener('click', () => {
      if (!current(view)) return
      try {
        error.textContent = ''
        download(view, new Blob([doc.json], { type: 'application/json;charset=utf-8' }),
          `${view.target.runId}-${view.target.attemptId}-${title.replace(/[^a-z0-9]+/gi, '-')}.json`)
      } catch (failure) { error.textContent = `JSON download failed: ${message(failure)}` }
    })
    details.append(content, exportButton, error)
    parent.append(details)
    return details
  }

  function artifact(view: SelectionView, reference: ArtifactReference) {
    const row = element('div')
    row.className = 'card'
    row.dataset.artifactId = reference.artifact_id
    row.append(element('p', `Artifact: ${reference.artifact_id}`))
    const provenance = element('details')
    provenance.append(element('summary', 'Artifact provenance'),
      element('p', `Committed manifest SHA-256: ${reference.manifest_sha256}`))
    row.append(provenance)
    const button = element('button', 'Verify and download artifact')
    button.type = 'button'
    button.className = 'btn btn-ghost'
    const status = element('span', 'Content not yet downloaded')
    status.className = 'status'
    status.setAttribute('role', 'status')
    button.addEventListener('click', async () => {
      if (!current(view) || button.disabled) return
      button.disabled = true
      status.className = 'status'
      status.textContent = 'Verifying saved content…'
      try {
        const result = await view.client.getArtifact(view.target, reference, view.abort.signal)
        if (!current(view)) return
        rawDocument(view, row, 'Verified artifact record', result.record)
        download(view, new Blob([result.bytes], { type: 'application/octet-stream' }), `${reference.artifact_id}.bin`)
        status.textContent = 'SHA-256 and size verified; download requested'
      } catch (failure) {
        if (current(view)) {
          status.className = 'status error'
          status.textContent = `Artifact download failed: ${message(failure)}`
        }
      } finally { if (current(view)) button.disabled = false }
    })
    row.append(button, status)
    view.artifacts.append(row)
  }

  function readErrors(view: SelectionView, errors: string[]) {
    view.readError.textContent = errors.join('\n')
    if (errors.length) {
      view.readHistory.parentElement!.hidden = false
      for (const error of errors) view.readHistory.append(element('li', error))
    }
  }

  function terminal(view: SelectionView, value: TerminalRecord) {
    if (view.terminal) {
      if (view.terminal.state !== value.state) throw new RunClientError('protocol', 'Stored terminal state changed')
      return // Keep existing download controls; every received raw document is retained separately.
    }
    view.terminal = value
    view.state.textContent = value.state
    view.reason.textContent = storedReason(value.result)
    view.artifacts.replaceChildren()
    if (!value.artifacts.length) view.artifacts.append(element('p', 'No committed artifact references'))
    for (const reference of value.artifacts) artifact(view, reference)
  }

  async function snapshots(view: SelectionView): Promise<boolean> {
    const results = await Promise.allSettled([
      view.client.getRun(view.target.runId, view.abort.signal),
      view.client.getAttempt(view.target, view.abort.signal),
    ])
    if (!current(view)) return false
    const [run, attempt] = results
    const errors: string[] = []
    if (run.status === 'fulfilled') {
      rawDocument(view, view.documents, 'Run snapshot / execution metadata', run.value)
      view.runState.textContent = `${run.value.value.state}; current attempt ${run.value.value.current_attempt_id}`
    } else {
      if (view.runState.textContent === 'Loading') view.runState.textContent = 'Unavailable'
      errors.push(`Run read failed: ${message(run.reason)}`)
    }
    if (attempt.status === 'fulfilled') {
      rawDocument(view, view.documents, 'Attempt snapshot / stored result', attempt.value)
      if (attempt.value.value.terminal) terminal(view, attempt.value.value.terminal)
      else if (!view.terminal) view.state.textContent = attempt.value.value.state
    } else {
      if (view.state.textContent === 'Loading') view.state.textContent = 'Unavailable'
      errors.push(`Attempt read failed: ${message(attempt.reason)}`)
    }
    readErrors(view, errors)
    return errors.length === 0
  }

  async function reconnect(view: SelectionView) {
    if (!current(view) || view.loading) return
    if (view.loaded) { connect(view); return }
    view.loading = true
    view.reconnect.disabled = true
    view.transport.textContent = 'Reading stored snapshots…'
    try {
      view.loaded = await snapshots(view)
      if (current(view) && view.loaded) connect(view)
    } catch (failure) {
      if (current(view)) readErrors(view, [`Stored read failed: ${message(failure)}`])
    } finally {
      view.loading = false
      if (current(view)) {
        view.reconnect.disabled = false
        view.reconnect.textContent = view.loaded ? 'Reconnect event stream' : 'Retry stored reads'
        if (!view.loaded) view.transport.textContent = 'Not connected; stored reads need retry'
      }
    }
  }

  function connect(view: SelectionView) {
    if (!current(view) || !view.loaded) return
    const resume = view.subscription?.cursor
    view.subscription?.unsubscribe()
    const connection = ++view.connection
    const listening = () => current(view) && view.connection === connection
    view.transport.className = 'status'
    view.transport.textContent = `Connecting; replay after ${resume?.sequence ?? '0'}`
    try {
      const subscription = view.client.subscribe(view.target, (doc: StoredDocument<StoredEvent>) => {
        if (!listening()) return
        const item = element('li')
        item.dataset.sequence = doc.value.sequence.toString()
        rawDocument(view, item, `Event ${doc.value.sequence}: ${doc.value.kind}`, doc).open = true
        view.events.append(item)
        view.lastEvent.textContent = `${doc.value.sequence}: ${doc.value.kind}`
        view.transport.textContent = 'Receiving persisted events'
        if (doc.value.kind === 'terminal') terminal(view, doc.value.payload as TerminalRecord)
        else if (!view.terminal && doc.value.kind === 'running') {
          // Replay begins at zero; an old queued event cannot regress a newer
          // running snapshot, nor can any replay event undo a terminal record.
          view.state.textContent = 'running'
        }
      }, resume)
      view.subscription = subscription
      void subscription.done.then(async end => {
        if (!listening()) return
        const labels = {
          terminal: 'Terminal event received; stream closed',
          terminal_acknowledged: 'Terminal already acknowledged (HTTP 204)',
          disconnected: 'Transport disconnected; stored outcome unchanged',
          unsubscribed: 'Unsubscribed; stored outcome unchanged',
        }
        view.transport.textContent = labels[end.reason]
        if (end.transportError !== undefined) view.transport.textContent += `: ${message(end.transportError)}`
        if (end.reason === 'terminal' || end.reason === 'terminal_acknowledged') await snapshots(view)
      }).catch(failure => {
        if (listening()) {
          view.transport.className = 'status error'
          view.transport.textContent = `Stream error: ${message(failure)}; stored outcome unchanged`
        }
      })
    } catch (failure) {
      view.transport.className = 'status error'
      view.transport.textContent = `Stream error: ${message(failure)}`
    }
  }

  async function select(selection: StoredRunSelection) {
    close(active)
    active = undefined
    if (disposed) return
    root.replaceChildren()
    const card = element('section')
    card.className = 'card'
    card.dataset.panel = 'stored-run'
    card.append(element('h3', 'Stored run'),
      element('p', 'Execution state is recorded independently of physical qualification. Qualification and unknown payloads remain in the original stored JSON.'))
    root.append(card)
    try {
      const client = new StoredRunClient(selection.endpoint, selection.storeId)
      const target = Object.freeze({ runId: selection.runId, attemptId: selection.attemptId })
      const source = element('details')
      source.append(element('summary', 'Source connection'), element('p', `Endpoint: ${client.endpoint}`),
        element('p', `Store: ${client.storeId}`))
      card.append(element('p', `Run: ${target.runId}`), element('p', `Selected attempt: ${target.attemptId}`), source)
      function field(label: string, role: string, initial: string) {
        const line = element('p')
        line.append(element('strong', `${label}: `))
        const value = element('span', initial)
        value.dataset.role = role
        line.append(value)
        card.append(line)
        return value
      }
      const runState = field('Run snapshot state', 'run-state', 'Loading')
      const state = field('Selected attempt state', 'attempt-state', 'Loading')
      const reason = field('Persisted terminal reason', 'terminal-reason', 'No terminal record loaded')
      const lastEvent = field('Last displayed event', 'last-event', 'None')
      const transport = field('Transport', 'transport', 'Not connected')
      transport.className = 'status'
      transport.setAttribute('role', 'status')
      const readError = element('p')
      readError.className = 'status error'
      readError.dataset.role = 'read-error'
      readError.setAttribute('role', 'alert')
      const retryButton = element('button', 'Reading stored snapshots…')
      retryButton.type = 'button'
      retryButton.className = 'btn btn-ghost'
      retryButton.disabled = true
      const errorHistory = element('details')
      errorHistory.hidden = true
      const readHistory = element('ul')
      readHistory.dataset.role = 'read-error-history'
      errorHistory.append(element('summary', 'Read error history'), readHistory)
      const documents = element('div')
      documents.dataset.role = 'documents'
      const events = element('ol')
      events.dataset.role = 'events'
      const artifacts = element('div', 'No terminal record loaded')
      artifacts.dataset.role = 'artifacts'
      card.append(retryButton, readError, errorHistory, element('h4', 'Stored snapshots'), documents,
        element('h4', 'Persisted events'), events, element('h4', 'Committed artifacts'), artifacts)
      const view: SelectionView = { client, target, abort: new AbortController(), connection: 0, loaded: false, loading: false,
        urls: new Map(), documents, events, artifacts, state, runState, reason, transport, readError, readHistory, lastEvent,
        reconnect: retryButton }
      active = view
      retryButton.addEventListener('click', () => { void reconnect(view) })
      await reconnect(view)
    } catch (failure) {
      // Asynchronous stale reads are handled by snapshots; this also reports
      // invalid explicit endpoint/identity input without selecting a fallback.
      if (!disposed && root.contains(card)) {
        const error = element('p', `Stored run read failed: ${message(failure)}`)
        error.className = 'status error'
        error.setAttribute('role', 'alert')
        card.append(error)
      }
    }
  }

  ready = select(initial)
  return {
    get ready() { return ready },
    get cursor() { return active?.subscription?.cursor },
    select(selection) { ready = select(selection); return ready },
    reconnect() { if (active) void reconnect(active) },
    dispose() {
      disposed = true
      close(active)
      active = undefined
      root.replaceChildren()
    },
  }
}
